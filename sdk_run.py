#!/usr/bin/env python3
"""Frisson Vision, driven end to end by the Comfy SDK.

The SDK talks the Comfy API v2. Point it at:
  - your own ComfyUI, through the open-source comfy-api-proxy that runs beside it:
        pip install comfy-sdk comfy-api-proxy
        comfy-api-proxy --max-upload-mb 2000          # serves http://127.0.0.1:8189
        set COMFY_BASE_URL=http://127.0.0.1:8189      # (macOS/Linux: export ...)
  - or a Comfy API deployment with the Frisson Vision nodes installed (COMFY_BASE_URL + COMFY_API_KEY).

    python sdk_run.py --song my_track.mp3 --pack crowd --format 9:16 --seconds 30
    python sdk_run.py --song my_track.mp3 --pack artist --artist me.png --framing mixed
    python sdk_run.py --song track.wav --mode motion --dance me_dancing.mp4 --pack wardrobe --preset fast

What happens:
  1. Director job: analyses the song (beats, bars, sections, drop), separates the vocal and
     drum stems, plans the cut, casts the performers and returns the run as jobs.json.
  2. This script submits every render job (portraits, lip-sync clips, twin keys, pose-matched
     twins; grouped by model) and waits for them.
  3. Take QC job: checks every clip (missing / frozen / blown / dead mouth / cut away / lip sync) and returns re-roll
     jobs with new seeds; the script submits those and checks again (--retries times).
  4. Assemble job: cut on the beat, audio-reactive FX, the finish, the song muxed in; the SDK
     downloads the MP4 (plus the EDL for your editor).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIRECTOR = HERE / "api" / "FrissonVision_director_api.json"


def node_id(wf, class_type: str) -> str:
    for nid, node in wf.json.items():
        if node.get("class_type") == class_type:
            return nid
    raise KeyError(class_type)


def fetch_json(job, suffix: str) -> dict:
    for o in job.outputs:
        if o.name.endswith(suffix):
            return json.loads(o.to_bytes().decode("utf-8"))
    raise RuntimeError(f"job {job.id} returned no {suffix}")


def run_all(client, graphs: list[dict], label: str, run: str, timeout: float) -> int:
    """Submit graphs in order (the server runs them in order, so models load once per group)
    and wait for all of them. Returns the number that failed."""
    jobs = [client.submit(client.workflows.from_json(g), metadata={"app": "frissonvision", "run": run})
            for g in graphs]
    failed = 0
    t0 = time.time()
    for i, j in enumerate(jobs, 1):
        j.wait(timeout=max(60.0, timeout - (time.time() - t0)))
        if j.status != "succeeded":
            failed += 1
        print(f"\r  {label}: {i}/{len(jobs)} done, {failed} failed, {(time.time() - t0) / 60:.1f} min",
              end="", flush=True)
    print()
    return failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--song", required=True, help="a song you have the rights to use")
    ap.add_argument("--mode", default="voice", choices=["voice", "motion"])
    ap.add_argument("--dance", default="", help="motion mode: the dance clip (path on the server)")
    ap.add_argument("--artist", default="", help="one-artist mode: a photo of the singer (path on the server)")
    ap.add_argument("--pack", default="crowd")
    ap.add_argument("--format", default="16:9", choices=["16:9", "9:16", "1:1"])
    ap.add_argument("--preset", default="quality", choices=["quality", "fast"])
    ap.add_argument("--framing", default="mixed", choices=["mixed", "close"])
    ap.add_argument("--seconds", type=float, default=30.0, help="0 = whole song")
    ap.add_argument("--start", type=float, default=-1.0, help="-1 = strongest stretch")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--twins", default="auto", choices=["auto", "loud", "all", "off"])
    ap.add_argument("--retries", type=int, default=2, help="take-QC re-roll rounds (0 = off)")
    ap.add_argument("--fx", type=float, default=1.0, help="FX intensity (0 = clean cut)")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--out", default="frissonvision_out")
    ap.add_argument("--timeout", type=float, default=8 * 3600)
    a = ap.parse_args()

    from comfy_sdk import Comfy

    run = a.run_name or f"sdk-{a.mode}-{a.pack}-{dt.datetime.now():%Y%m%d-%H%M%S}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Comfy() as client:
        # 1. plan
        wf = client.workflows.from_file(DIRECTOR)
        song, d = node_id(wf, "LoadAudio"), node_id(wf, "FVDirector")
        wf.set_input(song, "audio", client.assets.from_file(a.song))
        for k, v in (("mode", a.mode), ("pack", a.pack), ("format", a.format), ("preset", a.preset),
                     ("seconds", a.seconds), ("start", a.start), ("seed", a.seed), ("twins", a.twins),
                     ("run_name", run), ("dance_video_path", a.dance), ("artist_image_path", a.artist),
                     ("framing", a.framing), ("queue_renders", False), ("queue_assemble", False),
                     ("qc_retries", 0)):
            wf.set_input(d, k, v)
        job = client.submit(wf, metadata={"app": "frissonvision", "run": run}).result()
        bundle = fetch_json(job, "jobs.json")
        run_dir = bundle["run_dir"]
        print(f"run '{bundle['run_id']}': {len(bundle['jobs'])} jobs planned")

        # 2. renders
        run_all(client, [j["graph"] for j in bundle["jobs"]], "renders", run, a.timeout)

        # 3. take QC and re-rolls
        for p in range(a.retries + 1):
            qc = {"1": {"class_type": "FVQC", "inputs": {"run_dir": run_dir, "pass_index": p,
                                                         "max_passes": a.retries, "then_assemble": False,
                                                         "queue_rerolls": False}}}
            qj = client.submit(client.workflows.from_json(qc), metadata={"app": "frissonvision", "run": run}).result()
            q = fetch_json(qj, f"qc{p}.json")
            print(f"  {q['message']}")
            if q["final"] or not q["jobs"]:
                break
            run_all(client, [j["graph"] for j in q["jobs"]], f"re-rolls (pass {p + 1})", run, a.timeout)

        # 4. cut
        asm = {"1": {"class_type": "FVFreeMemory", "inputs": {"note": "before assembly"}}}
        client.submit(client.workflows.from_json(asm)).result()
        asm = {"1": {"class_type": "FVAssemble", "inputs": {"run_dir": run_dir, "filename_prefix": bundle["run_rel"],
                                                            "fx_intensity": a.fx, "finish": True}}}
        aj = client.submit(client.workflows.from_json(asm), metadata={"app": "frissonvision", "run": run}).result()
        saved = [o.to_file(out / o.name) for o in aj.outputs if o.name.lower().endswith((".mp4", ".mov", ".webm"))]
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    for f in saved:
        print(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
