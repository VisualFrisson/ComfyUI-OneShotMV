#!/usr/bin/env python3
"""Run Frisson Vision through the Comfy SDK.

The SDK talks the Comfy API v2. Point it at:
  - your own ComfyUI, through the open-source comfy-api-proxy that runs beside it:
        pip install comfy-sdk comfy-api-proxy
        comfy-api-proxy --max-upload-mb 2000          # serves http://127.0.0.1:8189
        set COMFY_BASE_URL=http://127.0.0.1:8189      # (macOS/Linux: export ...)
  - or a Comfy API deployment of the Director workflow (COMFY_BASE_URL + COMFY_API_KEY).

    python sdk_run.py --song my_track.mp3 --pack crowd --format 9:16 --seconds 30
    python sdk_run.py --song track.wav --mode motion --dance me_dancing.mp4 --pack wardrobe --preset fast

Step 1 submits the Director (song analysis, plan, casting; it queues every render on the
server, grouped by model). Step 2 submits Assemble for the same run: it sits behind the
renders in the queue, so when it finishes the video is ready, and the SDK downloads it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIRECTOR = HERE / "workflows" / "FrissonVision_director_api.json"
ASSEMBLE = HERE / "workflows" / "FrissonVision_assemble_api.json"


def node_id(wf, class_type: str) -> str:
    for nid, node in wf.json.items():
        if node.get("class_type") == class_type:
            return nid
    raise KeyError(class_type)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--song", required=True, help="a song you have the rights to use")
    ap.add_argument("--mode", default="voice", choices=["voice", "motion"])
    ap.add_argument("--dance", default="", help="motion mode: the dance clip (path on the server)")
    ap.add_argument("--pack", default="crowd")
    ap.add_argument("--format", default="9:16", choices=["9:16", "1:1", "16:9"])
    ap.add_argument("--preset", default="quality", choices=["quality", "fast"])
    ap.add_argument("--seconds", type=float, default=30.0, help="0 = whole song")
    ap.add_argument("--start", type=float, default=-1.0, help="-1 = strongest stretch")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--twins", default="auto", choices=["auto", "loud", "all", "off"])
    ap.add_argument("--fx", type=float, default=1.0)
    ap.add_argument("--run-name", default="")
    ap.add_argument("--out", default="frissonvision_out")
    ap.add_argument("--timeout", type=float, default=6 * 3600)
    a = ap.parse_args()

    from comfy_sdk import Comfy

    run = a.run_name or f"sdk-{a.mode}-{a.pack}-{dt.datetime.now():%Y%m%d-%H%M%S}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Comfy() as client:
        wf = client.workflows.from_file(DIRECTOR)
        song, d = node_id(wf, "LoadAudio"), node_id(wf, "FVDirector")
        wf.set_input(song, "audio", client.assets.from_file(a.song))
        for k, v in (("mode", a.mode), ("pack", a.pack), ("format", a.format), ("preset", a.preset),
                     ("seconds", a.seconds), ("start", a.start), ("seed", a.seed), ("twins", a.twins),
                     ("run_name", run), ("dance_video_path", a.dance), ("queue_assemble", False)):
            wf.set_input(d, k, v)
        t0 = time.time()
        job = client.submit(wf)
        print(f"director job {job.id}: planning and queueing run '{run}' ...")
        job.result()
        asm = client.workflows.from_file(ASSEMBLE)
        n = node_id(asm, "FVAssemble")
        asm.set_input(n, "run_dir", f"frissonvision/runs/{run}")
        asm.set_input(n, "fx_intensity", a.fx)
        job2 = client.submit(asm)
        print(f"assemble job {job2.id}: waits behind the renders ...")
        job2.result(timeout=a.timeout) if "timeout" in job2.result.__code__.co_varnames else job2.result()
        saved = [o.to_file(out / o.name) for o in job2.outputs
                 if o.name.lower().endswith((".mp4", ".mov", ".webm"))]
        print(f"done in {(time.time() - t0) / 60:.1f} min")
        for p in saved:
            print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
