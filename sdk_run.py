#!/usr/bin/env python3
"""OneShot MV, driven by the Comfy SDK: one song and one World line in, a music video out.

The Comfy SDK talks the Comfy API v2. Comfy Cloud doesn't install custom node packs yet,
so point it at your own ComfyUI through Comfy's open-source comfy-api-proxy, which gives a
local ComfyUI the same API. This script starts the proxy for you if it isn't running:

    pip install comfy-sdk comfy-api-proxy        (or run_sdk.bat on Windows, which does it)
    python sdk_run.py --song my_track.wav --theme "the gladiator games in ancient Rome" --seconds 30

Another deployment that runs the OneShot MV nodes: set COMFY_BASE_URL (and COMFY_API_KEY).

What happens:
  1. The song goes up as an asset, and a Director job analyses it (beats, sections, energy,
     stems), writes the World, casts the performers and plans every shot. The plan
     prints here. With --preview it stops there and renders nothing.
  2. The Director queues the run's own stages in ComfyUI (they jump the queue): portraits and
     b-roll stills, the vision check (re-makes bad stills), the lip-sync clips, take QC with
     SyncNet (re-rolls weak takes), then the cut with audio-reactive FX.
  3. A Run Result job waits behind those stages; when the run is finished it hands back the
     final MP4, which the SDK downloads.
  4. --recut RUN_DIR re-cuts a finished run at another FX level (no re-rendering).
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SDK_DIR = HERE / ".sdk"  # run_sdk.bat installs comfy-sdk and comfy-api-proxy here
if SDK_DIR.is_dir():
    sys.path.insert(0, str(SDK_DIR))
DIRECTOR = HERE / "api" / "OneShotMV_director_api.json"
RESULT = HERE / "api" / "OneShotMV_result_api.json"
ASSEMBLE = HERE / "api" / "OneShotMV_assemble_api.json"
LOCAL_PROXY = "http://127.0.0.1:8189"
CUTS = ["every 2 bars", "every bar", "to the music (subtle)", "to the music (intense)"]


def say(msg: str = "") -> None:
    print(msg, flush=True)


def healthy(base: str) -> bool:
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/api/v2/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_proxy(base: str, comfyui: str) -> None:
    """Start comfy-api-proxy beside the local ComfyUI unless one already answers at base."""
    if healthy(base):
        return
    port = base.rsplit(":", 1)[-1].strip("/")
    boot = ("import sys; sys.path.insert(0, %r); from comfy_api_proxy.cli import main; "
            "sys.exit(main(sys.argv[1:]))" % str(SDK_DIR))
    say(f"starting comfy-api-proxy on port {port} (in front of ComfyUI at {comfyui})")
    proc = subprocess.Popen([sys.executable, "-c", boot, "run", "--comfyui", comfyui, "--port", port,
                             "--max-upload-mb", "2000"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    atexit.register(proc.terminate)
    for _ in range(40):
        if healthy(base):
            return
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    sys.exit(f"comfy-api-proxy didn't start at {base}. Is ComfyUI running at {comfyui}?")


def node_id(wf, class_type: str) -> str:
    for nid, node in wf.json.items():
        if node.get("class_type") == class_type:
            return nid
    raise KeyError(class_type)


def fetch_json(job, name: str) -> dict:
    for o in job.outputs:
        if o.name == name or o.name.endswith(name):
            return json.loads(o.to_bytes().decode("utf-8"))
    raise RuntimeError(f"job {job.id} returned no {name}")


def videos(job) -> list:
    return [o for o in job.outputs if o.name.lower().endswith((".mp4", ".mov", ".webm"))]


def wait(job, label: str, t0: float, timeout: float):
    """Wait for a job, printing the elapsed time while it runs or waits in the queue."""
    while True:
        try:
            return job.wait(timeout=30)
        except TimeoutError:
            if time.time() - t0 > timeout:
                raise
            print(f"\r  {label}: {(time.time() - t0) / 60:5.1f} min", end="", flush=True)


def download(job, out: Path, prefix: str) -> list[Path]:
    saved = []
    for o in videos(job):
        saved.append(o.to_file(out / f"{prefix}{o.name}"))
    return saved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--song", help="a song you have the rights to use (wav, mp3, flac, m4a)")
    ap.add_argument("--theme", default="", help='the World in a line, e.g. "a 1965 airline: pilots, '
                    'stewardesses and passengers; the cabin, the tarmac; warm afternoon light"')
    ap.add_argument("--seconds", type=float, default=30.0, help="length of the video; 0 = the whole song")
    ap.add_argument("--start", type=float, default=-1.0, help="song time to start at; -1 = the strongest stretch")
    ap.add_argument("--format", default="16:9", choices=["16:9", "9:16", "1:1"])
    ap.add_argument("--preset", default="quality", choices=["quality", "fast"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fx", default="subtle", choices=["none", "subtle", "intense"],
                    help="audio-reactive effects (flash, zoom punch, glow, colour split)")
    ap.add_argument("--cuts-low", default=CUTS[0], choices=CUTS, help="cuts in low-energy sections")
    ap.add_argument("--cuts-medium", default=CUTS[1], choices=CUTS, help="cuts in medium-energy sections")
    ap.add_argument("--cuts-high", default=CUTS[2], choices=CUTS, help="cuts in high-energy sections")
    ap.add_argument("--cast-order", default="as written", choices=["as written", "mixed"])
    ap.add_argument("--crowd", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--broll", type=float, default=0.125, help="share of shots that become b-roll cutaways")
    ap.add_argument("--qc", type=int, default=2, help="take-QC re-roll rounds (0 = off)")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--preview", action="store_true", help="plan and print the shot list; render nothing")
    ap.add_argument("--recut", default="", metavar="RUN_DIR", help="re-cut a finished run at --fx")
    ap.add_argument("--out", default="oneshot_mv_out", help="where the MP4 is saved")
    ap.add_argument("--comfyui", default="http://127.0.0.1:8188", help="the ComfyUI the proxy fronts")
    ap.add_argument("--no-proxy", action="store_true", help="don't start comfy-api-proxy")
    ap.add_argument("--timeout", type=float, default=8.0, help="hours to wait for the run")
    a = ap.parse_args()
    if not a.song and not a.recut:
        ap.error("--song is required (or --recut RUN_DIR)")
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass

    if not os.environ.get("COMFY_BASE_URL", "").strip():
        os.environ["COMFY_BASE_URL"] = LOCAL_PROXY  # Comfy Cloud can't run custom node packs yet
    base = os.environ["COMFY_BASE_URL"]
    if not a.no_proxy and base.rstrip("/") == LOCAL_PROXY:
        ensure_proxy(base, a.comfyui)

    from comfy_sdk import Comfy

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    timeout = a.timeout * 3600
    with Comfy() as client:
        say(f"Comfy SDK -> {client!r}")

        if a.recut:  # re-cut a finished run at another FX level
            wf = client.workflows.from_file(ASSEMBLE)
            n = node_id(wf, "FVAssemble")
            wf.set_input(n, "run_dir", a.recut)
            wf.set_input(n, "audio_fx", a.fx)
            job = wait(client.submit(wf), "re-cut", t0, timeout)
            say()
            if job.status != "succeeded":
                say(f"re-cut failed: {job.error}")
                return 1
            for f in download(job, out, f"recut_{a.fx}_"):
                say(f"saved {f}")
            return 0

        # 1. the Director: song analysis, World, cast, shot plan
        wf = client.workflows.from_file(DIRECTOR)
        song, d = node_id(wf, "LoadAudio"), node_id(wf, "FVDirector")
        say(f"uploading {Path(a.song).name}")
        wf.set_input(song, "audio", client.assets.from_file(a.song))
        for k, v in (("theme", a.theme), ("seconds", a.seconds), ("start", a.start), ("format", a.format),
                     ("preset", a.preset), ("seed", a.seed), ("audio_fx", a.fx), ("cuts_low", a.cuts_low),
                     ("cuts_medium", a.cuts_medium), ("cuts_high", a.cuts_high), ("cast_order", a.cast_order),
                     ("crowd", a.crowd), ("broll_share", a.broll), ("qc_retries", a.qc),
                     ("run_name", a.run_name), ("queue_renders", not a.preview), ("queue_assemble", True)):
            wf.set_input(d, k, v)
        say("Director: analysing the song, writing the World, planning the shots")
        job = wait(client.submit(wf), "Director", t0, timeout)
        say()
        if job.status != "succeeded":
            say(f"the Director failed: {job.error}")
            return 1
        bundle = fetch_json(job, "jobs.json")
        say(bundle.get("summary") or f"run '{bundle['run_id']}' planned")
        say()
        if a.preview:
            say("Preview only: nothing was rendered. Change --theme if needed and run again "
                "without --preview (the song analysis and the World are cached).")
            return 0

        # 2-3. the run's stages render in ComfyUI; a Run Result job waits behind them
        run_dir = bundle["run_dir"]
        say(f"rendering run '{bundle['run_id']}' (the ComfyUI page shows each stage)")
        res = client.workflows.from_file(RESULT)
        rid = node_id(res, "FVRunResult")
        res.set_input(rid, "run_dir", run_dir)
        while True:
            job = wait(client.submit(res), "rendering", t0, timeout)
            if job.status != "succeeded":
                say(f"\nthe result check failed: {job.error}")
                return 1
            st = fetch_json(job, "status.json")
            print(f"\r  {st['message']}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
            if st["state"] == "done":
                break
            if st["state"] in ("stopped", "missing"):
                return 1
            if time.time() - t0 > timeout:
                say("timed out; the run may still finish in ComfyUI")
                return 1
            time.sleep(20)

        saved = download(job, out, f"{bundle['run_id']}_")
    say(f"done in {(time.time() - t0) / 60:.1f} min")
    for f in saved:
        say(f"saved {f}")
    other = "subtle" if a.fx == "intense" else "intense"
    say(f"re-cut it with other FX:  python sdk_run.py --recut \"{run_dir}\" --fx {other}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
