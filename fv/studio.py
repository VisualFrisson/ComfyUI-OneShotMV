"""OneShot MV Studio: the page (studio/ in the pack) and its small JSON API, served by ComfyUI
itself at /oneshot-mv/ (studio_routes.py). This module is reloaded on every request, so the
page and the API change without restarting ComfyUI.

API (all JSON):
  GET  api/system            GPU, queue, optional models
  GET  api/songs             audio files in ComfyUI's input folder
  GET  api/analyze?song=&seconds=&start=
                             duration, tempo, energy sections, a waveform, the part the run will use
  POST api/start             {song, theme, seconds, start, format, preset, fx, cuts_*, seed, render}
                             queues a Director job; returns its prompt id
  GET  api/job/<prompt id>   queued / running / done (with the run it made) / error
  GET  api/run/<run>         the plan, the World, every shot with its still, clip and state,
                             the stage, the notes from the vision check and take QC, the video
  POST api/recut             {run, fx}: a new cut at another FX level, nothing re-rendered
  POST api/cancel            {run}: removes the run's queued jobs and stops the one running
  GET  api/runs              recent runs, newest first
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import glob
import hashlib
import json
import os
import re
import subprocess
import urllib.parse

import numpy as np
from aiohttp import web

import folder_paths

AUDIO_EXT = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus", ".aiff", ".aif")
LEVEL = {"intro": "low", "break": "low", "outro": "low", "verse": "medium", "build": "medium",
         "chorus": "high", "drop": "high"}
WHY = {"people": "someone else in frame", "gender": "wrong gender", "beard": "unwanted beard", "glasses": "stray glasses",
       "head": "head cut off", "mono": "black and white", "crowd": "modern crowd", "text": "lettering", "modern": "modern items",
       "outfit": "wrong outfit", "unread": "unreadable", "frozen": "frozen", "exposure": "bad exposure", "bars": "black bars",
       "cut": "cut away", "noface": "no face", "headcut": "head out of frame", "sideways": "head sideways",
       "tooclose": "too close", "identity": "a different person", "drift": "drifted off the shot", "mouth": "mouth not moving",
       "sync": "weak lip sync", "missing": "missing"}


def _why(fs) -> str:
    return ", ".join(WHY.get(f, f) for f in fs)


STAGES = [("plan", "Plan", "song, World, shot list"), ("stills", "Stills", "Krea 2 portraits"),
          ("check", "Vision check", "re-makes bad stills"), ("clips", "Clips", "LTX lip sync"),
          ("takes", "Take check", "SyncNet, re-takes"), ("broll", "B-roll", "cutaways"),
          ("edit", "Edit", "beat cut, FX, grade")]


# ------------------------------------------------------------------ helpers
def _out() -> str:
    return folder_paths.get_output_directory()


def _runs_root() -> str:
    return os.path.join(_out(), "oneshot-mv", "runs")


def _view(path: str) -> str:
    """A file under ComfyUI's output or input folder as a /api/view URL."""
    for typ, root in (("output", _out()), ("input", folder_paths.get_input_directory())):
        try:
            rel = os.path.relpath(path, root)
        except ValueError:
            continue
        if not rel.startswith(".."):
            sub, name = os.path.split(rel)
            q = {"filename": name, "subfolder": sub.replace(os.sep, "/"), "type": typ}
            try:
                q["v"] = str(int(os.path.getmtime(path)))
            except OSError:
                pass
            return "/api/view?" + urllib.parse.urlencode(q)
    return ""


def _run_dir(name: str) -> str:
    name = os.path.basename(name.strip().replace("\\", "/").rstrip("/"))
    d = os.path.join(_runs_root(), name)
    if not name or not os.path.isfile(os.path.join(d, "plan.json")):
        raise web.HTTPNotFound(text=json.dumps({"error": f"no run '{name}'"}), content_type="application/json")
    return d


def _load(path: str, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _newest(paths: list[str]) -> str:
    paths = [p for p in paths if os.path.isfile(p)]
    return max(paths, key=os.path.getmtime) if paths else ""


def _clock(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60)}:{int(t % 60):02d}"


async def _blocking(fn, *a):
    return await asyncio.get_running_loop().run_in_executor(None, fn, *a)


def _queue_graph(graph: dict) -> str:
    from . import director
    return director.queue_prompt(graph)  # urllib to this server: only call from a worker thread


def _queue_lists():
    from server import PromptServer
    running, pending = PromptServer.instance.prompt_queue.get_current_queue()
    return running, pending


# ------------------------------------------------------------------ system and songs
def _system() -> dict:
    out = {"gpu": "", "vram_gb": 0, "queue": {"running": 0, "pending": 0}}
    try:
        import torch
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0).replace("NVIDIA ", "")
            out["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30)
    except Exception:
        pass
    try:
        running, pending = _queue_lists()
        out["queue"] = {"running": len(running), "pending": len(pending)}
    except Exception:
        pass
    try:
        from . import lyrics
        out["whisper"] = bool(lyrics.available())
    except Exception:
        out["whisper"] = False
    return out


def _songs() -> list[dict]:
    root = folder_paths.get_input_directory()
    found = []
    for dirpath, _dirs, files in os.walk(root):
        depth = os.path.relpath(dirpath, root).count(os.sep)
        if depth > 1:
            continue
        for f in files:
            if f.lower().endswith(AUDIO_EXT):
                p = os.path.join(dirpath, f)
                found.append({"name": os.path.relpath(p, root).replace(os.sep, "/"), "size": os.path.getsize(p),
                              "mtime": os.path.getmtime(p)})
    found.sort(key=lambda s: -s["mtime"])
    return found[:300]


def _song_path(name: str) -> str:
    root = folder_paths.get_input_directory()
    p = os.path.normpath(os.path.join(root, name))
    if not p.startswith(os.path.normpath(root)) or not os.path.isfile(p):
        raise web.HTTPNotFound(text=json.dumps({"error": f"no song '{name}' in the input folder"}),
                               content_type="application/json")
    return p


def _decode(path: str) -> tuple[np.ndarray, int]:
    if path.lower().endswith(".wav"):
        try:
            from . import stems
            return stems.read_wav(path)
        except Exception:
            pass
    from .assemble import ffmpeg_exe
    sr = 44100
    raw = subprocess.run([ffmpeg_exe(), "-v", "error", "-nostdin", "-i", path, "-f", "f32le", "-ac", "2",
                          "-ar", str(sr), "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32).reshape(-1, 2).T.copy()
    return x, sr


def _analysis(path: str) -> dict:
    """The song's timeline (cached per file), plus a 160-bar waveform."""
    st = os.stat(path)
    key = hashlib.sha1(f"{path}|{st.st_size}|{int(st.st_mtime)}|v1".encode()).hexdigest()[:16]
    cdir = os.path.join(_out(), "oneshot-mv", "studio_cache")
    cp = os.path.join(cdir, key + ".json")
    hit = _load(cp)
    if hit:
        return hit
    from . import beat
    x, sr = _decode(path)
    tl = beat.analyze(x, sr)
    mono = np.abs(x).mean(axis=0)
    n = 160
    edges = np.linspace(0, len(mono), n + 1).astype(int)
    env = np.array([float(np.sqrt(np.mean(mono[a:b] ** 2))) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])])
    env = (env / max(1e-9, float(env.max()))).round(3).tolist()
    keep = {k: tl[k] for k in ("duration", "tempo", "beats_per_bar", "sections", "drop", "downbeats", "beats") if k in tl}
    out = {"tl": keep, "env": env}
    os.makedirs(cdir, exist_ok=True)
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


def _analyze(name: str, seconds: float, start: float) -> dict:
    from . import plan as planner
    path = _song_path(name)
    a = _analysis(path)
    tl = a["tl"]
    t0, t1 = planner.choose(tl, float(seconds), float(start))
    secs = []
    for s in tl.get("sections") or []:
        lab = s.get("label", "")
        lvl = "high" if s.get("is_drop") else LEVEL.get(lab, "medium")
        if lab == "outro" and s.get("loud"):
            lvl = "high"
        secs.append({"start": round(s["start"], 2), "end": round(s["end"], 2), "label": lab, "level": lvl})
    drop = tl.get("drop") or {}
    return {"song": name, "duration": round(tl["duration"], 2), "tempo": round(float(tl.get("tempo") or 0), 1),
            "sections": secs, "env": a["env"], "drop": round(drop["time"], 2) if drop.get("time") else None,
            "window": [round(t0, 2), round(t1, 2)], "url": _view(path)}


# ------------------------------------------------------------------ start, job, recut, cancel
def _director_graph(b: dict) -> dict:
    cuts = {k: b.get(k) for k in ("cuts_low", "cuts_medium", "cuts_high") if b.get(k)}
    d = {"audio": ["1", 0], "mode": "voice", "pack": "crowd", "format": b.get("format", "16:9"),
         "preset": b.get("preset", "quality"), "seconds": float(b.get("seconds", 30)),
         "start": float(b.get("start", -1)), "seed": int(b.get("seed", 1)), "twins": "auto",
         "queue_renders": bool(b.get("render", False)), "theme": (b.get("theme") or "").strip(),
         "run_name": b.get("run_name", ""), "queue_assemble": True, "framing": "mixed",
         "broll_share": float(b.get("broll", 0.125)), "broll_style": "mix", "qc_retries": int(b.get("qc", 2)),
         "staged": True, "audio_fx": b.get("fx", "subtle"), "cast_order": b.get("cast_order", "as written"),
         "crowd": b.get("crowd", "auto"), **cuts}
    return {"1": {"class_type": "LoadAudio", "inputs": {"audio": b["song"]}},
            "2": {"class_type": "FVDirector", "inputs": d}}


def _job(pid: str) -> dict:
    from server import PromptServer
    q = PromptServer.instance.prompt_queue
    running, pending = q.get_current_queue()
    if any(x[1] == pid for x in running):
        return {"state": "running"}
    for i, x in enumerate(sorted(pending, key=lambda x: x[0])):
        if x[1] == pid:
            return {"state": "queued", "ahead": i + len(running)}
    h = q.get_history(prompt_id=pid).get(pid)
    if not h:
        return {"state": "unknown"}
    status = h.get("status") or {}
    if status.get("status_str") == "error":
        msg = next((d.get("exception_message", "") for ev, d in status.get("messages", [])
                    if ev == "execution_error" and isinstance(d, dict)), "")
        return {"state": "error", "error": msg.strip() or "the job failed (see the ComfyUI console)"}
    run = ""
    for node_out in (h.get("outputs") or {}).values():
        for it in node_out.get("text", []) or []:
            if isinstance(it, dict) and it.get("filename") == "jobs.json":
                run = it.get("subfolder", "").rstrip("/").split("/")[-1]
    return {"state": "done", "run": run}


def _run_jobs(name: str):
    running, pending = _queue_lists()
    mine = lambda x: name in json.dumps(x[2])
    return [x for x in running if mine(x)], [x for x in pending if mine(x)]


def _cancel(name: str) -> dict:
    from server import PromptServer
    running, pending = _run_jobs(name)
    q = PromptServer.instance.prompt_queue
    ids = {x[1] for x in pending}
    for _ in range(len(ids) + 5):  # delete_queue_item removes one match per call
        if not q.delete_queue_item(lambda a: a[1] in ids):
            break
    if running:
        try:
            import nodes as comfy_nodes
            comfy_nodes.interrupt_processing()
        except Exception:
            import comfy.model_management as mm
            mm.interrupt_current_processing()
    return {"removed": len(ids), "stopped": len(running)}


# ------------------------------------------------------------------ a run, for the page
def _who(p: dict, s: dict, cast: dict) -> dict:
    rs = [r for r in p.get("renders", []) if r.get("shot") == s["index"] and not r.get("unused")]
    br = next((r for r in rs if r.get("role") == "R"), None)
    a = next((r for r in rs if r.get("role") == "A"), None)
    if br is not None:
        ck = (br.get("broll") or {}).get("check") or {}
        return {"kind": "broll", "render": br, "role": ck.get("role") or "cutaway", "gender": ck.get("gender", ""),
                "age": ck.get("age", ""), "wear": ck.get("wear", ""), "place": ck.get("place", "")}
    if a is not None and a.get("performer") in cast:
        c = cast[a["performer"]]
        ck = c.get("check") or {}
        return {"kind": "sing", "render": a, "performer": a["performer"], "role": ck.get("role") or c.get("who", ""),
                "gender": ck.get("gender", ""), "age": ck.get("age", ""), "wear": ck.get("wear", ""),
                "place": ck.get("place", "")}
    return {"kind": "sing", "render": a, "role": "", "gender": "", "age": "", "wear": "", "place": ""}


def _still(d: str, info: dict, p: dict) -> str:
    r = info.get("render") or {}
    if info["kind"] == "broll":
        b = r.get("broll") or {}
        if b.get("still_job") is not None:
            got = _newest([x for x in glob.glob(os.path.join(d, f"k{r['id']:03d}R*.png")) if "_fit" not in x])
            if got:
                return got
        return r.get("key", "") if r.get("key") and os.path.isfile(r.get("key", "")) else ""
    if r.get("portrait") and os.path.isfile(r["portrait"]):
        return r["portrait"]
    if info.get("performer") is not None:
        n = info["performer"]
        cands = [x for x in glob.glob(os.path.join(d, f"p{n:03d}_*.png")) if "_fit" not in x]
        return _newest(cands)
    return ""


def _clip(d: str, r: dict) -> tuple[str, int]:
    if not r:
        return "", 0
    tag = "R" if r.get("role") == "R" else "A"
    takes = glob.glob(os.path.join(d, f"r{r['id']:03d}{tag}*.mp4"))
    if r.get("path") and os.path.isfile(r["path"]):
        return r["path"], len(takes)
    return _newest(takes), len(takes)


def _notes(d: str, p: dict, label_of: dict) -> list[dict]:
    notes = []
    for it in (p.get("critic") or {}).get("items", []):
        takes = it.get("takes") or []
        if len(takes) > 1:
            faults = sorted({f for t in takes[:-1] for f in (t.get("faults") or [])})
            who = label_of.get(("p", it.get("performer"))) or label_of.get(("r", it.get("render"))) or it.get("name")
            notes.append({"kind": "fix", "text": f"{who}: still re-made ({_why(faults) or 'failed the check'})."})
    qc = _load(os.path.join(d, "qc.json"), []) or []
    for ps in qc if isinstance(qc, list) else []:
        rep = ps.get("report") or []
        bad = [x for x in rep if not x.get("ok")]
        if rep:
            if bad:
                why = sorted({f for x in bad for f in (x.get("fail") or [])})
                notes.append({"kind": "redo", "text": f"Take check {ps.get('pass', 0) + 1}: {len(bad)} of {len(rep)} "
                                                      f"takes re-shot ({_why(why[:4])})."})
            else:
                notes.append({"kind": "ok", "text": f"Take check {ps.get('pass', 0) + 1}: all {len(rep)} takes kept."})
    return notes[-12:]


def _estimate_min(p: dict) -> float:
    dur = float(p.get("duration") or 30)
    per = 2.0 if p.get("preset") == "quality" else 1.0
    return 6 + per * dur


def _run(name: str) -> dict:
    d = _run_dir(name)
    p = _load(os.path.join(d, "plan.json"), {}) or {}
    th = p.get("theme") or {}
    cast = {c["id"]: c for c in p.get("cast") or []}
    tm = p.get("timing") or {}
    done = _load(os.path.join(d, "done.json"))
    if done and not os.path.isfile(done.get("video", "")):
        done = None
    shots, label_of, used = [], {}, set()
    n_stills = n_clips = n_sing = 0
    for s in p.get("shots", []):
        info = _who(p, s, cast)
        r = info.get("render") or {}
        still = _still(d, info, p)
        clip, takes = _clip(d, r)
        if info.get("performer") is not None:
            label_of[("p", info["performer"])] = info["role"] or f"shot {s['index'] + 1}"
            c = cast.get(info["performer"]) or {}
            if c.get("char") is not None:
                used.add(c["char"])
        if r:
            label_of[("r", r.get("id"))] = info["role"] or f"shot {s['index'] + 1}"
        n_stills += bool(still)
        if info["kind"] == "sing":
            n_sing += 1
            n_clips += bool(clip)
        state = "clip" if clip else ("still" if still else "planned")
        shots.append({"n": s["index"] + 1, "t0": round(s["t0"], 2), "t1": round(s["t1"], 2),
                      "time": _clock(float(p.get("song_t0", 0)) + s["t0"]), "len": round(s["t1"] - s["t0"], 1),
                      "energy": s.get("energy", ""), "size": s.get("size", ""), "kind": info["kind"],
                      "role": info["role"], "gender": info["gender"], "age": info["age"], "wear": info["wear"],
                      "place": info["place"], "still": _view(still) if still else "", "clip": _view(clip) if clip else "",
                      "takes": takes, "state": state})
    # where the run is
    queued_jobs = 0
    try:
        running, pending = _run_jobs(os.path.basename(d))
        queued_jobs = len(running) + len(pending)
    except Exception:
        pass
    started = tm.get("queued")
    keys = " ".join(tm.keys())
    if done:
        stage = "done"
    elif not started:
        stage = "preview"
    elif "qc_looks" in keys or (n_clips >= n_sing and "qc_leads" in keys and queued_jobs <= 3):
        stage = "broll" if queued_jobs > 1 else "edit"
    elif "qc_leads" in keys:
        stage = "takes"
    elif n_clips:
        stage = "clips"
    elif "critic" in keys:
        stage = "check"
    else:
        stage = "stills"
    order = [k for k, _, _ in STAGES]
    cur = order.index(stage) if stage in order else (len(order) if stage == "done" else 0)
    stages = [{"id": k, "name": nm, "meta": meta,
               "state": "done" if (stage == "done" or i < cur) else ("now" if i == cur and stage != "preview" else "next")}
              for i, (k, nm, meta) in enumerate(STAGES)]
    if stage == "preview":
        stages[0]["state"] = "done"
    for st_ in stages:
        if st_["id"] == "stills":
            st_["meta"] = f"{n_stills} of {len(shots)} made"
        if st_["id"] == "clips":
            st_["meta"] = f"{n_clips} of {n_sing} sung"
    weights = {"plan": 3, "stills": 12, "check": 5, "clips": 55, "takes": 15, "broll": 5, "edit": 5}
    if stage == "done":
        pct = 100
    elif stage == "preview":
        pct = 0
    else:
        pct = sum(weights[k] for k in order[:cur])
        if stage == "stills":
            pct += weights["stills"] * n_stills / max(1, len(shots))
        if stage == "clips":
            pct += weights["clips"] * n_clips / max(1, n_sing)
        pct = int(min(99, pct))
    elapsed = 0.0
    if started:
        try:
            t_end = _dt.datetime.fromisoformat(done["at"]) if done else _dt.datetime.now()
            elapsed = max(0.0, (t_end - _dt.datetime.fromisoformat(started)).total_seconds() / 60)
        except (ValueError, KeyError):
            pass
    est = _estimate_min(p)
    latest = _newest([x for x in glob.glob(os.path.join(d, "r*.mp4"))])
    chars = th.get("characters") or []
    left = [str(ch.get("role") or str(ch.get("who", "")).split(",")[0]) for i, ch in enumerate(chars) if i not in used]
    world = {"line": th.get("theme", ""), "headline": (th.get("theme", "").split(":")[0].split(",")[0]).strip(),
             "camera": th.get("camera", ""), "places": th.get("places") or [], "light": th.get("light") or [],
             "grade": th.get("grade") or {}, "characters": len(chars), "left_out": left[:30]}
    return {
        "run": os.path.basename(d), "stage": stage, "stages": stages, "pct": pct,
        "elapsed_min": round(elapsed, 1), "estimate_min": round(est), "left_min": round(max(0.0, est - elapsed)),
        "queued_jobs": queued_jobs, "stalled": bool(started and not done and queued_jobs == 0),
        "brief": {"format": p.get("format"), "preset": p.get("preset"), "seconds": round(float(p.get("duration") or 0), 1),
                  "from": _clock(p.get("song_t0", 0)), "to": _clock(p.get("song_t1", 0)), "tempo": p.get("tempo"),
                  "fx": p.get("fx_level") or "subtle", "cuts": p.get("cuts") or {}, "seed": p.get("seed"),
                  "crowd": p.get("crowd", "auto"), "cast_order": p.get("cast_order", "as written")},
        "world": world, "energy_map": p.get("energy_map") or [], "shots": shots,
        "notes": _notes(d, p, label_of),
        "latest": _view(latest) if latest else "",
        "video": _view(done["video"]) if done else "",
        "edl": _view(done.get("edl", "")) if done and done.get("edl") and os.path.isfile(done["edl"]) else "",
        "finished_at": (done or {}).get("at", ""),
        "song": _view(os.path.join(d, "song.wav")) if os.path.isfile(os.path.join(d, "song.wav")) else "",
    }


def _runs() -> list[dict]:
    root = _runs_root()
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*", "plan.json")), key=os.path.getmtime, reverse=True)[:60]:
        d = os.path.dirname(d)
        p = _load(os.path.join(d, "plan.json"), {}) or {}
        done = _load(os.path.join(d, "done.json"))
        th = p.get("theme") or {}
        thumb = _newest([x for x in glob.glob(os.path.join(d, "p0*_00001_.png")) if "_fit" not in x][:3])
        out.append({"run": os.path.basename(d), "world": (th.get("theme", "").split(":")[0].split(",")[0]).strip(),
                    "format": p.get("format"), "seconds": round(float(p.get("duration") or 0), 1),
                    "rendered": bool((p.get("timing") or {}).get("queued")),
                    "video": _view(done["video"]) if done and os.path.isfile(done.get("video", "")) else "",
                    "thumb": _view(thumb) if thumb else "",
                    "when": _dt.datetime.fromtimestamp(os.path.getmtime(d)).isoformat(timespec="minutes")})
    return out


# ------------------------------------------------------------------ the request handler
async def handle(request: web.Request, pack_dir: str) -> web.StreamResponse:
    tail = request.match_info.get("tail", "")
    if tail.startswith("api/"):
        return await _api(request, tail[4:])
    page_dir = os.path.join(pack_dir, "studio")
    path = os.path.normpath(os.path.join(page_dir, tail or "index.html"))
    if not path.startswith(os.path.normpath(page_dir)) or not os.path.isfile(path):
        path = os.path.join(page_dir, "index.html")
    resp = web.FileResponse(path)
    resp.headers["Cache-Control"] = "no-store"
    return resp


async def _api(request: web.Request, route: str) -> web.Response:
    q = request.query
    if route == "system":
        return web.json_response(await _blocking(_system))
    if route == "songs":
        return web.json_response(await _blocking(_songs))
    if route == "analyze":
        res = await _blocking(_analyze, q.get("song", ""), float(q.get("seconds", 30) or 30), float(q.get("start", -1) or -1))
        return web.json_response(res)
    if route == "runs":
        return web.json_response(await _blocking(_runs))
    if route.startswith("run/"):
        return web.json_response(await _blocking(_run, route[4:]))
    if route.startswith("job/"):
        return web.json_response(await _blocking(_job, route[4:]))
    if request.method == "POST":
        body = await request.json()
        if route == "start":
            if not body.get("song"):
                return web.json_response({"error": "pick a song first"}, status=400)
            _song_path(body["song"])
            pid = await _blocking(_queue_graph, _director_graph(body))
            return web.json_response({"prompt_id": pid})
        if route == "recut":
            d = _run_dir(body.get("run", ""))
            fx = body.get("fx", "subtle")
            if fx not in ("none", "subtle", "intense"):
                fx = "subtle"
            rel = os.path.relpath(d, _out()).replace(os.sep, "/")
            g = {"1": {"class_type": "FVAssemble", "inputs": {"run_dir": d, "filename_prefix": rel,
                                                            "fx_intensity": 1.0, "finish": True, "audio_fx": fx}}}
            pid = await _blocking(_queue_graph, g)
            return web.json_response({"prompt_id": pid})
        if route == "cancel":
            d = _run_dir(body.get("run", ""))
            return web.json_response(await _blocking(_cancel, os.path.basename(d)))
    return web.json_response({"error": f"unknown api '{route}'"}, status=404)
