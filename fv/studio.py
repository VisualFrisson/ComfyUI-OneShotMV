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
    key = hashlib.sha1(f"{path}|{st.st_size}|{int(st.st_mtime)}|v2".encode()).hexdigest()[:16]
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
    out = {"tl": tl, "env": env}
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


# ------------------------------------------------------------------ how long a preview takes
# Seconds per step of the Director's preview, measured on the RTX PRO 6000 (Oct 28: a new World
# for a 111 s window took 16 min, most of it writing 53 extra characters 8 at a time). A World
# already written for the same line and seed is reused, so a second preview is quick.
T = {"listen": 15, "load": 120, "world": 125, "roles": 85, "batch": 78, "first_chars": 12, "max_batches": 8,
     "gender": 55, "stems": 5, "stems_per_s": 0.25, "crowd": 45, "finish": 5}


def _bible(theme: str, seed: int) -> dict | None:
    if not theme.strip():
        return None
    from . import theme as themes
    return _load(os.path.join(themes.cache_dir(_out()), themes._key(theme, seed) + ".json"))


def _estimate(b: dict) -> dict:
    """The preview's steps and their expected seconds, from the song, the length and whether the
    World is already written for this line and seed."""
    import math
    from . import broll, casting, plan as planner
    path = _song_path(b["song"])
    tl = _analysis(path)["tl"]
    pk = casting.load_pack("crowd")
    cuts = {k[5:]: b[k] for k in ("cuts_low", "cuts_medium", "cuts_high") if b.get(k)}
    if cuts:
        pk["edit"] = dict(pk["edit"], **{f"cut_{lv}": m for lv, m in cuts.items()})
    p = planner.plan_voice(tl, pk, fmt=b.get("format", "16:9"), preset=b.get("preset", "quality"),
                           seconds=float(b.get("seconds", 30)), start=float(b.get("start", -1)),
                           seed=int(b.get("seed", 1)), framing="mixed")
    win = float(p["song_t1"] - p["song_t0"])
    theme = (b.get("theme") or "").strip()
    share = float(b.get("broll", 0.125))
    need = p["performers"] + 2 * broll.slots(p, share) + 2 if theme else 0
    bible = _bible(theme, int(b.get("seed", 1))) if theme else None
    have = len((bible or {}).get("characters") or [])
    parts = [{"id": "listen", "name": "Listening to the song", "s": T["listen"]}]
    if theme:
        batches = 0
        if not bible:
            have = T["first_chars"]
        if have < need:
            batches = min(T["max_batches"], math.ceil((need - have) / 8))
        llm = (not bible) or batches > 0 or not (bible or {}).get("genders_checked")
        if llm:
            parts.append({"id": "load", "name": "Loading the language model", "s": T["load"]})
        if not bible:
            parts.append({"id": "world", "name": "Writing the World", "s": T["world"]})
        if batches:
            parts.append({"id": "roles", "name": "Finding more characters", "s": T["roles"]})
            parts.append({"id": "cast", "name": f"Writing {min(need - have, 8 * batches)} more characters",
                          "s": T["batch"] * batches, "batches": batches})
        if llm:
            parts.append({"id": "gender", "name": "Checking every character", "s": T["gender"]})
    parts.append({"id": "stems", "name": "Separating the vocals", "s": round(T["stems"] + T["stems_per_s"] * win)})
    if theme and b.get("crowd", "auto") != "off":
        from . import theme as themes
        if (b.get("crowd") == "on" or themes.crowd_world(theme)) and not (bible or {}).get("crowd_dress"):
            parts.append({"id": "crowd", "name": "Dressing the crowd", "s": T["crowd"]})
    parts.append({"id": "finish", "name": "Planning every shot", "s": T["finish"]})
    total = sum(x["s"] for x in parts)
    return {"total_s": int(total), "parts": parts, "shots": len(p["shots"]), "need": need,
            "world_cached": bool(bible) and have >= need, "window": [round(p["song_t0"], 2), round(p["song_t1"], 2)],
            "render_min": round(6 + (2.0 if b.get("preset", "quality") == "quality" else 1.0) * win)}


def _project(d: str) -> str:
    """The project name the user gave in the Studio ("" for runs made elsewhere)."""
    return str((_load(os.path.join(d, "studio.json"), {}) or {}).get("project") or "")


def _file_stem(name: str) -> str:
    """A project name as a file name: letters, digits, spaces, dashes and underscores."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\- ]+", "", name)).strip()[:80]


def _jobs_dir() -> str:
    d = os.path.join(_out(), "oneshot-mv", "studio_jobs")
    os.makedirs(d, exist_ok=True)
    return d


def _progress(pid: str, rec: dict) -> dict:
    """Where a running Director is, read from its log lines, and the seconds left."""
    import time
    parts = (rec.get("estimate") or {}).get("parts") or []
    t_start = float(rec.get("t_run") or rec.get("t", time.time()))
    lines = []
    try:
        from app.logger import get_logs
        for e in list(get_logs() or []):
            t = e.get("t")
            try:
                ts = _dt.datetime.fromisoformat(t).timestamp() if isinstance(t, str) else float(t)
            except (TypeError, ValueError):
                continue
            if ts >= t_start - 1:
                lines.append((ts, str(e.get("m", ""))))
    except Exception:
        pass
    done, last_t, cast_n, batches_done = set(), t_start, 0, 0
    for ts, m in lines:
        if "Requested to load LTXAVTEModel" in m and "load" not in done:
            done.add("listen"); done.add("load"); last_t = ts
        if "[OneShotMV] Theme:" in m:
            done.update({"listen", "load", "world"}); last_t = ts
        if "new roles:" in m:
            done.update({"listen", "load", "world", "roles"}); last_t = ts
        mm = re.search(r"theme: cast (\d+) characters", m)
        if mm:
            batches_done += 1; cast_n = int(mm.group(1)); last_t = ts
        if "every role has its real gender" in m:
            done.update({"listen", "load", "world", "roles", "cast", "gender"}); last_t = ts
        if "voice plan:" in m:
            done.update({"listen", "load", "world", "roles", "cast", "gender"}); last_t = ts
        if "[OneShotMV] stems:" in m or "lyrics:" in m or "lyrics skipped" in m:  # older runs logged lyrics
            done.update({"listen", "load", "world", "roles", "cast", "gender", "stems"}); last_t = ts
    now = time.time()
    left, cur = 0.0, None
    for x in parts:
        if x["id"] in done:
            continue
        if x["id"] == "cast" and batches_done:
            left_b = max(0, int(x.get("batches", 1)) - batches_done)
            if left_b == 0:
                continue
            x = dict(x, s=left_b * T["batch"], name=f"Writing more characters ({cast_n} so far)")
        if cur is None:
            cur = x
            left += max(5.0, x["s"] - (now - last_t))
        else:
            left += x["s"]
    elapsed = now - t_start
    return {"elapsed_s": int(elapsed), "left_s": int(left), "step": (cur or {}).get("name", "Finishing the plan"),
            "total_s": int((rec.get("estimate") or {}).get("total_s") or 0)}


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
    rec = _load(os.path.join(_jobs_dir(), re.sub(r"[^0-9a-f-]", "", pid) + ".json"), {}) or {}
    if any(x[1] == pid for x in running):
        if not rec.get("t_run"):
            import time
            rec["t_run"] = time.time()
            try:
                with open(os.path.join(_jobs_dir(), re.sub(r"[^0-9a-f-]", "", pid) + ".json"), "w", encoding="utf-8") as f:
                    json.dump(rec, f)
            except OSError:
                pass
        out = {"state": "running"}
        if rec.get("estimate"):
            out.update(_progress(pid, rec))
        return out
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
    if run and rec.get("project"):
        try:
            sj = os.path.join(_run_dir(run), "studio.json")
            if not os.path.isfile(sj):
                with open(sj, "w", encoding="utf-8") as f:
                    json.dump({"project": rec["project"]}, f)
        except Exception:
            pass
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


def _notes(d: str, p: dict, label_of: dict, done: bool = False) -> list[dict]:
    notes = []
    for it in (p.get("critic") or {}).get("items", []):
        takes = it.get("takes") or []
        if len(takes) > 1:
            faults = sorted({f for t in takes[:-1] for f in (t.get("faults") or [])})
            nm = str(it.get("name") or "")
            who = label_of.get(("p", it.get("performer"))) or label_of.get(("r", it.get("render"))) or \
                ("A cutaway" if nm.startswith("k") else "A spare portrait")
            notes.append({"kind": "fix", "text": f"{who}: still re-made ({_why(faults) or 'failed the check'})."})
    qc = _load(os.path.join(d, "qc.json"), []) or []
    passes = [x for x in (qc if isinstance(qc, list) else []) if x.get("report")]
    for k, ps in enumerate(passes):
        rep = ps.get("report") or []
        bad = [x for x in rep if not x.get("ok")]
        if rep:
            if bad and done and k == len(passes) - 1:
                why = sorted({f for x in bad for f in (x.get("fail") or [])})
                notes.append({"kind": "redo", "text": f"Take check {ps.get('pass', 0) + 1}: {len(bad)} of {len(rep)} takes still "
                                                      f"weak ({_why(why[:4])}); kept the best take of each."})
            elif bad:
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
                      "takes": takes, "state": state, "render": r.get("id"), "pickup": int(r.get("pickup") or 0)})
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
        "run": os.path.basename(d), "project": _project(d), "stage": stage, "stages": stages, "pct": pct,
        "elapsed_min": round(elapsed, 1), "estimate_min": round(est), "left_min": 0 if done else round(max(0.0, est - elapsed)),
        "queued_jobs": queued_jobs, "stalled": bool(started and not done and queued_jobs == 0),
        "brief": {"format": p.get("format"), "preset": p.get("preset"), "seconds": round(float(p.get("duration") or 0), 1),
                  "from": _clock(p.get("song_t0", 0)), "to": _clock(p.get("song_t1", 0)), "tempo": p.get("tempo"),
                  "fx": p.get("fx_level") or "subtle", "cuts": p.get("cuts") or {}, "seed": p.get("seed"),
                  "crowd": p.get("crowd", "auto"), "cast_order": p.get("cast_order", "as written")},
        "world": world, "energy_map": p.get("energy_map") or [], "shots": shots,
        "notes": _notes(d, p, label_of, done=bool(done)),
        "latest": _view(latest) if latest else "",
        "video": _view(done["video"]) if done else "",
        "edl": _view(done.get("edl", "")) if done and done.get("edl") and os.path.isfile(done["edl"]) else "",
        "finished_at": (done or {}).get("at", ""),
        "song": _view(os.path.join(d, "song.wav")) if os.path.isfile(os.path.join(d, "song.wav")) else "",
        "versions": _versions(d, p), "pickup": _pickup_state(d, p, done),
        "finish": _finish_settings(p), "fx_now": (done or {}).get("fx") or p.get("fx_level") or "subtle",
        "fx_levels": (done or {}).get("fx_levels") or _fv("assemble").fx_levels(p),
        "looks": [{"id": "none", "name": "None"}] + [{"id": k, "name": _look_names().get(k, k)} for k in SHOWN_LOOKS],
        "looks_sig": _file_sig(_fv("finish").__file__),  # in the preview URLs, so browsers drop old tiles
    }


# ------------------------------------------------------------------ pickups: re-shoot chosen shots
# The Result screen's "fix a shot": a new take of the same singer (same still, new seed), or a
# new person (a new still from the portrait job with a new seed, then its clip), each with the
# user's note added to the prompt. They shoot together (stills first, then clips, so each model
# loads once), then the run is cut again. The cut before is kept as a version.
def _find_job(d: str, stem: str) -> str | None:
    """The job file that made <stem>_00001_.* (by its name, else by its save prefix)."""
    from . import director
    jf = director._job_file(d, stem)
    if jf:
        return jf
    jd = os.path.join(d, "jobs")
    for f in sorted(os.listdir(jd)) if os.path.isdir(jd) else []:
        g = _load(os.path.join(jd, f), {}) or {}
        for n in g.values() if isinstance(g, dict) else []:
            if isinstance(n, dict) and n.get("class_type") in ("SaveVideo", "SaveImage") and \
                    str((n.get("inputs") or {}).get("filename_prefix", "")).endswith("/" + stem):
                return os.path.join(jd, f)
    return None


def _stem(path: str) -> str:
    b = os.path.basename(str(path).replace("\\", "/"))
    b = b.rsplit("_00001_", 1)[0]
    return b[:-4] if b.endswith("_fit") else b


def _positive(g: dict) -> list[str]:
    """Ids of the text-encode nodes that feed a sampler's positive input."""
    ids = set()
    for n in g.values():
        ins = n.get("inputs") or {}
        for k in ("positive", "conditioning"):
            v = ins.get(k)
            if isinstance(v, list) and v and str(v[0]) in g and g[str(v[0])].get("class_type") == "CLIPTextEncode":
                ids.add(str(v[0]))
    if not ids:  # no wiring found: the longest prompt
        texts = [(len(str((n.get("inputs") or {}).get("text", ""))), k) for k, n in g.items()
                 if n.get("class_type") == "CLIPTextEncode"]
        if texts:
            ids.add(max(texts)[1])
    return sorted(ids)


DECADES = {"twenties": 25, "thirties": 35, "forties": 45, "fifties": 55, "sixties": 65, "seventies": 75,
           "eighties": 83, "nineties": 92}


def _note_age(note: str) -> int | None:
    m = re.search(r"\b(twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties)\b", note, re.I)
    if m:
        return DECADES[m.group(1).lower()]
    m = re.search(r"\b(?:aged?|age)\s*(\d{2})\b|\b(\d{2})\s*(?:-?\s*years?[ -]old|s\b)", note, re.I)
    if m:
        return int(m.group(1) or m.group(2))
    return None


def _with_note(g: dict, note: str) -> None:
    """The user's note right after the prompt's first sentence; an age in it replaces the
    prompt's own age (a note asking for a woman in her seventies beats "a woman aged 45")."""
    from . import director
    note = re.sub(r"\s+", " ", note or "").strip()
    if not note:
        return
    if not note.endswith((".", "!", "?")):
        note += "."
    age = _note_age(note)
    for k in _positive(g):
        ins = g[k]["inputs"]
        if isinstance(ins.get("text"), str):
            t = ins["text"]
            if age:
                t = re.sub(r"\baged \d{2}\b", f"aged {age}", t)
                dec = next((w for w, v in DECADES.items() if abs(v - age) <= 5), None)
                if dec:
                    t = re.sub(r"\b(in (?:his|her|their)) (?:twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties)\b",
                               lambda m: f"{m.group(1)} {dec}", t)
            ins["text"] = director._insert_fix(t, note[0].upper() + note[1:])


def _set_prefix(g: dict, cls: str, prefix: str) -> None:
    for n in g.values():
        if n.get("class_type") == cls:
            n["inputs"]["filename_prefix"] = prefix


def _versions(d: str, p: dict) -> list[dict]:
    vd = os.path.join(d, "versions")
    out = []
    for f in sorted(glob.glob(os.path.join(vd, "v*_*.mp4")), key=os.path.getmtime):
        out.append({"name": os.path.basename(f).split("_")[0], "url": _view(f),
                    "when": _dt.datetime.fromtimestamp(os.path.getmtime(f)).isoformat(timespec="minutes")})
    return out


def _pickup_state(d: str, p: dict, done: dict | None) -> dict:
    pk = (p.get("pickups") or [])
    if not pk:
        return {}
    last = pk[-1]
    ids = set(last.get("prompt_ids") or (_load(os.path.join(d, f"pickup{last['n']}_jobs.json"), {}) or {}).get("prompt_ids") or [])
    try:
        running, pending = _queue_lists()
        left = sum(1 for x in list(running) + list(pending) if x[1] in ids)
    except Exception:
        left = 0
    finished = bool(done) and str((done or {}).get("at", "")) >= str(last.get("at", ""))
    state = "done" if finished else ("running" if left else "stopped")
    total = len(ids)
    import time
    el = (time.time() - float(last.get("t", time.time()))) / 60
    est = float(last.get("estimate_min") or 0)
    return {"n": last["n"], "kind": last.get("kind", "pickups"), "state": state, "jobs": total, "jobs_left": left, "items": last.get("items", []),
            "elapsed_min": round(el, 1), "left_min": 0 if finished else round(max(0.5, est - el), 1)}


def _pickups(name: str, items: list, dry: bool = False) -> dict:
    import shutil
    import time
    from . import director, formats, graphs
    d = _run_dir(name)
    pj = os.path.join(d, "plan.json")
    p = _load(pj, {}) or {}
    if not items:
        raise ValueError("pick at least one shot")
    st = _pickup_state(d, p, _load(os.path.join(d, "done.json")))
    if st.get("state") == "running":
        raise ValueError("the last pickups are still shooting")
    if _run_jobs(os.path.basename(d))[1]:
        raise ValueError("this run still has jobs in the queue")
    rel = p.get("run_rel") or os.path.relpath(d, _out()).replace(os.sep, "/")
    k = len(p.get("pickups") or []) + 1
    shots = {s["index"]: s for s in p.get("shots", [])}
    stills, clips, changes = [], [], []
    delay0 = float(formats.PRESETS.get(p.get("preset", "quality"), {}).get("sync_delay", 0.0))
    for i, it in enumerate(items):
        idx = int(it.get("n", 0)) - 1
        mode = "recast" if it.get("mode") == "recast" else "take"
        note = str(it.get("note") or "")[:400]
        if idx not in shots:
            raise ValueError(f"no shot {idx + 1}")
        rs = [r for r in p["renders"] if r.get("shot") == idx and not r.get("unused")]
        r = next((x for x in rs if x.get("role") == "R"), None) or next((x for x in rs if x.get("role") == "A"), None)
        if r is None:
            raise ValueError(f"shot {idx + 1} has no clip to re-shoot")
        tagbase = f"r{r['id']:03d}{r['role']}"
        jf = _find_job(d, _stem(r.get("path", ""))) or _find_job(d, tagbase)
        if not jf:
            raise ValueError(f"shot {idx + 1}: its clip job is missing")
        g = director._bump(_load(jf, {}), 211 + 17 * k + i)
        tag = f"{tagbase}_u{k}"
        _set_prefix(g, "SaveVideo", f"{rel}/{tag}")
        change = {"shot": idx + 1, "render": r["id"], "mode": mode, "note": note,
                  "before": {"path": r.get("path"), "portrait": r.get("portrait"), "key": r.get("key")}}
        if mode == "recast":
            cur = r.get("key") if r["role"] == "R" else r.get("portrait")
            if not cur:  # the still the clip job reads
                cur = next((n["inputs"].get("path") for n in g.values() if n.get("class_type") == "FVLoadImagePath"), "")
            sj = _find_job(d, _stem(cur)) if cur else None
            if not sj:
                raise ValueError(f"shot {idx + 1}: its still's job is missing")
            gs = director._bump(_load(sj, {}), 307 + 17 * k + i)
            sbase = _stem(cur).split("_")[0]
            stag = f"{sbase}_u{k}"
            _set_prefix(gs, "SaveImage", f"{rel}/{stag}")
            _with_note(gs, note)
            new_still = os.path.join(d, f"{stag}_00001_.png")
            for n in g.values():
                if n.get("class_type") == "FVLoadImagePath":
                    n["inputs"]["path"] = new_still
            stills.append(gs)
            if not dry:
                json.dump(gs, open(os.path.join(d, "jobs", f"pickup{k}_{stag}.json"), "w", encoding="utf-8"), indent=1)
            if r["role"] == "R":
                r["key"] = new_still
            else:
                r["portrait"] = new_still
            change["still"] = new_still
        _with_note(g, note)
        if not dry:
            json.dump(g, open(os.path.join(d, "jobs", f"pickup{k}_{tag}.json"), "w", encoding="utf-8"), indent=1)
        clips.append(g)
        new_clip = os.path.join(d, f"{tag}_00001_.mp4")
        if "path_before" not in r or os.path.exists(r.get("path", "")):
            r["path_before"] = r.get("path")
            if "sync_delay" in r:
                r["sync_delay_before"] = r["sync_delay"]
        r["path"] = new_clip
        r["sync_delay"] = r.get("sync_delay_before", delay0) if mode == "take" else delay0
        r["pickup"] = k
        change["clip"] = new_clip
        changes.append(change)
    if dry:  # what would be queued, for checking (nothing is written or queued)
        def show(g):
            out = {}
            for n in g.values():
                c, ins = n.get("class_type"), n.get("inputs") or {}
                if c in ("SaveVideo", "SaveImage"):
                    out["save"] = ins.get("filename_prefix")
                if c == "FVLoadImagePath":
                    out["image"] = ins.get("path")
                    out["image_exists"] = os.path.exists(ins.get("path", ""))
            out["positive"] = [str(g[k]["inputs"].get("text", ""))[:260] for k in _positive(g)]
            return out
        return {"dry": True, "pickup": k, "stills": [show(g) for g in stills], "clips": [show(g) for g in clips],
                "changes": changes}
    # keep the cut as it is now
    done = _load(os.path.join(d, "done.json")) or {}
    if done.get("video") and os.path.isfile(done["video"]):
        os.makedirs(os.path.join(d, "versions"), exist_ok=True)
        nv = len(glob.glob(os.path.join(d, "versions", "v*_*.mp4"))) + 1
        shutil.copy2(done["video"], os.path.join(d, "versions", f"v{nv}_{os.path.basename(done['video'])}"))
    chain = []
    if stills:
        chain += [graphs.free_memory("before pickup stills").nodes] + stills
    chain += [graphs.free_memory("before pickup clips").nodes] + clips
    chain += [graphs.free_memory("before assembly").nodes,
              graphs.assemble(run_dir=d, prefix=rel).nodes]
    est = (3.0 + 0.5 * len(stills) if stills else 0.0) + 2.0 + 1.2 * len(clips) + \
        1.0 + 0.035 * float(p.get("duration") or 30)
    p.setdefault("pickups", []).append({"n": k, "at": _dt.datetime.now().isoformat(timespec="seconds"),
                                        "t": time.time(), "items": changes, "estimate_min": round(est, 1)})
    json.dump(p, open(pj, "w", encoding="utf-8"), indent=1)  # written in full before any job can read it
    # queued in order at the back: front-queueing a chain from an idle server lets its first
    # POST (the cut) start before the rest are queued (Oct 28: the cut ran before the clip)
    ids = [director.queue_prompt(g) for g in chain]
    json.dump({"prompt_ids": ids}, open(os.path.join(d, f"pickup{k}_jobs.json"), "w", encoding="utf-8"))
    return {"pickup": k, "jobs": len(chain), "estimate_min": round(est, 1)}


# ------------------------------------------------------------------ soft frames: re-decode older clips
def _decode_fix(g: dict) -> int:
    """Set every tiled decode in a clip job to one temporal tile (graphs.ltx_render since Oct 28).
    Returns how many decodes changed; 0 means the job already decodes in one tile."""
    frames = 0
    for n in g.values():
        if n.get("class_type") == "EmptyLTXVLatentVideo" and isinstance(n["inputs"].get("length"), int):
            frames = n["inputs"]["length"]
    want = min(512, (frames + 7) // 8 * 8 + 8) if frames else 512
    k = 0
    for n in g.values():
        if n.get("class_type") == "VAEDecodeTiled":
            ins = n["inputs"]
            if isinstance(ins.get("temporal_size", 64), int) and ins.get("temporal_size", 64) < want:
                ins["temporal_size"] = want
                k += 1
    return k


def _soft_clips(d: str, p: dict) -> list:
    """The clips in the cut that were decoded in 64-frame tiles: each has a soft frame where two
    tiles meet (frames 56, 112... of a 50 fps clip). [(render, job file)]"""
    used = {e["render"] for e in p.get("edl", [])}
    out = []
    for r in p.get("renders", []):
        if r["id"] not in used or not r.get("path") or _stem(r["path"]).endswith("_sd"):
            continue
        jf = _find_job(d, _stem(r["path"]))
        g = _load(jf, {}) if jf else None
        if g and _decode_fix(g):
            out.append((r, jf))
    return out


def _sharpen(name: str, dry: bool = False) -> dict:
    """Re-run each soft clip's own job (same seed, so the same take) with the clip decoded in one
    temporal tile, then cut again. Nothing else in the plan changes."""
    import shutil
    import time
    from . import director, graphs
    d = _run_dir(name)
    pj = os.path.join(d, "plan.json")
    p = _load(pj, {}) or {}
    st = _pickup_state(d, p, _load(os.path.join(d, "done.json")))
    if st.get("state") == "running":
        raise ValueError("the last pickups are still shooting")
    if _run_jobs(os.path.basename(d))[1]:
        raise ValueError("this run still has jobs in the queue")
    soft = _soft_clips(d, p)
    if not soft:
        raise ValueError("every clip in this cut is already sharp")
    rel = p.get("run_rel") or os.path.relpath(d, _out()).replace(os.sep, "/")
    k = len(p.get("pickups") or []) + 1
    byid = {r["id"]: r for r in p["renders"]}
    clips, changes = [], []
    for r0, jf in soft:
        r = byid[r0["id"]]
        g = _load(jf, {})
        _decode_fix(g)
        tag = _stem(r["path"]) + "_sd"
        _set_prefix(g, "SaveVideo", f"{rel}/{tag}")
        clips.append(g)
        new_clip = os.path.join(d, f"{tag}_00001_.mp4")
        changes.append({"shot": r.get("shot", 0) + 1, "render": r["id"], "mode": "sharpen", "clip": new_clip,
                        "before": {"path": r.get("path")}})
        if dry:
            continue
        json.dump(g, open(os.path.join(d, "jobs", f"sharpen{k}_{tag}.json"), "w", encoding="utf-8"), indent=1)
        if "path_before" not in r or os.path.exists(r.get("path", "")):
            r["path_before"] = r.get("path")
            if "sync_delay" in r:
                r["sync_delay_before"] = r["sync_delay"]
        r["path"] = new_clip  # the same take: its sync_delay stays
    est = 3.0 + 1.1 * len(clips) + 1.0 + 0.035 * float(p.get("duration") or 30)
    if dry:
        return {"dry": True, "clips": len(clips), "estimate_min": round(est, 1),
                "prefixes": [c["clip"] for c in changes][:5]}
    done = _load(os.path.join(d, "done.json")) or {}
    if done.get("video") and os.path.isfile(done["video"]):
        os.makedirs(os.path.join(d, "versions"), exist_ok=True)
        nv = len(glob.glob(os.path.join(d, "versions", "v*_*.mp4"))) + 1
        shutil.copy2(done["video"], os.path.join(d, "versions", f"v{nv}_{os.path.basename(done['video'])}"))
    chain = [graphs.free_memory("before re-decoding").nodes] + clips + \
        [graphs.free_memory("before assembly").nodes, graphs.assemble(run_dir=d, prefix=rel).nodes]
    p.setdefault("pickups", []).append({"n": k, "kind": "sharpen", "at": _dt.datetime.now().isoformat(timespec="seconds"),
                                        "t": time.time(), "items": changes, "estimate_min": round(est, 1)})
    json.dump(p, open(pj, "w", encoding="utf-8"), indent=1)
    ids = [director.queue_prompt(g) for g in chain]
    json.dump({"prompt_ids": ids}, open(os.path.join(d, f"pickup{k}_jobs.json"), "w", encoding="utf-8"))
    return {"pickup": k, "jobs": len(chain), "clips": len(clips), "estimate_min": round(est, 1)}


# ------------------------------------------------------------------ the finish: grade, grain, vignette
def _fv(name: str):
    """A core module, reloaded when its file changed: edits show up here without a restart
    (the nodes reload the same modules each time they run)."""
    import importlib
    mod = importlib.import_module("." + name, __package__)
    try:
        sig = _file_sig(mod.__file__)
        if getattr(mod, "_studio_sig", None) != sig:
            mod = importlib.reload(mod)
            mod._studio_sig = sig
    except Exception as e:  # keep the loaded module if an edit is broken
        print(f"[OneShotMV] studio reload {name} failed: {e}")
    return mod


# the 3x3 grid on the Result screen, in this order (from the user's ratings, Oct 28)
SHOWN_LOOKS = ["neon", "technicolor", "teal_orange",    # vivid
               "golden", "romantic", "eterna",          # warm and soft
               "lowkey", "matrix", "silver"]            # moody


def _look_names() -> dict:
    return _fv("finish").LOOKS


def _finish_cfg(p: dict) -> dict:
    """The finish the cut uses: the pack's, the World's grade, then the Studio's choices."""
    from . import casting
    cfg = dict(casting.load_pack(p.get("pack", "crowd")).get("finish") or {})
    if p.get("finish"):
        cfg = dict(p["finish"])
    cfg.update(_fv("finish").STUDIO_DEFAULTS)  # the Studio's starting settings
    cfg.update(p.get("finish_user") or {})
    return cfg


def _finish_settings(p: dict) -> dict:
    c = _finish_cfg(p)
    return {"look": c.get("look", "none"), "grain": round(float(c.get("grain", 0.05)), 3),
            "grain_size": round(float(c.get("grain_size", 0.5)), 2), "vignette": round(float(c.get("vignette", 0.0)), 3)}


def _preview_frame(d: str, p: dict, shot: int | None):
    """A frame (render size, 0..1 tensor) from the middle of a sung shot's clip."""
    import numpy as np_
    import torch
    from . import formats
    from .assemble import ClipReader
    rw, rh = formats.render_size(p["format"], p["preset"])
    shots = p.get("shots", [])
    if shot is None:  # a close-up in the loudest part, else the first sung shot
        cands = [s for s in shots if s.get("energy") == "high" and s.get("size") == "close"] or shots
        shot = cands[len(cands) // 2]["index"] if cands else 0
    rs = [r for r in p["renders"] if r.get("shot") == shot and r.get("role") == "A" and not r.get("unused")] or \
        [r for r in p["renders"] if r.get("role") == "A"]
    r = rs[0]
    pth = r.get("path", "")
    if not os.path.exists(pth) and r.get("path_before"):
        pth = r["path_before"]
    rd = ClipReader(pth, (rw, rh))
    f = rd.get(rd.n // 2) if rd.ok else None
    if f is None:
        raise ValueError("no frame to preview")
    return torch.from_numpy(np_.ascontiguousarray(f)).float() / 255, shot


def _lab_cache(d: str, p: dict):
    import torch
    from . import formats
    from .assemble import ClipReader
    lab_stats = _fv("finish").lab_stats
    cp = os.path.join(d, "studio_lab.json")
    hit = _load(cp)
    if hit:
        return torch.tensor(hit["mean"]), torch.tensor(hit["std"])
    rw, rh = formats.render_size(p["format"], p["preset"])
    samples = []
    for r in p["renders"][:: max(1, len(p["renders"]) // 12)]:
        rd = ClipReader(r.get("path", ""), (rw, rh))
        if rd.ok:
            f = rd.get(rd.n // 2)
            if f is not None:
                samples.append(torch.from_numpy(np.ascontiguousarray(f)).float() / 255)
    if not samples:
        return None
    m, sd = lab_stats(samples)
    json.dump({"mean": m.tolist(), "std": sd.tolist()}, open(cp, "w"))
    return m, sd


def _graded(d: str, p: dict, frame, look: str, overrides: dict | None = None, w: int = 560):
    import torch
    import torch.nn.functional as F
    from . import formats
    Finisher = _fv("finish").Finisher
    cfg = _finish_cfg(p)
    cfg.update(overrides or {})
    cfg["look"] = look
    y = Finisher(cfg, _lab_cache(d, p), seed=3)(frame, 0)
    rw, rh = formats.render_size(p["format"], p["preset"])
    cx, cy, cw, ch = formats.deliver_crop(p["format"], p["preset"])
    if (cw, ch) != (rw, rh):
        y = y[cy:cy + ch, cx:cx + cw]
    w = min(w, y.shape[1])
    h = int(round(w * y.shape[0] / y.shape[1]))
    y = F.interpolate(y.permute(2, 0, 1)[None], size=(h, w), mode="bilinear", align_corners=False, antialias=True)[0]
    return (y.permute(1, 2, 0).clamp(0, 1) * 255 + 0.5).byte().numpy()


def _file_sig(path: str) -> str:
    """A file's content hash: deploys can keep a file's old modification time (tar restores it)."""
    return hashlib.md5(open(path, "rb").read()).hexdigest()[:10]


def _look_preview(name: str, look: str, shot: int | None, w: int = 560) -> str:
    from PIL import Image
    if look not in _look_names():
        look = "none"
    d = _run_dir(name)
    p = _load(os.path.join(d, "plan.json"), {}) or {}
    w = max(240, min(1600, int(w)))
    # the grade alone: grain and vignette come after it, with their own sliders
    key = hashlib.sha1(json.dumps([look, shot, w, "grade-only",
                                   (p.get("finish") or {}).get("grade_mean"),
                                   _file_sig(_fv("finish").__file__)]).encode()).hexdigest()[:12]
    out = os.path.join(d, "studio_previews", f"{look}_{key}.jpg")
    if not os.path.exists(out):
        frame, _ = _preview_frame(d, p, shot)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        Image.fromarray(_graded(d, p, frame, look, {"grain": 0, "vignette": 0}, w=w)).save(out, quality=88)
    return out


def _look_sheet(name: str) -> str:
    """Every look on three frames of the run, for choosing which four the Result screen shows."""
    from PIL import Image, ImageDraw
    LOOKS = _fv("finish").LOOKS
    d = _run_dir(name)
    p = _load(os.path.join(d, "plan.json"), {}) or {}
    shots = [s for s in p.get("shots", []) if any(r.get("shot") == s["index"] and r.get("role") == "A" for r in p["renders"])]
    pick = [shots[len(shots) // 6]["index"], shots[len(shots) // 2]["index"], shots[(5 * len(shots)) // 6]["index"]]
    frames = [_preview_frame(d, p, i)[0] for i in pick]
    looks = list(LOOKS)
    tiles = [[_graded(d, p, f, lk, {"grain": 0, "vignette": 0}) for f in frames] for lk in looks]
    th, tw = tiles[0][0].shape[:2]
    lab = 34
    sheet = Image.new("RGB", (tw * 3, (th + lab) * len(looks)), (13, 16, 21))
    dr = ImageDraw.Draw(sheet)
    for i, lk in enumerate(looks):
        y0 = i * (th + lab)
        dr.text((10, y0 + 9), f"{i}. {LOOKS[lk]}  ({lk})", fill=(238, 240, 244))
        for j, t in enumerate(tiles[i]):
            sheet.paste(Image.fromarray(t), (j * tw, y0 + lab))
    out = os.path.join(d, "studio_previews", "look_sheet.jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sheet.save(out, quality=84)
    return out


def _grain_preview(name: str, look: str, grain: float, size: float, shot: int | None = None, cw_px: int = 640) -> str:
    """A 1:1 crop of a delivery-size frame with the grain on, made the way the cut makes it:
    the grade at render size, the upscale, then the grain."""
    import torch.nn.functional as F
    from PIL import Image
    from . import formats
    if look not in _look_names():
        look = "none"
    grain = round(min(0.2, max(0.0, float(grain))), 3)
    size = round(min(4.0, max(0.5, float(size))) * 2) / 2
    cw_px = max(160, min(960, int(cw_px)))
    d = _run_dir(name)
    p = _load(os.path.join(d, "plan.json"), {}) or {}
    key = hashlib.sha1(json.dumps([look, grain, size, shot, cw_px, _file_sig(_fv("finish").__file__),
                                   (p.get("finish") or {}).get("grade_mean")]).encode()).hexdigest()[:12]
    out = os.path.join(d, "studio_previews", f"grain_{key}.jpg")
    if os.path.exists(out):
        return out
    frame, _ = _preview_frame(d, p, shot)
    cfg = _finish_cfg(p)
    cfg.update({"look": look, "grain": grain, "grain_size": size, "vignette": 0})
    fin = _fv("finish").Finisher(cfg, _lab_cache(d, p), seed=3)
    y = fin.grade(frame)
    rw, rh = formats.render_size(p["format"], p["preset"])
    cx, cy, cw, ch = formats.deliver_crop(p["format"], p["preset"])
    dw, dh = formats.deliver_size(p["format"])
    y = y[cy:cy + ch, cx:cx + cw]
    if (dw, dh) != (cw, ch):
        y = F.interpolate(y.permute(2, 0, 1)[None], size=(dh, dw), mode="bicubic", align_corners=False,
                          antialias=True)[0].permute(1, 2, 0).clamp(0, 1)
    y = fin.texture(y, 0)
    w = min(cw_px, dw)  # the middle of the frame, pixel for pixel at the width it is shown
    h = min(int(round(w * 9 / 16)), dh)
    x0, y0 = (dw - w) // 2, max(0, (dh - h) // 2 - dh // 10)
    crop = (y[y0:y0 + h, x0:x0 + w].clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Image.fromarray(crop).save(out, quality=95)
    return out


def _xml(name: str) -> str:
    export = _fv("export")
    d = _run_dir(name)
    p = _load(os.path.join(d, "plan.json"), {}) or {}
    done = _load(os.path.join(d, "done.json")) or {}
    return export.write_xml(p, os.path.join(d, "final.xml"), done.get("video", ""), os.path.join(d, "song.wav"))


def _runs() -> list[dict]:
    root = _runs_root()
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*", "plan.json")), key=os.path.getmtime, reverse=True)[:60]:
        d = os.path.dirname(d)
        p = _load(os.path.join(d, "plan.json"), {}) or {}
        done = _load(os.path.join(d, "done.json"))
        th = p.get("theme") or {}
        thumb = _newest([x for x in glob.glob(os.path.join(d, "p0*_00001_.png")) if "_fit" not in x][:3])
        out.append({"run": os.path.basename(d), "project": _project(d), "world": (th.get("theme", "").split(":")[0].split(",")[0]).strip(),
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
    if route == "estimate":
        b = {k: q.get(k) for k in q}
        if not b.get("song"):
            return web.json_response({"error": "pick a song first"}, status=400)
        return web.json_response(await _blocking(_estimate, b))
    if route == "runs":
        return web.json_response(await _blocking(_runs))
    if route.startswith("xml/"):
        path = await _blocking(_xml, route[4:])
        stem = _file_stem(_project(os.path.dirname(path))) or os.path.basename(os.path.dirname(path))
        return web.FileResponse(path, headers={"Content-Disposition": f'attachment; filename="{stem}.xml"',
                                                "Cache-Control": "no-store"})
    if route.startswith("look/"):
        shot = q.get("shot")
        path = await _blocking(_look_preview, route[5:], q.get("look", "none"), int(shot) if shot not in (None, "") else None,
                               int(q.get("w") or 560))
        return web.FileResponse(path, headers={"Cache-Control": "max-age=3600"})
    if route.startswith("grain/"):
        shot = q.get("shot")
        path = await _blocking(_grain_preview, route[6:], q.get("look", "none"), q.get("g", 0.05), q.get("s", 1.0),
                               int(shot) if shot not in (None, "") else None, int(q.get("w") or 640))
        return web.FileResponse(path, headers={"Cache-Control": "max-age=3600"})
    if route.startswith("looksheet/"):
        path = await _blocking(_look_sheet, route[10:])
        return web.FileResponse(path, headers={"Cache-Control": "no-store"})
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
            try:
                est = await _blocking(_estimate, body)
            except Exception as e:
                est = {"error": str(e)}
            project = str(body.get("name") or "").strip()[:80]
            if project and not body.get("run_name"):
                slug = re.sub(r"[^a-z0-9]+", "-", project.lower()).strip("-")[:40] or "video"
                body["run_name"] = f"{slug}-{_dt.datetime.now().strftime('%Y%m%d-%H%M')}"
            pid = await _blocking(_queue_graph, _director_graph(body))
            import time
            with open(os.path.join(_jobs_dir(), re.sub(r"[^0-9a-f-]", "", pid) + ".json"), "w", encoding="utf-8") as f:
                json.dump({"t": time.time(), "estimate": est, "render": bool(body.get("render")), "project": project}, f)
            return web.json_response({"prompt_id": pid, "estimate": est})
        if route == "recut":
            d = _run_dir(body.get("run", ""))
            ok = ("none", "subtle", "intense")
            fx = body.get("fx", "subtle")  # one level, or {low, medium, high}: one per music energy
            if isinstance(fx, dict):
                fxl = {k: fx.get(k) if fx.get(k) in ok else "subtle" for k in ("low", "medium", "high")}
            else:
                fxl = {k: fx if fx in ok else "subtle" for k in ("low", "medium", "high")}
            # the choice stays with the run: the screen shows it, and later cuts (pickups) keep it
            pj = os.path.join(d, "plan.json")
            plan_ = _load(pj, {}) or {}
            if plan_:
                plan_["fx_levels"] = fxl
                plan_["fx_level"] = fxl["medium"]
                fu = dict(plan_.get("finish_user") or {})
                if body.get("look") is not None:
                    fu["look"] = str(body["look"]) if str(body["look"]) in _look_names() else "none"
                for k, lo, hi in (("grain", 0.0, 0.2), ("grain_size", 0.5, 4.0), ("vignette", 0.0, 0.95)):
                    if body.get(k) is not None:
                        fu[k] = round(min(hi, max(lo, float(body[k]))), 3)
                plan_["finish_user"] = fu
                json.dump(plan_, open(pj, "w", encoding="utf-8"), indent=1)
            rel = os.path.relpath(d, _out()).replace(os.sep, "/")
            g = {"1": {"class_type": "FVAssemble", "inputs": {"run_dir": d, "filename_prefix": rel,
                                                            "fx_intensity": 1.0, "finish": True, "audio_fx": "from the run"}}}
            pid = await _blocking(_queue_graph, g)
            return web.json_response({"prompt_id": pid})
        if route == "sharpen":
            try:
                res = await _blocking(_sharpen, body.get("run", ""), bool(body.get("dry")))
            except ValueError as e:
                return web.json_response({"error": str(e)}, status=400)
            return web.json_response(res)
        if route == "pickups":
            try:
                res = await _blocking(_pickups, body.get("run", ""), body.get("items") or [], bool(body.get("dry")))
            except ValueError as e:
                return web.json_response({"error": str(e)}, status=400)
            return web.json_response(res)
        if route == "cancel":
            d = _run_dir(body.get("run", ""))
            return web.json_response(await _blocking(_cancel, os.path.basename(d)))
    return web.json_response({"error": f"unknown api '{route}'"}, status=404)
