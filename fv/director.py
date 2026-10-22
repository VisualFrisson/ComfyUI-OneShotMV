"""The Director: song (+ dance clip) -> analysis, stems, plan, casting -> a queue of ComfyUI jobs.

Jobs are queued grouped by model so each model loads once per run:
  voice : Krea 2 portraits -> LTX lip-sync renders (A) -> twin keys -> LTX twin renders (B) -> assemble
  motion: Klein character keys -> LTX pose-guided renders -> assemble
Every job is also written to <run>/jobs/ so a single shot can be re-rolled by re-queueing it."""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import urllib.request

import numpy as np

from . import beat, casting, formats, graphs, plan as planner, stems


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-")[:80]


def server_url() -> str:
    try:
        from comfy.cli_args import args
        port = args.port
    except Exception:
        port = 8188
    return f"http://127.0.0.1:{port}"


def queue_prompt(graph: dict, url: str | None = None, client_id: str = "frissonvision", front: bool = False) -> str:
    """front=True jumps the queue. ComfyUI runs the *latest* front-queued prompt first, so queue
    a chain that must run in order with queue_chain_front()."""
    body = {"prompt": graph, "client_id": client_id}
    if front:
        body["front"] = True
    data = json.dumps(body).encode()
    req = urllib.request.Request((url or server_url()) + "/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode()).get("prompt_id", "")


def _node_available(name: str) -> bool:
    try:
        import nodes as _comfy_nodes  # ComfyUI's registry (absent outside ComfyUI)
        return name in _comfy_nodes.NODE_CLASS_MAPPINGS
    except Exception:
        return False


def queue_chain_front(graphs: list[dict], url: str | None = None) -> list[str]:
    """Queue graphs at the front of the queue so they run next, in the given order (follow-ups
    of a run finish before the next run's renders start)."""
    ids = [queue_prompt(g, url, front=True) for g in reversed(graphs)]
    return ids[::-1]


def clip_info(path: str) -> tuple[int, float]:
    import cv2
    cap = cv2.VideoCapture(path)
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    return n, fps


def direct(*, audio: np.ndarray, sr: int, out_root: str, mode: str = "voice", pack: str = "crowd",
           fmt: str = "16:9", preset: str = "quality", seconds: float = 30.0, start: float = -1.0,
           seed: int = 1, twins: str = "auto", dance_path: str = "", run_name: str = "",
           queue: bool = True, editor: str = "klein", guide_strength: float = 1.0,
           do_stems: bool = True, queue_assemble: bool = True, framing: str = "mixed",
           artist_path: str = "", qc_retries: int = 2, log=print) -> dict:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = _slug(run_name) if run_name else f"{mode}-{pack}-{fmt.replace(':', 'x')}-{preset}-{stamp}"
    base_id, k = run_id, 1
    while os.path.isdir(os.path.join(out_root, "frissonvision", "runs", run_id)) and \
            os.listdir(os.path.join(out_root, "frissonvision", "runs", run_id)):
        k += 1  # never reuse a run folder: ComfyUI's save counters would shift every file name
        run_id = f"{base_id}-{k}"
    rel = f"frissonvision/runs/{run_id}"
    run_dir = os.path.join(out_root, *rel.split("/"))
    os.makedirs(os.path.join(run_dir, "jobs"), exist_ok=True)
    stems.write_wav(os.path.join(run_dir, "song.wav"), audio, sr)

    tl = beat.analyze(audio, sr)
    pk = casting.load_pack(pack)
    label = preset
    preset = formats.effective_preset(mode, preset)  # singing always renders on the quality model
    if mode == "voice":
        if artist_path and pack != "artist":
            pk = casting.load_pack("artist")
            pack = "artist"
        if pk["casting"].get("artist") and not (artist_path and os.path.exists(artist_path)):
            raise ValueError("the artist pack needs artist_image_path (one photo of the artist)")
        p = planner.plan_voice(tl, pk, fmt=fmt, preset=preset, seconds=seconds, start=start, seed=seed,
                               twins=None if twins == "auto" else twins, framing=framing)
        p["artist_path"] = artist_path
    else:
        if not dance_path or not os.path.exists(dance_path):
            raise ValueError("motion mode needs a dance clip (dance_video_path)")
        n, cfps = clip_info(dance_path)
        p = planner.plan_motion(tl, pk, clip_frames=n, clip_fps=cfps, fmt=fmt, preset=preset,
                                seconds=seconds, start=start, seed=seed)
        p["dance_path"] = dance_path
    log(f"[FrissonVision] {mode} plan: {len(p['renders'])} renders, window {p['song_t0']:.2f}-{p['song_t1']:.2f}s, "
        f"tempo {p['tempo']}")

    # stems for the window (+1 s margins): vocals drive lip sync, drums drive the FX
    p["stems_t0"] = max(0.0, p["song_t0"] - 1.0)
    if do_stems:
        a0, a1 = int(p["stems_t0"] * sr), int(min(len(audio[0]), (p["song_t1"] + 1.0) * sr))
        st = stems.separate(audio[:, a0:a1], sr)
        stems.write_wav(os.path.join(run_dir, "vocals.wav"), st["vocals"], st["sr"])
        stems.write_wav(os.path.join(run_dir, "drums.wav"), st["drums"], st["sr"])
        vocal_path = os.path.join(run_dir, "vocals.wav")
    else:
        vocal_path = os.path.join(run_dir, "song.wav")
        p["stems_t0"] = 0.0

    rsize = formats.render_size(fmt, preset)
    jobs: list[tuple[str, dict]] = []
    if mode == "voice":
        sizes = ["close"] * p["performers"]
        for r in p["renders"]:
            sizes[r["performer"]] = r.get("size", "close")
        cast = casting.cast_voice(pk, p["performers"], seed, sizes, fmt=fmt)
        s1, s2 = formats.portrait_size(fmt)
        port = {}
        need_portrait = {r["performer"] for r in p["renders"] if r["role"] == "A" or editor == "qwen"}
        for c in cast:
            if c["id"] not in need_portrait:
                continue
            name = f"p{c['id']:03d}"
            if c.get("artist_edit"):
                jobs.append((name, graphs.artist_portrait(artist_path=artist_path, prompt=c["portrait"],
                                                          seed=c["seed"], prefix=f"{rel}/{name}", size2=s2).nodes))
            else:
                jobs.append((name, graphs.krea2_portrait(prompt=c["portrait"], seed=c["seed"],
                                                           prefix=f"{rel}/{name}", size1=s1, size2=s2,
                                                           turbo_only=(label == "fast")).nodes))
            port[c["id"]] = os.path.join(run_dir, f"{name}_00001_.png")
        jobs.append(("free_after_portraits", graphs.free_memory("after Krea 2 portraits").nodes))
        neg = pk["casting"]["negative"]
        by_id = {r["id"]: r for r in p["renders"]}
        for r in p["renders"]:
            r["path"] = os.path.join(run_dir, f"r{r['id']:03d}{r['role']}_00001_.mp4")
        seek = lambda r: round(r["t_start"] + p["song_t0"] - p["stems_t0"], 4)
        # dense vocals (rap) get the articulation prompt: it lifted lip sync on dense shots
        from . import lipsync
        vx, vsr = stems.read_wav(vocal_path) if os.path.exists(vocal_path) else (None, None)
        if vx is not None:
            vx = vx.mean(0) if vx.ndim == 2 else vx
            vref = float(np.sqrt(np.mean(vx ** 2) + 1e-12))
        dense_from = float(pk["casting"].get("dense_from", 0.6))
        sing_for = {}
        for r in [r for r in p["renders"] if r["role"] == "A"]:
            text = cast[r["performer"]]["sing"]
            if vx is not None:
                r["vocal_density"] = round(lipsync.vocal_density(vx, vsr, seek(r), r["frames"] / r["fps"], vref), 3)
                if r["vocal_density"] >= dense_from:
                    text = casting.articulate(text)
            sing_for[r["id"]] = text
        for r in [r for r in p["renders"] if r["role"] == "A"]:
            c = cast[r["performer"]]
            jobs.append((f"r{r['id']:03d}A", graphs.ltx_render(
                preset=preset, fmt=fmt, key_path=port[r["performer"]], prompt=sing_for[r["id"]], negative=neg,
                frames=r["frames"], seed=c["seed"], prefix=f"{rel}/r{r['id']:03d}A",
                audio_path=vocal_path, audio_seek=seek(r)).nodes))
        twins_ = [r for r in p["renders"] if r["role"] == "B"]
        for r in twins_:
            a = by_id[r["twin_of"]]
            r["key"] = os.path.join(run_dir, f"k{r['id']:03d}_00001_.png")
            c = cast[r["performer"]]
            # a chain of looks (hold grammar) needs a different look per twin: each twin
            # performer's own; one twin per lead (flash grammar) keeps the lead's twin look
            look_of = r["performer"] if "guide_start" in r else a["performer"]
            jobs.append((f"k{r['id']:03d}", graphs.twin_key(
                video_path=a["path"], portrait_path=port.get(r["performer"]), prefix=f"{rel}/k{r['id']:03d}",
                size=rsize, seed=seed + r["id"], editor=editor, start_s=r.get("guide_start", 0.0),
                prompt=(cast[look_of].get("twin_prompt") if c.get("artist_edit") else
                        " ".join(graphs.TWIN_TEXT.format(**c["twin"]).split())) if editor != "qwen" else None).nodes))
        for r in twins_:
            a = by_id[r["twin_of"]]
            c = cast[r["performer"]]
            text = casting.articulate(c["sing"]) if a.get("vocal_density", 0.0) >= dense_from else c["sing"]
            jobs.append((f"r{r['id']:03d}B", graphs.ltx_render(
                preset=preset, fmt=fmt, key_path=r["key"], prompt=text, negative=neg,
                frames=r["frames"], seed=c["seed"], prefix=f"{rel}/r{r['id']:03d}B",
                audio_path=vocal_path, audio_seek=seek(r), guide_video=a["path"],
                guide_start=r.get("guide_start", 0.0), guide_strength=guide_strength, guide_face=True).nodes))
    else:
        cast = casting.cast_motion(pk, len(pk["casting"]["cast"]), seed)
        neg = pk["casting"]["negative"]
        for r in p["renders"]:
            c = cast[r["performer"] % len(cast)]
            r["key"] = os.path.join(run_dir, f"k{r['id']:03d}_00001_.png")
            r["path"] = os.path.join(run_dir, f"r{r['id']:03d}{r['role']}_00001_.mp4")
            jobs.append((f"k{r['id']:03d}", graphs.motion_key(
                video_path=dance_path, start_s=r["src_start"], prompt=c["edit"],
                prefix=f"{rel}/k{r['id']:03d}", size=rsize, seed=c["seed"]).nodes))
        for r in p["renders"]:
            c = cast[r["performer"] % len(cast)]
            jobs.append((f"r{r['id']:03d}{r['role']}", graphs.ltx_render(
                preset=preset, fmt=fmt, key_path=r["key"], prompt=c["video"], negative=neg,
                frames=r["frames"], seed=c["seed"] + 17 * (r["role"] == "MB"),
                prefix=f"{rel}/r{r['id']:03d}{r['role']}",
                guide_video=dance_path, guide_start=r["src_start"], guide_strength=guide_strength,
                guide_face=False, key_strengths=(1.0, 1.0, 1.0) if preset == "quality" else (1.0, 1.0)).nodes))
    use_qc = qc_retries > 0 and queue_assemble and _node_available("FVQC")
    if use_qc:  # QC re-rolls bad takes, then queues free + assemble itself
        jobs.append(("zz_qc", graphs.qc(run_dir=run_dir, pass_index=0, max_passes=qc_retries).nodes))
    else:
        jobs.append(("free_before_assemble", graphs.free_memory("before assembly").nodes))
    jobs.append(("zz_assemble", graphs.assemble(run_dir=run_dir, prefix=rel).nodes))

    p["cast"] = cast
    p["run_dir"], p["run_rel"] = run_dir, rel
    json.dump(p, open(os.path.join(run_dir, "plan.json"), "w", encoding="utf-8"), indent=1)
    for i, (name, g) in enumerate(jobs):
        json.dump(g, open(os.path.join(run_dir, "jobs", f"{i:03d}_{name}.json"), "w", encoding="utf-8"), indent=1)
    # the whole run as one file: an SDK client can submit these itself (renders only; the
    # client then runs FVQC / FVAssemble), see sdk_run.py
    json.dump({"run_id": run_id, "run_dir": run_dir, "run_rel": rel,
               "jobs": [{"name": n, "graph": g} for n, g in jobs if not n.startswith("zz_")]},
              open(os.path.join(run_dir, "jobs.json"), "w", encoding="utf-8"))
    ids = []
    if queue:
        url = server_url()
        for name, g in jobs:
            if name == "zz_assemble" and (not queue_assemble or use_qc):
                continue
            ids.append(queue_prompt(g, url))
    summary = (f"Frisson Vision run '{run_id}': {mode} / {pack} / {fmt} / {label}\n"
               f"window {p['song_t0']:.2f}-{p['song_t1']:.2f}s at {p['tempo']} BPM, "
               f"{len(p.get('shots', []))} shots, {len(p['renders'])} renders, {len(jobs)} jobs "
               f"{'queued' if queue else 'written (not queued)'}.\nOutput folder: {run_dir}")
    return {"run_dir": run_dir, "run_id": run_id, "run_rel": rel, "jobs": len(jobs), "prompt_ids": ids,
            "summary": summary, "bundle": os.path.join(run_dir, "jobs.json")}


# ------------------------------------------------------------------ take QC + re-rolls
def _bump(graph: dict, k: int) -> dict:
    """New seeds for every sampler in a job graph."""
    for node in graph.values():
        ins = node.get("inputs", {})
        for key in ("noise_seed", "seed"):
            if isinstance(ins.get(key), int):
                ins[key] = (ins[key] + 7919 * k) % (2 ** 31 - 1)
    return graph


def _job_file(run_dir: str, name: str) -> str | None:
    hits = sorted(f for f in os.listdir(os.path.join(run_dir, "jobs")) if f.endswith(f"_{name}.json"))
    return os.path.join(run_dir, "jobs", hits[0]) if hits else None


def _articulate_prompt(g: dict) -> None:
    """A lead re-rolled for lip sync switches to the articulation prompt (casting.articulate)."""
    from . import casting
    for node in g.values():
        if node.get("class_type") != "LTXVConditioning":
            continue
        ref = node.get("inputs", {}).get("positive")
        enc = g.get(str(ref[0])) if isinstance(ref, list) and ref else None
        if enc and enc.get("class_type") == "CLIPTextEncode" and isinstance(enc["inputs"].get("text"), str):
            enc["inputs"]["text"] = casting.articulate(enc["inputs"]["text"])


def _refresh_prompt(g: dict) -> None:
    """Bring a saved LTX job's prompt up to date before re-rolling it (jobs queued before the
    continuity sentence existed would otherwise re-roll with the prompt that cut away)."""
    from . import graphs
    for node in g.values():
        if node.get("class_type") != "LTXVConditioning":
            continue
        ref = node.get("inputs", {}).get("positive")
        enc = g.get(str(ref[0])) if isinstance(ref, list) and ref else None
        if enc and enc.get("class_type") == "CLIPTextEncode":
            t = enc["inputs"].get("text", "")
            if isinstance(t, str) and graphs.CONTINUITY not in t:
                enc["inputs"]["text"] = f"{t.rstrip()} {graphs.CONTINUITY}"


def qc_pass(run_dir: str, *, pass_idx: int = 0, max_passes: int = 1, queue: bool = True,
            then_assemble: bool = True, queue_rerolls: bool = True, log=print) -> dict:
    """Check every render of a run; re-roll the bad ones (new seeds) and queue the next QC pass,
    or, when all pass / retries are spent, keep the best take of each shot and queue assembly."""
    from . import qc
    pj = os.path.join(run_dir, "plan.json")
    p = json.load(open(pj, encoding="utf-8"))
    voice = p.get("mode", "voice") == "voice"
    vocal = sr = None
    vp = os.path.join(run_dir, "vocals.wav")
    if voice and os.path.exists(vp):
        vocal, sr = stems.read_wav(vp)
    report, redo = [], []
    for r in p["renders"]:
        seek = r["t_start"] + p["song_t0"] - p.get("stems_t0", 0.0)
        takes = r.setdefault("takes", [])
        done = next((t for t in takes if t.get("path") == r.get("path") and t.get("v", 0) >= qc.VERSION
                     and os.path.exists(t["path"])), None)
        if done is not None:  # checked in an earlier pass: keep its verdict
            res = dict(done, ok=not done.get("fail"))
            report.append({"id": r["id"], "role": r["role"], **res})
            if not res["ok"]:
                redo.append(r)
            continue
        try:
            res = qc.check_clip(r.get("path", ""), expected_frames=r.get("frames", 0), voice=voice,
                                vocal=vocal, sr=sr or 44100, seek=seek, sync=voice and r["role"] == "A",
                                people=not voice)
        except Exception as e:  # never let QC block a cut
            res = {"path": r.get("path", ""), "ok": True, "fail": [], "score": 0.5, "error": str(e)}
        entry = {k: v for k, v in res.items() if k != "ok"}
        for t in takes:  # a re-check (new checks, e.g. `cut`) replaces the old verdict on a take
            if t["path"] == res["path"]:
                t.clear()
                t.update(entry)
                break
        else:
            takes.append(entry)
        report.append({"id": r["id"], "role": r["role"], **res})
        if not res["ok"]:
            redo.append(r)
    # re-roll budget: re-renders may add at most `qc_budget` x the run's own render frames
    # (the user's call: a run may take up to twice as long to get lip sync right). The worst
    # takes go first: broken ones (missing, frozen, cut away...), then the weakest lip sync.
    by_id = {r["id"]: r for r in p["renders"]}
    total = sum(r.get("frames", 0) for r in p["renders"])
    budget = p.setdefault("qc_budget_frames", int(total * float(p.get("qc_budget", 1.0))))
    spent = p.get("qc_spent_frames", 0)

    def severity(r):
        t = next((x for x in report if x["id"] == r["id"]), {})
        broken = bool(set(t.get("fail", [])) - {"sync"})
        return (0 if broken else 1, t.get("sync_conf", 0.0), r["role"] == "B", r["id"])

    chosen, redo_ids = [], set()
    for r in sorted(redo, key=severity):
        if r["id"] in redo_ids:
            continue
        # a re-rolled lead clip invalidates its pose-matched twins: re-render them on the new lead
        group = [r] + [t for t in p["renders"] if t.get("twin_of") == r["id"] and t["role"] == "B"
                       and t["id"] not in redo_ids]
        cost = sum(x.get("frames", 0) for x in group)
        if spent + cost > budget:
            continue
        spent += cost
        for x in group:
            chosen.append(x)
            redo_ids.add(x["id"])
    skipped = len({r["id"] for r in redo} - redo_ids)
    redo = chosen
    p["qc_spent_frames"] = spent
    last = pass_idx >= max_passes or not redo
    queued, rerolls = [], []
    if not last:
        redo.sort(key=lambda r: (r["role"] == "B", r["id"]))
        new_paths = {}
        rel = p.get("run_rel", "")
        lead_jobs, key_jobs, twin_jobs = [], [], []

        def swap(g, old, new):
            return json.loads(json.dumps(g).replace(json.dumps(old)[1:-1], json.dumps(new)[1:-1]))

        for r in redo:
            name = f"r{r['id']:03d}{r['role']}"
            jf = _job_file(run_dir, name)
            if not jf:
                continue
            g = _bump(json.load(open(jf, encoding="utf-8")), pass_idx + 1)
            _refresh_prompt(g)
            if r["role"] == "A" and "sync" in next((x.get("fail", []) for x in report if x["id"] == r["id"]), []):
                _articulate_prompt(g)
            tag = f"{name}_t{pass_idx + 1}"
            for node in g.values():
                ins = node.get("inputs", {})
                if node.get("class_type") == "SaveVideo" and str(ins.get("filename_prefix", "")).endswith(name):
                    ins["filename_prefix"] = f"{rel}/{tag}"
            fails = next((x.get("fail", []) for x in report if x["id"] == r["id"]), [])
            needs_key = False
            if r["role"] in ("M", "MB") and "people" in fails:
                # a second person came from the character keyframe: re-make it with a new seed
                kname = f"k{r['id']:03d}"
                kf = _job_file(run_dir, kname)
                if kf:
                    kg = _bump(json.load(open(kf, encoding="utf-8")), pass_idx + 1)
                    ktag = f"{kname}_t{pass_idx + 1}"
                    for node in kg.values():
                        ins = node.get("inputs", {})
                        if node.get("class_type") == "SaveImage" and str(ins.get("filename_prefix", "")).endswith(kname):
                            ins["filename_prefix"] = f"{rel}/{ktag}"
                    old_key = r.get("key") or os.path.join(run_dir, f"{kname}_00001_.png")
                    g = swap(g, old_key, os.path.join(run_dir, f"{ktag}_00001_.png"))
                    json.dump(kg, open(os.path.join(run_dir, "jobs", f"qc{pass_idx + 1}_{ktag}.json"), "w",
                                       encoding="utf-8"), indent=1)
                    key_jobs.append(kg)
                    needs_key = True
            if r["role"] == "B" and r.get("twin_of") in new_paths:  # follow the re-rolled lead clip
                old = by_id[r["twin_of"]]["takes"][0]["path"]
                g = swap(g, old, new_paths[r["twin_of"]])
                # and re-make the twin's keyframe from the new lead, so its first frame has the
                # new lead's pose (the old keyframe would snap from the old pose to the new)
                kname = f"k{r['id']:03d}"
                kf = _job_file(run_dir, kname)
                if kf:
                    kg = swap(json.load(open(kf, encoding="utf-8")), old, new_paths[r["twin_of"]])
                    ktag = f"{kname}_t{pass_idx + 1}"
                    for node in kg.values():
                        ins = node.get("inputs", {})
                        if node.get("class_type") == "SaveImage" and str(ins.get("filename_prefix", "")).endswith(kname):
                            ins["filename_prefix"] = f"{rel}/{ktag}"
                    old_key = r.get("key") or os.path.join(run_dir, f"{kname}_00001_.png")
                    g = swap(g, old_key, os.path.join(run_dir, f"{ktag}_00001_.png"))
                    json.dump(kg, open(os.path.join(run_dir, "jobs", f"qc{pass_idx + 1}_{ktag}.json"), "w",
                                       encoding="utf-8"), indent=1)
                    key_jobs.append(kg)
            new_paths[r["id"]] = os.path.join(run_dir, f"{tag}_00001_.mp4")
            r["path"] = new_paths[r["id"]]
            json.dump(g, open(os.path.join(run_dir, "jobs", f"qc{pass_idx + 1}_{tag}.json"), "w", encoding="utf-8"),
                      indent=1)
            # renders that wait on a new keyframe run after the keyframe jobs
            (twin_jobs if (r["role"] == "B" or needs_key) else lead_jobs).append(g)
        jobs = lead_jobs + key_jobs + twin_jobs  # grouped by model: LTX, Klein, LTX (renders after their keys)
        from . import graphs
        rerolls = list(jobs)
        if queue_rerolls:  # chain the next pass on the server
            jobs.append(graphs.qc(run_dir=run_dir, pass_index=pass_idx + 1, max_passes=max_passes,
                                  then_assemble=then_assemble).nodes)
        if queue and queue_rerolls:
            queued = queue_chain_front(jobs, server_url())
        msg = f"QC pass {pass_idx}: re-rolling {len(redo)} take(s): " + ", ".join(
            f"r{r['id']:03d}{r['role']}" for r in redo) + \
            (f" ({skipped} more over the re-render budget)" if skipped else "")
    else:
        for r in p["renders"]:  # best take of each shot
            takes = [t for t in r.get("takes", []) if t.get("path") and os.path.exists(t["path"])]
            for t in takes:  # takes judged by an older QC (before the cut / sync checks): judge again
                if t.get("v", 0) < qc.VERSION and t.get("frames"):
                    try:
                        seek = r.get("t_start", 0.0) + p.get("song_t0", 0.0) - p.get("stems_t0", 0.0)
                        res = qc.check_clip(t["path"], expected_frames=r.get("frames", 0), voice=voice,
                                            vocal=vocal, sr=sr or 44100, seek=seek,
                                            sync=voice and r["role"] == "A", people=not voice)
                        t.clear()
                        t.update({k: v for k, v in res.items() if k != "ok"})
                    except Exception:
                        pass
            if takes:
                # newest take wins ties (re-rolled twins follow their re-rolled lead clip)
                best = max(enumerate(takes), key=lambda it: (round(it[1].get("score", 0.0), 2), it[0]))[1]
                r["path"] = best["path"]
                # a twin only ever appears as a flash, so one that still fails is left out of the
                # cut rather than flashing a wrong person in (a lead clip has no stand-in and stays)
                if r["role"] in ("B", "MB") and best.get("fail"):
                    r["unusable"] = sorted(best["fail"])
                else:
                    r.pop("unusable", None)
                # the take's own lip-sync offset (SyncNet, 25 fps frames; negative = the mouth
                # leads the voice): show it that much later
                if "sync_offset" in best:
                    r["sync_delay"] = round(min(0.4, max(-0.1, -best["sync_offset"] / 25.0)), 3)
                    r["sync_conf"] = best.get("sync_conf")
        shots = {s["index"]: s for s in p.get("shots", []) if "index" in s}
        for r in p["renders"]:  # twins follow their lead's mouth, so they share its offset
            lead = by_id.get(r.get("twin_of"))
            if r["role"] == "B" and lead is not None and "sync_delay" in lead:
                r["sync_delay"] = lead["sync_delay"]
            # a delay can't reach back past the clip's own lead-in (it would hold the first
            # frame at the cut)
            if "sync_delay" in r:
                on = r.get("cut_at", shots.get(r.get("shot"), {}).get("t0", r["t_start"]))
                r["sync_delay"] = round(min(r["sync_delay"], max(0.0, on - r["t_start"] - 0.02)), 3)
        if queue and then_assemble and queue_rerolls:
            from . import graphs
            queued = queue_chain_front([graphs.free_memory("before assembly").nodes,
                                        graphs.assemble(run_dir=run_dir, prefix=p.get("run_rel", "")).nodes],
                                       server_url())
        bad = [f"r{x['id']:03d}{x['role']}:{'+'.join(x['fail'])}" for x in report if not x["ok"]]
        msg = f"QC pass {pass_idx}: {len(report) - len(bad)}/{len(report)} takes pass" + \
              (f"; kept best take for {', '.join(bad)}" if bad else "") + ("; assembling" if then_assemble else "")
    json.dump(p, open(pj, "w", encoding="utf-8"), indent=1)
    hist = os.path.join(run_dir, "qc.json")
    allq = json.load(open(hist, encoding="utf-8")) if os.path.exists(hist) else []
    allq.append({"pass": pass_idx, "report": report, "message": msg})
    json.dump(allq, open(hist, "w", encoding="utf-8"), indent=1)
    bundle = os.path.join(run_dir, f"qc{pass_idx}.json")
    json.dump({"pass": pass_idx, "final": last, "message": msg, "jobs": [{"graph": g} for g in rerolls]},
              open(bundle, "w", encoding="utf-8"))
    log(f"[FrissonVision] {msg}")
    return {"message": msg, "report": report, "queued": queued, "bundle": bundle, "final": last,
            "rerolls": len(rerolls)}
