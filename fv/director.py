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


def queue_prompt(graph: dict, url: str | None = None, client_id: str = "oneshot-mv", front: bool = False) -> str:
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
           artist_path: str = "", qc_retries: int = 2, theme: str = "", broll_share: float = 0.125,
           broll_style: str = "mix", staged: bool = True,
           look_groups: bool = False, cuts: dict | None = None, audio_fx: str = "",
           cast_order: str = "as written", crowd: str = "auto", log=print,
           lip_repair: bool = False) -> dict:  # lip_repair: retired Oct 28 (LatentSync removed), ignored;
    # still accepted so a ComfyUI that has not restarted (old nodes.py in memory) can call in
    """staged (voice runs queued here, with QC): the 4-load order. Krea makes every portrait and
    the b-roll candidate stills; LTX makes every singer clip and the b-roll clips, and QC
    re-rolls the singer shots while LTX stays loaded; then Klein makes every look keyframe from
    the final singer takes and LTX the look clips (QC passes and Klein edits keep LTX loaded,
    FVKeep). n46 spent 65 of 155 minutes on 12 model loads; this order loads 4 times and never
    re-makes a look because its singer was re-rolled.
    look_groups (staged, experimental, off by default): each drop group's looks are one LTX clip
    that switches person on the beat (the lead's pose guide, each look's keyframe pinned at its
    cut) instead of one clip per look; the edit reads each look from it. The n46 test cut
    cleanly (65 s instead of 156 s), but in n47 only 10 of 36 looks did (the clip left the
    first look within its first frames, morphed instead of cutting, or showed three people), so
    looks a group doesn't deliver get their own clip after all (_fallback_looks)."""
    # the language model (world bible, cast, b-roll lines) loads on top of whatever the last job
    # left in memory: at 01:41 UTC Oct 27 the Rome Director started right after an LTX clip and
    # ComfyUI went down without a word in its log. Start from empty memory.
    if theme and theme.strip():
        try:
            import gc
            import comfy.model_management as mm  # type: ignore
            mm.unload_all_models()
            mm.soft_empty_cache()
            gc.collect()
        except Exception:
            pass
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = _slug(run_name) if run_name else f"{mode}-{pack}-{fmt.replace(':', 'x')}-{preset}-{stamp}"
    base_id, k = run_id, 1
    while os.path.isdir(os.path.join(out_root, "oneshot-mv", "runs", run_id)) and \
            os.listdir(os.path.join(out_root, "oneshot-mv", "runs", run_id)):
        k += 1  # never reuse a run folder: ComfyUI's save counters would shift every file name
        run_id = f"{base_id}-{k}"
    rel = f"oneshot-mv/runs/{run_id}"
    run_dir = os.path.join(out_root, *rel.split("/"))
    os.makedirs(os.path.join(run_dir, "jobs"), exist_ok=True)
    stems.write_wav(os.path.join(run_dir, "song.wav"), audio, sr)

    tl = beat.analyze(audio, sr)
    pk = casting.load_pack(pack)
    label = preset
    preset = formats.effective_preset(mode, preset)  # singing always renders on the quality model
    if mode == "voice" and artist_path and pack != "artist":
        pk = casting.load_pack("artist")
        pack = "artist"
    bible = {}
    if theme and theme.strip():  # the theme sets the world: who, wear, places, light, actions, grade
        from . import theme as themes
        bible = themes.expand(theme, seed=seed, out_root=out_root, log=log)
        pk = casting.apply_theme(pk, bible)
        pk["finish"] = themes.finish_for(bible, pk.get("finish", {}))
        log("[OneShotMV] " + themes.summary(bible).replace("\n", "\n[OneShotMV] "))
    if cuts:  # the user's cut per energy level (low / medium / high), over the pack's defaults
        pk["edit"] = dict(pk["edit"], **{f"cut_{lv}": m for lv, m in cuts.items() if m})
    if mode == "voice":
        if pk["casting"].get("artist") and not (artist_path and os.path.exists(artist_path)):
            raise ValueError("the artist pack needs artist_image_path (one photo of the artist)")
        p = planner.plan_voice(tl, pk, fmt=fmt, preset=preset, seconds=seconds, start=start, seed=seed,
                               twins=None if twins == "auto" else twins, framing=framing)
        p["artist_path"] = artist_path
        if cuts:  # a re-applied theme starts from the pack again: keep the user's cuts
            p["cuts"] = planner.cut_modes(pk["edit"])
        if bible and (pk.get("casting") or {}).get("unique"):
            # every shot its own character (the user, Oct 26: a bigger cast, no repeats): enough
            # characters for every singer and every b-roll candidate
            from . import broll as _broll
            need = p["performers"] + 2 * _broll.slots(p, broll_share) + 2
            if len(bible.get("characters") or []) < need:
                bible = themes.extend(bible, need, seed=seed, out_root=out_root, log=log)
                ed_ = pk["edit"]
                pk = casting.apply_theme(casting.load_pack(pack), bible)
                pk["edit"] = ed_
                pk["finish"] = themes.finish_for(bible, pk.get("finish", {}))
            p["unique_cast"] = True
    else:
        if not dance_path or not os.path.exists(dance_path):
            raise ValueError("motion mode needs a dance clip (dance_video_path)")
        n, cfps = clip_info(dance_path)
        p = planner.plan_motion(tl, pk, clip_frames=n, clip_fps=cfps, fmt=fmt, preset=preset,
                                seconds=seconds, start=start, seed=seed)
        p["dance_path"] = dance_path
    log(f"[OneShotMV] {mode} plan: {len(p['renders'])} renders, window {p['song_t0']:.2f}-{p['song_t1']:.2f}s, "
        f"tempo {p['tempo']}")

    # stems for the window (+1 s margins): vocals drive lip sync, drums drive the FX
    p["stems_t0"] = max(0.0, p["song_t0"] - 1.0)
    if do_stems:
        a0, a1 = int(p["stems_t0"] * sr), int(min(len(audio[0]), (p["song_t1"] + 1.0) * sr))
        st = stems.separate(audio[:, a0:a1], sr)
        stems.write_wav(os.path.join(run_dir, "vocals.wav"), st["vocals"], st["sr"])
        stems.write_wav(os.path.join(run_dir, "drums.wav"), st["drums"], st["sr"])
        vocal_path = os.path.join(run_dir, "vocals.wav")
        log("[OneShotMV] stems: vocals and drums")
    else:
        vocal_path = os.path.join(run_dir, "song.wav")
        p["stems_t0"] = 0.0

    # b-roll cutaways (fv/broll.py): World characters in a candid moment, dancers in instrumental gaps
    if mode == "voice" and broll_share > 0:
        p["broll_share"] = float(broll_share)
        p["broll_style"] = broll_style if broll_style in ("mix", "world", "dance") else "mix"

    rsize = formats.render_size(fmt, preset)
    jobs: list[tuple[str, dict]] = []
    staged = bool(staged) and mode == "voice" and queue and queue_assemble and qc_retries > 0 \
        and _node_available("FVQC") and _node_available("FVKeep")
    groups = bool(look_groups) and staged
    keep = preset if staged else None
    free_idx = 0
    if mode == "voice":
        sizes = ["close"] * p["performers"]
        for r in p["renders"]:
            sizes[r["performer"]] = r.get("size", "close")
        leads = sorted({r["performer"] for r in p["renders"] if r["role"] == "A"})
        shot_of = {s_["index"]: s_ for s_ in p.get("shots", []) if "index" in s_}
        phases = ["quiet"] * p["performers"]
        for r in p["renders"]:
            phases[r["performer"]] = shot_of.get(r["shot"], {}).get("phase", "quiet")
        # the Director's cast order and crowd choices (casting._cast_theme)
        pk.setdefault("casting", {})["story"] = (cast_order or "as written") != "mixed"
        pk["casting"]["crowd_mode"] = crowd if crowd in ("on", "off") else "auto"
        if bible and pk["casting"]["crowd_mode"] != "off" and (pk["casting"]["crowd_mode"] == "on" or
                                                               themes.crowd_world(bible.get("theme", ""))):
            try:
                dress = themes.crowd_dress(bible, seed=seed, out_root=out_root, log=log)
                if dress and isinstance(pk["casting"].get("theme"), dict):
                    pk["casting"]["theme"]["crowd_dress"] = dress
            except Exception as e:  # never let it stop a run
                log(f"[OneShotMV] crowd dress skipped ({e})")
        p["cast_order"], p["crowd"] = (cast_order or "as written"), pk["casting"]["crowd_mode"]
        cast = casting.cast_voice(pk, p["performers"], seed, sizes, fmt=fmt, leads=leads, phases=phases)
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
        free_idx = len(jobs)
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
                identity_path=artist_path if (c.get("artist_edit") and artist_path) else None,
                prompt=(cast[look_of].get("twin_prompt") if c.get("artist_edit") else
                        " ".join(graphs.TWIN_TEXT.format(**c["twin"]).split()) + " " +
                        casting.edit_realism(cast[a["performer"]].get("size", "close") == "close"))
                if editor != "qwen" else None,
                keep=keep).nodes))
        if groups:  # each drop group's chain of looks: one clip, made after singer QC (_looks_jobs)
            p["look_groups"] = []
            for s in p.get("shots", []):
                ch = [by_id[i] for i in s.get("chain", []) if by_id[i]["role"] == "B" and "cut_at" in by_id[i]]
                if not ch:
                    continue
                a = by_id[s["A"]]
                first = cast[ch[0]["performer"]]
                sing = casting.articulate(first["sing"]) if a.get("vocal_density", 0.0) >= dense_from else first["sing"]
                p["look_groups"].append({"shot": s["index"], "lead": a["id"], "looks": [b["id"] for b in ch],
                                         "who": [_who_short(cast[b["performer"]]) for b in ch], "sing": sing,
                                         "seed": first["seed"], "neg": neg, "guide_strength": guide_strength,
                                         "shot_t1": s["t1"]})
                for b in ch:
                    b["group"] = s["index"]
        for r in twins_:
            if r.get("group") is not None:
                continue
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
    p["cast"] = cast
    from . import fx as _fx
    p["fx_level"] = audio_fx if audio_fx in _fx.LEVELS else (pk.get("fx") or {}).get("level", "subtle")
    if bible:
        p["theme"] = {k: v for k, v in bible.items() if k != "raw"}
        p["finish"] = pk["finish"]
    p["run_dir"], p["run_rel"] = run_dir, rel
    deferred = set()
    if staged:
        # the looks wait for the final singer takes (queued by the last singer QC pass)
        deferred = {n for n, _ in jobs if re.match(r"^(k\d{3}|r\d{3}B)$", n)}
        if broll_share > 0:  # b-roll candidates ride along with the portraits and the singer clips
            from . import broll
            try:
                stills, clips = broll.plan(p, run_dir, pk, log=log, candidates=True)
            except Exception as e:  # never let b-roll block a run
                log(f"[OneShotMV] b-roll skipped ({e})")
                stills, clips = [], []
            jobs[free_idx:free_idx] = [(f"brollk{i:02d}", g) for i, g in enumerate(stills)]
            last_a = max([i for i, (n, _) in enumerate(jobs) if re.match(r"^r\d{3}A$", n)] + [len(jobs) - 1])
            jobs[last_a + 1:last_a + 1] = [(f"brollr{i:02d}", g) for i, g in enumerate(clips)]
            p["broll_planned"] = True  # broll.finalize() picks the slots after singer QC
        p["staged"] = True
        if p.get("unique_cast") and _node_available("FVCritic"):
            # the vision check (fv/critic.py) between the stills and the videos: every portrait and
            # b-roll still is checked, the bad ones re-made, and only then are the clips queued
            items = []
            for c in cast:
                if c["id"] in port and c.get("check"):
                    items.append({"name": f"p{c['id']:03d}", "job": f"p{c['id']:03d}", "kind": "portrait",
                                  "performer": c["id"], "orig": port[c["id"]], "image": port[c["id"]],
                                  "check": c["check"], "drop": bool(c.get("drop")), "takes": []})
            for r in p["renders"]:
                b = r.get("broll") or {}
                if r.get("broll_candidate") and b.get("check") and b.get("still_job") is not None:
                    items.append({"name": f"k{r['id']:03d}R", "job": f"brollk{b['still_job']:02d}", "kind": "broll",
                                  "render": r["id"], "orig": r["key"], "image": r["key"], "check": b["check"],
                                  "takes": []})
            later = {n for n, _ in jobs if re.match(r"^(r\d{3}A|brollr\d{2})$", n)} | {"zz_qc"}  # QC is added below
            fi = next(i for i, (n, _) in enumerate(jobs) if n == "free_after_portraits")
            jobs.insert(fi + 1, ("zz_critic", graphs.critic(run_dir=run_dir, pass_index=0, max_passes=2).nodes))
            p["critic"] = {"items": items, "later": sorted(later), "max_passes": 2}
            deferred |= later
    use_qc = qc_retries > 0 and queue_assemble and _node_available("FVQC")
    if use_qc:  # QC re-rolls bad takes, then queues free + assemble itself
        jobs.append(("zz_qc", graphs.qc(run_dir=run_dir, pass_index=0, max_passes=qc_retries,
                                        stage="leads" if staged else "all", keep=keep).nodes))
    else:
        jobs.append(("free_before_assemble", graphs.free_memory("before assembly").nodes))
    jobs.append(("zz_assemble", graphs.assemble(run_dir=run_dir, prefix=rel).nodes))
    if staged:
        p["deferred_jobs"] = [f"{i:03d}_{n}.json" for i, (n, _) in enumerate(jobs) if n in deferred
                              and n not in (p.get("critic") or {}).get("later", [])]
        if p.get("critic"):
            p["critic"]["deferred"] = [f"{i:03d}_{n}.json" for i, (n, _) in enumerate(jobs)
                                       if n in p["critic"]["later"]]
        p["timing"] = {"queued": _dt.datetime.now().isoformat(timespec="seconds")}
    emap = _energy_map(p, pk)  # kept in the plan too (p["energy_map"])
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
        # at the front, like every later stage: a run's renders go before the next queued Director,
        # so several runs queued at once finish one after another (Oct 27: five Directors queued
        # behind sx05f ran before its portraits)
        first = [g for name, g in jobs
                 if not (name == "zz_assemble" and (not queue_assemble or use_qc)) and name not in deferred]
        ids = queue_chain_front(first, server_url())
    summary = (f"OneShot MV run '{run_id}': {mode} / {pack} / {fmt} / {label}\n"
               + (themes.summary(bible) + "\n" if bible else "") + emap +
               f"window {p['song_t0']:.2f}-{p['song_t1']:.2f}s at {p['tempo']} BPM, "
               f"{len(p.get('shots', []))} shots, {len(p['renders'])} renders, {len(jobs)} jobs "
               f"{'queued' if queue else 'written (not queued)'}.\nOutput folder: {run_dir}")
    try:
        summary += "\n\n" + plan_preview(p, queued=queue)
    except Exception as e:  # never let the preview stop a run
        log(f"[OneShotMV] preview skipped ({e})")
    try:  # the summary rides along in jobs.json, so an SDK client can print the plan
        bp = os.path.join(run_dir, "jobs.json")
        b = json.load(open(bp, encoding="utf-8"))
        b.update(summary=summary, queued=bool(queue))
        json.dump(b, open(bp, "w", encoding="utf-8"))
    except (OSError, ValueError) as e:
        log(f"[OneShotMV] summary not written to jobs.json ({e})")
    return {"run_dir": run_dir, "run_id": run_id, "run_rel": rel, "jobs": len(jobs), "prompt_ids": ids,
            "summary": summary, "bundle": os.path.join(run_dir, "jobs.json")}


def run_status(run_dir: str) -> dict:
    """Where a run is: done (with its final video) or still rendering, with what is made so far.
    FVRunResult hands this to API / SDK clients; sdk_run.py waits on it."""
    import glob
    pj = os.path.join(run_dir, "plan.json")
    if not run_dir or not os.path.exists(pj):
        return {"state": "missing", "video": "", "message": f"no OneShot MV run at '{run_dir}'"}
    p = json.load(open(pj, encoding="utf-8"))
    tm = p.get("timing") or {}
    queued = tm.get("queued", "")
    stills = len(glob.glob(os.path.join(run_dir, "*.png")))
    clips = len([f for f in glob.glob(os.path.join(run_dir, "*.mp4"))
                 if not os.path.basename(f).startswith("final")])
    planned = len(p.get("renders", []))
    steps = sorted((v, k) for k, v in tm.items() if isinstance(v, str))
    last_at, last = steps[-1] if steps else ("", "")
    done = None
    dj = os.path.join(run_dir, "done.json")
    if os.path.exists(dj):
        try:
            d = json.load(open(dj, encoding="utf-8"))
            if d.get("at", "") >= queued and os.path.exists(d.get("video", "")):
                done = d
        except (OSError, ValueError):
            pass
    st = {"state": "done" if done else "rendering", "run_dir": run_dir, "run_id": p.get("run_id", ""),
          "video": done["video"] if done else "", "stills": stills, "clips": clips,
          "renders_planned": planned, "last_step": last, "last_step_at": last_at,
          "missing_renders": (done or {}).get("missing_renders", [])}
    st["message"] = (f"done: {os.path.basename(done['video'])}" if done else
                     f"rendering: {stills} stills, {clips} clips so far ({planned} renders planned)"
                     + (f", last step {last} at {last_at[11:16]}" if last else ""))
    return st


def _clock(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def plan_preview(p: dict, queued: bool = True) -> str:
    """The plan as the user will see it, shot by shot (the user, Oct 27: show the hidden step, the
    World the tool wrote from a short theme, before anything renders): time, energy, framing and
    who is in each shot, where, in what; then the World's characters left out, and how to change it."""
    th = p.get("theme") or {}
    cast = {c["id"]: c for c in p.get("cast") or []}
    lines = ["PLAN" + ("" if queued else " (preview: nothing rendered; turn queue_renders on to render it)")]
    if th.get("theme"):
        lines.append("World: " + th["theme"].split(":")[0].strip())
    if th.get("camera"):
        lines.append("Camera: " + th["camera"])
    lines.append(f"Cast order: {p.get('cast_order', 'as written')}   Crowd behind singers: {p.get('crowd', 'auto')}")
    lines.append("")
    used = set()
    for s in p.get("shots", []):
        rs = [r for r in p["renders"] if r.get("shot") == s["index"] and not r.get("unused")]
        br = next((r for r in rs if r.get("role") == "R"), None)
        a = next((r for r in rs if r.get("role") == "A"), None)
        head = f"{s['index'] + 1:>3}  {_clock(s['t0'])}  {s['t1'] - s['t0']:4.1f}s  {s.get('energy', ''):<6} {s.get('size', ''):<6}"
        if br is not None and (br.get("broll") or {}).get("check"):
            ck = br["broll"]["check"]
            who = f"b-roll: {ck.get('role', '')} ({ck.get('gender', '')}, {ck.get('age', '')})"
            where = ck.get("place", "")
            wear = ck.get("wear", "")
        elif br is not None:
            who, where, wear = "b-roll: " + ((br.get("broll") or {}).get("line") or "a cutaway")[:60], "", ""
        elif a is not None and a.get("performer") in cast:
            c = cast[a["performer"]]
            ck = c.get("check") or {}
            if c.get("char") is not None:
                used.add(c["char"])
            who = f"{ck.get('role') or c.get('who', '')} ({ck.get('gender', '')}, {ck.get('age', '')})" if ck else c.get("who", "")
            where, wear = ck.get("place", ""), ck.get("wear", "")
        else:
            who, where, wear = "?", "", ""
        lines.append(f"{head}  {who}" + (f" | {wear}" if wear else "") + (f" | {where}" if where else ""))
    chars = th.get("characters") or []
    left = [str(ch.get("role") or ch.get("who", "").split(",")[0]) for i, ch in enumerate(chars) if i not in used]
    if left:
        lines.append("")
        lines.append(f"In the World but not in this video ({len(left)}): " + ", ".join(left[:30]))
    lines.append("")
    lines.append("To change it: edit the World line (theme). The characters it names are cast first, in the order "
                 "it names them; say who leads, where and in what. The same seed keeps this cast; a new seed "
                 "re-casts.")
    return "\n".join(lines)


def _energy_map(p: dict, pk: dict) -> str:
    """The song's sections in the window, the energy level each was given and how it is cut
    (the user, Oct 26: map the song into low / medium / high and edit each level its own way)."""
    ed = pk.get("edit") or {}
    if ed.get("cuts") != "energy" or not p.get("shots"):
        return ""
    modes = planner.cut_modes(ed)
    t0 = float(p.get("song_t0", 0.0))

    def mmss(t):
        return f"{int(t // 60)}:{t % 60:04.1f}"
    runs = []
    for s in p["shots"]:
        lv = planner.LEVEL_OF.get(s.get("phase"), "low")
        if runs and runs[-1][0] == lv:
            runs[-1][2] = s["t1"]
            runs[-1][3] += 1
        else:
            runs.append([lv, s["t0"], s["t1"], 1])
    labels = {round(x["start"], 2): x.get("label", "") for x in p.get("sections", [])}
    parts = []
    for lv, a, b, n in runs:
        ks = [k for k in labels if k <= a + 0.05]
        lab = labels[max(ks)] if ks else ""
        lab = "chorus" if lab == "outro" and lv == "high" else lab  # a loud last chorus is labelled outro
        parts.append(f"{mmss(t0 + a)}-{mmss(t0 + b)} {lv} ({lab}): {modes[lv]}, {n} shot{'s' if n > 1 else ''}")
    p["energy_map"] = parts
    return "energy map: " + " | ".join(parts) + "\n"


# ------------------------------------------------------------------ take QC + re-rolls
def _bump(graph: dict, k: int) -> dict:
    """New seeds for every sampler in a job graph."""
    for node in graph.values():
        ins = node.get("inputs", {})
        for key in ("noise_seed", "seed"):
            if isinstance(ins.get(key), int):
                ins[key] = (ins[key] + 7919 * k) % (2 ** 31 - 1)
    return graph


def _video_in(graph: dict) -> str | None:
    """The clip a job reads (a look's lead clip, for the pose guide)."""
    for node in graph.values():
        if node.get("class_type") == "FVLoadVideoPath":
            return node.get("inputs", {}).get("path")
    return None


def _job_file(run_dir: str, name: str) -> str | None:
    hits = sorted(f for f in os.listdir(os.path.join(run_dir, "jobs")) if f.endswith(f"_{name}.json"))
    return os.path.join(run_dir, "jobs", hits[0]) if hits else None


def _take_inputs(run_dir: str, path: str) -> tuple[str | None, str | None]:
    """The clip and the image a take's job read (a look take: the lead take it follows and its
    keyframe), from the saved job of that take (r039B_t2_00001_.mp4 -> jobs/qc2_r039B_t2.json)."""
    stem = os.path.basename(str(path).replace("\\", "/")).rsplit("_00001_", 1)[0]
    try:
        jf = _job_file(run_dir, stem)
        g = json.load(open(jf, encoding="utf-8")) if jf else {}
    except Exception:
        return None, None
    img = next((n.get("inputs", {}).get("path") for n in g.values() if n.get("class_type") == "FVLoadImagePath"),
               None)
    return _video_in(g), img


# LTX lands a pinned switch a little early (n46 test: 6 and 2 frames at 50 fps), so the pins sit
# this much after the beat; _attach_groups then lines each real switch up with its beat
PIN_LATE = 0.06
GROUP_MAX_SHIFT = 0.14  # s: how far a look may be moved off its lip sync to fit its beat


def _who_short(c: dict) -> str:
    """'a woman aged seventy-seven' from a cast entry (the look's keyframe shows the rest)."""
    w = (c.get("twin") or {}).get("who") or c.get("who") or "a person"
    return w.split(",")[0].split(" with ")[0].strip()


GROUP_OPEN = ("Head-and-shoulders shots of {n} different people, one after another, singing the same words "
              "straight into the camera: {who}. Hard cuts on the beat from one person to the next, each one in "
              "exactly the same pose, framing and head position as the one before; no dissolves and no morphing "
              "between faces.")
GROUP_NEG = ", dissolve, crossfade, morphing, blended faces"


def _group_prompt(gs: dict) -> str:
    who = gs["who"]
    seq = "first " + ", then ".join(who) if len(who) > 1 else who[0]
    rest = re.sub(r"^[^.]*\.\s*", "", gs["sing"], count=1)  # drop "Head-and-shoulders shot of X singing ..."
    rest = rest.replace(graphs.CONTINUITY, "").strip()
    return GROUP_OPEN.format(n=len(who), who=seq) + " " + rest


def _looks_jobs(p: dict, run_dir: str, log=print) -> tuple[list[dict], list[dict]]:
    """Staged runs, after the last singer QC pass: the look jobs (Klein keyframes, then the LTX
    look clips), following each lead's final take. A look group's later keyframes are taken from
    the lead's frame at their cut (minus the lead's lip-sync delay, so the switch lands on the
    beat on screen) and pinned there in the group's clip."""
    by_id = {r["id"]: r for r in p["renders"]}
    rel = p.get("run_rel", "")
    orig = {r["id"]: os.path.join(run_dir, f"r{r['id']:03d}A_00001_.mp4") for r in p["renders"] if r["role"] == "A"}
    grp_of = {i: gs for gs in p.get("look_groups", []) for i in gs["looks"]}

    def swap(g, old, new):
        return json.loads(json.dumps(g).replace(json.dumps(old)[1:-1], json.dumps(new)[1:-1]))

    key_jobs, clip_jobs = [], []
    for fn in p.get("deferred_jobs", []):
        path = os.path.join(run_dir, "jobs", fn)
        if not os.path.exists(path):
            continue
        g = json.load(open(path, encoding="utf-8"))
        name = fn.split("_", 1)[1][:-5]
        r = by_id.get(int(name[1:4]))
        if r is None or r.get("twin_of") not in by_id:
            continue
        a = by_id[r["twin_of"]]
        if a.get("path") and orig.get(a["id"]) != a["path"]:
            g = swap(g, orig[a["id"]], a["path"])
        gs = grp_of.get(r["id"])
        if name.startswith("k") and gs is not None and gs["looks"].index(r["id"]) > 0:
            lead_t = r["cut_at"] - a["t_start"] - float(a.get("sync_delay", 0.0) or 0.0)
            for node in g.values():
                if node.get("class_type") == "FVLoadVideoPath":
                    node["inputs"]["start_seconds"] = round(max(0.0, lead_t), 4)
        json.dump(g, open(path, "w", encoding="utf-8"), indent=1)  # the saved job is the one that runs
        (key_jobs if name.startswith("k") else clip_jobs).append(g)
    vocals = os.path.join(run_dir, "vocals.wav")
    vp = vocals if os.path.exists(vocals) else os.path.join(run_dir, "song.wav")
    for gs in p.get("look_groups", []):
        looks = [by_id[i] for i in gs["looks"]]
        a = by_id[gs["lead"]]
        d = float(a.get("sync_delay", 0.0) or 0.0)
        rfps = float(looks[0]["fps"])
        t0g = looks[0]["t_start"]
        end = looks[-1]["t_start"] + looks[-1]["frames"] / rfps
        frames = formats.ltx_frames(end - t0g, rfps)
        a_end = a["t_start"] + a["frames"] / rfps
        while frames > 9 and t0g + frames / rfps > a_end + 1e-6:  # the pose guide is the lead's clip
            frames -= 8
        pins = []
        late = int(round(PIN_LATE * rfps))
        for b in looks[1:]:
            tau = int(round((b["cut_at"] - t0g - d) * rfps)) + late
            if 4 <= tau <= frames - 6:
                pins.append((b["key"], tau, 1.0))
            else:
                b["unusable"] = ["group_pin"]
        name = f"g{gs['shot']:03d}"
        g = graphs.ltx_render(preset=p["preset"], fmt=p["format"], key_path=looks[0]["key"], prompt=_group_prompt(gs),
                              negative=gs["neg"] + GROUP_NEG, frames=frames, seed=gs["seed"], prefix=f"{rel}/{name}",
                              audio_path=vp, audio_seek=round(t0g + p["song_t0"] - p.get("stems_t0", 0.0), 4),
                              guide_video=a["path"], guide_start=looks[0].get("guide_start", 0.0),
                              guide_strength=gs.get("guide_strength", 1.0), guide_face=True, keyframes=pins,
                              continuity=False).nodes
        json.dump(g, open(os.path.join(run_dir, "jobs", f"look_{name}.json"), "w", encoding="utf-8"), indent=1)
        gs.update(t_start=t0g, frames=frames, fps=rfps, delay=d, pins=[t for _, t, _ in pins],
                  pinned=[b["id"] for b in looks[1:] if not b.get("unusable")],
                  path=os.path.join(run_dir, f"{name}_00001_.mp4"))
        clip_jobs.append(g)
    log(f"[OneShotMV] looks: {len(key_jobs)} keyframes, {len(clip_jobs)} clips "
        f"({len(p.get('look_groups', []))} look groups)")
    return key_jobs, clip_jobs


def _attach_groups(p: dict, run_dir: str, log=print) -> None:
    """Point each look of a rendered group at its stretch of the group's clip: find where the
    clip really switches person (qc.switch_frames), then give each look a start time that shows
    its own person for its whole beat, as close to lip sync as the switch allows (at most
    GROUP_MAX_SHIFT off); a look whose switch isn't a clean cut, or doesn't fit, is left out (the look before it
    plays on)."""
    from . import qc
    by_id = {r["id"]: r for r in p["renders"]}
    for gs in p.get("look_groups", []):
        if gs.get("attached") or not gs.get("path") or not os.path.exists(gs["path"]):
            continue
        looks = [by_id[i] for i in gs["looks"]]
        sw = qc.switch_frames(gs["path"], gs.get("pins", []))
        found = {pid: (x["frame"] if x["frame"] is not None and x["ratio"] >= 6 and x.get("spread", 1.0) <= 0.6
                       else None) for pid, x in zip(gs.get("pinned", []), sw)}
        fps, d, t0g = gs["fps"], gs["delay"], gs["t_start"]
        notes = []
        for j, b in enumerate(looks):
            sj = 0 if j == 0 else found.get(b["id"])  # where this look's person starts in the clip
            later = [k for k in range(j + 1, len(looks)) if looks[k]["id"] in found]
            # where the next person starts (None: that switch isn't a clean cut, so this look may
            # melt into the next one)
            sn = found[looks[later[0]]["id"]] if later else gs["frames"]
            if sj is None or sn is None or b.get("unusable"):
                b["unusable"] = sorted(set(b.get("unusable") or []) | {"group_switch"})
                notes.append(f"r{b['id']:03d}B out")
                continue
            nxt = looks[j + 1]["cut_at"] if j + 1 < len(looks) else gs.get("shot_t1", b["cut_at"] + 0.5)
            win = (nxt - b["cut_at"]) * fps
            tau = (b["cut_at"] - t0g - d) * fps
            o = min(max(tau, sj + 1), sn - 1 - win) if j + 1 < len(looks) else max(tau, sj + 1)
            if o < sj + 1 - 1e-6 or abs(o - tau) > GROUP_MAX_SHIFT * fps:
                b["unusable"] = sorted(set(b.get("unusable") or []) | {"group_timing"})
                notes.append(f"r{b['id']:03d}B out ({o - tau:+.0f} f)")
                continue
            b.update(path=gs["path"], t_start=round(b["cut_at"] - d - o / fps, 4), frames=int(sn),
                     group_offset=round(o - tau, 1), takes=[])
            b.pop("unusable", None)
            notes.append(f"r{b['id']:03d}B {o - tau:+.0f} f")
        gs["attached"], gs["switches"] = True, sw
        log(f"[OneShotMV] look group g{gs['shot']:03d}: switches {[x['frame'] for x in sw]} "
            f"(pins {gs.get('pins')}); " + ", ".join(notes))


def _fallback_looks(p: dict, run_dir: str, log=print) -> tuple[list[dict], list[dict], set]:
    """Looks a group clip didn't deliver (no clean switch, or it didn't fit its beat) get their
    own clip, as without look groups: a keyframe from the lead's frame at the look's own start
    (Klein) and a pose-guided LTX clip. n47: only 10 of 36 group looks came out clean (the
    clip left the first look in its first frames, or morphed instead of cutting)."""
    by_id = {r["id"]: r for r in p["renders"]}
    cast = p.get("cast", [])
    pk = casting.load_pack(p["pack"])
    dense_from = float(pk["casting"].get("dense_from", 0.6))
    grp = {gs["shot"]: gs for gs in p.get("look_groups", [])}
    rel = p.get("run_rel", "")
    vocals = os.path.join(run_dir, "vocals.wav")
    vp = vocals if os.path.exists(vocals) else os.path.join(run_dir, "song.wav")
    fb = set(GROUP_FAULTS)
    keys, clips, ids = [], [], []
    for r in p["renders"]:
        if r["role"] != "B" or r.get("group") is None or not (set(r.get("unusable") or []) & fb):
            continue
        gs, a = grp.get(r["group"]), by_id.get(r["twin_of"])
        kf = _job_file(run_dir, f"k{r['id']:03d}")
        if gs is None or a is None or kf is None or r["performer"] >= len(cast):
            continue
        kg = json.load(open(kf, encoding="utf-8"))
        name = f"k{r['id']:03d}_s"
        for node in kg.values():
            ins = node.get("inputs", {})
            if node.get("class_type") == "FVLoadVideoPath":
                ins["path"], ins["start_seconds"] = a["path"], float(r.get("guide_start", 0.0))
            if node.get("class_type") == "SaveImage":
                ins["filename_prefix"] = f"{rel}/{name}"
        json.dump(kg, open(os.path.join(run_dir, "jobs", f"fb_{name}.json"), "w", encoding="utf-8"), indent=1)
        key = os.path.join(run_dir, f"{name}_00001_.png")
        c = cast[r["performer"]]
        text = casting.articulate(c["sing"]) if a.get("vocal_density", 0.0) >= dense_from else c["sing"]
        rg = graphs.ltx_render(preset=p["preset"], fmt=p["format"], key_path=key, prompt=text, negative=gs["neg"],
                               frames=r["frames"], seed=c["seed"], prefix=f"{rel}/r{r['id']:03d}B",
                               audio_path=vp, audio_seek=round(r["t_start"] + p["song_t0"] - p.get("stems_t0", 0.0), 4),
                               guide_video=a["path"], guide_start=r.get("guide_start", 0.0),
                               guide_strength=gs.get("guide_strength", 1.0), guide_face=True).nodes
        json.dump(rg, open(os.path.join(run_dir, "jobs", f"fb_r{r['id']:03d}B.json"), "w", encoding="utf-8"), indent=1)
        keys.append(kg)
        clips.append(rg)
        r.update(path=os.path.join(run_dir, f"r{r['id']:03d}B_00001_.mp4"), key=key, takes=[],
                 group_fallback=r.pop("group"))
        r.pop("unusable", None)
        ids.append(r["id"])
    if ids:
        log(f"[OneShotMV] looks: {len(ids)} group looks get their own clip")
    return keys, clips, set(ids)


GROUP_FAULTS = ("group_switch", "group_timing", "group_pin")


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


STEADY = "The camera holds still at the same distance; the framing never gets closer."


def _steady_prompt(g: dict) -> None:
    """A lead re-rolled for being too close (LTX pushed in on the face, n41) loses its camera-move
    sentence and holds its distance instead."""
    import re
    for node in g.values():
        if node.get("class_type") != "LTXVConditioning":
            continue
        ref = node.get("inputs", {}).get("positive")
        enc = g.get(str(ref[0])) if isinstance(ref, list) and ref else None
        if enc and enc.get("class_type") == "CLIPTextEncode" and isinstance(enc["inputs"].get("text"), str):
            t = enc["inputs"]["text"]
            if STEADY in t:
                continue
            out = []
            for x in re.split(r"(?<=[.!?])\s+", t):
                if re.search(r"\bsing", x, re.I) or "lips" in x:  # the subject and the singing stay
                    x = re.sub(r"(?i)(handheld )?(extreme )?close-up", "head-and-shoulders shot", x)
                    out.append(x[:1].upper() + x[1:])
                elif not re.search(r"(?i)\b(camera|close)\b", x):  # the move sentence goes
                    out.append(x)
            enc["inputs"]["text"] = " ".join(out + [STEADY])


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


def _long(r: dict) -> bool:
    """Long enough for SyncNet to judge (the energy grammar's drop shots run about half a second:
    lip sync can't be read there, and a re-roll for it would be wasted)."""
    return r.get("frames", 0) / max(1.0, float(r.get("fps", 25) or 25)) >= 1.6


def qc_pass(run_dir: str, *, pass_idx: int = 0, max_passes: int = 1, queue: bool = True,
            then_assemble: bool = True, queue_rerolls: bool = True, stage: str = "all", log=print) -> dict:
    """Check every render of a run; re-roll the bad ones (new seeds) and queue the next QC pass,
    or, when all pass / retries are spent, keep the best take of each shot and queue assembly.
    stage (staged runs, see direct()): 'leads' checks and re-rolls the singer shots only and, when
    they are final, queues the looks made from them plus a 'looks' pass; 'looks' checks the looks
    and queues assembly. 'all' is the single-pass order."""
    from . import qc
    pj = os.path.join(run_dir, "plan.json")
    p = json.load(open(pj, encoding="utf-8"))
    if stage not in ("all", "leads", "looks"):
        stage = "all"
    keep = p.get("preset") if stage != "all" else None

    def in_stage(r):
        if stage == "leads":
            return r["role"] == "A"
        if stage == "looks":
            return r["role"] == "B"
        return not r.get("broll_candidate") or not r.get("unused")
    fb_keys, fb_clips, fb_ids = [], [], set()
    if stage == "looks":
        _attach_groups(p, run_dir, log=log)
        if pass_idx < max_passes and queue_rerolls:
            fb_keys, fb_clips, fb_ids = _fallback_looks(p, run_dir, log=log)
    tm = p.setdefault("timing", {})
    tm[f"qc_{stage}_{pass_idx}"] = _dt.datetime.now().isoformat(timespec="seconds")
    voice = p.get("mode", "voice") == "voice"
    vocal = sr = None
    vp = os.path.join(run_dir, "vocals.wav")
    if voice and os.path.exists(vp):
        vocal, sr = stems.read_wav(vp)
    report, redo = [], []
    # artist runs: the artist's face, to catch looks that turned into someone else
    ref = None
    if voice and p.get("artist_path"):
        try:
            from . import identity
            if identity.available() and os.path.exists(p["artist_path"]):
                ref = identity.embed_path(p["artist_path"])
        except Exception as e:
            log(f"[OneShotMV] identity check off ({e})")

    def checks(r):  # the face checks a render gets: framing for every sung shot, identity for the artist
        if not voice or r["role"] not in ("A", "B"):
            return {}
        # a look takes its lead's framing (Face Match), so only leads are judged for "too close"
        return {"framing": r.get("size", "close") if r["role"] == "A" else "look", "ref": ref,
                "key_path": r.get("key") if r["role"] == "B" else None}

    for r in p["renders"]:
        if not in_stage(r) or (stage == "looks" and r.get("unusable") and r.get("group") is not None) \
                or r["id"] in fb_ids:  # a look just given its own clip is checked next pass
            continue
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
                                vocal=vocal, sr=sr or 44100, seek=seek,
                                sync=voice and r["role"] == "A" and r.get("size", "close") != "wide" and _long(r),
                                people=not voice, **checks(r))
        except Exception as e:  # never let QC block a cut
            res = {"path": r.get("path", ""), "ok": True, "fail": [], "score": 0.5, "error": str(e)}
        if r.get("group") is not None and set(res.get("fail", [])) & {"cut", "drift"}:
            # a look group's clip cuts on purpose (its switches were checked by _attach_groups),
            # and its face sizes step from person to person
            res["fail"] = [f for f in res["fail"] if f not in ("cut", "drift")]
            res["ok"] = not res["fail"]
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
        if r.get("group") is not None:  # a look group isn't re-rolled: a bad look stays out of the cut
            continue
        # a re-rolled lead clip invalidates its pose-matched twins: re-render them on the new lead
        # (staged runs make the looks after their lead is final)
        group = [r] + ([t for t in p["renders"] if t.get("twin_of") == r["id"] and t["role"] == "B"
                        and t["id"] not in redo_ids] if stage == "all" else [])
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
    last = pass_idx >= max_passes or not (redo or fb_clips)
    queued, rerolls = [], []
    if not last:
        redo.sort(key=lambda r: (r["role"] == "B", r["id"]))
        new_paths = {}
        rel = p.get("run_rel", "")
        lead_jobs, key_jobs, twin_jobs, portrait_jobs = [], [], [], []

        def swap(g, old, new):
            return json.loads(json.dumps(g).replace(json.dumps(old)[1:-1], json.dumps(new)[1:-1]))

        for r in redo:
            name = f"r{r['id']:03d}{r['role']}"
            jf = _job_file(run_dir, name)
            if not jf:
                continue
            g = _bump(json.load(open(jf, encoding="utf-8")), pass_idx + 1)
            # the saved job is the first take's: bring it up to the lead clip and the keyframe this
            # render uses now (a look whose lead was re-rolled in an earlier pass follows the new
            # lead; a keyframe or portrait re-made in an earlier pass stays)
            lead = by_id.get(r.get("twin_of")) if r["role"] == "B" else None
            orig_lead = ((lead or {}).get("takes") or [{}])[0].get("path")
            cur_lead = (lead or {}).get("path")

            def follow_lead(graph):
                if orig_lead and cur_lead and orig_lead != cur_lead:
                    return swap(graph, orig_lead, cur_lead)
                return graph

            g = follow_lead(g)
            cur_key = r.get("portrait") if r["role"] == "A" else r.get("key")
            for gnode in g.values():
                if gnode.get("class_type") == "FVLoadImagePath":
                    if cur_key and gnode["inputs"].get("path") not in (None, cur_key):
                        g = swap(g, gnode["inputs"]["path"], cur_key)
                    break
            _refresh_prompt(g)
            if r["role"] == "A" and "sync" in next((x.get("fail", []) for x in report if x["id"] == r["id"]), []):
                _articulate_prompt(g)
            if r["role"] == "A" and set(next((x.get("fail", []) for x in report if x["id"] == r["id"]), [])) \
                    & {"tooclose", "noface", "headcut"}:
                _steady_prompt(g)
            tag = f"{name}_t{pass_idx + 1}"
            for node in g.values():
                ins = node.get("inputs", {})
                if node.get("class_type") == "SaveVideo" and str(ins.get("filename_prefix", "")).endswith(name):
                    ins["filename_prefix"] = f"{rel}/{tag}"
            fails = next((x.get("fail", []) for x in report if x["id"] == r["id"]), [])
            needs_key = False

            def key_in(graph, default):
                for gnode in graph.values():
                    if gnode.get("class_type") == "FVLoadImagePath":
                        return gnode["inputs"].get("path", default)
                return default

            def remake(kname, graph, bump=True):
                """Re-make a keyframe / portrait job (new seeds, following the current lead clip);
                returns the render graph switched to the new image, or None without a job."""
                kf = _job_file(run_dir, kname)
                if not kf:
                    return None, None
                kg = follow_lead(json.load(open(kf, encoding="utf-8")))
                if bump:
                    kg = _bump(kg, pass_idx + 1)
                ktag = f"{kname}_t{pass_idx + 1}"
                for node in kg.values():
                    ins = node.get("inputs", {})
                    if node.get("class_type") == "SaveImage" and str(ins.get("filename_prefix", "")).endswith(kname):
                        ins["filename_prefix"] = f"{rel}/{ktag}"
                json.dump(kg, open(os.path.join(run_dir, "jobs", f"qc{pass_idx + 1}_{ktag}.json"), "w",
                                   encoding="utf-8"), indent=1)
                # a new portrait is a Krea job: run between LTX clips it thrashed VRAM for 20+ min
                # (stress test st4); portraits go first, after a free-memory job
                (portrait_jobs if kname.startswith("p") else key_jobs).append(kg)
                new_key = os.path.join(run_dir, f"{ktag}_00001_.png")
                old_key = key_in(graph, os.path.join(run_dir, f"{kname}_00001_.png"))
                return swap(graph, old_key, new_key), new_key

            if r["role"] in ("M", "MB") and "people" in fails:
                # a second person came from the character keyframe: re-make it with a new seed
                g2, nk = remake(f"k{r['id']:03d}", g)
                if g2 is not None:
                    g, needs_key, r["key"] = g2, True, nk
            if r["role"] == "B" and r.get("twin_of") in new_paths:
                # follow the lead clip re-rolled in this pass, and re-make the look's keyframe from
                # it, so its first frame has the new lead's pose (the old keyframe would snap from
                # the old pose to the new); an identity fault gets new seeds with it
                old = cur_lead or orig_lead
                g = swap(g, old, new_paths[r["twin_of"]])
                cur_lead = new_paths[r["twin_of"]]
                g2, nk = remake(f"k{r['id']:03d}", g, bump="identity" in fails)
                if g2 is not None:
                    g, needs_key, r["key"] = g2, True, nk
            elif r["role"] == "B" and "identity" in fails:
                # the look's keyframe edit turned the artist into someone else: a new edit
                g2, nk = remake(f"k{r['id']:03d}", g)
                if g2 is not None:
                    g, needs_key, r["key"] = g2, True, nk
            if r["role"] == "A" and voice and set(fails) & {"noface", "tooclose", "identity", "headcut", "sideways", "people", "bars"}:
                # a framing / identity fault that is already in the portrait (the head out of
                # frame in the still itself) needs a new portrait, not just a new seed for LTX
                pname = f"p{r['performer']:03d}"
                old_p = key_in(g, os.path.join(run_dir, f"{pname}_00001_.png"))
                bad = qc.image_bars(old_p) >= qc.THRESH["bars"]  # the bars are in the still itself
                try:
                    from . import identity
                    rep = identity.image_report(old_p, ref) if identity.available() else {}
                    lim = qc.THRESH["tooclose"].get(r.get("size", ""))
                    bad = bad or (rep.get("face_frac", 1.0) < 1.0 or (lim is not None and rep.get("face_h", 0.0) > lim)
                           or rep.get("tilt", 0.0) > qc.THRESH["tilt"] or rep.get("second_face", 0.0) >= 0.5
                           or (r.get("size") in ("medium", "wide") and rep.get("head_top", 1.0) < 0.0)
                           or (ref is not None and rep.get("face_h", 0.0) >= qc.THRESH["id_face_h"]
                               and rep.get("id_sim", 1.0) < qc.THRESH["id_min_key"]))
                except Exception:
                    pass  # keep the bars verdict
                # the fault survived a re-roll on the same still (sx04w's store owner: the head out of
                # frame in all three takes, steady camera and all): the still is the problem
                if not bad and pass_idx >= 1 and set(fails) & {"headcut", "noface", "sideways", "people", "bars"}:
                    bad = True
                if bad:
                    g2, nk = remake(pname, g)
                    if g2 is not None:
                        g, needs_key, r["portrait"] = g2, True, nk
            new_paths[r["id"]] = os.path.join(run_dir, f"{tag}_00001_.mp4")
            r["path"] = new_paths[r["id"]]
            json.dump(g, open(os.path.join(run_dir, "jobs", f"qc{pass_idx + 1}_{tag}.json"), "w", encoding="utf-8"),
                      indent=1)
            # renders that wait on a new keyframe run after the keyframe jobs
            (twin_jobs if (r["role"] == "B" or needs_key) else lead_jobs).append(g)
        # grouped by model: LTX, Klein, LTX (renders after their keys)
        from . import graphs
        jobs = lead_jobs + key_jobs + fb_keys + twin_jobs + fb_clips
        if portrait_jobs:
            jobs = [graphs.free_memory("before QC portrait re-rolls").nodes] + portrait_jobs + jobs
        rerolls = list(jobs)
        if queue_rerolls:  # chain the next pass on the server
            jobs.append(graphs.qc(run_dir=run_dir, pass_index=pass_idx + 1, max_passes=max_passes,
                                  then_assemble=then_assemble, stage=stage, keep=keep).nodes)
        if queue and queue_rerolls:
            queued = queue_chain_front(jobs, server_url())
        msg = f"QC pass {pass_idx}: re-rolling {len(redo)} take(s): " + ", ".join(
            f"r{r['id']:03d}{r['role']}" for r in redo) + \
            (f"; {len(fb_clips)} group looks get their own clip" if fb_clips else "") + \
            (f" ({skipped} more over the re-render budget)" if skipped else "")
    else:
        # best take of each shot: leads first, so each look then picks among the takes made from
        # the lead take that is kept (n46: lead r037A kept its first re-roll while its looks kept
        # takes made from the original lead and from the second re-roll, so their poses and
        # mouths no longer followed the lead in the cut)
        for r in sorted(p["renders"], key=lambda x: x["role"] in ("B", "MB")):
            if not in_stage(r):
                continue
            takes = [t for t in r.get("takes", []) if t.get("path") and os.path.exists(t["path"])]
            lead = by_id.get(r.get("twin_of")) if r["role"] == "B" else None
            if lead is not None and len(takes) > 1:
                same = [t for t in takes if _take_inputs(run_dir, t["path"])[0] == lead.get("path")]
                takes = same or takes
            for t in takes:  # takes judged by an older QC (before the cut / sync checks): judge again
                if t.get("v", 0) < qc.VERSION and t.get("frames"):
                    try:
                        seek = r.get("t_start", 0.0) + p.get("song_t0", 0.0) - p.get("stems_t0", 0.0)
                        res = qc.check_clip(t["path"], expected_frames=r.get("frames", 0), voice=voice,
                                            vocal=vocal, sr=sr or 44100, seek=seek,
                                            sync=voice and r["role"] == "A" and r.get("size", "close") != "wide" and _long(r),
                                            people=not voice, **checks(r))
                        t.clear()
                        t.update({k: v for k, v in res.items() if k != "ok"})
                    except Exception:
                        pass
            if takes:
                # newest take wins ties (re-rolled twins follow their re-rolled lead clip)
                # a take with a visual fault (frozen, headless, barred...) loses to one that only
                # sings off the beat: sx09w kept a frozen take over two with weak lip sync
                fatal = {"missing", "frozen", "exposure", "bars", "headcut", "noface", "sideways", "people", "cut"}
                best = max(enumerate(takes), key=lambda it: (not (set(it[1].get("fail") or []) & fatal),
                                                             round(it[1].get("score", 0.0), 2), it[0]))[1]
                r["path"] = best["path"]
                if r["role"] == "B" and r.get("group") is None:  # the keyframe of the take that is kept
                    r["key"] = _take_inputs(run_dir, best["path"])[1] or r.get("key")
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
                # (runs from before Oct 28 may hold a LatentSync-repaired lead: its twins follow
                # the original take's offset)
                r["sync_delay"] = lead.get("sync_delay_ltx", lead["sync_delay"]) if lead.get("latentsync") \
                    else lead["sync_delay"]
            # a delay can't reach back past the clip's own lead-in (it would hold the first
            # frame at the cut)
            if "sync_delay" in r:
                on = r.get("cut_at", shots.get(r.get("shot"), {}).get("t0", r["t_start"]))
                r["sync_delay"] = round(min(r["sync_delay"], max(0.0, on - r["t_start"] - 0.02)), 3)
        # b-roll cutaways (over weak lip sync and instrumental gaps), made once, before assembly
        post = []
        if voice and float(p.get("broll_share", 0.0)) > 0 and not p.get("broll_planned"):
            try:
                from . import broll
                post = broll.plan(p, run_dir, casting.load_pack(p["pack"]), log=log)
            except Exception as e:  # never let b-roll block the cut
                log(f"[OneShotMV] b-roll skipped ({e})")
                post = []
            p["broll_planned"] = True
        rerolls = list(post)  # SDK clients run these themselves (qc bundle)
        looks_msg = ""
        if stage == "leads":
            # the singer takes are final: keep the best b-roll candidates, then make the looks from
            # the final singer takes (Klein keyframes, LTX look clips) and check them
            from . import graphs
            if p.get("broll_candidates"):
                try:
                    from . import broll
                    broll.finalize(p, log=log)
                except Exception as e:  # never let b-roll block the cut
                    log(f"[OneShotMV] b-roll finalize skipped ({e})")
            key_jobs, clip_jobs = _looks_jobs(p, run_dir, log=log)
            rerolls = key_jobs + clip_jobs
            looks_msg = f"; {len(key_jobs)} look keyframes and {len(clip_jobs)} look clips queued"
            if queue and queue_rerolls:
                queued = queue_chain_front(rerolls + [graphs.qc(run_dir=run_dir, pass_index=0, max_passes=1,
                                                                then_assemble=then_assemble, stage="looks",
                                                                keep=keep).nodes], server_url())
        elif queue and then_assemble and queue_rerolls:
            from . import graphs
            queued = queue_chain_front(post + [graphs.free_memory("before assembly").nodes,
                                               graphs.assemble(run_dir=run_dir, prefix=p.get("run_rel", "")).nodes],
                                       server_url())
        bad = [f"r{x['id']:03d}{x['role']}:{'+'.join(x['fail'])}" for x in report if not x["ok"]]
        nb = sum(1 for r in p["renders"] if r.get("role") == "R")
        msg = f"QC pass {pass_idx}: {len(report) - len(bad)}/{len(report)} takes pass" + \
              (f"; kept best take for {', '.join(bad)}" if bad else "") + \
              (f"; {nb} b-roll cutaway(s) queued" if post else "") + looks_msg + \
              ("; assembling" if then_assemble and stage != "leads" else "")
    if ref is not None:
        try:
            from . import identity
            identity.free()
        except Exception:
            pass
    json.dump(p, open(pj, "w", encoding="utf-8"), indent=1)
    hist = os.path.join(run_dir, "qc.json")
    allq = json.load(open(hist, encoding="utf-8")) if os.path.exists(hist) else []
    allq.append({"pass": pass_idx, "report": report, "message": msg})
    json.dump(allq, open(hist, "w", encoding="utf-8"), indent=1)
    bundle = os.path.join(run_dir, f"qc{pass_idx}.json")
    json.dump({"pass": pass_idx, "final": last, "message": msg, "jobs": [{"graph": g} for g in rerolls]},
              open(bundle, "w", encoding="utf-8"))
    log(f"[OneShotMV] {msg}")
    return {"message": msg, "report": report, "queued": queued, "bundle": bundle, "final": last,
            "rerolls": len(rerolls)}


# ------------------------------------------------------------------ vision check of the stills
def _insert_fix(prompt: str, fix: str) -> str:
    """The fix lines right after the prompt's first sentence (the picture's subject), where the
    image model weighs them most."""
    if not fix:
        return prompt
    m = re.search(r"(?<=[a-z0-9)'\"])\.\s+", prompt)
    if not m:
        return prompt.rstrip() + " " + fix
    return prompt[:m.end()] + fix + " " + prompt[m.end():]


def _wall_place(prompt: str) -> str:
    """The second re-make for other people in the picture: a wall of the place right behind the
    person, so there is no depth for anyone to stand in."""
    def sub(m):
        place = re.sub(r"(?i)^(?:an?|the)\s+", "", m.group(2).strip())
        place = re.sub(r"(?i)\b(?:crowded|busy|packed|bustling|mirrored)\s+", "", place)
        return f"{m.group(1)} a plain, weathered wall of the {place} right behind them, with nothing and nobody else in view."
    return re.sub(r"(?i)\b(Background:|Setting:)\s*([^.]+)\.", sub, prompt, count=1)


def _empty_place(prompt: str) -> str:
    """'Background: a mirrored dressing room.' -> 'Background: the same place, empty after everyone has
    left, with no mirrors.' (a picture re-made for other people in it)."""
    def sub(m):
        place = re.sub(r"(?i)^(?:an?|the)\s+", "", m.group(2).strip())
        place = re.sub(r"(?i)\b(?:crowded|busy|packed|bustling)\s+", "", place)
        mirrors = ""
        if re.search(r"(?i)\bmirror", place):  # a mirror shows the person twice, or someone behind
            place = re.sub(r"(?i)\bmirrored\s+|,?\s*\b(?:lined|ringed|filled|full)\s+with\s+mirrors?\b", "", place)
            mirrors = ", plain walls and no mirrors"
        return f"{m.group(1)} the {place}, empty after everyone has left{mirrors}."
    return re.sub(r"(?i)\b(Background:|Setting:)\s*([^.]+)\.", sub, prompt, count=1)


def critic_pass(run_dir: str, *, pass_idx: int = 0, max_passes: int = 2, queue: bool = True, log=print) -> dict:
    """Check every still of a unique-cast run (fv/critic.py); re-make the ones with faults (new
    seeds, a line that says what to fix) and check again, at most max_passes times; then keep the
    best picture of each, bring the drop faces to one size and place (critic.fit_face), point the
    clip jobs at the kept pictures and queue them."""
    import gc
    from . import critic, graphs
    pj = os.path.join(run_dir, "plan.json")
    p = json.load(open(pj, encoding="utf-8"))
    cr = p.get("critic") or {}
    items = cr.get("items", [])
    th = p.get("theme") or {}
    world = th.get("theme", "")
    era = critic.check_era(world)
    tm = p.setdefault("timing", {})
    tm[f"critic_{pass_idx}"] = _dt.datetime.now().isoformat(timespec="seconds")
    t0 = _dt.datetime.now()
    checked = 0
    for it in items:
        if any(t.get("image") == it["image"] for t in it["takes"]):
            continue
        take = {"image": it["image"]}
        if not os.path.exists(it["image"]):
            take.update(faults=["missing"], score=9)
        else:
            try:
                text = critic.ask(it["image"], critic.rubric(it["check"], world, era))
                ans = critic.parse(text)
                fs = critic.faults(ans, it["check"], it["kind"], era)
                try:
                    sat = critic.saturation(it["image"])
                    take["saturation"] = round(sat, 3)
                    if sat < critic.MONO_BELOW:
                        fs.append("mono")
                except Exception:
                    pass
                take.update(answer=ans if ans else (text or "")[:300], faults=fs, score=critic.score(fs),
                            modern=critic.modern_items(ans))
            except Exception as e:  # never let the check stop a run
                take.update(faults=[], score=0, error=str(e)[:200])
        it["takes"].append(take)
        checked += 1
    critic.free()
    gc.collect()
    secs = (_dt.datetime.now() - t0).total_seconds()
    bad = [it for it in items if it["takes"] and it["takes"][-1].get("faults")
           and it["takes"][-1]["faults"] != ["unread"]]
    rel = p.get("run_rel", "")
    queued = []
    if bad and pass_idx < max_passes:
        jobs = []
        for it in bad:
            jf = _job_file(run_dir, it["job"])
            if not jf:
                continue
            g = _bump(json.load(open(jf, encoding="utf-8")), 31 * (pass_idx + 1))
            fix = critic.fix_lines(it["takes"][-1]["faults"], it["check"], era)
            tag = f"{it['name']}_c{pass_idx + 1}"
            people = "people" in it["takes"][-1]["faults"]
            crowded = sum(1 for tk in it["takes"] if "people" in (tk.get("faults") or [])) >= 2
            for node in g.values():
                ins = node.get("inputs", {})
                if node.get("class_type") == "CLIPTextEncode" and isinstance(ins.get("text"), str):
                    t = ins["text"]
                    if "mono" in it["takes"][-1]["faults"]:  # sx05c: "muted" colour went grey
                        t = re.sub(r"(?i)\bMuted, true-to-life colour\b", "Rich, natural, true-to-life colour", t)
                    if crowded:  # sx07c: the emptied beach club, market and rave still filled up
                        t = _wall_place(t)
                    elif people:  # st2e: a new seed alone kept the dressing room's mirrors and the arena's crowd
                        t = _empty_place(t)
                    ins["text"] = _insert_fix(t, fix)
                if node.get("class_type") == "SaveImage":
                    ins["filename_prefix"] = f"{rel}/{tag}"
            json.dump(g, open(os.path.join(run_dir, "jobs", f"critic{pass_idx + 1}_{tag}.json"), "w",
                           encoding="utf-8"), indent=1)
            it["image"] = os.path.join(run_dir, f"{tag}_00001_.png")
            jobs.append(g)
        chain = [graphs.free_memory("before vision-check re-makes").nodes] + jobs + \
            [graphs.free_memory("after vision-check re-makes").nodes,
             graphs.critic(run_dir=run_dir, pass_index=pass_idx + 1, max_passes=max_passes).nodes]
        if queue:
            queued = queue_chain_front(chain, server_url())
        msg = (f"vision check pass {pass_idx}: {checked} pictures in {secs:.0f} s, {len(bad)} re-made: " +
               ", ".join(f"{it['name']} ({'+'.join(it['takes'][-1]['faults'])})" for it in bad))
    else:
        # the best picture of each (fewest and lightest faults; the newest on a tie: it was made
        # with the fix lines)
        by_id = {r["id"]: r for r in p["renders"]}
        for it in items:
            takes = [t for t in it["takes"] if os.path.exists(t.get("image", ""))]
            best = min(enumerate(takes), key=lambda x: (x[1].get("score", 0), -x[0]))[1] if takes else \
                {"image": it["orig"], "faults": []}
            it["final"] = best["image"]
            it["final_faults"] = best.get("faults", [])
        # the drop's match cuts: every drop face to one size and one place
        drop = [it for it in items if it["kind"] == "portrait" and it.get("drop")]
        fitted = 0
        if drop:
            try:
                boxes = {it["name"]: critic.face_box(it["final"]) for it in drop}
                tw, ty = critic.fit_targets(list(boxes.values()))
                for it in drop:
                    b = boxes.get(it["name"])
                    out = os.path.splitext(it["final"])[0].replace("_00001_", "") + "_fit_00001_.png"
                    res = critic.fit_face(it["final"], out, b, tw, ty)
                    it["fit"] = res
                    if res and res.get("scale", 1.0) > 1.0 and os.path.exists(out):
                        it["final"] = out
                        fitted += 1
                cr["fit_target"] = [round(tw, 4), round(ty, 4)]
            except Exception as e:  # never let the framing stop a run
                log(f"[OneShotMV] vision check: face fitting skipped ({e})")
        # point the clip jobs at the kept pictures; a b-roll still that still shows a wrong person
        # (another gender, a beard, glasses, someone else in it) gets no clip
        swaps, dropped = {}, set()
        for it in items:
            if it["final"] != it["orig"]:
                swaps[it["orig"]] = it["final"]
            if it["kind"] == "portrait":
                for r in p["renders"]:
                    if r["role"] == "A" and r["performer"] == it["performer"]:
                        r["portrait"] = it["final"]
            elif set(it.get("final_faults") or []) & {"people", "gender", "beard", "glasses"}:
                r = by_id.get(it["render"])
                if r is not None:
                    r["unused"] = True
                    r["critic_reject"] = it["final_faults"]
                    dropped.add(f"r{r['id']:03d}R")
            else:
                r = by_id.get(it["render"])
                if r is not None:
                    r["key"] = it["final"]
        later = []
        for fn in cr.get("deferred", []):
            path = os.path.join(run_dir, "jobs", fn)
            if not os.path.exists(path):
                continue
            txt = open(path, encoding="utf-8").read()
            for a, b in swaps.items():
                txt = txt.replace(json.dumps(a)[1:-1], json.dumps(b)[1:-1])
            g = json.loads(txt)
            pref = next((str(n["inputs"].get("filename_prefix", "")) for n in g.values()
                         if n.get("class_type") == "SaveVideo"), "")
            if any(pref.endswith(d) for d in dropped):
                continue
            open(path, "w", encoding="utf-8").write(txt)
            later.append(g)
        chain = [graphs.free_memory("after the vision check").nodes] + later
        if queue:
            queued = queue_chain_front(chain, server_url())
        left = [f"{it['name']} ({'+'.join(it['final_faults'])})" for it in items if it.get("final_faults")]
        msg = (f"vision check pass {pass_idx}: {checked} pictures in {secs:.0f} s; final: "
               f"{len(items) - len(left)}/{len(items)} clean" + (f", kept best for {', '.join(left)}" if left else "") +
               (f"; {fitted} drop faces re-framed" if fitted else "") +
               (f"; {len(dropped)} b-roll still(s) dropped" if dropped else "") +
               f"; {len(later)} clip jobs queued")
        cr["done"] = True
    cr["passes"] = pass_idx + 1
    p["critic"] = cr
    json.dump(p, open(pj, "w", encoding="utf-8"), indent=1)
    hist = os.path.join(run_dir, "critic.json")
    allc = json.load(open(hist, encoding="utf-8")) if os.path.exists(hist) else []
    allc.append({"pass": pass_idx, "message": msg,
                 "items": [{"name": it["name"], "role": it["check"].get("role"), "gender": it["check"].get("gender"),
                            "takes": it["takes"]} for it in items]})
    json.dump(allc, open(hist, "w", encoding="utf-8"), indent=1)
    log(f"[OneShotMV] {msg}")
    return {"message": msg, "queued": queued}
