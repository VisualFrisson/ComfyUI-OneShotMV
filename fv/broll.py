"""B-roll: a few cutaways, nobody singing, that show what the words are about.

About one shot in eight (`broll_share`), never on the drop and never the opening shot, never
two in a row. Slots are filled first where lip sync is still weak after the re-rolls (a sung
shot whose SyncNet confidence stayed low), then in instrumental gaps (a lead with almost no
vocal under it). Each cutaway illustrates the line sung there (Whisper, fv/lyrics.py), written
as an image prompt by the same local language model as themes (fv/theme.py), in the theme's or
pack's world; Krea 2 makes the still and LTX moves it."""
from __future__ import annotations

import json
import os

from . import formats, graphs, lyrics

PROMPT = """You are editing a music video{world}. Write the image prompt for one b-roll shot, a cutaway where nobody sings, for the moment when the singer sings: "{line}"
Show the idea of these words literally or as a visual metaphor: an object, a place, a detail, hands, a landscape, the weather, the sky, a street. No real or famous people, no brands, no logos, no text or lettering anywhere. Answer with one sentence of at most 45 words that describes the image, and nothing else."""

PROMPT_NO_WORDS = """You are editing a music video{world}. Write the image prompt for one atmospheric b-roll shot, a cutaway where nobody sings, that fits this world: a place, an object, a detail, the light, the weather. No real or famous people, no brands, no logos, no text or lettering anywhere. Answer with one sentence of at most 45 words that describes the image, and nothing else."""

STILL = " Cinematic film still, shallow depth of field, natural colour, subtle film grain. No text, no lettering."
MOTION = (" The camera moves slowly and smoothly; gentle natural motion in the scene. Nobody sings or speaks. "
          "No text appears.")
NEGATIVE = ("text, lettering, subtitles, watermark, logo, singing, talking, microphone, blurry, low quality, "
            "flicker, deformed hands, distorted faces")


def choose(p: dict, share: float = 0.125, sync_min: float = 4.5, sung_from: float = 0.55,
           gap_below: float = 0.3) -> list[tuple[dict, str]]:
    """[(shot, reason)] for the b-roll slots of a voice run."""
    if share <= 0 or not p.get("shots"):
        return []
    by_id = {r["id"]: r for r in p["renders"]}
    shots = p["shots"]
    n = max(1, int(round(len(shots) * share)))
    cands = [s for s in shots if s.get("phase", "quiet") in ("quiet", "build") and s.get("index", 0) > 0
             and s["t1"] - s["t0"] >= 1.0 and "A" in s]

    def lead(s):
        return by_id.get(s["A"], {})

    weak = [s for s in cands if lead(s).get("sync_conf") is not None and lead(s)["sync_conf"] < sync_min
            and lead(s).get("vocal_density", 1.0) >= sung_from and lead(s).get("size", "close") != "wide"]
    weak.sort(key=lambda s: lead(s)["sync_conf"])
    gaps = [s for s in cands if lead(s).get("vocal_density", 1.0) < gap_below and s not in weak]
    gaps.sort(key=lambda s: lead(s).get("vocal_density", 1.0))
    picks = []
    for s, why in [(s, "weak lip sync") for s in weak] + [(s, "instrumental gap") for s in gaps]:
        if len(picks) >= n:
            break
        if any(abs(s["index"] - q["index"]) <= 1 for q, _ in picks):
            continue
        picks.append((s, why))
    return sorted(picks, key=lambda x: x[0]["index"])


def _world(p: dict, pack: dict) -> str:
    th = (p.get("theme") or {}).get("theme")
    if th:
        return f' set in this world: "{th}"'
    about = pack.get("about", "")
    return f" ({about})" if about else ""


def _prompt_for(p: dict, pack: dict, shot: dict, seed: int, log=print) -> tuple[str, str]:
    """(image prompt, the line it illustrates)."""
    from . import theme as themes
    lns = (p.get("lyrics") or {}).get("lines", [])
    ln = lyrics.line_near(lns, shot["t0"], shot["t1"])
    q = PROMPT.format(world=_world(p, pack), line=ln["text"].replace('"', "'")) if ln else \
        PROMPT_NO_WORDS.format(world=_world(p, pack))
    text = themes.llm(q, seed=seed, max_length=160, log=log)
    if text:
        text = " ".join(text.strip().strip('"').split())
        text = text.split("\n")[0][:400]
    if not text:  # no language model: the world itself
        th = p.get("theme") or {}
        place = (th.get("places") or ["a quiet street at dusk"])[seed % max(1, len(th.get("places") or [1]))]
        text = f"An empty scene {place}, atmospheric, no people"
    return text, (ln or {}).get("text", "")


def plan(p: dict, run_dir: str, pack: dict, log=print) -> list[dict]:
    """Pick the slots, write the prompts, add the b-roll renders to the plan; returns the job
    graphs (stills first, then the LTX clips) to queue before assembly."""
    picks = choose(p, share=float(p.get("broll_share", 0.0)))
    if not picks:
        return []
    rel = p.get("run_rel", "")
    preset, fmt = p["preset"], p["format"]
    rfps = p.get("render_fps", formats.PRESETS[preset]["fps"])
    s1, s2 = formats.portrait_size(fmt)
    keys, clips, made = [], [], []
    for k, (s, why) in enumerate(picks):
        rid = len(p["renders"])
        img, line = _prompt_for(p, pack, s, seed=p.get("seed", 1) * 100 + k, log=log)
        t_start = max(-p["song_t0"], s["t0"] - 0.2)
        frames = formats.ltx_frames(s["t1"] - t_start + 0.15, rfps)
        key = os.path.join(run_dir, f"k{rid:03d}R_00001_.png")
        r = {"id": rid, "shot": s["index"], "role": "R", "performer": -1, "t_start": round(t_start, 4),
             "frames": frames, "fps": rfps, "size": "wide", "sync_delay": 0.0, "key": key,
             "path": os.path.join(run_dir, f"r{rid:03d}R_00001_.mp4"), "broll": {"why": why, "line": line,
                                                                                   "prompt": img}}
        p["renders"].append(r)
        s["broll"] = rid
        seed = p.get("seed", 1) * 1000 + 900 + k
        keys.append(graphs.krea2_portrait(prompt=img + STILL, seed=seed, prefix=f"{rel}/k{rid:03d}R",
                                          size1=s1, size2=s2).nodes)
        clips.append(graphs.ltx_render(preset=preset, fmt=fmt, key_path=key, prompt=img + MOTION,
                                       negative=NEGATIVE, frames=frames, seed=seed,
                                       prefix=f"{rel}/r{rid:03d}R").nodes)
        made.append(f"shot {s['index']} ({why}): {line!r} -> {img}")
    for i, g in enumerate(keys + clips):
        json.dump(g, open(os.path.join(run_dir, "jobs", f"broll_{i:02d}.json"), "w", encoding="utf-8"), indent=1)
    log("[FrissonVision] b-roll: " + " | ".join(made))
    return keys + [graphs.free_memory("after b-roll stills").nodes] + clips
