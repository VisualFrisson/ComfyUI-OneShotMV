"""B-roll (Second Unit): a few cutaways where nobody sings: shots of the World itself (a telling
detail of one of its places), and performers dancing to the track (medium shots, not full body) as
filler. Cutaways that illustrate the lyrics (broll_style 'lyrics') are still there, but no longer
the default: in the Oct 24 samples they came out as a rock on concrete and a fruit bowl.

About one shot in eight (`broll_share`), never on the drop and never the opening shot, never
two in a row. Slots are filled first where lip sync is still weak after the re-rolls (a sung
shot whose SyncNet confidence stayed low), then in instrumental gaps (a lead with almost no
vocal under it). Each world cutaway is written as an image prompt by the same local language
model as themes (fv/theme.py), from one of the World's places and lights; Krea 2 makes the still
and LTX moves it. broll_style 'mix' (default) puts the world cutaways over weak lip sync and the
dance filler in instrumental gaps; 'world', 'lyrics' or 'dance' uses one kind everywhere."""
from __future__ import annotations

import json
import os
import re

from . import formats, graphs, lyrics

PROMPT = """You are editing a music video{world}. Write the image prompt for one b-roll shot, a cutaway where nobody sings, for the moment when the singer sings: "{line}"
Show the idea of these words literally or as a visual metaphor: an object, a place, a detail, a landscape, the weather, the sky, a street. No people at all, no brands, no logos, no text or lettering anywhere, nothing explicit. Make it look like a real photograph of an ordinary moment someone actually took: specific, believable details, nothing magical, surreal or too perfect. Answer with one sentence of at most 45 words that describes the image, and nothing else."""

PROMPT_WORLD = """You are editing a music video set in this world: "{world}". Write the image prompt for one b-roll cutaway: an ordinary, everyday view {place}, {light}, exactly as it really looks at that moment, with the things normally found there, and nobody in it. Nothing odd, symbolic, dramatic, sad or out of place, no story and no metaphor. No people, no brands, no logos, no recognisable car or product models, no text, screens or lettering anywhere. Answer with one sentence of at most 40 words that describes the image, and nothing else."""

PROMPT_NO_WORDS = """You are editing a music video{world}. Write the image prompt for one atmospheric b-roll shot, a cutaway where nobody sings, that fits this world: a place, an object, a detail, the light, the weather. No people at all, no brands, no logos, no text or lettering anywhere. Make it look like a real photograph of an ordinary moment: specific, believable details, nothing surreal or too perfect. Answer with one sentence of at most 45 words that describes the image, and nothing else."""

# dance filler (the n34-n36 notes: pose-guided full-body dance doesn't hold up in LTX, but a
# performer grooving to the track in a medium shot works as a cutaway between sung lines)
DANCE_STILL = ("Medium shot on a 35mm lens of {who}, wearing {wear}, dancing to the music with loose "
               "shoulders and one arm raised, eyes half closed, mouth closed in a slight smile, not singing. "
               "{light}. Setting: {place}. Photorealistic film still, an invented person, the only person in frame, "
               "natural colour. No text, no lettering.")
DANCE_MOTION = ("Medium shot of {who} dancing to the beat: shoulders, hips and arms move with the rhythm, the head "
                "bobs, a relaxed smile, the mouth stays closed, not singing. The handheld camera sways with the "
                "rhythm. No one else in frame. No text appears.")
DANCE_ARTIST = ("Photo of the same person as in image 1, photographed again somewhere new: a medium shot on a 35mm "
                "lens. They are dancing to the music with loose shoulders and one arm raised, eyes half closed, "
                "mouth closed in a slight smile. Wearing {wear}. {light}. Setting: {place}. Keep their exact face, "
                "identity, skin tone and hair. Candid photograph, natural colour, subtle film grain.")
# no "cinematic": in text to video it brings letterbox bars with it (n39's cutaways)
# the n39 note: the text-to-video cutaways "have a real AI feel": cutaways start from a Krea 2
# still (the realism model the portraits use) written as a real photograph, and LTX moves it
STILL = (" A real photograph, not a render: a full frame with true-to-life textures and natural light, no "
         "glossy CGI look. Nobody in the picture. No text, no lettering.")
MOTION = (" The camera moves slowly and smoothly; gentle natural motion in the scene. Nobody is in the frame. "
          "No black bars. No text appears.")
NEGATIVE = ("text, lettering, subtitles, watermark, logo, singing, talking, microphone, blurry, low quality, "
            "flicker, deformed hands, distorted faces, letterbox, black bars, widescreen bars, frame border")
# a lyric cutaway has nobody in it (n39: a stranger turned up on the beach in an artist's video)
NEGATIVE_EMPTY = NEGATIVE + ", person, people, face, crowd, figure, silhouette"


def choose(p: dict, share: float = 0.125, sync_min: float = 4.5, sung_from: float = 0.55,
           gap_below: float = 0.3, n: int | None = None, only: set | None = None) -> list[tuple[dict, str]]:
    """[(shot, reason)] for the b-roll slots of a voice run. n: how many (default: share of the
    shots); only: shot indices to choose among (the candidates rendered up front)."""
    if share <= 0 or not p.get("shots"):
        return []
    by_id = {r["id"]: r for r in p["renders"]}
    shots = p["shots"]
    n = n or max(1, int(round(len(shots) * share)))
    cands = [s for s in shots if s.get("phase", "quiet") in ("quiet", "build") and s.get("index", 0) > 0
             and s["t1"] - s["t0"] >= 1.0 and "A" in s and (only is None or s["index"] in only)]

    def lead(s):
        return by_id.get(s["A"], {})

    weak = [s for s in cands if lead(s).get("sync_conf") is not None and lead(s)["sync_conf"] < sync_min
            and lead(s).get("vocal_density", 1.0) >= sung_from and lead(s).get("size", "close") != "wide"]
    weak.sort(key=lambda s: lead(s)["sync_conf"])
    gaps = [s for s in cands if lead(s).get("vocal_density", 1.0) < gap_below and s not in weak]
    gaps.sort(key=lambda s: lead(s).get("vocal_density", 1.0))
    # when lip sync holds up everywhere and there are no gaps, cutaways still mix things up
    # (the n31b note: "not many, but to keep things interesting"): the weakest remaining shots
    rest = [s for s in cands if s not in weak and s not in gaps]
    rest.sort(key=lambda s: (lead(s).get("sync_conf") is None, lead(s).get("sync_conf") or 0.0))
    picks = []
    for s, why in [(s, "weak lip sync") for s in weak] + [(s, "instrumental gap") for s in gaps] + \
            [(s, "variety") for s in rest]:
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


def _clean(text: str) -> str:
    from . import theme as themes
    text = " ".join(text.strip().strip('"').split()).split("\n")[0][:400]
    # a recognisable car model is a brand (the Y2K sample opened on a blue sports car)
    text = re.sub(r"(?i)\b(sports|supercar|luxury|vintage|classic|muscle)\s+cars?\b", "car", text)
    return themes._debrand(text)


def _prompt_for(p: dict, pack: dict, shot: dict, seed: int, k: int = 0, style: str = "mix",
                log=print) -> tuple[str, str]:
    """(image prompt, the line it illustrates: '' for world cutaways)."""
    from . import theme as themes
    th = p.get("theme") or {}
    if style == "lyrics":
        lns = (p.get("lyrics") or {}).get("lines", [])
        ln = lyrics.line_near(lns, shot["t0"], shot["t1"])
        if ln:
            q = PROMPT.format(world=_world(p, pack), line=ln["text"].replace('"', "'"))
            text = themes.llm(q, seed=seed, max_length=160, log=log)
            if text:
                return _clean(text), ln["text"]
    if th.get("theme") and th.get("places"):
        # the characters' own workplaces first: the extra places drift off-world (Night Shift
        # retest: a record store, a bowling alley, a theatre lobby)
        places = [c["place"] for c in th.get("characters") or [] if c.get("place")] or th["places"]
        lights = th.get("light") or ["natural light"]
        place = places[(seed + k * 5) % len(places)]
        light = lights[(seed + k * 3) % len(lights)]
        q = PROMPT_WORLD.format(world=th["theme"].replace('"', "'"), place=place, light=light)
        text = themes.llm(q, seed=seed, max_length=160, log=log)
        if text:
            return _clean(text), ""
        return f"An ordinary, empty view {place}, {light}, no people", ""
    text = themes.llm(PROMPT_NO_WORDS.format(world=_world(p, pack)), seed=seed, max_length=160, log=log)
    if text:
        return _clean(text), ""
    return "An empty quiet street at dusk, atmospheric, no people", ""


def _dancer(p: dict, pack: dict, k: int) -> dict:
    """Who dances in a filler shot: someone from the run's own cast (theme or pack world)."""
    import random
    rng = random.Random(p.get("seed", 1) * 31 + k)
    th = p.get("theme") or {}
    if th.get("characters"):  # a whole character, so the outfit fits the dancer
        from .casting import _no_feet
        ch = rng.choice(th["characters"])
        place = ch.get("place") or rng.choice(th["places"])
        return {"who": ch["who"], "wear": _no_feet(ch["wear"]), "place": place.split(" ", 1)[-1],
                "light": rng.choice(th["light"])}
    if th.get("who"):
        return {"who": rng.choice(th["who"]), "wear": rng.choice(th["wear"]),
                "place": rng.choice(th["places"]).split(" ", 1)[-1], "light": rng.choice(th["light"])}
    cast = [c for c in p.get("cast", []) if c.get("twin")] or [{"twin": {}}]
    c = rng.choice(cast)["twin"]
    c2 = (pack.get("casting") or {})
    wear = (c.get("wear") or "").replace("Wearing ", "").rstrip(".") or \
        rng.choice(c2.get("wear") or ["their own clothes"])
    place = (c.get("background") or rng.choice(c2.get("places") or ["a dim club with coloured lights"]))
    return {"who": c.get("who") or "a person", "wear": wear, "light": c.get("light") or "warm practical light",
            "place": place.replace(" behind", "")}


def kind_for(why: str, style: str) -> str:
    """'words' (a cutaway that shows the lyric) or 'dance' (a performer dancing, not singing)."""
    if style == "dance":
        return "dance"
    if style in ("lyrics", "world"):
        return "words"
    return "dance" if why == "instrumental gap" else "words"  # mix


def plan(p: dict, run_dir: str, pack: dict, log=print, candidates: bool = False):
    """Pick the slots, write the prompts, add the b-roll renders to the plan; returns the job
    graphs (stills first, then the LTX clips) to queue before assembly.
    candidates (staged runs): made before any QC, so the weak lip-sync shots aren't known yet:
    twice the slots are rendered up front (stills with the portraits, clips with the singer
    shots, so no model swap is spent on them) and finalize() keeps the best after singer QC.
    Returns (stills, clips) then."""
    share = float(p.get("broll_share", 0.0))
    n = max(1, int(round(len(p.get("shots", [])) * share))) if share > 0 else 0
    picks = choose(p, share=share, n=2 * n if candidates else None)
    if not picks:
        return ([], []) if candidates else []
    style = p.get("broll_style", "mix")
    drums = os.path.join(run_dir, "drums.wav")
    rel = p.get("run_rel", "")
    preset, fmt = p["preset"], p["format"]
    rfps = p.get("render_fps", formats.PRESETS[preset]["fps"])
    s1, s2 = formats.portrait_size(fmt)
    keys, clips, made = [], [], []
    for k, (s, why) in enumerate(picks):
        rid = len(p["renders"])
        kind = kind_for(why, style)
        t_start = max(-p["song_t0"], s["t0"] - 0.2)
        frames = formats.ltx_frames(s["t1"] - t_start + 0.15, rfps)
        key = os.path.join(run_dir, f"k{rid:03d}R_00001_.png")
        seed = p.get("seed", 1) * 1000 + 900 + k
        line = ""
        # a Krea 2 still first, then LTX image to video: text to video alone looked AI-made (the n39
        # note); the model swap costs a few minutes per run. broll_stills False goes back to it
        stills = bool(p.get("broll_stills", True))
        if kind == "dance" and p.get("artist_path"):
            kind = "words"  # the artist's own dance would need an identity edit (another model swap)
        if kind == "dance":
            d = _dancer(p, pack, k)
            d["light"] = d["light"][:1].upper() + d["light"][1:]
            short = d["who"].split(",")[0].split(" with ")[0]
            img = " ".join(DANCE_STILL.format(**d).split())
            motion = " ".join(DANCE_MOTION.format(who=d["who"]).split()) + " " + \
                f"{d['light']}. Setting: {d['place']}. Photorealistic, full frame with no black bars, natural colour, " \
                "subtle film grain."
            if stills:
                keys.append(graphs.krea2_portrait(prompt=img, seed=seed, prefix=f"{rel}/k{rid:03d}R",
                                                  size1=s1, size2=s2).nodes)
            # the drum stem drives the movement (no vocal, so nobody starts singing)
            seek = round(t_start + p["song_t0"] - p.get("stems_t0", 0.0), 4)
            clips.append(graphs.ltx_render(preset=preset, fmt=fmt, key_path=key if stills else None,
                                           prompt=motion, negative=NEGATIVE, frames=frames, seed=seed,
                                           prefix=f"{rel}/r{rid:03d}R",
                                           audio_path=drums if os.path.exists(drums) else None,
                                           audio_seek=seek).nodes)
        else:
            img, line = _prompt_for(p, pack, s, seed=p.get("seed", 1) * 100 + k, k=k, style=style, log=log)
            if stills:
                keys.append(graphs.krea2_portrait(prompt=img + STILL, seed=seed, prefix=f"{rel}/k{rid:03d}R",
                                                  size1=s1, size2=s2).nodes)
            clips.append(graphs.ltx_render(preset=preset, fmt=fmt, key_path=key if stills else None,
                                           prompt=img + STILL + MOTION, negative=NEGATIVE_EMPTY, frames=frames,
                                           seed=seed, prefix=f"{rel}/r{rid:03d}R").nodes)
        r = {"id": rid, "shot": s["index"], "role": "R", "performer": -1, "t_start": round(t_start, 4),
             "frames": frames, "fps": rfps, "size": "medium" if kind == "dance" else "wide", "sync_delay": 0.0,
             "key": key, "path": os.path.join(run_dir, f"r{rid:03d}R_00001_.mp4"),
             "broll": {"kind": kind, "why": why, "line": line, "prompt": img}}
        p["renders"].append(r)
        if candidates:
            r["broll_candidate"] = True
            p.setdefault("broll_candidates", []).append(rid)
        else:
            s["broll"] = rid
        made.append(f"shot {s['index']} ({why}, {kind}): {line!r} -> {img[:120]}")
    for i, g in enumerate(keys + clips):
        json.dump(g, open(os.path.join(run_dir, "jobs", f"broll_{i:02d}.json"), "w", encoding="utf-8"), indent=1)
    log("[FrissonVision] b-roll" + (" candidates" if candidates else "") + ": " + " | ".join(made))
    if candidates:
        return keys, clips
    return (keys + [graphs.free_memory("after b-roll stills").nodes] if keys else []) + clips


def finalize(p: dict, log=print) -> list[int]:
    """Staged runs, after singer QC: keep the b-roll candidates whose shots are now the best
    slots (weak lip sync first, then instrumental gaps), as plan() would have chosen them."""
    by_id = {r["id"]: r for r in p["renders"]}
    cand = [by_id[i] for i in p.get("broll_candidates", []) if i in by_id]
    ok = [r for r in cand if r.get("path") and os.path.exists(r["path"])]
    shots = {s["index"]: s for s in p.get("shots", [])}
    share = float(p.get("broll_share", 0.0))
    n = max(1, int(round(len(shots) * share))) if share > 0 else 0
    picks = choose(p, share=share, n=n, only={r["shot"] for r in ok})
    chosen = []
    for s, why in picks:
        r = next(x for x in ok if x["shot"] == s["index"])
        s["broll"] = r["id"]
        r["broll"]["why_final"] = why
        chosen.append(r["id"])
    for r in cand:
        if r["id"] not in chosen:
            r["unused"] = True
    log(f"[FrissonVision] b-roll: kept {len(chosen)} of {len(cand)} candidates")
    return chosen
