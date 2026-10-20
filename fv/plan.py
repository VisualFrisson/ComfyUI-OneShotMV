"""Frisson Vision planners: song timeline + style pack -> renders + edit decision list.

Voice mode
  * The song is cut into shots on bar lines: `shot_bars_loud` bars in loud sections,
    `shot_bars_quiet` in quiet ones (the edamame edit: 2 bars loud, 1 bar quiet).
  * Every shot gets a new performer (A) lip-synced to that stretch of the vocal stem.
  * In loud sections a pose-matched twin (B: same pose, same vocals, new person) is rendered
    too, and the edit flashes to B on every `insert_every_beats`-th beat for a few frames,
    ghosting back to A. With probability `pingpong` the shot's second half switches to B.
Motion mode
  * Delegates the cut pattern to dsplan (a new character every N beats); each segment is a
    pose-guided render of the dance clip from a character keyframe.

Times in the plan are seconds relative to the window start; `song_t0` maps back to the song."""
from __future__ import annotations

import math
import random

from . import beat, dsplan, formats


def _sections_with_loudness(w: dict, thr: float) -> list[dict]:
    secs = w.get("sections") or [{"start": 0.0, "end": w["duration"], "energy": 1.0}]
    out = []
    for s in secs:
        e = float(s.get("energy", 1.0))
        out.append({"start": float(s["start"]), "end": float(s["end"]), "energy": e,
                    "loud": bool(e >= thr or s.get("is_drop"))})
    return out


def _loud_at(secs: list[dict], t: float) -> bool:
    for s in secs:
        if s["start"] - 1e-3 <= t < s["end"]:
            return s["loud"]
    return secs[-1]["loud"] if secs else True


def choose(tl: dict, seconds: float, start: float) -> tuple[float, float]:
    if seconds <= 0:
        return 0.0, tl["duration"]
    return beat.choose_window(tl, seconds, drop_at=0.25, start=start if start >= 0 else None)


def assign_framing(shots: list[dict], framing: str, seed: int) -> None:
    """'close': every shot a close-up (the edamame look). 'mixed': quiet stretches lean wide and
    medium, loud ones punch in close (with an occasional medium), never the same size twice in
    a row in quiet sections."""
    rng = random.Random(seed * 31 + 7)
    prev = None
    for k, s in enumerate(shots):
        if framing != "mixed":
            s["size"] = "close"
            continue
        if s["loud"]:
            size = "medium" if (rng.random() < 0.25 and prev == "close") else "close"
        else:
            size = rng.choice([x for x in ("wide", "medium") if x != prev] or ["wide"])
        s["size"] = size
        prev = size


def plan_voice(tl: dict, pack: dict, *, fmt: str = "9:16", preset: str = "quality",
               seconds: float = 30.0, start: float = -1.0, seed: int = 1,
               twins: str | None = None, preroll: float = 0.2, tail: float = 0.12,
               out_fps: float = 24.0, framing: str = "close") -> dict:
    ed = pack["edit"]
    twins = twins or ed.get("twins", "loud")
    t0, t1 = choose(tl, seconds, start)
    w = beat.slice_timeline(tl, t0, t1)
    dur = w["duration"]
    bars = [b for b in w["downbeats"] if 0 <= b < dur - 0.05]
    if not bars or bars[0] > 0.05:
        bars = [0.0] + bars
    beats = sorted(set(round(b, 4) for b in w["beats"] if 0 <= b < dur))
    secs = _sections_with_loudness(w, ed.get("loud_threshold", 0.45))
    sec_starts = sorted(s["start"] for s in secs)
    rfps = formats.PRESETS[preset]["fps"]
    rng = random.Random(seed)

    # ---- shots on bar lines; a section start always starts a new shot
    shots, i = [], 0
    while i < len(bars):
        a = bars[i]
        loud = _loud_at(secs, a + 0.01)
        nb = ed["shot_bars_loud"] if loud else ed["shot_bars_quiet"]
        j = min(len(bars), i + nb)
        b = bars[j] if j < len(bars) else dur
        nxt = [s for s in sec_starts if a + 0.3 < s < b - 0.3]
        if nxt:  # stop at the next section start (snapped to its bar)
            k = min(range(len(bars)), key=lambda x: abs(bars[x] - nxt[0]))
            if k > i:
                j, b = k, bars[k]
        shots.append({"t0": round(float(a), 4), "t1": round(float(b), 4), "loud": bool(loud)})
        i = j if j > i else i + 1
    if shots and dur - shots[-1]["t1"] > 0.05:
        shots[-1]["t1"] = round(dur, 4)
    # a final sliver shorter than a beat joins the previous shot
    if len(shots) > 1 and shots[-1]["t1"] - shots[-1]["t0"] < 60.0 / max(1.0, w["tempo"]):
        shots[-2]["t1"] = shots.pop()["t1"]

    assign_framing(shots, framing, seed)

    # ---- performers + renders
    renders, performers = [], 0
    for si, s in enumerate(shots):
        s["index"] = si
        r0 = max(-t0, s["t0"] - preroll)  # can't start before the song
        length = formats.ltx_frames(s["t1"] - r0 + tail, rfps)
        a = {"id": len(renders), "shot": si, "role": "A", "performer": performers,
             "t_start": round(r0, 4), "frames": length, "fps": rfps, "size": s["size"]}
        renders.append(a)
        s["A"] = a["id"]
        performers += 1
        if twins == "all" or (twins == "loud" and s["loud"]):
            b = dict(a, id=len(renders), role="B", performer=performers, twin_of=a["id"])
            renders.append(b)
            s["B"] = b["id"]
            performers += 1

    edl = voice_edl(shots, renders, beats, ed, out_fps, seed)
    return {"version": 1, "mode": "voice", "pack": pack["name"], "format": fmt, "preset": preset,
            "seed": seed, "song_t0": t0, "song_t1": t1, "duration": round(dur, 4),
            "out_fps": out_fps, "render_fps": rfps, "tempo": w["tempo"], "timeline": w,
            "sections": secs, "shots": shots, "renders": renders, "edl": edl,
            "performers": performers, "twins": twins, "framing": framing}


def voice_edl(shots: list[dict], renders: list[dict], beats: list[float], ed: dict, out_fps: float,
              seed: int) -> list[dict]:
    """The cut. Every shot plays its lead performer (A). Loud shots get the edamame flicker:
    on most beats (and many off-beats) a 3-5 frame insert of another face, then a 2-3 frame
    ghost back to A. Inserts alternate between the shot's pose-matched twin (B: same pose,
    same words) and other performers' clips (sampled mid-clip; at 3-5 frames lip sync is
    invisible, the face change is what reads). Some loud shots hand the second half to B.

    Older packs with only `insert_every_beats` keep the original sparse pattern."""
    rng = random.Random(seed * 7919 + 3)
    edl = []
    dense = "insert_on_beat" in ed
    fr = ed.get("insert_frames", 4)
    ins_lo, ins_hi = (fr, fr) if isinstance(fr, (int, float)) else (int(fr[0]), int(fr[1]))
    ghost_n = int(ed.get("ghost_frames", 3))
    every = max(1, int(ed.get("insert_every_beats", 2)))
    by_id = {r["id"]: r for r in renders}
    a_pool = [r for r in renders if r["role"] == "A"]
    beat_len = (beats[1] - beats[0]) if len(beats) > 1 else 0.5
    for s in shots:
        f0, f1 = int(round(s["t0"] * out_fps)), int(round(s["t1"] * out_fps))
        events = [{"f0": f0, "f1": f1, "render": s["A"], "kind": "main"}]
        sb = [b for b in beats if s["t0"] + 0.05 < b < s["t1"] - 0.15]
        has_b = "B" in s
        pp = has_b and rng.random() < ed.get("pingpong", 0.0) and len(sb) >= 4
        half = sb[len(sb) // 2] if pp else None
        if dense and s["loud"]:
            slots = []
            for b in sb:
                if rng.random() < ed["insert_on_beat"]:
                    slots.append(b)
                ob = b + beat_len / 2
                if ob < s["t1"] - 0.15 and rng.random() < ed.get("insert_off_beat", 0.0):
                    slots.append(ob)
            # other faces of the same framing read as match cuts
            same = [r for r in a_pool if r["shot"] != s["index"] and r.get("size") == s.get("size")] or \
                [r for r in a_pool if r["shot"] != s["index"]]
            last_end = f0 + 2
            for t in sorted(slots):
                fb = int(round(t * out_fps))
                n = rng.randint(ins_lo, ins_hi)
                if fb < last_end or fb + n > f1 - 1:
                    continue
                lead = s["B"] if (half is not None and t >= half) else s["A"]
                use_twin = has_b and lead != s.get("B") and (not same or rng.random() < ed.get("insert_twin_share", 0.5))
                if use_twin:
                    src, dt = s["B"], 0.0
                elif same:
                    o = rng.choice(same)
                    o_t0 = o["t_start"] + 0.2
                    o_t1 = o["t_start"] + o["frames"] / o["fps"] - 0.25
                    dt = round(o_t0 + (o_t1 - o_t0) * rng.uniform(0.2, 0.8) - fb / out_fps, 4)
                    src = o["id"]
                else:
                    continue
                events.append({"f0": fb, "f1": fb + n, "render": src, "kind": "insert", "dt": dt})
                g1 = min(f1, fb + n + ghost_n)
                events.append({"f0": fb + n, "f1": g1, "render": lead, "kind": "ghost", "from": src, "from_dt": dt})
                last_end = g1
        elif has_b:
            hits = [b for b in sb if (beats.index(b) % every) == (every - 1)]
            for b in hits:
                fb = int(round(b * out_fps))
                if half is not None and b >= half:
                    continue
                events.append({"f0": fb, "f1": min(f1, fb + int(ins_hi)), "render": s["B"], "kind": "insert"})
                events.append({"f0": min(f1, fb + int(ins_hi)), "f1": min(f1, fb + int(ins_hi) + ghost_n),
                               "render": s["A"], "kind": "ghost", "from": s["B"]})
        if half is not None:
            fh = int(round(half * out_fps))
            events.append({"f0": fh, "f1": f1, "render": s["B"], "kind": "switch"})
        edl.extend(events)
    return edl


def plan_motion(tl: dict, pack: dict, *, clip_frames: int, clip_fps: float, fmt: str = "9:16",
                preset: str = "quality", seconds: float = 16.0, start: float = -1.0,
                seed: int = 1, switch_every: str | None = None, clip_offset_s: float = 0.0,
                pad: float = 0.12, out_fps: float = 24.0) -> dict:
    """A new character every `switch_every`; each segment is its own pose-guided render that
    starts on the cut, so every cut is pose-aligned (a match cut)."""
    ed = pack["edit"]
    switch_every = switch_every or ed.get("switch_every", "2 beats")
    n_cast = len(pack["casting"]["cast"])
    looks = [f"c{i:02d}" for i in range(n_cast)]
    rng = random.Random(seed)
    rng.shuffle(looks)
    clip_offset = int(round(clip_offset_s * clip_fps))
    p = dsplan.plan_dropshift(tl, clip_frames=clip_frames, fps=clip_fps, looks=looks,
                              max_seconds=seconds, song_start=start, mode="every-n-bars",
                              switch_every=switch_every, render_mode="segments",
                              clip_offset=clip_offset)
    t0, t1 = p["window"]
    w = p["timeline"]
    secs = _sections_with_loudness(w, 0.45)
    rfps = formats.PRESETS[preset]["fps"]
    renders = []
    for r in p["renders"]:
        ts = r["f0"] / clip_fps
        dur = (r["f1"] - r["f0"]) / clip_fps
        renders.append({"id": r["id"], "role": "M", "performer": int(r["look"][1:]),
                        "t_start": round(ts, 4), "src_start": round(r["src_f0"] / clip_fps, 4),
                        "frames": formats.ltx_frames(dur + pad, rfps), "fps": rfps})
    edl, shots = [], []
    for s in p["segments"]:
        f0 = int(round(s["f0"] / clip_fps * out_fps))
        f1 = int(round(s["f1"] / clip_fps * out_fps))
        if f1 <= f0:
            continue
        edl.append({"f0": f0, "f1": f1, "render": s.get("render"), "kind": "main"})
        shots.append({"t0": round(f0 / out_fps, 4), "t1": round(f1 / out_fps, 4),
                      "loud": _loud_at(secs, f0 / out_fps + 0.01), "A": s.get("render")})
    return {"version": 1, "mode": "motion", "pack": pack["name"], "format": fmt, "preset": preset,
            "seed": seed, "song_t0": t0, "song_t1": t1, "duration": round(t1 - t0, 4),
            "out_fps": out_fps, "render_fps": rfps, "clip_fps": clip_fps, "tempo": w["tempo"],
            "timeline": w, "sections": secs, "shots": shots, "renders": renders, "edl": edl,
            "switch_every": switch_every, "performers": n_cast}
