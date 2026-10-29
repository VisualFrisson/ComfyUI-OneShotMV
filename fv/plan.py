"""OneShot MV planners: song timeline + style pack -> renders + edit decision list.

Voice mode
  * The song is cut into shots on bar lines: `shot_bars_loud` bars in loud sections,
    `shot_bars_quiet` in quiet ones (the reference AI edit: 2 bars loud, 1 bar quiet).
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
import os
import random

import numpy as np

from . import beat, dsplan, formats


def _sections_with_loudness(w: dict, thr: float) -> list[dict]:
    secs = w.get("sections") or [{"start": 0.0, "end": w["duration"], "energy": 1.0}]
    out = []
    for s in secs:
        e = float(s.get("energy", 1.0))
        out.append({"start": float(s["start"]), "end": float(s["end"]), "energy": e,
                    "loud": bool(e >= thr or s.get("is_drop")), "label": s.get("label", ""),
                    "is_drop": bool(s.get("is_drop"))})
    return out


def _loud_at(secs: list[dict], t: float) -> bool:
    for s in secs:
        if s["start"] - 1e-3 <= t < s["end"]:
            return s["loud"]
    return secs[-1]["loud"] if secs else True


def choose(tl: dict, seconds: float, start: float) -> tuple[float, float]:
    if seconds <= 0:
        return 0.0, tl["duration"]
    return beat.choose_window(tl, seconds, drop_at=0.25, start=start if start >= 0 else None, phrase_bars=4)


def assign_framing(shots: list[dict], framing: str, seed: int, drop_size: str = "close",
                   quiet_weights: dict | None = None) -> None:
    """'close': every shot a close-up (the reference AI edit's look). 'mixed': quiet stretches lean wide and
    medium, loud ones punch in close (with an occasional medium), never the same size twice in
    a row in quiet sections. With phases (the hold grammar): quiet shots move between wide,
    medium and close (`quiet_weights`, a pack's edit `quiet_sizes`); builds tighten (medium,
    close); drops are close-ups (or the pack's `drop_size`)."""
    rng = random.Random(seed * 31 + 7)
    qw = {"wide": 0.4, "medium": 0.4, "close": 0.2, **(quiet_weights or {})}
    prev = None
    for k, s in enumerate(shots):
        if framing != "mixed":
            s["size"] = "close"
            continue
        ph = s.get("phase")
        if ph == "quiet":
            opts = [x for x in ("wide", "medium", "close") if x != prev]
            size = rng.choices(opts, weights=[qw[x] for x in opts])[0]
        elif ph == "build":
            size = "close" if prev == "medium" else "medium"
        elif ph == "drop":
            size = drop_size
        elif s["loud"]:
            size = "medium" if (ph is None and rng.random() < 0.25 and prev == "close") else "close"
        else:
            size = rng.choice([x for x in ("wide", "medium") if x != prev] or ["wide"])
        s["size"] = size
        prev = size


INTENSE = ("drop", "chorus")


def _intense(s: dict) -> bool:
    """The song's intense parts: the drop and the choruses (by the section analysis). Loudness
    alone can't tell them apart in songs that stay loud throughout (rap verses are as loud as
    the hook); without labels, the loud sections count."""
    if s.get("label"):
        # a final chorus is often labelled "outro" (a test track from 3:10, st2d: the last chorus,
        # 4 dB under the loudest section, planned as all holds): a loud outro counts too
        return s.get("is_drop") or s["label"] in INTENSE or \
            (s["label"] == "outro" and s.get("loud") and s.get("energy", 0.0) >= 0.6)
    return s["loud"]


def _phases(secs: list[dict], bar_len: float, build_bars: float, medium: tuple = ()):
    """quiet / build / drop at a time: the intense sections (_intense) are drops; the last
    `build_bars` bars before one are the build; the rest holds. medium: section labels that count
    as build too (the energy grammar: verses and builds are the song's medium energy, intros,
    breaks and quiet outros its low energy; beat.label_sections labels by energy tertile)."""
    hot = [s for s in secs if _intense(s)]
    mid = [s for s in secs if medium and not _intense(s) and s.get("label") in medium]
    # a window that opens inside a chorus (a song loud from end to end: sx01, a test track, was
    # one 32 s chorus, so every shot was a close-up on the beat and no cutaway could go in) still
    # gets its build: the first `build_bars` bars, at most a third of the window
    total = max((s["end"] for s in secs), default=0.0)
    lead = min(build_bars * bar_len, total / 3) if hot and hot[0]["start"] <= 0.05 else 0.0

    def phase(t: float) -> str:
        if t + 0.01 < lead:
            return "build"
        if any(s["start"] - 1e-3 <= t + 0.01 < s["end"] for s in hot):
            return "drop"
        nxt = [s["start"] for s in hot if s["start"] > t + 0.01]
        if nxt and nxt[0] - t <= build_bars * bar_len + 0.05:
            return "build"
        if any(s["start"] - 1e-3 <= t + 0.01 < s["end"] for s in mid):
            return "build"
        return "quiet"
    return phase


# How each energy level is cut (the user, Oct 26: a drop-down per level; "to the music, intense" is
# the beat cut of the first tests, too fast and chaotic for some songs, so the default for high
# energy is the subtle one, a little faster than a bar)
CUT_CHOICES = ["every 2 bars", "every bar", "to the music (subtle)", "to the music (intense)"]
CUT_DEFAULTS = {"low": "every 2 bars", "medium": "every bar", "high": "to the music (subtle)"}
LEVEL_OF = {"quiet": "low", "build": "medium", "drop": "high"}


def cut_modes(ed: dict) -> dict:
    """{'low': mode, 'medium': mode, 'high': mode} from the pack's edit (cut_low, cut_medium, cut_high)."""
    out = {}
    for lv, d in CUT_DEFAULTS.items():
        m = str(ed.get(f"cut_{lv}") or d).strip().lower()
        out[lv] = m if m in CUT_CHOICES else d
    return out


def subtle_beats(bar_len: float, beat_len: float) -> int:
    """'To the music, subtle': two or four beats, whichever lands nearer 1.2 s, and never faster
    than twice the intense cut (a rap test track, tracked at double time, 180 BPM: half a bar was 0.67 s,
    the same pace as intense; now a bar of 1.33 s, against 0.67 s for intense)."""
    k = min((2, 4), key=lambda b: abs(b * beat_len - 1.2))
    return max(k, 2 * intense_beats(beat_len))


def intense_beats(beat_len: float) -> int:
    """'To the music, intense': one or two beats, whichever lands nearer 0.6 s (one beat at 107
    BPM, 0.56 s, the pace of the first new-system tests)."""
    return min((1, 2), key=lambda b: abs(b * beat_len - 0.6))


def drop_beats(tempo: float, ed: dict) -> int:
    """Beats per shot on the drop (the energy grammar's fast cuts on the rhythm): the whole number
    of beats closest to `drop_cut_seconds` (0.75 s: one beat at 100 BPM, two at 128)."""
    target = float(ed.get("drop_cut_seconds", 0.75))
    beat_len = 60.0 / max(40.0, float(tempo or 120.0))
    return max(1, int(round(target / beat_len)))


def _split_drop(shots: list[dict], beats: list[float], k, min_len) -> list[dict]:
    """Each drop shot (or each shot of a phase in k, when k is {phase: beats}) cut every k beats:
    one new person per cut. A last piece shorter than min_len joins the one before it."""
    out = []
    ks = k if isinstance(k, dict) else {"drop": k}
    mins = min_len if isinstance(min_len, dict) else {"drop": min_len}
    for s in shots:
        if s.get("phase") not in ks:
            out.append(s)
            continue
        k, min_len = ks[s["phase"]], mins[s["phase"]]
        inner = [b for b in beats if s["t0"] + 0.1 < b < s["t1"] - 0.1]
        cuts = [s["t0"]] + inner[k - 1::k] + [s["t1"]]
        pieces = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            if pieces and b - a < min_len:
                pieces[-1]["t1"] = round(float(b), 4)
                continue
            pieces.append(dict(s, t0=round(float(a), 4), t1=round(float(b), 4)))
        if len(pieces) > 1 and pieces[-1]["t1"] - pieces[-1]["t0"] < min_len:
            last = pieces.pop()
            pieces[-1]["t1"] = last["t1"]
        out.extend(pieces)
    return out


def _hold_shots(bars: list[float], dur: float, secs: list[dict], ed: dict, bar_len: float,
                beats: list[float] | None = None, tempo: float = 120.0) -> list[dict]:
    """The hold grammar's shots: long holds in quiet stretches, one bar each in the build,
    one-bar groups in the drop (cut on the beat inside each group, see _hold_renders).
    The energy grammar (pack edit "cuts": "energy", the user's rule of Oct 26): low energy
    (intro, breaks, quiet outro) a cut every `shot_bars_quiet` (2) bars, medium (verses, build)
    every bar, high (choruses, drops) a cut every drop_beats() beats, each a new person."""
    energy = ed.get("cuts") == "energy"
    medium = tuple(ed.get("medium_labels", ("verse", "build"))) if energy else ()
    phase = _phases(secs, bar_len, ed.get("build_bars", 4), medium)
    nbars = {"quiet": int(ed.get("shot_bars_quiet", 2)), "build": int(ed.get("shot_bars_build", 1)),
             "drop": int(ed.get("drop_group_bars", 1))}
    modes = cut_modes(ed) if energy else {}
    if energy:  # the level's cut: bars, or one-bar groups cut to the music below
        nbars = {ph: 2 if modes[LEVEL_OF[ph]] == "every 2 bars" else 1 for ph in nbars}
        if bar_len < 1.4:  # a double-time grid (a rap test track at 180 BPM: 1.33 s bars): bars in pairs, or
            # "every bar" is the same pace as the subtle cut on the drop and the drop doesn't lift
            nbars = {ph: 2 * n for ph, n in nbars.items()}
    sec_starts = sorted(s["start"] for s in secs)
    shots, i = [], 0
    while i < len(bars):
        a = bars[i]
        ph = phase(a)
        j = i + 1
        while j < len(bars) and j - i < nbars[ph] and phase(bars[j]) == ph and \
                not any(abs(bars[j] - s) < 0.3 for s in sec_starts):
            j += 1
        b = bars[j] if j < len(bars) else dur
        shots.append({"t0": round(float(a), 4), "t1": round(float(b), 4), "loud": ph == "drop", "phase": ph})
        i = j
    if shots and dur - shots[-1]["t1"] > 0.05:
        shots[-1]["t1"] = round(dur, 4)
    if energy and beats:
        beat_len = 60.0 / max(40.0, tempo)
        ks = {}
        for ph, lv in LEVEL_OF.items():
            if modes[lv] == "to the music (intense)":
                ks[ph] = intense_beats(beat_len)
            elif modes[lv] == "to the music (subtle)":
                ks[ph] = subtle_beats(bar_len, beat_len)
        if ks:
            shots = _split_drop(shots, beats, ks, {ph: 0.6 * k * beat_len for ph, k in ks.items()})
        for s in shots:
            s["energy"] = {"quiet": "low", "build": "medium", "drop": "high"}[s["phase"]]
            s["burst"] = s["phase"] == "drop"
        return shots
    # long drops: beat cuts for the first `drop_burst_bars` bars, then every other bar, so a
    # 30-bar chorus doesn't strobe from end to end
    burst = int(ed.get("drop_burst_bars", 8))
    run = 0
    for s in shots:
        if s["phase"] != "drop":
            run = 0
            continue
        run += 1
        s["burst"] = run <= burst or (run - burst) % 2 == 0
    return shots


def plan_voice(tl: dict, pack: dict, *, fmt: str = "16:9", preset: str = "quality",
               seconds: float = 30.0, start: float = -1.0, seed: int = 1,
               twins: str | None = None, preroll: float | None = None, tail: float = 0.12,
               out_fps: float | None = None, framing: str = "mixed") -> dict:
    ed = pack["edit"]
    twins = twins or ed.get("twins", "loud")
    # lead-in before each shot: room for the per-take lip-sync delay (up to ~0.4 s) without
    # holding a clip's first frame at the cut
    preroll = preroll if preroll is not None else (0.4 if ed.get("grammar") == "hold" else 0.2)
    t0, t1 = choose(tl, seconds, start)
    w = beat.slice_timeline(tl, t0, t1)
    dur = w["duration"]
    bars = [b for b in w["downbeats"] if 0 <= b < dur - 0.05]
    if not bars or bars[0] > 0.05:
        bars = [0.0] + bars
    # beat tracking often stops in a fade-out: carry the bar grid on to the end so the last
    # shot isn't one long take
    bar_len = float(np.median(np.diff(bars))) if len(bars) > 2 else 4 * 60.0 / max(1.0, w["tempo"])
    while bars[-1] + 1.5 * bar_len < dur:
        bars.append(round(bars[-1] + bar_len, 4))
    beats = sorted(set(round(b, 4) for b in w["beats"] if 0 <= b < dur))
    secs = _sections_with_loudness(w, ed.get("loud_threshold", 0.6))
    sec_starts = sorted(s["start"] for s in secs)
    rfps = formats.PRESETS[preset]["fps"]
    # the cut runs at an exact divisor of the render rate: 50 fps clips cut at 25 fps take every
    # other frame; at 24 fps every twelfth step skips a frame, which reads as stutter on the mouth
    out_fps = float(out_fps or (rfps / 2.0 if rfps >= 48 else rfps))
    rng = random.Random(seed)
    if ed.get("grammar") == "hold":
        shots = _hold_shots(bars, dur, secs, ed, bar_len, beats=beats, tempo=w["tempo"])
        if len(shots) > 1 and shots[-1]["t1"] - shots[-1]["t0"] < 60.0 / max(1.0, w["tempo"]):
            shots[-1]["t1"] = shots.pop()["t1"]
        assign_framing(shots, framing, seed, drop_size=ed.get("drop_size", "close"),
                       quiet_weights=ed.get("quiet_sizes"))
        renders, performers = _hold_renders(shots, beats, ed, rfps, t0, preroll, tail)
        edl = voice_edl(shots, renders, beats, ed, out_fps, seed)
        return {"version": 2, "mode": "voice", "pack": pack["name"], "format": fmt, "preset": preset,
                "seed": seed, "song_t0": t0, "song_t1": t1, "duration": round(dur, 4),
                "out_fps": out_fps, "render_fps": rfps, "tempo": w["tempo"], "timeline": w,
                "sections": secs, "shots": shots, "renders": renders, "edl": edl,
                "performers": performers, "twins": "drop", "framing": framing, "grammar": "hold"}

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
        shots[-1]["t1"] = shots.pop()["t1"]

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
              seed: int, faces: dict | None = None, delay: float = 0.0) -> list[dict]:
    """The cut. Every shot plays its lead performer (A). Loud shots get the reference edit's flicker:
    on most beats (and many off-beats) a 3-5 frame insert of another face, then a 2-3 frame
    ghost back to A. Inserts alternate between the shot's pose-matched twin (B: same pose,
    same words) and other performers' clips (sampled mid-clip; at 3-5 frames lip sync is
    invisible, the face change is what reads). Some loud shots hand the second half to B.

    MATCH CUTS (when face tracks are given): every insert is chosen so its face sits where the
    lead's face is (centre and size), and the assembler nudges it into place (`align`). An insert
    with no good match is dropped, so fast passages stay readable: fewer flashes, all of them
    match cuts. `delay` is the clip-to-song sync offset the assembler applies.

    Older packs with only `insert_every_beats` keep the original sparse pattern.
    Packs with "grammar": "hold" use _hold_edl instead (no flashes back to a shot)."""
    import importlib
    from . import faces as fz
    fz = importlib.reload(fz)
    if ed.get("grammar") == "hold":
        return _hold_edl(fz, shots, renders, ed, out_fps, faces, delay)
    rng = random.Random(seed * 7919 + 3)
    edl = []
    dense = "insert_on_beat" in ed
    fr = ed.get("insert_frames", 4)
    ins_lo, ins_hi = (fr, fr) if isinstance(fr, (int, float)) else (int(fr[0]), int(fr[1]))
    ghost_n = int(ed.get("ghost_frames", 3))
    every = max(1, int(ed.get("insert_every_beats", 2)))
    by_id = {r["id"]: r for r in renders}
    a_pool = [r for r in renders if r["role"] == "A" and not r.get("unusable")]
    beat_len = (beats[1] - beats[0]) if len(beats) > 1 else 0.5
    for s in shots:
        f0, f1 = int(round(s["t0"] * out_fps)), int(round(s["t1"] * out_fps))
        events = [{"f0": f0, "f1": f1, "render": s["A"], "kind": "main"}]
        sb = [b for b in beats if s["t0"] + 0.05 < b < s["t1"] - 0.15]
        has_b = "B" in s and not by_id.get(s["B"], {}).get("unusable")
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
                tf = fb / out_fps
                al = None
                if faces is not None:
                    pick = _match_insert(fz, faces, by_id, lead, tf, s, same, has_b, delay, ed, rng)
                    if pick is None:
                        continue
                    src, dt, al = pick
                else:
                    use_twin = has_b and lead != s.get("B") and (not same or rng.random() < ed.get("insert_twin_share", 0.5))
                    if use_twin:
                        src, dt = s["B"], 0.0
                    elif same:
                        o = rng.choice(same)
                        o_t0 = o["t_start"] + 0.2
                        o_t1 = o["t_start"] + o["frames"] / o["fps"] - 0.25
                        dt = round(o_t0 + (o_t1 - o_t0) * rng.uniform(0.2, 0.8) - tf, 4)
                        src = o["id"]
                    else:
                        continue
                ins = {"f0": fb, "f1": fb + n, "render": src, "kind": "insert", "dt": dt}
                gh = {"f0": fb + n, "f1": min(f1, fb + n + ghost_n), "render": lead, "kind": "ghost",
                      "from": src, "from_dt": dt}
                if al is not None:
                    ins["align"] = gh["from_align"] = al
                events += [ins, gh]
                last_end = gh["f1"]
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


def _hold_renders(shots: list[dict], beats: list[float], ed: dict, rfps: float, t0: float,
                  preroll: float, tail: float) -> tuple[list[dict], int]:
    """One lead (A) per shot, lip-synced to its stretch of the vocal. Each drop group adds a
    chain of pose-matched looks (B): the lead's clip re-rendered from the lead's own pose and
    words as someone (or somewhere) new, each covering just its beat. Every cut inside the group
    then lands on the same face position, pose and syllable. `drop_looks` looks per group
    (the lead included), one per `drop_cut_beats` beats."""
    renders, performers = [], 0
    looks = max(1, int(ed.get("drop_looks", 4)))
    every = max(1, int(ed.get("drop_cut_beats", 1)))
    tpre = float(ed.get("twin_preroll", 0.2))
    for si, s in enumerate(shots):
        s["index"] = si
        r0 = max(-t0, s["t0"] - preroll)  # can't start before the song
        a = {"id": len(renders), "shot": si, "role": "A", "performer": performers,
             "t_start": round(r0, 4), "frames": formats.ltx_frames(s["t1"] - r0 + tail, rfps), "fps": rfps,
             "size": s["size"]}
        renders.append(a)
        s["A"] = a["id"]
        performers += 1
        if s.get("phase") != "drop" or not s.get("burst", True) or looks < 2:
            continue
        a_end = r0 + a["frames"] / rfps
        sb = [b for b in beats if s["t0"] + 0.1 < b < s["t1"] - 0.1]
        cuts = sb[every - 1::every][:looks - 1]
        s["chain"] = []
        for k, c in enumerate(cuts):
            end = cuts[k + 1] if k + 1 < len(cuts) else s["t1"]
            ts = max(r0, c - tpre)
            n = formats.ltx_frames(end - ts + tail, rfps)
            while n > 9 and ts + n / rfps > a_end + 1e-6:  # the pose guide is the lead's clip
                n -= 8
            b = {"id": len(renders), "shot": si, "role": "B", "performer": performers, "twin_of": a["id"],
                 "t_start": round(ts, 4), "frames": n, "fps": rfps, "size": s["size"],
                 "guide_start": round(ts - r0, 4), "cut_at": round(c, 4)}
            renders.append(b)
            s["chain"].append(b["id"])
            performers += 1
    return renders, performers


def _hold_edl(fz, shots: list[dict], renders: list[dict], ed: dict, out_fps: float,
              faces: dict | None, delay: float) -> list[dict]:
    """The hold grammar's cut. Each shot plays its lead; a drop group steps through its chain of
    pose-matched looks on the beat and never returns to an earlier one. With face tracks, a
    look is used only if its face is found and lands on the lead's (then nudged exactly into
    place); one that QC could not save, or that can't be checked, is skipped and the look
    before it plays on, so every fast cut is a match cut. The cut into a new drop group (a new
    lead on the bar line) is nudged too, so its face starts where the last one was."""
    by_id = {r["id"]: r for r in renders}
    thr = ed.get("match_threshold", 0.16) * ed.get("twin_tolerance", 2.0)
    edl = []
    prev_face = None  # (face box on screen at the end of the previous shot, previous phase)

    def face(rid, t):
        r = by_id[rid]
        return fz.face_at(faces.get(rid), t - r["t_start"] - r.get("sync_delay", delay)) if faces else None

    for s in shots:
        f0, f1 = int(round(s["t0"] * out_fps)), int(round(s["t1"] * out_fps))
        br = by_id.get(s.get("broll"))
        if br is not None and br.get("path") and not br.get("unusable") and os.path.exists(br["path"]):
            # a b-roll cutaway (fv/broll.py) replaces the shot's lead
            edl.append({"f0": f0, "f1": f1, "render": br["id"], "kind": "main", "broll": True})
            prev_face = None
            continue
        a = by_id[s["A"]]
        a_al = None
        if faces is not None and s.get("phase") == "drop" and prev_face is not None and prev_face[1] == "drop":
            lf0 = face(s["A"], s["t0"] + 0.02)
            if lf0 is not None and fz.match_cost(prev_face[0], lf0) < 3 * thr:
                a_al = fz.align(prev_face[0], lf0, max_scale=ed.get("match_max_scale", 1.12), min_scale=0.92,
                                max_shift=ed.get("match_max_shift", 0.08))
        segs = [(s["A"], f0, a_al)]
        cands = []
        for rid in s.get("chain", []):
            r = by_id.get(rid)
            if r is None or r.get("unusable"):
                continue
            fc = int(round(r["cut_at"] * out_fps))
            if not (segs[-1][1] + 2 <= fc <= f1 - 2):
                continue
            al = None
            if faces is not None:
                tf = r["cut_at"]
                lf_raw = face(s["A"], tf)
                lf = fz.apply_align(a_al, lf_raw)  # where the lead's face is on screen
                bf = face(rid, tf)
                # a look is rendered (and face-matched) against the lead's own frame: judge it there,
                # then move it to wherever the group's alignment put the lead on screen
                if lf is None or bf is None or fz.match_cost(lf_raw, bf) > thr:
                    continue  # can't be shown to line up: no fast cut here
                al = fz.align(lf, bf)
            cands.append((rid, fc, al))
        # each look's clip only runs to the next planned cut: a look may stay up only until the
        # next look that is actually shown, or it would freeze on its last frame (n31 at 0:40).
        # Pick, from the end of the bar back, the looks whose clips reach the next one shown.
        need, keep = f1, []
        for rid, fc, al in reversed(cands):
            r = by_id[rid]
            end = r["t_start"] + r.get("sync_delay", delay) + (r["frames"] - 1) / r["fps"]
            if end * out_fps >= need - 0.5:
                keep.insert(0, (rid, fc, al))
                need = fc
        segs += keep
        for k, (rid, fa, al) in enumerate(segs):
            fb = segs[k + 1][1] if k + 1 < len(segs) else f1
            ev = {"f0": fa, "f1": fb, "render": rid, "kind": "main" if k == 0 else "switch"}
            if al is not None:
                ev["align"] = al
            edl.append(ev)
        if faces is not None:
            last_rid, _, last_al = segs[-1]
            prev_face = (fz.apply_align(last_al, face(last_rid, s["t1"] - 0.02)), s.get("phase"))
            if prev_face[0] is None:
                prev_face = None
    return edl


def _match_insert(fz, faces, by_id, lead, tf, shot, same, has_b, delay, ed, rng):
    """(source render, dt, align) for an insert at window time tf, or None if nothing lines up."""
    lr = by_id[lead]
    lf = fz.face_at(faces.get(lead), tf - lr["t_start"] - delay)
    if lf is None:
        return None
    thr = ed.get("match_threshold", 0.16)
    cands = []
    twin = shot.get("B") if has_b and lead != shot.get("B") else None
    if twin is not None:
        tr = by_id[twin]
        bf = fz.face_at(faces.get(twin), tf - tr["t_start"] - delay)
        if bf is not None:  # the twin shares the lead's pose by construction: a looser bar
            cands.append((fz.match_cost(lf, bf) / ed.get("twin_tolerance", 2.0), twin, 0.0, bf, True))
    for o in same:
        tr = faces.get(o["id"])
        if not tr:
            continue
        lo = o["t_start"] + 0.25
        hi = o["t_start"] + o["frames"] / o["fps"] - 0.3
        best = None
        x = lo
        while x <= hi:
            bf = fz.face_at(tr, x - o["t_start"] - delay)
            c = fz.match_cost(lf, bf)
            if best is None or c < best[0]:
                best = (c, o["id"], round(x - tf, 4), bf, False)
            x += 0.08
        if best is not None:
            cands.append(best)
    good = [c for c in cands if c[0] <= thr]
    if not good:
        return None
    twins = [c for c in good if c[4]]
    others = sorted([c for c in good if not c[4]], key=lambda c: c[0])[:3]
    if twins and (not others or rng.random() < ed.get("insert_twin_share", 0.5)):
        c = twins[0]
    else:
        c = rng.choice(others) if others else twins[0]
    return c[1], c[2], fz.align(lf, c[3])


def plan_motion(tl: dict, pack: dict, *, clip_frames: int, clip_fps: float, fmt: str = "16:9",
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
    secs = _sections_with_loudness(w, ed.get("loud_threshold", 0.6))
    rfps = formats.PRESETS[preset]["fps"]
    renders = []
    for r in p["renders"]:
        ts = r["f0"] / clip_fps
        dur = (r["f1"] - r["f0"]) / clip_fps
        renders.append({"id": r["id"], "role": "M", "performer": int(r["look"][1:]),
                        "t_start": round(ts, 4), "src_start": round(r["src_f0"] / clip_fps, 4),
                        "frames": formats.ltx_frames(dur + pad, rfps), "fps": rfps})
    shots = []
    phase = dsplan.section_phase(w, ed.get("build_bars", 4))
    for s in p["segments"]:
        f0 = int(round(s["f0"] / clip_fps * out_fps))
        f1 = int(round(s["f1"] / clip_fps * out_fps))
        if f1 <= f0:
            continue
        shots.append({"index": len(shots), "t0": round(f0 / out_fps, 4), "t1": round(f1 / out_fps, 4),
                      "loud": _loud_at(secs, f0 / out_fps + 0.01), "A": s.get("render"),
                      "phase": phase(f0 / out_fps + 0.01)})
    # motion twins: the same stretch of the dance rendered as a second character, so the fast
    # flicker inserts are pose-perfect match cuts
    mt = ed.get("motion_twins", "off")
    if mt in ("loud", "all", "drop"):
        by_id = {r["id"]: r for r in renders}
        for sh in shots:
            a = by_id.get(sh["A"])
            if a is None or (mt == "loud" and not sh["loud"]) or (mt == "drop" and sh["phase"] != "drop"):
                continue
            other = (a["performer"] + max(1, n_cast // 2) + sh["index"] % 3) % n_cast
            b = dict(a, id=len(renders), role="MB", performer=other, twin_of=a["id"])
            renders.append(b)
            sh["B"] = b["id"]
    beats = sorted(set(round(b, 4) for b in w["beats"] if 0 <= b < t1 - t0))
    edl = motion_edl(shots, beats, ed, out_fps, seed)
    return {"version": 1, "mode": "motion", "pack": pack["name"], "format": fmt, "preset": preset,
            "seed": seed, "song_t0": t0, "song_t1": t1, "duration": round(t1 - t0, 4),
            "out_fps": out_fps, "render_fps": rfps, "clip_fps": clip_fps, "tempo": w["tempo"],
            "timeline": w, "sections": secs, "shots": shots, "renders": renders, "edl": edl,
            "switch_every": switch_every, "performers": n_cast}


def motion_edl(shots: list[dict], beats: list[float], ed: dict, out_fps: float, seed: int,
               renders: list[dict] | None = None) -> list[dict]:
    """Motion cut: each segment plays its character; segments with a twin flicker to it on
    beats and off-beats (3-4 frames, then a short ghost back). Same dance, same moment, so
    every flicker is a match cut."""
    rng = random.Random(seed * 104729 + 11)
    bad = {r["id"] for r in renders or [] if r.get("unusable")}  # twins QC could not save
    fr = ed.get("insert_frames", [3, 4])
    lo, hi = (fr, fr) if isinstance(fr, (int, float)) else (int(fr[0]), int(fr[1]))
    ghost_n = int(ed.get("ghost_frames", 2))
    beat_len = (beats[1] - beats[0]) if len(beats) > 1 else 0.5
    edl = []
    for s in shots:
        f0, f1 = int(round(s["t0"] * out_fps)), int(round(s["t1"] * out_fps))
        edl.append({"f0": f0, "f1": f1, "render": s["A"], "kind": "main"})
        if "B" not in s or s["B"] in bad:
            continue
        slots = []
        for b in beats:
            if s["t0"] + 0.1 < b < s["t1"] - 0.12 and rng.random() < ed.get("insert_on_beat", 0.6):
                slots.append(b)
            ob = b + beat_len / 2
            if s["t0"] + 0.1 < ob < s["t1"] - 0.12 and rng.random() < ed.get("insert_off_beat", 0.3):
                slots.append(ob)
        last = f0 + 2
        for t in sorted(slots):
            fb = int(round(t * out_fps))
            n = rng.randint(lo, hi)
            if fb < last or fb + n > f1 - 1:
                continue
            edl.append({"f0": fb, "f1": fb + n, "render": s["B"], "kind": "insert"})
            g1 = min(f1, fb + n + ghost_n)
            edl.append({"f0": fb + n, "f1": g1, "render": s["A"], "kind": "ghost", "from": s["B"]})
            last = g1
    return edl
