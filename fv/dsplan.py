"""dsplan: turn a song timeline + a clip into a Drop Shift render plan.

The plan cuts the clip into segments on beat or bar lines and gives each segment a look
("real" = untouched footage). Default mode "drop-reveal": real footage until the drop,
then a new look every `switch_every` (a beat, two beats, a bar, ...), cycling through
the pack.

Rendering is planned separately from cutting. Each segment points at a render
("render", "render_offset"):
  - "segments": every styled segment is rendered on its own (fresh keyframe per cut);
  - "layers":   every look is rendered once as a continuous layer over the whole styled
                stretch, and the edit cuts between layers on the beat. Costs the same
                however fast the looks switch, and the cut pattern can be changed
                afterwards without re-rendering (see recut()).
  - "auto":     per look, whichever of the two is cheaper.
"""
from __future__ import annotations

import math

SWITCH_CHOICES = ["1 beat", "2 beats", "1 bar", "2 bars", "4 bars", "accelerate", "by section"]
# "by section": calm in the quiet stretches, a bar a look in the build, fast on the drop
SECTION_STEPS = {"quiet": "2 bars", "build": "1 bar", "drop": "2 beats"}
RENDER_MODES = ["auto", "layers", "segments"]
# Fixed cost of one render (keyframe edit, control maps, model passes, decode) in
# frame-equivalents: a 17-frame LTX render takes roughly a third of a 129-frame one.
RENDER_OVERHEAD_FRAMES = 48


def ltx_len(n: int) -> int:
    """Smallest LTX-valid frame count (8k+1) >= n."""
    return 8 * max(1, math.ceil((n - 1) / 8)) + 1


def _slot_beats(choice: str, bpb: int) -> int:
    fixed = {"1 beat": 1, "2 beats": 2, "1 bar": bpb, "2 bars": 2 * bpb, "4 bars": 4 * bpb}
    if choice in fixed:
        return fixed[choice]
    if choice in ("accelerate", "by section"):
        return 1
    return bpb * max(1, int(str(choice).split()[0]))  # "3 bars" etc.


def _grid(w: dict) -> tuple[list[float], list[float], float]:
    dur = w["duration"]
    bars = [b for b in w["downbeats"] if 0 <= b < dur]
    if not bars or bars[0] > 1e-3:
        bars = [0.0] + bars
    beats = sorted(set([round(b, 4) for b in w["beats"] if 0 <= b < dur] + bars))
    return beats, bars, dur


def section_phase(w: dict, build_bars: float = 4):
    """quiet / build / drop at a window time: the drop and choruses are drops (loud sections
    when the analysis has no labels), the last `build_bars` bars before one are the build."""
    secs = w.get("sections") or []
    hot = [s for s in secs if s.get("is_drop") or s.get("label") in ("drop", "chorus")]
    if not hot and not any(s.get("label") for s in secs):
        hot = [s for s in secs if s.get("energy", 0.0) >= 0.6]
    bar = 4 * 60.0 / max(1.0, float(w.get("tempo") or 120.0))

    def phase(t: float) -> str:
        if any(s["start"] - 1e-3 <= t + 0.01 < s["end"] for s in hot):
            return "drop"
        nxt = [s["start"] for s in hot if s["start"] > t + 0.01]
        if nxt and nxt[0] - t <= build_bars * bar + 0.05:
            return "build"
        return "quiet"
    return phase


def _switch_times(start: float, end: float, beats: list[float], choice: str,
                  bpb: int, phase=None, steps: dict | None = None) -> list[float]:
    """Cut times from `start` to `end`. 'accelerate' ramps from a new look every bar to
    every two beats to every beat; 'by section' takes the step from the song's phase at each
    cut (SECTION_STEPS: long in quiet stretches, fast on the drop)."""
    grid = [b for b in beats if start - 1e-3 <= b < end - 1e-3]
    if not grid or abs(grid[0] - start) > 1e-3:
        grid = [start] + grid
    out, i, span = [], 0, max(1e-6, end - start)
    while i < len(grid):
        out.append(grid[i])
        if choice == "accelerate":
            p = (grid[i] - start) / span
            step = bpb if p < 0.4 else (2 if p < 0.7 else 1)
        elif choice == "by section" and phase is not None:
            ph = phase(grid[i])
            step = _slot_beats((steps or SECTION_STEPS)[ph], bpb)
            # a phase change cuts on its own bar line, not mid-step
            nxt = next((j for j in range(i + 1, min(len(grid), i + step)) if phase(grid[j]) != ph), None)
            if nxt is not None:
                step = nxt - i
        else:
            step = _slot_beats(choice, bpb)
        i += step
    return out


def _styled_start(w: dict, bars: list[float], looks: list[str]) -> float:
    drop_t = (w.get("drop") or {}).get("time")
    start = drop_t if drop_t is not None else (bars[min(len(bars) - 1, 2)]
                                               if len(bars) > 2 else 0.0)
    return start if (start > 0.05 and "real" in looks) else 0.0


def _cut_segments(w: dict, fps: float, n_out: int, looks: list[str], mode: str,
                  switch_every: str, bpb: int, tempo: float) -> list[dict]:
    """The edit: [{f0, f1, look}] covering [0, n_out), cut on the beat grid."""
    beats, bars, dur = _grid(w)
    styled = [x for x in looks if x != "real"] or ["anime"]
    cuts = []
    if mode == "drop-reveal":
        start = _styled_start(w, bars, looks)
        if start > 0:
            cuts.append((0.0, "real"))
        for k, t in enumerate(_switch_times(start, dur, beats, switch_every, bpb)):
            cuts.append((t, styled[k % len(styled)]))
    elif mode == "per-section":
        secs = w.get("sections") or [{"start": 0.0}]
        for i, s in enumerate(secs):
            cuts.append((s["start"], looks[i % len(looks)]))
    else:  # "every-n-bars": cycle all looks, real included, from the start
        for k, t in enumerate(_switch_times(0.0, dur, beats, switch_every, bpb, section_phase(w))):
            cuts.append((t, looks[k % len(looks)]))
    merged = []
    for t, look in sorted(cuts):
        if merged and (merged[-1][1] == look or t - merged[-1][0] < 0.05):
            if t - merged[-1][0] < 0.05:
                merged[-1] = (merged[-1][0], look)
            continue
        merged.append((t, look))
    # a final look that would last less than half its slot is folded into the previous one
    if len(merged) >= 2 and mode != "per-section":
        if dur - merged[-1][0] < 0.5 * _slot_beats(switch_every, bpb) * 60.0 / max(1e-6, tempo):
            merged.pop()
    segs = []
    for i, (t, look) in enumerate(merged):
        te = merged[i + 1][0] if i + 1 < len(merged) else dur
        f0, f1 = int(round(t * fps)), min(n_out, int(round(te * fps)))
        if f1 > f0:
            segs.append({"look": look, "f0": f0, "f1": f1})
    if segs:
        segs[0]["f0"], segs[-1]["f1"] = 0, n_out
    return segs


def _chunks(a: int, b: int, max_gen: int, lines: list[int]) -> list[tuple[int, int]]:
    """Split [a, b) into balanced pieces <= max_gen, snapped to the given cut lines."""
    out = []
    while b - a > max_gen:
        n_pieces = math.ceil((b - a) / max_gen)
        target = a + (b - a) / n_pieces
        cand = [c for c in lines if a + 8 < c <= a + max_gen - 1]
        c = min(cand, key=lambda x: abs(x - target)) if cand else a + max_gen - 1
        out.append((a, c))
        a = c
    out.append((a, b))
    return out


def _plan_renders(segs: list[dict], bar_frames: list[int], max_gen: int, clip_offset: int,
                  render_mode: str, styled_looks: list[str]) -> list[dict]:
    """Decide the renders. Returns renders [{id, look, f0, f1, src_f0, gen_len}]."""
    renders: list[dict] = []

    def add(look, a, b):
        renders.append({"look": look, "f0": a, "f1": b, "src_f0": clip_offset + a,
                        "gen_len": ltx_len(b - a)})

    styled = [s for s in segs if s["look"] != "real"]
    if not styled:
        return renders
    span = (min(s["f0"] for s in styled), max(s["f1"] for s in styled))
    cut_lines = sorted({s["f0"] for s in segs} | {s["f1"] for s in segs} | set(bar_frames))
    if render_mode == "layers":
        # every look over the whole styled stretch, chunked on bar lines only, so the
        # renders do not depend on the cut pattern and any re-cut can reuse them
        for look in styled_looks:
            for p0, p1 in _chunks(span[0], span[1], max_gen, bar_frames):
                add(look, p0, p1)
    else:
        by_look: dict[str, list[dict]] = {}
        for s in styled:
            by_look.setdefault(s["look"], []).append(s)

        def cost(pieces):
            return sum(RENDER_OVERHEAD_FRAMES + ltx_len(p1 - p0) for p0, p1 in pieces)
        for look, ss in by_look.items():
            own = [(p0, p1) for s in ss for (p0, p1) in _chunks(s["f0"], s["f1"], max_gen, cut_lines)]
            lay = _chunks(ss[0]["f0"], ss[-1]["f1"], max_gen, cut_lines)
            use = lay if (render_mode == "auto" and len(ss) > 1 and cost(lay) <= cost(own)) else own
            for p0, p1 in use:
                add(look, p0, p1)
    renders.sort(key=lambda r: (r["f0"], r["look"]))
    for i, r in enumerate(renders):
        r["id"] = i
    return renders


def _attach(segs: list[dict], renders: list[dict], fps: float, clip_offset: int) -> list[dict]:
    """Point every styled segment (split where its renders split) at a render."""
    out = []
    for s in segs:
        if s["look"] == "real":
            out.append(dict(s, render=None, render_offset=0, continues=False))
            continue
        rs = sorted((r for r in renders if r["look"] == s["look"]
                     and r["f0"] < s["f1"] and r["f1"] > s["f0"]), key=lambda r: r["f0"])
        a, j = s["f0"], 0
        for r in rs:
            if r["f1"] <= a or r["f0"] > a:
                continue
            b = min(s["f1"], r["f1"])
            out.append({"look": s["look"], "f0": a, "f1": b, "render": r["id"],
                        "render_offset": a - r["f0"], "continues": j > 0})
            a, j = b, j + 1
            if a >= s["f1"]:
                break
        if a < s["f1"]:  # not covered by any render (should not happen): show the source
            out.append(dict(s, f0=a, look="real", render=None, render_offset=0, continues=j > 0))
    for i, s in enumerate(out):
        s.update(index=i, t0=round(s["f0"] / fps, 4), t1=round(s["f1"] / fps, 4),
                 src_f0=clip_offset + s["f0"])
    return out


def _changes(segs: list[dict]) -> list[float]:
    return [s["t0"] for i, s in enumerate(segs[1:], 1) if s["look"] != segs[i - 1]["look"]]


def plan_dropshift(tl: dict, clip_frames: int, fps: float, looks: list[str], *,
                   max_seconds: float = 20.0, song_start: float = -1.0, drop_at: float = 0.4,
                   mode: str = "drop-reveal", switch_bars: int = 2, max_gen_frames: int = 241,
                   clip_offset: int = 0, switch_every: str | None = None,
                   render_mode: str = "auto") -> dict:
    """tl: full-song timeline from beatcore.analyze. looks: e.g. ["real","anime","clay"].
    switch_every: one of SWITCH_CHOICES (None: every `switch_bars` bars).
    render_mode: "auto", "layers" or "segments" (see module doc).
    Returns a JSON-safe plan: window (song seconds), segments in output frames, renders."""
    from .beat import choose_window, slice_timeline  # local import for both layouts

    avail = (clip_frames - clip_offset) / fps
    length = max(1.0, min(max_seconds, avail, tl["duration"]))
    t0, t1 = choose_window(tl, length, drop_at=drop_at,
                           start=song_start if song_start >= 0 else None)
    w = slice_timeline(tl, t0, t1)
    n_out = int(math.floor((t1 - t0) * fps))
    bpb = int(tl.get("beats_per_bar") or 4)
    tempo = float(tl.get("tempo") or 120.0)
    if switch_every is None:
        switch_every = {1: "1 bar", 2: "2 bars", 4: "4 bars"}.get(int(switch_bars),
                                                                 f"{int(switch_bars)} bars")
    _, bars, _ = _grid(w)
    bar_frames = [int(round(b * fps)) for b in bars]
    segs = _cut_segments(w, fps, n_out, looks, mode, switch_every, bpb, tempo)
    styled_looks = list(dict.fromkeys(x for x in looks if x != "real")) or ["anime"]
    if mode == "per-section":
        styled_looks = [x for x in styled_looks if any(s["look"] == x for s in segs)]
    renders = _plan_renders(segs, bar_frames, max_gen_frames, clip_offset, render_mode,
                            styled_looks)
    segs = _attach(segs, renders, fps, clip_offset)
    return {"version": 2, "fps": fps, "window": [t0, t1], "frames": n_out,
            "clip_offset": clip_offset, "mode": mode, "switch_every": switch_every,
            "render_mode": render_mode, "looks": list(looks), "beats_per_bar": bpb,
            "tempo": tempo, "segments": segs, "renders": renders,
            "look_changes": _changes(segs), "timeline": w}


def recut(plan: dict, switch_every: str, mode: str | None = None) -> dict:
    """Re-cut a "layers" plan on a different rhythm, reusing its renders. Plans rendered
    per segment are returned unchanged (their renders only cover their own cuts)."""
    if switch_every in (None, "", "as planned") or plan.get("render_mode") != "layers":
        return plan
    mode = mode or plan["mode"]
    segs = _cut_segments(plan["timeline"], plan["fps"], plan["frames"], plan["looks"], mode,
                         switch_every, int(plan.get("beats_per_bar", 4)),
                         float(plan.get("tempo", 120.0)))
    segs = _attach(segs, plan["renders"], plan["fps"], int(plan.get("clip_offset", 0)))
    out = dict(plan, segments=segs, switch_every=switch_every, mode=mode)
    out["look_changes"] = _changes(segs)
    return out
