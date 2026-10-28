"""Timeline export: CMX3600 EDL of the cut (one event per edit decision), so the edit can be
re-opened in Resolve / Premiere with the rendered clips as sources."""
from __future__ import annotations

import os


def _tc(frames: int, fps: float) -> str:
    fr = int(round(fps))
    h, rem = divmod(int(frames), 3600 * fr)
    m, rem = divmod(rem, 60 * fr)
    s, f = divmod(rem, fr)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def write_edl(plan: dict, path: str) -> str:
    fps = plan["out_fps"]
    renders = {r["id"]: r for r in plan["renders"]}
    events = sorted(plan["edl"], key=lambda e: (e["f0"], -1 if e["kind"] == "main" else 0))
    # flatten overlapping events into a single track
    flat = []
    for e in events:
        if e["f1"] <= e["f0"]:
            continue
        flat = [x for x in flat if not (e["f0"] <= x["f0"] and x["f1"] <= e["f1"])]
        cut = []
        for x in flat:
            if x["f0"] < e["f0"] < x["f1"]:
                cut.append(dict(x, f1=e["f0"]))
                if e["f1"] < x["f1"]:
                    cut.append(dict(x, f0=e["f1"]))
            elif x["f0"] < e["f1"] < x["f1"]:
                cut.append(dict(x, f0=e["f1"]))
            else:
                cut.append(x)
        flat = sorted(cut + [e], key=lambda x: x["f0"])
    lines = [f"TITLE: OneShot MV {plan['pack']} {plan['format']}", "FCM: NON-DROP FRAME", ""]
    for i, e in enumerate(flat, 1):
        r = renders.get(e["render"], {})
        name = os.path.basename(r.get("path", f"render_{e['render']}"))
        src_in = int(round((e["f0"] / fps - r.get("t_start", 0.0)) * fps))
        dur = e["f1"] - e["f0"]
        reel = f"R{int(e['render'] or 0):03d}"
        lines.append(f"{i:03d}  {reel:<8} V     C        {_tc(src_in, fps)} {_tc(src_in + dur, fps)} "
                     f"{_tc(e['f0'], fps)} {_tc(e['f1'], fps)}")
        lines.append(f"* FROM CLIP NAME: {name}")
        lines.append(f"* FV EVENT: {e['kind']}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
