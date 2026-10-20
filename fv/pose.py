"""Lead-dancer pose guides.

DWPose draws every person it finds: backup dancers, people at the back of the room. LTX's
union-control guide turns every skeleton into a person, so a dance clip with a crew behind the
lead dancer puts extra people into every motion shot. `pick_lead` chooses one person per frame
(the tallest, most central person when it first sees them, then whoever stays closest to where
the lead was) so the guide can be redrawn with that person alone.

Works on DWPose's POSE_KEYPOINT output: one OpenPose-style dict per frame,
{"people": [{"pose_keypoints_2d": [x, y, c] * 18, ...}], "canvas_width": W, "canvas_height": H},
with c = 1 for a detected point and 0 for a missing one. x, y may be normalised (0..1) or pixels.
"""
from __future__ import annotations

import math


def _box(person: dict, W: float, H: float):
    """(cx, cy, h, n) of a person's body keypoints, normalised to the canvas, or None."""
    k = person.get("pose_keypoints_2d") or []
    pts = [(k[i], k[i + 1]) for i in range(0, len(k) - 2, 3) if k[i + 2] > 0]
    if len(pts) < 4:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if max(max(xs), max(ys)) > 1.5:  # pixel coordinates
        xs, ys = [x / max(1.0, W) for x in xs], [y / max(1.0, H) for y in ys]
    return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(1e-3, max(ys) - min(ys)), len(pts))


def boxes(frame: dict) -> list:
    W, H = float(frame.get("canvas_width") or 1), float(frame.get("canvas_height") or 1)
    return [_box(p, W, H) for p in frame.get("people") or []]


def _score(b) -> float:
    cx, cy, h, n = b
    return h * (1.0 - 0.8 * min(0.5, abs(cx - 0.5))) * math.sqrt(n / 18.0)


def pick_lead(frames: list[dict], jump: float = 0.35, hold: int = 6) -> list:
    """For each frame, where to take the lead dancer from: (frame index, person index), or None.
    When the lead drops out for a few frames (a missed detection, a spin), their last pose is held
    rather than jumping to someone else; after `hold` frames the tallest, most central person is
    picked again."""
    out, prev, last, lost = [], None, None, 0
    for f, fr in enumerate(frames):
        bs = boxes(fr) if isinstance(fr, dict) else []
        cand = [(i, b) for i, b in enumerate(bs) if b is not None]
        pick = None
        if cand and prev is not None:
            pcx, pcy, ph, _ = prev
            d, i = min((abs(b[0] - pcx) + abs(b[1] - pcy) + 0.5 * abs(math.log(b[2] / ph)), i) for i, b in cand)
            if d <= jump:
                pick = i
        if pick is None and prev is not None and last is not None and lost < hold:
            lost += 1
            out.append(last)  # hold the lead's last pose
            continue
        if pick is None and cand:  # first frame, or the lead is gone: tallest, most central person
            pick = max(cand, key=lambda ib: _score(ib[1]))[0]
        if pick is None:
            out.append(None)
            continue
        prev, last, lost = bs[pick], (f, pick), 0
        out.append(last)
    return out


def only(frames: list[dict], src) -> dict:
    """Frame dict holding just the person `src` = (frame index, person index) points to."""
    if src is None:
        f0 = frames[0] if frames else {}
        return {**f0, "people": []}
    f, i = src
    people = frames[f].get("people") or []
    return {**frames[f], "people": [people[i]] if 0 <= i < len(people) else []}
