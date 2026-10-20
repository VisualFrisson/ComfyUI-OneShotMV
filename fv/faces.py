"""Face tracks for match cuts: where the face is (centre, size) through each clip.

OpenCV's frontal-face cascade on downscaled frames, every few frames, with gaps filled by
interpolation (short) or left empty (long). Tracks are cached as JSON beside the run so a
re-cut costs nothing. Coordinates are normalised to the frame: cx, cy in 0..1, w = face
width / frame width.
"""
from __future__ import annotations

import json
import os

import numpy as np

_CASCADE = None


def _cascade():
    global _CASCADE
    if _CASCADE is None:
        import cv2
        try:
            _CASCADE = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,
                                                          "haarcascade_frontalface_default.xml"))
        except Exception:
            _CASCADE = False
    return _CASCADE


def track(path: str, every: int = 3, size: int = 320) -> dict | None:
    """{fps, n, every, boxes: [[cx, cy, w] or None per sampled frame]} or None."""
    import cv2
    casc = _cascade()
    if not casc or not path or not os.path.exists(path):
        return None
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    boxes, k, prev = [], 0, None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if k % every == 0:
            h, w = f.shape[:2]
            s = size / max(h, w)
            g = cv2.cvtColor(cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            gh, gw = g.shape
            fs = casc.detectMultiScale(g, 1.1, 4, minSize=(int(0.08 * min(gh, gw)),) * 2)
            b = None
            if len(fs):
                if prev is not None:  # keep the same face if several are found
                    pc = np.array([prev[0] * gw, prev[1] * gh])
                    x, y, bw, bh = min(fs, key=lambda q: np.hypot(q[0] + q[2] / 2 - pc[0], q[1] + q[3] / 2 - pc[1])
                                       - 0.5 * q[2])
                else:
                    x, y, bw, bh = max(fs, key=lambda q: q[2] * q[3])
                b = [round((x + bw / 2) / gw, 4), round((y + bh / 2) / gh, 4), round(bw / gw, 4)]
                prev = b
            boxes.append(b)
        k += 1
    cap.release()
    # fill gaps of up to 3 samples by linear interpolation
    idx = [i for i, b in enumerate(boxes) if b is not None]
    for a, c in zip(idx, idx[1:]):
        if 1 < c - a <= 4:
            for i in range(a + 1, c):
                u = (i - a) / (c - a)
                boxes[i] = [round(boxes[a][j] * (1 - u) + boxes[c][j] * u, 4) for j in range(3)]
    return {"fps": fps, "n": k, "every": every, "boxes": boxes}


def tracks_for_run(run_dir: str, renders: list[dict], log=None) -> dict:
    """{render id: track} for every render clip, cached in <run>/faces.json."""
    cache_p = os.path.join(run_dir, "faces.json")
    try:
        cache = json.load(open(cache_p, encoding="utf-8"))
    except Exception:
        cache = {}
    out, changed = {}, False
    for r in renders:
        path = r.get("path", "")
        name = os.path.basename(path.replace("\\", "/"))
        if not name:
            continue
        local = os.path.join(run_dir, name)
        if not os.path.exists(local):
            continue
        key = f"{name}:{os.path.getsize(local)}"
        if key not in cache:
            t = track(local)
            if t is None:
                continue
            cache[key] = t
            changed = True
        out[r["id"]] = cache[key]
    if changed:
        try:
            json.dump(cache, open(cache_p, "w", encoding="utf-8"))
        except Exception:
            pass
    return out


def face_at(tr: dict | None, t_local: float):
    """Face box at clip-local time (seconds) or None."""
    if not tr:
        return None
    i = int(round(t_local * tr["fps"] / tr["every"]))
    b = tr["boxes"]
    if not 0 <= i < len(b) or b[i] is None:
        return None
    near = [b[k] for k in (i - 1, i, i + 1) if 0 <= k < len(b) and b[k] is not None]
    return [sum(q[j] for q in near) / len(near) for j in range(3)]  # steadies detector jitter


def match_cost(a, b) -> float:
    """How far apart two face boxes read on screen (0 = identical)."""
    if a is None or b is None:
        return 9.0
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + 1.2 * abs(np.log(max(1e-3, b[2]) / max(1e-3, a[2])))


def align(lead, src, max_scale=1.35, min_scale=0.9, max_shift=0.18):
    """Transform that puts the source face where the lead face is: (scale, dx, dy), with dx, dy
    the shift of the source frame in frame-widths / heights after scaling about its centre."""
    s = float(np.clip(lead[2] / max(1e-3, src[2]), min_scale, max_scale))
    dx = float(np.clip(lead[0] - (0.5 + (src[0] - 0.5) * s), -max_shift, max_shift))
    dy = float(np.clip(lead[1] - (0.5 + (src[1] - 0.5) * s), -max_shift, max_shift))
    return [round(s, 4), round(dx, 4), round(dy, 4)]
