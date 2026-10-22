"""Face tracks for match cuts: where the face is (centre, size) through each clip.

The S3FD face detector (installed with the SyncNet weights, see fv/lipsync.py) when available,
else OpenCV's frontal-face cascade, on downscaled frames every few frames, with gaps filled by
interpolation (short) or left empty (long). S3FD also finds the extreme close-ups and turned
heads the cascade misses. Tracks are cached as JSON beside the run so a re-cut costs nothing.
Coordinates are normalised to the frame: cx, cy in 0..1, w = face width / frame width.
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


def method() -> str:
    try:
        from . import lipsync
        if lipsync.available():
            return "s3fd"
    except Exception:
        pass
    return "haar" if _cascade() else ""


def _s3fd_detector():
    from . import lipsync
    dev = lipsync._device()
    return lipsync, lipsync._load(dev)[1], dev


def detect_all(rgb, size: int = 320) -> list:
    """Every face in one RGB uint8 frame: [[cx, cy, w], ...] normalised, largest first."""
    import cv2
    how = method()
    if not how:
        return []
    h, w = rgb.shape[:2]
    s = size / max(h, w)
    small = cv2.resize(np.ascontiguousarray(rgb), (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    gh, gw = small.shape[:2]
    if how == "s3fd":
        ls, det, dev = _s3fd_detector()
        fs = [(q[0], q[1], q[2] - q[0], q[3] - q[1]) for q in ls._detect(det, small.copy(), dev, 1.0, 0.8)]
    else:
        fs = _cascade().detectMultiScale(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), 1.1, 4,
                                         minSize=(int(0.04 * min(gh, gw)),) * 2)
    out = [[float((x + bw / 2) / gw), float((y + bh / 2) / gh), float(bw / gw)] for x, y, bw, bh in fs]
    return sorted(out, key=lambda b: -b[2])


def detect(rgb, size: int = 320):
    """Largest face in one RGB uint8 frame: [cx, cy, w] normalised, or None."""
    import cv2
    how = method()
    if not how:
        return None
    h, w = rgb.shape[:2]
    s = size / max(h, w)
    small = cv2.resize(np.ascontiguousarray(rgb), (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    gh, gw = small.shape[:2]
    if how == "s3fd":
        ls, det, dev = _s3fd_detector()
        fs = [(q[0], q[1], q[2] - q[0], q[3] - q[1]) for q in ls._detect(det, small.copy(), dev, 1.0, 0.8)]
    else:
        fs = _cascade().detectMultiScale(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), 1.1, 4,
                                         minSize=(int(0.08 * min(gh, gw)),) * 2)
    if not len(fs):
        return None
    x, y, bw, bh = max(fs, key=lambda q: q[2] * q[3])
    return [float((x + bw / 2) / gw), float((y + bh / 2) / gh), float(bw / gw)]


def track(path: str, every: int = 3, size: int = 320) -> dict | None:
    """{fps, n, every, boxes: [[cx, cy, w] or None per sampled frame]} or None."""
    import cv2
    how = method()
    if not how or not path or not os.path.exists(path):
        return None
    casc = _cascade() if how == "haar" else None
    if how == "s3fd":
        ls, det, dev = _s3fd_detector()
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
            small = cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            gh, gw = small.shape[:2]
            if how == "s3fd":  # (x, y, w, h) like the cascade's
                fs = [(q[0], q[1], q[2] - q[0], q[3] - q[1]) for q in
                      ls._detect(det, small[:, :, ::-1].copy(), dev, 1.0, 0.8)]
            else:
                g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                fs = casc.detectMultiScale(g, 1.1, 4, minSize=(int(0.08 * min(gh, gw)),) * 2)
            b = None
            if len(fs):
                if prev is not None:  # keep the same face if several are found
                    pc = np.array([prev[0] * gw, prev[1] * gh])
                    x, y, bw, bh = min(fs, key=lambda q: np.hypot(q[0] + q[2] / 2 - pc[0], q[1] + q[3] / 2 - pc[1])
                                       - 0.5 * q[2])
                else:
                    x, y, bw, bh = max(fs, key=lambda q: q[2] * q[3])
                b = [round(float((x + bw / 2) / gw), 4), round(float((y + bh / 2) / gh), 4), round(float(bw / gw), 4)]
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
        key = f"{name}:{os.path.getsize(local)}:{method()}"
        if key not in cache:
            t = track(local)
            if t is None:
                continue
            cache[key] = t
            changed = True
        out[r["id"]] = cache[key]
    if changed:
        try:  # write whole or not at all (a failed dump used to leave a truncated cache)
            txt = json.dumps(cache)
            with open(cache_p, "w", encoding="utf-8") as fh:
                fh.write(txt)
        except Exception as e:
            print(f"[FrissonVision] face cache not saved: {e}")
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


def apply_align(al, box):
    """Where a face box lands after the frame is warped by `al` (see align)."""
    if not al or box is None:
        return box
    s, dx, dy = al
    return [0.5 + (box[0] - 0.5) * s + dx, 0.5 + (box[1] - 0.5) * s + dy, box[2] * s]


def align(lead, src, max_scale=1.35, min_scale=0.9, max_shift=0.18):
    """Transform that puts the source face where the lead face is: (scale, dx, dy), with dx, dy
    the shift of the source frame in frame-widths / heights after scaling about its centre."""
    s = float(np.clip(lead[2] / max(1e-3, src[2]), min_scale, max_scale))
    dx = float(np.clip(lead[0] - (0.5 + (src[0] - 0.5) * s), -max_shift, max_shift))
    dy = float(np.clip(lead[1] - (0.5 + (src[1] - 0.5) * s), -max_shift, max_shift))
    return [round(s, 4), round(dx, 4), round(dy, 4)]
