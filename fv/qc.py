"""Take QC: catch the renders that would spoil a cut, so they can be re-rolled automatically.

Checks per clip (numpy + OpenCV only, a few seconds per clip):
  missing  - the file is not there or has far fewer frames than planned
  frozen   - almost no motion (LTX sometimes returns a near-still image)
  exposure - many frames blown to white or crushed to black
  cut      - a hard scene change inside the clip: LTX sometimes cuts away to another person or
             place mid-take (one consecutive-frame difference far above the clip's own typical
             motion; calibrated on n5/n11, where the bad takes jump to 0.24-0.30 and good ones stay
             below 0.16)
  mouth    - voice renders: the mouth barely moves (closed-mouth / dead takes)
             (mouth region from a face detector; activity = frame-to-frame change of the
              mouth crop relative to the whole face, plus how much the dark mouth cavity varies)
Mouth-vs-vocal correlation is reported for information only: on continuous singing it is
too noisy to reject takes with (calibrated on known-good renders).
"""
from __future__ import annotations

import os

import numpy as np

THRESH = {"min_frames_frac": 0.85, "frozen": 0.006, "exposure_frac": 0.25, "mouth_ratio": 0.5,
          "mouth_dark_std": 0.01, "cut_abs": 0.18, "cut_rel": 5.0}


def cut_jump(frames) -> tuple[float, float, int]:
    """Largest consecutive-frame change (mean abs, 48x80 colour thumbnails), the clip's median
    change, and where the largest one happens. The first and last two steps are ignored
    (LTX's first latent and the trailing frame can jump without being a cut)."""
    import cv2
    th = [cv2.resize(f, (48, 80), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 for f in frames]
    d = np.array([np.abs(b - a).mean() for a, b in zip(th, th[1:])])
    if len(d) < 8:
        return 0.0, 0.0, -1
    inner = d[2:-2]
    i = int(np.argmax(inner))
    return float(inner[i]), float(np.median(d)), i + 3


def _read(path: str, size: int = 256, max_frames: int = 400):
    import cv2
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    out = []
    while len(out) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        s = size / max(h, w)
        out.append(cv2.resize(f, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA))
    cap.release()
    return out, fps


_CASCADE = None


def _faces(gray):
    import cv2
    global _CASCADE
    if _CASCADE is None:
        try:
            _CASCADE = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
        except Exception:  # OpenCV builds without the cascade module: skip the mouth check
            _CASCADE = False
    if _CASCADE is False:
        return []
    h, w = gray.shape
    return _CASCADE.detectMultiScale(gray, 1.1, 4, minSize=(int(0.12 * min(h, w)),) * 2)


def face_box(frames):
    """Median face box over a few frames (x, y, w, h) or None."""
    import cv2
    boxes = []
    for f in frames[:: max(1, len(frames) // 8)][:8]:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        fs = _faces(g)
        if len(fs):
            boxes.append(max(fs, key=lambda b: b[2] * b[3]))
    if not boxes:
        return None
    return tuple(int(v) for v in np.median(np.array(boxes), 0))


def mouth_series(frames, box):
    """Per-frame activity of the mouth crop and of the whole face crop."""
    import cv2
    x, y, w, h = box
    my0, my1 = y + int(0.62 * h), min(frames[0].shape[0], y + int(1.02 * h))
    mx0, mx1 = x + int(0.22 * w), x + int(0.78 * w)
    m_prev = f_prev = None
    ma, fa, dark = [], [], []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        m = g[my0:my1, mx0:mx1]
        fc = g[y:y + h, x:x + w]
        if m.size == 0 or fc.size == 0:
            return None
        dark.append(float((m < np.percentile(fc, 12)).mean()))  # open mouth = dark cavity
        if m_prev is not None:
            ma.append(float(np.abs(m - m_prev).mean()))
            fa.append(float(np.abs(fc - f_prev).mean()))
        m_prev, f_prev = m, fc
    return np.array(ma), np.array(fa), np.array(dark)


def vocal_env(vocal: np.ndarray | None, sr: int, seek: float, n: int, fps: float):
    if vocal is None:
        return None
    x = vocal.mean(0) if vocal.ndim == 2 else vocal
    out = np.zeros(n, np.float32)
    half = int(sr / fps / 2) + 1
    for k in range(n):
        c = int((seek + k / fps) * sr)
        seg = x[max(0, c - half):c + half]
        out[k] = np.sqrt((seg ** 2).mean()) if len(seg) else 0.0
    return out


def check_clip(path: str, *, expected_frames: int = 0, voice: bool = False, vocal=None, sr: int = 44100,
               seek: float = 0.0) -> dict:
    r = {"path": path, "ok": True, "fail": [], "score": 1.0}
    if not path or not os.path.exists(path):
        r.update(ok=False, fail=["missing"], score=0.0)
        return r
    frames, fps = _read(path)
    r["frames"] = len(frames)
    if len(frames) < 4 or (expected_frames and len(frames) < THRESH["min_frames_frac"] * min(expected_frames, 400)):
        r.update(ok=False, fail=["missing"], score=0.0)
        return r
    g = np.stack([f.mean(2) for f in frames]).astype(np.float32) / 255
    motion = float(np.abs(np.diff(g, axis=0)).mean())
    r["motion"] = round(motion, 5)
    if motion < THRESH["frozen"]:
        r["fail"].append("frozen")
    flat = g.reshape(len(g), -1)
    lum, det = flat.mean(1), flat.std(1)
    # blown / black = extreme level AND no picture left (a high-key white-studio shot is fine)
    bad = float((((lum > 0.93) | (lum < 0.04)) & (det < 0.04)).mean())
    r["exposure_bad"] = round(bad, 3)
    if bad > THRESH["exposure_frac"]:
        r["fail"].append("exposure")
    jmax, jmed, jat = cut_jump(frames)
    r["jump"], r["jump_at"] = round(jmax, 3), jat
    if jmax > THRESH["cut_abs"] and jmax > THRESH["cut_rel"] * max(jmed, 1e-3):
        r["fail"].append("cut")
    if voice:
        box = face_box(frames)
        r["face"] = box is not None
        if box is not None:
            ms = mouth_series(frames, box)
            if ms is not None:
                ma, fa, dark = ms
                ratio = float(ma.mean() / max(1e-6, fa.mean()))
                r["mouth_ratio"] = round(ratio, 3)
                r["mouth_open_std"] = round(float(dark.std()), 4)
                env = vocal_env(vocal, sr, seek, len(dark), fps)
                if env is not None and env.std() > 1e-6 and dark.std() > 1e-6:
                    c = float(np.corrcoef(dark, env)[0, 1])
                    r["mouth_corr"] = round(c, 3)
                    singing = float((env > 0.25 * env.max()).mean())
                    r["singing_frac"] = round(singing, 2)
                if ratio < THRESH["mouth_ratio"] and dark.std() < THRESH["mouth_dark_std"]:
                    r["fail"].append("mouth")
    r["fail"] = sorted(set(r["fail"]))
    r["ok"] = not r["fail"]
    r["score"] = (0.0 if "missing" in r["fail"] else 1.0) - 0.4 * len(r["fail"]) \
        + 0.1 * min(2.0, r.get("mouth_ratio", 1.0)) + 5 * min(0.03, motion)
    return r
