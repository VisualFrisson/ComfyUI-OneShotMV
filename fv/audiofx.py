"""Audio-reactive control signals, one value per output video frame.

kick / snare / hat : attack-decay envelopes from onsets in the drums stem (or the mix)
vocal              : smoothed vocal-stem loudness (0..1)
loud               : smoothed mix loudness (0..1) relative to the whole window
"""
from __future__ import annotations

import numpy as np

HOP_S = 0.01  # 100 Hz analysis


def _mono(x: np.ndarray) -> np.ndarray:
    return x.mean(0) if x.ndim == 2 else x


def _band_env(x: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    """Log energy in [lo, hi] Hz at 100 Hz."""
    x = _mono(x).astype(np.float32)
    hop = int(sr * HOP_S)
    n_fft = 2048
    win = np.hanning(n_fft).astype(np.float32)
    n = max(1, 1 + (len(x) - n_fft) // hop)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
        n = 1
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = (freqs >= lo) & (freqs < hi)
    out = np.empty(n, np.float32)
    for i0 in range(0, n, 2048):  # chunked to bound memory
        idx = np.arange(i0, min(n, i0 + 2048))
        frames = np.stack([x[i * hop:i * hop + n_fft] for i in idx]) * win
        mag = np.abs(np.fft.rfft(frames, axis=1))[:, band]
        out[idx] = np.log1p(10.0 * (mag ** 2).sum(1))
    return out


def _onsets(env: np.ndarray, sens: float = 1.0) -> list[tuple[int, float]]:
    """Peak-picked onsets of a log-energy curve -> [(index, strength 0..1)]."""
    d = np.maximum(0.0, np.diff(env, prepend=env[:1]))
    if d.max() <= 0:
        return []
    k = 15
    pad = np.pad(d, k, mode="edge")
    med = np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(len(d))])
    thr = med + sens * 0.5 * d.std()
    out, last = [], -10
    for i in range(1, len(d) - 1):
        if d[i] > thr[i] and d[i] >= d[i - 1] and d[i] >= d[i + 1] and i - last >= 6:
            out.append((i, float(d[i])))
            last = i
    if out:
        m = np.percentile([s for _, s in out], 95) or 1.0
        out = [(i, min(1.0, s / m)) for i, s in out]
    return out


def _decay_env(onsets, n_frames: int, fps: float, t0: float, tau: float) -> np.ndarray:
    t = t0 + np.arange(n_frames) / fps
    env = np.zeros(n_frames, np.float32)
    for i, s in onsets:
        to = i * HOP_S
        dt = t - to
        m = (dt >= -0.5 / fps) & (dt < 6 * tau)
        env[m] = np.maximum(env[m], s * np.exp(-np.maximum(dt[m], 0.0) / tau))
    return env


def _smooth_level(x, sr, n_frames, fps, t0, win_s=0.25):
    x = _mono(x)
    t = t0 + np.arange(n_frames) / fps
    half = int(win_s * sr / 2)
    out = np.zeros(n_frames, np.float32)
    for k, tt in enumerate(t):
        c = int(tt * sr)
        seg = x[max(0, c - half):c + half]
        out[k] = np.sqrt((seg ** 2).mean()) if len(seg) else 0.0
    db = 20 * np.log10(out + 1e-5)
    lo, hi = np.percentile(db, 10), np.percentile(db, 98)
    return np.clip((db - lo) / max(1e-3, hi - lo), 0.0, 1.0)


def controls(mix: np.ndarray, sr: int, t0: float, t1: float, fps: float,
             drums: np.ndarray | None = None, vocals: np.ndarray | None = None,
             stem_sr: int | None = None) -> dict[str, np.ndarray]:
    """Envelopes for output frames covering song time [t0, t1)."""
    n = int(round((t1 - t0) * fps))
    d, dsr = (drums, stem_sr) if drums is not None else (mix, sr)
    kick = _decay_env(_onsets(_band_env(d, dsr, 30, 150)), n, fps, t0, 0.12)
    snare = _decay_env(_onsets(_band_env(d, dsr, 1500, 6000), 1.1), n, fps, t0, 0.10)
    hat = _decay_env(_onsets(_band_env(d, dsr, 7000, 16000), 1.3), n, fps, t0, 0.05)
    voc = _smooth_level(vocals if vocals is not None else mix, stem_sr or sr, n, fps, t0, 0.15)
    loud = _smooth_level(mix, sr, n, fps, t0, 0.6)
    return {"kick": kick, "snare": snare, "hat": hat, "vocal": voc, "loud": loud}
