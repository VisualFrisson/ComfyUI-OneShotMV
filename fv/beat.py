"""beatcore: song structure analysis in pure numpy.

Finds the beat grid, the first beat of every bar, sections and the drop of a
song, which is what Drop Shift and Trend Remix time their cuts and looks to.
No librosa / scipy dependency so the module runs inside any ComfyUI install.

Main entry point: analyze(samples, sr, beats_per_bar=4) -> dict (JSON-safe).
"""
from __future__ import annotations

import math

import numpy as np

ANALYSIS_SR = 22050
N_FFT = 2048
HOP = 441  # 50 envelope frames per second at 22.05 kHz
ENV_FPS = ANALYSIS_SR / HOP


# --------------------------------------------------------------------- audio
def to_mono(samples) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 3:  # (batch, channels, n) as in ComfyUI AUDIO
        x = x[0]
    if x.ndim == 2:
        # (channels, n) or (n, channels)
        if x.shape[0] <= 8 and x.shape[1] > x.shape[0]:
            x = x.mean(axis=0)
        else:
            x = x.mean(axis=1)
    return x.astype(np.float32)


def resample(x: np.ndarray, sr: int, target: int = ANALYSIS_SR) -> np.ndarray:
    if sr == target:
        return x
    ratio = target / sr
    if ratio < 1.0:
        # cheap anti-alias: moving average over the decimation span
        k = max(1, int(round(1.0 / ratio)))
        if k > 1:
            kernel = np.ones(k, dtype=np.float32) / k
            x = np.convolve(x, kernel, mode="same")
    n_out = int(round(len(x) * ratio))
    t_out = np.arange(n_out, dtype=np.float64) / target
    t_in = np.arange(len(x), dtype=np.float64) / sr
    return np.interp(t_out, t_in, x).astype(np.float32)


def stft_mag(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    pad = n_fft // 2
    x = np.pad(x, (pad, pad), mode="reflect" if len(x) > pad else "constant")
    n_frames = 1 + (len(x) - n_fft) // hop
    if n_frames <= 0:
        return np.zeros((0, n_fft // 2 + 1), np.float32)
    window = np.hanning(n_fft).astype(np.float32)
    out = np.empty((n_frames, n_fft // 2 + 1), np.float32)
    step = 2048  # frames per chunk keeps memory flat on long songs
    for i0 in range(0, n_frames, step):
        i1 = min(n_frames, i0 + step)
        idx = (np.arange(i0, i1)[:, None] * hop) + np.arange(n_fft)[None, :]
        frames = x[idx] * window
        out[i0:i1] = np.abs(np.fft.rfft(frames, axis=1))
    return out


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 64, fmin: float = 30.0,
                   fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sr / 2

    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel2hz(m):
        return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)

    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz = mel2hz(mels)
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(bins)), np.float32)
    for m in range(n_mels):
        lo, ce, hi = hz[m], hz[m + 1], hz[m + 2]
        up = (bins - lo) / max(ce - lo, 1e-6)
        down = (hi - bins) / max(hi - ce, 1e-6)
        fb[m] = np.maximum(0.0, np.minimum(up, down))
    fb /= np.maximum(fb.sum(axis=1, keepdims=True), 1e-8)
    return fb


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    k = np.hanning(n + 2)[1:-1]
    k /= k.sum()
    return np.convolve(x, k, mode="same")


# ------------------------------------------------------------------ features
class Features:
    """Frame-level features at ENV_FPS (50/s)."""

    def __init__(self, x: np.ndarray, sr: int):
        y = resample(to_mono(x), sr, ANALYSIS_SR)
        peak = np.max(np.abs(y)) if len(y) else 0.0
        if peak > 0:
            y = y / peak
        self.duration = len(y) / ANALYSIS_SR
        mag = stft_mag(y)
        fb = mel_filterbank(ANALYSIS_SR, N_FFT, n_mels=64)
        mel = mag @ fb.T  # (frames, 64)
        self.logmel = np.log1p(100.0 * mel).astype(np.float32)
        freqs = np.fft.rfftfreq(N_FFT, 1.0 / ANALYSIS_SR)
        power = mag ** 2
        tot = power.sum(axis=1) + 1e-10
        self.rms_db = 10 * np.log10(tot / N_FFT + 1e-10)
        low = power[:, freqs < 150].sum(axis=1)
        mid = power[:, (freqs >= 150) & (freqs < 2000)].sum(axis=1)
        high = power[:, freqs >= 2000].sum(axis=1)
        self.low_db = 10 * np.log10(low / N_FFT + 1e-10)
        self.mid_db = 10 * np.log10(mid / N_FFT + 1e-10)
        self.high_db = 10 * np.log10(high / N_FFT + 1e-10)
        self.centroid = (power * freqs[None, :]).sum(axis=1) / tot
        # spectral flux onset envelope (log-mel, half-wave rectified)
        d = np.diff(self.logmel, axis=0, prepend=self.logmel[:1])
        flux = np.maximum(d, 0.0).mean(axis=1)
        flux = flux - _smooth(flux, int(ENV_FPS))  # remove slow trend
        self.onset = np.maximum(flux, 0.0).astype(np.float32)
        lowmel = self.logmel[:, :6]  # ~30-150 Hz bands: kick drum
        dl = np.diff(lowmel, axis=0, prepend=lowmel[:1])
        self.onset_low = np.maximum(dl, 0.0).mean(axis=1).astype(np.float32)
        self.n = len(self.onset)

    def t2f(self, t: float) -> int:
        return int(round(t * ENV_FPS))


# --------------------------------------------------------------------- tempo
def tempo_curve(onset: np.ndarray, bpm_lo: float = 40.0, bpm_hi: float = 400.0,
                step: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Fourier tempogram of the whole onset envelope: periodicity strength per BPM."""
    o = onset - onset.mean()
    t = np.arange(len(o)) / ENV_FPS
    bpms = np.arange(bpm_lo, bpm_hi + step, step)
    mag = np.empty(len(bpms))
    for i0 in range(0, len(bpms), 128):  # chunked to keep memory flat
        b = bpms[i0:i0 + 128]
        mag[i0:i0 + 128] = np.abs(np.exp(-2j * np.pi * np.outer(b / 60.0, t)) @ o)
    return bpms, mag


def estimate_tempo(onset: np.ndarray, bpm_min: float = 70.0, bpm_max: float = 180.0,
                   prior_bpm: float = 120.0) -> tuple[float, np.ndarray]:
    """Global tempo. A beat period also shows up at its double tempo (8th notes),
    so each candidate is scored on itself, 2x and 0.5x, then weighted by a
    broad log-normal prior around prior_bpm to settle octave ambiguity."""
    if len(onset) < 8:
        return prior_bpm, np.zeros(1)
    bpms, mag = tempo_curve(onset)

    def at(b):
        return np.interp(b, bpms, mag)

    cand = np.arange(bpm_min, bpm_max + 0.001, 0.05)
    score = at(cand) + 0.5 * at(2 * cand) + 0.25 * at(cand / 2)
    prior = np.exp(-0.5 * (np.log2(cand / prior_bpm) / 0.9) ** 2)
    score = score * prior
    i = int(np.argmax(score))
    return float(cand[i]), score


def track_beats(onset: np.ndarray, bpm: float, tightness: float = 400.0) -> np.ndarray:
    """Ellis (2007) dynamic-programming beat tracker. Returns envelope frame indices."""
    n = len(onset)
    period = 60.0 * ENV_FPS / bpm
    if n == 0 or period <= 1:
        return np.zeros(0, int)
    o = onset / (onset.std() + 1e-9)
    # local score: onset envelope smoothed with a gaussian ~ beat width
    w = int(round(period / 4)) * 2 + 1
    g = np.exp(-0.5 * (np.arange(-(w // 2), w // 2 + 1) / (period / 32)) ** 2)
    local = np.convolve(o, g, mode="same")
    backlink = -np.ones(n, int)
    cum = local.copy()
    lo, hi = int(round(-2 * period)), -int(round(period / 2))
    window = np.arange(lo, hi + 1)
    txcost = -tightness * (np.log(-window / period) ** 2)
    for i in range(n):
        z = i + window
        valid = z >= 0
        if not np.any(valid):
            continue
        cand = cum[z[valid]] + txcost[valid]
        j = int(np.argmax(cand))
        cum[i] = local[i] + cand[j]
        backlink[i] = z[valid][j]
    # start from the best-scoring frame in the last period
    tail = np.arange(max(0, n - int(period)), n)
    # choose a local max of cum near the end whose local score is decent
    end = int(tail[np.argmax(cum[tail])])
    beats = [end]
    while backlink[beats[-1]] >= 0:
        beats.append(int(backlink[beats[-1]]))
    beats = np.array(beats[::-1], int)
    # trim weak leading / trailing beats (silence)
    thr = 0.5 * np.median(local[beats]) if len(beats) else 0
    keep = np.where(local[beats] > thr)[0]
    if len(keep):
        beats = beats[keep[0]: keep[-1] + 1]
    return beats


def refine_beat_times(beat_frames: np.ndarray, onset: np.ndarray, bpm: float) -> np.ndarray:
    """Least-squares fit of a straight grid (constant tempo) through tracked beats,
    then snap each grid beat to the nearest onset peak within +-1/8 beat."""
    if len(beat_frames) < 4:
        return beat_frames / ENV_FPS
    idx = np.arange(len(beat_frames))
    A = np.vstack([idx, np.ones_like(idx)]).T.astype(np.float64)
    period, offset = np.linalg.lstsq(A, beat_frames.astype(np.float64), rcond=None)[0]
    grid = offset + period * idx
    # deviation of tracked beats from the straight grid tells whether tempo drifts
    dev = np.abs(beat_frames - grid)
    if np.median(dev) > period / 8:
        base = beat_frames.astype(np.float64)  # tempo changes: trust the tracker
    else:
        base = grid
    out = []
    rad = max(1, int(round(period / 8)))
    for b in base:
        c = int(round(b))
        lo, hi = max(0, c - rad), min(len(onset), c + rad + 1)
        if hi > lo and onset[lo:hi].max() > 0:
            j = lo + int(np.argmax(onset[lo:hi]))
            # only snap when the peak is clearly stronger than the grid position
            if onset[j] > 1.5 * onset[min(max(c, 0), len(onset) - 1)]:
                out.append(0.5 * (b + j))
                continue
        out.append(b)
    return np.asarray(out) / ENV_FPS


# ----------------------------------------------------------------- downbeats
def estimate_downbeat_phase(beat_times: np.ndarray, feats: Features, beats_per_bar: int) -> int:
    """Which beat (0..bpb-1) starts a bar: kicks land there, and the sound changes
    most right before it (chords, sections and fills all turn over on bar lines)."""
    if len(beat_times) < beats_per_bar * 2:
        return 0
    bf = np.clip((beat_times * ENV_FPS).round().astype(int), 0, feats.n - 1)
    kick = np.array([feats.onset_low[max(0, f - 1): f + 2].max() for f in bf])
    # spectral change across each beat: mean logmel of the beat after vs before
    change = np.zeros(len(bf))
    for i in range(1, len(bf) - 1):
        a = feats.logmel[bf[i - 1]: bf[i]].mean(axis=0)
        b = feats.logmel[bf[i]: bf[i + 1]].mean(axis=0)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0 and nb > 0:
            change[i] = 1.0 - float(a @ b) / (na * nb)
    kick = kick / (kick.mean() + 1e-9)
    change = change / (change.mean() + 1e-9)
    # long-range change: compare the bar before vs the bar after each beat
    bar = beats_per_bar
    lchange = np.zeros(len(bf))
    for i in range(bar, len(bf) - bar):
        a = feats.logmel[bf[i - bar]: bf[i]].mean(axis=0)
        b = feats.logmel[bf[i]: bf[i + bar]].mean(axis=0)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0 and nb > 0:
            lchange[i] = 1.0 - float(a @ b) / (na * nb)
    lchange = lchange / (lchange.mean() + 1e-9)
    score = []
    for p in range(beats_per_bar):
        sel = np.arange(p, len(bf), beats_per_bar)
        score.append(0.5 * kick[sel].mean() + 0.25 * change[sel].mean()
                     + 0.75 * np.sort(lchange[sel])[-max(1, len(sel) // 6):].mean())
    return int(np.argmax(score))


# ------------------------------------------------------------------ sections
def bar_feature_matrix(bar_starts: np.ndarray, end_time: float, feats: Features) -> np.ndarray:
    edges = list(bar_starts) + [end_time]
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        fa = max(0, feats.t2f(a))  # a pickup bar can start before 0 s
        fb = max(fa + 1, feats.t2f(b))
        fa, fb = min(fa, feats.n - 1), min(fb, feats.n)
        sl = slice(fa, fb)
        rows.append(np.concatenate([
            feats.logmel[sl].mean(axis=0),
            [feats.rms_db[sl].mean(), feats.low_db[sl].mean(), feats.high_db[sl].mean(),
             feats.onset[sl].mean() * 10, np.log(feats.centroid[sl].mean() + 1)],
        ]))
    return np.asarray(rows, np.float32)


def novelty_curve(F: np.ndarray, kernel_bars: int = 4) -> np.ndarray:
    n = len(F)
    if n < 3:
        return np.zeros(n)
    Z = F - F.mean(axis=0)
    Z = Z / (Z.std(axis=0) + 1e-6)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    S = Z @ Z.T
    k = kernel_bars
    ck = np.ones((2 * k, 2 * k))
    ck[:k, k:] = -1
    ck[k:, :k] = -1
    gx = np.exp(-0.5 * ((np.arange(2 * k) - k + 0.5) / (k * 0.6)) ** 2)
    ck = ck * np.outer(gx, gx)
    Sp = np.pad(S, k, mode="edge")
    nov = np.zeros(n)
    for i in range(n):
        nov[i] = (Sp[i: i + 2 * k, i: i + 2 * k] * ck).sum()
    nov = np.maximum(nov, 0)
    return nov / (nov.max() + 1e-9)


def pick_boundaries(nov: np.ndarray, energy: np.ndarray, low: np.ndarray,
                    high: np.ndarray | None = None, min_bars: int = 4,
                    phrase: int = 4) -> list[int]:
    """Section starts (bar indices): prominent peaks of timbre novelty or of
    loudness / bass / brightness jumps (builds, drops, breakdowns), preferring
    phrase-aligned bars. Prominence is judged locally, so a subtle change next
    to a dramatic one elsewhere in the song still counts."""
    n = len(nov)
    if n == 0:
        return [0]

    def jumps(v):
        if v is None:
            return np.zeros(n)
        z = (v - np.median(v)) / (np.std(v) + 1e-6)
        j = np.zeros(n)
        for i in range(1, n):
            j[i] = abs(z[i: min(n, i + 2)].mean() - z[max(0, i - 2): i].mean())
        return j / (j.max() + 1e-9)

    score = 0.45 * nov + 0.25 * jumps(low) + 0.15 * jumps(energy) + 0.15 * jumps(high)
    # phrase grid offset: songs with a pickup or an odd-length intro start their
    # 4/8-bar phrases on a bar other than bar 0
    offs = [float(score[o::phrase].sum()) / max(1, len(score[o::phrase])) for o in range(phrase)]
    off = int(np.argmax(offs))
    align = np.array([1.0 if (i - off) % phrase == 0 else (0.7 if (i - off) % 2 == 0 else 0.45)
                      for i in range(n)])
    score = score * align
    cands = []
    for i in range(1, n - 1):
        if not (score[i] >= score[i - 1] and score[i] >= score[i + 1]):
            continue
        lo, hi = max(0, i - 8), min(n, i + 9)
        local = np.median(score[lo:hi])
        if score[i] > 0.08 and score[i] > 2.0 * local:
            cands.append(i)
    cands.sort(key=lambda i: -score[i])
    chosen = [0]
    for c in cands:
        if all(abs(c - b) >= min_bars for b in chosen) and n - c >= 2:
            chosen.append(c)
    return sorted(chosen)


def label_sections(sections: list[dict]) -> None:
    """Section labels by energy. The user's rule (Oct 27): low energy (intro, break, outro) is only
    for intros and minimal bridges, sections well below the song's usual level; once the song
    gets going, medium (verse, build) is the norm; the drop and the clearly loudest sections are
    high (chorus). Tertiles marked a loud rap verse as a break (Big Dawgs 0:21-0:32 at 0.84 of the
    peak, the 33rd percentile being 0.85)."""
    if not sections:
        return
    en = np.array([s["energy"] for s in sections])
    hi = max(float(np.percentile(en, 75)), 0.6)
    lo = 0.55 * float(np.median(en))
    last = len(sections) - 1
    for i, s in enumerate(sections):
        if s.get("is_drop"):
            s["label"] = "drop"
        elif s["energy"] >= hi:
            s["label"] = "chorus"
        elif s["energy"] <= lo:
            s["label"] = "intro" if i == 0 else ("outro" if i == last else "break")
        elif i == last:
            s["label"] = "outro"  # a loud last section: plan._intense treats it as a chorus when loud
        else:
            s["label"] = "verse"
        nxt = sections[i + 1] if i + 1 < len(sections) else None
        if nxt is not None and nxt.get("is_drop") and s["label"] != "chorus":
            s["label"] = "build"


def find_drop(sections: list[dict], bar_energy: np.ndarray, bar_low: np.ndarray,
              bar_high: np.ndarray | None = None) -> int | None:
    """Index of the section that is the drop: the first section that reaches the
    song's peak bass level with a clear rise into it (after a build, the kick and
    bass come back in and the risers stop). Values are in dB."""
    if len(sections) < 2:
        return None
    n = len(bar_low)
    heads = []
    for s in sections:
        a0, a1 = s["bar_start"], min(s["bar_end"], s["bar_start"] + 4)
        b1 = max(s["bar_end"], a0 + 1)
        heads.append((0.5 * float(bar_low[a0:a1].mean()) + 0.5 * float(bar_low[a0:b1].mean()),
                      0.5 * float(bar_energy[a0:a1].mean()) + 0.5 * float(bar_energy[a0:b1].mean())))
    l_max = max(h[0] for h in heads)
    e_max = max(h[1] for h in heads)
    best, best_i = -1e9, None
    for i in range(1, len(sections)):
        a0 = sections[i]["bar_start"]
        b0 = max(0, a0 - 2)
        if a0 <= b0:
            continue
        l_sec, e_sec = heads[i]
        level = math.exp(-max(0.0, l_max - l_sec) / 6.0) * math.exp(-max(0.0, e_max - e_sec) / 8.0)
        rise = float(np.clip((l_sec - bar_low[b0:a0].mean()) / 12.0, 0.0, 1.0))
        e_rise = float(np.clip((e_sec - bar_energy[b0:a0].mean()) / 6.0, -1.0, 1.0))
        bright = 0.0
        if bar_high is not None:
            bright = float(np.clip((bar_high[b0:a0].mean() - bar_high[a0:a0 + 2].mean()) / 10.0,
                                   0.0, 1.0))
        score = level * (0.55 + 0.35 * rise + 0.1 * max(0.0, e_rise)) + 0.15 * bright \
            - 0.3 * (a0 / max(1, n))
        sections[i]["drop_score"] = round(score, 3)
        if level >= 0.6 and (rise >= 0.35 or bright >= 0.5) and score > best:
            best, best_i = score, i
    return best_i


# ------------------------------------------------------------------- analyze
def analyze(samples, sr: int, beats_per_bar: int = 4, min_section_bars: int = 4,
            bpm_min: float = 70.0, bpm_max: float = 180.0) -> dict:
    feats = Features(samples, sr)
    bpm, _ = estimate_tempo(feats.onset, bpm_min, bpm_max)
    bframes = track_beats(feats.onset, bpm)
    beat_times = refine_beat_times(bframes, feats.onset, bpm)
    if len(beat_times) >= 2:
        bpm = 60.0 / float(np.median(np.diff(beat_times)))
    phase = estimate_downbeat_phase(beat_times, feats, beats_per_bar)
    downbeats = beat_times[phase::beats_per_bar]
    bar_period = beats_per_bar * 60.0 / bpm
    # extend bars backwards to cover a pickup before the first downbeat
    bar_starts = list(downbeats)
    while bar_starts and bar_starts[0] - bar_period > -0.25 * bar_period and bar_starts[0] > 0.3:
        bar_starts.insert(0, max(0.0, bar_starts[0] - bar_period))
    bar_starts = np.asarray(bar_starts)
    end_time = feats.duration
    sections = []
    drop = None
    if len(bar_starts) >= 2:
        F = bar_feature_matrix(bar_starts, end_time, feats)
        bar_energy = F[:, -5]
        bar_low = F[:, -4]
        bar_high = F[:, -3]
        nov = novelty_curve(F, kernel_bars=2)
        if len(F) >= 16:
            nov = 0.5 * nov + 0.5 * novelty_curve(F, kernel_bars=4)
        starts = pick_boundaries(nov, bar_energy, bar_low, bar_high, min_bars=min_section_bars)
        ends = starts[1:] + [len(bar_starts)]
        for a, b in zip(starts, ends):
            t0 = float(bar_starts[a])
            t1 = float(bar_starts[b]) if b < len(bar_starts) else end_time
            sections.append({"bar_start": int(a), "bar_end": int(b), "start": round(t0, 4),
                             "end": round(t1, 4), "energy": float(bar_energy[a:b].mean()),
                             "low": float(bar_low[a:b].mean())})
        di = find_drop(sections, bar_energy, bar_low, bar_high)
        if di is not None:
            sections[di]["is_drop"] = True
            drop = {"section": di, "time": sections[di]["start"], "bar": sections[di]["bar_start"]}
        label_sections(sections)
        # energy 0..1 on an absolute dB scale below the loudest section (12 dB down = 0), so
        # two equally loud halves of a song stay "loud" (a min-max scale would call one quiet)
        e = np.array([s["energy"] for s in sections])
        for s in sections:
            s["energy_db"] = round(float(s["energy"] - e.max()), 2)
            s["energy"] = round(float(np.clip(1.0 + (s["energy"] - e.max()) / 12.0, 0.0, 1.0)), 3)
            s.pop("low", None)
    return {
        "version": 1,
        "duration": round(end_time, 4),
        "tempo": round(float(bpm), 2),
        "beats_per_bar": beats_per_bar,
        "beats": [round(float(t), 4) for t in beat_times],
        "downbeats": [round(float(t), 4) for t in bar_starts],
        "sections": sections,
        "drop": drop,
        "onset_env_fps": ENV_FPS,
    }


# ------------------------------------------------------------ window helpers
def choose_window(tl: dict, length_s: float, drop_at: float = 0.4, start: float | None = None,
                  phrase_bars: int = 0) -> tuple[float, float]:
    """Pick the stretch of song a clip of `length_s` seconds is cut to.
    Default: the drop lands `drop_at` of the way in, snapped to a bar line. phrase_bars > 0: the
    end moves to the nearest bar line that closes a phrase of that many bars (counted from the
    drop, or from the start without one), so the cut ends where the music does."""
    dur = tl["duration"]
    if start is not None and start >= 0:
        s = start
    elif tl.get("drop"):
        s = tl["drop"]["time"] - drop_at * length_s
    else:
        # no drop: the section-start window with the most energy over its whole length
        # (starting at the loudest section alone can run the clip into the outro)
        secs = tl.get("sections") or []

        def mean_energy(a):
            b, tot = a + length_s, 0.0
            for x in secs:
                ov = max(0.0, min(b, x["end"]) - max(a, x["start"]))
                tot += ov * x["energy"]
            return tot / max(1e-6, min(b, dur) - a) - (0.5 if b > dur + 0.5 else 0.0)
        cands = sorted({max(0.0, x["start"]) for x in secs}) or [0.0]
        s = max(cands, key=lambda a: (round(mean_energy(a), 2), -a))
    bars = tl.get("downbeats") or [0.0]
    # snap to the bar line at or before s (a start within half a beat of a bar line counts
    # as that bar, so a typed start like 7.48 for a bar at 7.49 isn't pushed a bar back)
    tol = 30.0 / max(float(tl.get("tempo") or 120.0), 1.0)
    prior = [b for b in bars if b <= s + tol]
    s = prior[-1] if prior else bars[0]
    if phrase_bars > 0 and (start is None or start < 0) and tl.get("drop") and len(bars) > 2:
        # an automatic start also lands on a phrase line before the drop (4 bars, not 5)
        i_drop = min(range(len(bars)), key=lambda i: abs(bars[i] - tl["drop"]["time"]))
        i_s = bars.index(s) if s in bars else i_drop
        k = max(phrase_bars, int(round((i_drop - i_s) / phrase_bars)) * phrase_bars)
        if i_drop - k >= 0:
            s = bars[i_drop - k]
    s = max(0.0, min(s, max(0.0, dur - length_s)))
    e = min(dur, s + length_s)
    if phrase_bars > 0 and len(bars) > 2 and e < dur - 0.05:
        ref = tl["drop"]["time"] if tl.get("drop") else s
        i_ref = min(range(len(bars)), key=lambda i: abs(bars[i] - ref))
        ends = [b for i, b in enumerate(bars) if (i - i_ref) % phrase_bars == 0
                and 0.75 * length_s <= b - s <= 1.25 * length_s]
        if ends:
            e = min(ends, key=lambda b: abs(b - (s + length_s)))
    return round(s, 4), round(min(dur, e), 4)


def slice_timeline(tl: dict, t0: float, t1: float) -> dict:
    """Timeline re-based to a window [t0, t1) of the song."""
    def keep(ts):
        return [round(t - t0, 4) for t in ts if t0 - 1e-6 <= t < t1]
    secs = []
    for s in tl.get("sections", []):
        a, b = max(s["start"], t0), min(s["end"], t1)
        if b - a > 0.05:
            d = dict(s)
            d["start"], d["end"] = round(a - t0, 4), round(b - t0, 4)
            secs.append(d)
    drop = None
    if tl.get("drop") and t0 <= tl["drop"]["time"] < t1:
        drop = {"time": round(tl["drop"]["time"] - t0, 4)}
    out = dict(tl)
    out.update(duration=round(t1 - t0, 4), beats=keep(tl["beats"]),
               downbeats=keep(tl["downbeats"]), sections=secs, drop=drop,
               window=[t0, t1])
    return out


def frame_of(t: float, fps: float) -> int:
    return int(round(t * fps))
