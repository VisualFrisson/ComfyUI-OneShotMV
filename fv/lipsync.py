"""Lip-sync scoring with SyncNet (Chung and Zisserman, 2016).

SyncNet embeds 0.2 s of mouth video and 0.2 s of speech in a shared space; in-sync pairs land
close together. Sliding the audio against the video gives, per take:

  offset - how far the mouth is from the voice, in 25 fps frames. Negative = the mouth moves
           before the sound (LTX's usual error); the cut can show that take later by -offset/25 s.
  conf   - how much better the best alignment is than a typical one (LSE-C in lip-sync papers;
           higher is better). Real talking-head video scores about 7-9, a dead or mushy mouth
           under 3. Singing over a vocal stem scores lower than speech.
  dist   - the best alignment's feature distance (LSE-D; lower is better).

Weights are not shipped (research licence, see fv/syncnet/NOTICE): put syncnet_v2.model and
sfd_face.pth in ComfyUI/models/syncnet/ (or this pack's weights/ folder). Without them the
scorer reports itself unavailable and Take QC skips the check.

The pipeline follows syncnet_python's: 25 fps, S3FD face boxes (smoothed), a 224x224 crop
around the face reaching down past the chin, 16 kHz MFCCs (13 coefficients, 100 per second).
"""
from __future__ import annotations

import math
import os
import threading

import numpy as np

FPS = 25
SR = 16000
_LOCK = threading.Lock()
_MODELS: dict = {}
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def weight_paths() -> tuple[str, str] | None:
    dirs = [os.environ.get("FV_SYNCNET_DIR", "")]
    try:
        import folder_paths  # ComfyUI
        dirs.append(os.path.join(folder_paths.models_dir, "syncnet"))
    except Exception:
        pass
    dirs.append(os.path.join(_HERE, "weights"))
    for d in dirs:
        a, b = os.path.join(d, "syncnet_v2.model"), os.path.join(d, "sfd_face.pth")
        if d and os.path.isfile(a) and os.path.isfile(b):
            return a, b
    return None


def available() -> bool:
    if weight_paths() is None:
        return False
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load(device: str):
    import torch
    from .syncnet.model import S
    from .syncnet.s3fd import S3FDNet
    with _LOCK:
        if device not in _MODELS:
            sn, sfd = weight_paths()
            s = S()
            s.load_state_dict(torch.load(sn, map_location=device))
            s.eval().to(device)
            d = S3FDNet(device=device)
            d.load_state_dict(torch.load(sfd, map_location=device))
            d.eval().to(device)
            _MODELS[device] = (s, d)
        return _MODELS[device]


def unload():
    with _LOCK:
        _MODELS.clear()


# ---- MFCC: python_speech_features 0.6 (MIT), the features SyncNet was trained on ---------------

def _hz2mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)


def _mel2hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def mfcc(sig: np.ndarray, sr: int = SR, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512,
         preemph=0.97, ceplifter=22) -> np.ndarray:
    """(frames, 13) MFCCs, numerically the same as python_speech_features.mfcc defaults."""
    from scipy.fftpack import dct
    sig = np.asarray(sig, np.float64)
    sig = np.append(sig[0], sig[1:] - preemph * sig[:-1])
    flen, fstep = int(round(winlen * sr)), int(round(winstep * sr))
    n = 1 if len(sig) <= flen else 1 + int(math.ceil((len(sig) - flen) / fstep))
    pad = np.concatenate([sig, np.zeros((n - 1) * fstep + flen - len(sig))])
    idx = np.arange(flen)[None, :] + np.arange(n)[:, None] * fstep
    frames = pad[idx]
    pspec = 1.0 / nfft * np.square(np.abs(np.fft.rfft(frames, nfft)))
    energy = pspec.sum(1)
    energy = np.where(energy == 0, np.finfo(float).eps, energy)
    mel = np.linspace(_hz2mel(0), _hz2mel(sr / 2), nfilt + 2)
    bins = np.floor((nfft + 1) * _mel2hz(mel) / sr)
    fb = np.zeros((nfilt, nfft // 2 + 1))
    for j in range(nfilt):
        for i in range(int(bins[j]), int(bins[j + 1])):
            fb[j, i] = (i - bins[j]) / (bins[j + 1] - bins[j])
        for i in range(int(bins[j + 1]), int(bins[j + 2])):
            fb[j, i] = (bins[j + 2] - i) / (bins[j + 2] - bins[j + 1])
    feat = pspec @ fb.T
    feat = np.where(feat == 0, np.finfo(float).eps, feat)
    feat = dct(np.log(feat), type=2, axis=1, norm="ortho")[:, :numcep]
    lift = 1 + (ceplifter / 2.0) * np.sin(np.pi * np.arange(numcep) / ceplifter)
    feat = feat * lift
    feat[:, 0] = np.log(energy)
    return feat


def _resample(x: np.ndarray, sr: int, to: int = SR) -> np.ndarray:
    if sr == to:
        return x
    from scipy.signal import resample_poly
    g = math.gcd(int(sr), int(to))
    return resample_poly(x, to // g, int(sr) // g)


# ---- faces ------------------------------------------------------------------------------------

def _detect(det, rgb: np.ndarray, device: str, scale: float, conf_th: float):
    """S3FD boxes (x0, y0, x1, y1, score) on one RGB frame (syncnet_python's detect_faces)."""
    import cv2
    import torch
    h, w = rgb.shape[:2]
    img = cv2.resize(rgb, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    img = img.transpose(2, 0, 1)[[2, 1, 0]].astype(np.float32)
    img -= np.array([104.0, 117.0, 123.0], np.float32)[:, None, None]
    img = img[[2, 1, 0]]
    with torch.no_grad():
        y = det(torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).to(device))
    d = y.data
    out = []
    for i in range(d.size(1)):
        j = 0
        while j < d.size(2) and d[0, i, j, 0] > conf_th:
            pt = (d[0, i, j, 1:] * torch.tensor([w, h, w, h], device=d.device)).cpu().numpy()
            out.append((pt[0], pt[1], pt[2], pt[3], float(d[0, i, j, 0])))
            j += 1
    return out


def _track(boxes: list, w: int, h: int):
    """One face per frame: the largest box, gaps interpolated, then median-smoothed centre and
    size (as syncnet_python). Returns per-frame (cx, cy, half-size) and the detected fraction."""
    from scipy.signal import medfilt
    n = len(boxes)
    have = [i for i, b in enumerate(boxes) if b is not None]
    if not have:
        return None, 0.0
    arr = np.array([boxes[i][:4] for i in have], np.float64)
    full = np.stack([np.interp(np.arange(n), have, arr[:, k]) for k in range(4)], 1)
    s = np.maximum(full[:, 3] - full[:, 1], full[:, 2] - full[:, 0]) / 2
    x, y = (full[:, 0] + full[:, 2]) / 2, (full[:, 1] + full[:, 3]) / 2
    k = min(13, n if n % 2 else n - 1)
    if k >= 3:
        s, x, y = medfilt(s, k), medfilt(x, k), medfilt(y, k)
    return np.stack([x, y, s], 1), len(have) / n


def _crops(frames_bgr: list, track: np.ndarray, crop_scale: float = 0.40) -> np.ndarray:
    import cv2
    out = []
    for f, (x, y, bs) in zip(frames_bgr, track):
        cs = crop_scale
        pad = int(bs * (1 + 2 * cs))
        p = cv2.copyMakeBorder(f, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(110, 110, 110))
        my, mx = y + pad, x + pad
        y1, y2 = int(my - bs), int(my + bs * (1 + 2 * cs))
        x1, x2 = int(mx - bs * (1 + cs)), int(mx + bs * (1 + cs))
        c = p[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
        out.append(cv2.resize(c, (224, 224)))
    return np.stack(out)


# ---- scoring ----------------------------------------------------------------------------------

def to_25fps(frames: list, fps: float) -> list:
    n = len(frames)
    t = int(n / fps * FPS)
    return [frames[min(n - 1, int(round(i * fps / FPS)))] for i in range(t)]


def score(frames_bgr: list, fps: float, audio: np.ndarray, sr: int, *, vshift: int = 15,
          batch: int = 20, device: str | None = None, det_scale: float | None = None) -> dict:
    """SyncNet offset / confidence of one take against its audio.

    frames_bgr: the clip's frames (BGR uint8, any size) at `fps`.
    audio: mono float samples (-1..1) that start at the clip's first frame, at `sr`.
    """
    import torch
    device = device or _device()
    net, det = _load(device)
    fr = to_25fps(frames_bgr, fps)
    if len(fr) < 12:
        return {"ok": False, "why": "too short"}
    h, w = fr[0].shape[:2]
    # S3FD wants faces of roughly 40-300 px at its input: scale big frames down
    scale = det_scale or min(1.0, 480.0 / max(h, w))
    boxes = []
    for f in fr:
        rgb = f[:, :, ::-1]
        b = _detect(det, rgb, device, scale, 0.9) or _detect(det, rgb, device, scale, 0.5)
        boxes.append(max(b, key=lambda z: (z[2] - z[0]) * (z[3] - z[1])) if b else None)
    track, face_frac = _track(boxes, w, h)
    if track is None or face_frac < 0.5:
        return {"ok": False, "why": "no face", "face_frac": round(face_frac, 2)}
    crops = _crops(fr, track)  # T x 224 x 224 x 3 (BGR, 0-255: what SyncNet was trained on)

    x = _resample(np.asarray(audio, np.float64), sr, SR)
    x = np.clip(x, -1, 1) * 32767.0  # syncnet_python reads int16 wav samples
    cc = mfcc(x, SR).T  # 13 x N (100 per second)
    n = min(len(crops), len(x) // 640)
    last = n - 5
    if last < 8:
        return {"ok": False, "why": "too short"}
    im = torch.from_numpy(crops[:n].astype(np.float32)).permute(3, 0, 1, 2).unsqueeze(0)  # 1x3xTx224x224
    ct = torch.from_numpy(cc.astype(np.float32))[None, None]
    vf, af = [], []
    with torch.no_grad():
        for i in range(0, last, batch):
            idx = range(i, min(last, i + batch))
            v = torch.cat([im[:, :, k:k + 5] for k in idx], 0).to(device)
            a = torch.cat([ct[:, :, :, k * 4:k * 4 + 20] for k in idx], 0).to(device)
            vf.append(net.forward_lip(v).cpu())
            af.append(net.forward_aud(a).cpu())
    vf, af = torch.cat(vf), torch.cat(af)
    afp = torch.nn.functional.pad(af, (0, 0, vshift, vshift))
    win = 2 * vshift + 1
    dists = torch.stack([torch.nn.functional.pairwise_distance(vf[[i]].repeat(win, 1), afp[i:i + win])
                         for i in range(len(vf))], 1)  # win x T
    mdist = dists.mean(1)
    minval, minidx = torch.min(mdist, 0)
    offset = vshift - int(minidx)
    conf = float(torch.median(mdist) - minval)
    # framewise confidence at the best offset: where in the take the sync holds or breaks
    fconf = (torch.median(mdist) - dists[int(minidx)]).numpy()
    from scipy.signal import medfilt
    fconf = medfilt(fconf, min(9, len(fconf) if len(fconf) % 2 else len(fconf) - 1))
    return {"ok": True, "offset": offset, "conf": round(conf, 3), "dist": round(float(minval), 3),
            "face_frac": round(face_frac, 2), "frames25": n,
            "conf_low_frac": round(float((fconf < 1.0).mean()), 3),
            "curve": [round(float(v), 3) for v in mdist.numpy()]}


def score_file(path: str, audio: np.ndarray, sr: int, seek: float = 0.0, **kw) -> dict:
    """Score a clip file against `audio` (mono or (ch, n)) starting `seek` seconds in."""
    import cv2
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        return {"ok": False, "why": "unreadable"}
    x = audio.mean(0) if audio.ndim == 2 else audio
    a0 = int(max(0.0, seek) * sr)
    seg = x[a0:a0 + int((len(frames) / fps + 0.5) * sr)]
    if seek < 0:  # the clip starts before the audio: pad with silence
        seg = np.concatenate([np.zeros(int(-seek * sr)), seg])
    return score(frames, fps, seg, sr, **kw)


def score_run(run_dir: str, *, leads_only: bool = True, every_take: bool = False, log=print) -> dict:
    """Score a finished run's clips against its vocal stem; writes <run>/lipsync.json. Read-only
    for the run itself (the plan is not touched)."""
    import json
    from . import stems
    p = json.load(open(os.path.join(run_dir, "plan.json"), encoding="utf-8"))
    vocal, sr = stems.read_wav(os.path.join(run_dir, "vocals.wav"))
    x = vocal.mean(0) if vocal.ndim == 2 else vocal
    out = {}
    for r in p["renders"]:
        if leads_only and r["role"] != "A":
            continue
        seek = r["t_start"] + p["song_t0"] - p.get("stems_t0", 0.0)
        paths = [t["path"] for t in r.get("takes", []) if t.get("path")] if every_take else []
        paths = paths or [r.get("path", "")]
        for path in dict.fromkeys(paths):
            local = os.path.join(run_dir, os.path.basename(path.replace("\\", "/")))
            if not os.path.exists(local):
                continue
            try:
                res = score_file(local, x, sr, seek=seek)
            except Exception as e:  # keep going
                res = {"ok": False, "why": f"error: {e}"}
            res.pop("curve", None)
            out[os.path.basename(local)] = dict(res, id=r["id"], role=r["role"], size=r.get("size"))
            log(f"[FrissonVision] lip sync {os.path.basename(local)}: {res}")
    json.dump(out, open(os.path.join(run_dir, "lipsync.json"), "w", encoding="utf-8"), indent=1)
    confs = [v["conf"] for v in out.values() if v.get("ok")]
    msg = (f"{len(confs)} takes scored: SyncNet confidence median {np.median(confs):.2f} "
           f"(min {min(confs):.2f}, max {max(confs):.2f})" if confs else "nothing scored")
    return {"message": msg, "scores": out}
