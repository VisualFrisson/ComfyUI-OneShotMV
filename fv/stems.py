"""Stem separation (vocals / drums) with torchaudio's Hybrid Demucs, plus WAV I/O.

Runs inside ComfyUI (torch + torchaudio are ComfyUI requirements). The model weights are
fetched once by torchaudio into the torch hub cache."""
from __future__ import annotations

import os
import wave

import numpy as np


def write_wav(path: str, x: np.ndarray, sr: int) -> str:
    """x: (C, N) float32 in [-1, 1] -> 16-bit PCM WAV."""
    if x.ndim == 1:
        x = x[None, :]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").T.copy()
    with wave.open(path, "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return path


def read_wav(path: str, seek: float = 0.0, duration: float = 0.0) -> tuple[np.ndarray, int]:
    """16/24/32-bit PCM WAV -> (C, N) float32."""
    with wave.open(path, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        start = int(round(seek * sr))
        count = n - start if duration <= 0 else int(round(duration * sr))
        w.setpos(max(0, min(n, start)))
        raw = w.readframes(max(0, count))
    if sw == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        x = v.astype(np.float32) / float(1 << 23)
    elif sw == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / float(1 << 31)
    else:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    x = x.reshape(-1, ch).T
    if duration > 0 and x.shape[1] < int(round(duration * sr)):  # pad past the end with silence
        x = np.pad(x, ((0, 0), (0, int(round(duration * sr)) - x.shape[1])))
    return x, sr


def separate(x: np.ndarray, sr: int, segment: float = 10.0, overlap: float = 0.1,
             device: str | None = None) -> dict[str, np.ndarray]:
    """Hybrid Demucs (HDEMUCS_HIGH_MUSDB_PLUS). x: (C, N) -> {'drums','bass','other','vocals'}
    each (2, N') at the model rate (44.1 kHz); also returns 'sr'."""
    import torch
    from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
    from torchaudio.transforms import Fade, Resample

    if device is None:
        try:
            import comfy.model_management as mm
            device = mm.get_torch_device()
        except Exception:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model().to(device).eval()
    msr = bundle.sample_rate
    mix = torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(device)
    if mix.shape[0] == 1:
        mix = mix.repeat(2, 1)
    mix = mix[:2]
    if sr != msr:
        mix = Resample(sr, msr).to(device)(mix)
    ref = mix.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    mix = ((mix - mean) / std)[None]
    length = mix.shape[-1]
    chunk = int(round(segment * msr))
    ov = int(round(overlap * msr))
    out = torch.zeros(1, len(model.sources), 2, length, device=device)
    fade = Fade(fade_in_len=0, fade_out_len=ov, fade_shape="linear")
    start = 0
    with torch.no_grad():
        while start < length - ov:
            end = min(length, start + chunk)
            part = model(mix[..., start:end])
            fade.fade_in_len = ov if start else 0
            fade.fade_out_len = ov if end < length else 0
            out[..., start:end] += fade(part)
            if end >= length:
                break
            start += chunk - ov
    out = out * std + mean
    res = {name: out[0, i].float().cpu().numpy() for i, name in enumerate(model.sources)}
    res["sr"] = msr
    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return res
