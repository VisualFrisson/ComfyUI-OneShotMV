"""Lyrics with word timings, from the separated vocal (Whisper large-v3-turbo).

Used for b-roll that matches the words (fv/broll.py). The model lives in
ComfyUI/models/whisper/whisper-large-v3-turbo (the same folder comfy-mtb's Whisper loader uses):
the Hugging Face repo openai/whisper-large-v3-turbo, about 1.6 GB. Without it, b-roll falls
back to the theme or pack world."""
from __future__ import annotations

import os
import re

import numpy as np

MODEL = "whisper-large-v3-turbo"


def model_dir() -> str | None:
    roots = []
    try:
        import folder_paths  # type: ignore
        try:
            roots += folder_paths.get_folder_paths("whisper")
        except Exception:
            pass
        roots.append(os.path.join(folder_paths.models_dir, "whisper"))
    except Exception:
        pass
    for r in roots:
        d = os.path.join(r, MODEL)
        if os.path.exists(os.path.join(d, "model.safetensors")) and os.path.exists(os.path.join(d, "config.json")):
            return d
    return None


def available() -> bool:
    return model_dir() is not None


def _resample(x: np.ndarray, sr: int, to: int = 16000) -> np.ndarray:
    if sr == to:
        return x.astype(np.float32)
    try:
        import torch
        import torchaudio.functional as AF
        return AF.resample(torch.from_numpy(np.ascontiguousarray(x, np.float32)), sr, to).numpy()
    except Exception:
        n = int(round(len(x) * to / sr))
        return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def transcribe(vocal: np.ndarray, sr: int, log=print) -> list[dict]:
    """[{"w": word, "t0": s, "t1": s}] on the vocal's own time axis; [] without the model."""
    d = model_dir()
    if d is None:
        return []
    import torch
    x = vocal.mean(0) if vocal.ndim == 2 else vocal
    x = _resample(np.asarray(x, np.float32), sr)
    peak = float(np.abs(x).max() or 1.0)
    x = x / peak * 0.9
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    try:
        from transformers import pipeline
        try:
            asr = pipeline("automatic-speech-recognition", model=d, dtype=dt, device=dev)
        except TypeError:  # transformers < 4.56 calls it torch_dtype
            asr = pipeline("automatic-speech-recognition", model=d, torch_dtype=dt, device=dev)
        out = asr({"raw": x, "sampling_rate": 16000}, return_timestamps="word", chunk_length_s=30,
                  batch_size=4, generate_kwargs={"task": "transcribe"})
    except Exception as e:
        log(f"[FrissonVision] lyrics: Whisper failed ({e})")
        return []
    finally:
        try:
            del asr
        except Exception:
            pass
        if dev == "cuda":
            torch.cuda.empty_cache()
    words = []
    for c in out.get("chunks", []):
        t0, t1 = c.get("timestamp", (None, None))
        w = str(c.get("text", "")).strip()
        if not w or t0 is None:
            continue
        t1 = t1 if t1 is not None else t0 + 0.3
        words.append({"w": w, "t0": round(float(t0), 3), "t1": round(float(t1), 3)})
    return _drop_hallucinations(words)


def _drop_hallucinations(words: list[dict]) -> list[dict]:
    """Whisper repeats a phrase or invents 'Thank you.' over silence: drop runs of 3+ identical
    words and words longer than 2.5 s."""
    out = []
    for w in words:
        if w["t1"] - w["t0"] > 2.5:
            continue
        if len(out) >= 2 and out[-1]["w"].lower() == out[-2]["w"].lower() == w["w"].lower():
            continue
        out.append(w)
    return out


def lines(words: list[dict], gap: float = 0.5, max_words: int = 10) -> list[dict]:
    """Group words into sung lines at pauses (or every max_words words)."""
    out, cur = [], []
    for w in words:
        if cur and (w["t0"] - cur[-1]["t1"] > gap or len(cur) >= max_words):
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    res = []
    for ln in out:
        text = " ".join(x["w"] for x in ln)
        text = re.sub(r"\s+([,.!?'])", r"\1", text).strip()
        res.append({"text": text, "t0": ln[0]["t0"], "t1": ln[-1]["t1"]})
    return res


def line_near(lns: list[dict], t0: float, t1: float) -> dict | None:
    """The line sung most within [t0, t1], else the nearest one (for b-roll in a gap)."""
    best, bo = None, 0.0
    for ln in lns:
        o = min(t1, ln["t1"]) - max(t0, ln["t0"])
        if o > bo:
            best, bo = ln, o
    if best is not None:
        return best
    if not lns:
        return None
    mid = 0.5 * (t0 + t1)
    return min(lns, key=lambda ln: min(abs(ln["t0"] - mid), abs(ln["t1"] - mid)))
