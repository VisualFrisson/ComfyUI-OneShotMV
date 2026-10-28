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


LAST_ERROR = ""
LAST_TEXT = ""


def _degenerate(text: str) -> bool:
    """Whisper looping on a sung hook ('S-D-D-D-D...'): few distinct tokens for the length."""
    toks = re.findall(r"[\w']+", text.lower())
    return len(toks) >= 12 and len(set(toks)) / len(toks) < 0.25


def _pieces(x: np.ndarray, sr: int = 16000, most: float = 28.0, least: float = 18.0) -> list[tuple[int, int]]:
    """Cut the vocal into pieces Whisper takes in one pass (< 30 s), at the quietest moment."""
    out, a, n = [], 0, len(x)
    hop = sr // 50
    while n - a > most * sr:
        lo, hi = a + int(least * sr), a + int(most * sr)
        seg = x[lo:hi]
        e = np.array([np.mean(seg[i:i + hop] ** 2) for i in range(0, len(seg) - hop, hop)])
        cut = lo + int(np.argmin(e)) * hop + hop // 2
        out.append((a, cut))
        a = cut
    out.append((a, n))
    return [(p, q) for p, q in out if q - p > sr // 2]


def transcribe(vocal: np.ndarray, sr: int, log=print, language: str | None = None, device=None) -> list[dict]:
    """[{"w": word, "t0": s, "t1": s}] on the vocal's own time axis; [] without the model.
    Word timings need the attention weights, so the model runs with eager attention; repeats are
    blocked (sung hooks send Whisper into loops) and a looping answer is retried in English."""
    global LAST_ERROR
    LAST_ERROR = ""
    d = model_dir()
    if d is None:
        LAST_ERROR = "no model"
        return []
    import torch
    x = vocal.mean(0) if vocal.ndim == 2 else vocal
    x = _resample(np.asarray(x, np.float32), sr)
    peak = float(np.abs(x).max() or 1.0)
    x = x / peak * 0.9
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float16 if dev == "cuda" else torch.bfloat16
    asr = None
    out = None
    try:
        # transformers' ASR pipeline (word timings, its own decoding setup) without its torchcodec
        # check: on a ComfyUI whose torchcodec can't find its FFmpeg libraries that import fails,
        # and raw arrays don't need it
        import transformers.pipelines.automatic_speech_recognition as _asr_mod
        _asr_mod.is_torchcodec_available = lambda: False
        from transformers import pipeline
        kw = {"model_kwargs": {"attn_implementation": "eager"}}
        try:
            asr = pipeline("automatic-speech-recognition", model=d, dtype=dt, device=dev, **kw)
        except TypeError:  # transformers < 4.56 calls it torch_dtype
            asr = pipeline("automatic-speech-recognition", model=d, torch_dtype=dt, device=dev, **kw)
        chunks = []
        for a, b in _pieces(x):
            res = None
            for lang in ([language] if language else [None, "en"]):
                gk = {"task": "transcribe", "max_new_tokens": 220, "no_repeat_ngram_size": 3}
                if lang:
                    gk["language"] = lang
                with torch.inference_mode():
                    res = asr({"raw": x[a:b], "sampling_rate": 16000}, return_timestamps="word", generate_kwargs=gk)
                if not _degenerate(res.get("text", "")):
                    break
            if res is None or _degenerate(res.get("text", "")):
                continue  # a looping piece is dropped, the rest kept
            off = a / 16000.0
            for c in res.get("chunks", []):
                t0, t1 = c.get("timestamp", (None, None))
                if t0 is None:
                    continue
                chunks.append({"text": c.get("text", ""), "timestamp": (t0 + off, None if t1 is None else t1 + off)})
        out = {"chunks": chunks, "text": " ".join(c["text"] for c in chunks)}
        global LAST_TEXT
        LAST_TEXT = out["text"]
    except Exception as e:
        LAST_ERROR = f"{type(e).__name__}: {e}"[:300]
        log(f"[OneShotMV] lyrics: Whisper failed ({LAST_ERROR})")
        return []
    finally:
        del asr
        if dev == "cuda":
            torch.cuda.empty_cache()
    if not out.get("chunks"):
        LAST_ERROR = "Whisper heard no words (or looped on every piece)"
        return []
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
