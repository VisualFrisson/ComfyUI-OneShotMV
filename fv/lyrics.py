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
    model = None
    out = None
    try:
        # the model directly (not transformers' ASR pipeline, whose audio path imports torchcodec:
        # on a ComfyUI whose torchcodec can't find its FFmpeg libraries that import fails)
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        proc = WhisperProcessor.from_pretrained(d)
        try:
            model = WhisperForConditionalGeneration.from_pretrained(d, dtype=dt, attn_implementation="eager")
        except TypeError:  # transformers < 4.56 calls it torch_dtype
            model = WhisperForConditionalGeneration.from_pretrained(d, torch_dtype=dt, attn_implementation="eager")
        model = model.to(dev).eval()
        tok = proc.tokenizer
        special = set(tok.all_special_ids)
        chunks = []
        for a, b in _pieces(x):
            feats = proc(x[a:b], sampling_rate=16000, return_tensors="pt").input_features.to(dev, dt)
            res = None
            for lang in ([language] if language else [None, "en"]):
                gk = {"task": "transcribe", "max_new_tokens": 220, "no_repeat_ngram_size": 3,
                      "return_token_timestamps": True, "return_dict_in_generate": True}
                if lang:
                    gk["language"] = lang
                with torch.inference_mode():
                    g = model.generate(feats, **gk)
                seq = g["sequences"][0].tolist()
                ts = g["token_timestamps"][0].tolist()
                text = tok.decode([t for t in seq if t not in special])
                res = (seq, ts, text)
                if not _degenerate(text):
                    break
            if res is None or _degenerate(res[2]):
                continue  # a looping piece is dropped, the rest kept
            seq, ts, _ = res
            off = a / 16000.0
            cur = None
            for i, (t, at) in enumerate(zip(seq, ts)):
                if t in special or t >= len(tok):
                    continue
                piece = tok.decode([t])
                if cur is None or piece.startswith(" "):
                    if cur and cur["text"].strip():
                        cur["timestamp"] = (cur["timestamp"][0], at + off)
                        chunks.append(cur)
                    cur = {"text": piece.strip(), "timestamp": (at + off, None)}
                else:
                    cur["text"] += piece
            if cur and cur["text"].strip():
                chunks.append(cur)
        out = {"chunks": chunks, "text": " ".join(c["text"] for c in chunks)}
    except Exception as e:
        LAST_ERROR = f"{type(e).__name__}: {e}"[:300]
        log(f"[FrissonVision] lyrics: Whisper failed ({LAST_ERROR})")
        return []
    finally:
        del model
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
