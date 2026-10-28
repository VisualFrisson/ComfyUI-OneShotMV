"""Lip-sync repair: LatentSync 1.6 re-draws the mouth of a take whose lip sync stayed weak after
the re-rolls (fv/latentsync/). It runs inside the final QC pass on the leads SyncNet still
scores low; a repaired take replaces the original only when SyncNet scores it clearly higher.

Needs, besides diffusers / kornia / onnxruntime (all in a usual ComfyUI install):
  custom_nodes/ComfyUI-OneShotMV/weights/latentsync/latentsync_unet.pt   (ByteDance/LatentSync-1.6)
  .../weights/latentsync/whisper/tiny.pt                                      (same repo)
  .../weights/latentsync/sd-vae-ft-mse/{config.json, diffusion_pytorch_model.safetensors}
  ComfyUI/models/insightface/models/buffalo_l/{det_10g.onnx, 2d106det.onnx}"""
from __future__ import annotations

import os
import subprocess

import numpy as np

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(PACK, "weights", "latentsync")
_MODEL = None


def available() -> bool:
    try:
        from .latentsync import face, run
        import diffusers  # noqa: F401
        import kornia  # noqa: F401
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return run.available(WEIGHTS) and face.model_dir() is not None


def model(log=print):
    global _MODEL
    if _MODEL is None:
        import torch
        from .latentsync.run import LatentSync
        try:  # make room: the LTX and Gemma models are not needed for this step
            import comfy.model_management as mm  # type: ignore
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception:
            pass
        log("[OneShotMV] lip-sync repair: loading LatentSync 1.6")
        _MODEL = LatentSync(WEIGHTS, device="cuda" if torch.cuda.is_available() else "cpu")
    return _MODEL


def free():
    global _MODEL
    _MODEL = None
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _read(path: str):
    import cv2
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def _write(path: str, frames: np.ndarray, fps: float):
    from .assemble import ffmpeg_exe
    h, w = frames.shape[1:3]
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p",
           path]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    pr.stdin.write(np.ascontiguousarray(frames).tobytes())
    pr.stdin.close()
    pr.wait()


def repair(path: str, vocal: np.ndarray, sr: int, seek: float, out_path: str, *, seed: int = 1247,
           log=print) -> dict | None:
    """Re-draw the mouth of the take at `path` to the vocal from `seek` (stem time, s). Writes a
    25 fps clip to out_path; returns {"path", "fps", "frames", "face_frac"} or None."""
    from .lyrics import _resample
    src, fps = _read(path)
    if len(src) < 8:
        return None
    n25 = int((len(src) - 1) * 25.0 / fps) + 1
    frames = np.stack([src[min(len(src) - 1, int(round(k * fps / 25.0)))] for k in range(n25)])
    x = vocal.mean(0) if vocal.ndim == 2 else vocal
    a0 = int(round(max(0.0, seek) * sr))
    seg = np.asarray(x[a0:a0 + int(np.ceil(n25 / 25.0 * sr)) + sr // 10], np.float32)
    if seek < 0:  # the clip starts before the stem: pad the start with silence
        seg = np.concatenate([np.zeros(int(round(-seek * sr)), np.float32), seg])
    a16 = _resample(seg, sr, 16000)
    need = int(np.ceil(n25 / 25.0 * 16000))
    if len(a16) < need:
        a16 = np.pad(a16, (0, need - len(a16)))
    out, found = model(log)(frames, a16, fps=25.0, seed=seed, log=log)
    if out is None:
        return None
    _write(out_path, out, 25.0)
    return {"path": out_path, "fps": 25, "frames": int(n25), "face_frac": round(float(found), 2)}
