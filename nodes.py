"""Frisson Vision ComfyUI nodes."""
from __future__ import annotations

import importlib
import os
import random

import numpy as np
import torch

import folder_paths

from . import fv as _fvpkg  # noqa: F401
from .fv import assemble as fv_assemble
from .fv import audiofx, beat, casting, director, export, faces, finish, formats, fx, graphs, plan, qc, stems, dsplan

CATEGORY = "Frisson Vision"
_MODULES = [beat, dsplan, formats, casting, faces, plan, graphs, stems, audiofx, fx, finish, export, qc, fv_assemble, director]


def _reload():
    """Pick up edits to the core without restarting ComfyUI (cheap; modules are small)."""
    if os.environ.get("FV_NO_RELOAD"):
        return
    for m in _MODULES:
        try:
            importlib.reload(m)
        except Exception as e:  # keep the old module if an edit is broken
            print(f"[FrissonVision] reload {m.__name__} failed: {e}")


def _resolve(path: str) -> str:
    path = (path or "").strip().strip('"')
    if not path:
        return path
    if os.path.isabs(path):
        return path
    for base in (folder_paths.get_input_directory(), folder_paths.get_output_directory()):
        p = os.path.join(base, path)
        if os.path.exists(p):
            return p
    return os.path.join(folder_paths.get_input_directory(), path)


def _file_ref(path: str) -> dict:
    """A file under the output folder as a ComfyUI ui entry (so API / SDK clients can fetch it)."""
    rel = os.path.relpath(path, folder_paths.get_output_directory())
    sub, name = os.path.split(rel)
    return {"filename": name, "subfolder": sub.replace(os.sep, "/"), "type": "output"}


def _inputs_dir() -> str:
    d = os.path.join(folder_paths.get_output_directory(), "frissonvision", "inputs")
    os.makedirs(d, exist_ok=True)
    return d


def _save_image_input(image) -> str:
    """IMAGE tensor -> PNG under output/frissonvision/inputs (named by content, so reruns reuse it)."""
    import hashlib
    from PIL import Image
    arr = (image[0].detach().cpu().float().clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)
    path = os.path.join(_inputs_dir(), f"artist_{hashlib.sha1(arr.tobytes()).hexdigest()[:12]}.png")
    if not os.path.exists(path):
        Image.fromarray(arr).save(path)
    return path


def _video_input_path(video) -> str:
    """VIDEO input -> a file path the render graphs can read."""
    try:
        src = video.get_stream_source()
        if isinstance(src, str) and os.path.exists(src):
            return src
    except Exception:
        pass
    import time
    path = os.path.join(_inputs_dir(), f"dance_{int(time.time())}.mp4")
    video.save_to(path)
    return path


# ------------------------------------------------------------------ loaders
class FVLoadImagePath:
    """Load an image from a path. Unlike Load Image it does not check the file when the job is
    queued, so it can read a file an earlier job in the same queue will write."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = CATEGORY + "/io"

    def load(self, path):
        from PIL import Image, ImageOps
        im = ImageOps.exif_transpose(Image.open(_resolve(path))).convert("RGB")
        return (torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)[None],)

    @classmethod
    def IS_CHANGED(cls, path):
        p = _resolve(path)
        return os.path.getmtime(p) if os.path.exists(p) else random.random()


class FVLoadVideoPath:
    """Frames of a video from a path (no check at queue time). Center-cropped to width x height
    (0 = keep), resampled to force_fps (0 = keep), starting at start_seconds."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": ""}),
                             "width": ("INT", {"default": 0, "min": 0, "max": 8192}),
                             "height": ("INT", {"default": 0, "min": 0, "max": 8192}),
                             "frame_cap": ("INT", {"default": 0, "min": 0, "max": 100000}),
                             "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.001}),
                             "force_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01})}}

    RETURN_TYPES = ("IMAGE", "FLOAT", "INT")
    RETURN_NAMES = ("images", "fps", "frame_count")
    FUNCTION = "load"
    CATEGORY = CATEGORY + "/io"

    def load(self, path, width, height, frame_cap, start_seconds, force_fps):
        import cv2
        p = _resolve(path)
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise FileNotFoundError(p)
        sfps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ofps = force_fps or sfps
        frames, want = [], frame_cap or 10 ** 9
        src = []
        first = int(round(start_seconds * sfps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        idx = first
        while True:
            ok, f = cap.read()
            if not ok:
                break
            src.append(f)
            idx += 1
            if len(src) / sfps > (want + 2) / ofps + 0.5:
                break
        cap.release()
        if not src:
            raise ValueError(f"no frames read from {p} at {start_seconds}s")
        k = 0
        while len(frames) < want:
            si = int(round(k * sfps / ofps))
            if si >= len(src):
                if frame_cap and len(frames) < frame_cap:
                    si = len(src) - 1  # hold the last frame to reach the requested length
                else:
                    break
            f = src[si]
            if width and height:
                f = fv_assemble.cover(f, (width, height), cv2)
            frames.append(f[:, :, ::-1])
            k += 1
        arr = np.stack(frames).astype(np.float32) / 255.0
        return (torch.from_numpy(arr), float(ofps), len(frames))

    @classmethod
    def IS_CHANGED(cls, path, **kw):
        p = _resolve(path)
        return os.path.getmtime(p) if os.path.exists(p) else random.random()


class FVLoadAudioPath:
    """Audio from a WAV path (seek + duration; silence past the end)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": ""}),
                             "seek_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.001}),
                             "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.001})}}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load"
    CATEGORY = CATEGORY + "/io"

    def load(self, path, seek_seconds, duration):
        p = _resolve(path)
        if p.lower().endswith(".wav"):
            x, sr = stems.read_wav(p, seek_seconds, duration)
        else:
            import torchaudio
            w, sr = torchaudio.load(p)
            a = int(seek_seconds * sr)
            b = a + int(duration * sr) if duration > 0 else w.shape[1]
            x = w[:, a:b].numpy()
        return ({"waveform": torch.from_numpy(np.ascontiguousarray(x, np.float32))[None], "sample_rate": int(sr)},)


# ------------------------------------------------------------------ the app
class FVDirector:
    """Frisson Vision in one press: analyses the song, plans the cut in the chosen style pack,
    casts the performers and queues every render (grouped by model) plus the final assembly."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "mode": (["voice", "motion"], {"default": "voice"}),
            "pack": (casting.list_packs(), {"default": "crowd"}),
            "format": (formats.FORMAT_NAMES, {"default": "9:16"}),
            "preset": (formats.PRESET_NAMES, {"default": "quality"}),
            "seconds": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 600.0, "step": 0.5,
                                  "tooltip": "Length of the edit; 0 = the whole song"}),
            "start": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 3600.0, "step": 0.01,
                                "tooltip": "Song time to start at; -1 = pick the strongest stretch"}),
            "seed": ("INT", {"default": 1, "min": 0, "max": 2 ** 31 - 1}),
            "twins": (["auto", "loud", "all", "off"], {"default": "auto",
                      "tooltip": "Pose-matched twins for flash inserts (voice mode)"}),
            "queue_renders": ("BOOLEAN", {"default": True}),
        }, "optional": {
            "dance_video_path": ("STRING", {"default": "", "tooltip": "Motion mode: path to the dance clip"}),
            "run_name": ("STRING", {"default": ""}),
            "twin_editor": (["klein", "qwen"], {"default": "klein"}),
            "guide_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "queue_assemble": ("BOOLEAN", {"default": True,
                               "tooltip": "Also queue the final Assemble (off when a script assembles itself)"}),
            "framing": (["close", "mixed"], {"default": "close",
                        "tooltip": "close = close-ups only; mixed = medium and wide shots in quiet stretches"}),
            "artist_image_path": ("STRING", {"default": "",
                                  "tooltip": "One-artist mode: a photo of the singer (uses the artist pack)"}),
            "artist_image": ("IMAGE", {"tooltip": "One-artist mode: connect a Load Image with the singer's photo "
                                                  "(instead of artist_image_path)"}),
            "dance_video": ("VIDEO", {"tooltip": "Motion mode: connect a Load Video with the dance clip "
                                                 "(instead of dance_video_path)"}),
            "qc_retries": ("INT", {"default": 1, "min": 0, "max": 3,
                           "tooltip": "Check every take (frozen / dead mouth / blown / missing) and re-roll bad ones "
                                      "this many times before the cut; 0 = off"}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("summary", "run_dir")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kw):
        return random.random()

    def run(self, audio, mode, pack, format, preset, seconds, start, seed, twins, queue_renders,
            dance_video_path="", run_name="", twin_editor="klein", guide_strength=1.0, queue_assemble=True,
            framing="close", artist_image_path="", qc_retries=1, artist_image=None, dance_video=None):
        _reload()
        if artist_image is not None:
            artist_image_path = _save_image_input(artist_image)
        if dance_video is not None:
            dance_video_path = _video_input_path(dance_video)
        wf = audio["waveform"][0].detach().cpu().float().numpy()
        res = director.direct(audio=wf, sr=int(audio["sample_rate"]), out_root=folder_paths.get_output_directory(),
                              mode=mode, pack=pack, fmt=format, preset=preset, seconds=seconds, start=start,
                              seed=seed, twins=twins, dance_path=_resolve(dance_video_path), run_name=run_name,
                              queue=queue_renders, editor=twin_editor, guide_strength=guide_strength,
                              queue_assemble=queue_assemble, framing=framing,
                              artist_path=_resolve(artist_image_path) if artist_image_path else "",
                              qc_retries=qc_retries)
        ui = {"text": [res["summary"]]}
        ui["text"].append(_file_ref(res["bundle"]))  # jobs.json as a job output (SDK clients)
        return {"ui": ui, "result": (res["summary"], res["run_dir"])}


class FVAssemble:
    """Cuts a Frisson Vision run together on the beat, with audio-reactive FX and the finish.
    Queued automatically as the last job of a run; queue it again to re-cut."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"run_dir": ("STRING", {"default": ""}),
                             "filename_prefix": ("STRING", {"default": "frissonvision/final"})},
                "optional": {"fx_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                             "finish": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview", "video_path")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kw):
        return random.random()

    def run(self, run_dir, filename_prefix, fx_intensity=1.0, finish=True):
        _reload()
        import comfy.utils
        pbar = comfy.utils.ProgressBar(100)
        res = fv_assemble.assemble(_resolve(run_dir), fx_intensity=fx_intensity, do_finish=finish,
                                   progress=lambda i, n: pbar.update_absolute(int(100 * i / max(1, n)), 100))
        prev = torch.from_numpy(np.stack(res["previews"]).astype(np.float32) / 255.0) if res["previews"] \
            else torch.zeros(1, 64, 64, 3)
        msg = f"{res['video']}  ({res['frames']} frames)"
        if res["missing_renders"]:
            msg += f"  missing renders: {res['missing_renders']}"
        ui = {"text": [msg]}
        out_dir = folder_paths.get_output_directory()
        try:  # expose the MP4 as a job output (UI preview, API / SDK download)
            rel = os.path.relpath(res["video"], out_dir)
            if not rel.startswith(".."):
                sub, name = os.path.split(rel)
                ui["gifs"] = [{"filename": name, "subfolder": sub, "type": "output", "format": "video/h264-mp4"}]
        except ValueError:
            pass
        return {"ui": ui, "result": (prev, res["video"])}


class FVQC:
    """Checks every take of a run (missing, frozen, blown exposure, dead mouth), re-rolls the bad
    ones with new seeds and queues itself again; when everything passes (or retries run out) it
    keeps the best take of each shot and queues the final Assemble."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"run_dir": ("STRING", {"default": ""}),
                             "pass_index": ("INT", {"default": 0, "min": 0, "max": 9}),
                             "max_passes": ("INT", {"default": 1, "min": 0, "max": 9})},
                "optional": {"then_assemble": ("BOOLEAN", {"default": True}),
                             "queue_rerolls": ("BOOLEAN", {"default": True,
                                               "tooltip": "off = return the re-roll jobs (qcN.json) for an "
                                                          "SDK client to submit"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kw):
        return random.random()

    def run(self, run_dir, pass_index, max_passes, then_assemble=True, queue_rerolls=True):
        _reload()
        res = director.qc_pass(_resolve(run_dir), pass_idx=pass_index, max_passes=max_passes,
                               then_assemble=then_assemble, queue_rerolls=queue_rerolls)
        return {"ui": {"text": [res["message"], _file_ref(res["bundle"])]}, "result": (res["message"],)}


class FVFinish:
    """The Frisson Vision finish on any frames: learned grade, diffusion, halation, contrast,
    gate vignette, grain (settings from a style pack)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "pack": (casting.list_packs(), {"default": "crowd"}),
                             "amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images, pack, amount):
        _reload()
        pk = casting.load_pack(pack)
        dev = images.device if images.is_cuda else ("cuda" if torch.cuda.is_available() else "cpu")
        frames = [images[i].to(dev) for i in range(images.shape[0])]
        fin = finish.Finisher(pk["finish"], finish.lab_stats(frames[:: max(1, len(frames) // 16)]))
        out = [(fin(f, i) * amount + f * (1 - amount)).cpu() for i, f in enumerate(frames)]
        return (torch.stack(out),)


class FVOffloadToCPU:
    """Keeps a model's weight backups (made when LoRAs are patched in) in system RAM instead of
    VRAM. With --highvram ComfyUI parks them on the GPU, which doubles a 22B LTX model's
    footprint once LoRAs are applied; on a 96 GB card that spills into shared memory and
    sampling runs ~30x slower."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"
    CATEGORY = CATEGORY + "/io"

    def run(self, model):
        m = model.clone()
        m.offload_device = torch.device("cpu")
        return (m,)


class FVFreeMemory:
    """Unloads every model and clears ComfyUI's cache after this job (asks the server through its
    own /free endpoint). The Director queues one between model groups so RAM stays bounded."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"note": ("STRING", {"default": "free models between groups"})}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY + "/io"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return random.random()

    def run(self, note):
        import gc
        import json as _json
        import urllib.request
        try:
            req = urllib.request.Request(director.server_url() + "/free",
                                         data=_json.dumps({"unload_models": True, "free_memory": True}).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
        except Exception as e:
            print(f"[FrissonVision] /free failed: {e}")
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception:
            pass
        gc.collect()
        return {"ui": {"text": ["models unloaded"]}}


NODE_CLASS_MAPPINGS = {
    "FVFreeMemory": FVFreeMemory,
    "FVOffloadToCPU": FVOffloadToCPU,
    "FVDirector": FVDirector,
    "FVAssemble": FVAssemble,
    "FVFinish": FVFinish,
    "FVQC": FVQC,
    "FVLoadImagePath": FVLoadImagePath,
    "FVLoadVideoPath": FVLoadVideoPath,
    "FVLoadAudioPath": FVLoadAudioPath,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FVDirector": "Frisson Vision: Director",
    "FVFreeMemory": "Frisson Vision: Free Memory",
    "FVOffloadToCPU": "Frisson Vision: Weight Backups to RAM",
    "FVAssemble": "Frisson Vision: Assemble",
    "FVFinish": "Frisson Vision: Finish",
    "FVQC": "Frisson Vision: Take QC",
    "FVLoadImagePath": "Frisson Vision: Load Image (path)",
    "FVLoadVideoPath": "Frisson Vision: Load Video (path)",
    "FVLoadAudioPath": "Frisson Vision: Load Audio (path)",
}
