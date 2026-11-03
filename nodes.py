"""OneShot MV ComfyUI nodes."""
from __future__ import annotations

import importlib
import os
import random

import numpy as np
import torch

import folder_paths

from . import fv as _fvpkg  # noqa: F401
from .fv import assemble as fv_assemble
from .fv import audiofx, beat, casting, director, export, faces, finish, formats, fx, graphs, lipsync, plan, qc, stems
from .fv import setup_check
from .fv import theme as fv_theme
from .fv import broll as fv_broll
from .fv import identity as fv_identity
from .fv import critic as fv_critic
from .fv import world_images as fv_world_images

CATEGORY = "OneShot MV"
_MODULES = [beat, formats, casting, faces, plan, graphs, setup_check, fv_theme, fv_broll, fv_identity, fv_critic,
            fv_world_images, stems, audiofx, fx, finish, export, lipsync, qc, fv_assemble, director]


def _reload():
    """Pick up edits to the core without restarting ComfyUI (cheap; modules are small)."""
    if os.environ.get("FV_NO_RELOAD"):
        return
    for m in _MODULES:
        try:
            importlib.reload(m)
        except Exception as e:  # keep the old module if an edit is broken
            print(f"[OneShotMV] reload {m.__name__} failed: {e}")


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
    d = os.path.join(folder_paths.get_output_directory(), "oneshot-mv", "inputs")
    os.makedirs(d, exist_ok=True)
    return d


def _save_image_input(image) -> str:
    """IMAGE tensor -> PNG under output/oneshot-mv/inputs (named by content, so reruns reuse it)."""
    import hashlib
    from PIL import Image
    arr = (image[0].detach().cpu().float().clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)
    path = os.path.join(_inputs_dir(), f"artist_{hashlib.sha1(arr.tobytes()).hexdigest()[:12]}.png")
    if not os.path.exists(path):
        Image.fromarray(arr).save(path)
    return path


class FVLoadImagePath:
    """Load an image from a path. Unlike Load Image it does not check the file when the job is
    queued, so it can read a file an earlier job in the same queue will write."""
    DESCRIPTION = "Loads an image from a path (for the Director's jobs)."

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
    DESCRIPTION = "Loads a video from a path (for the Director's jobs)."

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
    DESCRIPTION = "Loads audio from a path (for the Director's jobs)."

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
    """OneShot MV in one press: a song and a World line in, a music video out. Analyses the song,
    writes the World's cast, plans the cut and queues every render plus the final assembly."""
    DESCRIPTION = ("A song and a World line (or a folder of your images) in, a lip-synced music video out: "
                   "plans, casts and queues the whole run.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "world": ("STRING", {"default": "", "multiline": True,
                      "tooltip": "The World in a line or two, e.g. 'a 1965 airline: pilots, stewardesses and "
                                 "passengers, warm cabin light'. It sets the people, what they wear, the places, "
                                 "the light and the grade. People you describe keep your words, and rules like "
                                 "'no men', 'present day' or 'no people, all robots' are kept. Empty: invented "
                                 "people in close-ups."}),
            "world_images": ("STRING", {"default": "",
                             "tooltip": "Instead of a World line: a folder of your own images (a path on this "
                                        "computer, or a folder inside ComfyUI's input or output folder). Images "
                                        "with a face sing, the rest become cutaways. The World line and cast are "
                                        "then not used."}),
            "image_order": (["in order", "random order"], {"default": "in order",
                            "tooltip": "World images: by file name, or shuffled. Images repeat when there are more "
                                       "shots than images."}),
            "image_fit": (list(fv_world_images.FITS), {"default": "crop",
                          "tooltip": "World images that aren't the video's shape. crop: cut around the face. "
                                     "extend: keep the whole image while Qwen-Image 2.1 draws the missing sides (its "
                                     "licence is reported to be research-only). With professional on, always crop."}),
            "cast": (["crowd", "single artist"], {"default": "crowd",
                     "tooltip": "crowd: a different invented person sings each line. single artist: one person "
                                "sings every line, in the World's outfits and places, from a photo (artist_photo) "
                                "or a description (artist_description)."}),
            "seconds": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 600.0, "step": 0.5,
                                  "tooltip": "Length of the video; 0 = the whole song"}),
            "start": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 3600.0, "step": 0.01,
                                "tooltip": "Song time to start at; -1 = pick the strongest stretch"}),
            "format": (formats.FORMAT_NAMES, {"default": "16:9",
                       "tooltip": "16:9 (1920×1080), 9:16 for phones (1080×1920) or 1:1 (1080×1080)"}),
            "quality": (formats.PRESET_NAMES, {"default": "quality",
                        "tooltip": "quality: 50 fps clips, the sharpest detail and the best lip sync. fast: "
                                   "smaller 25 fps clips in fewer steps, for checking the cast and the cut"}),
            "models": (["auto", "24gb", "full"], {"default": "auto",
                       "tooltip": "24gb: fp8 files for 24 GB GPUs. full: bf16 files for 48 GB or more. auto: full "
                                  "on a 48 GB+ GPU when its files are installed, otherwise 24gb."}),
            "professional": ("BOOLEAN", {"default": False,
                             "tooltip": "For paid or client work: Single Artist uses Qwen-Image-Edit 2511 "
                                        "(Apache-2.0) instead of Qwen-Image 2.1, whose licence is reported to be "
                                        "research-only (2511 matches faces a little less closely), and World images "
                                        "are always cropped. Check the other models' licences too (README, Models "
                                        "and licences)."}),
            "preview_only": ("BOOLEAN", {"default": False,
                             "tooltip": "On: plan the video and list every shot in the summary, rendering nothing. "
                                        "Turn it off and queue again to render; the song analysis and the World are "
                                        "cached."}),
        }, "optional": {
            "seed": ("INT", {"default": 1, "min": 0, "max": 2 ** 31 - 1,
                     "tooltip": "Same seed, same cast and shots; change it for new ones"}),
            "cuts_low": (plan.CUT_CHOICES, {"default": plan.CUT_DEFAULTS["low"],
                         "tooltip": "How low-energy sections (intro, breaks, a quiet outro) are cut"}),
            "cuts_medium": (plan.CUT_CHOICES, {"default": plan.CUT_DEFAULTS["medium"],
                            "tooltip": "How medium-energy sections (verses, the build into a drop) are cut"}),
            "cuts_high": (plan.CUT_CHOICES, {"default": plan.CUT_DEFAULTS["high"],
                          "tooltip": "How high-energy sections (choruses, drops) are cut. 'To the music' cuts "
                                     "on the beat to a new face (Single Artist: a new look) in the same spot: subtle "
                                     "about every half bar, intense about every beat"}),
            "audio_fx": (["none", "subtle", "intense"], {"default": "subtle",
                         "tooltip": "Effects on the drum hits: flash, zoom punch, glow, color split. subtle: "
                                    "gentle. intense: stronger, with a shake on the big hits. none: off (the color "
                                    "grade and grain stay)"}),
            "crowd": (["auto", "on", "off"], {"default": "auto",
                      "tooltip": "People behind the singers. auto: a soft-focus crowd when the World is a packed "
                                 "spectacle (arena, stadium, crowded stands), otherwise nobody else in frame "
                                 "(Single Artist: only with on). on / off: always / never"}),
            "cast_order": (["as written", "mixed"], {"default": "as written",
                           "tooltip": "as written: the characters the World line names are cast first, in the "
                                      "order it names them (write who leads first); mixed: shuffled"}),
            "broll_share": ("FLOAT", {"default": 0.125, "min": 0.0, "max": 0.5, "step": 0.025,
                            "tooltip": "Share of shots that become cutaways with nobody singing: first where lip "
                                       "sync stayed weak, then in instrumental gaps, never on the drop. 0 = none"}),
            "qc_retries": ("INT", {"default": 2, "min": 0, "max": 3,
                           "tooltip": "Check every take (missing, frozen, blown out, dead mouth, drifting off the "
                                      "shot, and lip sync if the SyncNet weights are installed) and shoot bad ones "
                                      "again, up to this many rounds; at most doubles the render time. 0 = off, which "
                                      "also turns off the cutaways, the vision check and review_images"}),
            "run_name": ("STRING", {"default": "", "tooltip": "Folder name for the run; empty = made from the settings"}),
            "artist_photo": ("IMAGE", {"tooltip": "Single artist: a photo of the singer, only of yourself or of "
                                                  "someone who agreed to it. Best: one sharp photo of just them, "
                                                  "facing the camera in soft even light, the whole face showing"}),
            "artist_description": ("STRING", {"default": "", "multiline": True,
                                   "tooltip": "Single artist without a photo: describe a new person, e.g. 'a singer "
                                              "in her late twenties with a shaved head and gold hoops'. They look "
                                              "the same in every shot"}),
            "review_images": ("BOOLEAN", {"default": False,
                              "tooltip": "Stop once every image is made, so you can check them. Then open the video "
                                         "in the Studio (the OneShot MV button, then Your videos) to fix any image "
                                         "and make the video. Needs qc_retries above 0."}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("summary", "run_dir")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kw):
        return random.random()

    def run(self, audio, world, cast, seconds, start, format, quality, models, preview_only, world_images="",
            image_order="in order", image_fit="crop", professional=False, seed=1, cuts_low=None, cuts_medium=None,
            cuts_high=None, audio_fx="subtle", crowd="auto", cast_order="as written", broll_share=0.125, qc_retries=2,
            run_name="", artist_photo=None, artist_description="", review_images=False):
        _reload()
        artist_path, described = "", ""
        if (world_images or "").strip():
            if cast == "single artist":
                print("[OneShotMV] World images: Single Artist is off; your images are the cast")
            cast = "crowd"  # the images are the cast
        if cast == "single artist":
            if artist_photo is not None:
                artist_path = _save_image_input(artist_photo)
            elif (artist_description or "").strip():
                described = artist_description.strip()
            else:
                raise ValueError("Single artist needs a photo or a description: un-bypass the Artist photo node "
                                 "(or connect a Load Image to artist_photo), or describe a new person in "
                                 "artist_description.")
        wf = audio["waveform"][0].detach().cpu().float().numpy()
        res = director.direct(audio=wf, sr=int(audio["sample_rate"]), out_root=folder_paths.get_output_directory(),
                              theme=world or "", fmt=format, preset=quality, seconds=seconds, start=start,
                              seed=seed, run_name=run_name, queue=not preview_only, qc_retries=qc_retries,
                              broll_share=broll_share,
                              cuts={"low": cuts_low, "medium": cuts_medium, "high": cuts_high},
                              audio_fx=audio_fx or "", cast_order=cast_order or "as written",
                              crowd=crowd or "auto", models=models or "auto",
                              artist_path=artist_path, artist_description=described,
                              world_images=(world_images or "").strip(), image_order=image_order or "in order",
                              image_fit=image_fit or "crop", professional=bool(professional),
                              review=bool(review_images), log=_stoppable_log())
        ui = {"text": [res["summary"]]}
        ui["text"].append(_file_ref(res["bundle"]))  # jobs.json as a job output (SDK clients)
        return {"ui": ui, "result": (res["summary"], res["run_dir"])}


def _stoppable_log():
    """The Director's log, which also stops it when the job is cancelled (the Studio's Cancel, or
    ComfyUI's): every step logs a line, and once ComfyUI's interrupt flag is up each later line raises
    again, so a step that catches errors to carry on can't swallow the stop. The flag is left up for
    ComfyUI to clear when the next prompt starts."""
    import comfy.model_management as mm
    stopped = [False]

    def log(*a, **k):
        print(*a, **k)
        flag = getattr(mm, "processing_interrupted", None)
        if stopped[0] or (flag() if flag else False):
            stopped[0] = True
            raise mm.InterruptProcessingException()
    return log


class FVAssemble:
    """Cuts a OneShot MV run together on the beat, with audio-reactive FX and the finish.
    Queued automatically as the last job of a run; queue it again to re-cut."""
    DESCRIPTION = "Cuts a run together with FX and the finish. Run it again to re-cut; nothing re-renders."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"run_dir": ("STRING", {"default": "",
                                         "tooltip": "The run folder, e.g. oneshot-mv/runs/<run> (the Director "
                                                    "prints it)"}),
                             "filename_prefix": ("STRING", {"default": "oneshot-mv/final",
                                                 "tooltip": "Not used for the file name: the new cut replaces "
                                                            "final_<format>.mp4 in the run folder"})},
                "optional": {"fx_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                                              "tooltip": "Scales every effect: 0 none, 1 normal, up to 3"}),
                             "audio_fx": (["from the run", "none", "subtle", "intense"], {"default": "from the run",
                                          "tooltip": "Cut again with another audio-reactive FX level"}),
                             "finish": ("BOOLEAN", {"default": True,
                                        "tooltip": "The color grade, grain and vignette; off keeps the clips' own "
                                                   "colors"})}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview", "video_path")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kw):
        return random.random()

    def run(self, run_dir, filename_prefix, fx_intensity=1.0, finish=True, audio_fx="from the run"):
        _reload()
        import comfy.utils
        pbar = comfy.utils.ProgressBar(100)
        res = fv_assemble.assemble(_resolve(run_dir), fx_intensity=fx_intensity, do_finish=finish,
                                   fx_level=None if audio_fx == "from the run" else audio_fx,
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
    DESCRIPTION = ("Checks every take, lip sync included, shoots bad ones again, then queues the cut (the "
                   "Director queues it).")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"run_dir": ("STRING", {"default": ""}),
                             "pass_index": ("INT", {"default": 0, "min": 0, "max": 9}),
                             "max_passes": ("INT", {"default": 1, "min": 0, "max": 9})},
                "optional": {"then_assemble": ("BOOLEAN", {"default": True}),
                             "queue_rerolls": ("BOOLEAN", {"default": True,
                                               "tooltip": "off = return the re-roll jobs (qcN.json) for an "
                                                          "SDK client to submit"}),
                             "stage": (["all", "leads", "looks"], {"default": "all",
                                       "tooltip": "all = every take at once; leads / looks = the Director's "
                                                  "staged order: singer shots first, then the looks made from "
                                                  "the final singer takes"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kw):
        return random.random()

    def run(self, run_dir, pass_index, max_passes, then_assemble=True, queue_rerolls=True, stage="all"):
        _reload()
        res = director.qc_pass(_resolve(run_dir), pass_idx=pass_index, max_passes=max_passes,
                               then_assemble=then_assemble, queue_rerolls=queue_rerolls, stage=stage)
        return {"ui": {"text": [res["message"], _file_ref(res["bundle"])]}, "result": (res["message"],)}


def _video_ref(path: str) -> list:
    """An MP4 under the output folder as a ComfyUI 'gifs' ui entry (UI preview, API / SDK download)."""
    try:
        rel = os.path.relpath(path, folder_paths.get_output_directory())
    except ValueError:
        return []
    if rel.startswith(".."):
        return []
    sub, name = os.path.split(rel)
    return [{"filename": name, "subfolder": sub.replace(os.sep, "/"), "type": "output", "format": "video/h264-mp4"}]


class FVRunResult:
    """Where a run is, and its final video once it is finished, as job outputs. For API and
    Comfy SDK clients (sdk_run.py): queued after a Director, it runs once the run's own stages
    (which jump the queue) are done."""
    DESCRIPTION = "Where a run is, and its finished video once it's done (the SDK uses it)."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"run_dir": ("STRING", {"default": "",
                                         "tooltip": "The run folder (the Director's run_dir output)"})}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "video_path")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kw):
        return random.random()

    def run(self, run_dir):
        _reload()
        import json
        st = director.run_status(_resolve(run_dir))
        if st["state"] == "rendering":
            try:  # nothing else queued and no video: one of the run's stages failed
                from server import PromptServer
                _running, pending = PromptServer.instance.prompt_queue.get_current_queue()
                if not pending:
                    st["state"] = "stopped"
                    st["message"] += ("; nothing is left in the queue, so the run stopped early "
                                      "(the ComfyUI console shows the error)")
            except Exception:
                pass
        ui = {"text": [st["message"]]}
        if st.get("run_dir"):
            sp = os.path.join(st["run_dir"], "status.json")
            json.dump(st, open(sp, "w", encoding="utf-8"), indent=1)
            ui["text"].append(_file_ref(sp))
        if st["state"] == "done":
            ui["gifs"] = _video_ref(st["video"])
        return {"ui": ui, "result": (st["message"], st.get("video", ""))}


class FVCritic:
    """The vision check (fv/critic.py): Gemma 3 looks at every portrait and b-roll still of a run
    against what its character should be (gender, facial hair, glasses, other people, the head in
    frame, writing, anything from after the World's period, the outfit). Bad ones are re-made with
    new seeds and a line that says what to fix, and checked again; then the clips are queued."""
    DESCRIPTION = "Checks every image of a run and makes the wrong ones again (the Director queues it)."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"run_dir": ("STRING", {"default": ""}),
                             "pass_index": ("INT", {"default": 0, "min": 0, "max": 9}),
                             "max_passes": ("INT", {"default": 2, "min": 0, "max": 9})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kw):
        return random.random()

    def run(self, run_dir, pass_index, max_passes):
        _reload()
        res = director.critic_pass(_resolve(run_dir), pass_idx=pass_index, max_passes=max_passes)
        return {"ui": {"text": [res["message"]]}, "result": (res["message"],)}


class FVLipSyncScore:
    """Scores a run's lip sync with SyncNet: per take, the confidence (higher = the mouth follows
    the words) and the offset (25 fps frames; negative = the mouth leads the voice). Writes
    <run>/lipsync.json; the run itself is not changed. Needs the SyncNet weights (see README)."""
    DESCRIPTION = "Scores a finished run's lip sync with SyncNet, take by take."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"run_dir": ("STRING", {"default": ""}),
                             "which": (["lead clips", "all clips"], {"default": "lead clips"}),
                             "every_take": ("BOOLEAN", {"default": False,
                                            "tooltip": "also score the re-rolled takes QC kept aside"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kw):
        return random.random()

    def run(self, run_dir, which, every_take):
        _reload()
        if not lipsync.available():
            msg = ("SyncNet weights not found: put syncnet_v2.model and sfd_face.pth in "
                   "ComfyUI/models/syncnet/ (see the OneShot MV README)")
            return {"ui": {"text": [msg]}, "result": (msg,)}
        res = lipsync.score_run(_resolve(run_dir), leads_only=(which == "lead clips"), every_take=every_take)
        return {"ui": {"text": [res["message"]]}, "result": (res["message"],)}


class FVFaceMatch:
    """Moves and scales an edited keyframe so its face sits where the reference frame's face is.
    Image edits asked for a new outfit tend to pull the camera back; a pose-matched look made
    from such a keyframe stays smaller and higher than the lead, so the cut doesn't match."""
    DESCRIPTION = "Puts a look's face where its lead's face is (Single Artist's looks)."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "reference": ("IMAGE",),
                             "max_scale": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 4.0, "step": 0.05})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, reference, max_scale=2.2):
        _reload()
        return (faces.face_match(image, reference, max_scale),)


class FVFinish:
    """The OneShot MV finish on any frames: learned grade, diffusion, halation, contrast,
    gate vignette, grain (the settings every run uses)."""
    DESCRIPTION = "Applies the finish (color grade, grain, vignette) to any frames."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",),
                             "amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images, amount):
        _reload()
        pk = casting.load_pack("crowd")
        dev = images.device if images.is_cuda else ("cuda" if torch.cuda.is_available() else "cpu")
        frames = [images[i].to(dev) for i in range(images.shape[0])]
        fin = finish.Finisher(pk["finish"], finish.lab_stats(frames[:: max(1, len(frames) // 16)]))
        out = [(fin(f, i) * amount + f * (1 - amount)).cpu() for i, f in enumerate(frames)]
        return (torch.stack(out),)


class FVFitAspect:
    """An image brought to another shape without cropping: the short side is extended with its own
    edge pixels (Single Artist with Qwen-Image 2.1, whose output takes image 1's shape: a tall phone
    photo would otherwise be centre-cropped to 16:9 and lose the head)."""
    DESCRIPTION = "Brings an image to the video's shape without cropping, so a tall photo keeps its head."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "width": ("INT", {"default": 1344, "min": 16, "max": 8192}),
                             "height": ("INT", {"default": 768, "min": 16, "max": 8192})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY + "/io"

    def run(self, image, width, height):
        _, h, w, _ = image.shape
        want = width / height
        if abs(w / h - want) < 0.01:
            return (image,)
        if w / h < want:  # too tall: extend left and right
            add = int(round(h * want)) - w
            pad = (add // 2, add - add // 2, 0, 0)
        else:  # too wide: extend top and bottom
            add = int(round(w / want)) - h
            pad = (0, 0, add // 2, add - add // 2)
        x = torch.nn.functional.pad(image.permute(0, 3, 1, 2), pad, mode="replicate")
        return (x.permute(0, 2, 3, 1),)


class FVExtendCanvas:
    """World images with fit "extend": an image on a canvas of the video's shape. The image is fitted
    inside width x height and centred, the new edges are plain grey; `fill` marks the new edges (reaching
    a little into the image) and `keep` is the image with a soft edge, to paste the original pixels back
    over the redrawn canvas."""
    DESCRIPTION = ("Puts an image on a canvas of the video's shape, with masks to paste the original back (World "
                   "images with Extend).")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "width": ("INT", {"default": 2016, "min": 16, "max": 8192}),
                             "height": ("INT", {"default": 1152, "min": 16, "max": 8192})}}

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("canvas", "fill", "keep")
    FUNCTION = "run"
    CATEGORY = CATEGORY + "/io"

    def run(self, image, width, height):
        return fv_world_images.extend_canvas(image[:1], width, height)


class FVOffloadToCPU:
    """Keeps a model's weight backups (made when LoRAs are patched in) in system RAM instead of
    VRAM. With --highvram ComfyUI parks them on the GPU, which doubles a 22B LTX model's
    footprint once LoRAs are applied; on a 96 GB card that spills into shared memory and
    sampling runs ~30x slower."""
    DESCRIPTION = ("Keeps LoRA weight backups in system RAM instead of on the GPU (a memory helper for the "
                   "Director's jobs).")

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
    DESCRIPTION = "Frees GPU memory between a run's stages (a memory helper for the Director's jobs)."

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
        try:  # the most GPU memory the stage before this used (to check a run fits a smaller card)
            import torch
            if torch.cuda.is_available():
                print(f"[OneShotMV] peak GPU memory before '{note}': "
                      f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB allocated, "
                      f"{torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB reserved")
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        try:
            req = urllib.request.Request(director.server_url() + "/free",
                                         data=_json.dumps({"unload_models": True, "free_memory": True}).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
        except Exception as e:
            print(f"[OneShotMV] /free failed: {e}")
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception:
            pass
        gc.collect()
        return {"ui": {"text": ["models unloaded"]}}


class FVKeep:
    """Keeps models loaded across jobs that don't use them. ComfyUI drops a cached model when a
    job without its loader runs (a QC pass, an image edit), and LTX-2.3 then reloads from disk
    (5-9 min on a slow drive). The Director wires LTX's loaders in here in those jobs: they
    come from the cache instantly, and the model stays loaded for the next video job."""
    DESCRIPTION = "Keeps a run's models loaded between jobs (a memory helper for the Director's jobs)."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {},
                "optional": {"model": ("MODEL",), "clip": ("CLIP",), "vae": ("VAE",), "audio_vae": ("VAE",),
                             "upscaler": ("LATENT_UPSCALE_MODEL",)}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY + "/io"

    def run(self, **kw):
        return {"ui": {"text": [f"kept {len(kw)} model(s) loaded"]}}


class FVTheme:
    """Preview what a World line turns into before a run: the characters, outfits, places, light,
    in-scene actions and colour grade the Director will cast from (Gemma 3, the LTX-2.3 text
    encoder). The result is cached, so a Director run with the same theme and seed reuses it."""
    DESCRIPTION = ("Shows what a World line turns into (characters, outfits, places, light, grade) before you "
                   "spend a run on it.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"theme": ("STRING", {"default": "", "multiline": True, "tooltip": "A World line"}),
                             "seed": ("INT", {"default": 1, "min": 0, "max": 2 ** 31 - 1,
                                      "tooltip": "Same seed, same World (cached)"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("world",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def run(self, theme, seed):
        _reload()
        b = fv_theme.expand(theme, seed=seed, out_root=folder_paths.get_output_directory())
        if not b:
            return {"ui": {"text": ["(no World line)"]}, "result": ("",)}
        lines = [f"World: {b['theme']}  (grade: {b.get('grade')})"]
        if b.get("camera"):
            lines.append(f"camera: {b['camera']}")
        chars = b.get("characters") or []
        if chars:  # whole characters, as the cast uses them
            lines.append("\ncharacters:")
            for c in chars:
                lines.append(f"  - {c['who']}")
                lines.append(f"      wearing {c['wear']}")
                for k, lab in (("place", ""), ("light", "light: "), ("doing", "wide shot: "), ("face", "close-up: ")):
                    if c.get(k):
                        lines.append(f"      {lab}{c[k]}")
        for k in (("places", "light", "actions") if chars else ("who", "wear", "places", "light", "actions")):
            lines.append(f"\n{k}:")
            lines += [f"  - {x}" for x in b.get(k, [])]
        text = "\n".join(lines)
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "FVFreeMemory": FVFreeMemory,
    "FVKeep": FVKeep,
    "FVOffloadToCPU": FVOffloadToCPU,
    "FVFitAspect": FVFitAspect,
    "FVExtendCanvas": FVExtendCanvas,
    "FVDirector": FVDirector,
    "FVAssemble": FVAssemble,
    "FVRunResult": FVRunResult,
    "FVFinish": FVFinish,
    "FVQC": FVQC,
    "FVCritic": FVCritic,
    "FVLipSyncScore": FVLipSyncScore,
    "FVFaceMatch": FVFaceMatch,
    "FVTheme": FVTheme,
    "FVLoadImagePath": FVLoadImagePath,
    "FVLoadVideoPath": FVLoadVideoPath,
    "FVLoadAudioPath": FVLoadAudioPath,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FVDirector": "OneShot MV: Director",
    "FVFreeMemory": "OneShot MV: Free Memory",
    "FVKeep": "OneShot MV: Keep Models Loaded",
    "FVOffloadToCPU": "OneShot MV: Weight Backups to RAM",
    "FVFitAspect": "OneShot MV: Fit to Shape",
    "FVExtendCanvas": "OneShot MV: Extend Canvas",
    "FVAssemble": "OneShot MV: Assemble",
    "FVRunResult": "OneShot MV: Run Result",
    "FVFinish": "OneShot MV: Finish",
    "FVQC": "OneShot MV: Take QC",
    "FVCritic": "OneShot MV: Vision Check",
    "FVLoadImagePath": "OneShot MV: Load Image (path)",
    "FVLoadVideoPath": "OneShot MV: Load Video (path)",
    "FVLoadAudioPath": "OneShot MV: Load Audio (path)",
    "FVLipSyncScore": "OneShot MV: Lip Sync Score",
    "FVFaceMatch": "OneShot MV: Face Match",
    "FVTheme": "OneShot MV: World Preview",
}
