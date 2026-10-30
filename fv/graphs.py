"""ComfyUI API-format graph builders for every OneShot MV job.

All file inputs go through the pack's path loaders (FVLoadImagePath / FVLoadVideoPath /
FVLoadAudioPath), which do not check that the file exists when the job is queued, so the
Director can queue a whole run up front (later jobs read files earlier jobs will write)."""
from __future__ import annotations

import json
import os

from . import formats

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDERS = ("checkpoints", "diffusion_models", "unet", "loras", "text_encoders", "clip", "vae",
                 "latent_upscale_models")
MODEL_EXT = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf")


def _model_index():
    """(names ComfyUI lists, {file name: listed name}) across the model folders; None outside ComfyUI."""
    try:
        import folder_paths  # type: ignore
    except Exception:
        return None
    have, by_name = set(), {}
    for kind in MODEL_FOLDERS:
        if kind not in getattr(folder_paths, "folder_names_and_paths", {}):
            continue
        try:
            names = folder_paths.get_filename_list(kind)
        except Exception:
            continue
        for n in names:
            have.add(n)
            by_name.setdefault(os.path.basename(n.replace("\\", "/")).lower(), n)
    return have, by_name


def _resolve_models(m: dict, idx=None) -> tuple[dict, list]:
    """Each model file as this ComfyUI lists it, and the files it can't find. A name that isn't
    found as written is looked up by its file name in the model folders, so the files can sit in
    any sub-folder and the path separator doesn't matter. Names that aren't found are left as
    written (the job then says which file is missing)."""
    idx = idx if idx is not None else _model_index()
    if idx is None:
        return dict(m), []
    have, by_name = idx
    missing = []

    def fix(k, v):
        if not isinstance(v, str) or not v.lower().endswith(MODEL_EXT) or v in have:
            return v
        alt = v.replace("\\", os.sep).replace("/", os.sep)
        if alt in have:
            return alt
        hit = by_name.get(os.path.basename(v.replace("\\", "/")).lower())
        if hit is None:
            missing.append((k, os.path.basename(v.replace("\\", "/"))))
        return hit or v

    out = {k: [[fix(k, x[0]), *x[1:]] if isinstance(x, list) else fix(k, x) for x in v] if isinstance(v, list)
           else fix(k, v) for k, v in m.items()}
    return out, missing


# ---- model sets: "24gb" (models.json, fp8, the default) and "full" (models_full.json on top, bf16)
MODEL_SETS = ("24gb", "full")
FULL_MIN_GB = 46          # the full set wants a 48 GB card or bigger
RUN_KEYS = ("krea_turbo", "krea_raw", "krea_turbo_lora", "krea_te", "krea_vae", "ltx23_ckpt", "ltx23_te",
            "ltx23_upscaler", "ltx23_loras")  # what a default voice run loads


def _set_files(name: str) -> dict:
    m = json.load(open(os.path.join(HERE, "models.json"), encoding="utf-8"))
    if name == "full":
        m.update({k: v for k, v in json.load(open(os.path.join(HERE, "models_full.json"), encoding="utf-8")).items()
                  if not k.startswith("_")})
    return m


def reserved_gb() -> float:
    """GPU memory ComfyUI was told to leave alone with --reserve-vram (0 if not set)."""
    try:
        from comfy.cli_args import args  # type: ignore
        return float(args.reserve_vram or 0)
    except Exception:
        return 0.0


def vram_gb() -> float:
    """The GPU memory ComfyUI may use in GB: the card's total minus any --reserve-vram (0 without CUDA)."""
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2 ** 30
            return max(total - reserved_gb(), 0.0)
    except Exception:
        pass
    return 0.0


def missing_files(name: str, idx=None) -> list:
    """The files of a model set that a default run needs and this ComfyUI doesn't have."""
    m = {k: v for k, v in _set_files(name).items() if k in RUN_KEYS}
    if m.get("krea_raw") == m.get("krea_turbo"):
        m.pop("krea_turbo_lora", None)  # one Krea file for both passes: the Turbo LoRA isn't used
    return sorted(set(f for _, f in _resolve_models(m, idx)[1]))


def pick_models(requested: str | None = "auto") -> str:
    """The model set for a run: the requested one, or with "auto" the full set on cards with 48 GB or
    more when its files are installed, otherwise the 24 GB set."""
    if requested in MODEL_SETS:
        return requested
    idx = _model_index()
    big = vram_gb() >= FULL_MIN_GB
    if big and not missing_files("full", idx):
        return "full"
    if missing_files("24gb", idx) and big:
        return "full"  # neither set complete: the one this card can run
    return "24gb"


def models_report() -> dict:
    """For the Studio: the GPU's memory, the set "auto" picks, and the missing files of each set."""
    idx = _model_index()
    return {"vram_gb": round(vram_gb(), 1), "reserved_gb": round(reserved_gb(), 1),
            "auto": pick_models("auto"), "full_min_gb": FULL_MIN_GB,
            "missing": {n: missing_files(n, idx) for n in MODEL_SETS}}


def use_models(name: str | None = "auto") -> str:
    """Point M at a model set (in place, so every module that holds graphs.M sees it)."""
    name = pick_models(name)
    resolved, _ = _resolve_models(_set_files(name))
    M.clear()
    M.update(resolved)
    M["_set"] = name
    return name


def _have_sage() -> bool:
    """SageAttention is optional: without the package the LTX jobs skip the patch."""
    try:
        import importlib.util
        return importlib.util.find_spec("sageattention") is not None
    except Exception:
        return False


M: dict = {}
use_models("auto")

R8_S2 = "0.909, 0.880, 0.840, 0.780, 0.700, 0.600, 0.500, 0.400, 0.250, 0.100, 0.0"
R8_S3 = "0.909375, 0.725, 0.421875, 0.0"
D25_S1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
D25_S2 = "0.909375, 0.725, 0.421875, 0.0"


class Graph:
    def __init__(self):
        self.nodes, self._n = {}, 0

    def add(self, class_type, **inputs):
        self._n += 1
        self.nodes[str(self._n)] = {"class_type": class_type, "inputs": inputs}
        return str(self._n)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.nodes, f, indent=1)
        return path


def O(nid, slot=0):
    return [nid, slot]


def _save_video(g, images, fps, prefix, audio=None):
    kw = {"images": images, "fps": float(fps)}
    if audio is not None:
        kw["audio"] = audio
    v = g.add("CreateVideo", **kw)
    g.add("SaveVideo", video=O(v), filename_prefix=prefix,
          **{"format": "mp4", "format.codec": "auto"})


# ------------------------------------------------------------------ Krea 2 W5
def krea2_portrait(*, prompt, seed, prefix, size1, size2, denoise=0.25, turbo_only=False):
    """Krea 2 W5 hybrid (Turbo composes, RAW + Turbo LoRA redetails). turbo_only=True uses the
    Turbo model for both passes: one model to load instead of two (the fast preset).
    denoise: how hard the RAW pass redetails the upscaled image; 0.40 looks too gritty, so 0.25
    keeps Turbo's cleaner composition and adds less texture. A set that names the same file for
    Turbo and RAW also runs Turbo for both passes (the Turbo LoRA is only for the RAW model)."""
    g = Graph()
    turbo = g.add("UNETLoader", unet_name=M["krea_turbo"], weight_dtype="default")
    if turbo_only or M["krea_raw"] == M["krea_turbo"]:
        lora = turbo
    else:
        raw = g.add("UNETLoader", unet_name=M["krea_raw"], weight_dtype="default")
        lora = g.add("LoraLoaderModelOnly", model=O(raw), lora_name=M["krea_turbo_lora"], strength_model=0.6)
    clip = g.add("CLIPLoader", clip_name=M["krea_te"], type="krea2", device="default")
    vae = g.add("VAELoader", vae_name=M["krea_vae"])
    pos = g.add("CLIPTextEncode", clip=O(clip), text=prompt)
    neg = g.add("ConditioningZeroOut", conditioning=O(pos))
    lat = g.add("EmptyLatentImage", width=size1[0], height=size1[1], batch_size=1)
    ks = g.add("KSampler", model=O(turbo), seed=seed, steps=8, cfg=1.0, sampler_name="euler",
               scheduler="simple", positive=O(pos), negative=O(neg), latent_image=O(lat), denoise=1.0)
    dec = g.add("VAEDecode", samples=O(ks), vae=O(vae))
    up = g.add("ImageScale", image=O(dec), upscale_method="lanczos", width=size2[0], height=size2[1],
               crop="disabled")
    enc = g.add("VAEEncode", pixels=O(up), vae=O(vae))
    cs = g.add("ClownsharKSampler_Beta", eta=0.5, sampler_name="multistep/deis_2m", scheduler="beta",
               steps=8, steps_to_run=-1, denoise=denoise, cfg=1.0, seed=seed, sampler_mode="standard",
               bongmath=True, model=O(lora), positive=O(pos), negative=O(neg), latent_image=O(enc))
    dec2 = g.add("VAEDecode", samples=O(cs, 0), vae=O(vae))
    g.add("SaveImage", images=O(dec2), filename_prefix=prefix)
    return g


def artist_portrait(*, artist_path, prompt, seed, prefix, size2, resolution=1024):
    """One-artist mode: the artist photo re-staged by Qwen-Image 2.1 (identity kept)."""
    g = Graph()
    ref = g.add("FVLoadImagePath", path=artist_path)
    out = _qwen21(g, O(ref), prompt, seed=seed, resolution=resolution)
    out = O(g.add("ImageScale", image=out, upscale_method="lanczos", width=size2[0], height=size2[1], crop="center"))
    g.add("SaveImage", images=out, filename_prefix=prefix)
    return g


# ------------------------------------------------------------------ LTX models
def _has_node(name: str) -> bool:
    """Whether the server this graph is built on has a node class (True when not inside ComfyUI)."""
    try:
        import nodes as _comfy_nodes
        return name in _comfy_nodes.NODE_CLASS_MAPPINGS
    except Exception:
        return True


def _ram_backups(g, model):
    """LoRA weight backups to system RAM (see FVOffloadToCPU); skipped on servers without it."""
    if not _has_node("FVOffloadToCPU"):
        return model
    return O(g.add("FVOffloadToCPU", model=model))


def _ltx23(g, union=False):
    ck = g.add("CheckpointLoaderSimple", ckpt_name=M["ltx23_ckpt"])
    model = _ram_backups(g, O(ck, 0))
    if M.get("ltx23_sage") and _have_sage():
        model = O(g.add("PathchSageAttentionKJ", model=model, sage_attention=M["ltx23_sage"],
                        allow_compile=False))
    for name, st in M["ltx23_loras"]:
        model = O(g.add("LoraLoaderModelOnly", model=model, lora_name=name, strength_model=st))
    if M.get("ltx23_av_tuner"):
        vs, as_, a2v, v2a = M["ltx23_av_tuner"]
        model = O(g.add("LTX2AttentionTunerPatch", model=model, blocks="", video_scale=vs,
                        audio_scale=as_, audio_to_video_scale=a2v, video_to_audio_scale=v2a,
                        triton_kernels=False))
    dsf = 1.0
    if union:
        lo = g.add("LTXICLoRALoaderModelOnly", model=model, lora_name=M["ltx_union_lora"], strength_model=1.0)
        model, dsf = O(lo), O(lo, 1)
    avae = g.add("LTXVAudioVAELoader", ckpt_name=M["ltx23_ckpt"])
    clip = g.add("LTXAVTextEncoderLoader", text_encoder=M["ltx23_te"], ckpt_name=M["ltx23_ckpt"], device="default")
    ups = g.add("LatentUpscaleModelLoader", model_name=M["ltx23_upscaler"])
    return dict(model=model, dsf=dsf, vae=O(ck, 2), avae=O(avae), clip=O(clip), ups=O(ups))


def _ltx25(g, union=False):
    unet = g.add("UNETLoader", unet_name=M["ltx25_unet"], weight_dtype="default")
    model, dsf = _ram_backups(g, O(unet)), 1.0
    if union:
        lo = g.add("LTXICLoRALoaderModelOnly", model=model, lora_name=M["ltx_union_lora"], strength_model=1.0)
        model, dsf = O(lo), O(lo, 1)
    vae = g.add("VAELoader", vae_name=M["ltx25_vae"])
    avae = g.add("VAELoader", vae_name=M["ltx25_audio_vae"])
    clip = g.add("CLIPLoader", clip_name=M["ltx25_te"], type="ltxv", device="default")
    ups = g.add("LatentUpscaleModelLoader", model_name=M["ltx25_upscaler"])
    return dict(model=model, dsf=dsf, vae=O(vae), avae=O(avae), clip=O(clip), ups=O(ups))


def _pose(g, video_path, w, h, frames, fps, start_s=0.0, face=True, lead_only=False):
    """DWPose guide frames. lead_only: keep just the lead dancer (FVLeadPose) so backup dancers
    in the source clip don't turn into extra people."""
    src = g.add("FVLoadVideoPath", path=video_path, width=w, height=h, frame_cap=frames,
                start_seconds=float(start_s), force_fps=float(fps))
    dw = g.add("DWPreprocessor", image=O(src), detect_hand="enable", detect_body="enable",
               detect_face="enable" if face else "disable", resolution=512,
               bbox_detector=M["dwpose_bbox"], pose_estimator=M["dwpose_pose"],
               scale_stick_for_xinsr_cn="disable")
    if lead_only and _has_node("FVLeadPose"):
        return O(g.add("FVLeadPose", pose_keypoint=O(dw, 1), image=O(dw, 0)))
    return O(dw)


# LTX (the fast LTX-2.5 distilled model especially) sometimes cuts away mid-clip to another
# person or place, as if editing a music video. At cfg 1 a negative prompt can't stop it, so every
# clip says, in positive terms, that it is one unbroken take. QC's cut check catches the rest.
CONTINUITY = "One continuous take with no cuts: the same person in the same place, filmed by the same camera from the first frame to the last."


def ltx_render(*, preset, fmt, key_path, prompt, negative, frames, seed, prefix,
               audio_path=None, audio_seek=0.0, guide_video=None, guide_start=0.0,
               guide_strength=1.0, guide_face=True, save_pose=False, key_strengths=None,
               keyframes=None, continuity=True):
    """One LTX clip. audio_path -> lip sync on a frozen vocal latent (voice);
    guide_video -> DWPose union-control guide (twins, motion).
    keyframes: [(image_path, frame_idx, strength)] pinned at later frames of the clip, in every
    stage (a look group: one clip that switches person on the beat, each look's keyframe pinned
    at its cut frame; continuity=False, since the clip is meant to change person)."""
    pr = formats.PRESETS[preset]
    fps = pr["fps"]
    sizes = formats.stage_sizes(fmt, preset)
    g = Graph()
    union = guide_video is not None
    m = _ltx23(g, union) if pr["engine"] == "ltx23_r8" else _ltx25(g, union)
    if continuity and CONTINUITY not in prompt:
        prompt = f"{prompt.rstrip()} {CONTINUITY}"

    def pin(stage, pos_, neg_, lat_):
        for per, idx, st in extra:
            ag = g.add("LTXVAddGuide", positive=pos_, negative=neg_, vae=m["vae"], latent=lat_,
                       image=per[stage], frame_idx=idx, strength=st)
            pos_, neg_, lat_ = O(ag, 0), O(ag, 1), O(ag, 2)
        return pos_, neg_, lat_
    pos = g.add("CLIPTextEncode", clip=m["clip"], text=prompt)
    neg = g.add("CLIPTextEncode", clip=m["clip"], text=negative)
    cond = g.add("LTXVConditioning", positive=O(pos), negative=O(neg), frame_rate=float(fps))
    p_ref, n_ref = O(cond, 0), O(cond, 1)
    keys = []
    if key_path:  # image to video; without a key it is text to video (b-roll)
        key = g.add("FVLoadImagePath", path=key_path)
        for (w, h) in sizes:
            k = g.add("ImageScale", image=O(key), upscale_method="lanczos", width=w, height=h, crop="center")
            keys.append(O(g.add("LTXVPreprocess", image=O(k), img_compression=33)))
    extra = []  # per later keyframe: its image at each stage size (after the main key: the
    for path, idx, st in (keyframes or []):  # first image a job loads is its keyframe)
        im = g.add("FVLoadImagePath", path=path)
        per = []
        for (w, h) in sizes:
            k = g.add("ImageScale", image=O(im), upscale_method="lanczos", width=w, height=h, crop="center")
            per.append(O(g.add("LTXVPreprocess", image=O(k), img_compression=33)))
        extra.append((per, int(idx), float(st)))
    ks = key_strengths or ((0.7, 0.75, 0.8) if pr["stages"] == 3 else (0.9, 0.9))
    w1, h1 = sizes[0]
    lat = g.add("EmptyLTXVLatentVideo", width=w1, height=h1, length=frames, batch_size=1)
    lat_ref = O(g.add("LTXVImgToVideoInplace", vae=m["vae"], image=keys[0], latent=O(lat),
                      strength=ks[0], bypass=False)) if keys else O(lat)
    if union:
        ctl = _pose(g, guide_video, w1, h1, frames, fps, guide_start, guide_face, lead_only=not guide_face)
        gd = g.add("LTXAddVideoICLoRAGuide", positive=p_ref, negative=n_ref, vae=m["vae"], latent=lat_ref,
                   image=ctl, frame_idx=0, strength=guide_strength, latent_downscale_factor=m["dsf"],
                   crop="disabled", use_tiled_encode=False, tile_size=256, tile_overlap=64)
        p_ref, n_ref, lat_ref = O(gd, 0), O(gd, 1), O(gd, 2)
        if save_pose:
            _save_video(g, ctl, fps, prefix + "_pose")
    if extra:
        p_ref, n_ref, lat_ref = pin(0, p_ref, n_ref, lat_ref)
    au = None
    if audio_path:
        au = O(g.add("FVLoadAudioPath", path=audio_path, seek_seconds=float(audio_seek),
                     duration=round(frames / float(fps), 4)))
        alat = g.add("LTXVAudioVAEEncode", audio=au, audio_vae=m["avae"])
        mask = g.add("SolidMask", value=0.0, width=512, height=512)
        a_ref = O(g.add("SetLatentNoiseMask", samples=O(alat), mask=O(mask)))
    else:
        a_ref = O(g.add("LTXVEmptyLatentAudio", frames_number=frames, frame_rate=int(fps), batch_size=1,
                        audio_vae=m["avae"]))
    # stage 1
    av = g.add("LTXVConcatAVLatent", video_latent=lat_ref, audio_latent=a_ref)
    guider = g.add("CFGGuider", model=m["model"], positive=p_ref, negative=n_ref, cfg=1.0)
    if pr["engine"] == "ltx23_r8":
        sig = g.add("LTXVScheduler", steps=int(pr.get("steps1", 30)), max_shift=2.05, base_shift=0.95, stretch=True,
                    terminal=0.1, latent=O(av))
        smp = g.add("KSamplerSelect", sampler_name="res_2s")
    else:
        sig = g.add("ManualSigmas", sigmas=D25_S1)
        smp = g.add("KSamplerSelect", sampler_name="euler_ancestral")
    s = g.add("SamplerCustomAdvanced", noise=O(g.add("RandomNoise", noise_seed=seed)), guider=O(guider),
              sampler=O(smp), sigmas=O(sig), latent_image=O(av))
    sep = g.add("LTXVSeparateAVLatent", av_latent=O(s))
    crop = g.add("LTXVCropGuides", positive=p_ref, negative=n_ref, latent=O(sep, 0))
    vlat, cpos, cneg = O(crop, 2), O(crop, 0), O(crop, 1)
    plan = ([("dpmpp_sde", R8_S2), ("dpmpp_2m_sde", R8_S3)] if pr["engine"] == "ltx23_r8"
            else [("euler_ancestral", D25_S2)])
    for i, (sname, sigs) in enumerate(plan, start=1):
        up = g.add("LTXVLatentUpsampler", samples=vlat, upscale_model=m["ups"], vae=m["vae"])
        li = O(g.add("LTXVImgToVideoInplace", vae=m["vae"], image=keys[i], latent=O(up), strength=ks[i],
                     bypass=False)) if keys else O(up)
        spos, sneg = cpos, cneg
        if extra:  # pin the later keyframes again at this stage's size, crop them after
            spos, sneg, li = pin(i, cpos, cneg, li)
        av = g.add("LTXVConcatAVLatent", video_latent=li, audio_latent=a_ref if audio_path else O(sep, 1))
        gd2 = g.add("CFGGuider", model=m["model"], positive=spos, negative=sneg, cfg=1.0)
        s = g.add("SamplerCustomAdvanced", noise=O(g.add("RandomNoise", noise_seed=12345 + seed % 1000)),
                  guider=O(gd2), sampler=O(g.add("KSamplerSelect", sampler_name=sname)),
                  sigmas=O(g.add("ManualSigmas", sigmas=sigs)), latent_image=O(av))
        sep = g.add("LTXVSeparateAVLatent", av_latent=O(s))
        vlat = O(sep, 0)
        if extra:
            cr = g.add("LTXVCropGuides", positive=spos, negative=sneg, latent=vlat)
            vlat, cpos, cneg = O(cr, 2), O(cr, 0), O(cr, 1)
    # the whole clip in one temporal tile: 64-frame tiles leave a soft frame at every seam
    # (frames 56, 112... of a 50 fps clip); spatial tiles keep the memory in check
    t_size = min(512, (frames + 7) // 8 * 8 + 8)
    dec = g.add("VAEDecodeTiled", samples=vlat, vae=m["vae"], tile_size=512, overlap=64,
                temporal_size=t_size, temporal_overlap=8)
    _save_video(g, O(dec), fps, prefix, audio=au)
    return g


# ------------------------------------------------------------------ image edits
def _klein(g, image_ref, prompt, refs=(), seed=7, steps=4):
    unet = g.add("UNETLoader", unet_name=M["klein_unet"], weight_dtype="default")
    clip = g.add("CLIPLoader", clip_name=M["klein_te"], type="flux2", device="default")
    vae = g.add("VAELoader", vae_name=M["klein_vae"])
    size = g.add("GetImageSize", image=image_ref)
    pos = g.add("CLIPTextEncode", clip=O(clip), text=prompt)
    neg = g.add("ConditioningZeroOut", conditioning=O(pos))
    p_ref, n_ref = O(pos), O(neg)
    for im in [image_ref] + list(refs):
        enc = g.add("VAEEncode", pixels=im, vae=O(vae))
        p_ref = O(g.add("ReferenceLatent", conditioning=p_ref, latent=O(enc)))
        n_ref = O(g.add("ReferenceLatent", conditioning=n_ref, latent=O(enc)))
    guider = g.add("CFGGuider", model=O(unet), positive=p_ref, negative=n_ref, cfg=1.0)
    sched = g.add("Flux2Scheduler", steps=steps, width=O(size, 0), height=O(size, 1))
    lat = g.add("EmptyFlux2LatentImage", width=O(size, 0), height=O(size, 1), batch_size=1)
    smp = g.add("SamplerCustomAdvanced", noise=O(g.add("RandomNoise", noise_seed=seed)), guider=O(guider),
                sampler=O(g.add("KSamplerSelect", sampler_name="euler")), sigmas=O(sched), latent_image=O(lat))
    return O(g.add("VAEDecode", samples=O(smp), vae=O(vae)))


def _qwen21(g, image_ref, prompt, refs=(), seed=7, steps=25, resolution=1024):
    unet = g.add("UNETLoader", unet_name=M["qwen21_unet"], weight_dtype="default")
    cache = g.add("QwenImage21Cache", model=O(unet), device="auto", dtype="default")
    clip = g.add("CLIPLoader", clip_name=M["qwen21_te"], type="qwen_image", device="default")
    vae = g.add("VAELoader", vae_name=M["qwen21_vae"])
    imgs = {"images.image_1": image_ref}
    for i, r in enumerate(refs):
        imgs[f"images.image_{i + 2}"] = r
    enc = g.add("TextEncodeQwenImage21", clip=O(clip), prompt=prompt, negative_prompt="",
                resolution=resolution, vae=O(vae), **imgs)
    ks = g.add("KSampler", model=O(cache), seed=seed, steps=steps, cfg=1.0, sampler_name="euler",
               scheduler="simple", positive=O(enc, 0), negative=O(enc, 1), latent_image=O(enc, 2), denoise=1.0)
    return O(g.add("VAEDecode", samples=O(ks), vae=O(vae)))


TWIN_PROMPT = ("Recreate image 1 with the person from image 2. Keep image 1's exact camera angle, framing, "
               "head position, head tilt, mouth shape and facial expression. Use image 2's identity, face, "
               "hair, skin, makeup, jewelry, clothing, lighting colour and background. Photorealistic.")


TWIN_IDENTITY = ("This is the same person as in image 2: keep their exact face, eyes, nose, lips, jawline, skin "
                 "tone, hair colour and age. Only the hairstyle, outfit and place change.")


TWIN_TEXT = ("Change only the person into {who}. {detail} {wear} Light: {light}. Background: {background}. "
             "Keep the camera exactly as in the image: the same distance, lens, angle and framing, the same space "
             "above the head, the same amount of shoulders and body in frame, the face the same size and in the "
             "same place. Keep the head position, head tilt and the open, mid-word mouth exactly as in the image. "
             "A completely different person. No text or lettering anywhere.")  # + casting.edit_realism()
# no "the same tight crop" and no close-up skin brief: with them Klein redraws every look as a
# tight, mouth-shut face shot (in one test all 24 looks came out tighter than their lead's frame)


def twin_key(*, video_path, prefix, size, seed=7, editor="klein", prompt=None, portrait_path=None,
             start_s=0.0, identity_path=None, keep=None):
    """The frame of performer A's clip at `start_s` re-cast as performer B (same pose and mouth).
    editor 'klein' (default): text-described B, pose kept by construction (fast, no portrait).
    editor 'qwen': B's Krea 2 portrait as the identity reference (tends to copy the portrait's pose).
    identity_path (artist runs, Klein): the artist photo as a second reference image, so a new
    outfit, hairstyle and place keep the artist's face (without it the looks drift into other
    people: ArcFace similarity to the artist 0.20-0.37 without, 0.33-0.63 with)."""
    g = Graph()
    fr = g.add("FVLoadVideoPath", path=video_path, width=size[0], height=size[1], frame_cap=1,
               start_seconds=float(start_s), force_fps=0.0)
    if editor == "qwen":
        ref = g.add("FVLoadImagePath", path=portrait_path)
        out = _qwen21(g, O(fr), prompt or TWIN_PROMPT, [O(ref)], seed=seed)
    elif identity_path:
        ref = g.add("FVLoadImagePath", path=identity_path)
        out = _klein(g, O(fr), f"{prompt} {TWIN_IDENTITY}", refs=[O(ref)], seed=seed)
    else:
        out = _klein(g, O(fr), prompt, seed=seed)
    out = O(g.add("ImageScale", image=out, upscale_method="lanczos", width=size[0], height=size[1], crop="center"))
    if _has_node("FVFaceMatch"):  # the edit pulls the camera back: put the face back where the lead's is
        out = O(g.add("FVFaceMatch", image=out, reference=O(fr), max_scale=2.2))
    g.add("SaveImage", images=out, filename_prefix=prefix)
    if keep:  # staged runs: LTX stays loaded between the singer clips and the look clips
        keep_ltx(g, keep)
    return g


def motion_key(*, video_path, start_s, prompt, prefix, size, seed=7):
    """The dance frame at `start_s` re-cast as a character (Klein edit, keeps the pose)."""
    g = Graph()
    fr = g.add("FVLoadVideoPath", path=video_path, width=size[0], height=size[1], frame_cap=1,
               start_seconds=float(start_s), force_fps=0.0)
    img = O(fr)
    if _has_node("FVLeadMask"):  # grey out backup dancers so the edit keeps one person
        dw = g.add("DWPreprocessor", image=img, detect_hand="enable", detect_body="enable",
                   detect_face="disable", resolution=512, bbox_detector=M["dwpose_bbox"],
                   pose_estimator=M["dwpose_pose"], scale_stick_for_xinsr_cn="disable")
        img = O(g.add("FVLeadMask", pose_keypoint=O(dw, 1), image=img))
    out = _klein(g, img, prompt, seed=seed)
    g.add("SaveImage", images=out, filename_prefix=prefix)
    return g


def free_memory(note="free"):
    g = Graph()
    g.add("FVFreeMemory", note=note)
    return g


def assemble(*, run_dir, prefix):
    g = Graph()
    g.add("FVAssemble", run_dir=run_dir, filename_prefix=prefix)
    return g


def keep_ltx(g, preset="quality"):
    """Wire LTX's loaders into an FVKeep node of graph g, so a job that doesn't use LTX (a QC
    pass, a Klein edit) keeps it loaded for the next video job instead of a reload from disk."""
    if not _has_node("FVKeep"):
        return g
    pr = formats.PRESETS.get(preset, formats.PRESETS["quality"])
    m = _ltx23(g) if pr["engine"] == "ltx23_r8" else _ltx25(g)
    g.add("FVKeep", model=m["model"], clip=m["clip"], vae=m["vae"], audio_vae=m["avae"], upscaler=m["ups"])
    return g


def qc(*, run_dir, pass_index=0, max_passes=1, then_assemble=True, stage="all", keep=None):
    """keep: the preset whose LTX stays loaded through this pass (staged runs)."""
    g = Graph()
    kw = {"stage": stage} if stage != "all" else {}
    g.add("FVQC", run_dir=run_dir, pass_index=pass_index, max_passes=max_passes, then_assemble=then_assemble, **kw)
    if keep:
        keep_ltx(g, keep)
    return g


def critic(*, run_dir, pass_index=0, max_passes=2):
    """The vision check of a run's stills (director.critic_pass)."""
    g = Graph()
    g.add("FVCritic", run_dir=run_dir, pass_index=pass_index, max_passes=max_passes)
    return g
