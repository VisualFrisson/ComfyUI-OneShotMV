"""ComfyUI API-format graph builders for every Frisson Vision job.

All file inputs go through the pack's path loaders (FVLoadImagePath / FVLoadVideoPath /
FVLoadAudioPath), which do not check that the file exists when the job is queued, so the
Director can queue a whole run up front (later jobs read files earlier jobs will write)."""
from __future__ import annotations

import json
import os

from . import formats

HERE = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(HERE, "models.json"), encoding="utf-8"))

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
def krea2_portrait(*, prompt, seed, prefix, size1, size2, denoise=0.40):
    g = Graph()
    turbo = g.add("UNETLoader", unet_name=M["krea_turbo"], weight_dtype="default")
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


# ------------------------------------------------------------------ LTX models
def _ltx23(g, union=False):
    ck = g.add("CheckpointLoaderSimple", ckpt_name=M["ltx23_ckpt"])
    model = O(ck, 0)
    if M.get("ltx23_sage"):
        model = O(g.add("PathchSageAttentionKJ", model=model, sage_attention=M["ltx23_sage"],
                        allow_compile=False))
    for name, st in M["ltx23_loras"]:
        model = O(g.add("LoraLoaderModelOnly", model=model, lora_name=name, strength_model=st))
    if M.get("ltx23_av_tuner"):
        vs, as_, a2v, v2a = M["ltx23_av_tuner"]
        model = O(g.add("LTX2AttentionTunerPatch", model=model, blocks="", video_scale=vs,
                        audio_scale=as_, audio_to_video_scale=a2v, video_to_audio_scale=v2a,
                        triton_kernels=True))
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
    model, dsf = O(unet), 1.0
    if union:
        lo = g.add("LTXICLoRALoaderModelOnly", model=model, lora_name=M["ltx_union_lora"], strength_model=1.0)
        model, dsf = O(lo), O(lo, 1)
    vae = g.add("VAELoader", vae_name=M["ltx25_vae"])
    avae = g.add("VAELoader", vae_name=M["ltx25_audio_vae"])
    clip = g.add("CLIPLoader", clip_name=M["ltx25_te"], type="ltxv", device="default")
    ups = g.add("LatentUpscaleModelLoader", model_name=M["ltx25_upscaler"])
    return dict(model=model, dsf=dsf, vae=O(vae), avae=O(avae), clip=O(clip), ups=O(ups))


def _pose(g, video_path, w, h, frames, fps, start_s=0.0, face=True):
    src = g.add("FVLoadVideoPath", path=video_path, width=w, height=h, frame_cap=frames,
                start_seconds=float(start_s), force_fps=float(fps))
    return O(g.add("DWPreprocessor", image=O(src), detect_hand="enable", detect_body="enable",
                   detect_face="enable" if face else "disable", resolution=512,
                   bbox_detector=M["dwpose_bbox"], pose_estimator=M["dwpose_pose"],
                   scale_stick_for_xinsr_cn="disable"))


def ltx_render(*, preset, fmt, key_path, prompt, negative, frames, seed, prefix,
               audio_path=None, audio_seek=0.0, guide_video=None, guide_start=0.0,
               guide_strength=1.0, guide_face=True, save_pose=False, key_strengths=None):
    """One LTX clip. audio_path -> lip sync on a frozen vocal latent (voice);
    guide_video -> DWPose union-control guide (twins, motion)."""
    pr = formats.PRESETS[preset]
    fps = pr["fps"]
    sizes = formats.stage_sizes(fmt, preset)
    g = Graph()
    union = guide_video is not None
    m = _ltx23(g, union) if pr["engine"] == "ltx23_r8" else _ltx25(g, union)
    pos = g.add("CLIPTextEncode", clip=m["clip"], text=prompt)
    neg = g.add("CLIPTextEncode", clip=m["clip"], text=negative)
    cond = g.add("LTXVConditioning", positive=O(pos), negative=O(neg), frame_rate=float(fps))
    p_ref, n_ref = O(cond, 0), O(cond, 1)
    key = g.add("FVLoadImagePath", path=key_path)
    keys = []
    for (w, h) in sizes:
        k = g.add("ImageScale", image=O(key), upscale_method="lanczos", width=w, height=h, crop="center")
        keys.append(O(g.add("LTXVPreprocess", image=O(k), img_compression=33)))
    ks = key_strengths or ((0.7, 0.75, 0.8) if pr["stages"] == 3 else (0.9, 0.9))
    w1, h1 = sizes[0]
    lat = g.add("EmptyLTXVLatentVideo", width=w1, height=h1, length=frames, batch_size=1)
    lat_ref = O(g.add("LTXVImgToVideoInplace", vae=m["vae"], image=keys[0], latent=O(lat),
                      strength=ks[0], bypass=False))
    if union:
        ctl = _pose(g, guide_video, w1, h1, frames, fps, guide_start, guide_face)
        gd = g.add("LTXAddVideoICLoRAGuide", positive=p_ref, negative=n_ref, vae=m["vae"], latent=lat_ref,
                   image=ctl, frame_idx=0, strength=guide_strength, latent_downscale_factor=m["dsf"],
                   crop="disabled", use_tiled_encode=False, tile_size=256, tile_overlap=64)
        p_ref, n_ref, lat_ref = O(gd, 0), O(gd, 1), O(gd, 2)
        if save_pose:
            _save_video(g, ctl, fps, prefix + "_pose")
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
        sig = g.add("LTXVScheduler", steps=30, max_shift=2.05, base_shift=0.95, stretch=True,
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
        li = g.add("LTXVImgToVideoInplace", vae=m["vae"], image=keys[i], latent=O(up), strength=ks[i], bypass=False)
        av = g.add("LTXVConcatAVLatent", video_latent=O(li), audio_latent=a_ref if audio_path else O(sep, 1))
        gd2 = g.add("CFGGuider", model=m["model"], positive=cpos, negative=cneg, cfg=1.0)
        s = g.add("SamplerCustomAdvanced", noise=O(g.add("RandomNoise", noise_seed=12345 + seed % 1000)),
                  guider=O(gd2), sampler=O(g.add("KSamplerSelect", sampler_name=sname)),
                  sigmas=O(g.add("ManualSigmas", sigmas=sigs)), latent_image=O(av))
        sep = g.add("LTXVSeparateAVLatent", av_latent=O(s))
        vlat = O(sep, 0)
    dec = g.add("VAEDecodeTiled", samples=vlat, vae=m["vae"], tile_size=512, overlap=64,
                temporal_size=64, temporal_overlap=8)
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


TWIN_TEXT = ("Change the person into {who}. {detail} {wear} Light: {light}. Background: {background}. "
             "Keep the exact same camera angle, framing, head position, head tilt, mouth shape and facial "
             "expression. A completely different person. Photorealistic photo with natural skin texture.")


def twin_key(*, video_path, prefix, size, seed=7, editor="klein", prompt=None, portrait_path=None):
    """First frame of performer A's clip re-cast as performer B (same pose and mouth).
    editor 'klein' (default): text-described B, pose kept by construction (fast, no portrait).
    editor 'qwen': B's Krea 2 portrait as the identity reference (tends to copy the portrait's pose)."""
    g = Graph()
    fr = g.add("FVLoadVideoPath", path=video_path, width=size[0], height=size[1], frame_cap=1,
               start_seconds=0.0, force_fps=0.0)
    if editor == "qwen":
        ref = g.add("FVLoadImagePath", path=portrait_path)
        out = _qwen21(g, O(fr), prompt or TWIN_PROMPT, [O(ref)], seed=seed)
    else:
        out = _klein(g, O(fr), prompt, seed=seed)
    out = O(g.add("ImageScale", image=out, upscale_method="lanczos", width=size[0], height=size[1], crop="center"))
    g.add("SaveImage", images=out, filename_prefix=prefix)
    return g


def motion_key(*, video_path, start_s, prompt, prefix, size, seed=7):
    """The dance frame at `start_s` re-cast as a character (Klein edit, keeps the pose)."""
    g = Graph()
    fr = g.add("FVLoadVideoPath", path=video_path, width=size[0], height=size[1], frame_cap=1,
               start_seconds=float(start_s), force_fps=0.0)
    out = _klein(g, O(fr), prompt, seed=seed)
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
