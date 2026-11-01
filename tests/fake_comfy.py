"""A stand-in ComfyUI for the tests: its node registry (with the dropdown lists ComfyUI checks a job
against), its model folders, its queue, and the SageAttention package and GPU. Each option is one way
a user's ComfyUI can differ from the one OneShot MV was made on."""
import importlib.machinery
import os
import sys
import tempfile
import types

from fv import director, graphs, setup_check

# the 24 GB set, in sub-folders and with Windows separators, as a user might keep it
FILES = {
    "diffusion_models": ["Krea2\\krea2_turbo_fp8_scaled.safetensors"],
    "text_encoders": ["qwen3vl_4b_fp8_scaled.safetensors", "gemma\\gemma_3_12B_it_fp8_scaled.safetensors"],
    "vae": ["qwen_image_vae.safetensors"],
    "checkpoints": ["LTX-2\\ltx-2.3-22b-dev-fp8.safetensors"],
    "loras": ["LTX-2\\ltx-2.3-22b-distilled-lora-384-1.1.safetensors", "ltx-2-19b-ic-lora-detailer.safetensors",
              "LTX2.3_Crisp_Enhance.safetensors", "LTX2.3_Soft_Enhance.safetensors"],
    "latent_upscale_models": ["ltx-2.3-spatial-upscaler-x2-1.1.safetensors"],
}
PACKS = ("RES4LYF", "ComfyUI-KJNodes", "ComfyUI-LTXVideo")
ARTIST_FILES = {"diffusion_models": ["Flux-2 Klein\\flux-2-klein-9b-fp8.safetensors"],
                "text_encoders": ["qwen_3_8b_fp8mixed.safetensors"], "vae": ["full_encoder_small_decoder.safetensors"],
                "loras": ["ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"]}
KJ_MODES = ["disabled", "auto", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton",
            "sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda++", "sageattn3", "sageattn3_per_block_mean"]
CLIP_TYPES = ["stable_diffusion", "ltxv", "flux2", "qwen_image", "krea2"]
CORE_SAMPLERS = ["euler", "euler_ancestral", "dpmpp_sde", "dpmpp_2m_sde"]


def _cls(name, inputs=None, v3=False):
    """A node class whose INPUT_TYPES gives these inputs (V1 lists, or V3 "COMBO" with options)."""
    def it(cls):
        return {"required": dict(inputs or {})}
    return type(name, (), {"INPUT_TYPES": classmethod(it), "_v3": v3})


def node_classes() -> set:
    """Every node class a default run's jobs use (built with no ComfyUI around, so no node is left out)."""
    saved = sys.modules.pop("nodes", None)
    graphs._SAGE[graphs.M.get("ltx23_sage")] = ("sageattn_qk_int8_pv_fp8_cuda++", "")  # include the patch, quietly
    try:
        gs = setup_check.probe(render=True, theme=True, artist=True)
        gs.append(graphs.twin_key(video_path="v", prefix="k", size=(1280, 768)).nodes)
        gs.append(graphs.artist_portrait(artist_path="a.png", prompt="x", seed=1, prefix="a", size1=(1344, 768),
                                         size2=(2016, 1152), editor="qwen").nodes)  # Single Artist on Qwen
    finally:
        if saved is not None:
            sys.modules["nodes"] = saved
        graphs._SAGE.clear()
    return {n["class_type"] for g in gs for n in g.values()} | {"PathchSageAttentionKJ", "FVQC", "FVKeep", "FVCritic"}


ALL = node_classes()


def sage1():
    m = types.ModuleType("sageattention")
    m.__spec__ = importlib.machinery.ModuleSpec("sageattention", None)
    m.sageattn = m.sageattn_varlen = lambda *a, **k: None
    m.core = types.ModuleType("sageattention.core")
    return m


def sage2(sm80=True, sm89=True, sm90=False):
    m = sage1()
    for f in ("sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp8_cuda",
              "sageattn_qk_int8_pv_fp8_cuda_sm90"):
        setattr(m, f, lambda *a, **k: None)
    m.core.SM80_ENABLED, m.core.SM89_ENABLED, m.core.SM90_ENABLED = sm80, sm89, sm90
    return m


def torch_with(cap, hip=False):
    t = types.ModuleType("torch")
    t.cuda = types.SimpleNamespace(is_available=lambda: cap is not None, current_device=lambda: 0,
                                   get_device_capability=lambda i=0: cap,
                                   get_device_properties=lambda i=0: types.SimpleNamespace(total_memory=24 * 2 ** 30))
    t.version = types.SimpleNamespace(hip="6.2" if hip else None)
    t.nn = types.ModuleType("torch.nn")
    t.nn.functional = types.ModuleType("torch.nn.functional")
    return t


def install(monkeypatch, *, packs=PACKS, missing=(), drop_files=(), old_comfy=False, kj_modes=KJ_MODES,
            sage=None, cap=(8, 9), hip=False, comfy=True, artist=False):
    """Put a fake ComfyUI in place. Returns the list the fake queue fills with posted jobs.
    artist: also what Single Artist needs (Klein's files, the union-control LoRA, DWPose's pack)."""
    all_files = {k: list(v) + list(ARTIST_FILES.get(k, []) if artist else []) for k, v in FILES.items()}
    if artist:
        packs = tuple(packs) + ("comfyui_controlnet_aux",)
    files = {k: [f for f in v if os.path.basename(f.replace("\\", "/")) not in drop_files] for k, v in all_files.items()}
    out_dir = tempfile.mkdtemp()
    fp = types.ModuleType("folder_paths")
    fp.folder_names_and_paths = {k: ([], set()) for k in graphs.MODEL_FOLDERS}
    fp.get_filename_list = lambda kind: list(files.get(kind, []))
    fp.get_output_directory = lambda: out_dir
    fp.get_input_directory = lambda: out_dir
    monkeypatch.setitem(sys.modules, "folder_paths", fp)

    samplers = CORE_SAMPLERS + (["res_2s"] if "RES4LYF" in packs else [])
    specs = {
        "UNETLoader": {"unet_name": (files["diffusion_models"],), "weight_dtype": (["default", "fp8_e4m3fn"],)},
        "CheckpointLoaderSimple": {"ckpt_name": (files["checkpoints"],)},
        "CLIPLoader": {"clip_name": (files["text_encoders"],),
                       "type": ([t for t in CLIP_TYPES if not (old_comfy and t == "krea2")],),
                       "device": (["default", "cpu"],)},
        "VAELoader": {"vae_name": (files["vae"],)},
        "LoraLoaderModelOnly": {"lora_name": (files["loras"],)},
        "LatentUpscaleModelLoader": {"model_name": (files["latent_upscale_models"],)},
        "LTXAVTextEncoderLoader": {"text_encoder": (files["text_encoders"],), "ckpt_name": (files["checkpoints"],),
                                   "device": (["default", "cpu"],)},
        "LTXVAudioVAELoader": {"ckpt_name": (files["checkpoints"],)},
        "KSamplerSelect": {"sampler_name": ("COMBO", {"options": samplers})},  # a V3 node
        "KSampler": {"sampler_name": (samplers,), "scheduler": (["simple", "beta"],)},
        "ClownsharKSampler_Beta": {"sampler_name": (["multistep/deis_2m", "res_2s"],), "scheduler": (["beta", "simple"],)},
        "PathchSageAttentionKJ": {"sage_attention": (list(kj_modes),), "allow_compile": ("BOOLEAN",)},
    }
    reg = {}
    for c in ALL:
        pack = setup_check.NODE_PACKS.get(c)
        if c in missing or (pack and pack not in packs) or (old_comfy and c == "TextGenerate"):
            continue
        reg[c] = _cls(c, specs.get(c))
    if comfy:
        mod = types.ModuleType("nodes")
        mod.NODE_CLASS_MAPPINGS = reg
        monkeypatch.setitem(sys.modules, "nodes", mod)
    t = torch_with(cap, hip)
    monkeypatch.setitem(sys.modules, "torch", t)
    monkeypatch.setitem(sys.modules, "torch.nn", t.nn)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", t.nn.functional)
    if sage is None:
        monkeypatch.setitem(sys.modules, "sageattention", None)  # import fails, as when it isn't installed
    else:
        monkeypatch.setitem(sys.modules, "sageattention", sage)
        monkeypatch.setitem(sys.modules, "sageattention.core", sage.core)
    graphs._SAGE.clear()
    graphs.use_models("auto")  # as when ComfyUI loads the pack: names resolved in these model folders
    posted = []
    monkeypatch.setattr(director, "queue_prompt",
                        lambda g, url=None, client_id="", front=False: posted.append(g) or f"id{len(posted)}")
    return posted
