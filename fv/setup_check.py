"""Is this ComfyUI ready for OneShot MV? One check, run wherever a problem can show up first: the
Studio's Brief (as soon as the page opens), Make the video and Preview (before anything starts), the
Director (before it plans), and before any chain of jobs is queued. It reads ComfyUI's own node
registry and input lists, so it catches exactly what ComfyUI would refuse with "HTTP Error 400":
a node pack that isn't installed or didn't load, a pack or ComfyUI too old for a node or option, and
a model file ComfyUI doesn't list. Each problem comes with the fix, in plain words."""
from __future__ import annotations

import inspect
import os

from . import formats, graphs

# the node classes a run can use that don't come with ComfyUI itself: their pack and where to get it
PACKS = {"RES4LYF": "https://github.com/ClownsharkBatwing/RES4LYF",
         "ComfyUI-KJNodes": "https://github.com/kijai/ComfyUI-KJNodes",
         "ComfyUI-LTXVideo": "https://github.com/Lightricks/ComfyUI-LTXVideo",
         "comfyui_controlnet_aux": "https://github.com/Fannovel16/comfyui_controlnet_aux"}
NODE_PACKS = {"ClownsharKSampler_Beta": "RES4LYF", "LTX2AttentionTunerPatch": "ComfyUI-KJNodes",
              "PathchSageAttentionKJ": "ComfyUI-KJNodes", "LTXICLoRALoaderModelOnly": "ComfyUI-LTXVideo",
              "LTXAddVideoICLoRAGuide": "ComfyUI-LTXVideo", "DWPreprocessor": "comfyui_controlnet_aux"}
SAMPLER_PACKS = {"res_2s": "RES4LYF"}  # samplers another pack adds to ComfyUI's list
FOLDERS = {"unet_name": "diffusion_models", "ckpt_name": "checkpoints", "clip_name": "text_encoders",
           "text_encoder": "text_encoders", "lora_name": "loras", "vae_name": "vae",
           "model_name": "latent_upscale_models"}


class NotReady(ValueError):
    """This ComfyUI can't run the jobs yet; the message says what to install or update."""


def _registry():
    try:
        import nodes as comfy_nodes  # ComfyUI's node registry (absent outside ComfyUI)
        return comfy_nodes.NODE_CLASS_MAPPINGS
    except Exception:
        return None


def _self_checked(cls) -> tuple[set, bool]:
    """The inputs a node checks itself (ComfyUI then skips its list check for them), as ComfyUI
    decides it in execution.validate_inputs."""
    fn = None
    try:
        from comfy_api.internal import _ComfyNodeInternal, first_real_override  # type: ignore
        fn = first_real_override(cls, "validate_inputs") if issubclass(cls, _ComfyNodeInternal) \
            else getattr(cls, "VALIDATE_INPUTS", None)
    except Exception:
        fn = getattr(cls, "VALIDATE_INPUTS", None)
    if fn is None:
        return set(), False
    try:
        spec = inspect.getfullargspec(fn)
        return set(spec.args), spec.varkw is not None
    except TypeError:
        return set(), True


def _options(spec) -> list | None:
    """A dropdown's choices from an input spec (V1 list, or V3 "COMBO" with options)."""
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    kind, extra = spec[0], (spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {})
    if extra.get("multiselect"):
        return None
    if isinstance(kind, (list, tuple)):
        return list(kind)
    if kind == "COMBO" and isinstance(extra.get("options"), (list, tuple)):
        return list(extra["options"])
    return None


def problems(job_graphs, reg=None) -> dict:
    """What ComfyUI would refuse in these jobs: {"nodes": [class], "files": {file: folder},
    "values": [(class, input, value)]}. Empty lists outside ComfyUI."""
    reg = _registry() if reg is None else reg
    out = {"nodes": [], "files": {}, "values": []}
    if reg is None:
        return out
    specs, checked = {}, {}
    for g in job_graphs:
        for n in g.values():
            ct = n.get("class_type") if isinstance(n, dict) else None
            if not ct:
                continue
            if ct not in reg:
                if ct not in out["nodes"]:
                    out["nodes"].append(ct)
                continue
            if ct not in specs:
                cls = reg.get(ct) if hasattr(reg, "get") else None
                try:
                    it = cls.INPUT_TYPES()
                    specs[ct] = {**(it.get("required") or {}), **(it.get("optional") or {})}
                except Exception:
                    specs[ct] = {}
                checked[ct] = _self_checked(cls) if cls is not None else (set(), False)
            args, any_ = checked[ct]
            for k, v in (n.get("inputs") or {}).items():
                if not isinstance(v, str) or any_ or k in args:
                    continue
                opts = _options(specs[ct].get(k))
                if opts is None or v in opts:
                    continue
                if v.lower().endswith(graphs.MODEL_EXT):
                    out["files"][os.path.basename(v.replace("\\", "/"))] = FOLDERS.get(k, "")
                elif (ct, k, v) not in out["values"]:
                    out["values"].append((ct, k, v))
    return out


def fixes(p: dict, model_set: str = "") -> list[str]:
    """Each problem as the thing to do about it, one line each."""
    lines = []
    by_pack = {}
    for c in p.get("nodes") or []:
        by_pack.setdefault(NODE_PACKS.get(c, ""), []).append(c)
    for pack, cs in sorted(by_pack.items()):
        if pack:
            lines.append(f"Install or update {pack} ({PACKS[pack]}), then restart ComfyUI; it provides "
                         f"{', '.join(cs)}. If it's installed, it didn't load: look for IMPORT FAILED next to "
                         "it in the ComfyUI console when ComfyUI starts.")
        else:
            lines.append(f"Update ComfyUI: this one has no {', '.join(cs)} node.")
    for ct, k, v in p.get("values") or []:
        if k == "sampler_name" and v in SAMPLER_PACKS:
            pack = SAMPLER_PACKS[v]
            if pack in by_pack:  # the pack's own line says it already
                continue
            by_pack[pack] = []
            lines.append(f"Install or update {pack} ({PACKS[pack]}), then restart ComfyUI; it adds the "
                         f"'{v}' sampler. If it's installed, look for IMPORT FAILED next to it in the console.")
        elif NODE_PACKS.get(ct):
            lines.append(f"Update {NODE_PACKS[ct]}: its {ct} node has no '{v}' {k}.")
        else:
            lines.append(f"Update ComfyUI: its {ct} node has no '{v}' {k}.")
    files = p.get("files") or {}
    if files:
        which = {"24gb": " of the 24 GB set", "full": " of the full set"}.get(model_set, "")
        listed = ", ".join(f"{f} (models/{d})" if d else f for f, d in sorted(files.items()))
        lines.append(f"Download the model file{'s' if len(files) > 1 else ''}{which} that ComfyUI can't find: "
                     f"{listed}. The README's model tables have the links. A file saved under another name counts "
                     "as missing: rename it, or change its name in fv/models.json.")
    return lines


def message(p: dict, model_set: str = "", lead: str = "") -> str:
    return " ".join([lead] + fixes(p, model_set)).strip() if fixes(p, model_set) else ""


# ------------------------------------------------------------------ what a run will use, before it plans
def probe(*, preset: str = "quality", fmt: str = "16:9", render: bool = True, theme: bool = True,
          artist: bool = False, images: bool = False, editor: str | None = None,
          extend: bool = False) -> list[dict]:
    """images: a World from the user's images (no Krea 2; Qwen-Image 2.1 only with extend, when images
    are extended to the video's shape).
    editor: Single Artist's image editor ("qwen" or "qwen_edit"; default graphs.artist_editor()).
    Sample jobs with the nodes and model files a run uses, built the way the Director builds
    them, so the check can run before the run plans (writing the World takes minutes). The full list
    of the run's own jobs is checked again before they are queued."""
    gs = []
    if theme:  # the World, the cast and the vision check: Gemma 3 through ComfyUI's Generate Text
        gs.append({"1": {"class_type": "LTXAVTextEncoderLoader",
                         "inputs": {"text_encoder": graphs.M["ltx23_te"], "ckpt_name": graphs.M["ltx23_ckpt"],
                                    "device": "default"}},
                   "2": {"class_type": "TextGenerate", "inputs": {"clip": ["1", 0], "prompt": "x"}}})
    if render:
        eff = formats.effective_preset(preset)
        s1, s2 = formats.portrait_size(fmt)
        if images:
            gs.append(graphs.world_image(path="probe.png", prefix="probe/i", size=s2, crop=(0, 0, 64, 36),
                                         extend=extend).nodes)
        else:
            gs.append(graphs.krea2_portrait(prompt="x", seed=1, prefix="probe/p", size1=s1, size2=s2,
                                            turbo_only=preset == "fast").nodes)
            gs.append(graphs.krea2_portrait(prompt="x", seed=1, prefix="probe/k", size1=s1, size2=s2).nodes)  # b-roll
        gs.append(graphs.ltx_render(preset=eff, fmt=fmt, key_path="probe.png", prompt="x", negative="x", frames=9,
                                    seed=1, prefix="probe/r", audio_path="probe.wav").nodes)
        gs.append(graphs.ltx_render(preset=eff, fmt=fmt, key_path=None, prompt="x", negative="x", frames=9,
                                    seed=1, prefix="probe/b").nodes)
        for g in (graphs.free_memory(), graphs.critic(run_dir="probe"), graphs.qc(run_dir="probe", keep=eff),
                  graphs.assemble(run_dir="probe", prefix="probe")):
            gs.append(g.nodes)
        if artist:  # Single Artist: the editor's portraits, the looks' keyframes and pose-guided clips
            gs.append(graphs.artist_portrait(artist_path="probe.png", prompt="x", seed=1, prefix="probe/a",
                                             size1=s1, size2=s2, editor=editor).nodes)
            gs.append(graphs.twin_key(video_path="probe.mp4", prefix="probe/k", size=formats.render_size(fmt, eff),
                                      identity_path="probe.png", prompt="x", editor=editor).nodes)
            gs.append(graphs.ltx_render(preset=eff, fmt=fmt, key_path="probe.png", prompt="x", negative="x", frames=9,
                                        seed=1, prefix="probe/t", audio_path="probe.wav",
                                        guide_video="probe.mp4").nodes)
    return gs


def check(job_graphs, model_set: str = "", lead: str = "It can't start yet, so nothing was queued.") -> None:
    """Raise NotReady, with what to do, if ComfyUI would refuse any of these jobs."""
    msg = message(problems(job_graphs), model_set, lead)
    if msg:
        raise NotReady(msg)


# the model files a default run loads (graphs.RUN_KEYS) and the folder each goes in
KEY_FOLDERS = {"krea_turbo": "diffusion_models", "krea_raw": "diffusion_models", "krea_turbo_lora": "loras",
               "krea_te": "text_encoders", "krea_vae": "vae", "ltx23_ckpt": "checkpoints",
               "ltx23_te": "text_encoders", "ltx23_upscaler": "latent_upscale_models", "ltx23_loras": "loras",
               "qwen21_unet": "diffusion_models", "qwen21_te": "text_encoders", "qwen21_vae": "vae",
               "qwen_edit_unet": "diffusion_models", "qwen_edit_te": "text_encoders", "qwen_edit_vae": "vae",
               "qwen_edit_lora": "loras",
               "ltx_union_lora": "loras"}


def set_files(name: str, artist: bool = False, images: bool = False, editor: str | None = None,
              extend: bool = False) -> dict:
    """{file: folder} of a model set's files that this ComfyUI doesn't list (graphs.missing_files),
    without switching the set the running jobs use. artist: with Single Artist's editor and look files.
    extend: World images extended to the video's shape (Qwen-Image 2.1)."""
    keys = set(graphs.RUN_KEYS) | (set(graphs.ARTIST_KEYS[editor or graphs.artist_editor()]) if artist else set())
    if images:  # a World from images makes no Krea 2 stills
        keys = {k for k in keys if not k.startswith("krea")} | (set(graphs.EXTEND_KEYS) if extend else set())
    m = {k: v for k, v in graphs._set_files(name).items() if k in keys}
    if m.get("krea_raw") == m.get("krea_turbo"):
        m.pop("krea_turbo_lora", None)
    return {f: KEY_FOLDERS.get(k, "") for k, f in graphs._resolve_models(m)[1]}


# ------------------------------------------------------------------ what a video will load, for the Studio
KREA = "Krea 2 Community License: free under US$1M revenue (images too)"
LTX = "LTX-2 Community License: free under US$10M revenue"
QWEN21 = "Reported research-only (Qwen Research License)"


def _precision(f: str) -> str:
    f = f.lower()
    return next((t for t in ("fp8", "int8", "bf16") if t in f), "")


def uses(models: str = "auto", *, artist: bool = False, images: bool = False, extend: bool = False,
         professional: bool = False) -> dict:
    """The models a video with these settings loads, one row per job, for the Studio's Models card:
    {"models": set, "rows": [{"role", "model", "files", "missing", "licence", "note"}]}. Files are looked up
    the way a run finds them (any sub-folder); "missing" is what this ComfyUI doesn't list. The optional
    checks (SyncNet, insightface) are files outside ComfyUI's model lists: "installed" says whether they are."""
    name = graphs.pick_models(models)
    sf = graphs._set_files(name)
    idx = graphs._model_index()
    rows = []

    def miss(files):
        return sorted({os.path.basename(f) for _, f in graphs._resolve_models(dict(enumerate(files)), idx)[1]})

    def row(role, model, keys, licence, note="", prec=True):
        files = list(dict.fromkeys(sf[k] for k in keys if sf.get(k)))
        prec = (_precision(files[0]) or ("bf16" if name == "full" else "")) if files and prec else ""
        rows.append({"role": role, "model": model + (f" ({prec})" if prec else ""), "files": files,
                     "missing": miss(files), "licence": licence, "note": note})

    if not images:
        two = sf.get("krea_raw") != sf.get("krea_turbo")
        row("Stills", "Krea 2 Turbo and Raw" if two else "Krea 2 Turbo",
            ["krea_turbo"] + (["krea_raw", "krea_turbo_lora"] if two else []) + ["krea_te", "krea_vae"], KREA,
            "Raw redetails with the Turbo LoRA" if two else "")
    elif extend and not professional:
        row("Extending", "Qwen-Image 2.1", graphs.EXTEND_KEYS, QWEN21)
    if artist:
        if professional:
            row("The artist", "Qwen-Image-Edit 2511", graphs.ARTIST_KEYS["qwen_edit"][:4], "Apache-2.0",
                "With the 4-step Lightning LoRA")
        else:
            row("The artist", "Qwen-Image 2.1", graphs.ARTIST_KEYS["qwen"][:3], QWEN21)
        row("Looks", "LTX-2.3 union-control LoRA and DWPose", ["ltx_union_lora"], LTX + "; DWPose Apache-2.0",
            prec=False)
    row("Video", "LTX-2.3 22B dev", ["ltx23_ckpt", "ltx23_upscaler"], LTX)
    loras = [x[0] for x in sf.get("ltx23_loras") or []]
    official = [f for f in loras if "enhance" not in f.lower()]
    community = [f for f in loras if "enhance" in f.lower()]
    if official:
        rows.append({"role": "Video LoRAs", "model": "Distilled and Detailer (Lightricks)", "files": official,
                     "missing": miss(official), "licence": LTX, "note": ""})
    if community:
        rows.append({"role": "Video LoRAs", "model": "Crisp and Soft Enhance (community)",
                     "files": community, "missing": miss(community), "licence": "No licence stated",
                     "note": "To run without them, remove them from ltx23_loras in fv/models.json"})
    row("Language", "Gemma 3 12B", ["ltx23_te"], "Gemma Terms of Use")
    rows.append({"role": "Vocals", "model": "Hybrid Demucs", "files": [], "missing": [],
                 "licence": "Code MIT; no weights licence published", "note": ""})
    from . import identity, lipsync
    rows.append({"role": "Checks", "model": "SyncNet and S3FD", "files": [], "missing": [],
                 "installed": lipsync.weight_paths() is not None,
                 "licence": "Research use, CC-BY (Oxford VGG)", "note": ""})
    rows.append({"role": "Checks", "model": "insightface buffalo_l", "files": [], "missing": [],
                 "installed": identity.model_dir() is not None,
                 "licence": "Non-commercial research use only", "note": ""})
    return {"models": name, "rows": rows}


def report(models: str = "auto", preset: str = "quality", fmt: str = "16:9", artist: bool = False,
           images: bool = False, editor: str | None = None, extend: bool = False) -> dict:
    """For the Studio, as soon as the Brief opens: what stops a preview and what stops a render (the
    things to do, one line each), and SageAttention's state. Never switches graphs.M (a run may be
    building jobs with it), so the model files are looked up separately."""
    name = graphs.pick_models(models)
    files = set_files(name, artist, images, editor, extend=images and extend)
    sf = graphs._set_files(name)
    llm = {os.path.basename(sf["ltx23_te"]), os.path.basename(sf["ltx23_ckpt"])}  # the World's language model
    # a World from images needs no language model, and an image editor only to extend images
    plan_p = dict(problems(probe(render=False, theme=not images)),
                  files={} if images else {f: d for f, d in files.items() if f in llm})
    render_p = dict(problems(probe(preset=preset, fmt=fmt, render=True, theme=not images, artist=artist,
                                   images=images, editor=editor, extend=images and extend)),
                    files=files)
    return {"models": name, "plan": fixes(plan_p, name), "render": fixes(render_p, name),
            "sage": graphs.sage_report()}


# ------------------------------------------------------------------ ComfyUI's refusal, read
def refused(reply, code: int) -> str:
    """ComfyUI's refusal of a job ({"error": ..., "node_errors": ...}, server.py) in plain words."""
    if not isinstance(reply, dict):
        return f"ComfyUI refused a OneShot MV job (HTTP {code}); the ComfyUI console says why."
    err = reply.get("error") if isinstance(reply.get("error"), dict) else {}
    p = {"nodes": [], "files": {}, "values": []}
    other = []
    if err.get("type") == "missing_node_type":
        p["nodes"].append((err.get("extra_info") or {}).get("class_type") or "?")
    for ne in (reply.get("node_errors") or {}).values():
        ct = ne.get("class_type", "")
        for e in ne.get("errors") or []:
            x = e.get("extra_info") or {}
            k, v = x.get("input_name", ""), x.get("received_value")
            if e.get("type") == "value_not_in_list" and isinstance(v, str) and v.lower().endswith(graphs.MODEL_EXT):
                p["files"][os.path.basename(v.replace("\\", "/"))] = FOLDERS.get(k, "")
            elif e.get("type") == "value_not_in_list":
                p["values"].append((ct, k, str(v)))
            else:
                det = f" ({str(e['details'])[:160]})" if e.get("details") else ""
                other.append(f"{ct}: {e.get('message', '')}{det}.")
    lines = fixes(p, graphs.M.get("_set", "")) + list(dict.fromkeys(other))
    if not lines:
        lines.append(f"{err.get('message') or f'HTTP {code}'}; the ComfyUI console says more.")
    return "ComfyUI refused a OneShot MV job. " + " ".join(lines)
