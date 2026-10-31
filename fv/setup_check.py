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
def probe(*, preset: str = "quality", fmt: str = "16:9", render: bool = True, theme: bool = True) -> list[dict]:
    """Sample jobs with the nodes and model files a run uses, built the way the Director builds
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
    return gs


def check(job_graphs, model_set: str = "", lead: str = "It can't start yet, so nothing was queued.") -> None:
    """Raise NotReady, with what to do, if ComfyUI would refuse any of these jobs."""
    msg = message(problems(job_graphs), model_set, lead)
    if msg:
        raise NotReady(msg)


# the model files a default run loads (graphs.RUN_KEYS) and the folder each goes in
KEY_FOLDERS = {"krea_turbo": "diffusion_models", "krea_raw": "diffusion_models", "krea_turbo_lora": "loras",
               "krea_te": "text_encoders", "krea_vae": "vae", "ltx23_ckpt": "checkpoints",
               "ltx23_te": "text_encoders", "ltx23_upscaler": "latent_upscale_models", "ltx23_loras": "loras"}


def set_files(name: str) -> dict:
    """{file: folder} of a model set's files that this ComfyUI doesn't list (graphs.missing_files),
    without switching the set the running jobs use."""
    m = {k: v for k, v in graphs._set_files(name).items() if k in graphs.RUN_KEYS}
    if m.get("krea_raw") == m.get("krea_turbo"):
        m.pop("krea_turbo_lora", None)
    return {f: KEY_FOLDERS.get(k, "") for k, f in graphs._resolve_models(m)[1]}


def report(models: str = "auto", preset: str = "quality", fmt: str = "16:9") -> dict:
    """For the Studio, as soon as the Brief opens: what stops a preview and what stops a render (the
    things to do, one line each), and SageAttention's state. Never switches graphs.M (a run may be
    building jobs with it), so the model files are looked up separately."""
    name = graphs.pick_models(models)
    files = set_files(name)
    sf = graphs._set_files(name)
    llm = {os.path.basename(sf["ltx23_te"]), os.path.basename(sf["ltx23_ckpt"])}  # the World's language model
    plan_p = dict(problems(probe(render=False, theme=True)), files={f: d for f, d in files.items() if f in llm})
    render_p = dict(problems(probe(preset=preset, fmt=fmt, render=True, theme=True)), files=files)
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
