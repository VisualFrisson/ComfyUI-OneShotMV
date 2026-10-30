"""When ComfyUI refuses a job, the user should see why, not "HTTP Error 400: Bad Request".

Runs without ComfyUI or a GPU:  python -m pytest tests/
The 400 replies below have the shape ComfyUI 0.43 sends (server.py post_prompt, execution.validate_prompt)."""
import http.server
import json
import os
import sys
import tempfile
import threading
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fv import director, graphs  # noqa: E402

KREA_JOB = graphs.krea2_portrait(prompt="a portrait", seed=1, prefix="t/p000", size1=(1344, 768),
                                 size2=(2016, 1152)).nodes

MISSING_NODE = {"error": {"type": "missing_node_type",
                          "message": "Node 'ClownsharKSampler_Beta' not found. The custom node may not be installed.",
                          "details": "Node ID '#11'",
                          "extra_info": {"node_id": "11", "class_type": "ClownsharKSampler_Beta",
                                         "node_title": "ClownsharKSampler_Beta"}},
                "node_errors": {}}


def _not_in_list(cls, name, value, choices):
    return {"error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation",
                      "details": "", "extra_info": {}},
            "node_errors": {"3": {"errors": [{"type": "value_not_in_list", "message": "Value not in list",
                                              "details": f"{name}: {value!r} not in {choices}",
                                              "extra_info": {"input_name": name, "input_config": None,
                                                             "received_value": value}}],
                                  "dependent_outputs": ["14"], "class_type": cls}}}


MISSING_FILE = _not_in_list("UNETLoader", "unet_name", "krea2_turbo_fp8_scaled.safetensors",
                            ["flux1-dev.safetensors", "wan2.2_t2v.safetensors"])
OLD_COMFY = _not_in_list("CLIPLoader", "type", "krea2", ["stable_diffusion", "flux2", "ltxv"])


@pytest.fixture
def comfy():
    """A stand-in for ComfyUI's /prompt that answers with the reply the test sets."""
    reply = {"status": 200, "body": {"prompt_id": "abc"}}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            data = json.dumps(reply["body"]).encode()
            self.send_response(reply["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def answer(status, body):
        reply.update(status=status, body=body)
        return f"http://127.0.0.1:{srv.server_port}"
    yield answer
    srv.shutdown()


def test_accepted_job_returns_its_id(comfy):
    assert director.queue_prompt(KREA_JOB, comfy(200, {"prompt_id": "abc"})) == "abc"


def test_missing_node_pack_is_named(comfy):
    with pytest.raises(Exception) as e:
        director.queue_prompt(KREA_JOB, comfy(400, MISSING_NODE))
    msg = str(e.value)
    assert "RES4LYF" in msg and "ClownsharKSampler_Beta" in msg


def test_missing_model_file_is_named(comfy):
    with pytest.raises(Exception) as e:
        director.queue_prompt(KREA_JOB, comfy(400, MISSING_FILE))
    msg = str(e.value)
    assert "krea2_turbo_fp8_scaled.safetensors" in msg
    assert "flux1-dev" not in msg  # not the whole list of the user's models


def test_old_comfyui_is_named(comfy):
    with pytest.raises(Exception) as e:
        director.queue_prompt(KREA_JOB, comfy(400, OLD_COMFY))
    msg = str(e.value)
    assert "krea2" in msg and "update ComfyUI" in msg


# ------------------------------------------------------------------ the check before anything is queued
LISTED = {  # where a user might keep the 24 GB set (sub-folders and Windows separators on purpose)
    "diffusion_models": ["Krea2\\krea2_turbo_fp8_scaled.safetensors"],
    "text_encoders": ["qwen3vl_4b_fp8_scaled.safetensors", "gemma\\gemma_3_12B_it_fp8_scaled.safetensors"],
    "vae": ["qwen_image_vae.safetensors"],
    "checkpoints": ["LTX-2\\ltx-2.3-22b-dev-fp8.safetensors"],
    "loras": ["LTX-2\\ltx-2.3-22b-distilled-lora-384-1.1.safetensors", "ltx-2-19b-ic-lora-detailer.safetensors",
              "LTX2.3_Crisp_Enhance.safetensors", "LTX2.3_Soft_Enhance.safetensors"],
    "latent_upscale_models": ["ltx-2.3-spatial-upscaler-x2-1.1.safetensors"],
}


class _Registry(dict):
    """ComfyUI's NODE_CLASS_MAPPINGS with some node classes not installed."""

    def __init__(self, missing=()):
        super().__init__()
        self.missing = set(missing)

    def __contains__(self, name):
        return name not in self.missing


def _song(seconds=45, sr=44100):
    t = np.arange(int(sr * seconds)) / sr
    clicks = (np.mod(t, 0.5) < 0.03).astype(np.float32)
    env = np.where(t > 20, 1.0, 0.4)
    x = (0.6 * clicks * env + 0.2 * np.sin(2 * np.pi * 220 * t) * env).astype(np.float32)
    return np.stack([x, x]), sr


@pytest.fixture
def comfyui(monkeypatch):
    """A fake ComfyUI around the Director: its node registry, its model lists and its queue."""
    posted = []

    def setup(missing_nodes=(), drop_files=()):
        fp = types.ModuleType("folder_paths")
        fp.folder_names_and_paths = {k: ([], set()) for k in graphs.MODEL_FOLDERS}
        fp.get_filename_list = lambda kind: [f for f in LISTED.get(kind, [])
                                             if os.path.basename(f.replace("\\", "/")) not in drop_files]
        reg = types.ModuleType("nodes")
        reg.NODE_CLASS_MAPPINGS = _Registry(missing_nodes)
        monkeypatch.setitem(sys.modules, "folder_paths", fp)
        monkeypatch.setitem(sys.modules, "nodes", reg)
        monkeypatch.setattr(director, "queue_prompt",
                            lambda g, url=None, client_id="", front=False: posted.append(g) or f"id{len(posted)}")
        return posted
    yield setup
    graphs.use_models("auto")


def _direct(**kw):
    audio, sr = _song()
    return director.direct(audio=audio, sr=sr, out_root=tempfile.mkdtemp(), seconds=20, do_stems=False,
                           log=lambda *a: None, **kw)


def test_nothing_is_queued_when_packs_and_files_are_missing(comfyui):
    posted = comfyui(missing_nodes={"ClownsharKSampler_Beta", "LTX2AttentionTunerPatch"},
                     drop_files={"krea2_turbo_fp8_scaled.safetensors"})
    with pytest.raises(ValueError) as e:
        _direct(models="24gb")
    msg = str(e.value)
    assert posted == []
    assert "RES4LYF" in msg and "ComfyUI-KJNodes" in msg
    assert "krea2_turbo_fp8_scaled.safetensors" in msg
    assert "ltx-2.3-22b-dev-fp8" not in msg  # found in its sub-folder


def test_a_complete_setup_queues_as_before(comfyui):
    posted = comfyui()
    res = _direct(models="24gb")
    assert len(posted) == len(res["prompt_ids"]) > 0
    # the files were found by name in their sub-folders and the jobs use ComfyUI's listed names
    assert any(n["inputs"].get("unet_name") == "Krea2\\krea2_turbo_fp8_scaled.safetensors"
               for g in posted for n in g.values())


def test_a_preview_checks_nothing(comfyui):
    posted = comfyui(missing_nodes={"ClownsharKSampler_Beta"})
    _direct(models="24gb", queue=False)
    assert posted == []


if __name__ == "__main__":  # python tests/test_queue_errors.py (pytest from the pack root would import the pack itself)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.exit(pytest.main([__file__, "-q", f"--confcutdir={here}", f"--rootdir={here}", "--import-mode=importlib",
                          "-p", "no:cacheprovider"]))
