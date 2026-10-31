"""When ComfyUI refuses a job anyway, the user sees why, not "HTTP Error 400: Bad Request".

Runs without ComfyUI or a GPU:  python -m pytest tests/
The 400 replies below have the shape ComfyUI 0.43 sends (server.py post_prompt, execution.validate_prompt)."""
import http.server
import json
import os
import sys
import threading

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
    assert "krea2" in msg and "Update ComfyUI" in msg


# the checks before anything is queued: tests/test_setup_check.py


if __name__ == "__main__":  # python tests/test_queue_errors.py (pytest from the pack root would import the pack itself)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.exit(pytest.main([__file__, "-q", f"--confcutdir={here}", f"--rootdir={here}", "--import-mode=importlib",
                          "-p", "no:cacheprovider"]))
