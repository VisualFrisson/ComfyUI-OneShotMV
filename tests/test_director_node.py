"""The Director node and everything that drives it agree: the Studio's Brief, the SDK's API graph
and the example workflows send exactly the inputs the node has (Oct 31: one set of choices,
Song and World, the same in the Studio and the workflow; packs and motion mode removed).

Runs without ComfyUI or torch:  python tests/test_director_node.py"""
import importlib
import inspect
import json
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import fake_comfy as fc  # noqa: E402


@pytest.fixture
def node(monkeypatch):
    fc.install(monkeypatch)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    pkg = types.ModuleType("osmv")
    pkg.__path__ = [ROOT]
    monkeypatch.setitem(sys.modules, "osmv", pkg)
    for k in [k for k in sys.modules if k.startswith("osmv.")]:
        monkeypatch.delitem(sys.modules, k)
    return importlib.import_module("osmv.nodes").FVDirector


def _inputs(cls):
    it = cls.INPUT_TYPES()
    return list(it["required"]), list(it.get("optional", {}))


def test_the_node_shows_song_world_and_settings(node):
    req, opt = _inputs(node)
    assert req == ["audio", "world", "seconds", "start", "format", "quality", "models", "preview_only"]
    gone = {"mode", "pack", "twins", "twin_editor", "guide_strength", "dance_video", "dance_video_path", "framing",
            "artist_image", "artist_image_path", "broll_style", "staged", "look_groups", "queue_assemble",
            "queue_renders", "theme", "preset"}
    assert not gone & set(req + opt)
    params = inspect.signature(node.run).parameters
    assert set(req + opt) == set(params) - {"self"}


def test_the_studio_sends_what_the_node_has(node):
    from fv import studio
    req, opt = _inputs(node)
    g = studio._director_graph({"song": "a.mp3", "theme": "a 1965 airline", "render": True, "cuts_high": "every beat"})
    d = g["2"]["inputs"]
    assert set(req) <= set(d) and set(d) <= set(req + opt)
    assert d["world"] == "a 1965 airline" and d["preview_only"] is False and d["cuts_high"] == "every beat"
    assert studio._director_graph({"song": "a.mp3"})["2"]["inputs"]["preview_only"] is True


def test_the_sdk_graph_sends_what_the_node_has(node):
    req, opt = _inputs(node)
    g = json.load(open(os.path.join(ROOT, "api", "OneShotMV_director_api.json")))
    d = next(n["inputs"] for n in g.values() if n["class_type"] == "FVDirector")
    assert set(req) <= set(d) and set(d) <= set(req + opt)


def test_the_studio_asks_for_a_restart_after_an_update(node, monkeypatch):
    """The Studio reloads on every request, the node only when ComfyUI restarts: until then the old
    node refuses the new inputs, so the Brief says to restart instead of starting."""
    from fv import studio

    class OldDirector:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"audio": ("AUDIO",), "mode": (["voice", "motion"],), "theme": ("STRING",)}}
    reg = sys.modules["nodes"].NODE_CLASS_MAPPINGS
    monkeypatch.setitem(reg, "FVDirector", OldDirector)
    assert studio._cant_start({"render": True}) == [studio.RESTART]
    monkeypatch.setitem(reg, "FVDirector", node)
    assert studio.RESTART not in studio._cant_start({"render": True})


def test_the_studio_follows_the_plan_step(node, monkeypatch):
    """The planning progress reads the Director's log; its plan line lost "voice" with motion mode."""
    import time
    from fv import studio
    now = time.time()
    logger = types.ModuleType("app.logger")
    logger.get_logs = lambda: [{"t": now + 1, "m": "[OneShotMV] plan: 16 renders, window 48.83-78.83s, tempo 94"}]
    monkeypatch.setitem(sys.modules, "app", types.ModuleType("app"))
    monkeypatch.setitem(sys.modules, "app.logger", logger)
    parts = [{"id": "listen", "name": "Listening to the song", "s": 10},
             {"id": "stems", "name": "Separating the vocals", "s": 10}]
    assert studio._progress("x", {"t_run": now, "estimate": {"parts": parts}})["step"] == "Separating the vocals"


@pytest.mark.parametrize("name", ["OneShotMV.json", "OneShotMV_App.json"])
def test_the_example_workflows_fit_the_node(node, name):
    """A saved workflow stores the node's widget values by position; a value in the wrong place
    loads as the wrong setting (or fails validation)."""
    w = json.load(open(os.path.join(ROOT, "example_workflows", name), encoding="utf-8"))
    d = next(n for n in w["nodes"] if n["type"] == "FVDirector")
    it = node.INPUT_TYPES()
    specs = [(k, v) for part in ("required", "optional") for k, v in it[part].items()
             if isinstance(v[0], list) or v[0] in ("STRING", "FLOAT", "INT", "BOOLEAN")]
    vals = list(d["widgets_values"])
    for k, spec in specs:
        v = vals.pop(0)
        if isinstance(spec[0], list):
            assert v in spec[0], (k, v)
        else:
            assert isinstance(v, {"STRING": str, "FLOAT": (int, float), "INT": int, "BOOLEAN": bool}[spec[0]]), (k, v)
        if k == "seed":
            assert vals.pop(0) in ("fixed", "increment", "decrement", "randomize")
    assert vals == []
    assert {n["type"] for n in w["nodes"]} <= {"MarkdownNote", "LoadAudio", "FVDirector", "PreviewAny"}


def test_no_motion_workflow():
    assert sorted(os.listdir(os.path.join(ROOT, "example_workflows"))) == [
        "OneShotMV.json", "OneShotMV_App.json", "OneShotMV_Recut.json"]


if __name__ == "__main__":  # pytest from the pack root would import the pack itself
    sys.exit(pytest.main([__file__, "-q", f"--confcutdir={HERE}", f"--rootdir={HERE}", "--import-mode=importlib",
                          "-p", "no:cacheprovider"]))
