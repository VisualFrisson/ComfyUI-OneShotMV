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


if __name__ == "__main__":  # pytest from the pack root would import the pack itself
    sys.exit(pytest.main([__file__, "-q", f"--confcutdir={HERE}", f"--rootdir={HERE}", "--import-mode=importlib",
                          "-p", "no:cacheprovider"]))
