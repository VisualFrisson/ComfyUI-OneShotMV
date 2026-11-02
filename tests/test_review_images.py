"""Images first (Nov 1, 2026): the Plan's "Generate images and review" queues only the stills; the
Rendering screen then re-makes any the user picks, with a note, and "Make the video" queues the rest.
Also the Studio's Cancel for a Director that is still planning.

Runs without ComfyUI:  python tests/test_review_images.py"""
import json
import os
import sys
import tempfile
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import fake_comfy as fc  # noqa: E402
from fv import director, graphs  # noqa: E402


class Queue:
    """ComfyUI's prompt queue, enough for the Studio: what is running and pending, and deleting."""
    def __init__(self):
        self.running, self.pending, self.hist = [], [], {}

    def get_current_queue(self):
        return list(self.running), list(self.pending)

    def delete_queue_item(self, fn):
        for x in self.pending:
            if fn(x):
                self.pending.remove(x)
                return True
        return False

    def get_history(self, max_items=None, prompt_id=None):
        return self.hist


@pytest.fixture
def studio(monkeypatch):
    posted = fc.install(monkeypatch, artist=True)
    q = Queue()
    srv = types.ModuleType("server")
    srv.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=None, prompt_queue=q))
    monkeypatch.setitem(sys.modules, "server", srv)
    from fv import studio as st
    monkeypatch.setattr(st, "folder_paths", sys.modules["folder_paths"], raising=False)  # this test's output folder
    yield st, q, posted
    graphs.use_models("auto")


def _direct(**kw):
    import folder_paths
    sr = 44100
    t = np.arange(int(sr * 45)) / sr
    env = np.where(t > 20, 1.0, 0.4)
    x = (0.6 * (np.mod(t, 0.5) < 0.03) * env + 0.2 * np.sin(2 * np.pi * 220 * t) * env).astype(np.float32)
    return director.direct(audio=np.stack([x, x]), sr=sr, out_root=folder_paths.get_output_directory(), seconds=20,
                           do_stems=False, log=lambda *a: None, **kw)


def _photo():
    from PIL import Image
    path = os.path.join(tempfile.mkdtemp(), "artist.png")
    Image.new("RGB", (600, 1000), (120, 100, 90)).save(path)
    return path


def _types(g):
    return {n["class_type"] for n in g.values()}


def _saves(g):
    return [n["inputs"].get("filename_prefix", "") for n in g.values() if n["class_type"] in ("SaveImage", "SaveVideo")]


def test_images_first_queues_only_the_stills(studio):
    st, q, posted = studio
    res = _direct(artist_path=_photo(), review=True)
    assert posted and not any("LTXVAudioVAELoader" in _types(g) or "FVQC" in _types(g) for g in posted)
    assert any(s.endswith("/p000") or "/p0" in s for g in posted for s in _saves(g))  # the portraits
    p = json.load(open(os.path.join(res["run_dir"], "plan.json")))
    held = p["review"]["held"]
    assert p["review"]["state"] == "stills"
    assert any(f.endswith("A.json") for f in held) and any(f.endswith("_zz_qc.json") for f in held)
    run = st._run(os.path.basename(res["run_dir"]))
    assert run["stage"] == "review" and not run["stalled"]  # nothing queued, waiting for the user
    sung = [s["n"] for s in run["shots"] if s["kind"] == "sing"]
    assert run["review"]["state"] == "ready" and set(run["review"]["remake"]) == set(sung)
    # a plain run queues everything, as before
    posted.clear()
    _direct(artist_path=_photo())
    assert any("FVQC" in _types(g) for g in posted)


def test_remake_with_a_note_then_make_the_video(studio):
    st, q, posted = studio
    res = _direct(artist_path=_photo(), review=True)
    d, name = res["run_dir"], os.path.basename(res["run_dir"])
    shot = st._run(name)["review"]["remake"][0]
    posted.clear()
    out = st._remake(name, [{"n": shot, "note": "older, in a raincoat"}])
    assert out["stills"] == 1
    still = next(g for g in posted if "SaveImage" in _types(g))
    enc = next(n for n in still.values() if n["class_type"] == "TextEncodeQwenImage21")["inputs"]["prompt"]
    assert "Older, in a raincoat." in enc  # the note reaches the image editor's prompt
    new = os.path.join(d, os.path.basename(_saves(still)[0]) + "_00001_.png")
    assert new.endswith("_v1_00001_.png")
    p = json.load(open(os.path.join(d, "plan.json")))
    clip = next(f for f in p["review"]["held"] if f.endswith("A.json") and
                json.dumps(json.load(open(os.path.join(d, "jobs", f)))).count(json.dumps(new)[1:-1]))
    assert clip  # the held clip now reads the new still
    assert p["review"]["rounds"][0]["items"][0]["note"] == "older, in a raincoat"
    # while it is made, nothing else can start
    q.pending.append((1, "x", {"1": {"inputs": {"p": d}}}, {}, []))
    with pytest.raises(ValueError, match="wait"):
        st._release(name)
    q.pending.clear()
    posted.clear()
    got = st._release(name)
    assert got["jobs"] == len(posted) and any("FVQC" in _types(g) for g in posted)
    assert any("LTXVAudioVAELoader" in _types(g) for g in posted)
    assert st._run(name)["review"]["state"] == "released"
    with pytest.raises(ValueError, match="already"):
        st._release(name)
    with pytest.raises(ValueError, match="Result screen"):
        st._remake(name, [{"n": shot, "note": ""}])


def test_your_own_cropped_images_have_nothing_to_remake(studio, monkeypatch):
    st, q, posted = studio
    from PIL import Image
    from fv import faces
    folder = tempfile.mkdtemp()
    for i in range(3):
        Image.new("RGB", (2016, 1152), (120, 110, 100)).save(os.path.join(folder, f"face_{i}.png"))
    monkeypatch.setattr(faces, "method", lambda: "haar")
    monkeypatch.setattr(faces, "detect_all", lambda rgb, size=320: [[0.5, 0.4, 0.3]])
    res = _direct(world_images=folder, review=True)
    name = os.path.basename(res["run_dir"])
    assert st._run(name)["review"]["remake"] == []
    with pytest.raises(ValueError, match="only cropped"):
        st._remake(name, [{"n": 1, "note": "x"}])
    assert st._release(name)["jobs"] > 1


def test_cancel_a_director_still_planning(studio, monkeypatch):
    st, q, posted = studio
    import nodes
    stopped = []
    monkeypatch.setattr(nodes, "interrupt_processing", lambda value=True: stopped.append(value), raising=False)
    q.pending.append((5, "pid-queued", {"1": {"class_type": "FVDirector", "inputs": {}}}, {}, []))
    assert st._cancel_job("pid-queued")["removed"] == 1 and not q.pending and not stopped
    q.running.append((6, "pid-running", {"1": {"class_type": "FVDirector", "inputs": {}}}, {}, []))
    q.running.append((7, "someone-else", {}, {}, []))
    assert st._cancel_job("pid-running")["stopped"] == 1 and stopped == [True]
    stopped.clear()
    assert st._cancel_job("not-mine")["stopped"] == 0 and not stopped  # never another job


def test_the_studio_asks_for_images_first(studio):
    st, q, posted = studio
    base = {"song": "a.mp3", "theme": "x"}
    assert st._director_graph(dict(base, render=True, review=True))["2"]["inputs"]["review_images"] is True
    assert st._director_graph(dict(base, render=False, review=True))["2"]["inputs"]["review_images"] is False
    assert st._director_graph(dict(base, render=True))["2"]["inputs"]["review_images"] is False


if __name__ == "__main__":  # pytest from the pack root would import the pack itself
    sys.exit(pytest.main([__file__, "-q", f"--confcutdir={HERE}", f"--rootdir={HERE}", "--import-mode=importlib",
                          "-p", "no:cacheprovider"]))
