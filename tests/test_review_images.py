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
    cuts = [s["n"] for s in run["shots"] if s["kind"] == "broll"]
    assert run["review"]["state"] == "ready" and set(run["review"]["remake"]) == set(sung + cuts)
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
    assert st._run(name)["review"]["remake"] == [] and st._run(name)["review"]["swap"]
    with pytest.raises(ValueError, match="only cropped"):
        st._remake(name, [{"n": 1, "note": "x"}])
    assert st._release(name)["jobs"] > 1


def test_cutaways_can_be_made_again(studio):
    st, q, posted = studio
    res = _direct(artist_path=_photo(), review=True)
    d, name = res["run_dir"], os.path.basename(res["run_dir"])
    run = st._run(name)
    cuts = [s["n"] for s in run["shots"] if s["kind"] == "broll"]
    assert cuts and set(cuts) <= set(run["review"]["remake"])
    posted.clear()
    assert st._remake(name, [{"n": cuts[0], "note": "at dusk, wider"}])["stills"] == 1
    still = next(g for g in posted if "SaveImage" in _types(g))
    assert any("At dusk, wider." in str(n["inputs"].get("text", "")) for n in still.values())
    new = os.path.join(d, os.path.basename(_saves(still)[0]) + "_00001_.png")
    p = json.load(open(os.path.join(d, "plan.json")))
    held = [json.dumps(json.load(open(os.path.join(d, "jobs", f)))) for f in p["review"]["held"]]
    assert any(json.dumps(new)[1:-1] in h and "R\"" in h for h in held)  # the cutaway's clip reads it


def _faces(monkeypatch):
    """A face in every bright picture, none in a dark one. Again after the Studio has read a run:
    the edit plan reloads fv.faces."""
    from fv import faces
    monkeypatch.setattr(faces, "method", lambda: "haar")
    monkeypatch.setattr(faces, "detect_all", lambda rgb, size=320: [[0.5, 0.4, 0.3]] if rgb[0, 0, 0] > 60 else [])


def _own_images(monkeypatch, fit="crop"):
    """Three faces that sing and a dark place picture (a cutaway)."""
    from PIL import Image
    folder = tempfile.mkdtemp()
    for i in range(3):
        Image.new("RGB", (2016, 1152), (120 + i, 110, 100)).save(os.path.join(folder, f"face_{i}.png"))
    Image.new("RGB", (2016, 1152), (10, 10, 10)).save(os.path.join(folder, "place_0.png"))
    _faces(monkeypatch)
    return folder, _direct(world_images=folder, image_fit=fit, review=True)


def test_a_new_image_of_your_own_for_one_shot(studio, monkeypatch):
    st, q, posted = studio
    folder, res = _own_images(monkeypatch)
    d, name = res["run_dir"], os.path.basename(res["run_dir"])
    run = st._run(name)
    _faces(monkeypatch)
    assert run["review"]["remake"] == [] and run["review"]["swap"] == [s["n"] for s in run["shots"]]
    got = st._review_images(name)
    assert [x["name"] for x in got["images"]] == ["face_0.png", "face_1.png", "face_2.png", "place_0.png"]
    assert [x["sings"] for x in got["images"]] == [True, True, True, False]
    assert st._review_thumb(name, "world", "face_1.png")[:2] == b"\xff\xd8"
    sung = [s["n"] for s in run["shots"] if s["kind"] == "sing"]
    shown = st._shot_images(d, json.load(open(os.path.join(d, "plan.json"))))
    old = shown[sung[0]]["image"]
    sharing = [n for n, v in shown.items() if v["image"] == old and n != sung[0]]
    # a sung shot needs a face that can sing
    with pytest.raises(ValueError, match="no face big enough"):
        st._remake(name, [{"n": sung[0], "image": "place_0.png", "src": "world"}])
    with pytest.raises(ValueError, match="no image"):
        st._remake(name, [{"n": sung[0], "image": "../plan.json", "src": "world"}])
    posted.clear()
    out = st._remake(name, [{"n": sung[0], "image": "face_2.png", "src": "world"}])
    assert out["picked"] == 1 and out["stills"] == 0
    crop = next(g for g in posted if "SaveImage" in _types(g))
    assert any(str(n["inputs"].get("path", "")).endswith("face_2.png") for n in crop.values())
    p = json.load(open(os.path.join(d, "plan.json")))
    after = st._shot_images(d, p)
    assert after[sung[0]]["image"] == os.path.join(d, f"pick1s{sung[0]:03d}_00001_.png")
    assert all(after[n]["image"] == old for n in sharing)  # only that shot changes
    assert p["review"]["rounds"][0]["items"][0]["image"] == "face_2.png"


def test_an_added_image_for_a_cutaway(studio, monkeypatch):
    st, q, posted = studio
    import folder_paths
    from PIL import Image
    folder, res = _own_images(monkeypatch)
    d, name = res["run_dir"], os.path.basename(res["run_dir"])
    run = st._run(name)
    cut = next(s["n"] for s in run["shots"] if s["kind"] == "broll")
    _faces(monkeypatch)
    added = os.path.join(folder_paths.get_input_directory(), "oneshot-mv", "added", name)
    os.makedirs(added)
    Image.new("RGB", (1200, 1200), (12, 12, 12)).save(os.path.join(added, "street.jpg"))
    got = st._review_images(name)
    assert got["added"] == f"oneshot-mv/added/{name}"
    assert {"name": "street.jpg", "src": "added", "sings": False} == {k: v for k, v in got["images"][-1].items() if k != "url"}
    posted.clear()
    assert st._remake(name, [{"n": cut, "image": "street.jpg", "src": "added"}])["picked"] == 1
    p = json.load(open(os.path.join(d, "plan.json")))
    assert st._shot_images(d, p)[cut]["image"].endswith(f"pick1s{cut:03d}_00001_.png")
    r = next(r for r in p["renders"] if r["role"] == "R" and r["shot"] == cut - 1 and not r.get("unused"))
    assert r["key"].endswith(f"pick1s{cut:03d}_00001_.png")


def test_a_new_image_crops_around_the_face_or_extends(studio, monkeypatch):
    st, q, posted = studio
    import io
    import folder_paths
    from PIL import Image
    folder, res = _own_images(monkeypatch, fit="extend")
    d, name = res["run_dir"], os.path.basename(res["run_dir"])
    run = st._run(name)
    sung = next(s["n"] for s in run["shots"] if s["kind"] == "sing")
    _faces(monkeypatch)
    added = os.path.join(folder_paths.get_input_directory(), "oneshot-mv", "added", name)
    os.makedirs(added)
    Image.new("RGB", (800, 1200), (130, 120, 110)).save(os.path.join(added, "tall.png"))
    assert st._review_images(name)["fit"] == "extend"
    # crop: the video's window placed by the face (40% down), not in the middle
    got = st._review_fit(name, "added", "tall.png", "crop")
    qs = dict(x.split("=") for x in got["preview"].split("?")[1].split("&"))
    x, y, w, h = [int(v) for v in qs["box"].split("%2C")]
    assert got["fit"] == "crop" and got["face"] and (x, w) == (0, 800) and y < (1200 - h) / 2 - 50
    jpg = Image.open(io.BytesIO(st._review_thumb(name, "added", "tall.png", "crop", qs["box"].replace("%2C", ","))))
    assert jpg.width == 480 and abs(jpg.width / jpg.height - w / h) < 0.02
    # extend: the whole image, the sides to be drawn
    got = st._review_fit(name, "added", "tall.png", "extend")
    assert got["fit"] == "extend" and "box" not in got["preview"]
    jpg = Image.open(io.BytesIO(st._review_thumb(name, "added", "tall.png", "extend")))
    assert jpg.width == 480 and abs(jpg.width / jpg.height - w / h) < 0.02
    assert jpg.getpixel((5, jpg.height // 2))[0] < 110 and jpg.getpixel((240, jpg.height // 2))[0] > 115
    posted.clear()
    assert st._remake(name, [{"n": sung, "image": "tall.png", "src": "added", "fit": "extend"}])["picked"] == 1
    assert any("FVExtendCanvas" in _types(g) for g in posted)
    p = json.load(open(os.path.join(d, "plan.json")))
    assert p["review"]["rounds"][0]["items"][0]["fit"] == "extend"
    # For professional use only crops
    p["professional"] = True
    json.dump(p, open(os.path.join(d, "plan.json"), "w"))
    assert st._review_fit(name, "added", "tall.png", "extend")["fit"] == "crop"
    assert st._review_images(name)["extend"] is False


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
