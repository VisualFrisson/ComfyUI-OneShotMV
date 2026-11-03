"""Single Artist in a World from the user's images (Nov 2, 2026): a folder of images, Single Artist and a
photo of the artist. Every image with one person gets the artist: their face and hair, or, where the
image's person is another gender, a child or much older (the vision model looks first), the whole artist
in a new outfit that suits the place. Images with two people or nobody stay as they are.

Runs without ComfyUI:  python tests/test_artist_in_images.py"""
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
from fv import critic, director, faces, graphs  # noqa: E402

# what the vision model says about each picture, by file name
PEOPLE = {"artist.png": {"gender": "woman", "age": "young adult"},
          "face_01.png": {"gender": "woman", "age": "adult", "outfit": "a summer dress"},
          "face_02.png": {"gender": "man", "age": "adult", "outfit": "red lifeguard shorts and a whistle"},
          "small_01.png": {"gender": "woman", "age": "young adult", "outfit": "a sun hat"}}


@pytest.fixture
def world(monkeypatch):
    """A folder: two faces that sing, a small face (a cutaway), two people side by side and a view;
    a face detector and a vision model that know them by name; and the artist's photo."""
    from PIL import Image
    folder, opened = tempfile.mkdtemp(), []
    for name in ("face_01.png", "face_02.png", "small_01.png", "twins_01.png", "view_01.png"):
        Image.new("RGB", (2016, 1152), (120, 110, 100)).save(os.path.join(folder, name))
    photo = os.path.join(tempfile.mkdtemp(), "artist.png")
    Image.new("RGB", (600, 800), (130, 120, 110)).save(photo)
    real_open = Image.open

    def open_(path, *a, **k):
        opened.append(os.path.basename(str(path)))
        return real_open(path, *a, **k)

    def detect(rgb, size=320):
        name = opened[-1] if opened else ""
        return {"face": [[0.5, 0.4, 0.3]], "small": [[0.5, 0.4, 0.03]],
                "twins": [[0.3, 0.4, 0.2], [0.7, 0.4, 0.2]]}.get(name.split("_")[0], [])
    monkeypatch.setattr(Image, "open", open_)
    monkeypatch.setattr(faces, "method", lambda: "haar")
    monkeypatch.setattr(faces, "detect_all", detect)
    asked, freed = [], []
    monkeypatch.setattr(critic, "ask", lambda path, prompt, max_length=320: (
        asked.append(os.path.basename(path)) or json.dumps(PEOPLE.get(os.path.basename(path), {}))))
    monkeypatch.setattr(critic, "free", lambda: freed.append(1))
    return types.SimpleNamespace(folder=folder, photo=photo, asked=asked, freed=freed)


def _direct(folder, **kw):
    sr = 44100
    t = np.arange(int(sr * 45)) / sr
    env = np.where(t > 20, 1.0, 0.4)
    x = (0.6 * (np.mod(t, 0.5) < 0.03) * env + 0.2 * np.sin(2 * np.pi * 220 * t) * env).astype(np.float32)
    return director.direct(audio=np.stack([x, x]), sr=sr, out_root=tempfile.mkdtemp(), seconds=20, do_stems=False,
                           log=lambda *a: None, world_images=folder, **kw)


def _staged(res):
    """{the World image's file name: its staging job}."""
    d = os.path.join(res["run_dir"], "jobs")
    out = {}
    for f in sorted(os.listdir(d)):
        if f.split("_", 1)[1].startswith("img"):
            g = json.load(open(os.path.join(d, f)))
            src = next(n["inputs"]["path"] for n in g.values() if n["class_type"] == "FVLoadImagePath")
            out[os.path.basename(src)] = g
    return out


def _prompts(g):
    return [n["inputs"].get("prompt", "") for n in g.values() if n["class_type"].startswith("TextEncodeQwenImage")]


def _refs(g):
    return [n["inputs"]["path"] for n in g.values() if n["class_type"] == "FVLoadImagePath"]


def test_the_artist_goes_into_each_image_with_one_person(monkeypatch, world):
    fc.install(monkeypatch, artist=True)
    res = _direct(world.folder, artist_path=world.photo)
    p = json.load(open(os.path.join(res["run_dir"], "plan.json")))
    staged = _staged(res)
    # a woman artist: the woman's image gets her face and hair, the man's image the whole artist, newly dressed
    assert _prompts(staged["face_01.png"]) == [graphs.SWAP_FACE]
    body = _prompts(staged["face_02.png"])[0]
    assert "Dress them in red lifeguard shorts and a whistle, clothes that suit this place." in body
    assert world.photo in _refs(staged["face_01.png"]) and world.photo in _refs(staged["face_02.png"])
    # two people side by side and a place stay as they are
    assert "twins_01.png" in staged
    for name in ("twins_01.png", "view_01.png"):
        if name in staged:
            assert _prompts(staged[name]) == [] and world.photo not in _refs(staged[name])
    assert "small_01.png" in staged  # a small face in a cutaway gets the artist too
    assert _prompts(staged["small_01.png"]) == [graphs.SWAP_FACE]
    assert world.asked[0] == "artist.png" and "view_01.png" not in world.asked and "twins_01.png" not in world.asked
    assert world.freed  # the vision model is let go before the images are made
    sw = {it["name"]: it for it in p["world_images"]["swap"]["items"]}
    assert sw["face_02.png"]["how"] == "body" and sw["face_02.png"]["why"] == "a man in the image"
    assert sw["face_01.png"]["how"] == "face"
    # the take check compares every sung face with the artist's photo
    assert p["artist_path"] == world.photo and p["editor"] == "qwen"
    assert any(n["class_type"] == "FVFaceMatch" for n in staged["face_01.png"].values())


def test_without_the_vision_model_every_image_gets_the_face(monkeypatch, world):
    fc.install(monkeypatch, artist=True)

    def broken(*a, **k):
        raise RuntimeError("no Generate Text node")
    monkeypatch.setattr(critic, "ask", broken)
    res = _direct(world.folder, artist_path=world.photo)
    staged = _staged(res)
    assert _prompts(staged["face_02.png"]) == [graphs.SWAP_FACE]


def test_an_extended_image_loads_qwen_image_21_once(monkeypatch, world):
    fc.install(monkeypatch, artist=True)
    from PIL import Image
    Image.new("RGB", (800, 1200), (120, 110, 100)).save(os.path.join(world.folder, "face_03.png"))  # a tall photo
    res = _direct(world.folder, artist_path=world.photo, image_fit="extend")
    g = _staged(res)["face_03.png"]
    assert len(_prompts(g)) == 2 and _prompts(g)[0] == graphs.EXTEND and _prompts(g)[1] == graphs.SWAP_FACE
    assert sum(n["class_type"] == "UNETLoader" for n in g.values()) == 1


def test_for_professional_use_2511_puts_the_artist_in(monkeypatch, world):
    fc.install(monkeypatch, artist=True)
    res = _direct(world.folder, artist_path=world.photo, professional=True)
    g = _staged(res)["face_01.png"]
    assert any(n["class_type"] == "TextEncodeQwenImageEditPlus" for n in g.values())
    assert not any(n["class_type"] == "TextEncodeQwenImage21" for n in g.values())


def test_no_artist_leaves_the_images_as_they_are(monkeypatch, world):
    fc.install(monkeypatch, artist=True)
    res = _direct(world.folder)
    assert all(_prompts(g) == [] for g in _staged(res).values()) and not world.asked
    assert not json.load(open(os.path.join(res["run_dir"], "plan.json"))).get("artist_path")


def test_the_mismatch_rule():
    woman = {"gender": "woman", "age": "young adult"}
    assert director._mismatch({"gender": "woman", "age": "adult"}, woman) == ""
    assert director._mismatch({"gender": "man", "age": "young adult"}, woman) == "a man in the image"
    assert director._mismatch({"gender": "woman", "age": "child"}, woman) == "a child in the image"
    assert director._mismatch({"gender": "woman", "age": "older adult"}, woman) == "an older adult in the image"
    assert director._mismatch({"gender": "unclear", "age": "teen"}, woman) == ""


def test_the_studio_sends_the_artist_with_your_images(monkeypatch):
    fc.install(monkeypatch)
    srv = types.ModuleType("server")
    srv.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=None, prompt_queue=None))
    monkeypatch.setitem(sys.modules, "server", srv)
    from fv import studio
    base = {"song": "a.mp3", "world_mode": "images", "world_images": "E:/pics", "cast": "single artist"}
    g = studio._director_graph(dict(base, artist_image="oneshot-mv/me.png", artist_mode="describe",
                                    artist_description="x"))
    d = g["2"]["inputs"]
    assert d["cast"] == "single artist" and d["artist_photo"] == ["3", 0] and "artist_description" not in d
    assert g["3"]["inputs"]["image"] == "oneshot-mv/me.png" and d["world"] == ""
    assert studio._director_graph(base)["2"]["inputs"]["cast"] == "crowd"  # no photo: the images sing as they are
    folder = tempfile.mkdtemp()
    from PIL import Image
    Image.new("RGB", (64, 96)).save(os.path.join(folder, "face_01.png"))
    assert "Add a photo of the artist" in studio._cant_start(dict(base, world_images=folder))[0]


def test_the_setup_check_and_models_card(monkeypatch):
    from fv import setup_check
    fc.install(monkeypatch)  # the 24 GB set, without either image editor or DWPose
    need = " ".join(setup_check.report("24gb", artist=True, images=True)["render"])
    assert "qwen_image_2.1_int8_convrot" in need and "comfyui_controlnet_aux" not in need and "union" not in need
    assert setup_check.report("24gb", artist=True, images=True)["plan"] == []
    fc.install(monkeypatch, artist=True)
    assert setup_check.report("24gb", artist=True, images=True)["render"] == []
    groups = {g["name"]: [f["name"] for f in g["files"]] for g in setup_check.uses("24gb", artist=True, images=True)["groups"]}
    assert list(groups) == ["Video", "Image Edit", "Language"]
    assert not any("union" in f for f in groups["Video"])


if __name__ == "__main__":  # pytest from the pack root would import the pack itself
    sys.exit(pytest.main([__file__, "-q", f"--confcutdir={HERE}", f"--rootdir={HERE}", "--import-mode=importlib",
                          "-p", "no:cacheprovider"]))
