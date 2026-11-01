"""Single Artist (Oct 31, 2026): one person sings every line, from a photo or from a description.
The portraits come out in the video's shape (a tall phone photo keeps its head), Qwen-Image 2.1 makes
the portraits and the looks (Qwen-Image-Edit 2511 for professional use), a described artist's headshot
is the run's first image,
the crowd stays off unless it's turned on, and the setup check asks for the editor's files.

Runs without ComfyUI:  python tests/test_single_artist.py"""
import os
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import fake_comfy as fc  # noqa: E402
from fv import director, graphs, setup_check  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    graphs.use_models("auto")
    yield
    graphs.use_models("auto")
    graphs.M.pop("artist_editor_test", None)


def _direct(**kw):
    sr = 44100
    t = np.arange(int(sr * 45)) / sr
    env = np.where(t > 20, 1.0, 0.4)
    x = (0.6 * (np.mod(t, 0.5) < 0.03) * env + 0.2 * np.sin(2 * np.pi * 220 * t) * env).astype(np.float32)
    return director.direct(audio=np.stack([x, x]), sr=sr, out_root=tempfile.mkdtemp(), seconds=20, do_stems=False,
                           log=lambda *a: None, **kw)


def _photo():
    from PIL import Image
    path = os.path.join(tempfile.mkdtemp(), "artist.png")
    Image.new("RGB", (600, 1000), (120, 100, 90)).save(path)  # a tall phone photo
    return path


def _jobs(res):
    import json
    d = os.path.join(res["run_dir"], "jobs")
    return [(f, json.load(open(os.path.join(d, f)))) for f in sorted(os.listdir(d))]


def _types(g):
    return [n["class_type"] for n in g.values()]


def test_qwen_image_21_restages_the_photo_and_dresses_the_looks(monkeypatch):
    """The default editor (the Oct 31 test: likeness 0.84 portraits, 0.85 looks): the photo padded to the
    video's shape (a tall phone photo keeps its head), and each look on the drop with the artist photo as
    image 2. No Klein anywhere."""
    fc.install(monkeypatch, artist=True)
    res = _direct(artist_path=_photo())
    jobs = _jobs(res)
    ports = [g for f, g in jobs if "_p0" in f]
    looks = [g for f, g in jobs if f.split("_", 1)[1].startswith("k")]
    assert ports and looks
    for g in ports:
        fit = next(n for n in g.values() if n["class_type"] == "FVFitAspect")
        assert (fit["inputs"]["width"], fit["inputs"]["height"]) == (1344, 768)  # 16:9, not the photo's shape
    for g in ports + looks:
        t_ = _types(g)
        assert "TextEncodeQwenImage21" in t_ and "ReferenceLatent" not in t_ and "EmptyFlux2LatentImage" not in t_
        unet = next(n for n in g.values() if n["class_type"] == "UNETLoader")["inputs"]["unet_name"]
        assert unet.endswith("qwen_image_2.1_int8_convrot.safetensors")  # the 24 GB set
    for g in looks:
        enc = next(n for n in g.values() if n["class_type"] == "TextEncodeQwenImage21")["inputs"]
        assert "images.image_2" in enc and "same person" in enc["prompt"]


def test_for_professional_use_qwen_image_edit_2511_does_both(monkeypatch):
    """professional: Qwen-Image-Edit 2511 (Apache-2.0) for the portraits and the looks, with the 4-step
    Lightning LoRA."""
    fc.install(monkeypatch, artist=True)
    res = _direct(artist_path=_photo(), professional=True)
    jobs = _jobs(res)
    ports = [g for f, g in jobs if "_p0" in f]
    looks = [g for f, g in jobs if f.split("_", 1)[1].startswith("k")]
    assert ports and looks
    for g in ports + looks:
        assert "TextEncodeQwenImageEditPlus" in _types(g) and "TextEncodeQwenImage21" not in _types(g)
        ks = next(n for n in g.values() if n["class_type"] == "KSampler")["inputs"]
        assert (ks["steps"], ks["cfg"]) == (4, 1.0)  # the Lightning LoRA
        unet = next(n for n in g.values() if n["class_type"] == "UNETLoader")["inputs"]["unet_name"]
        assert unet.endswith("qwen_image_edit_2511_fp8mixed.safetensors")
    for g in looks:  # the artist photo rides along as image 2
        enc = next(n for n in g.values() if n["class_type"] == "TextEncodeQwenImageEditPlus")["inputs"]
        assert "image2" in enc and "same person" in enc["prompt"]
    import json
    assert json.load(open(os.path.join(res["run_dir"], "plan.json")))["editor"] == "qwen_edit"
    graphs.M["qwen_edit_lora"] = ""  # without the LoRA: ComfyUI's 40 steps at CFG 4
    g = graphs.artist_portrait(artist_path="a.png", prompt="x", seed=1, prefix="a", size1=(1344, 768),
                               size2=(2016, 1152), editor="qwen_edit").nodes
    ks = next(n for n in g.values() if n["class_type"] == "KSampler")["inputs"]
    assert (ks["steps"], ks["cfg"]) == (40, 4.0)


def test_each_editor_has_a_24gb_file_and_a_full_one():
    assert graphs._set_files("24gb")["qwen21_unet"] == "qwen_image_2.1_int8_convrot.safetensors"
    assert graphs._set_files("full")["qwen21_unet"] == "qwen_image_2.1_bf16.safetensors"
    assert graphs._set_files("24gb")["qwen_edit_unet"] == "qwen_image_edit_2511_fp8mixed.safetensors"
    assert graphs._set_files("full")["qwen_edit_unet"] == "qwen_image_edit_2511_bf16.safetensors"


def test_a_described_artist_is_made_first(monkeypatch):
    fc.install(monkeypatch, artist=True)
    res = _direct(artist_description="A singer in her late twenties with a shaved head and gold hoops.")
    jobs = _jobs(res)
    assert jobs[0][0].endswith("free_before_portraits.json") and jobs[1][0].endswith("artist_ref.json")
    ref = jobs[1][1]
    prompt = next(n["inputs"]["text"] for n in ref.values() if n["class_type"] == "CLIPTextEncode")
    assert "a singer in her late twenties with a shaved head and gold hoops" in prompt
    want = os.path.join(res["run_dir"], "artist_ref_00001_.png")
    for f, g in jobs[2:]:
        if "_p0" in f:
            assert any(n["class_type"] == "FVLoadImagePath" and n["inputs"]["path"] == want for n in g.values())


def test_the_crowd_stays_off_unless_turned_on(monkeypatch):
    fc.install(monkeypatch, artist=True)
    import json
    photo = _photo()
    text = lambda res: " ".join(c["portrait"] for c in json.load(open(os.path.join(res["run_dir"], "plan.json")))["cast"])
    assert "crowd" not in text(_direct(artist_path=photo)).lower()  # auto: the singer alone
    assert "crowd" in text(_direct(artist_path=photo, crowd="on")).lower()


def test_a_crowd_world_doesnt_dress_a_crowd_for_one_singer(monkeypatch):
    """The crowd's dress is a Gemma call made for the crowd cast; a single artist never uses it."""
    fc.install(monkeypatch, artist=True)
    from fv import theme as themes
    bible = {"theme": "a roller derby bout, a packed arena crowd", "characters": [],
             "wear": [f"a skater's kit number {i}" for i in range(6)], "light": ["hard white arena light"] * 4,
             "places": [f"on the rink, spot {i}" for i in range(6)], "actions": ["skating a lap"] * 4}
    dressed = []
    monkeypatch.setattr(themes, "expand", lambda theme, **k: dict(bible))
    monkeypatch.setattr(themes, "crowd_world", lambda theme: True)
    monkeypatch.setattr(themes, "crowd_dress", lambda b, **k: dressed.append(1) or "a sea of team colours")
    _direct(artist_path=_photo(), theme=bible["theme"])
    _direct(artist_path=_photo(), theme=bible["theme"], crowd="on")
    assert dressed == []


def test_the_setup_check_asks_for_the_editors_files(monkeypatch):
    fc.install(monkeypatch)  # the 24 GB set, without either editor or DWPose
    crowd = setup_check.report("24gb", artist=False)
    artist = " ".join(setup_check.report("24gb", artist=True)["render"])
    assert crowd["render"] == []
    assert "qwen_image_2.1_int8_convrot" in artist and "comfyui_controlnet_aux" in artist
    assert "klein" not in artist.lower() and "2511" not in artist
    pro = " ".join(setup_check.report("24gb", artist=True, editor="qwen_edit")["render"])
    assert "qwen_image_edit_2511_fp8mixed" in pro and "Lightning" in pro and "qwen_image_2.1" not in pro
    fc.install(monkeypatch, artist=True)
    assert setup_check.report("24gb", artist=True)["render"] == []
    assert setup_check.report("24gb", artist=True, editor="qwen_edit")["render"] == []


def test_the_studio_sends_the_artist(monkeypatch):
    fc.install(monkeypatch)
    import types
    srv = types.ModuleType("server")
    srv.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=None, prompt_queue=None))
    monkeypatch.setitem(sys.modules, "server", srv)
    from fv import studio
    base = {"song": "a.mp3", "theme": "x", "cast": "single artist"}
    g = studio._director_graph(dict(base, artist_mode="photo", artist_image="oneshot-mv/me.png"))
    assert g["3"] == {"class_type": "LoadImage", "inputs": {"image": "oneshot-mv/me.png"}}
    assert g["2"]["inputs"]["artist_photo"] == ["3", 0] and g["2"]["inputs"]["cast"] == "single artist"
    g = studio._director_graph(dict(base, artist_mode="describe", artist_image="oneshot-mv/me.png",
                                    artist_description="a crooner in his sixties"))
    assert "3" not in g and g["2"]["inputs"]["artist_description"] == "a crooner in his sixties"
    assert studio._cant_start(dict(base, artist_mode="describe")) == ["Add a photo of the artist, or describe a new one."]
    assert studio._director_graph({"song": "a.mp3"})["2"]["inputs"]["cast"] == "crowd"
    assert studio._director_graph(dict(base, artist_mode="describe", artist_description="x",
                                       professional=True))["2"]["inputs"]["professional"] is True
    assert studio._director_graph({"song": "a.mp3", "professional": True})["2"]["inputs"]["professional"] is False


if __name__ == "__main__":  # pytest from the pack root would import the pack itself
    sys.exit(pytest.main([__file__, "-q", f"--confcutdir={HERE}", f"--rootdir={HERE}", "--import-mode=importlib",
                          "-p", "no:cacheprovider"]))
