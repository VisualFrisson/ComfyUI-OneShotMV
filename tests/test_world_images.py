"""World from a folder of images (Oct 31, 2026): the user's own images are the World and the cast.
Images with a face sing, the rest are cutaways; in order or random order, repeating when there are
more shots than images; images not in the video's shape are cropped around the face, or with
image_fit "extend" the ones far off it get new edges from Qwen-Image 2.1; no Krea 2, no Gemma and no
vision check.

Runs without ComfyUI:  python tests/test_world_images.py"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import fake_comfy as fc  # noqa: E402
from fv import director, faces, graphs, world_images as wi  # noqa: E402


def _folder(spec):
    """spec: [(name, (w, h))]; a file named face_* has a face in the middle, place_* none."""
    from PIL import Image
    d = tempfile.mkdtemp()
    for name, size in spec:
        Image.new("RGB", size, (120, 110, 100)).save(os.path.join(d, name))
    return d


@pytest.fixture
def detector(monkeypatch):
    """A face detector that finds a face in images whose colour is (120, 110, 100) and the file is
    a face_* image: the test reads the name from the last image opened."""
    opened = []
    from PIL import Image
    real_open = Image.open

    def open_(path, *a, **k):
        opened.append(os.path.basename(str(path)))
        return real_open(path, *a, **k)
    monkeypatch.setattr(Image, "open", open_)
    monkeypatch.setattr(faces, "method", lambda: "haar")
    monkeypatch.setattr(faces, "detect_all", lambda rgb, size=320: [[0.5, 0.4, 0.3]]
                        if opened and opened[-1].startswith("face") else [])
    return opened


def _direct(folder, **kw):
    sr = 44100
    t = np.arange(int(sr * 45)) / sr
    env = np.where(t > 20, 1.0, 0.4)
    x = (0.6 * (np.mod(t, 0.5) < 0.03) * env + 0.2 * np.sin(2 * np.pi * 220 * t) * env).astype(np.float32)
    return director.direct(audio=np.stack([x, x]), sr=sr, out_root=tempfile.mkdtemp(), seconds=20, do_stems=False,
                           log=lambda *a: None, world_images=folder, **kw)


def _jobs(res):
    d = os.path.join(res["run_dir"], "jobs")
    return [(f, json.load(open(os.path.join(d, f)))) for f in sorted(os.listdir(d))]


def _types(g):
    return [n["class_type"] for n in g.values()]


def test_faces_sing_and_the_rest_are_cutaways(monkeypatch, detector):
    posted = fc.install(monkeypatch, artist=True)
    folder = _folder([("face_01.png", (1664, 2496)), ("face_02.png", (2016, 1152)), ("place_01.png", (2016, 1152))])
    res = _direct(folder)
    p = json.load(open(os.path.join(res["run_dir"], "plan.json")))
    jobs = _jobs(res)
    names = [f.split("_", 1)[1][:-5] for f, _ in jobs]
    # no Krea 2 and no Gemma: the images are staged (cropped or extended), then sung
    assert not any("krea" in str(n["inputs"].get("unet_name", "")).lower() or n["inputs"].get("type") == "krea2"
                   for _, g in jobs for n in g.values())
    assert "zz_critic" not in names
    staged = {n: g for n, g in zip(names, (g for _, g in jobs)) if n.startswith("img")}
    assert len(staged) == 3
    for g in staged.values():  # cropped to the video's shape, nothing drawn in
        assert "ImageCrop" in _types(g) and "UNETLoader" not in _types(g)
    tall = next(g for g in staged.values() if any(n["inputs"].get("path", "").endswith("face_01.png")
                                                  for n in g.values()))
    crop = next(n for n in tall.values() if n["class_type"] == "ImageCrop")["inputs"]
    assert (crop["width"], crop["height"]) == (1664, 951)  # a 2:3 photo's 16:9 band, around the face
    assert crop["y"] == round(0.4 * 2496 - 0.4 * 951)  # the face (40% down the photo) 40% down the crop
    # every singer's clip starts from a face image; the cutaways from the place image
    sung = [r for r in p["renders"] if r["role"] == "A"]
    imgs = {c["id"]: c["image"] for c in p["cast"]}
    assert {os.path.basename(imgs[r["performer"]]) for r in sung} == {"face_01.png", "face_02.png"}
    assert [os.path.basename(imgs[r["performer"]]) for r in sung][:4] == ["face_01.png", "face_02.png"] * 2  # in order
    cut_keys = {os.path.basename(c) for c in p["world_images"]["cutaways"]}
    place_job = next(n for n, g in staged.items() if any(x["inputs"].get("path", "").endswith("place_01.png")
                                                          for x in g.values()))
    assert cut_keys == {f"{place_job}_00001_.png"}
    assert p["world_images"]["n_cutaway"] == 1
    assert p["finish"]["grade_amount"] == 0 and p["finish"]["halation"] == 0  # the images keep their colours
    assert posted  # queued


def test_extend_keeps_the_whole_image(monkeypatch, detector):
    """image_fit "extend": a 2:3 photo in 16:9 goes on a canvas and Qwen-Image 2.1 draws the sides, the
    original pasted back over the middle; a 3:2 photo (close to the shape) is still cropped a little."""
    fc.install(monkeypatch, artist=True)
    folder = _folder([("face_01.png", (1664, 2496)), ("face_02.png", (1728, 1152)), ("place_01.png", (2016, 1152))])
    res = _direct(folder, image_fit="extend")
    p = json.load(open(os.path.join(res["run_dir"], "plan.json")))
    assert p["world_images"]["fit"] == "extend"
    assert [it["fit"] for it in p["world_images"]["images"]] == ["extend", "crop", "crop"]
    path = lambda g: next(n["inputs"]["path"] for n in g.values() if n["class_type"] == "FVLoadImagePath")
    staged = {os.path.basename(path(g)): g for f, g in _jobs(res) if f.split("_", 1)[1].startswith("img")}
    tall = _types(staged["face_01.png"])
    assert {"FVExtendCanvas", "TextEncodeQwenImage21", "ImageCompositeMasked"} <= set(tall)
    unet = next(n for n in staged["face_01.png"].values() if n["class_type"] == "UNETLoader")["inputs"]["unet_name"]
    assert unet.endswith("qwen_image_2.1_int8_convrot.safetensors")
    for name in ("face_02.png", "place_01.png"):
        assert "ImageCrop" in _types(staged[name]) and "UNETLoader" not in _types(staged[name])
    # the extended photo's face on the final frame: 0.3 of 1664 px, scaled to fit 1152 px of height
    face_h = next(it for it in p["world_images"]["images"] if it["name"] == "face_01.png")["face_h"]
    assert face_h == round(0.3 * 1664 * (1152 / 2496) / 1152, 3)
    assert wi.fit_of(1664, 2496, 1152, 2016, "extend") == "crop"  # 2:3 in 9:16 loses only 16% of its width
    # For professional use crops: Extend draws with Qwen-Image 2.1 (reported research-only)
    pro = json.load(open(os.path.join(_direct(folder, image_fit="extend", professional=True)["run_dir"], "plan.json")))
    assert pro["world_images"]["fit"] == "crop" and pro["professional"] is True
    assert {it["fit"] for it in pro["world_images"]["images"]} == {"crop"}
    assert wi.fit_of(1664, 2496, 2016, 1152, "crop") == "crop"


def test_two_people_side_by_side_are_a_cutaway(monkeypatch, tmp_path):
    from PIL import Image
    Image.new("RGB", (2016, 1152)).save(tmp_path / "pair.png")
    Image.new("RGB", (2016, 1152)).save(tmp_path / "lead.png")
    monkeypatch.setattr(faces, "method", lambda: "haar")
    seen = []
    monkeypatch.setattr(faces, "detect_all", lambda rgb, size=320: seen.append(1) or (
        [[0.3, 0.4, 0.2], [0.7, 0.4, 0.18]] if len(seen) == 2 else [[0.5, 0.4, 0.3], [0.9, 0.9, 0.05]]))
    lead, pair = wi.inspect([str(tmp_path / "lead.png"), str(tmp_path / "pair.png")], 2016, 1152, log=lambda *a: None)
    assert lead["sings"] and not lead["group"]  # a small face far behind doesn't count
    assert pair["group"] and not pair["sings"]


def test_random_order_repeats_without_twice_in_a_row():
    items = [{"path": f"{i}.png"} for i in range(5)]
    seq = wi.sequence(items, 23, "random order", seed=3)
    assert len(seq) == 23 and all(a is not b for a, b in zip(seq, seq[1:]))
    assert {it["path"] for it in seq[:5]} == {it["path"] for it in items}  # each image once before any repeats
    assert wi.sequence(items, 7, "in order")[5:] == items[:2]


def test_the_crop_keeps_the_face():
    # a 2:3 photo in 16:9: the full width and a band of the height, the face's centre 40% down it
    assert wi.crop_box(1664, 2496, 2016, 1152, [0.5, 0.25, 0.3]) == [0, 244, 1664, 951]
    assert wi.crop_box(1664, 2496, 2016, 1152, [0.5, 0.05, 0.3]) == [0, 0, 1664, 951]  # kept inside the photo
    assert wi.crop_box(1664, 2496, 2016, 1152, None) == [0, 772, 1664, 951]  # no face: the middle
    # a close-up too big for that: the crop moves down for the mouth and chin, as far as the eyes allow
    cy, fh = 0.45 * 2496, 1.25 * 0.7 * 1664
    x, y, w, h = wi.crop_box(1664, 2496, 2016, 1152, [0.5, 0.45, 0.7])
    assert y > cy - 0.4 * h and y <= cy - 0.1 * fh <= y + 0.15 * h and cy + 0.3 * fh < y + h
    # in 9:16 the same photo loses width instead, the face kept in
    x, y, w, h = wi.crop_box(1664, 2496, 1152, 2016, [0.8, 0.3, 0.2])
    assert (y, h) == (0, 2496) and x <= 0.8 * 1664 <= x + w
    assert wi.size_of(0.4) == "close" and wi.size_of(0.2) == "medium" and wi.size_of(0.11) == "wide"


def test_no_face_anywhere_says_so(monkeypatch, detector):
    fc.install(monkeypatch, artist=True)
    folder = _folder([("place_01.png", (2016, 1152))])
    with pytest.raises(ValueError, match="face big enough to sing"):
        _direct(folder)
    with pytest.raises(ValueError, match="wasn't found"):
        _direct(os.path.join(folder, "nope"))


def test_only_extending_needs_qwen_image_21(monkeypatch):
    from fv import setup_check
    fc.install(monkeypatch)  # the 24 GB set, without either image editor
    assert not setup_check.problems(setup_check.probe(images=True))["files"]
    assert any("qwen_image_2.1" in f for f in setup_check.problems(setup_check.probe(images=True, extend=True))["files"])
    assert "qwen_image_2.1" not in " ".join(setup_check.report("24gb", images=True)["render"])
    assert "qwen_image_2.1_int8_convrot" in " ".join(setup_check.report("24gb", images=True, extend=True)["render"])
    fc.install(monkeypatch, artist=True)
    assert setup_check.report("24gb", images=True, extend=True)["render"] == []


def test_the_studio_sends_the_folder(monkeypatch):
    fc.install(monkeypatch)
    import types
    srv = types.ModuleType("server")
    srv.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=None, prompt_queue=None))
    monkeypatch.setitem(sys.modules, "server", srv)
    from fv import studio
    base = {"song": "a.mp3", "theme": "a night market", "cast": "single artist", "artist_mode": "describe",
            "artist_description": "x"}
    g = studio._director_graph(dict(base, world_mode="images", world_images="E:/pics", image_order="random order"))
    d = g["2"]["inputs"]
    assert d["world_images"] == "E:/pics" and d["image_order"] == "random order"
    assert d["world"] == "" and d["cast"] == "crowd" and "artist_description" not in d
    assert d["image_fit"] == "crop"  # the default
    g = studio._director_graph(dict(base, world_mode="images", world_images="E:/pics", image_fit="extend"))
    assert g["2"]["inputs"]["image_fit"] == "extend"
    g = studio._director_graph(dict(base, world_mode="describe", world_images="E:/pics"))
    assert g["2"]["inputs"]["world_images"] == "" and g["2"]["inputs"]["world"] == "a night market"
    assert studio._cant_start({"song": "a.mp3", "world_mode": "images"}) == ["Pick a folder of images for the World."]
    folder = _folder([("face_01.png", (64, 96))])
    info = studio._images(folder)
    assert info["count"] == 1 and info["names"] == ["face_01.png"]


def test_a_test_machine_can_skip_the_finish(monkeypatch, tmp_path):
    """fv/local.json {"finish": "plain"}: no grade, grain or vignette on any cut (test runs)."""
    from fv import finish
    assert finish.plain({"plain": True}) and not finish.plain({})
    local = tmp_path / "local.json"
    local.write_text('{"finish": "plain"}')
    monkeypatch.setattr(finish, "local_settings", lambda: json.loads(local.read_text()))
    assert finish.plain({})


if __name__ == "__main__":  # pytest from the pack root would import the pack itself
    sys.exit(pytest.main([__file__, "-q", f"--confcutdir={HERE}", f"--rootdir={HERE}", "--import-mode=importlib",
                          "-p", "no:cacheprovider"]))
