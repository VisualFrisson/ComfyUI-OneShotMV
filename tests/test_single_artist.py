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
    want = os.path.join(res["run_dir"], "artist_ref_00001_.png")
    faces = [(f, g) for f, g in jobs[2:] if "_p0" in f]
    assert faces and not any(f.endswith("d.json") for f, _ in faces)  # no drawn shots: the headshot is edited
    for f, g in faces:
        assert any(n["class_type"] == "FVLoadImagePath" and n["inputs"]["path"] == want for n in g.values())
        enc = next(n for n in g.values() if n["class_type"] == "TextEncodeQwenImage21")["inputs"]
        assert "Keep only their face from image 1" in enc["prompt"]


def test_with_drawn_shots_a_described_artist_is_the_look_alike(monkeypatch):
    fc.install(monkeypatch, artist=True)
    from fv import casting
    monkeypatch.setattr(casting, "DRAWN_SHOTS", True)
    res = _direct(artist_description="A singer in her late twenties with a shaved head and gold hoops.")
    jobs = _jobs(res)
    assert jobs[0][0].endswith("free_before_portraits.json") and jobs[1][0].endswith("artist_ref.json")
    ref = jobs[1][1]
    prompt = next(n["inputs"]["text"] for n in ref.values() if n["class_type"] == "CLIPTextEncode")
    assert "a singer in her late twenties with a shaved head and gold hoops" in prompt
    want = os.path.join(res["run_dir"], "artist_ref_00001_.png")
    faces = [(f, g) for f, g in jobs[2:] if "_p0" in f and not f.endswith("d.json")]
    drafts = [(f, g) for f, g in jobs[2:] if "_p0" in f and f.endswith("d.json")]
    assert faces and drafts  # the description is the look-alike each shot is drawn with
    for f, g in faces:
        assert any(n["class_type"] == "FVLoadImagePath" and n["inputs"]["path"] == want for n in g.values())
    for f, g in drafts:
        text = next(n["inputs"]["text"] for n in g.values() if n["class_type"] == "CLIPTextEncode")
        assert text.startswith("Photo of a singer in her late twenties with a shaved head and gold hoops, their hair styled in")


WHO = "a woman in her thirties with olive skin and short dark hair"


def test_each_shot_is_drawn_then_the_face_goes_in(monkeypatch):
    """Nov 2, 2026 (off by default since Nov 3, casting.DRAWN_SHOTS): Krea 2 draws each main image with a
    look-alike (the photo described by the vision model), in a hairstyle of its own, then the editor puts the
    artist's face in; the face check follows the portraits."""
    fc.install(monkeypatch, artist=True)
    from fv import casting
    monkeypatch.setattr(casting, "DRAWN_SHOTS", True)
    monkeypatch.setattr(director, "_artist_who", lambda photo, log=print: WHO)
    import json
    photo = _photo()
    res = _direct(artist_path=photo)
    jobs = _jobs(res)
    names = [f.split("_", 1)[1][:-5] for f, _ in jobs]
    drafts = [(n, g) for n, (f, g) in zip(names, jobs) if n.startswith("p") and n.endswith("d")]
    faces = [(n, g) for n, (f, g) in zip(names, jobs) if n.startswith("p") and not n.endswith("d")]
    assert drafts and len(drafts) == len(faces)
    texts = [next(x["inputs"]["text"] for x in g.values() if x["class_type"] == "CLIPTextEncode") for _, g in drafts]
    assert all(t.startswith(f"Photo of {WHO}, their hair styled in") for t in texts)
    assert len({t.split("styled in ")[1].split(".")[0].split(":")[0] for t in texts}) > 1  # the hair changes
    for n, g in faces:
        paths = [x["inputs"]["path"] for x in g.values() if x["class_type"] == "FVLoadImagePath"]
        assert paths[0].endswith(f"{n}d_00001_.png") and paths[1] == photo
        enc = next(x for x in g.values() if x["class_type"] == "TextEncodeQwenImage21")["inputs"]
        assert enc["prompt"] == graphs.FACE_IN and "images.image_2" in enc
    # one Krea 2 load (the drawn shots and the cutaways' stills), then one editor load
    free = names.index("free_after_drafts")
    assert max(names.index(n) for n, _ in drafts) < free < min(names.index(n) for n, _ in faces)
    assert all(names.index(n) < free for n in names if n.startswith("brollk"))
    p = json.load(open(os.path.join(res["run_dir"], "plan.json")))
    assert p["artist_who"] == WHO and p["artist_shots"] == "drawn"
    items = p["critic"]["items"]
    assert {it["name"] for it in items if it["kind"] == "artist"} == {n for n, _ in faces}
    assert all("Keep only their face" in it["fallback"] for it in items if it["kind"] == "artist")
    assert "zz_critic" in names and names.index("zz_critic") > names.index("free_after_portraits")


def test_by_default_every_shot_edits_the_photo_keeping_only_the_face(monkeypatch):
    """Nov 3, 2026: after the check run (the face went in on 2 of 16 drawn shots), every main image edits the
    photo keeping only the face; the photo isn't described and nothing is drawn first."""
    fc.install(monkeypatch, artist=True)
    asked = []
    monkeypatch.setattr(director, "_artist_who", lambda photo, log=print: asked.append(photo) or WHO)
    import json
    photo = _photo()
    res = _direct(artist_path=photo)
    jobs = _jobs(res)
    mains = [(f, g) for f, g in jobs if "_p0" in f]
    assert mains and not asked and not any(f.endswith("d.json") for f, _ in mains)
    for f, g in mains:
        assert any(n["class_type"] == "FVLoadImagePath" and n["inputs"]["path"] == photo for n in g.values())
        enc = next(n for n in g.values() if n["class_type"] == "TextEncodeQwenImage21")["inputs"]
        assert "Keep only their face from image 1" in enc["prompt"]
    p = json.load(open(os.path.join(res["run_dir"], "plan.json")))
    assert p["artist_shots"] == "photo" and not any(it["kind"] == "artist" for it in (p.get("critic") or {}).get("items", []))


def test_without_a_description_the_photo_keeps_only_the_face(monkeypatch):
    fc.install(monkeypatch, artist=True)
    from fv import casting
    monkeypatch.setattr(casting, "DRAWN_SHOTS", True)
    monkeypatch.setattr(director, "_artist_who", lambda photo, log=print: "")
    import json
    res = _direct(artist_path=_photo())
    jobs = _jobs(res)
    assert not any(f.endswith("d.json") and "_p0" in f for f, _ in jobs)
    for f, g in jobs:
        if "_p0" in f:
            enc = next(n for n in g.values() if n["class_type"] == "TextEncodeQwenImage21")["inputs"]
            assert "Keep only their face from image 1" in enc["prompt"]
    assert json.load(open(os.path.join(res["run_dir"], "plan.json")))["artist_shots"] == "photo"


def test_the_face_check_remakes_then_falls_back_to_the_photo(monkeypatch):
    """A drawn shot whose face isn't the artist's is made again (a new seed); if it still isn't, the last try
    edits the photo itself, keeping only the face; the closest take is kept."""
    fc.install(monkeypatch, artist=True)
    from fv import casting
    monkeypatch.setattr(casting, "DRAWN_SHOTS", True)
    monkeypatch.setattr(director, "_artist_who", lambda photo, log=print: WHO)
    import json
    from PIL import Image
    from fv import identity
    photo = _photo()
    res = _direct(artist_path=photo)
    d = res["run_dir"]
    monkeypatch.setattr(identity, "available", lambda: True)
    # the photo and the fallback's pictures (_c2) are the artist; the drawn shots and their re-makes are not
    monkeypatch.setattr(identity, "embed_path", lambda path: np.array([1.0, 0.0]) if (path == photo or "_c2" in path)
                        else np.array([0.3, 0.954]))
    p = json.load(open(os.path.join(d, "plan.json")))
    for it in p["critic"]["items"]:
        Image.new("RGB", (64, 64)).save(it["image"])
    director.critic_pass(d, pass_idx=0, queue=False, log=lambda *a: None)
    remade = sorted(f for f in os.listdir(os.path.join(d, "jobs")) if f.startswith("critic1_"))
    assert remade and all(f.endswith("_c1.json") for f in remade)
    g = json.load(open(os.path.join(d, "jobs", remade[0])))
    assert any(n["class_type"] == "TextEncodeQwenImage21" and n["inputs"]["prompt"] == graphs.FACE_IN for n in g.values())
    p = json.load(open(os.path.join(d, "plan.json")))
    for it in p["critic"]["items"]:
        Image.new("RGB", (64, 64)).save(it["image"])
    director.critic_pass(d, pass_idx=1, queue=False, log=lambda *a: None)
    fall = sorted(f for f in os.listdir(os.path.join(d, "jobs")) if f.startswith("critic2_"))
    g = json.load(open(os.path.join(d, "jobs", fall[0])))
    enc = next(n for n in g.values() if n["class_type"] == "TextEncodeQwenImage21")["inputs"]
    assert "Keep only their face from image 1" in enc["prompt"]
    assert next(n for n in g.values() if n["class_type"] == "FVLoadImagePath")["inputs"]["path"] == photo
    p = json.load(open(os.path.join(d, "plan.json")))
    for it in p["critic"]["items"]:
        Image.new("RGB", (64, 64)).save(it["image"])
    director.critic_pass(d, pass_idx=2, queue=False, log=lambda *a: None)
    p = json.load(open(os.path.join(d, "plan.json")))
    assert p["critic"]["done"] and all(it["final"].endswith("_c2_00001_.png") for it in p["critic"]["items"])
    a = next(r for r in p["renders"] if r["role"] == "A")
    assert a["portrait"].endswith("_c2_00001_.png")


@pytest.mark.parametrize("said, who", [
    ("A woman in her mid-twenties with fair skin and long blonde hair.", "a woman in her mid-twenties with fair skin and long blonde hair"),
    ("**a man in his forties, olive skin, short grey hair**", "a man in his forties, olive skin, short grey hair"),
    ("a woman resembling a famous singer, with fair skin", "a woman"),
    ("Sorry, I can't describe people in photos.", ""),
])
def test_the_photo_is_described_without_names(monkeypatch, said, who):
    import types
    fake = types.SimpleNamespace(ask=lambda *a, **k: said, free=lambda: None)
    monkeypatch.setitem(sys.modules, "fv.critic", fake)
    import fv
    monkeypatch.setattr(fv, "critic", fake, raising=False)
    assert director._artist_who("photo.png", log=lambda *a: None) == who


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


def test_a_period_world_locks_the_artists_outfits_and_places():
    """Nov 3, 2026: a Single Artist in a period World gets the period lock and the World's camera a Crowd gets,
    on the drawn shot, its fallback and the looks; whoever walks into a clip is dressed for the period too."""
    from fv import casting
    camera = "Shot on 35mm colour film with a long lens"
    bible = {"theme": "the gladiator games in ancient Rome", "camera": camera,
             "wear": [f"a bronze helmet and leather armour, style {i}" for i in range(6)],
             "light": ["hard midday sun"] * 4, "places": [f"on the arena sand, spot {i}" for i in range(6)],
             "actions": ["raising a sword"] * 4}
    pk = casting.apply_theme(casting.load_pack("artist"), bible)
    cast = casting.cast_voice(pk, 3, 1, ["close", "close", "close"])
    lock = casting.era_line(bible["theme"])
    assert lock and "ancient Rome" in lock
    for c in cast:
        for key in ("portrait", "draft", "portrait_a", "twin_prompt"):
            assert lock in c[key], key
        assert camera in c["draft"] and camera in c["portrait"]
        assert "ancient Rome" in c["sing"]  # casting.period_background
        assert "Photo of @WHO@" in c["draft"]  # still the artist, not the World's characters
    modern = casting.cast_voice(casting.apply_theme(casting.load_pack("artist"), dict(bible, theme="a beach party")),
                                2, 1, ["close", "close"])
    assert all("period-accurate" not in c["draft"] for c in modern)


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
    # since 1.0.0 there is no For professional use: an old brief's flag isn't sent
    assert "professional" not in studio._director_graph({"song": "a.mp3", "professional": True})["2"]["inputs"]


def test_the_models_card_lists_what_a_video_loads(monkeypatch):
    """The Studio's Models card: each file's name, its ComfyUI folder and whether it's installed, in four
    groups, each model with its LoRAs."""
    fc.install(monkeypatch)  # the 24 GB set, without either image editor
    def groups(set_="24gb", **k):
        return {g["name"]: {f["name"]: (f["folder"], f["have"]) for f in g["files"]}
                for g in setup_check.uses(set_, **k)["groups"]}
    crowd = groups()
    assert list(crowd) == ["Video", "Image", "Language"]
    assert crowd["Image"] == {"krea2_turbo_fp8_scaled.safetensors": ("diffusion_models", True),
                               "qwen3vl_4b_fp8_scaled.safetensors": ("text_encoders", True),
                               "qwen_image_vae.safetensors": ("vae", True)}
    assert crowd["Video"]["ltx-2.3-22b-dev-fp8.safetensors"] == ("checkpoints", True)
    assert crowd["Video"]["ltx-2.3-spatial-upscaler-x2-1.1.safetensors"][0] == "latent_upscale_models"
    assert len(crowd["Video"]) == 6 and crowd["Language"] == {"gemma_3_12B_it_fp8_scaled.safetensors":
                                                                    ("text_encoders", True)}
    one = groups(artist=True)
    assert list(one) == ["Video", "Image", "Image Edit", "Language"]
    assert one["Image Edit"]["qwen_image_2.1_int8_convrot.safetensors"] == ("diffusion_models", False)  # not here
    assert "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors" in one["Video"]
    pro = groups(artist=True, professional=True)
    assert "qwen_image_edit_2511_fp8mixed.safetensors" in pro["Image Edit"] and not any("2.1" in f for f in pro["Image Edit"])
    assert pro["Image Edit"]["Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"][0] == "loras"
    assert "qwen_image_vae.safetensors" in pro["Image"] and "qwen_image_vae.safetensors" not in pro["Image Edit"]
    full = groups("full")
    assert {"krea2_raw_bf16.safetensors", "krea2_turbo_lora_rank_64_bf16.safetensors"} <= set(full["Image"])
    assert list(groups(images=True)) == ["Video", "Language"]  # your own images: nothing to make
    assert list(groups(images=True, extend=True)) == ["Video", "Image Edit", "Language"]
    assert "Image Edit" not in groups(images=True, extend=True, professional=True)  # professional crops
    # every file of both sets, the image editors included, links to its download
    for set_ in ("24gb", "full"):
        for k in ({}, {"artist": True}, {"artist": True, "professional": True}):
            for g in setup_check.uses(set_, **k)["groups"]:
                for f in g["files"]:
                    assert f["url"].startswith("https://huggingface.co/") and f["url"].endswith("/" + f["name"]), f


if __name__ == "__main__":  # pytest from the pack root would import the pack itself
    sys.exit(pytest.main([__file__, "-q", f"--confcutdir={HERE}", f"--rootdir={HERE}", "--import-mode=importlib",
                          "-p", "no:cacheprovider"]))
