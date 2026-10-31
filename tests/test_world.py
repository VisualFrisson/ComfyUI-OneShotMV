"""The World line wins over the defaults: what it says about the cast (no people, only women, set
today) is a rule, a character it describes keeps its words, and a creature cast gets creature
prompts and checks all the way to the clips. Reported Oct 30 by the author: an elemental World came
back as 60 people with jobs, and a "No men" World as 7 women and 5 men.

The language model is a scripted stand-in (it records every prompt and answers like Gemma), so this
runs without ComfyUI or a GPU:  python tests/test_world.py"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from fv import casting, critic, theme  # noqa: E402

ELEMENTS = ("A hidden northern valley at golden hour where the land itself has come alive, played straight like a "
            "quiet, photoreal fantasy nature film, all invented elemental creatures and no people, each with deep-set "
            "eyes and a wide mouth made of its own element: a snowdrift creature of packed powder with an icicle "
            "fringe, a glacier creature of clear blue ice with meltwater running down its sides, a moss creature with "
            "a back of ferns, a tide creature of sea foam and kelp with sea-glass eyes; a blue glacier edge, a mossy "
            "old-growth forest, a rocky tide line; low golden-hour sun, long blue shadows")
NO_MEN = ("A sunny beach in modern time, on a hot day, every single character is a woman in a swimsuit. No men. "
          "All characters are from 2020s.")
VILLAGE = ("A frozen fishing village in 1890, played straight: a net mender in her seventies with a weathered face and "
           "a grey bun, the harbour master in a brass-buttoned coat; the frozen harbour, the smokehouse; low winter sun")


def _char(role, gender="", age="", look="", wear="", place="on the frozen harbour", kind=None):
    c = {"role": role, "gender": gender, "age": age, "look": look, "wear": wear, "place": place,
         "light": "low winter sun from the south", "doing": "turning slowly toward the light", "face": "calm and proud"}
    if kind:
        c["kind"] = kind
    return c


def _world(chars):
    return json.dumps({"characters": chars, "places": [f"on place {i}" for i in range(12)],
                       "light": [f"low golden sun from the west number {i}" for i in range(8)],
                       "camera": "Shot on an unretouched modern digital camera in full colour.",
                       "actions": [f"turning slowly toward the light {i}" for i in range(10)],
                       "poses": [f"a calm, proud look number {i}" for i in range(10)], "grade": "golden"})


@pytest.fixture
def gemma(monkeypatch):
    """A scripted Gemma: answers each kind of prompt the way the real one does and records them."""
    asked = []
    state = {"world": "[]", "roles": []}

    def llm(prompt, seed=1, max_length=1600, log=print):
        asked.append(prompt)
        if "Invent the world of this video" in prompt:
            return state["world"]
        if "already in the video" in prompt:
            return json.dumps({"roles": state["roles"]})
        if prompt.startswith("A music video is set in this world") and "For each role below" in prompt:
            return "{}"
        m = re.search(r"in this order, one each: (.*?)\.\n", prompt, re.S)
        if m:  # FILL: one character per wanted item, written the way it was asked for
            out = []
            for item in m.group(1).split("; "):
                role = re.sub(r"^(?:an?|the)\s+", "", item.split(" (")[0].split(",")[0]).strip()
                g = (re.search(r"\((woman|man)\)", item) or [None, ""])[1]
                creature = "creatures, never people" in prompt
                out.append(_char(role, gender=g, age="" if creature else "40",
                                 look=f"made of grey stone like a {role}, with two dark hollows" if creature
                                 else f"short grey hair, the build of a {role}",
                                 wear="" if creature else f"the plain work clothes of a {role}",
                                 place=f"on the {role}'s own spot", kind="creature" if creature else None))
            return json.dumps({"characters": out})
        return ""
    monkeypatch.setattr(theme, "llm", llm)
    return asked, state


def _cast(bible, n=6):
    pack = casting.apply_theme(casting.load_pack("crowd"), bible)
    sizes = (["close", "medium", "wide"] * n)[:n]
    return casting.cast_voice(pack, n, 1, sizes, fmt="16:9", leads=list(range(n)), phases=["quiet"] * n)


# ------------------------------------------------------------------ what the World line says
@pytest.mark.parametrize("line, want", [
    (ELEMENTS, {"cast": "creatures", "gender": "", "now": False}),
    (NO_MEN, {"cast": "people", "gender": "woman", "now": True}),
    (VILLAGE, {"cast": "people", "gender": "", "now": False}),
    ("the gladiator games in ancient Rome", {"cast": "people", "gender": "", "now": False}),
    ("A 1965 airline, nothing modern: pilots, stewardesses and passengers", {"cast": "people", "gender": "", "now": False}),
    ("a barbershop quartet contest in 1910, men only", {"cast": "people", "gender": "man", "now": False}),
    ("a car factory run entirely by robots, no humans", {"cast": "creatures", "gender": "", "now": False}),
])
def test_rules(line, want):
    assert theme.rules(line) == want


def test_the_world_lines_own_creatures_keep_their_words():
    named = dict(theme.named_characters(ELEMENTS))
    assert named["snowdrift creature"] == "snowdrift creature of packed powder with an icicle fringe"
    assert named["tide creature"] == "tide creature of sea foam and kelp with sea-glass eyes"
    assert list(named) == ["snowdrift creature", "glacier creature", "moss creature", "tide creature"]


# ------------------------------------------------------------------ a creature World, to the clips
def test_a_creature_world_stays_creatures_even_when_the_model_writes_people(gemma, monkeypatch):
    asked, st = gemma
    st["world"] = _world([  # the first two as Gemma wrote them before: people with jobs
        _char("snowdrift creature", "man", "62", "weathered face, pale skin, silver hair", "a fur hat with ear flaps"),
        _char("glacier creature", "woman", "35", "fair skin, auburn hair in a Gibson roll", "a tweed skirt suit"),
        *[_char(f"{k} creature", look=f"made of {k} with two dark hollows and a wide mouth", kind="creature")
          for k in ("heather", "bark", "pebble", "foam", "frost", "fern")]])
    st["roles"] = ["lichen creature", "icicle creature", "seed creature", "driftwood creature"]
    b = theme.expand(ELEMENTS, seed=1, log=lambda *a: None)
    b = theme.extend(b, 12, seed=1, log=lambda *a: None)
    # what Gemma was asked: creatures, under the World line's rules, never the gender check
    assert "The theme is the brief and it wins" in asked[0] and "never a person" in asked[0]
    assert "different creatures" in asked[0] and '"gender"' not in asked[0]
    assert not any("For each role below" in a for a in asked)
    fill = next(a for a in asked if "one each:" in a)
    assert "as the theme describes it: moss creature with a back of ferns" in fill
    chars = b["characters"]
    assert len(chars) >= 12 and all(c.get("kind") == "creature" and not c.get("gender") for c in chars)
    snow = next(c for c in chars if c["role"] == "snowdrift creature")
    assert snow["who"] == "a snowdrift creature of packed powder with an icicle fringe"  # its words, not the man
    assert snow["wear"] == ""  # and not his fur hat
    for c in chars:
        assert not re.search(r"(?i)\b(aged \d|a (?:wo)?man\b|pale skin|silver hair|gibson)", c["who"]), c["who"]
    # the prompts the stills and clips are made from
    cast = _cast(b, 8)
    joined = " ".join(c["portrait"] for c in cast)
    assert "packed powder with an icicle fringe" in joined
    for c in cast:
        assert not re.search(r"(?i)invented person|grown adult|visible pores|flyaway hairs|aged \d", c["portrait"]), c["portrait"]
        assert "A single creature in frame" in c["portrait"]
        assert "lips" not in c["sing"] and "its mouth opens on every sung phrase" in c["sing"]
        assert not re.search(r"(?i)head-and-shoulders|waist-up", c["sing"]), c["sing"]
        assert c["check"]["kind"] == "creature"
    # the vision check asks about a creature, and a human-looking still is re-made
    rub = critic.rubric(cast[0]["check"], ELEMENTS, "")
    assert "exactly one creature" in rub and '"gender"' not in rub and '"human"' in rub
    assert critic.faults({"people": 0, "creatures": 1, "human": True, "head_in_frame": True, "text": False,
                          "matches": True}, cast[0]["check"]) == ["human"]
    assert critic.faults({"people": 1, "creatures": 1, "human": False, "head_in_frame": True, "text": False,
                          "matches": True}, cast[0]["check"]) == ["people"]
    assert "creature made only of" in critic.fix_lines(["human"], cast[0]["check"], "")


def test_a_creature_run_queues_creature_stills_checks_and_cutaways(gemma, monkeypatch):
    """The whole Director under the stand-in ComfyUI: the stills, the vision-check items, the
    cutaways and the take check's choices for a creature World."""
    import numpy as np
    import fake_comfy as fc
    from fv import director
    asked, st = gemma
    st["world"] = _world([_char(f"{k} creature", look=f"made of {k}, with two dark hollows and a wide mouth",
                                kind="creature", place=f"on the {k} slope")
                          for k in ("snowdrift", "glacier", "moss", "tide", "heather", "pebble", "frost", "fern")])
    st["roles"] = [f"{k} creature" for k in ("lichen", "icicle", "seed", "driftwood", "kelp", "cinder", "dew",
                                             "bramble", "slate", "pollen", "reed", "ember", "mist", "loam")]
    posted = fc.install(monkeypatch)
    sr = 44100
    t = np.arange(int(sr * 45)) / sr
    env = np.where(t > 20, 1.0, 0.4)
    x = (0.6 * (np.mod(t, 0.5) < 0.03) * env + 0.2 * np.sin(2 * np.pi * 220 * t) * env).astype(np.float32)
    res = director.direct(audio=np.stack([x, x]), sr=sr, out_root=tempfile.mkdtemp(), seconds=20, do_stems=False,
                          theme=ELEMENTS, log=lambda *a: None)
    p = json.load(open(os.path.join(res["run_dir"], "plan.json"), encoding="utf-8"))
    assert p["cast"] and all(c["check"]["kind"] == "creature" for c in p["cast"] if c.get("check"))
    stills = [n["inputs"]["text"] for g in posted for n in g.values()
              if n["class_type"] == "CLIPTextEncode" and "A single creature in frame" in str(n["inputs"].get("text"))]
    assert stills and "packed powder with an icicle fringe" in " ".join(stills)
    texts = " ".join(str(n["inputs"].get("text", "")) for g in posted for n in g.values() if n["class_type"] == "CLIPTextEncode")
    assert not re.search(r"(?i)an invented person|single grown adult|visible pores", texts)
    assert all(it["check"].get("kind") == "creature" for it in p["critic"]["items"] if it["kind"] == "portrait")
    cutaways = [r for r in p["renders"] if r.get("role") == "R"]
    assert all((r.get("broll") or {}).get("kind", "words") not in ("moment", "dance") for r in cutaways)
    assert "crowd_dress" not in (p.get("theme") or {})


# ------------------------------------------------------------------ one gender, set today
def test_no_men_means_no_men_and_today_means_today(gemma):
    asked, st = gemma
    st["world"] = _world([_char("lifeguard", "woman", "29", "tanned skin, sun-bleached ponytail", "a red swimsuit"),
                          _char("surfer", "man", "31", "sandy hair", "board shorts"),
                          _char("1950s pin-up model", "woman", "26", "victory rolls", "a polka-dot swimsuit"),
                          _char("1920s flapper", "woman", "24", "a black bob", "a beaded dress"),
                          _char("volleyball player", "woman", "27", "long black braids", "a sports swimsuit"),
                          _char("ice cream seller", "man", "45", "a grey beard", "a white apron"),
                          _char("yoga teacher", "woman", "38", "short silver hair", "a sage swimsuit"),
                          _char("paddleboarder", "woman", "33", "freckles, copper curls", "a navy swimsuit")])
    st["roles"] = ["sunbather", "1970s roller-disco queen", "kite surfer", "lifeguard captain", "beach photographer",
                   "snorkel guide", "sand sculptor", "lido swimmer", "beach runner"]
    b = theme.expand(NO_MEN, seed=1, log=lambda *a: None)
    b = theme.extend(b, 10, seed=1, log=lambda *a: None)
    assert "Every character is a woman" in asked[0] and "The theme is set today" in asked[0]
    assert '"gender": "woman" for every character' in asked[0] and "costume historian" not in asked[0]
    assert not any("For each role below" in a for a in asked)  # no gender re-check to flip anyone
    fills = [a for a in asked if "one each:" in a]
    assert fills and all("(man)" not in a for a in fills)
    chars = b["characters"]
    assert len(chars) >= 10
    assert {c["gender"] for c in chars} == {"woman"}
    assert not any(re.search(r"19\d0s", c["role"]) for c in chars), [c["role"] for c in chars]


# ------------------------------------------------------------------ a person World with described characters
def test_described_characters_keep_their_words_and_the_rest_fills_in(gemma):
    asked, st = gemma
    st["world"] = _world([_char("net mender", "woman", "72", "olive skin, short black hair", "a wool shawl"),
                          _char("harbour master", "man", "50", "a ruddy face and a white beard", "a navy coat"),
                          *[_char(f"fisher {k}", "man" if k % 2 else "woman", "40", "weathered skin", "oilskins")
                            for k in range(5)]])
    b = theme.expand(VILLAGE, seed=1, log=lambda *a: None)
    mender = next(c for c in b["characters"] if c["role"] == "net mender")
    master = next(c for c in b["characters"] if c["role"] == "harbour master")
    # her look is described: the line's words, not the model's olive skin and black hair
    assert mender["who"] == "a net mender in her seventies with a weathered face and a grey bun"
    assert mender["look"] == "net mender in her seventies with a weathered face and a grey bun"
    # his look isn't: the line's words first, then an age, a gender and the model's face
    assert master["who"] == "a harbour master in a brass-buttoned coat, a man aged 50, a ruddy face and a white beard"
    assert "The theme is the brief and it wins" in asked[0]


# ------------------------------------------------------------------ nothing else changes for people
def test_people_characters_are_read_as_before():
    """Main's _characters and this one give the same people for the same answer (only the World
    line's own characters and its rules change anything)."""
    old_src = subprocess.run(["git", "-C", ROOT, "show", "main:fv/theme.py"], capture_output=True, text=True).stdout
    if not old_src:
        pytest.skip("main isn't in this checkout")
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "fvold"))
    open(os.path.join(d, "fvold", "theme.py"), "w").write(old_src)
    for m in ("casting", "formats"):  # theme imports these lazily
        open(os.path.join(d, "fvold", f"{m}.py"), "w").write(
            subprocess.run(["git", "-C", ROOT, "show", f"main:fv/{m}.py"], capture_output=True, text=True).stdout)
    open(os.path.join(d, "fvold", "__init__.py"), "w").close()
    spec = importlib.util.spec_from_file_location("fvold.theme", os.path.join(d, "fvold", "theme.py"),
                                                  submodule_search_locations=None)
    sys.path.insert(0, d)
    try:
        old = importlib.import_module("fvold.theme")
    finally:
        sys.path.remove(d)
    answer = [_char("murmillo gladiator", "man", "34", "deep olive skin, short black curls and a broken nose",
                    "a bronze crested helmet and a manica on the sword arm"),
              _char("Vestal Virgin", "woman", "19", "pale skin, hair in six braids", "a white stola and a veil"),
              _char("senator", "", "61", "thin grey hair, a hooked nose", "a toga with a broad purple stripe"),
              _char("chariot racer", "man", "27", "sun-darkened skin, a shaved head, glasses",
                    "a short green tunic and leather wraps")]
    for line in ("the gladiator games in ancient Rome", "a 1965 airline: pilots, stewardesses and passengers"):
        assert theme._characters(answer, rules=theme.rules(line)) == old._characters(answer)
    assert spec is not None


if __name__ == "__main__":  # pytest from the pack root would import the pack itself
    sys.exit(pytest.main([__file__, "-q", f"--confcutdir={HERE}", f"--rootdir={HERE}", "--import-mode=importlib",
                          "-p", "no:cacheprovider"]))
