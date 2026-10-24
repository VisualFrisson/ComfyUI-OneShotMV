"""Style packs and casting: deterministic prompt combinatorics per performer."""
from __future__ import annotations

import glob
import json
import os
import random
import re

PACK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "packs")


def list_packs(mode: str | None = None) -> list[str]:
    out = []
    for f in sorted(glob.glob(os.path.join(PACK_DIR, "*.json"))):
        p = json.load(open(f, encoding="utf-8"))
        if mode is None or p.get("mode") == mode:
            out.append(p["name"])
    return out


def load_pack(name_or_path: str) -> dict:
    path = name_or_path if name_or_path.endswith(".json") else os.path.join(PACK_DIR, name_or_path + ".json")
    return json.load(open(path, encoding="utf-8"))


def _pick(rng: random.Random, seq, used: dict, key: str):
    """Pick from seq, avoiding recent repeats so consecutive performers differ."""
    seq = [s for s in seq if s is not None]
    if not seq:
        return ""
    recent = used.setdefault(key, [])
    choices = [s for s in seq if s not in recent] or seq
    v = rng.choice(choices)
    recent.append(v)
    if len(recent) > max(1, len(seq) // 2):
        recent.pop(0)
    return v


DEFAULT_FRAMING = {"close": {"shot": "Extreme close-up portrait", "lens": "on a wide-angle lens very close to the face",
                             "sing": "Handheld close-up of {who} singing straight into the camera. The lips form each word "
                                     "clearly with wide open mouth movements and jaw motion, natural breathing. Both hands stay below the frame."}}


# Dense vocals (rap, fast flows): a prompt that asks for a mouth that snaps on every syllable
# lifted SyncNet confidence on every dense shot it was tried on (n28: 3.3-4.0 -> 4.3-4.5,
# 2.2-2.6 -> 4.0); on sparse stretches no prompt, face or setting helped. See fv/lipsync.py.
ARTICULATE = ("The mouth moves sharply on every syllable: the lips close on the consonants and open wide on the "
              "vowels, the jaw snapping with each word, in time with the voice.")
_LIPS = re.compile(r"The lips form each word clearly with [^.]*\.")


def articulate(text: str, sentence: str | None = None) -> str:
    """Swap a sing prompt's mouth sentence for the articulation one (no-op if absent or done)."""
    sentence = sentence or ARTICULATE
    if sentence in text:
        return text
    m = _LIPS.search(text)
    if not m:
        return text
    tail = " Both hands stay below the frame." if "hands below" in m.group(0) and "Both hands" not in text else ""
    return text[:m.start()] + sentence + tail + text[m.end():]


def _frame(c: dict, size: str) -> dict:
    fr = c.get("framing") or DEFAULT_FRAMING
    return fr.get(size) or fr.get("close") or next(iter(fr.values()))


def cast_voice(pack: dict, n: int, seed: int, sizes: list[str] | None = None, fmt: str | None = None) -> list[dict]:
    """n performers, every portrait prompt finished with the realism brief (finish_prompt)."""
    out = _cast_voice(pack, n, seed, sizes, fmt)
    camera = ((pack.get("casting") or {}).get("theme") or {}).get("camera")
    for c in out:
        c["portrait"] = finish_prompt(c["portrait"], c.get("size", "close"), camera=camera)
        if c.get("twin_prompt"):
            c["twin_prompt"] = c["twin_prompt"].rstrip() + " " + edit_realism(c.get("size", "close") == "close")
    return out


def _cast_voice(pack: dict, n: int, seed: int, sizes: list[str] | None = None, fmt: str | None = None) -> list[dict]:
    """n performers for a voice pack: portrait prompt + singing (video) prompt.
    sizes[i] = 'close' | 'medium' | 'wide' framing for performer i (default close).
    fmt '16:9': a pack's `framing_landscape` overrides its framing (an extreme close-up on a
    wide frame crops the face at the eyes)."""
    c = pack["casting"]
    if fmt == "16:9" and c.get("framing_landscape"):
        c = dict(c, framing={**(c.get("framing") or DEFAULT_FRAMING), **c["framing_landscape"]})
    sizes = sizes or ["close"] * n
    if c.get("theme"):
        return _cast_theme(c, n, seed, sizes)
    if c.get("artist"):
        return _cast_artist(c, n, seed, sizes)
    if "people" in c:  # packs of coherent archetypes (who, setting, light)
        return _cast_people(c, n, seed, sizes)
    rng, used, out = random.Random(seed), {}, []
    for i in range(n):
        fr = _frame(c, sizes[i] if i < len(sizes) else "close")
        who = _pick(rng, c["who"], used, "who")
        skin = _pick(rng, c.get("skin", [""]), used, "skin")
        f = {k: _pick(rng, c.get(k, [""]), used, k) for k in
             ("hair", "detail", "angle", "light", "background", "wear", "time")}
        traits = [t for t in (skin, f["hair"]) if t]
        who_full = who + (" with " + " and ".join(traits) if traits else "")
        portrait = c["template"].format(who_full=who_full, who=who, shot=fr["shot"], lens=fr["lens"], **f)
        portrait = " ".join(portrait.replace(" ,", ",").replace(",,", ",").split())
        short = who + (" with " + f["hair"] if f["hair"] else "")
        sing = (c.get("sing") or fr["sing"]).format(who=short)
        sing = sing[0].upper() + sing[1:]
        twin = {"who": who_full, "detail": (f["detail"][:1].upper() + f["detail"][1:] + ".") if f["detail"] else "",
                "wear": ("Wearing " + f["wear"] + ".") if f["wear"] else "", "light": f["light"],
                "background": f["background"]}
        out.append({"id": i, "portrait": portrait, "sing": sing, "who": short, "twin": twin,
                    "size": sizes[i] if i < len(sizes) else "close", "seed": seed * 1000 + i * 7 + 1})
    return out


def _cast_artist(c: dict, n: int, seed: int, sizes: list[str]) -> list[dict]:
    """Same person every shot: the portrait is an identity edit of the artist photo. Close-ups
    change outfit, light and wall; medium and wide shots (with `places` / `actions` in the pack)
    put the artist somewhere new doing something: dancing, walking, sitting on a car hood."""
    rng, used, out = random.Random(seed), {}, []
    who = c.get("who", "the person")
    for i in range(n):
        size = sizes[i] if i < len(sizes) else "close"
        fr = _frame(c, size)
        f = {k: _pick(rng, c.get(k, [""]), used, k) for k in ("angle", "light", "background", "wear")}
        f["light"] = f["light"][:1].upper() + f["light"][1:]
        move = _pick(rng, (c.get("moves") or {}).get(size, []), used, "move_" + size) or \
            "The handheld camera sways gently."
        action = ""
        if size != "close" and c.get("places") and c.get("edit_wide"):
            action = _pick(rng, (c.get("actions") or {}).get(size, ["singing"]), used, "action_" + size)
            edit = c["edit_wide"].format(shot=fr["shot"][:1].lower() + fr["shot"][1:], lens=fr["lens"],
                                         angle=_pick(rng, c.get("angle_wide", c["angle"]), used, "angle_wide"),
                                         action=action, place=_pick(rng, c["places"], used, "place"),
                                         wear=f["wear"], light=f["light"])
            edit = " ".join(edit.split())
        else:
            edit = " ".join(c["edit"].format(shot=fr["shot"], lens=fr["lens"], **f).split())
        sing = fr["sing"].format(who=who, move=move, action=action or "singing")
        sing = " ".join((sing[0].upper() + sing[1:]).split())
        # the twin of this performer: same person, a different look
        g = {k: _pick(rng, c.get(k, [""]), used, k) for k in ("light", "background", "wear", "hair")}
        g["place"] = _pick(rng, c.get("places") or [g["background"]], used, "twin_place")
        g["place"] = re.sub(r"^(in|on|at)\s+", "", g["place"])
        g["soft_light"] = _pick(rng, c.get("twin_light") or c.get("light", [""]), used, "twin_light")
        twin_prompt = " ".join(c["twin"].format(**g).split())
        out.append({"id": i, "portrait": edit, "artist_edit": True, "sing": sing, "who": c.get("who", "the person"),
                    "twin_prompt": twin_prompt, "twin": {"who": "", "detail": "", "wear": g["wear"],
                                                         "light": g["light"], "background": g["background"]},
                    "size": size, "seed": seed * 1000 + i * 7 + 1})
    return out


# ------------------------------------------------------------------ in-scene shots
# Medium and wide shots where the performer does something in their world (walking, leaning in
# a doorway, working) instead of standing and singing at the lens. The n31 notes: "the walking
# shots were a great step up from the standard portrait-looking shots".
SCENE_ACTIONS = {
    "medium": ["leaning in a doorway", "sitting and leaning toward the camera", "leaning back against a wall",
               "walking slowly beside the camera", "turning toward the camera", "swaying to the beat",
               "looking up between lines, then back to the camera", "resting their arms on a railing"],
    "wide": ["walking slowly toward the camera", "striding past the camera", "sitting on a low wall",
             "dancing alone with small loose moves", "pacing slowly", "turning in a slow circle",
             "working with their hands, then looking up", "walking through the scene"],
}
SCENE_MOVES = {
    # close moves keep their distance: "very close" / "around the face" made LTX push in until the
    # face filled the frame (n41, the n30c "all mouth" note)
    "close": ["The handheld camera sways gently at the same distance.",
              "The camera drifts slowly sideways, keeping its distance.",
              "The handheld camera bobs gently with the beat, keeping its distance.",
              "The camera holds steady at head-and-shoulders distance."],
    "medium": ["The camera tracks slowly alongside.", "The camera holds at the same distance, waist-up.",
               "The handheld camera sways with the rhythm.", "The camera arcs slowly around them."],
    "wide": ["The camera follows from a few meters away.", "The camera pushes in slowly from far away.",
             "The camera arcs slowly around them.", "The camera holds still on a tripod."],
}
SCENE_SHOT = {"medium": ("Cinematic medium shot", "on a 35mm lens at chest height"),
              "wide": ("Cinematic wide full-body shot", "on a 24mm lens a few meters away")}
# The user's realism brief (Oct 24, 12:40: "the look feels really AI"): every image prompt ends with
# REALISM, close-up portraits also with CLOSE_SKIN (finish_prompt). The templates themselves only
# say no text; finish_prompt adds the photograph wording last.
REALISM_PERSON = ("Unretouched photograph with true-to-life detail: natural skin with visible pores, fine facial "
                  "hair, uneven tone and small blemishes, slightly asymmetrical features, flyaway hairs, creased "
                  "and worn fabric, everyday wear and clutter in the surroundings.")
REALISM_LIGHT = ("Light comes from real sources with mixed colour temperatures and natural falloff, a few clipped "
                 "highlights and deep, dense shadows.")
REALISM_LOOK = ("Muted, true-to-life colour, moderate depth of field, fine grain, slight lens softness toward the "
                "edges, imperfect off-centre framing.")
REALISM_CAMERA = REALISM_LIGHT + " " + REALISM_LOOK
REALISM = REALISM_PERSON + " " + REALISM_CAMERA
CLOSE_SKIN = ("Close-up skin detail: visible pores across the nose and cheeks, faint redness around the nostrils, "
              "fine lines at the corners of the eyes, a light natural sheen on the forehead, natural eye whites "
              "with tiny visible veins, a single soft catchlight, dry texture on the lips.")
# nobody in the picture (cutaways): the brief without the skin and hair
REALISM_SCENE = ("Unretouched photograph with true-to-life detail: everyday wear and clutter in the surroundings. "
                 + REALISM_CAMERA)
REAL_PHOTO = "No text, signs or lettering anywhere."
# one grown adult and nobody else: the Y2K retest drew two identical boy-band members in one
# portrait and photographers in a mirror; and nobody may look like a child (user, Oct 24)
ONE_ADULT = "A single grown adult in frame and nobody else, no other people in mirrors or reflections."
# the older photo wording at the end of the packs' templates, replaced by the brief
_OLD_TAIL = re.compile(r"(?i)(?:,\s*)?(?:visible pores and natural skin texture|natural colou?r|subtle film grain|"
                       r"photorealistic(?: film still)?|candid(?: [a-z-]+)? photograph|photorealistic photo with "
                       r"natural skin texture)\b\.?")


def _camera_part(camera: str | None) -> str:
    """The brief's camera half: the modern unretouched look, or the world's own period camera
    (theme bible "camera", e.g. a 35mm point-and-shoot with direct flash for 1999-2002: the user,
    Oct 24, on Y2K v3, "I don't feel that nostalgia from the time period")."""
    cam = " ".join((camera or "").split()).strip(" .")
    if not cam:
        return REALISM_CAMERA
    return f"{REALISM_LIGHT} {cam[0].upper() + cam[1:]}. Imperfect off-centre framing."


def finish_prompt(prompt: str, size: str = "close", people: bool = True, camera: str | None = None) -> str:
    """An image prompt with the realism brief at the end (CLOSE_SKIN too for close-ups); camera: the
    world's period camera in place of the modern look."""
    p = _OLD_TAIL.sub("", prompt)
    p = re.sub(r"\s*,\s*,", ",", p)
    p = re.sub(r"\s+([,.])", r"\1", p)
    p = re.sub(r"(?:,\s*)+\.", ".", p)
    p = re.sub(r"\.\s*\.", ".", p).strip(" ,")
    if not p.endswith("."):
        p += "."
    cam = _camera_part(camera)
    if not people:
        return " ".join(f"{p} {REALISM_SCENE.replace(REALISM_CAMERA, cam)}".split())
    tail = ONE_ADULT + " " + REALISM_PERSON + " " + cam + (" " + CLOSE_SKIN if size == "close" else "")
    return " ".join(f"{p} {tail}".split())


def edit_realism(close: bool = True) -> str:
    """For image edits that must keep the frame (the looks on the drop): the skin and fabric part of
    the brief, without the framing and depth of field."""
    return REALISM_PERSON + (" " + CLOSE_SKIN if close else "")
SCENE_PORTRAIT = ("{shot} {lens} of {who}, {action}, singing toward the camera with the mouth open mid-word, "
                  "{attitude}. {wear}Setting: {place}. {light}. An invented person, the only person in frame. "
                  + REAL_PHOTO)
SCENE_SING = {
    "medium": "Medium shot of {who} {action}, singing toward the camera. The lips form each word clearly with "
              "open mouth movements, shoulders and head move with the rhythm, natural breathing. {move} "
              "No one else in frame.",
    "wide": "Wide shot of {who} {action}, singing along, the lips forming the words, moving freely with the "
            "rhythm. {move} No one else in frame.",
}


def _scene_shot(rng, used, size: str, who: str, short: str, place: str, light: str, wear: str = "",
                actions: dict | None = None, attitude: str = "") -> tuple[str, str]:
    """(portrait prompt, singing prompt) for an in-scene medium or wide shot."""
    acts = (actions or SCENE_ACTIONS).get(size) or SCENE_ACTIONS[size]
    action = _pick(rng, acts, used, "scene_action_" + size)
    shot, lens = SCENE_SHOT[size]
    light = light[:1].upper() + light[1:] if light else "Natural light"
    portrait = SCENE_PORTRAIT.format(shot=shot, lens=lens, who=who, action=action, place=place, light=light,
                                     wear=("Wearing " + wear + ". ") if wear else "",
                                     attitude=attitude or ATTITUDE)
    move = _pick(rng, SCENE_MOVES[size], used, "scene_move_" + size)
    sing = SCENE_SING[size].format(who=short, action=action, move=move)
    return " ".join(portrait.split()), " ".join((sing[0].upper() + sing[1:]).split())


def _bg_place(bg: str) -> str:
    """'a dusty corral and adobe walls behind' -> 'a dusty corral and adobe walls'."""
    return re.sub(r"\s+behind$", "", bg.strip().rstrip("."))


def _cast_people(c: dict, n: int, seed: int, sizes: list[str] | None = None) -> list[dict]:
    rng = random.Random(seed)
    srng, used = random.Random(seed * 13 + 5), {}
    people = list(c["people"])
    rng.shuffle(people)
    scene_share = float(c.get("scene_medium_share", 0.5))
    out = []
    for i in range(n):
        who, bg, light = people[i % len(people)]
        size = sizes[i] if sizes and i < len(sizes) else "close"
        short = who.split(" with ")[0].split(" in ")[0]
        if size == "wide" or (size == "medium" and srng.random() < scene_share):
            portrait, sing = _scene_shot(srng, used, size, who, short, _bg_place(bg), light,
                                         actions=c.get("actions"))
        else:
            portrait = c["template"].format(who=who, background=bg, light=light)
            sing = c["sing"].format(who=short)
            sing = sing[0].upper() + sing[1:]
        twin = {"who": who, "detail": "", "wear": "", "light": light, "background": bg}
        out.append({"id": i, "portrait": portrait, "sing": sing, "who": short, "twin": twin, "size": size,
                    "seed": seed * 1000 + i * 7 + 1})
    return out


# ------------------------------------------------------------------ themes
THEME_CLOSE = ("{shot} of {who}, {attitude}, singing straight into the lens with the mouth open mid-word, shot "
               "from {angle} {lens}. Wearing {wear}. {light}. Background: {place}. An invented person, the only "
               "one in frame. " + REAL_PHOTO)
ATTITUDE = "expressive and a little over the top"  # without a theme's own poses
THEME_CLOSE_SING = ("Head-and-shoulders shot of {who} singing straight into the camera. The lips form each word "
                    "clearly with wide open mouth movements, expressive lips and jaw moving with the lyrics, natural "
                    "breathing, small head bobs and eyebrow raises on the beat. {move} The framing stays "
                    "head-and-shoulders and never moves in closer. Both hands stay below the frame. No one else "
                    "in frame.")
# themed close-ups are head-and-shoulders (n41: they lip-sync far better than medium shots and
# don't read as "all mouth"); a pack's own 16:9 close framing still wins
THEME_CLOSE_FRAME = {"shot": "Head-and-shoulders portrait",
                     "lens": "from about one and a half metres away on a 50mm lens, with clear space above the "
                             "head and both shoulders in frame, the face about a third of the frame height"}
THEME_ANGLES = ["slightly below", "eye level", "slightly above", "a low angle", "a three-quarter angle"]


def _place_noun(p: str) -> str:
    return re.sub(r"^(in|on|at|inside|atop)\s+", "", p.strip(), flags=re.I)


def _short_who(who: str) -> str:
    return re.split(r",| with ", who)[0].strip()


_FEET = (r"shoes?|sneakers?|trainers|sandals?|slippers?|boots?|loafers?|heels|socks?|clogs?|flip-flops?|"
         r"footwear|wedges|moccasins?|espadrilles?|platforms?|mules|pumps|stilettos?|slingbacks?|brogues|oxfords")
FEET = re.compile(rf"\b({_FEET})\b", re.I)
# a woman's cut on a man reads as costume (the Y2K sample: men in white lace corsets)
WOMENS_WEAR = re.compile(r"\b(dress|skirt|miniskirt|corset|bustier|bralette|bra|halter|lingerie|gown|blouse|"
                         r"camisole|tube top|bodysuit|heels|stilettos?|pumps)\b", re.I)
MAN = re.compile(r"\b(man|men|guy|boy|gentleman|male|father|grandfather|husband)\b", re.I)
WOMAN = re.compile(r"\b(woman|women|girl|lady|female|mother|grandmother|wife)\b", re.I)


def _no_feet(wear: str) -> str:
    """The outfit without its footwear, for close and medium shots (n47: a look edit drew a head
    standing on shoes). Drops the clause that names the shoes instead of dropping the outfit:
    filtering whole outfits left two in the Y2K and Florida samples, so everyone wore them."""
    if not FEET.search(wear):
        return wear
    out = wear
    for _ in range(3):
        # from the last joiner before the shoes ("and", "with", a comma) to the end of that clause
        seg = r"(?:(?!\s(?:and|with|paired with|plus|over)\s)[^,;])"
        nxt = re.sub(rf"(?:,\s*|\s+(?:and|with|paired with|plus|over)\s+){seg}*?\b(?:{_FEET})\b{seg}*",
                     "", out, flags=re.I)
        if nxt == out:
            break
        out = nxt
    out = out.strip(" ,;")
    return out if len(out) >= 8 and not FEET.search(out) else wear


def _cast_theme(c: dict, n: int, seed: int, sizes: list[str]) -> list[dict]:
    """Voice casting from a theme's world bible (fv/theme.py): every performer a new character,
    outfit, place and light from the theme; wide shots (and half the medium ones) are in-scene
    shots of the character doing something there. Bibles with whole characters (who + outfit +
    place) keep them together, so a role reads at a glance and the outfit fits the person."""
    th = c["theme"]
    fr = c.get("framing") or DEFAULT_FRAMING
    rng, used, out = random.Random(seed), {}, []
    acts = {"medium": SCENE_ACTIONS["medium"], "wide": th.get("actions") or SCENE_ACTIONS["wide"]}
    chars = [ch for ch in (th.get("characters") or []) if ch.get("who") and ch.get("wear")]
    # every character once before any repeats (the Y2K retest showed one of 14 in four of 16 shots)
    order = []
    for i in range(n):
        size = sizes[i] if i < len(sizes) else "close"
        if chars:
            if not order:
                order = chars[:]
                rng.shuffle(order)
            ch = order.pop()
            who, wear = ch["who"], ch["wear"]
            place = ch.get("place") or _pick(rng, th["places"], used, "place")
        else:
            who = _pick(rng, th["who"], used, "who")
            pool = th["wear"]
            if MAN.search(who) and not WOMAN.search(who):
                pool = [w for w in pool if not WOMENS_WEAR.search(w)] or pool
            wear = _pick(rng, pool, used, "wear")
            place = _pick(rng, th["places"], used, "place")
        # shoes in a head-and-shoulders frame: n47's look edit drew a head standing on shoes
        if size != "wide":
            wear = _no_feet(wear)
        # the character's own light where the bible has one (the light of its place: v4's first
        # preview put studio softboxes in a shopping mall), else one from the world's list
        light = (ch.get("light") if chars else None) or _pick(rng, th["light"], used, "light")
        short = _short_who(who)
        # the theme's own poses, in its tone (fun, but a real photograph: the user's n43 brief)
        attitude = _pick(rng, th["poses"], used, "pose") if th.get("poses") else ATTITUDE
        attitude = attitude[:1].lower() + attitude[1:]
        if size == "wide" or (size == "medium" and rng.random() < 0.5):
            portrait, sing = _scene_shot(rng, used, size, who, short, _place_noun(place), light, wear, acts,
                                         attitude=attitude)
        else:
            f = THEME_CLOSE_FRAME if size == "close" else (fr.get(size) or DEFAULT_FRAMING["close"])
            portrait = THEME_CLOSE.format(shot=f["shot"], lens=f["lens"], who=who, wear=wear, attitude=attitude,
                                          angle=_pick(rng, THEME_ANGLES, used, "angle"),
                                          light=light[:1].upper() + light[1:], place=_place_noun(place))
            move = _pick(rng, SCENE_MOVES["close" if size == "close" else "medium"], used, "move")
            sing = THEME_CLOSE_SING.format(who=short, move=move)
        twin = {"who": who, "detail": "", "wear": "Wearing " + wear + ".", "light": light,
                "background": _place_noun(place)}
        out.append({"id": i, "portrait": " ".join(portrait.split()), "sing": " ".join(sing.split()),
                    "who": short, "twin": twin, "size": size, "seed": seed * 1000 + i * 7 + 1})
    return out


def apply_theme(pack: dict, bible: dict) -> dict:
    """A copy of the pack whose casting comes from the theme (who, wear, places, light, actions);
    the pack keeps its edit, FX and finish (the grade comes from theme.finish_for)."""
    if not bible:
        return pack
    p = json.loads(json.dumps(pack))
    c = p["casting"]
    neg = c.get("negative", "")
    neg = neg.replace("modern clothing in a period scene, ", "")  # the eras pack's period lock
    if p.get("mode") == "motion":
        rng = random.Random(len(bible["who"]) * 7 + 3)
        cast, used = [], {}
        for k in range(max(40, len(c.get("cast", [])))):
            who = _pick(rng, bible["who"], used, "who")
            wear = _pick(rng, bible["wear"], used, "wear")
            place = _pick(rng, bible["places"], used, "place")
            light = _pick(rng, bible["light"], used, "light")
            cast.append([f"{who}, wearing {wear}", f"{_place_noun(place)}, {light}"])
        c["cast"] = cast
    elif c.get("artist"):
        c["wear"] = bible["wear"]
        c["light"] = bible["light"]
        c["places"] = bible["places"]
        c["background"] = [_place_noun(pl) + " behind" for pl in bible["places"]]
        c["actions"] = {"medium": SCENE_ACTIONS["medium"], "wide": bible["actions"]}
    else:
        p["casting"] = {"theme": bible, "framing": c.get("framing"),
                        "framing_landscape": c.get("framing_landscape"), "negative": neg,
                        "dense_from": c.get("dense_from", 0.6)}
        return p
    c["negative"] = neg
    return p


def cast_motion(pack: dict, n: int, seed: int) -> list[dict]:
    """n characters for a motion pack: keyframe edit prompt + video prompt."""
    c = pack["casting"]
    rng = random.Random(seed)
    cast = list(c["cast"])
    rng.shuffle(cast)
    out = []
    for i in range(n):
        who, where = cast[i % len(cast)]
        edit = f"Replace the dancer with {who}. Replace the room with {where}. {c['keep']}"
        video = c["video"].format(who=who[0].upper() + who[1:], where="in " + where)
        out.append({"id": i, "edit": edit, "video": video, "who": who, "seed": seed * 1000 + i * 7 + 1})
    return out
