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


def cast_voice(pack: dict, n: int, seed: int, sizes: list[str] | None = None) -> list[dict]:
    """n performers for a voice pack: portrait prompt + singing (video) prompt.
    sizes[i] = 'close' | 'medium' | 'wide' framing for performer i (default close)."""
    c = pack["casting"]
    sizes = sizes or ["close"] * n
    if c.get("artist"):
        return _cast_artist(c, n, seed, sizes)
    if "people" in c:  # packs of coherent archetypes (who, setting, light)
        return _cast_people(c, n, seed)
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
        g = {k: _pick(rng, c.get(k, [""]), used, k) for k in ("light", "background", "wear")}
        twin_prompt = " ".join(c["twin"].format(**g).split())
        out.append({"id": i, "portrait": edit, "artist_edit": True, "sing": sing, "who": c.get("who", "the person"),
                    "twin_prompt": twin_prompt, "twin": {"who": "", "detail": "", "wear": g["wear"],
                                                         "light": g["light"], "background": g["background"]},
                    "size": size, "seed": seed * 1000 + i * 7 + 1})
    return out


def _cast_people(c: dict, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    people = list(c["people"])
    rng.shuffle(people)
    out = []
    for i in range(n):
        who, bg, light = people[i % len(people)]
        portrait = c["template"].format(who=who, background=bg, light=light)
        short = who.split(" with ")[0].split(" in ")[0]
        sing = c["sing"].format(who=short)
        sing = sing[0].upper() + sing[1:]
        twin = {"who": who, "detail": "", "wear": "", "light": light, "background": bg}
        out.append({"id": i, "portrait": portrait, "sing": sing, "who": short, "twin": twin,
                    "seed": seed * 1000 + i * 7 + 1})
    return out


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
