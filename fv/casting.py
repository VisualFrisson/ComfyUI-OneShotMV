"""Style packs and casting: deterministic prompt combinatorics per performer."""
from __future__ import annotations

import glob
import json
import os
import random

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


def cast_voice(pack: dict, n: int, seed: int) -> list[dict]:
    """n performers for a voice pack: portrait prompt + singing (video) prompt."""
    c = pack["casting"]
    if "people" in c:  # packs of coherent archetypes (who, setting, light)
        return _cast_people(c, n, seed)
    rng, used, out = random.Random(seed), {}, []
    for i in range(n):
        who = _pick(rng, c["who"], used, "who")
        skin = _pick(rng, c.get("skin", [""]), used, "skin")
        f = {k: _pick(rng, c.get(k, [""]), used, k) for k in
             ("hair", "detail", "angle", "light", "background", "wear", "time")}
        traits = [t for t in (skin, f["hair"]) if t]
        who_full = who + (" with " + " and ".join(traits) if traits else "")
        portrait = c["template"].format(who_full=who_full, who=who, **f)
        portrait = " ".join(portrait.replace(" ,", ",").replace(",,", ",").split())
        short = who + (" with " + f["hair"] if f["hair"] else "")
        sing = c["sing"].format(who=short)
        sing = sing[0].upper() + sing[1:]
        twin = {"who": who_full, "detail": (f["detail"][:1].upper() + f["detail"][1:] + ".") if f["detail"] else "",
                "wear": ("Wearing " + f["wear"] + ".") if f["wear"] else "", "light": f["light"],
                "background": f["background"]}
        out.append({"id": i, "portrait": portrait, "sing": sing, "who": short, "twin": twin,
                    "seed": seed * 1000 + i * 7 + 1})
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
