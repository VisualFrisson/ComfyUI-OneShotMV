"""Vision check: every portrait and b-roll still is looked at before any video is made from it.

The user, Oct 26: "girls with beards, guys wearing girls clothes", modern glasses in ancient Rome.
Gemma 3 12B (the LTX-2.3 text encoder every install already has, run through ComfyUI's Generate
Text with an image) answers a fixed set of yes/no questions about the picture against what the
character should be: how many people, which gender, facial hair, glasses, the whole head in frame,
any writing, anything from after the World's period, and whether the clothes match. The Oct 26
test on 28 frames: glasses on all 12 frames that had them and on none of the 16 that didn't,
the crowds counted, the bearded "priestess" caught; about 7 s a picture. Its free-form "what is
modern" list also names the camera and the lighting, so those are filtered out.

fit_face(): the drop's match cuts land the next face where the last one was. Every drop portrait
is re-framed (scaled up and re-centred, never down) so the face has one size and one place."""
from __future__ import annotations

import json
import os
import re

import numpy as np

_CLIP = None

RUBRIC = """You are checking one picture for a music video before it is used. It should show exactly one person: {who}. Wearing: {wear}. The video is set in: {world}.
Look carefully at the whole picture and answer with one JSON object only, no other text:
{{"people": <how many people you can see anywhere in the picture, counting small, blurred or far-away figures and reflections>, "gender": "man" or "woman" or "unclear", "facial_hair": true if this person has a beard, a moustache or stubble, else false, "glasses": true if this person wears glasses, sunglasses or goggles, else false, "head_in_frame": true if the whole head and face are inside the picture, else false, "text": true if any letters, words, numbers or logos are visible anywhere, else false, "{modern_key}": [{modern_q}], "outfit_matches": true if the clothes match the description, else false, "problems": "<short list of what is wrong, or none>"}}"""
MODERN_PERIOD = ("objects, clothes or hairstyles you can see that did not exist in {era}; leave out the photograph "
                 "itself, the camera, the lighting and the picture's style")
MODERN_NOW = "things you can see that clearly do not belong in this world; leave out the photograph itself, the camera and the lighting"

# what the "modern" answer must not count: the photo itself (Oct 26 test: "camera", "modern lighting",
# "digital camera/lighting" on clean portraits) and vague style words ("apron style", "fabric of the shirt")
_NOT_A_THING = re.compile(r"(?i)\b(camera|lighting|light|lens|photo|photograph|picture|image|film|flash|studio|"
                          r"style|aesthetic|quality|resolution|depth of field|bokeh|blur|grain|colou?r|tone|"
                          r"fabric|material|texture|look|expression|pose|background|composition|framing|"
                          r"likely|possibly|maybe|unclear)\b")

# how bad each fault is (the best take has the lowest total)
WEIGHT = {"people": 3, "gender": 3, "beard": 3, "glasses": 3, "head": 3, "text": 1, "modern": 1, "outfit": 1,
          "unread": 0}


def _clip():
    global _CLIP
    import nodes as comfy_nodes  # type: ignore
    from . import graphs
    from .theme import _call
    if _CLIP is None:
        loader = comfy_nodes.NODE_CLASS_MAPPINGS["LTXAVTextEncoderLoader"]
        _CLIP = _call(loader, text_encoder=graphs.M["ltx23_te"], ckpt_name=graphs.M["ltx23_ckpt"], device="default")[0]
    return _CLIP


def free():
    global _CLIP
    _CLIP = None


def ask(image_path: str, prompt: str, max_length: int = 320) -> str | None:
    """Gemma 3's answer about one picture (greedy, so the same picture gets the same answer)."""
    import torch
    import nodes as comfy_nodes  # type: ignore
    from PIL import Image, ImageOps
    from .theme import _call
    gen = comfy_nodes.NODE_CLASS_MAPPINGS.get("TextGenerate")
    if gen is None or not os.path.exists(image_path):
        return None
    im = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    im.thumbnail((1024, 1024))
    t = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)[None]
    out = _call(gen, clip=_clip(), prompt=prompt, max_length=max_length, sampling_mode={"sampling_mode": "off"},
                image=t, thinking=False, use_default_template=True)
    return str(out[0])


def rubric(check: dict, world: str, era: str) -> str:
    g = check.get("gender") or ""
    age = check.get("age") or ""
    from .theme import _article
    role = check.get("role") or ""
    who = f"{_article(role)} {role}" if role and not re.match(r"(?i)^(a|an|the)\s", role) else (role or "a person")
    if g:
        who += f", a {g}" + (f" aged about {age}" if age else "")
    mq = MODERN_PERIOD.format(era=era) if era else MODERN_NOW
    return RUBRIC.format(who=who, wear=check.get("wear") or "their own clothes", world=world.split(":")[0].strip(),
                         modern_key="modern" if era else "out_of_place", modern_q=mq)


def parse(text: str | None) -> dict | None:
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    t = m.group(0)
    t = re.sub(r",\s*([}\]])", r"\1", t)
    t = re.sub(r"\btrue/false\b", "false", t)
    try:
        return json.loads(t)
    except Exception:
        pass
    # one key per line, no commas (rare): pick the fields out one by one
    out = {}
    for k in ("people", "gender", "facial_hair", "glasses", "head_in_frame", "text", "outfit_matches"):
        mm = re.search(rf'"{k}"\s*:\s*("?)([^",}}\n]+)\1', t)
        if mm:
            v = mm.group(2).strip().lower()
            out[k] = True if v == "true" else False if v == "false" else (int(v) if v.isdigit() else v)
    mm = re.search(r'"(?:modern|out_of_place)"\s*:\s*\[([^\]]*)\]', t)
    if mm:
        out["modern"] = [x.strip().strip('"') for x in mm.group(1).split(",") if x.strip().strip('"')]
    return out or None


def _yes(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "yes", "1")


def faults(ans: dict | None, check: dict, kind: str = "portrait") -> list[str]:
    """The faults in one answer (empty: the picture passes)."""
    if not ans:
        return ["unread"]
    out = []
    try:
        people = int(ans.get("people", 1))
    except (TypeError, ValueError):
        people = 1
    if people >= 2:
        out.append("people")
    want = (check.get("gender") or "").lower()
    got = str(ans.get("gender", "")).strip().lower()
    if want in ("man", "woman") and got in ("man", "woman") and got != want:
        out.append("gender")
    if (want == "woman" or (not want and got == "woman")) and _yes(ans.get("facial_hair")):
        out.append("beard")
    if _yes(ans.get("glasses")) and not check.get("eyewear"):
        out.append("glasses")
    if kind == "portrait" and "head_in_frame" in ans and not _yes(ans.get("head_in_frame")):
        out.append("head")
    if _yes(ans.get("text")):
        out.append("text")
    # only a period World's "from after {era}" list counts: in a present-day World the "out of place"
    # list named the World's own things (st2e: bookshelves in a record store, bottles behind a bar,
    # the disco ball the World line asks for)
    modern = ans.get("modern") or []
    if isinstance(modern, str):
        modern = [modern] if modern.strip().lower() not in ("", "none", "[]") else []
    things = [str(x) for x in modern if str(x).strip() and not _NOT_A_THING.search(str(x))
              and str(x).strip().lower() not in ("none", "nothing", "n/a")]
    # glasses are judged on their own (an eyewear-free world names them as modern too)
    things = [x for x in things if not re.search(r"(?i)\b(glasses|spectacles|sunglasses|goggles)\b", x)]
    if things:
        out.append("modern")
    if "outfit_matches" in ans and not _yes(ans.get("outfit_matches")):
        out.append("outfit")
    return out


def score(fs: list[str]) -> int:
    return sum(WEIGHT.get(f, 1) for f in fs)


def modern_items(ans: dict | None) -> list[str]:
    modern = (ans or {}).get("modern", (ans or {}).get("out_of_place")) or []
    if isinstance(modern, str):
        modern = [modern]
    return [str(x) for x in modern if str(x).strip() and not _NOT_A_THING.search(str(x))
            and not re.search(r"(?i)\b(glasses|spectacles|sunglasses|goggles|none|nothing)\b", str(x))]


def fix_lines(fs: list[str], check: dict, era: str, ans: dict | None = None) -> str:
    """Sentences for the re-made picture's prompt, said as what should be there (naming the
    unwanted thing draws it: 'glasses' in the period line put glasses on every other face)."""
    g = (check.get("gender") or "").lower()
    out = []
    if "gender" in fs and g:
        out.append(f"The person is clearly a {g}.")
    if "beard" in fs:
        out.append("Her face is smooth and clean, with bare skin on the upper lip, chin and jaw.")
    if "glasses" in fs:
        out.append("Bare eyes and a clear, uncovered face.")
    if "people" in fs:
        out.append("The person is completely alone: an empty background with nobody else anywhere, near or far.")
    if "head" in fs:
        out.append("The whole head is in the picture with space above the hair.")
    if "text" in fs:
        out.append("Plain, unmarked surfaces everywhere, with no writing or symbols.")
    if "modern" in fs and era:
        out.append(f"Every object, garment and hairdo in the picture existed in {era}.")
    if "outfit" in fs and check.get("wear"):
        out.append(f"The outfit is exactly this: {check['wear']}.")
    return " ".join(out)


def era_of(theme: str) -> str:
    """'1965', 'ancient Rome', ... from the World line (casting.era_line's match), or ''."""
    from .casting import _ERA, _ERA_NAMES
    m = _ERA.search(theme or "")
    if not m:
        return ""
    if m.group(1):
        return m.group(1) + (" to " + m.group(2) if m.group(2) else "")
    if m.group(3):
        return "the " + m.group(3)
    return _ERA_NAMES[m.group(4).lower()]


def check_era(theme: str) -> str:
    """The era the 'from a later time' question is asked about: none for recent decades (sx07c,
    Ibiza 1999: it called pink hair, butterfly clips and crop tops modern)."""
    era = era_of(theme)
    m = re.search(r"\b(1[5-9]\d\d|20\d\d)", era)
    return "" if m and int(m.group(1)) >= 1980 else era


# ------------------------------------------------------------------ one face size, one place
def face_box(path: str):
    """[cx, cy, w] of the largest face in an image file (normalised), or None."""
    from PIL import Image
    from . import faces
    try:
        rgb = np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None
    return faces.detect(rgb, size=480)


def fit_targets(boxes: list, max_scale: float = 1.6) -> tuple[float, float]:
    """(face width, face centre height) every drop portrait is brought to: a little larger than the
    typical face (only ever scaled up), at the typical height."""
    ws = sorted(b[2] for b in boxes if b)
    cys = sorted(b[1] for b in boxes if b)
    if not ws:
        return 0.0, 0.42
    med = ws[len(ws) // 2]
    # a touch above the largest face, so even that one can be moved into place, but never more
    # than a third above the typical one (the zoom costs resolution and head room)
    # (st2e at 1.35: hats and hair touched the top of the frame on the drop)
    tw = min(max(ws) * 1.06, med * 1.2, med * max_scale)
    ty = float(np.clip(cys[len(cys) // 2], 0.36, 0.46))
    return tw, ty


def fit_face(path: str, out_path: str, box, tw: float, ty: float, max_scale: float = 1.6) -> dict | None:
    """Scale the picture up about its face (never down) and move it so the face sits at the
    centre, ty from the top, tw wide; the crop stays inside the picture. Returns what it did."""
    from PIL import Image
    if not box or tw <= 0:
        return None
    im = Image.open(path).convert("RGB")
    W, H = im.size
    cx, cy, w = box
    s = float(np.clip(tw / max(1e-3, w), 1.0, max_scale))
    cw, ch = W / s, H / s
    x0 = float(np.clip(cx * W - 0.5 * cw, 0, W - cw))
    y0 = float(np.clip(cy * H - ty * ch, 0, H - ch))
    if s < 1.01 and abs(x0) < 1 and abs(y0) < 1:
        return {"scale": 1.0, "dx": 0.0, "dy": 0.0}
    out = im.resize((W, H), Image.LANCZOS, box=(x0, y0, x0 + cw, y0 + ch))
    out.save(out_path)
    return {"scale": round(s, 3), "x0": round(x0 / W, 3), "y0": round(y0 / H, 3),
            "face": [round((cx * W - x0) / cw, 3), round((cy * H - y0) / ch, 3), round(w * s, 3)]}
