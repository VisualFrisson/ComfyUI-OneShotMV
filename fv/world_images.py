"""World from a folder of images (Oct 31, 2026): the user's own finished images instead of a
described World. The tool skips its image stage: images with a face big enough to sing become the
singers, the rest become cutaways with camera moves. In order (by file name) or random order;
images repeat when there are more shots than images. Each image is cropped to the video's shape
around its face (a 2:3 portrait in 16:9 keeps the face and loses the legs); nothing is drawn in
(the Oct 31 editor test: extending left grey bands and seams).

No Gemma, no World bible, no vision check on stills: the images are the cast. The take check runs
as usual."""
from __future__ import annotations

import os
import random
import re

import numpy as np

EXTS = (".png", ".jpg", ".jpeg", ".webp")
FACE_AT = 0.4       # where the face's centre sits in a crop that loses height: 40% down, leaving headroom
MIN_FACE_H = 0.1    # face height / frame height: smaller faces don't lip-sync well, so the image is a cutaway
GROUP_FACE = 0.6    # a second face at least this big against the first: two people, so a cutaway
SIZES = ((0.3, "close"), (0.16, "medium"))  # face height on the frame -> the shot size QC judges it by

# the person in the image sings; nothing about framing or wardrobe (the image already has them)
SING = ("The person in the picture sings along to the song the way a real person sings a song they love. The lips "
        "shape each word in time with the voice, a relaxed jaw, small natural head movements on the beat, a blink now "
        "and then, natural breathing. The handheld camera sways gently. They stay where they are, in the same clothes "
        "and the same light, holding nothing new: no microphone appears, and nobody else comes into the picture or "
        "starts singing. No text appears.")
# the user's own colours: the house grade, glow, halation and contrast push off (the first test, Oct 31,
# came out orange-red); the Studio's grain and looks still apply on top
FINISH = {"grade_amount": 0.0, "glow": 0.0, "halation": 0.0, "contrast": 0.0, "saturation": 1.0, "lift": 0.0}
# a cutaway from an image: the scene comes to life, nobody sings
MOTION = ("The scene in the picture comes gently to life with natural, unhurried movement, and the camera moves "
          "slowly and smoothly. Anyone in it stays quiet with the mouth closed, not singing or talking. Full frame "
          "with no black bars. No text appears.")


def _natural(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def resolve(folder: str) -> str:
    """A folder path as typed: absolute, or inside ComfyUI's input or output folder."""
    folder = (folder or "").strip().strip('"').strip("'")
    if not folder or os.path.isabs(folder):
        return folder
    try:
        import folder_paths  # type: ignore
        bases = [folder_paths.get_input_directory(), folder_paths.get_output_directory()]
    except Exception:
        bases = []
    for b in bases:
        p = os.path.join(b, folder)
        if os.path.isdir(p):
            return p
    return os.path.join(bases[0], folder) if bases else folder


def list_images(folder: str) -> list[str]:
    d = resolve(folder)
    if not d or not os.path.isdir(d):
        raise ValueError(f"The World images folder wasn't found: {folder}")
    names = sorted((f for f in os.listdir(d) if f.lower().endswith(EXTS) and not f.startswith(".")), key=_natural)
    if not names:
        raise ValueError(f"The World images folder has no images (JPG, PNG or WebP): {folder}")
    return [os.path.join(d, n) for n in names]


def crop_box(w: int, h: int, W: int, H: int, face=None) -> list[int]:
    """[x, y, width, height] of the largest W:H window in a w x h image, placed around the face
    (face = [cx, cy, width] as faces.detect_all gives it, cx and cy as fractions of the image): centred
    on it across, its centre FACE_AT down; centred when there is no face. Kept inside the image."""
    if w * H > W * h:  # wider than the video: full height, part of the width
        ch, cw = h, max(1, min(w, round(h * W / H)))
    else:  # taller: full width, part of the height
        cw, ch = w, max(1, min(h, round(w * H / W)))
    cx = face[0] * w if face else w / 2
    cy = face[1] * h if face else h / 2
    x = round(min(max(cx - cw / 2, 0), w - cw))
    y = round(min(max(cy - (FACE_AT * ch if face else ch / 2), 0), h - ch))
    return [int(x), int(y), int(cw), int(ch)]


def size_of(face_h: float) -> str:
    for lim, name in SIZES:
        if face_h >= lim:
            return name
    return "wide"


def inspect(paths: list[str], W: int, H: int, log=print) -> list[dict]:
    """Each image's shape, its crop to the video's shape (around the face), and its largest face on
    the final frame."""
    from PIL import Image, ImageOps
    from . import faces
    how = faces.method()
    if not how:
        log("[OneShotMV] World images: no face detector (SyncNet weights, or OpenCV 4's face cascade), so every "
            "image sings")
    out = []
    for path in paths:
        try:
            im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        except Exception as e:
            log(f"[OneShotMV] World images: skipped {os.path.basename(path)} ({e})")
            continue
        w, h = im.size
        face, group = None, False
        if how:
            small = im.copy()
            small.thumbnail((640, 640))
            fs = faces.detect_all(np.asarray(small), size=640)
            face = fs[0] if fs else None
            # two people about as big as each other (twins side by side): both mouths would move, and
            # the take check fails the second face, so the image is a cutaway
            group = len(fs) > 1 and fs[1][2] >= GROUP_FACE * fs[0][2]
        # two people: the crop is centred between them
        aim = [(fs[0][0] + fs[1][0]) / 2, (fs[0][1] + fs[1][1]) / 2, fs[0][2]] if group else face
        crop = crop_box(w, h, W, H, aim)
        face_h = round(face[2] * w * (W / crop[2]) / H, 3) if face else 0.0
        out.append({"path": path, "name": os.path.basename(path), "w": w, "h": h, "fit": "crop", "crop": crop,
                    "face": [round(float(v), 3) for v in face] if face else None, "face_h": face_h,
                    "group": group, "sings": (face_h >= MIN_FACE_H and not group) if how else True,
                    "size": size_of(face_h) if how else "medium"})
    return out


def sequence(items: list, n: int, order: str = "in order", seed: int = 1) -> list:
    """n items: in order, repeating from the start; or shuffled, each round in a new order and
    never the same image twice in a row."""
    if not items or n <= 0:
        return []
    if order != "random order":
        return [items[i % len(items)] for i in range(n)]
    rng, out = random.Random(seed * 7 + 11), []
    while len(out) < n:
        cyc = list(items)
        rng.shuffle(cyc)
        if out and len(cyc) > 1 and cyc[0] is out[-1]:
            cyc.append(cyc.pop(0))
        out += cyc
    return out[:n]


def cast(p: dict, singers: list[dict], order: str, seed: int) -> list[dict]:
    """One cast entry per performer (numbered in the order their shots play): the image it sings
    from. Every sung render takes its image's shot size, so the take check judges the framing the
    image has."""
    seq = sequence(singers, p["performers"], order, seed)
    out = [{"id": i, "seed": seed * 1000 + i * 7 + 1, "who": it["name"], "image": it["path"], "sing": SING,
            "portrait": "", "size": it["size"]} for i, it in enumerate(seq)]
    shots = {s.get("index"): s for s in p.get("shots", [])}
    for r in p["renders"]:
        if r["role"] in ("A", "B") and r["performer"] < len(out):
            r["size"] = out[r["performer"]]["size"]
            if r["role"] == "A" and r.get("shot") in shots:
                shots[r["shot"]]["size"] = r["size"]
    return out


def summary(items: list[dict]) -> str:
    sing = sum(1 for it in items if it["sings"])
    return f"World images: {len(items)} images, {sing} sing, {len(items) - sing} are cutaways; each cropped around its face"
