"""World from a folder of images (Oct 31, 2026): the user's own finished images instead of a
described World. The tool skips its image stage: images with a face big enough to sing become the
singers, the rest become cutaways with camera moves. In order (by file name) or random order;
images repeat when there are more shots than images. An image close to the video's shape is
centre-cropped; one further off is extended to fit (FLUX.2 Klein draws the new edges, and the
original pixels are pasted back over the middle).

No Gemma, no World bible, no vision check on stills: the images are the cast. The take check runs
as usual."""
from __future__ import annotations

import os
import random
import re

import numpy as np

EXTS = (".png", ".jpg", ".jpeg", ".webp")
CROP_WITHIN = 0.2   # shape difference up to which an image is centre-cropped (a 2:3 photo in 9:16 loses 16% of its width)
MIN_FACE_H = 0.1    # face height / frame height: smaller faces don't lip-sync well, so the image is a cutaway
SIZES = ((0.3, "close"), (0.16, "medium"))  # face height on the frame -> the shot size QC judges it by

# the person in the image sings; nothing about framing or wardrobe (the image already has them)
SING = ("The person in the picture sings along to the song the way a real person sings a song they love. The lips "
        "shape each word in time with the voice, a relaxed jaw, small natural head movements on the beat, a blink now "
        "and then, natural breathing. The handheld camera sways gently. They stay where they are, in the same clothes "
        "and the same light, and nobody else starts singing. No text appears.")
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


def fit_of(w: int, h: int, W: int, H: int) -> str:
    """'crop' when the image is close to the video's shape, else 'extend'."""
    r = (w / h) / (W / H)
    return "crop" if max(r, 1 / r) - 1 <= CROP_WITHIN else "extend"


def size_of(face_h: float) -> str:
    for lim, name in SIZES:
        if face_h >= lim:
            return name
    return "wide"


def inspect(paths: list[str], W: int, H: int, log=print) -> list[dict]:
    """Each image's shape, how it fits the video, and its largest face on the final frame."""
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
        fit = fit_of(w, h, W, H)
        s = max(W / w, H / h) if fit == "crop" else min(W / w, H / h)
        face = None
        if how:
            small = im.copy()
            small.thumbnail((640, 640))
            fs = faces.detect_all(np.asarray(small), size=640)
            face = fs[0] if fs else None
        face_h = round(face[2] * w * s / H, 3) if face else 0.0
        if fit == "crop" and face:  # a face cropped off the side or top can't sing
            cx, cy = face[0] * w * s - (w * s - W) / 2, face[1] * h * s - (h * s - H) / 2
            if not (0 < cx < W and 0 < cy < H):
                face_h = 0.0
        out.append({"path": path, "name": os.path.basename(path), "w": w, "h": h, "fit": fit,
                    "face": [round(float(v), 3) for v in face] if face else None, "face_h": face_h,
                    "sings": (face_h >= MIN_FACE_H) if how else True,
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
    ext = sum(1 for it in items if it["fit"] == "extend")
    return (f"World images: {len(items)} images, {sing} sing, {len(items) - sing} are cutaways"
            + (f"; {ext} extended to the video's shape" if ext else ""))


def _ramp(torch, n_total, a, b, band):
    """1-D keep profile: 0 outside [a, b), 1 inside, rising over `band` pixels on a side that has new
    space next to it (a side that touches the canvas edge stays sharp)."""
    k = torch.zeros(n_total)
    k[a:b] = 1.0
    if a > 0 and band > 0:
        e = min(b, a + band)
        k[a:e] = torch.linspace(0.0, 1.0, e - a)
    if b < n_total and band > 0:
        s = max(a, b - band)
        k[s:b] = torch.minimum(k[s:b], torch.linspace(1.0, 0.0, b - s))
    return k


def extend_canvas(image, width, height):
    """(canvas, fill, keep) for OneShot MV: Extend Canvas. The image fitted inside width x height and
    centred on plain grey; `keep` is 1 over the image and fades to 0 over its last few per cent on a
    side with new space (the seam), `fill` is 1 - keep: where Klein draws, softly over the seam, so
    the new edges grow out of the image instead of meeting it at a line (Oct 31: a hard 60 px band
    left visible seams on a street scene)."""
    import torch
    _, h, w, _ = image.shape
    s = min(width / w, height / h)
    nw, nh = max(1, min(width, int(round(w * s)))), max(1, min(height, int(round(h * s))))
    x0, y0 = (width - nw) // 2, (height - nh) // 2
    x = torch.nn.functional.interpolate(image.permute(0, 3, 1, 2).float(), size=(nh, nw), mode="bicubic",
                                        align_corners=False, antialias=True).clamp(0, 1).permute(0, 2, 3, 1)
    canvas = torch.full((1, height, width, 3), 0.5)
    canvas[:, y0:y0 + nh, x0:x0 + nw] = x.cpu()
    band = max(16, int(round(0.12 * min(nw, nh))))
    keep = _ramp(torch, height, y0, y0 + nh, band)[:, None] * _ramp(torch, width, x0, x0 + nw, band)[None, :]
    return canvas, (1.0 - keep)[None], keep[None]
