"""Output formats and render presets.

LTX latents are 1/32 of the pixel size and the multi-stage pipeline upsamples the latent x2
per stage, so the stage-1 size must be a multiple of 32. Sizes are given as the final
(last-stage) render size; delivery sizes are what the finished video is scaled to."""
from __future__ import annotations

import math

FORMATS = {
    # name: (render w, h at quality), (render w, h at fast), delivery (w, h)
    "9:16": {"quality": (640, 1152), "fast": (448, 768), "deliver": (1080, 1920)},
    "1:1": {"quality": (1152, 1152), "fast": (768, 768), "deliver": (1080, 1080)},
    "16:9": {"quality": (1152, 640), "fast": (768, 448), "deliver": (1920, 1080)},
}
FORMAT_NAMES = list(FORMATS)

PRESETS = {
    # quality = the user's R8 LTX lip-sync recipe: LTX-2.3 22B dev + distilled LoRA 0.45,
    # three stages (1/4 -> 1/2 -> full), 50 fps.
    "quality": {"stages": 3, "fps": 50, "engine": "ltx23_r8"},
    # fast = LTX-2.5 22B distilled, two stages (1/2 -> full), 25 fps.
    "fast": {"stages": 2, "fps": 25, "engine": "ltx25_distilled"},
}
PRESET_NAMES = list(PRESETS)


def render_size(fmt: str, preset: str) -> tuple[int, int]:
    return FORMATS[fmt][preset]


def stage_sizes(fmt: str, preset: str) -> list[tuple[int, int]]:
    """Sizes of every stage, smallest first."""
    w, h = render_size(fmt, preset)
    n = PRESETS[preset]["stages"]
    out = [(w >> (n - 1 - i), h >> (n - 1 - i)) for i in range(n)]
    for (sw, sh) in out[:1]:
        assert sw % 32 == 0 and sh % 32 == 0, f"stage-1 size {sw}x{sh} not a multiple of 32"
    return out


def deliver_size(fmt: str) -> tuple[int, int]:
    return FORMATS[fmt]["deliver"]


def ltx_frames(seconds: float, fps: float) -> int:
    """Smallest LTX-valid frame count (8k+1) covering `seconds`."""
    n = max(1, int(math.ceil(seconds * fps - 1e-6)))
    return 8 * max(1, math.ceil((n - 1) / 8)) + 1


def portrait_size(fmt: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Krea 2 pass-1 size (about 1 MP) and redetail size for a format."""
    return {"9:16": ((768, 1344), (1152, 2016)),
            "1:1": ((1024, 1024), (1536, 1536)),
            "16:9": ((1344, 768), (2016, 1152))}[fmt]
