"""Output formats and render presets.

LTX latents are 1/32 of the pixel size and the multi-stage pipeline upsamples the latent x2
per stage. The union-control IC-LoRA (pose guides for twins and motion) encodes its guide at
half the stage-1 size, so the stage-1 size must be a multiple of 64 (an odd stage-1 latent
crashes the guide encode). Render sizes are therefore a touch wider than the exact aspect;
the assembler centre-crops to the delivery aspect before scaling.
Sizes are given as the final (last-stage) render size."""
from __future__ import annotations

import math

FORMATS = {
    # name: (render w, h at quality), (render w, h at fast), delivery (w, h)
    "16:9": {"quality": (1280, 768), "fast": (896, 512), "deliver": (1920, 1080)},
    "9:16": {"quality": (768, 1280), "fast": (512, 896), "deliver": (1080, 1920)},
    "1:1": {"quality": (1024, 1024), "fast": (768, 768), "deliver": (1080, 1080)},
}
for _f in FORMATS.values():
    _f["fast_voice"] = _f["fast"]  # singing at fast: the smaller size too
FORMAT_NAMES = list(FORMATS)  # 16:9 first: the default

PRESETS = {
    # quality = the 3-stage LTX-2.3 lip-sync schedule (engine "ltx23_r8"): LTX-2.3 22B dev +
    # distilled LoRA 0.45, three stages (1/4 -> 1/2 -> full), 50 fps.
    # sync_delay: LTX lip movements lead their audio; measured -0.22 s and -0.26 s (peak
    # mouth/vocal cross-correlation over 38 clips in two runs). Lips naturally open ~60-80 ms before the
    # sound, so the clips are shown 0.16 s later to land on the beat of the syllable.
    "quality": {"stages": 3, "fps": 50, "engine": "ltx23_r8", "sync_delay": 0.16},
    # fast = LTX-2.5 22B distilled, two stages (1/2 -> full), 25 fps.
    # fast: LTX-2.5 lips lead by only ~0.12 s (peak over 11 clips), so a small nudge
    "fast": {"stages": 2, "fps": 25, "engine": "ltx25_distilled", "sync_delay": 0.05},
    # fast for singing: LTX-2.5's lip sync scores lower under SyncNet, so voice runs at "fast"
    # keep the quality model (LTX-2.3 and its LoRAs) at the fast size: two stages (1/2 -> full,
    # the second with the quality schedule's last 3 steps), 25 fps, 20 first-stage steps
    "fast_voice": {"stages": 2, "fps": 25, "engine": "ltx23_r8", "sync_delay": 0.16, "steps1": 20,
                   "key_strengths": (0.7, 0.8)},
}
PRESET_NAMES = ["quality", "fast"]  # what users pick; fast_voice is chosen for voice + fast


def effective_preset(mode: str, preset: str) -> str:
    """Singing shots always use the quality model (fast_voice), motion keeps LTX-2.5 at fast."""
    return "fast_voice" if (mode == "voice" and preset == "fast") else preset


def render_size(fmt: str, preset: str) -> tuple[int, int]:
    return FORMATS[fmt][preset]


def stage_sizes(fmt: str, preset: str) -> list[tuple[int, int]]:
    """Sizes of every stage, smallest first."""
    w, h = render_size(fmt, preset)
    n = PRESETS[preset]["stages"]
    out = [(w >> (n - 1 - i), h >> (n - 1 - i)) for i in range(n)]
    for (sw, sh) in out[:1]:
        assert sw % 64 == 0 and sh % 64 == 0, f"stage-1 size {sw}x{sh} not a multiple of 64"
    return out


def deliver_size(fmt: str) -> tuple[int, int]:
    return FORMATS[fmt]["deliver"]


def deliver_crop(fmt: str, preset: str) -> tuple[int, int, int, int]:
    """(x, y, w, h) centre crop of a render frame with the delivery aspect."""
    rw, rh = render_size(fmt, preset)
    dw, dh = deliver_size(fmt)
    if rw * dh > dw * rh:  # render is wider than delivery
        w = int(round(rh * dw / dh / 2)) * 2
        return (rw - w) // 2, 0, w, rh
    h = int(round(rw * dh / dw / 2)) * 2
    return 0, (rh - h) // 2, rw, h


def ltx_frames(seconds: float, fps: float) -> int:
    """Smallest LTX-valid frame count (8k+1) covering `seconds`."""
    n = max(1, int(math.ceil(seconds * fps - 1e-6)))
    return 8 * max(1, math.ceil((n - 1) / 8)) + 1


def portrait_size(fmt: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Krea 2 pass-1 size (about 1 MP) and redetail size for a format."""
    return {"9:16": ((768, 1344), (1152, 2016)),
            "1:1": ((1024, 1024), (1536, 1536)),
            "16:9": ((1344, 768), (2016, 1152))}[fmt]
