"""Audio-reactive effects (torch, GPU).

fx_params(): per output frame, from the edit events and the audio controls ->
    flash (exposure blowout), bloom, rgb (channel split px), zoom, shake (px), echo.
apply_fx(): renders those onto a frame (H, W, 3 float tensor in 0..1).
"""
from __future__ import annotations

import math

import numpy as np


def _hash01(i: int, k: int) -> float:
    return ((i * 73856093) ^ (k * 19349663)) % 1000 / 1000.0


def fx_params(f: int, ev: dict | None, ctl: dict, loud: bool, cfg: dict, shot_start: bool, j: int,
              fps: float = 24.0) -> dict:
    """f: output frame; ev: active edit event; j: frames since that event started."""
    I = cfg.get("intensity", 1.0) * (1.0 if loud else cfg.get("quiet_scale", 0.25))
    kick = float(ctl["kick"][f]) if f < len(ctl["kick"]) else 0.0
    snare = float(ctl["snare"][f]) if f < len(ctl["snare"]) else 0.0
    hat = float(ctl["hat"][f]) if f < len(ctl["hat"]) else 0.0
    lv = float(ctl["loud"][f]) if f < len(ctl["loud"]) else 0.5
    pulse = float(ctl["beatpulse"][f]) if "beatpulse" in ctl and f < len(ctl["beatpulse"]) else 0.0
    drop = float(ctl["drop"][f]) if "drop" in ctl and f < len(ctl["drop"]) else 0.0
    flash = 0.3 * cfg["flash"] * snare * lv
    rgb = 0.5 * cfg["rgb"] * snare
    zoom = cfg["zoom"] * kick
    shake = cfg["shake"] * max(kick, snare) * lv
    bloom = 0.35 * snare * lv
    echo = 0.0
    if drop > 0 and pulse > 0:  # strobe on every beat right after the drop
        flash = max(flash, 0.75 * cfg["flash"] * pulse)
        rgb = max(rgb, cfg["rgb"] * pulse)
        zoom = max(zoom, cfg["zoom"] * 1.3 * pulse)
    kind = ev["kind"] if ev else "main"
    if kind == "insert":
        n = max(1, ev["f1"] - ev["f0"])
        a = 1.0 - j / n
        flash = max(flash, cfg["flash"] * (0.6 + 0.4 * a))
        bloom = max(bloom, 0.7 * a + 0.3)
        rgb = max(rgb, cfg["rgb"] * (0.5 + 0.5 * a))
        zoom = max(zoom, cfg["zoom"] * 1.2)
    elif kind == "ghost":
        echo = cfg.get("ghost", 0.6)
        rgb = max(rgb, cfg["rgb"] * 0.4)
    if shot_start and j < 3:
        a = 1.0 - j / 3.0
        flash = max(flash, 0.9 * cfg["flash"] * a)
        rgb = max(rgb, cfg["rgb"] * a)
        bloom = max(bloom, 0.6 * a)
        zoom = max(zoom, cfg["zoom"] * 0.8 * a)
    if hat > 0.6 and loud:
        echo = max(echo, 0.2 * hat)
    ang = 2 * math.pi * _hash01(f, 1)
    return {"flash": min(1.0, flash * I), "bloom": min(1.0, bloom * I), "rgb": rgb * I,
            "zoom": zoom * I, "dx": shake * I * math.cos(ang), "dy": shake * I * math.sin(ang),
            "echo": min(0.9, echo * (1.0 if loud else 0.5))}


# ------------------------------------------------------------------ torch side
def _blur(t, sigma_frac=0.02):
    """Cheap wide blur: downsample, box blur twice, upsample. t: (1,3,H,W)."""
    import torch.nn.functional as F
    H, W = t.shape[-2:]
    s = max(4, int(min(H, W) * 0.125))
    small = F.interpolate(t, size=(max(8, H * s // max(H, W)), max(8, W * s // max(H, W))),
                          mode="area")
    k = max(3, int(round(min(small.shape[-2:]) * sigma_frac * 8)) | 1)
    for _ in range(2):
        small = F.avg_pool2d(F.pad(small, (k // 2,) * 4, mode="replicate"), k, stride=1)
    return F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)


def apply_fx(img, p: dict, prev=None):
    """img: (H, W, 3) float tensor 0..1 (any device). prev: previous output frame or None."""
    import torch
    import torch.nn.functional as F
    t = img.permute(2, 0, 1)[None]
    H, W = t.shape[-2:]
    if p["zoom"] > 1e-4 or abs(p["dx"]) + abs(p["dy"]) > 0.05:
        s = 1.0 / (1.0 + p["zoom"])
        theta = torch.tensor([[[s, 0.0, -2 * p["dx"] / W], [0.0, s, -2 * p["dy"] / H]]],
                             dtype=t.dtype, device=t.device)
        grid = F.affine_grid(theta, t.shape, align_corners=False)
        t = F.grid_sample(t, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    px = int(round(p["rgb"]))
    if px >= 1:
        r = torch.roll(t[:, 0:1], shifts=-px, dims=3)
        b = torch.roll(t[:, 2:3], shifts=px, dims=3)
        t = torch.cat([r, t[:, 1:2], b], 1)
    if p["bloom"] > 0.01:
        hi = torch.clamp((t.mean(1, keepdim=True) - 0.55) / 0.45, 0, 1) * t
        g = _blur(hi)
        t = 1 - (1 - t) * (1 - g * p["bloom"] * 1.4)
    if p["flash"] > 0.01:
        # a hot, warm over-exposure like an on-camera flash: blacks stay deep (no milky
        # white-out), highlights roll off softly and bloom, colour gets a little richer
        f = p["flash"]
        t = t * (1 + 1.5 * f)
        t = t * torch.tensor([1 + 0.12 * f, 1 + 0.02 * f, 1 - 0.10 * f], dtype=t.dtype,
                             device=t.device).view(1, 3, 1, 1)
        knee = 0.72
        t = torch.where(t < knee, t, knee + (1 - knee) * torch.tanh((t - knee) / (1 - knee)))
        hi = torch.clamp((t.mean(1, keepdim=True) - 0.55) / 0.45, 0, 1) * t
        t = 1 - (1 - t) * (1 - _blur(hi) * 0.9 * f)
        m = t.mean(1, keepdim=True)
        t = m + (t - m) * (1 + 0.25 * f)
        t = 1 - (1 - t) * (1 - 0.06 * f)
    out = torch.clamp(t, 0, 1)[0].permute(1, 2, 0)
    if prev is not None and p["echo"] > 0.01:
        out = torch.maximum(out, prev * p["echo"])
    return out
