"""The finish: a grade learned from a reference edit (Lab mean/std), diffusion + warm halation,
S-curve contrast, saturation, lifted blacks, a rounded gate vignette and coloured film grain.
torch, works on (H, W, 3) float frames in 0..1."""
from __future__ import annotations

import math

# sRGB D65 matrices
_M = [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
_MI = [[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]]
_WHITE = (0.95047, 1.0, 1.08883)


def rgb_to_lab(x):
    import torch
    lin = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    M = torch.tensor(_M, dtype=x.dtype, device=x.device)
    xyz = lin @ M.T / torch.tensor(_WHITE, dtype=x.dtype, device=x.device)
    e = 216 / 24389
    f = torch.where(xyz > e, xyz.clamp(min=1e-8) ** (1 / 3), (24389 / 27 * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return torch.stack([L, a, b], -1)


def lab_to_rgb(lab):
    import torch
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    e = 216 / 24389
    def inv(f):
        return torch.where(f ** 3 > e, f ** 3, (116 * f - 16) / (24389 / 27))
    xyz = torch.stack([inv(fx), inv(fy), inv(fz)], -1) * torch.tensor(_WHITE, dtype=lab.dtype, device=lab.device)
    MI = torch.tensor(_MI, dtype=lab.dtype, device=lab.device)
    lin = (xyz @ MI.T).clamp(0, 1)
    return torch.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin.clamp(min=1e-8) ** (1 / 2.4) - 0.055).clamp(0, 1)


def lab_stats(frames):
    """Mean/std of Lab over a list of (H, W, 3) tensors (subsampled)."""
    import torch
    labs = [rgb_to_lab(f[::8, ::8].float()).reshape(-1, 3) for f in frames]
    allv = torch.cat(labs)
    return allv.mean(0), allv.std(0) + 1e-3


class Finisher:
    def __init__(self, cfg: dict, src_stats=None, seed: int = 3):
        self.cfg, self.src = cfg, src_stats
        self.seed = seed
        self._vig = None

    def _vignette(self, H, W, dev, dt):
        import torch
        if self._vig is None or self._vig.shape[:2] != (H, W):
            y = torch.linspace(-1, 1, H, device=dev, dtype=dt)[:, None]
            x = torch.linspace(-1, 1, W, device=dev, dtype=dt)[None, :]
            pw = self.cfg.get("vignette_shape", 4.0)
            r = (x.abs() ** pw + y.abs() ** pw) ** (1 / pw)
            v = 1 - self.cfg.get("vignette", 0.55) * ((r - 0.72) / 0.4).clamp(0, 1) ** 1.6
            self._vig = v[..., None]
        return self._vig

    def __call__(self, img, frame_index: int = 0):
        import torch
        import torch.nn.functional as F
        c = self.cfg
        x = img.float()
        H, W = x.shape[:2]
        # 1) grade toward the reference look
        if self.src is not None and c.get("grade_amount", 0) > 0:
            sm, ss = self.src
            rm = torch.tensor(c["grade_mean"], dtype=x.dtype, device=x.device)
            rs = torch.tensor(c["grade_std"], dtype=x.dtype, device=x.device)
            amt = c["grade_amount"]
            lab = rgb_to_lab(x)
            lab = (lab - sm) / ss * (ss + (rs - ss) * amt) + (sm + (rm - sm) * amt)
            x = lab_to_rgb(lab)
        # 2) diffusion + warm halation
        t = x.permute(2, 0, 1)[None]
        k = max(3, int(W * 0.012 * 3) | 1)
        small = F.interpolate(t, scale_factor=0.25, mode="area")
        ks = max(3, (k // 4) | 1)
        glow = F.avg_pool2d(F.pad(small, (ks // 2,) * 4, mode="replicate"), ks, stride=1)
        glow = F.interpolate(glow, size=(H, W), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        gl = c.get("glow", 0.18)
        x = x * (1 - gl) + torch.maximum(x, glow) * gl
        lum = x.mean(-1, keepdim=True)
        hi = ((lum - 0.6) / 0.4).clamp(0, 1)
        hal = glow * hi * torch.tensor([1.0, 0.45, 0.25], dtype=x.dtype, device=x.device)
        x = 1 - (1 - x) * (1 - hal * c.get("halation", 0.55))
        # 3) tone: S-curve, saturation, lifted blacks
        x = x.clamp(0, 1)
        x = x + c.get("contrast", 0.35) * 1.2 * (x - 0.5) * (1 - (x - 0.5).abs() * 2)
        l = x.mean(-1, keepdim=True)
        x = (l + (x - l) * c.get("saturation", 1.15)).clamp(0, 1)
        lift = c.get("lift", 0.035)
        x = (lift + x * (1 - lift * 1.4)).clamp(0, 1) ** 1.08
        # 4) gate vignette
        x = x * self._vignette(H, W, x.device, x.dtype)
        # 5) coloured grain, strongest in the mids
        gamt = c.get("grain", 0.05)
        if gamt > 0:
            gen = torch.Generator(device=x.device).manual_seed(self.seed * 100003 + frame_index)
            g = torch.randn((1, 3, H, W), generator=gen, device=x.device, dtype=x.dtype)
            g = F.avg_pool2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), 3, stride=1)[0].permute(1, 2, 0) * 2.2
            g = g.mean(-1, keepdim=True) * 0.7 + g * 0.3
            x = x + g * gamt * (0.4 + 0.6 * 4 * lum * (1 - lum))
        return x.clamp(0, 1)
