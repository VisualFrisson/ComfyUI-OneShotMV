"""The finish: a grade learned from a reference edit (Lab mean/std), diffusion + warm halation,
S-curve contrast, saturation, lifted blacks, the Naturalistic base, the Studio's colour grade,
coloured film grain, then a rounded gate vignette (the audio-reactive FX come before all of it,
in assemble).
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


# ------------------------------------------------------------------ colour grades (the Studio's
# "Colour grade" choice, applied after the World's own grade and tone, before vignette and grain)
LOOKS = {
    "none": "As made",
    "teal_orange": "Teal & Orange",
    "print_2383": "Print Film",
    "eterna": "Soft Eterna",
    "bleach": "Bleach Bypass",
    "golden": "Golden Hour",
    "moonlit": "Moonlit Blue",
    "faded70s": "Faded 70s",
    "silver": "Silver Mono",
    # the looks from Mark Bone's "Breaking Down Every Major Color Grade in Film" (Oct 2026)
    "neon": "Neon Saturated",
    "natural": "Naturalistic",
    "lowcon": "Low Contrast",
    "lowkey": "Low Key",
    "tobacco": "Earthy Tobacco",
    "romantic": "Warm Romantic",
    "technicolor": "Technicolor",
    "pushed_to": "Pushed Teal & Orange",
    "matrix": "Industrial Green",
    "hardmono": "Hard Mono",
}

BASE_LOOK = "natural"  # under every grade, "As made" included; not offered as a grade itself

# the reference and what each look is after (the grade rating page shows these)
LOOK_NOTES = {
    "teal_orange": ("Transformers, Mad Max: Fury Road", "Teal shadows, warm skin and highlights, pushed halfway to Fury Road: complementary colours make faces jump out."),
    "print_2383": ("Kodak 2383 print stock", "Denser colour, highlights that roll off softly, slightly warm."),
    "eterna": ("Fujifilm Eterna", "Soft, low-saturation film stock with quiet greens and gentle contrast."),
    "bleach": ("Saving Private Ryan", "Silver retention: muted colour, deeper blacks, harder contrast; textures jump out."),
    "golden": ("Late-afternoon sun", "Warm amber across the frame, gentle lift in the blacks."),
    "moonlit": ("Day for night", "Darker, cool blue, low saturation."),
    "faded70s": ("1970s prints", "Faded blacks, warm highlights, muted colour."),
    "silver": ("Soft black and white", "Gentle mono with a slight cool tint in the shadows."),
    "neon": ("Good Time, The Neon Demon, Mandy", "Punchy, saturated colour: magenta in the shadows, cool highlights."),
    "natural": ("No Country for Old Men", "Believable daylight and skin, restrained saturation, warm tungsten."),
    "lowcon": ("Arrival, Nomadland", "Blacks never reach black, soft highlights, gentle tonal steps, a little murky."),
    "lowkey": ("The Godfather", "Most of the frame falls into deep amber shadow, while faces and lights stay bright."),
    "tobacco": ("O Brother, Where Art Thou?", "Dusty yellow-brown over everything; greens turned to straw."),
    "romantic": ("Her", "Peach, coral, cream and warm neutrals; blue held back; a soft mist on the highlights."),
    "technicolor": ("La La Land, The Wizard of Oz", "Rich, saturated primaries across the full rainbow."),
    "pushed_to": ("Mad Max: Fury Road", "Teal and orange pushed hard: burning orange, deep teal shadows."),
    "matrix": ("The Matrix", "A green, contaminated, industrial cast."),
    "hardmono": ("The Lighthouse", "Harsh, high-contrast black and white with an orthochromatic look: darker skin, more texture."),
}


def apply_look(x, name: str, amount: float = 1.0):
    """x: (H, W, 3) in 0..1. Each look is lift/gain, split toning, saturation and a contrast curve."""
    import torch
    if not name or name == "none" or amount <= 0:
        return x
    dev, dt = x.device, x.dtype

    def v(*c):
        return torch.tensor(c, dtype=dt, device=dev)

    def lum(y):
        return (y * v(0.2126, 0.7152, 0.0722)).sum(-1, keepdim=True)

    def split(y, sh, hi, bal=0.45):
        lm = lum(y)
        ws = (1 - lm / bal).clamp(0, 1) ** 1.4
        wh = ((lm - bal) / (1 - bal)).clamp(0, 1) ** 1.4
        return y + ws * v(*sh) + wh * v(*hi)

    def sat(y, k):
        lm = lum(y)
        return lm + (y - lm) * k

    def con(y, k, pivot=0.45):
        y = pivot + (y - pivot) * k
        return y.clamp(0, 1)

    def lift(y, b, tint=(0, 0, 0)):
        return v(*tint) + b + y * (1 - b)

    def scurve(y, k):  # film-style S-curve: more midtone contrast, blacks and whites stay put
        t = y.clamp(0, 1)
        return t + k * (t * t * (3 - 2 * t) - t)

    def mist(y, k):  # soft glow from the highlights (a diffusion filter)
        import torch.nn.functional as F
        H, W = y.shape[:2]
        t = y.permute(2, 0, 1)[None]
        b = F.interpolate(F.avg_pool2d(t, 12, ceil_mode=True), size=(H, W), mode="bilinear", align_corners=False)
        b = F.interpolate(F.avg_pool2d(b, 5, 1, 2), size=(H, W), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        hl = ((b - 0.5) / 0.5).clamp(0, 1) * k
        return 1 - (1 - y) * (1 - hl)

    y = x
    if name == "teal_orange":  # halfway between the classic and the Fury Road push (Oct 28)
        y = split(y, (-0.0475, 0.016, 0.06), (0.07, 0.02, -0.06), bal=0.435)
        y = y * v(1.03, 0.995, 0.95)
        y = con(sat(y, 1.19), 1.15, 0.435)
    elif name == "print_2383":
        y = y.clamp(0, 1) ** 1.06
        y = split(y, (-0.012, 0.012, 0.02), (0.04, 0.018, -0.022))
        y = (1 - torch.exp(-y.clamp(0, 1) * 2.2)) / (1 - torch.exp(v(-2.2)))  # highlights roll off like print
        y = con(sat(y, 0.94), 1.12, 0.42)
    elif name == "eterna":
        y = y * v(0.98, 1.0, 1.02)
        g = y[..., 1:2]
        y = torch.cat([y[..., 0:1], g * 0.97 + y[..., 2:3] * 0.03, y[..., 2:3]], -1)
        y = split(y, (-0.01, 0.008, 0.02), (0.01, 0.01, 0.0))
        y = lift(con(sat(y, 0.80), 0.86), 0.025)
        y = scurve(y, 0.45)
    elif name == "bleach":
        lm = lum(y)
        y = y * 0.45 + lm * 0.55
        y = con(y * v(0.98, 1.0, 1.02), 1.32, 0.42)
    elif name == "golden":
        y = y * v(1.08, 1.0, 0.86)
        y = split(y, (0.02, 0.008, -0.01), (0.04, 0.02, -0.03))
        y = lift(con(sat(y, 1.06), 1.04), 0.02, (0.012, 0.004, 0.0))
        y = scurve(y, 0.3)
    elif name == "moonlit":
        y = y * v(0.84, 0.95, 1.12) * 0.9
        y = split(y, (-0.01, 0.0, 0.03), (-0.01, 0.01, 0.03))
        y = con(sat(y, 0.72), 1.06, 0.38)
    elif name == "faded70s":
        y = split(y, (0.01, 0.02, 0.0), (0.035, 0.02, -0.035))
        y = lift(con(sat(y, 0.84), 0.9), 0.06, (0.008, 0.012, 0.0))
    elif name == "silver":
        lm = lum(y).expand_as(y)
        y = con(lm, 1.16, 0.45) + v(0.0, 0.004, 0.014) * (1 - lm)
    elif name == "neon":
        y = sat(y, 1.5)
        y = split(y, (0.045, -0.02, 0.065), (-0.01, 0.015, 0.04))
        y = con(y, 1.14, 0.42)
    elif name == "natural":
        y = y * v(1.03, 1.0, 0.95)
        y = split(y, (0.0, 0.004, 0.008), (0.012, 0.006, -0.012))
        y = con(sat(y, 0.88), 1.06)
    elif name == "lowcon":
        y = y * v(0.97, 1.0, 1.03) * 0.93
        y = (1 - torch.exp(-y.clamp(0, 1) * 1.6)) / (1 - torch.exp(v(-1.6)))
        y = lift(con(sat(y, 0.76), 0.76, 0.42), 0.075, (0.0, 0.004, 0.01))
        y = y * 0.93
    elif name == "lowkey":
        x0 = y.clamp(0, 1)
        y = x0 ** 1.45  # deep shadows
        # the highs and whites come back up (Oct 28: the faces and lights were too dim)
        w = ((lum(x0) - 0.4) / 0.6).clamp(0, 1) ** 1.3
        y = y + (x0 * 1.06 - y) * w * 0.85
        y = y * v(1.08, 0.99, 0.8)
        y = split(y, (0.008, 0.002, -0.012), (0.035, 0.016, -0.03), bal=0.35)
        y = con(sat(y, 0.9), 1.08, 0.3)
    elif name == "tobacco":
        r, g, b = y[..., 0:1], y[..., 1:2], y[..., 2:3]
        r = r + 0.35 * (g - r).clamp(min=0)  # greens to straw
        y = torch.cat([r, g, b * 0.92], -1)
        y = sat(y, 0.72) * v(1.07, 1.0, 0.74)
        y = lift(con(y, 1.05), 0.025, (0.02, 0.012, 0.0))
    elif name == "romantic":
        y = y * v(1.05, 0.99, 0.86)
        y = split(y, (0.03, 0.004, -0.008), (0.035, 0.018, -0.002))
        y = lift(con(sat(y, 0.96), 0.92), 0.03, (0.016, 0.006, 0.0))
        y = scurve(y, 0.35)
        y = mist(y, 0.35)
    elif name == "technicolor":
        y = y.clamp(0, 1) ** 1.06
        y = con(sat(y, 1.42), 1.1, 0.43)
        y = split(y, (0.0, 0.0, 0.01), (0.02, 0.01, -0.01))
    elif name == "pushed_to":
        y = split(y, (-0.06, 0.02, 0.075), (0.085, 0.022, -0.075), bal=0.42)
        y = y * v(1.06, 0.99, 0.9)
        y = con(sat(y, 1.32), 1.2, 0.42)
    elif name == "matrix":
        y = y * v(0.9, 1.05, 0.84)
        y = split(y, (-0.01, 0.03, 0.004), (0.0, 0.028, -0.02))
        y = con(sat(y, 0.78), 1.12, 0.42)
    elif name == "hardmono":
        lm = (y * v(0.05, 0.55, 0.40)).sum(-1, keepdim=True).expand_as(y)  # orthochromatic: red reads dark
        y = con(lm.clamp(0, 1) ** 1.1, 1.45, 0.46)
    y = y.clamp(0, 1)
    return x + (y - x) * min(1.0, float(amount))


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
        # 3b) the house base: Naturalistic on every video (the user's call, Oct 28), then the
        # Studio's colour grade on top
        base = c.get("base_look", BASE_LOOK)
        if base and base != "none":
            x = apply_look(x, base, c.get("base_amount", 1.0))
        if c.get("look") and c["look"] != "none":
            x = apply_look(x, c["look"], c.get("look_amount", 1.0))
        # 4) coloured grain, strongest in the mids (after the grade: the grade never tints the grain)
        gamt = c.get("grain", 0.05)
        if gamt > 0:
            lum = x.clamp(0, 1).mean(-1, keepdim=True)
            gen = torch.Generator(device=x.device).manual_seed(self.seed * 100003 + frame_index)
            size = max(1.0, float(c.get("grain_size", 1.0)))  # 1 = fine, 4 = coarse
            gh, gw = max(8, int(H / size)), max(8, int(W / size))
            g = torch.randn((1, 3, gh, gw), generator=gen, device=x.device, dtype=x.dtype)
            g = F.avg_pool2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
            if (gh, gw) != (H, W):
                g = F.interpolate(g, size=(H, W), mode="bilinear", align_corners=False) * (1 + 0.18 * (size - 1))
            g = g[0].permute(1, 2, 0) * 2.2
            g = g.mean(-1, keepdim=True) * 0.7 + g * 0.3
            x = x + g * gamt * (0.4 + 0.6 * 4 * lum * (1 - lum))
        # 5) gate vignette, last: it darkens the grain with the picture
        if c.get("vignette", 0.55) > 0:
            x = x * self._vignette(H, W, x.device, x.dtype)
        return x.clamp(0, 1)
