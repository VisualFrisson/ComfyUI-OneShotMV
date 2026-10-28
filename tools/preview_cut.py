"""CPU preview of a OneShot MV cut (numpy + OpenCV, no torch / GPU).

Mirrors fv/fx.py and fv/finish.py closely enough to judge the edit, the effects and the
finish while the GPU is busy. Not used by the nodes.

    python tools/preview_cut.py <run_dir> [--width 540] [--out preview.mp4] [--fx 1.0]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from fv import audiofx, casting, formats, fx, stems  # noqa: E402
from fv import plan as planner  # noqa: E402
from fv.assemble import PRIORITY, _active  # noqa: E402

_M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]], np.float32)
_MI = np.linalg.inv(_M).astype(np.float32)
_WHITE = np.array([0.95047, 1.0, 1.08883], np.float32)


def rgb_to_lab(x):
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M.T / _WHITE
    e = 216 / 24389
    f = np.where(xyz > e, np.cbrt(np.maximum(xyz, 1e-8)), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_rgb(lab):
    fy = (lab[..., 0] + 16) / 116
    fx_, fz = fy + lab[..., 1] / 500, fy - lab[..., 2] / 200
    e = 216 / 24389

    def inv(f):
        return np.where(f ** 3 > e, f ** 3, (116 * f - 16) / (24389 / 27))
    xyz = np.stack([inv(fx_), inv(fy), inv(fz)], -1) * _WHITE
    lin = np.clip(xyz @ _MI.T, 0, 1)
    return np.clip(np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.maximum(lin, 1e-8) ** (1 / 2.4) - 0.055), 0, 1)


def blur(t, frac=0.02):
    H, W = t.shape[:2]
    s = max(8, int(min(H, W) * 0.125))
    small = cv2.resize(t, (max(8, W * s // max(H, W)), max(8, H * s // max(H, W))), interpolation=cv2.INTER_AREA)
    k = max(3, int(round(min(small.shape[:2]) * frac * 8)) | 1)
    for _ in range(2):
        small = cv2.blur(small, (k, k), borderType=cv2.BORDER_REPLICATE)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


def apply_fx(t, p, prev=None):
    H, W = t.shape[:2]
    if p["zoom"] > 1e-4 or abs(p["dx"]) + abs(p["dy"]) > 0.05:
        s = 1.0 + p["zoom"]
        M = np.array([[s, 0, (1 - s) * W / 2 + p["dx"]], [0, s, (1 - s) * H / 2 + p["dy"]]], np.float32)
        t = cv2.warpAffine(t, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    px = int(round(p["rgb"]))
    if px >= 1:
        t = np.stack([np.roll(t[..., 0], -px, 1), t[..., 1], np.roll(t[..., 2], px, 1)], -1)
    if p["bloom"] > 0.01:
        hi = np.clip((t.mean(-1, keepdims=True) - 0.55) / 0.45, 0, 1) * t
        t = 1 - (1 - t) * (1 - blur(hi) * p["bloom"] * 1.4)
    if p["flash"] > 0.01:
        f = p["flash"] * min(1.0, max(0.15, (0.75 - float(t.mean())) / 0.35))  # high-key frames: less
        t = t * (1 + 1.5 * f) * np.array([1 + 0.12 * f, 1 + 0.02 * f, 1 - 0.10 * f], np.float32)
        knee = 0.72
        t = np.where(t < knee, t, knee + (1 - knee) * np.tanh((t - knee) / (1 - knee)))
        hi = np.clip((t.mean(-1, keepdims=True) - 0.55) / 0.45, 0, 1) * t
        t = 1 - (1 - t) * (1 - blur(hi) * 0.9 * f)
        m = t.mean(-1, keepdims=True)
        t = m + (t - m) * (1 + 0.25 * f)
        t = 1 - (1 - t) * (1 - 0.06 * f)
    out = np.clip(t, 0, 1).astype(np.float32)
    if prev is not None and p["echo"] > 0.01:
        out = np.maximum(out, prev * p["echo"])
    return out


class Finisher:
    def __init__(self, c, src):
        self.c, self.src, self.vig = c, src, None

    def __call__(self, x, fi):
        c = self.c
        H, W = x.shape[:2]
        if self.src is not None and c.get("grade_amount", 0) > 0:
            sm, ss = self.src
            rm, rs, a = np.array(c["grade_mean"]), np.array(c["grade_std"]), c["grade_amount"]
            lab = rgb_to_lab(x)
            x = lab_to_rgb((lab - sm) / ss * (ss + (rs - ss) * a) + (sm + (rm - sm) * a)).astype(np.float32)
        small = cv2.resize(x, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        ks = max(3, (max(3, int(W * 0.012 * 3) | 1) // 4) | 1)
        glow = cv2.resize(cv2.blur(small, (ks, ks), borderType=cv2.BORDER_REPLICATE), (W, H))
        gl = c.get("glow", 0.18)
        x = x * (1 - gl) + np.maximum(x, glow) * gl
        lum = x.mean(-1, keepdims=True)
        hal = glow * np.clip((lum - 0.6) / 0.4, 0, 1) * np.array([1.0, 0.45, 0.25], np.float32)
        x = np.clip(1 - (1 - x) * (1 - hal * c.get("halation", 0.55)), 0, 1)
        x = x + c.get("contrast", 0.35) * 1.2 * (x - 0.5) * (1 - np.abs(x - 0.5) * 2)
        l = x.mean(-1, keepdims=True)
        x = np.clip(l + (x - l) * c.get("saturation", 1.15), 0, 1)
        lift = c.get("lift", 0.035)
        x = np.clip(lift + x * (1 - lift * 1.4), 0, 1) ** 1.08
        if self.vig is None:
            y = np.linspace(-1, 1, H)[:, None]
            xx = np.linspace(-1, 1, W)[None, :]
            pw = c.get("vignette_shape", 4.0)
            r = (np.abs(xx) ** pw + np.abs(y) ** pw) ** (1 / pw)
            self.vig = (1 - c.get("vignette", 0.55) * np.clip((r - 0.72) / 0.4, 0, 1) ** 1.6)[..., None].astype(np.float32)
        x = x * self.vig
        g = np.random.default_rng(fi).standard_normal((H, W, 3)).astype(np.float32)
        g = cv2.blur(g, (3, 3)) * 2.2
        g = g.mean(-1, keepdims=True) * 0.7 + g * 0.3
        x = x + g * c.get("grain", 0.05) * (0.4 + 0.6 * 4 * lum * (1 - lum))
        return np.clip(x, 0, 1).astype(np.float32)


class Reader:
    def __init__(self, path, size):
        self.cap = cv2.VideoCapture(path) if path and os.path.exists(path) else None
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap else 0
        self.size, self.pos, self.last = size, 0, None

    def get(self, idx):
        if not self.n:
            return None
        idx = int(min(max(0, idx), self.n - 1))
        if idx < self.pos - 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.pos = idx
        while self.pos <= idx:
            ok, f = self.cap.read()
            if not ok:
                break
            self.last, self.pos = f, self.pos + 1
        if self.last is None:
            return None
        W, H = self.size
        h, w = self.last.shape[:2]
        s = max(W / w, H / h)
        r = cv2.resize(self.last, (max(W, round(w * s)), max(H, round(h * s))), interpolation=cv2.INTER_AREA)
        y, x = (r.shape[0] - H) // 2, (r.shape[1] - W) // 2
        return r[y:y + H, x:x + W, ::-1].astype(np.float32) / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--width", type=int, default=540)
    ap.add_argument("--out", default="")
    ap.add_argument("--fx", type=float, default=1.0)
    ap.add_argument("--clips", default="", help="folder holding the run's clips if not run_dir")
    ap.add_argument("--pack", default="", help="override the pack (name or .json path)")
    ap.add_argument("--delay", type=float, default=None, help="override the lip-sync delay (s)")
    a = ap.parse_args()
    run = a.run_dir
    plan = json.load(open(os.path.join(run, "plan.json"), encoding="utf-8"))
    pack = casting.load_pack(a.pack or plan["pack"])
    fps, fmt = plan["out_fps"], plan["format"]
    cx, cy, cw, ch = formats.deliver_crop(fmt, plan["preset"])
    rw, rh = formats.render_size(fmt, plan["preset"])
    sc = a.width / cw
    W, H = a.width, int(round(ch * sc / 2)) * 2
    RW, RH = int(round(rw * sc / 2)) * 2, int(round(rh * sc / 2)) * 2
    n_out = int(round(plan["duration"] * fps))
    mix, sr = stems.read_wav(os.path.join(run, "song.wav"))
    drums = vocals = ssr = None
    if os.path.exists(os.path.join(run, "drums.wav")):
        drums, ssr = stems.read_wav(os.path.join(run, "drums.wav"))
        vocals, _ = stems.read_wav(os.path.join(run, "vocals.wav"))
        pad = int(round(plan.get("stems_t0", plan["song_t0"]) * ssr))
        drums, vocals = np.pad(drums, ((0, 0), (pad, 0))), np.pad(vocals, ((0, 0), (pad, 0)))
    ctl = audiofx.controls(mix, sr, plan["song_t0"], plan["song_t1"], fps, drums, vocals, ssr)
    tlw = plan.get("timeline", {})
    ctl["beatpulse"] = audiofx.beat_pulses(tlw.get("beats", []), n_out, fps)
    bar_s = 4 * 60.0 / max(1.0, float(plan.get("tempo") or 120.0))
    ctl["drop"] = audiofx.drop_mask((tlw.get("drop") or {}).get("time"), bar_s, n_out, fps)
    cfg = dict(pack["fx"])
    cfg["intensity"] = cfg.get("intensity", 1.0) * a.fx
    for k in ("rgb", "shake"):
        cfg[k] = cfg[k] * sc  # pixel-sized effects scale with the preview
    delay = float(plan.get("sync_delay", formats.PRESETS.get(plan["preset"], {}).get("sync_delay", 0.0))) \
        if a.delay is None else a.delay
    clipdir = a.clips or run
    if plan.get("mode") == "voice" and plan.get("shots"):
        faces = None
        if pack["edit"].get("match_cut", False):
            from fv import faces as fz
            faces = fz.tracks_for_run(clipdir, plan["renders"])
            print("face tracks:", len(faces))
        beats_w = sorted(set(round(b, 4) for b in tlw.get("beats", []) if 0 <= b < plan["duration"]))
        plan["edl"] = planner.voice_edl(plan["shots"], plan["renders"], beats_w, pack["edit"], fps,
                                        plan.get("seed", 1), faces=faces, delay=delay)
    else:
        delay = 0.0
    renders = {r["id"]: r for r in plan["renders"]}
    readers = {rid: Reader(os.path.join(clipdir, os.path.basename(r.get("path", "").replace("\\", "/"))), (RW, RH))
               for rid, r in renders.items()}

    def frame_for(rid, t):
        r, rd = renders.get(rid), readers.get(rid)
        if r is None or not rd.n:
            alt = r.get("twin_of") if r else None
            return frame_for(alt, t) if alt is not None and readers[alt].n else None
        return rd.get(int(round((t - r["t_start"] - delay) * r["fps"])))

    def warp(x, al):
        if not al:
            return x
        sc, ddx, ddy = al
        h, w = x.shape[:2]
        cxp, cyp = w / 2, h / 2
        M = np.array([[1 / sc, 0, cxp - (cxp + ddx * w) / sc], [0, 1 / sc, cyp - (cyp + ddy * h) / sc]], np.float32)
        return cv2.warpAffine(x, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)

    samples = [rd.get(rd.n // 2) for rd in readers.values() if rd.n]
    src = None
    if samples:
        labs = np.concatenate([rgb_to_lab(s[::4, ::4]).reshape(-1, 3) for s in samples])
        src = (labs.mean(0), labs.std(0) + 1e-3)
    fin = Finisher(pack["finish"], src)
    shot_starts = {int(round(s["t0"] * fps)) for s in plan.get("shots", [])}

    def loud_at(f):
        t = f / fps
        for s in plan["sections"]:
            if s["start"] - 1e-3 <= t < s["end"]:
                return s["loud"]
        return True
    out = a.out or os.path.join(run, "preview.mp4")
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
           "-i", "-", "-ss", str(plan["song_t0"]), "-t", str(plan["duration"]), "-i", os.path.join(run, "song.wav"),
           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "23", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-shortest", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    prev, starts = None, {}
    ox, oy = int(round(cx * sc)), int(round(cy * sc))
    for f in range(n_out):
        t = f / fps
        ev = _active(plan["edl"], f)
        j = 0
        if ev is not None:
            key = (ev["kind"], ev["f0"], ev["render"])
            starts.setdefault(key, f)
            j = f - starts[key]
        img = frame_for(ev["render"], t + ev.get("dt", 0.0)) if ev else None
        x = prev if img is None and prev is not None else (img if img is not None else np.zeros((RH, RW, 3), np.float32))
        if img is not None and ev is not None and ev.get("align"):
            x = warp(x, ev["align"])
        if ev is not None and ev["kind"] == "ghost" and ev.get("from") is not None:
            other = frame_for(ev["from"], t + ev.get("from_dt", 0.0))
            if other is not None:
                other = warp(other, ev.get("from_align"))
                al = cfg.get("ghost", 0.6) * (1 - j / max(1, ev["f1"] - ev["f0"]))
                x = x * (1 - al) + other * al
        p = fx.fx_params(f, ev, ctl, loud_at(f), cfg, f in shot_starts or (ev is not None and ev["kind"] == "switch"), j, fps)
        y = apply_fx(x, p, prev)
        prev = y
        z = fin(y, f)[oy:oy + H, ox:ox + W]
        proc.stdin.write((z * 255 + 0.5).astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    print(out)


if __name__ == "__main__":
    main()
