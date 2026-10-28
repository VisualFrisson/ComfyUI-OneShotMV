"""Cut a planned run together: edit decision list -> frames -> audio-reactive FX -> finish ->
delivery size -> MP4 with the song. Runs inside ComfyUI (torch on the GPU, OpenCV to read
clips, ffmpeg to encode)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np

from . import audiofx, casting, export, formats, fx, stems
from .finish import Finisher, lab_stats

PRIORITY = {"insert": 3, "ghost": 2, "switch": 1, "main": 0}


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:  # VideoHelperSuite keeps one
        from custom_nodes.ComfyUI_VideoHelperSuite.videohelpersuite.utils import ffmpeg_path  # type: ignore
        if ffmpeg_path:
            return ffmpeg_path
    except Exception:
        pass
    raise RuntimeError("ffmpeg not found: pip install imageio-ffmpeg")


class ClipReader:
    """Forward-only frame reader (each clip is visited in increasing time)."""

    def __init__(self, path: str, size: tuple[int, int], trim_bars: bool = False):
        import cv2
        self.cv2, self.path, self.size = cv2, path, size
        self.cap = cv2.VideoCapture(path) if path and os.path.exists(path) else None
        self.pos, self.last = 0, None
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap else 0
        # text-to-video b-roll sometimes comes back letterboxed (n39): crop the bars, then cover
        self.bars = letterbox(path, cv2) if trim_bars and self.ok else (0, 0)

    @property
    def ok(self):
        return self.cap is not None and self.n > 0

    def get(self, idx: int):
        if not self.ok:
            return None
        idx = int(min(max(0, idx), self.n - 1))
        if idx < self.pos - 1:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, idx)
            self.pos = idx
        while self.pos <= idx:
            ok, f = self.cap.read()
            if not ok:
                break
            self.last = f
            self.pos += 1
        if self.last is None:
            return None
        f = self.last
        top, bot = self.bars
        if top or bot:
            f = f[top:f.shape[0] - bot]
        if (f.shape[1], f.shape[0]) != self.size:
            f = cover(f, self.size, self.cv2)
        return f[:, :, ::-1]  # RGB


def letterbox(path: str, cv2, n: int = 5, dark: float = 18.0) -> tuple[int, int]:
    """Rows of black bar at the top and bottom of a clip (the same in all n sampled frames),
    (0, 0) when there are none; a bar must be at least 3% of the height to count."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    tops, bots = [], []
    for i in np.linspace(0, max(0, total - 1), n).round().astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        rows = f.mean(axis=(1, 2))
        h = len(rows)
        t = 0
        while t < h // 3 and rows[t] < dark:
            t += 1
        b = 0
        while b < h // 3 and rows[h - 1 - b] < dark:
            b += 1
        tops.append(t)
        bots.append(b)
    cap.release()
    if not tops:
        return 0, 0
    h = int(cv2.VideoCapture(path).get(cv2.CAP_PROP_FRAME_HEIGHT) or 1)
    t, b = min(tops), min(bots)
    # a letterbox has a bar at both ends, about the same height (a dark sky or floor has one)
    if min(t, b) < 0.03 * h or abs(t - b) > 0.03 * h:
        return 0, 0
    return t + 2, b + 2  # a couple of rows more: the bar edge is soft


def cover(f, size, cv2):
    W, H = size
    h, w = f.shape[:2]
    s = max(W / w, H / h)
    r = cv2.resize(f, (max(W, int(round(w * s))), max(H, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    y, x = (r.shape[0] - H) // 2, (r.shape[1] - W) // 2
    return r[y:y + H, x:x + W]


def _active(edl: list[dict], f: int):
    best, start = None, None
    for e in edl:
        if e["f0"] <= f < e["f1"] and (best is None or PRIORITY[e["kind"]] >= PRIORITY[best["kind"]]):
            best = e
    return best


def assemble(run_dir: str, *, fx_intensity: float = 1.0, do_finish: bool = True, device=None,
             out_name: str = "final", progress=None, fx_level: str | None = None) -> dict:
    import torch
    import torch.nn.functional as F
    plan = json.load(open(os.path.join(run_dir, "plan.json"), encoding="utf-8"))
    pack = casting.load_pack(plan["pack"])
    if plan.get("finish"):  # a theme's grade (fv/theme.py)
        pack["finish"] = plan["finish"]
    if plan.get("finish_user"):  # the Studio's colour grade, grain and vignette for this run
        pack["finish"] = dict(pack["finish"], **plan["finish_user"])
    fps = plan["out_fps"]
    fmt = plan["format"]
    rw, rh = formats.render_size(fmt, plan["preset"])
    dw, dh = formats.deliver_size(fmt)
    cx, cy, cw, ch = formats.deliver_crop(fmt, plan["preset"]) if rw * rh else (0, 0, rw, rh)
    n_out = int(round(plan["duration"] * fps))
    if device is None:
        try:
            import comfy.model_management as mm
            device = mm.get_torch_device()
        except Exception:
            device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- audio controls
    mix, sr = stems.read_wav(os.path.join(run_dir, "song.wav"))
    drums = vocals = None
    ssr = None
    so = plan.get("stems_t0", plan["song_t0"])
    if os.path.exists(os.path.join(run_dir, "drums.wav")):
        drums, ssr = stems.read_wav(os.path.join(run_dir, "drums.wav"))
        vocals, _ = stems.read_wav(os.path.join(run_dir, "vocals.wav"))
        # stems start at stems_t0; shift them onto song time by padding
        pad = int(round(so * ssr))
        drums = np.pad(drums, ((0, 0), (pad, 0)))
        vocals = np.pad(vocals, ((0, 0), (pad, 0)))
    ctl = audiofx.controls(mix, sr, plan["song_t0"], plan["song_t1"], fps, drums, vocals, ssr)
    tlw = plan.get("timeline", {})
    ctl["beatpulse"] = audiofx.beat_pulses(tlw.get("beats", []), n_out, fps)
    bar_s = 4 * 60.0 / max(1.0, float(plan.get("tempo") or 120.0))
    ctl["drop"] = audiofx.drop_mask((tlw.get("drop") or {}).get("time"), bar_s, n_out, fps)
    # the run's audio-reactive FX level (Director input audio_fx; none / subtle / intense)
    cfg = fx.level_cfg(pack["fx"], fx_level or plan.get("fx_level"))
    cfg["intensity"] = cfg.get("intensity", 1.0) * fx_intensity
    # the twitch is up to `twitch` px of the delivered frame (frames are drawn at render size)
    cfg["twitch"] = cfg.get("twitch", 0) * cw / max(1, dw)

    # ---- the cut: re-derived from the shots with the pack's current edit settings, so a
    # re-assemble picks up edit changes without re-rendering
    # lip-sync offset: LTX's mouths lead its audio slightly; show each clip `delay` seconds later
    delay = float(plan.get("sync_delay", formats.PRESETS.get(plan["preset"], {}).get("sync_delay", 0.0)))
    if plan.get("mode") == "voice" and plan.get("shots") and not plan.get("edl_locked"):
        from . import plan as planner
        faces = None
        if pack["edit"].get("match_cut", False):
            import importlib
            from . import faces as fz
            fz = importlib.reload(fz)  # nodes.py's hot reload predates this module
            faces = fz.tracks_for_run(run_dir, plan["renders"])
            if not faces:
                print("[OneShotMV] face tracking unavailable (this OpenCV build has no Haar "
                      "cascades, e.g. OpenCV 5): flashes will not be face-aligned. "
                      "pip install \"opencv-python<5\" to enable match cuts.")
        beats_w = sorted(set(round(b, 4) for b in tlw.get("beats", []) if 0 <= b < plan["duration"]))
        plan["edl"] = planner.voice_edl(plan["shots"], plan["renders"], beats_w, pack["edit"], fps,
                                        plan.get("seed", 1), faces=faces, delay=delay)
    if plan.get("mode") != "voice":
        delay = 0.0  # motion clips carry no lip sync
        if plan.get("shots") and not plan.get("edl_locked") and any("B" in s for s in plan["shots"]):
            from . import plan as planner
            beats_w = sorted(set(round(b, 4) for b in tlw.get("beats", []) if 0 <= b < plan["duration"]))
            plan["edl"] = planner.motion_edl(plan["shots"], beats_w, pack["edit"], fps, plan.get("seed", 1),
                                             renders=plan.get("renders"))

    # ---- clips
    renders = {r["id"]: r for r in plan["renders"]}

    def clip_path(r):
        # a pickup (Studio re-shoot) whose new take didn't render falls back to the take it replaced
        pth = r.get("path", "")
        if r.get("path_before") and not os.path.exists(pth) and os.path.exists(r["path_before"]):
            print(f"[OneShotMV] r{r['id']:03d}: the pickup take is missing, using the take before it")
            if "sync_delay_before" in r:
                r["sync_delay"] = r["sync_delay_before"]
            return r["path_before"]
        return pth
    readers = {rid: ClipReader(clip_path(r), (rw, rh), trim_bars=r.get("role") == "R")
               for rid, r in renders.items()}
    missing = [rid for rid, rd in readers.items() if not rd.ok]

    held = {}  # output frames that had to hold a clip's first / last frame (shows as a pause)

    def frame_for(rid, t):
        r = renders.get(rid)
        rd = readers.get(rid)
        if r is None or rd is None or not rd.ok:
            alt = r.get("twin_of") if r else None
            if alt is not None and readers[alt].ok:
                return frame_for(alt, t)
            return None
        # per-take lip-sync offset measured by take QC (SyncNet), else the preset's average
        idx = int(round((t - r["t_start"] - r.get("sync_delay", delay)) * r["fps"]))
        if idx < 0 or idx > rd.n - 1:
            held[rid] = held.get(rid, 0) + 1
        return rd.get(idx)

    def warp(x, al):
        """Scale/shift a frame so its face lands where the lead's face is (see faces.align)."""
        if not al:
            return x
        sc, ddx, ddy = al
        th = torch.tensor([[[1 / sc, 0.0, -2 * ddx / sc], [0.0, 1 / sc, -2 * ddy / sc]]], dtype=x.dtype,
                          device=x.device)
        t4 = x.permute(2, 0, 1)[None]
        grid = F.affine_grid(th, t4.shape, align_corners=False)
        return F.grid_sample(t4, grid, mode="bilinear", padding_mode="reflection",
                             align_corners=False)[0].permute(1, 2, 0)

    shots = plan.get("shots", [])
    shot_starts = {int(round(s["t0"] * fps)) for s in shots}

    def loud_at(f):
        t = f / fps
        for s in plan["sections"]:
            if s["start"] - 1e-3 <= t < s["end"]:
                return s["loud"]
        return True

    # FX level by phase (hold grammar): low in quiet stretches, rising through the build, full on
    # the drop, so the drop lands as a jump in energy
    qs = cfg.get("quiet_scale", 0.25)
    level = None
    if any("phase" in s for s in shots):
        level = np.full(n_out, qs, np.float32)
        k = 0
        while k < len(shots):
            s = shots[k]
            if s.get("phase") == "build":
                j = k
                while j + 1 < len(shots) and shots[j + 1].get("phase") == "build":
                    j += 1
                a, b = int(round(s["t0"] * fps)), int(round(shots[j]["t1"] * fps))
                if b > a:
                    level[a:b] = np.linspace(qs, cfg.get("build_scale", 0.6), b - a)
                k = j + 1
                continue
            if s.get("phase") == "drop":
                level[int(round(s["t0"] * fps)):int(round(s["t1"] * fps))] = 1.0
            k += 1

    # ---- finish stats from the middle frame of every clip
    fin = None
    if do_finish:
        samples = []
        for rid, r in renders.items():
            rd = ClipReader(r.get("path", ""), (rw, rh))
            if rd.ok:
                f = rd.get(rd.n // 2)
                samples.append(torch.from_numpy(np.ascontiguousarray(f)).to(device).float() / 255)
        fin = Finisher(pack["finish"], lab_stats(samples) if samples else None)

    # ---- encode
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, f"{out_name}_{fmt.replace(':', 'x')}.mp4")
    cmd = [ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{dw}x{dh}",
           "-r", str(fps), "-i", "-", "-ss", str(plan["song_t0"]), "-t", str(plan["duration"]),
           "-i", os.path.join(run_dir, "song.wav"), "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
           "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k",
           "-shortest", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    prev, previews = None, []
    starts = {}
    black = np.zeros((rh, rw, 3), np.uint8)
    for f in range(n_out):
        t = f / fps
        ev = _active(plan["edl"], f)
        if ev is not None:
            key = (ev["kind"], ev["f0"], ev["render"])
            starts.setdefault(key, f)
            j = f - starts[key]
        else:
            j = 0
        rid = ev["render"] if ev else None
        img = frame_for(rid, t + (ev.get("dt", 0.0) if ev else 0.0)) if rid is not None else None
        if img is None and prev is not None:
            x = prev
        else:
            x = torch.from_numpy(np.ascontiguousarray(img if img is not None else black)).to(device).float() / 255
            if img is not None and ev is not None and ev.get("align"):
                x = warp(x, ev["align"])
        if ev is not None and ev["kind"] == "ghost" and ev.get("from") is not None:
            other = frame_for(ev["from"], t + ev.get("from_dt", 0.0))
            if other is not None:
                n = max(1, ev["f1"] - ev["f0"])
                a = cfg.get("ghost", 0.6) * (1 - j / n)
                xo = torch.from_numpy(np.ascontiguousarray(other)).to(device).float() / 255
                xo = warp(xo, ev.get("from_align"))
                x = x * (1 - a) + xo * a
        p = fx.fx_params(f, ev, ctl, loud_at(f), cfg, f in shot_starts or (ev is not None and ev["kind"] == "switch"), j, fps,
                         level=None if level is None else float(level[min(f, n_out - 1)]))
        y = fx.apply_fx(x, p, prev)
        prev = y
        # colour at render size (cheaper), then the upscale to delivery, then grain and vignette
        # at delivery size so the grain is full-resolution grain
        z = fin.grade(y) if fin is not None else y
        if (cw, ch) != (rw, rh):
            z = z[cy:cy + ch, cx:cx + cw]
        if (dw, dh) != (cw, ch):
            z = F.interpolate(z.permute(2, 0, 1)[None], size=(dh, dw), mode="bicubic",
                              align_corners=False, antialias=True)[0].permute(1, 2, 0).clamp(0, 1)
        if fin is not None:
            z = fin.texture(z, f)
        u8 = (z * 255 + 0.5).byte().cpu().numpy()
        proc.stdin.write(u8.tobytes())
        if f % max(1, n_out // 8) == 0 and len(previews) < 8:
            previews.append(u8)
        if progress:
            progress(f + 1, n_out)
    proc.stdin.close()
    proc.wait()
    if held:
        print(f"[OneShotMV] {sum(held.values())} lookups held a clip's first/last frame: "
              + ", ".join(f"r{k:03d} x{v}" for k, v in sorted(held.items())))
    edl_path = export.write_edl(plan, os.path.join(run_dir, f"{out_name}.edl"))
    try:  # the run is finished: FVRunResult (and so an SDK client) hands this video out
        import datetime as _dt
        json.dump({"video": out_path, "edl": edl_path, "frames": n_out, "missing_renders": missing,
                   "fx": fx_level or plan.get("fx_level") or "subtle", "finish": plan.get("finish_user") or {},
                   "at": _dt.datetime.now().isoformat(timespec="seconds")},
                  open(os.path.join(run_dir, "done.json"), "w", encoding="utf-8"), indent=1)
    except OSError:
        pass
    return {"video": out_path, "edl": edl_path, "frames": n_out, "missing_renders": missing,
            "previews": previews}
