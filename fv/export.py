"""Timeline export: CMX3600 EDL of the cut (one event per edit decision), so the edit can be
re-opened in Resolve / Premiere with the rendered clips as sources."""
from __future__ import annotations

import os


def _tc(frames: int, fps: float) -> str:
    fr = int(round(fps))
    h, rem = divmod(int(frames), 3600 * fr)
    m, rem = divmod(rem, 60 * fr)
    s, f = divmod(rem, fr)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def _flatten(plan: dict) -> list[dict]:
    """The cut's events as one track: later inserts cut into the shots under them."""
    events = sorted(plan["edl"], key=lambda e: (e["f0"], -1 if e["kind"] == "main" else 0))
    flat = []
    for e in events:
        if e["f1"] <= e["f0"]:
            continue
        flat = [x for x in flat if not (e["f0"] <= x["f0"] and x["f1"] <= e["f1"])]
        cut = []
        for x in flat:
            if x["f0"] < e["f0"] < x["f1"]:
                cut.append(dict(x, f1=e["f0"]))
                if e["f1"] < x["f1"]:
                    cut.append(dict(x, f0=e["f1"]))
            elif x["f0"] < e["f1"] < x["f1"]:
                cut.append(dict(x, f0=e["f1"]))
            else:
                cut.append(x)
        flat = sorted(cut + [e], key=lambda x: x["f0"])
    return flat


def write_edl(plan: dict, path: str) -> str:
    fps = plan["out_fps"]
    renders = {r["id"]: r for r in plan["renders"]}
    flat = _flatten(plan)
    lines = [f"TITLE: OneShot MV {plan['pack']} {plan['format']}", "FCM: NON-DROP FRAME", ""]
    for i, e in enumerate(flat, 1):
        r = renders.get(e["render"], {})
        name = os.path.basename(r.get("path", f"render_{e['render']}"))
        src_in = int(round((e["f0"] / fps - r.get("t_start", 0.0)) * fps))
        dur = e["f1"] - e["f0"]
        reel = f"R{int(e['render'] or 0):03d}"
        lines.append(f"{i:03d}  {reel:<8} V     C        {_tc(src_in, fps)} {_tc(src_in + dur, fps)} "
                     f"{_tc(e['f0'], fps)} {_tc(e['f1'], fps)}")
        lines.append(f"* FROM CLIP NAME: {name}")
        lines.append(f"* FV EVENT: {e['kind']}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _base(path: str) -> str:
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]


def _url(path: str) -> str:
    from urllib.parse import quote
    p = os.path.abspath(path).replace("\\", "/")
    if not p.startswith("/"):
        p = "/" + p  # E:/x -> /E:/x
    return "file://localhost" + quote(p, safe="/:")


def write_xml(plan: dict, path: str, final_video: str = "", song: str = "") -> str:
    """Final Cut Pro 7 XML (xmeml 5) of the cut, which Premiere Pro and DaVinci Resolve import:
    track 1 is every shot's rendered clip with its in and out points, track 2 the finished
    video (off, for reference), and the song on the audio track."""
    from xml.sax.saxutils import escape as x
    fps = float(plan["out_fps"])
    tb = int(round(fps))
    renders = {r["id"]: r for r in plan["renders"]}
    flat = _flatten(plan)
    n_out = int(round(float(plan["duration"]) * fps))
    from . import formats
    dw, dh = formats.deliver_size(plan["format"])
    rw, rh = formats.render_size(plan["format"], plan["preset"])
    delay0 = float(plan.get("sync_delay", formats.PRESETS.get(plan["preset"], {}).get("sync_delay", 0.0)))
    rate = f"<rate><timebase>{tb}</timebase><ntsc>FALSE</ntsc></rate>"
    files_done = set()

    def file_xml(fid, p, frames, w=None, h=None, audio=False, file_fps=None):
        if fid in files_done:
            return f'<file id="{fid}"/>'
        files_done.add(fid)
        media = (f"<media><video><samplecharacteristics><width>{w}</width><height>{h}</height>"
                 f"</samplecharacteristics></video>" + ("<audio><channelcount>2</channelcount></audio>" if audio else "") +
                 "</media>") if w else \
            ("<media><audio><samplecharacteristics><depth>16</depth><samplerate>48000</samplerate>"
             "</samplecharacteristics><channelcount>2</channelcount></audio></media>")
        ffps = int(round(file_fps or fps))
        frate = f"<rate><timebase>{ffps}</timebase><ntsc>FALSE</ntsc></rate>"
        fdur = int(round(frames * ffps / fps))
        return (f'<file id="{fid}"><name>{x(_base(p))}</name><pathurl>{x(_url(p))}</pathurl>{frate}'
                f"<duration>{fdur}</duration>{media}</file>")

    v1 = []
    for i, e in enumerate(flat, 1):
        r = renders.get(e["render"], {})
        p = r.get("path") or ""
        if r.get("path_before") and not os.path.exists(p):
            p = r["path_before"]
        clip_frames = int(round(float(r.get("frames", 0)) / float(r.get("fps", fps) or fps) * fps)) or (e["f1"] - e["f0"])
        src_in = int(round((e["f0"] / fps - float(r.get("t_start", 0.0)) - float(r.get("sync_delay", delay0))) * fps))
        src_in = max(0, min(src_in, max(0, clip_frames - (e["f1"] - e["f0"]))))
        fid = f"file-r{int(e['render'] or 0):03d}-{_base(p)}"
        v1.append(f'<clipitem id="clipitem-{i}"><name>{x(_base(p))}</name><enabled>TRUE</enabled>'
                  f"<duration>{clip_frames}</duration>{rate}<start>{e['f0']}</start><end>{e['f1']}</end>"
                  f"<in>{src_in}</in><out>{src_in + e['f1'] - e['f0']}</out>"
                  f"{file_xml(fid, p, clip_frames, rw, rh, file_fps=float(r.get('fps', fps) or fps))}"
                  f"<comments><mastercomment1>{x(e.get('kind', 'main'))}</mastercomment1></comments></clipitem>")
    v2 = []
    if final_video and os.path.exists(final_video):
        v2.append(f'<clipitem id="clipitem-final"><name>{x(_base(final_video))}</name><enabled>FALSE</enabled>'
                  f"<duration>{n_out}</duration>{rate}<start>0</start><end>{n_out}</end><in>0</in><out>{n_out}</out>"
                  f"{file_xml('file-final', final_video, n_out, dw, dh, audio=True)}</clipitem>")
    a1 = []
    if song and os.path.exists(song):
        s_in = int(round(float(plan.get("song_t0", 0.0)) * fps))
        a1.append(f'<clipitem id="clipitem-song"><name>{x(_base(song))}</name><enabled>TRUE</enabled>'
                  f"<duration>{s_in + n_out}</duration>{rate}<start>0</start><end>{n_out}</end>"
                  f"<in>{s_in}</in><out>{s_in + n_out}</out>{file_xml('file-song', song, s_in + n_out)}"
                  "<sourcetrack><mediatype>audio</mediatype><trackindex>1</trackindex></sourcetrack></clipitem>")
    name = f"OneShot MV {_base(os.path.dirname(path))}"
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n<xmeml version="5">'
           f'<sequence id="sequence-1"><name>{x(name)}</name><duration>{n_out}</duration>{rate}'
           f"<timecode>{rate}<string>00:00:00:00</string><frame>0</frame><displayformat>NDF</displayformat></timecode>"
           "<media><video><format><samplecharacteristics>"
           f"{rate}<width>{dw}</width><height>{dh}</height><pixelaspectratio>square</pixelaspectratio>"
           "</samplecharacteristics></format>"
           f"<track>{''.join(v1)}</track>" + (f"<track>{''.join(v2)}</track>" if v2 else "") +
           "</video><audio>" + (f"<track>{''.join(a1)}</track>" if a1 else "") + "</audio></media></sequence></xmeml>\n")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(xml)
    return path
