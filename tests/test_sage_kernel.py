"""SageAttention is chosen for the machine, so it never stops a run: the kernel in fv/models.json when
it can run here, else "auto" (SageAttention 2 picks the kernel for the card), else none. A clip that
still doesn't render is re-shot without it. Reported Oct 30 by pxl-pshr: with SageAttention 1 the
default kernel failed every LTX clip, and the run was cut with no singer clips.

Runs without ComfyUI, torch or a GPU:  python tests/test_sage_kernel.py
The stand-ins export what the real packages do: SageAttention 1.0.6 (PyPI's latest) has only sageattn
and sageattn_varlen; 2.2 also has the CUDA kernels, each only if it was built for the card
(core.SM80_ENABLED, SM89_ENABLED, SM90_ENABLED), and sageattn picks one by compute capability."""
import os
import re
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import fake_comfy as fc  # noqa: E402
from fv import director, graphs  # noqa: E402

FP8 = "sageattn_qk_int8_pv_fp8_cuda++"  # the default in fv/models.json
OLD_KJ = [m for m in fc.KJ_MODES if m != FP8]


@pytest.fixture(autouse=True)
def _clean():
    yield
    graphs._SAGE.clear()


@pytest.mark.parametrize("label, sage, cap, mode, kw, want", [
    ("RTX 4090, SageAttention 2", fc.sage2(), (8, 9), FP8, {}, FP8),
    ("RTX 5090", fc.sage2(), (12, 0), FP8, {}, FP8),
    ("RTX 3090: its card has no fp8 kernel, auto picks one", fc.sage2(), (8, 6), FP8, {}, "auto"),
    ("A100", fc.sage2(), (8, 0), FP8, {}, "auto"),
    ("H100 build without the sm89 kernels", fc.sage2(sm89=False, sm90=True), (9, 0), FP8, {}, "auto"),
    ("RTX 4090, a build without the sm89 kernels", fc.sage2(sm89=False), (8, 9), FP8, {}, None),
    ("pip install sageattention (version 1)", fc.sage1(), (8, 9), FP8, {}, None),
    ("version 1, auto", fc.sage1(), (8, 6), "auto", {}, None),
    ("RTX 2080: no SageAttention kernel", fc.sage2(), (7, 5), FP8, {}, None),
    ("not installed", None, (8, 9), FP8, {}, None),
    ("no CUDA (Mac)", fc.sage2(), None, FP8, {}, None),
    ("AMD (ROCm)", fc.sage2(), (11, 0), FP8, {"hip": True}, None),
    ("KJNodes without the ++ mode", fc.sage2(), (8, 9), FP8, {"kj_modes": OLD_KJ}, "auto"),
    ("KJNodes not loaded", fc.sage2(), (8, 9), FP8, {"packs": ("RES4LYF",)}, None),
    ("SageAttention 3 asked for, not installed", fc.sage2(), (12, 0), "sageattn3", {}, "auto"),
    ("turned off in fv/models.json", fc.sage2(), (8, 9), "", {}, None),
    ("auto on an RTX 3090", fc.sage2(), (8, 6), "auto", {}, "auto"),
])
def test_sage_mode(monkeypatch, label, sage, cap, mode, kw, want):
    fc.install(monkeypatch, sage=sage, cap=cap, **kw)
    assert graphs.sage_mode(mode) == want, label


def _clip_sage(g):
    return [n["inputs"]["sage_attention"] for n in g.values() if n["class_type"] == "PathchSageAttentionKJ"]


def _clip():
    return graphs.ltx_render(preset="quality", fmt="16:9", key_path="/k.png", prompt="p", negative="n", frames=49,
                             seed=1, prefix="t/r000A", audio_path="/v.wav").nodes


def test_clip_leaves_sage1_out(monkeypatch):
    fc.install(monkeypatch, sage=fc.sage1(), cap=(8, 9))
    assert _clip_sage(_clip()) == []


def test_clip_on_a_3090_uses_auto(monkeypatch):
    fc.install(monkeypatch, sage=fc.sage2(), cap=(8, 6))
    assert _clip_sage(_clip()) == ["auto"]


def test_clip_on_a_4090_keeps_the_configured_kernel(monkeypatch):
    fc.install(monkeypatch, sage=fc.sage2(), cap=(8, 9))
    assert _clip_sage(_clip()) == [FP8]


def test_the_brief_says_what_sage_does_here(monkeypatch):
    fc.install(monkeypatch, sage=fc.sage1(), cap=(8, 9))
    assert graphs.sage_report().startswith("SageAttention is off: the installed SageAttention")
    fc.install(monkeypatch, sage=fc.sage2(), cap=(8, 6))
    assert "picks the kernel for this card" in graphs.sage_report()
    fc.install(monkeypatch, sage=None, cap=(8, 9))
    assert graphs.sage_report().startswith("SageAttention isn't installed")


def test_strip_sage_keeps_the_graph_connected(monkeypatch):
    fc.install(monkeypatch, sage=fc.sage2(), cap=(8, 9))
    g = _clip()
    assert graphs.strip_sage(g) == 1 and _clip_sage(g) == []
    links = [v for n in g.values() for v in n["inputs"].values() if isinstance(v, list) and len(v) == 2]
    assert all(str(k) in g for k, _ in links)


def test_clips_that_did_not_render_are_reshot_without_sage(monkeypatch):
    """A whole run under the stand-in ComfyUI on an RTX 4090 with SageAttention 2: the first takes
    use the fp8 kernel; when none of them renders (Sage failed in a way the checks can't see), the
    take check re-shoots every one of them without Sage."""
    posted = fc.install(monkeypatch, sage=fc.sage2(), cap=(8, 9))
    sr = 44100
    t = np.arange(int(sr * 45)) / sr
    env = np.where(t > 20, 1.0, 0.4)
    x = (0.6 * (np.mod(t, 0.5) < 0.03) * env + 0.2 * np.sin(2 * np.pi * 220 * t) * env).astype(np.float32)
    res = director.direct(audio=np.stack([x, x]), sr=sr, out_root=tempfile.mkdtemp(), seconds=20, do_stems=False,
                          log=lambda *a: None)
    leads = [g for g in posted if any(n["class_type"] == "SaveVideo" and re.search(r"/r\d{3}A$", n["inputs"]["filename_prefix"])
                                      for n in g.values())]  # the singing clips
    assert leads and all(_clip_sage(g) == [FP8] for g in leads)
    n0 = len(posted)
    director.qc_pass(res["run_dir"], pass_idx=0, max_passes=2, stage="leads", log=lambda *a: None)
    reshoots = [g for g in posted[n0:] if any(n["class_type"] == "SaveVideo" for n in g.values())]
    assert len(reshoots) == len(leads) and all(_clip_sage(g) == [] for g in reshoots)


if __name__ == "__main__":  # pytest from the pack root would import the pack itself
    sys.exit(pytest.main([__file__, "-q", f"--confcutdir={HERE}", f"--rootdir={HERE}", "--import-mode=importlib",
                          "-p", "no:cacheprovider"]))
