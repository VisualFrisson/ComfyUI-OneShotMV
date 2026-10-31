"""The LTX jobs use SageAttention only when the installed package and the GPU have the kernel that
fv/models.json names. Otherwise KJNodes' patch fails every LTX clip and the run is cut with no
singer clips (reported Oct 30 by pxl-pshr, with SageAttention 1.x).

Runs without ComfyUI, torch or a GPU:  python tests/test_sage_kernel.py
The stand-in packages export what the real ones do: SageAttention 1.0.6 (PyPI's latest) has only
sageattn and sageattn_varlen; 2.2 also has the CUDA kernels, and its fp8 kernel stops with "SM89
kernel is not available" when it wasn't built for the card (core.SM89_ENABLED)."""
import importlib.machinery
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fv import graphs  # noqa: E402

FP8 = "sageattn_qk_int8_pv_fp8_cuda++"  # the default in fv/models.json


def _sage1():
    m = types.ModuleType("sageattention")
    m.__spec__ = importlib.machinery.ModuleSpec("sageattention", None)  # "installed", for find_spec
    m.sageattn = m.sageattn_varlen = lambda *a, **k: None
    m.core = types.ModuleType("sageattention.core")
    return m


def _sage2(sm89_built=True):
    m = _sage1()
    for f in ("sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp8_cuda",
              "sageattn_qk_int8_pv_fp8_cuda_sm90"):
        setattr(m, f, lambda *a, **k: None)
    m.core.SM80_ENABLED, m.core.SM89_ENABLED, m.core.SM90_ENABLED = True, sm89_built, False
    return m


def _torch(cap):
    t = types.ModuleType("torch")
    t.cuda = types.SimpleNamespace(is_available=lambda: cap is not None, current_device=lambda: 0,
                                   get_device_capability=lambda i=0: cap)
    return t


@pytest.fixture
def machine(monkeypatch):
    def setup(sage, cap):
        monkeypatch.setitem(sys.modules, "torch", _torch(cap))
        if sage is None:
            monkeypatch.setitem(sys.modules, "sageattention", None)  # import fails, as when not installed
        else:
            monkeypatch.setitem(sys.modules, "sageattention", sage)
            monkeypatch.setitem(sys.modules, "sageattention.core", sage.core)
        getattr(graphs, "_SAGE", {}).clear()
    yield setup
    getattr(graphs, "_SAGE", {}).clear()


@pytest.mark.parametrize("sage, cap, mode, want", [
    (_sage2(), (8, 9), FP8, FP8),                       # RTX 4090 with SageAttention 2: as configured
    (_sage2(), (12, 0), FP8, FP8),                      # RTX 5090
    (_sage1(), (8, 9), FP8, None),                      # pip install sageattention (1.0.6): no fp8 kernel
    (_sage2(), (8, 6), FP8, None),                      # RTX 3090: fp8 needs compute 8.9
    (_sage2(sm89_built=False), (8, 9), FP8, None),      # built without the sm89 kernels
    (None, (8, 9), FP8, None),                          # not installed (as before)
    (_sage2(), None, FP8, None),                        # no CUDA
    (_sage2(), (8, 6), "auto", "auto"),                 # auto: Sage picks the kernel for the card
    (_sage1(), (8, 6), "auto", "auto"),
    (_sage2(), (7, 5), "auto", None),                   # RTX 20: below what SageAttention supports
    (_sage2(), (8, 9), "", None),                       # Sage turned off in models.json
])
def test_sage_mode(machine, sage, cap, mode, want):
    machine(sage, cap)
    assert graphs.sage_mode(mode) == want


def _clip_nodes():
    g = graphs.ltx_render(preset="quality", fmt="16:9", key_path="/k.png", prompt="p", negative="n", frames=49,
                          seed=1, prefix="t/r000A", audio_path="/v.wav").nodes
    return [n for n in g.values() if n["class_type"] == "PathchSageAttentionKJ"]


def test_clip_leaves_the_patch_out_on_sage1(machine, monkeypatch):
    machine(_sage1(), (8, 9))
    monkeypatch.setitem(graphs.M, "ltx23_sage", FP8)
    assert _clip_nodes() == []


def test_clip_keeps_the_patch_on_a_4090(machine, monkeypatch):
    machine(_sage2(), (8, 9))
    monkeypatch.setitem(graphs.M, "ltx23_sage", FP8)
    assert [n["inputs"]["sage_attention"] for n in _clip_nodes()] == [FP8]


if __name__ == "__main__":  # pytest from the pack root would import the pack itself
    here = os.path.dirname(os.path.abspath(__file__))
    sys.exit(pytest.main([__file__, "-q", f"--confcutdir={here}", f"--rootdir={here}", "--import-mode=importlib",
                          "-p", "no:cacheprovider"]))
