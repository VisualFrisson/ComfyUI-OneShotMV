"""The setup check (fv/setup_check.py): each way a user's ComfyUI can differ gets one plain line
saying what to do, as early as it can be known, and nothing is queued when ComfyUI would refuse a job.

Runs without ComfyUI, torch or a GPU:  python tests/test_setup_check.py"""
import os
import sys
import tempfile
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import fake_comfy as fc  # noqa: E402
from fv import assemble, beat, director, graphs, setup_check  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    graphs.use_models("auto")
    yield
    graphs._SAGE.clear()
    graphs.use_models("auto")


def _song(seconds=45, sr=44100):
    t = np.arange(int(sr * seconds)) / sr
    env = np.where(t > 20, 1.0, 0.4)
    x = (0.6 * (np.mod(t, 0.5) < 0.03) * env + 0.2 * np.sin(2 * np.pi * 220 * t) * env).astype(np.float32)
    return np.stack([x, x]), sr


def _direct(out=None, **kw):
    audio, sr = _song()
    return director.direct(audio=audio, sr=sr, out_root=out or tempfile.mkdtemp(), seconds=20, do_stems=False,
                           log=lambda *a: None, **kw)


def _todo(**kw):
    return setup_check.fixes(setup_check.problems(setup_check.probe(**kw)), "24gb")


# ------------------------------------------------------------------ one line per problem, with the fix
def test_a_ready_comfyui_has_nothing_to_do(monkeypatch):
    fc.install(monkeypatch)
    assert _todo() == []
    rep = setup_check.report()
    assert rep["plan"] == [] and rep["render"] == []


def test_missing_res4lyf_is_one_line_with_the_fix(monkeypatch):
    fc.install(monkeypatch, packs=("ComfyUI-KJNodes", "ComfyUI-LTXVideo"))
    todo = _todo()
    assert len(todo) == 1  # its node and its res_2s sampler, in one line
    assert "RES4LYF" in todo[0] and "github.com/ClownsharkBatwing/RES4LYF" in todo[0]
    assert "restart ComfyUI" in todo[0] and "IMPORT FAILED" in todo[0]


def test_missing_kjnodes(monkeypatch):
    fc.install(monkeypatch, packs=("RES4LYF", "ComfyUI-LTXVideo"))
    todo = " ".join(_todo())
    assert "ComfyUI-KJNodes" in todo and "LTX2AttentionTunerPatch" in todo


def test_comfyui_too_old_for_krea2_and_generate_text(monkeypatch):
    fc.install(monkeypatch, old_comfy=True)
    todo = " ".join(_todo())
    assert "Update ComfyUI: its CLIPLoader node has no 'krea2' type." in todo
    assert "Update ComfyUI: this one has no TextGenerate node." in todo


def test_a_missing_or_renamed_model_file_names_its_folder(monkeypatch):
    fc.install(monkeypatch, drop_files={"krea2_turbo_fp8_scaled.safetensors", "LTX2.3_Soft_Enhance.safetensors"})
    todo = " ".join(_todo())
    assert "krea2_turbo_fp8_scaled.safetensors (models/diffusion_models)" in todo
    assert "LTX2.3_Soft_Enhance.safetensors (models/loras)" in todo
    assert "24 GB set" in todo and "fv/models.json" in todo
    assert "ltx-2.3-22b-dev-fp8" not in todo  # found by name in its sub-folder


def test_a_preview_needs_only_the_language_model(monkeypatch):
    fc.install(monkeypatch, packs=(), drop_files={"krea2_turbo_fp8_scaled.safetensors"})
    rep = setup_check.report()
    assert rep["plan"] == []  # the World can be written; the render can't
    assert any("RES4LYF" in x for x in rep["render"])
    assert any("krea2_turbo_fp8_scaled" in x for x in rep["render"])


def test_the_report_leaves_the_running_jobs_model_set_alone(monkeypatch):
    fc.install(monkeypatch)
    graphs.use_models("24gb")
    setup_check.report("full")
    assert graphs.M["_set"] == "24gb"


# ------------------------------------------------------------------ as early as it can be known
def test_the_director_stops_before_planning(monkeypatch):
    fc.install(monkeypatch, packs=("ComfyUI-KJNodes", "ComfyUI-LTXVideo"))

    def planned(*a, **k):
        raise AssertionError("the song was analysed before the check")
    monkeypatch.setattr(beat, "analyze", planned)
    out = tempfile.mkdtemp()
    with pytest.raises(setup_check.NotReady) as e:
        _direct(out)
    assert "RES4LYF" in str(e.value) and str(e.value).startswith("It can't start yet.")
    assert not os.path.isdir(os.path.join(out, "oneshot-mv", "runs"))


def test_a_preview_still_plans_without_the_render_packs(monkeypatch):
    posted = fc.install(monkeypatch, packs=())
    res = _direct(queue=False)
    assert posted == [] and os.path.isfile(os.path.join(res["run_dir"], "plan.json"))


def test_every_chain_is_checked_before_any_job_is_queued(monkeypatch):
    posted = fc.install(monkeypatch, packs=("ComfyUI-KJNodes", "ComfyUI-LTXVideo"))
    chain = [graphs.free_memory().nodes,
             graphs.krea2_portrait(prompt="x", seed=1, prefix="p", size1=(1344, 768), size2=(2016, 1152)).nodes]
    with pytest.raises(setup_check.NotReady):
        director.queue_chain_front(chain)
    assert posted == []


def test_a_complete_setup_queues_the_run(monkeypatch):
    posted = fc.install(monkeypatch)
    res = _direct()
    assert len(posted) == len(res["prompt_ids"]) > 20
    assert any(n["inputs"].get("unet_name") == "Krea2\\krea2_turbo_fp8_scaled.safetensors"
               for g in posted for n in g.values())


# ------------------------------------------------------------------ while it renders
def test_a_cut_with_no_singing_clips_says_why_instead_of_cutting(monkeypatch):
    fc.install(monkeypatch)
    res = _direct()
    with pytest.raises(RuntimeError) as e:  # nothing rendered (every clip failed)
        assemble.assemble(res["run_dir"])
    assert "None of the" in str(e.value) and "nothing to cut" in str(e.value)


def test_krea_raw_keeps_its_lora_backups_in_ram(monkeypatch):
    """Oct 31: the full set's RAW pass ran at 150-300 s a step under --highvram (the LoRA's weight
    backups filled the card); LTX already had Weight Backups to RAM, the stills didn't."""
    fc.install(monkeypatch)
    graphs.use_models("full")
    g = graphs.krea2_portrait(prompt="p", seed=1, prefix="t/p000", size1=(1024, 576), size2=(1600, 900)).nodes
    lora = next(n for n in g.values() if n["class_type"] == "LoraLoaderModelOnly")
    assert g[str(lora["inputs"]["model"][0])]["class_type"] == "FVOffloadToCPU"


# ------------------------------------------------------------------ the Studio
@pytest.fixture
def studio(monkeypatch):
    fc.install(monkeypatch)  # folder_paths, which the Studio imports
    srv = types.ModuleType("server")
    srv.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=None, prompt_queue=None))
    monkeypatch.setitem(sys.modules, "server", srv)
    from fv import studio as st
    return st, srv


def test_studio_refuses_make_the_video_at_once(monkeypatch, studio):
    st, _ = studio
    fc.install(monkeypatch, packs=("ComfyUI-KJNodes", "ComfyUI-LTXVideo"))
    assert any("RES4LYF" in x for x in st._cant_start({"render": True}))
    assert st._cant_start({"render": False}) == []  # the plan can still be written


def test_studio_shows_why_a_job_of_the_run_failed(monkeypatch, studio):
    st, srv = studio
    run = "my-video-20261030-2015"

    def job(prefix, status, msg=""):
        g = {"1": {"class_type": "SaveVideo", "inputs": {"filename_prefix": prefix}}}
        st_ = {"status_str": status, "messages": [["execution_error", {"node_type": "PathchSageAttentionKJ",
                                                                      "exception_message": msg}]] if msg else []}
        return {"prompt": (1, "id", g, {}, []), "status": st_}
    hist = {"a": job(f"oneshot-mv/runs/{run}/r000A", "error", "cannot import name 'sageattn_qk_int8_pv_fp8_cuda'"),
            "b": job(f"oneshot-mv/runs/{run}-2/r000A", "error", "another run's error"),
            "c": job(f"oneshot-mv/runs/{run}/r001A", "success")}
    srv.PromptServer.instance.prompt_queue = types.SimpleNamespace(get_history=lambda max_items=None: hist)
    errs = st._run_errors(run)
    assert [e["node"] for e in errs] == ["PathchSageAttentionKJ"]
    assert "sageattn_qk_int8_pv_fp8_cuda" in errs[0]["error"]


if __name__ == "__main__":  # pytest from the pack root would import the pack itself
    sys.exit(pytest.main([__file__, "-q", f"--confcutdir={HERE}", f"--rootdir={HERE}", "--import-mode=importlib",
                          "-p", "no:cacheprovider"]))
