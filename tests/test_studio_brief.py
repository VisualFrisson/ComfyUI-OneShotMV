"""Going back to a video's Brief works in any browser: the Studio keeps the brief a run was started
with in the run's folder, and rebuilds it from the plan for runs it didn't start (the song from
ComfyUI's history). Reported Oct 31: a run started in one browser opened as "made outside the
Studio" in another, showing the last brief instead.

Runs without ComfyUI:  python tests/test_studio_brief.py"""
import json
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import fake_comfy as fc  # noqa: E402

RUN = "demo-song-20261031-1314"
PLAN = {"format": "16:9", "preset": "quality", "seed": 3, "song_t0": 48.83, "song_t1": 78.83, "duration": 30.0,
        "cuts": {"low": "every 2 bars", "medium": "every bar", "high": "to the music (subtle)"},
        "crowd": "auto", "cast_order": "as written", "broll_share": 0.125, "fx_level": "subtle", "models": "full",
        "theme": {"theme": "Present day, all women: roller derby bout night"}, "shots": []}


def _history(song="Demo Artist - Demo Song.mp3"):
    g = {"1": {"class_type": "LoadAudio", "inputs": {"audio": song}}, "2": {"class_type": "FVDirector", "inputs": {}}}
    return {"abc123": {"prompt": (1, "abc123", g, {}, []), "status": {"status_str": "success", "messages": []},
                    "outputs": {"2": {"text": [{"filename": "jobs.json", "subfolder": f"oneshot-mv/runs/{RUN}"}]}}}}


@pytest.fixture
def studio(monkeypatch):
    fc.install(monkeypatch)
    hist = _history()
    q = types.SimpleNamespace(get_history=lambda max_items=None, prompt_id=None: hist if prompt_id in (None, "abc123") else {},
                              get_current_queue=lambda: ([], []))
    srv = types.ModuleType("server")
    srv.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=None, prompt_queue=q))
    monkeypatch.setitem(sys.modules, "server", srv)
    from fv import studio as st
    d = os.path.join(st._runs_root(), RUN)
    os.makedirs(d, exist_ok=True)
    json.dump(PLAN, open(os.path.join(d, "plan.json"), "w"))
    return st, d


def test_a_run_started_elsewhere_rebuilds_its_brief(studio):
    st, d = studio
    f = st._form(d, PLAN)
    assert f["song"] == "Demo Artist - Demo Song.mp3"
    assert f["theme"] == "Present day, all women: roller derby bout night"
    assert (f["seconds"], f["start"], f["seed"], f["preset"], f["cuts_medium"]) == (30.0, 48.83, 3, "quality", "every bar")


def test_the_studio_keeps_the_brief_it_started_with(studio):
    st, d = studio
    brief = {"name": "Demo Song", "song": "Demo Artist - Demo Song.mp3", "theme": "Present day, all women",
             "seconds": 30, "start": -1, "seed": 1}
    with open(os.path.join(st._jobs_dir(), "abc123.json"), "w") as fh:
        json.dump({"t": 0, "project": "Demo Song", "brief": brief}, fh)
    assert st._job("abc123") == {"state": "done", "run": RUN}
    assert st._form(d, PLAN) == brief  # "start": -1 kept, so a new video picks its own window again
    assert st._project(d) == "Demo Song"


if __name__ == "__main__":  # pytest from the pack root would import the pack itself
    sys.exit(pytest.main([__file__, "-q", f"--confcutdir={HERE}", f"--rootdir={HERE}", "--import-mode=importlib",
                          "-p", "no:cacheprovider"]))
