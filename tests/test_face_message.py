"""When the cut can't face-align its flashes, it says why, and the take check runs without OpenCV's
face files. Reported Oct 31 by a tester: in a run
whose clips all failed (SageAttention), the console said "this OpenCV build has no Haar cascades
... pip install opencv-python<5", though OpenCV 4.13 was fine, and sent them the wrong way.

Runs without ComfyUI:  python tests/test_face_message.py"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from fv import faces  # noqa: E402

RENDERS = [{"id": 0, "path": "/runs/x/r000A_00001.mp4"}, {"id": 1, "path": "/runs/x/r001A_00001.mp4"}]


def test_no_clips_is_not_blamed_on_opencv(tmp_path, monkeypatch):
    monkeypatch.setattr(faces, "method", lambda: "")
    assert faces.why_no_tracks(str(tmp_path), RENDERS) == "no clips rendered to look for faces in."


def test_no_detector_says_how_to_get_one(tmp_path, monkeypatch):
    for r in RENDERS:
        (tmp_path / os.path.basename(r["path"])).write_bytes(b"x")
    monkeypatch.setattr(faces, "method", lambda: "")
    msg = faces.why_no_tracks(str(tmp_path), RENDERS)
    assert "SyncNet weights" in msg and "opencv-python<5" in msg


def test_clips_without_faces(tmp_path, monkeypatch):
    for r in RENDERS:
        (tmp_path / os.path.basename(r["path"])).write_bytes(b"x")
    monkeypatch.setattr(faces, "method", lambda: "haar")
    assert faces.why_no_tracks(str(tmp_path), RENDERS) == "no faces were found in the 2 rendered clips."


def test_the_take_check_runs_without_face_cascades(monkeypatch):
    """OpenCV 5 has no cascade files; its empty classifier threw inside qc.check_clip, the Director
    caught it and passed every clip unchecked."""
    import numpy as np
    from fv import qc
    monkeypatch.setattr(qc, "_CASCADE", None)
    import cv2
    monkeypatch.setattr(cv2.data, "haarcascades", "/nowhere/")
    assert list(qc._faces(np.zeros((360, 640), np.uint8))) == [] and qc._CASCADE is False


if __name__ == "__main__":  # pytest from the pack root would import the pack itself
    sys.exit(pytest.main([__file__, "-q", f"--confcutdir={HERE}", f"--rootdir={HERE}", "--import-mode=importlib",
                          "-p", "no:cacheprovider"]))
