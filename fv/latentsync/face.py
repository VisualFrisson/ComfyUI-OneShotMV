"""Face box + 106 landmarks for LatentSync's alignment, without the insightface package.

LatentSync finds the face with insightface's buffalo_l models (SCRFD det_10g + 2d106det). The
insightface package needs a C++ build on Windows, so this runs the same two ONNX models with
onnxruntime and the same pre/post-processing (after insightface's scrfd.py / landmark.py, MIT).
The models are looked for in ComfyUI/models/insightface/(models/)buffalo_l/."""
from __future__ import annotations

import os

import cv2
import numpy as np

DET_SIZE = 512  # LatentSync's INSIGHTFACE_DETECT_SIZE


def model_dir() -> str | None:
    roots = []
    try:
        import folder_paths  # type: ignore
        roots.append(os.path.join(folder_paths.models_dir, "insightface"))
    except Exception:
        pass
    roots.append(os.path.join(os.path.expanduser("~"), ".insightface"))
    for r in roots:
        for sub in ("models/buffalo_l", "buffalo_l"):
            d = os.path.join(r, *sub.split("/"))
            if os.path.exists(os.path.join(d, "det_10g.onnx")) and os.path.exists(os.path.join(d, "2d106det.onnx")):
                return d
    return None


def _session(path: str, cuda: bool = True):
    import onnxruntime as ort
    prov = ["CPUExecutionProvider"]
    if cuda and "CUDAExecutionProvider" in ort.get_available_providers():
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(path, providers=prov)


def _nms(dets: np.ndarray, thresh: float = 0.4) -> list[int]:
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        ovr = w * h / (areas[i] + areas[order[1:]] - w * h)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


class Detector:
    """SCRFD det_10g at 512 x 512 (strides 8 / 16 / 32, two anchors each)."""

    def __init__(self, path: str, cuda: bool = True):
        self.sess = _session(path, cuda)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.strides, self.na = [8, 16, 32], 2

    def __call__(self, img: np.ndarray, thresh: float = 0.5) -> np.ndarray:
        """[[x1, y1, x2, y2, score], ...] for one image (as given, 3 channels)."""
        h0, w0 = img.shape[:2]
        size = DET_SIZE
        if h0 / w0 > 1.0:
            nh, nw = size, int(size / (h0 / w0))
        else:
            nw, nh = size, int(size * (h0 / w0))
        scale = nh / h0
        canvas = np.zeros((size, size, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (size, size), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.sess.run(self.output_names, {self.input_name: blob})
        fmc = len(self.strides)
        boxes, scores = [], []
        for k, st in enumerate(self.strides):
            sc = outs[k].reshape(-1)
            bb = outs[k + fmc].reshape(-1, 4) * st
            hh, ww = size // st, size // st
            centers = np.stack(np.mgrid[:hh, :ww][::-1], axis=-1).astype(np.float32)
            centers = (centers * st).reshape(-1, 2)
            centers = np.stack([centers] * self.na, axis=1).reshape(-1, 2)
            pos = np.where(sc >= thresh)[0]
            if not len(pos):
                continue
            c = centers[pos]
            d = bb[pos]
            boxes.append(np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], -1))
            scores.append(sc[pos])
        if not boxes:
            return np.zeros((0, 5), np.float32)
        b = np.concatenate(boxes) / scale
        s = np.concatenate(scores)
        dets = np.hstack([b, s[:, None]]).astype(np.float32)
        return dets[_nms(dets)]


class Landmarks:
    """2d106det: 106 points from a 192 x 192 crop around the face box (scale 1.5)."""

    def __init__(self, path: str, cuda: bool = True):
        self.sess = _session(path, cuda)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.size = 192

    def __call__(self, img: np.ndarray, box) -> np.ndarray:
        x1, y1, x2, y2 = box[:4]
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        s = self.size / (max(w, h) * 1.5)
        M = np.array([[s, 0, self.size / 2 - s * cx], [0, s, self.size / 2 - s * cy]], np.float32)
        crop = cv2.warpAffine(img, M, (self.size, self.size), borderValue=0.0)
        # the model normalises itself (its graph starts with Sub / Mul): mean 0, std 1
        blob = cv2.dnn.blobFromImage(crop, 1.0, (self.size, self.size), (0.0, 0.0, 0.0), swapRB=True)
        pred = self.sess.run(self.output_names, {self.input_name: blob})[0][0].reshape(-1, 2)[-106:]
        pred = (pred + 1) * (self.size // 2)
        IM = cv2.invertAffineTransform(M)
        return (pred @ IM[:, :2].T + IM[:, 2]).astype(np.float32)


# LatentSync's subset of the 106 points that outlines the lower face (face_detector.py)
LMK_ADAPT = [1, 10, 12, 14, 16, 3, 5, 7, 0, 23, 21, 19, 32, 30, 28, 26, 17, 43, 48, 49, 51, 50, 102, 103,
             104, 105, 101, 73, 74, 86]


class FaceDetector:
    """Drop-in for LatentSync's FaceDetector: (box, 106 landmarks) of the biggest face, or (None, None)."""

    def __init__(self, cuda: bool = True):
        d = model_dir()
        if d is None:
            raise FileNotFoundError("insightface buffalo_l models (det_10g.onnx, 2d106det.onnx) not found in "
                                    "ComfyUI/models/insightface/")
        self.det = Detector(os.path.join(d, "det_10g.onnx"), cuda)
        self.lmk = Landmarks(os.path.join(d, "2d106det.onnx"), cuda)

    def __call__(self, frame: np.ndarray, threshold: float = 0.5):
        f_h, f_w = frame.shape[:2]
        best, best_area = None, 0
        for d in self.det(frame, threshold):
            x1, y1, x2, y2, sc = d
            w, h = x2 - x1, y2 - y1
            if w < 50 or h < 80 or w / h > 1.5 or w / h < 0.2 or sc < threshold:
                continue
            if w * h > best_area:
                best, best_area = d, w * h
        if best is None:
            return None, None
        lmk = np.round(self.lmk(frame, best)).astype(np.int_)
        half = np.mean([lmk[74], lmk[73]], axis=0)
        sub = lmk[LMK_ADAPT]
        dist = np.max(sub[:, 1]) - half[1]
        x1, y1, x2, y2 = np.min(sub[:, 0]), int(half[1] - dist), np.max(sub[:, 0]), np.max(sub[:, 1])
        if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
            x1, y1, x2, y2 = best[:4].astype(np.int_).tolist()
        y2 += int((x2 - x1) * 0.1)
        x1 -= int((x2 - x1) * 0.05)
        x2 += int((x2 - x1) * 0.05)
        return (max(0, x1), max(0, y1), min(f_w, x2), min(f_h, y2)), lmk
