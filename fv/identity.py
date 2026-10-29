"""Who is in the frame, and how they are framed: face boxes with five landmarks (SCRFD det_10g)
and identity embeddings (ArcFace w600k_r50), from insightface's buffalo_l models, run with
onnxruntime (no insightface package; the pre- and post-processing follow insightface's scrfd.py,
arcface_onnx.py and face_align.py, MIT).

Used by take QC (fv/qc.py):
  identity - artist runs: a look whose face is no longer the artist's (an image edit that changes
             the hairstyle sometimes changes the person too, for example into a young man)
  framing  - the head out of frame in a wide shot (a walking shot that loses the head), or a
             medium shot that came out as a tight close-up ("all mouth")
The models are looked for in ComfyUI/models/insightface/(models/)buffalo_l/."""
from __future__ import annotations

import os

import numpy as np

ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)
_MODELS: dict = {}


def model_dir() -> str | None:
    roots = []
    try:
        import folder_paths  # type: ignore
        roots.append(os.path.join(folder_paths.models_dir, "insightface"))
    except Exception:
        pass
    roots.append(os.path.join(os.path.expanduser("~"), ".insightface"))
    extra = os.environ.get("FV_INSIGHTFACE_DIR")
    if extra:
        roots.insert(0, extra)
    for r in roots:
        for sub in ("models/buffalo_l", "buffalo_l", ""):
            d = os.path.join(r, *[s for s in sub.split("/") if s])
            if os.path.exists(os.path.join(d, "det_10g.onnx")) and os.path.exists(os.path.join(d, "w600k_r50.onnx")):
                return d
    return None


def available() -> bool:
    if model_dir() is None:
        return False
    try:
        import onnxruntime  # noqa: F401
        import cv2  # noqa: F401
    except Exception:
        return False
    return True


def _session(path: str):
    import onnxruntime as ort
    prov = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(path, providers=prov)


def _models():
    if not _MODELS:
        d = model_dir()
        if d is None:
            raise FileNotFoundError("insightface buffalo_l (det_10g.onnx, w600k_r50.onnx) not found")
        _MODELS["det"] = _session(os.path.join(d, "det_10g.onnx"))
        _MODELS["rec"] = _session(os.path.join(d, "w600k_r50.onnx"))
    return _MODELS["det"], _MODELS["rec"]


def free() -> None:
    _MODELS.clear()


def _nms(dets: np.ndarray, thresh: float = 0.4) -> list[int]:
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        order = order[np.where(inter / (areas[i] + areas[order[1:]] - inter) <= thresh)[0] + 1]
    return keep


def detect(img_bgr: np.ndarray, thresh: float = 0.5, size: int = 640) -> list[dict]:
    """Faces, biggest first: {"box": (x1, y1, x2, y2), "score", "kps": 5x2} in image pixels."""
    import cv2
    det, _ = _models()
    h0, w0 = img_bgr.shape[:2]
    if h0 / w0 > 1.0:
        nh, nw = size, int(size * w0 / h0)
    else:
        nw, nh = size, int(size * h0 / w0)
    scale = nh / h0
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:nh, :nw] = cv2.resize(img_bgr, (nw, nh))
    blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (size, size), (127.5, 127.5, 127.5), swapRB=True)
    name = det.get_inputs()[0].name
    outs = det.run(None, {name: blob})
    strides, na = [8, 16, 32], 2
    fmc = len(strides)
    boxes, scores, kpss = [], [], []
    for k, st in enumerate(strides):
        sc = outs[k].reshape(-1)
        bb = outs[k + fmc].reshape(-1, 4) * st
        kp = outs[k + 2 * fmc].reshape(-1, 10) * st if len(outs) >= 3 * fmc else None
        hh, ww = size // st, size // st
        centers = np.stack(np.mgrid[:hh, :ww][::-1], axis=-1).astype(np.float32)
        centers = (centers * st).reshape(-1, 2)
        centers = np.stack([centers] * na, axis=1).reshape(-1, 2)
        pos = np.where(sc >= thresh)[0]
        if not len(pos):
            continue
        c, d = centers[pos], bb[pos]
        boxes.append(np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], -1))
        scores.append(sc[pos])
        if kp is not None:
            kk = kp[pos].reshape(-1, 5, 2) + c[:, None, :]
            kpss.append(kk)
    if not boxes:
        return []
    b = np.concatenate(boxes) / scale
    s = np.concatenate(scores)
    kps = np.concatenate(kpss) / scale if kpss else None
    dets = np.hstack([b, s[:, None]]).astype(np.float32)
    keep = _nms(dets)
    out = [{"box": tuple(float(v) for v in dets[i, :4]), "score": float(dets[i, 4]),
            "kps": kps[i] if kps is not None else None} for i in keep]
    out.sort(key=lambda f: -(f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))
    return out


def embed_face(img_bgr: np.ndarray, face: dict) -> np.ndarray | None:
    """Unit-length ArcFace embedding of one detected face."""
    import cv2
    if face.get("kps") is None:
        return None
    _, rec = _models()
    M, _ = cv2.estimateAffinePartial2D(np.asarray(face["kps"], np.float32), ARCFACE_DST, method=cv2.LMEDS)
    if M is None:
        return None
    crop = cv2.warpAffine(img_bgr, M, (112, 112), borderValue=0.0)
    blob = cv2.dnn.blobFromImage(crop, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
    v = rec.run(None, {rec.get_inputs()[0].name: blob})[0][0].astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else None


def embed(img_bgr: np.ndarray) -> np.ndarray | None:
    """Embedding of the biggest face in an image, None without a face."""
    fs = detect(img_bgr)
    return embed_face(img_bgr, fs[0]) if fs else None


def embed_path(path: str) -> np.ndarray | None:
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return embed(img) if img is not None else None


def similarity(a, b) -> float | None:
    if a is None or b is None:
        return None
    return float(np.dot(a, b))


def sample_frames(path: str, n: int = 5, skip_end: int = 2) -> list:
    """n evenly spaced full-size frames of a clip (BGR), skipping LTX's trailing frames."""
    import cv2
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    if total > 0:
        for i in np.linspace(2, max(2, total - 1 - skip_end), n).round().astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                out.append(f)
    cap.release()
    return out


def image_report(path: str, ref=None) -> dict:
    """Framing and identity of a still (a keyframe or portrait), as clip_report reports them."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return clip_report([img], ref, n=1) if img is not None else {}


def clip_report(frames_bgr: list, ref=None, n: int = 5) -> dict:
    """Face framing and identity over n evenly spaced frames of a clip (full-size BGR frames):
    face_frac (frames with a face), face_h (median face-box height / frame height), head_top
    (median face-box top / frame height; under 0 = the head runs off the top), id_sim (median
    similarity to `ref`), tilt (median degrees the eye line is off level: a sideways head is ~90),
    second_face (share of frames with a second face at least 0.6 the main one's height)."""
    if not frames_bgr:
        return {}
    idx = np.linspace(0, len(frames_bgr) - 1, n).round().astype(int)
    hs, tops, sims, found, tilts, twos = [], [], [], 0, [], 0
    for i in idx:
        f = frames_bgr[i]
        H = f.shape[0]
        fs = detect(f)
        if not fs:
            continue
        found += 1
        x1, y1, x2, y2 = fs[0]["box"]
        hs.append((y2 - y1) / H)
        tops.append(y1 / H)
        # the eye line's tilt (a portrait drawn lying sideways) and a second face of about the same
        # size (the same person drawn twice)
        k = fs[0].get("kps")
        if k is not None:
            dx, dy = float(k[1][0] - k[0][0]), float(k[1][1] - k[0][1])
            tilts.append(abs(np.degrees(np.arctan2(dy, dx))))
        if len(fs) > 1:
            h2 = fs[1]["box"][3] - fs[1]["box"][1]
            if h2 >= 0.6 * (y2 - y1):
                twos += 1
        if ref is not None:
            e = embed_face(f, fs[0])
            if e is not None:
                sims.append(float(np.dot(e, ref)))
    out = {"face_frac": round(found / len(idx), 2)}
    if found:
        out["second_face"] = round(twos / found, 2)
    if tilts:
        t = float(np.median(tilts))
        out["tilt"] = round(min(t, 180.0 - t), 1)  # eye line off level, either way round
    if hs:
        out["face_h"] = round(float(np.median(hs)), 3)
        out["head_top"] = round(float(np.median(tops)), 3)
    if sims:
        out["id_sim"] = round(float(np.median(sims)), 3)
    return out
