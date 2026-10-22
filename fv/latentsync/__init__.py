"""LatentSync 1.6 (ByteDance, Apache-2.0), vendored for Frisson Vision's lip-sync repair pass.

Only the inference parts: the 3D UNet (models/), the Whisper-tiny audio encoder (whisper/, from
openai/whisper via MuseTalk, MIT) and the face alignment (affine_transform.py). The face
detector is replaced by fv/latentsync/face.py (insightface's SCRFD + 106-point landmark ONNX
models run directly with onnxruntime) and the video/audio I/O by fv/latentsync/run.py.
Upstream: https://github.com/bytedance/LatentSync (see LICENSE and NOTICE)."""
