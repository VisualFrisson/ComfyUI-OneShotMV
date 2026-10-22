"""LatentSync 1.6 inference on frames in memory (the upstream pipeline without its file / ffmpeg
I/O): re-draws the mouth region of every frame to match the audio, 16 frames at a time, at
512 x 512 on the aligned face, then pastes it back.

weights_dir holds latentsync_unet.pt, whisper/tiny.pt and sd-vae-ft-mse/ (config.json +
diffusion_pytorch_model.safetensors)."""
from __future__ import annotations

import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# configs/unet/stage2_512.yaml, model section (LatentSync 1.6)
UNET_CONFIG = {
    "act_fn": "silu", "add_audio_layer": True, "attention_head_dim": 8,
    "block_out_channels": [320, 640, 1280, 1280], "center_input_sample": False, "cross_attention_dim": 384,
    "down_block_types": ["CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D"],
    "mid_block_type": "UNetMidBlock3DCrossAttn",
    "up_block_types": ["UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"],
    "downsample_padding": 1, "flip_sin_to_cos": True, "freq_shift": 0, "in_channels": 13, "layers_per_block": 2,
    "mid_block_scale_factor": 1, "norm_eps": 1e-5, "norm_num_groups": 32, "out_channels": 4, "sample_size": 64,
    "resnet_time_scale_shift": "default", "use_motion_module": True, "motion_module_resolutions": [1, 2, 4, 8],
    "motion_module_mid_block": False, "motion_module_decoder_only": False, "motion_module_type": "Vanilla",
    "motion_module_kwargs": {"num_attention_heads": 8, "num_transformer_block": 1,
                             "attention_block_types": ["Temporal_Self", "Temporal_Self"],
                             "temporal_position_encoding": True, "temporal_position_encoding_max_len": 24,
                             "temporal_attention_dim_div": 1, "zero_initialize": True},
}
NUM_FRAMES = 16
RES = 512
AUDIO_FEAT_LENGTH = (2, 2)


def weight_paths(weights_dir: str) -> dict:
    return {"unet": os.path.join(weights_dir, "latentsync_unet.pt"),
            "whisper": os.path.join(weights_dir, "whisper", "tiny.pt"),
            "vae": os.path.join(weights_dir, "sd-vae-ft-mse")}


def available(weights_dir: str) -> bool:
    w = weight_paths(weights_dir)
    return os.path.exists(w["unet"]) and os.path.exists(w["whisper"]) and \
        os.path.exists(os.path.join(w["vae"], "diffusion_pytorch_model.safetensors"))


class LatentSync:
    def __init__(self, weights_dir: str, device="cuda"):
        import torch
        from diffusers import AutoencoderKL, DDIMScheduler
        from .models.unet import UNet3DConditionModel
        from .whisper.model import ModelDimensions, Whisper
        from .face import FaceDetector
        w = weight_paths(weights_dir)
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.scheduler = DDIMScheduler.from_pretrained(HERE)  # scheduler_config.json
        ck = torch.load(w["whisper"], map_location="cpu", weights_only=True)
        wm = Whisper(ModelDimensions(**ck["dims"]))
        wm.load_state_dict(ck["model_state_dict"])
        self.whisper = wm.to(self.device).eval()
        self.vae = AutoencoderKL.from_pretrained(w["vae"], torch_dtype=self.dtype).to(self.device).eval()
        self.vae.config.scaling_factor = 0.18215
        self.vae.config.shift_factor = 0
        unet, _ = UNet3DConditionModel.from_pretrained(dict(UNET_CONFIG), w["unet"], device="cpu")
        self.unet = unet.to(self.device, dtype=self.dtype).eval()
        self.face = FaceDetector(cuda=self.device.type == "cuda")
        self.mask = self._load_mask()

    def _load_mask(self):
        import cv2
        import torch
        m = cv2.imread(os.path.join(HERE, "mask.png"))
        m = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
        m = cv2.resize(m, (RES, RES), interpolation=cv2.INTER_LANCZOS4) / 255.0
        return torch.from_numpy(m).permute(2, 0, 1).float()  # c h w

    # ---------------------------------------------------------------- audio
    def audio_features(self, audio16k: np.ndarray):
        """Whisper-tiny encoder embeddings at 50 per second: (T, 5, 384)."""
        import torch
        from .whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
        mel = log_mel_spectrogram(torch.from_numpy(np.ascontiguousarray(audio16k, np.float32)))
        out = []
        n = mel.shape[-1]
        seek = 0
        with torch.no_grad():
            while seek < n:
                end = min(seek + 3000, n)
                seg = pad_or_trim(mel[:, seek:seek + 3000], N_FRAMES).to(self.device)[None]
                if self.dtype == torch.float16:
                    seg = seg.half()
                _, emb = self.whisper.encoder(seg, include_embeddings=True)
                emb = emb.transpose(0, 2, 1, 3).squeeze(0)  # (1500, layers, 384)
                out.append(emb[: (end - seek) // 2])
                seek += 3000
        return torch.from_numpy(np.concatenate(out, axis=0))

    def audio_chunks(self, feat, n_frames: int, fps: float = 25.0):
        import torch
        dim = feat.shape[-1]
        L = len(feat)
        chunks = []
        for i in range(n_frames):
            c = int(i * 50 / fps)
            idx = [min(L - 1, max(0, k)) for k in range(c - AUDIO_FEAT_LENGTH[0] * 2, c + (AUDIO_FEAT_LENGTH[1] + 1) * 2)]
            chunks.append(torch.cat([feat[k] for k in idx], 0).reshape(-1, dim))
        return chunks

    # ---------------------------------------------------------------- faces
    def align(self, frames: np.ndarray):
        """Aligned 512 crops (f c h w, uint8 values as float), restore boxes and affine matrices.
        Frames without a face borrow the nearest found face's landmarks."""
        import cv2
        import torch
        from .affine_transform import AlignRestore
        self.restorer = AlignRestore(resolution=RES, device=self.device, dtype=self.dtype)
        lm = []
        for f in frames:
            _, l106 = self.face(f)
            lm.append(None if l106 is None else np.round([
                np.mean(l106[[43, 48, 49, 51, 50]], axis=0), np.mean(l106[101:106], axis=0),
                np.mean(l106[[74, 77, 83, 86]], axis=0)]))
        found = [i for i, x in enumerate(lm) if x is not None]
        if not found:
            return None
        for i in range(len(lm)):
            if lm[i] is None:
                lm[i] = lm[min(found, key=lambda j: abs(j - i))]
        faces, boxes, mats = [], [], []
        for f, l3 in zip(frames, lm):
            face, M = self.restorer.align_warp_face(f.copy(), landmarks3=l3, smooth=True)
            boxes.append([0, 0, face.shape[1], face.shape[0]])
            face = cv2.resize(face, (RES, RES), interpolation=cv2.INTER_LANCZOS4)
            faces.append(torch.from_numpy(face).permute(2, 0, 1))
            mats.append(M)
        return torch.stack(faces), boxes, mats, len(found) / len(frames)

    # ---------------------------------------------------------------- the pass
    def __call__(self, frames: np.ndarray, audio16k: np.ndarray, *, fps: float = 25.0, steps: int = 20,
                 guidance: float = 1.5, seed: int = 1247, log=print):
        """frames: (N, H, W, 3) uint8 RGB at `fps`; audio16k: the matching audio, 16 kHz mono.
        Returns (frames, share of frames with a face found) or (None, 0) without a face."""
        import torch
        import torch.nn.functional as F
        import torchvision.transforms.functional as TF
        from einops import rearrange
        al = self.align(frames)
        if al is None:
            return None, 0.0
        faces, boxes, mats, found = al
        n = len(frames)
        chunks = self.audio_chunks(self.audio_features(audio16k), n, fps)
        g = torch.Generator(device=self.device).manual_seed(seed)
        lat_ch = self.vae.config.latent_channels
        vsf = 2 ** (len(self.vae.config.block_out_channels) - 1)
        all_lat = torch.randn((1, lat_ch, 1, RES // vsf, RES // vsf), generator=g, device=self.device,
                              dtype=self.dtype).repeat(1, 1, n, 1, 1)
        self.scheduler.set_timesteps(steps, device=self.device)
        all_lat = all_lat * self.scheduler.init_noise_sigma
        cfg = guidance > 1.0
        mask = self.mask.to(self.device)
        out = []
        with torch.no_grad():
            for i in range(math.ceil(n / NUM_FRAMES)):
                sl = slice(i * NUM_FRAMES, (i + 1) * NUM_FRAMES)
                ae = torch.stack(chunks[sl]).to(self.device, dtype=self.dtype)
                if cfg:
                    ae = torch.cat([torch.zeros_like(ae), ae])
                fc = faces[sl].to(self.device).float()
                pix = (fc / 255.0 - 0.5) / 0.5  # normalize to -1..1
                masked = pix * mask
                m1 = mask[0:1].expand(len(fc), 1, RES, RES)
                m_lat = F.interpolate(m1, size=(RES // vsf, RES // vsf)).to(self.dtype)
                mi_lat = self.vae.encode(masked.to(self.dtype)).latent_dist.sample(generator=g) * 0.18215
                ref_lat = self.vae.encode(pix.to(self.dtype)).latent_dist.sample(generator=g) * 0.18215
                m_lat = rearrange(m_lat, "f c h w -> 1 c f h w")
                mi_lat = rearrange(mi_lat, "f c h w -> 1 c f h w")
                ref_lat = rearrange(ref_lat, "f c h w -> 1 c f h w")
                if cfg:
                    m_lat, mi_lat, ref_lat = (torch.cat([x] * 2) for x in (m_lat, mi_lat, ref_lat))
                lat = all_lat[:, :, sl]
                for t in self.scheduler.timesteps:
                    x = torch.cat([lat] * 2) if cfg else lat
                    x = self.scheduler.scale_model_input(x, t)
                    x = torch.cat([x, m_lat, mi_lat, ref_lat], dim=1)
                    eps = self.unet(x, t, encoder_hidden_states=ae).sample
                    if cfg:
                        eu, ea = eps.chunk(2)
                        eps = eu + guidance * (ea - eu)
                    lat = self.scheduler.step(eps, t, lat, eta=0.0, generator=g).prev_sample
                dec = self.vae.decode(rearrange(lat / 0.18215, "b c f h w -> (b f) c h w")).sample
                # keep the original pixels outside the mouth (the mask is 1 where the face is kept)
                keep = mask[0:1].to(dec.dtype)
                dec = dec * (1 - keep) + pix.to(dec.dtype) * keep
                out.append(dec)
        dec = torch.cat(out)  # f c h w in -1..1
        res = []
        for k in range(n):
            x1, y1, x2, y2 = boxes[k]
            face = TF.resize(dec[k], size=[int(y2 - y1), int(x2 - x1)],
                             interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
            res.append(self.restorer.restore_img(frames[k], face, mats[k]))
        return np.stack(res), found
