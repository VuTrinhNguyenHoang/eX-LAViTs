from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import torchaudio
    _HAS_TORCHAUDIO = True
except Exception:
    _HAS_TORCHAUDIO = False

@dataclass
class Speech2ImageConfig:
    sample_rate: int = 16000

    # window length: 2.24s -> 224 frames ~ 10ms hop
    window_sec: float = 2.24

    # STFT / Mel (classic)
    n_fft: int = 512
    win_length: int = 400    # 25ms @16k
    hop_length: int = 160    # 10ms @16k
    n_mels: int = 128
    fmin: float = 0.0
    fmax: float = 8000.0

    # output image size
    out_hw: int = 224

    # normalization
    eps: float = 1e-6


def pad_or_crop_1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    x: [T] or [1,T]
    returns same rank, padded/cropped to target_len on last dim.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)  # [1,T]
    T = x.size(-1)
    if T == target_len:
        return x
    if T > target_len:
        return x[..., :target_len]
    # pad at end
    pad = target_len - T
    return F.pad(x, (0, pad), mode="constant", value=0.0)


def pad_or_crop_2d(spec: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """
    spec: [F, T]
    pad/crop to [H,W] = target_hw, without interpolation.
    - freq axis padded symmetrically (center) to avoid shifting.
    - time axis padded at end (right).
    """
    H, W = target_hw
    Freq, Time = spec.shape

    # time
    if Time > W:
        spec = spec[:, :W]
    elif Time < W:
        spec = F.pad(spec, (0, W - Time), mode="constant", value=0.0)

    # freq
    if Freq > H:
        spec = spec[:H, :]
    elif Freq < H:
        pad_total = H - Freq
        pad_top = pad_total // 2
        pad_bot = pad_total - pad_top
        spec = F.pad(spec, (0, 0, pad_top, pad_bot), mode="constant", value=0.0)

    return spec  # [H,W]


def minmax_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mn = x.amin()
    mx = x.amax()
    return (x - mn) / (mx - mn + eps)


class Speech2Image:
    """
    Convert waveform -> [3,224,224] "image" tensor for ViT.
    Preferred backend: torchaudio. If torchaudio is missing, we raise a clear error.
    """
    def __init__(self, cfg: Speech2ImageConfig = Speech2ImageConfig(), device: Optional[torch.device] = None):
        self.cfg = cfg
        self.device = device

        if not _HAS_TORCHAUDIO:
            raise RuntimeError(
                "torchaudio is not available. Install torchaudio or implement a torch-only mel frontend."
            )

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            f_min=cfg.fmin,
            f_max=cfg.fmax,
            n_mels=cfg.n_mels,
            power=2.0,
            center=False,
            normalized=False,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

        if self.device is not None:
            self.mel = self.mel.to(self.device)
            self.db = self.db.to(self.device)

        self.target_wav_len = int(round(cfg.window_sec * cfg.sample_rate))
        self.out_hw = cfg.out_hw

    @torch.no_grad()
    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: [T] or [1,T] (mono). Returns image: [3,224,224] float32 in [0,1].
        """
        if self.device is not None:
            wav = wav.to(self.device)

        wav = pad_or_crop_1d(wav, self.target_wav_len)  # [1,T]

        spec = self.mel(wav)         # [1, n_mels, frames]
        spec = self.db(spec)         # log scale (dB)
        spec = spec.squeeze(0)       # [n_mels, frames]

        # normalize to [0,1]
        spec = minmax_norm(spec, eps=self.cfg.eps)

        # pad/crop to 224x224 without interpolation
        spec = pad_or_crop_2d(spec, (self.out_hw, self.out_hw))  # [224,224]

        # 1ch -> 3ch
        img = spec.unsqueeze(0).repeat(3, 1, 1).contiguous()     # [3,224,224]
        return img.float().cpu()