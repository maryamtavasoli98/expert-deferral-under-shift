"""MedMNIST loading and the controlled distribution shifts (written from scratch, torch only).

Images are float tensors in [0, 1] with shape (N, 3, 28, 28). Five corruption families with five
severities each emulate the usual deployment shifts (sensor noise, defocus, exposure, contrast,
resolution loss); the parameters are fixed in ``PARAMS`` so the shift is reproducible.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .common import RAW

PARAMS = {
    "gaussian_noise": [0.04, 0.08, 0.12, 0.18, 0.26],  # std of additive noise
    "gaussian_blur": [0.5, 0.8, 1.1, 1.5, 2.0],  # sigma in pixels
    "brightness": [0.1, 0.2, 0.3, 0.4, 0.5],  # additive offset
    "contrast": [0.75, 0.55, 0.4, 0.3, 0.2],  # factor around the per-image mean
    "pixelate": [24, 20, 16, 12, 8],  # side length after downsampling
}


def load_npz(name: str) -> dict[str, np.ndarray]:
    z = np.load(RAW / f"{name}.npz")
    return {k: z[k] for k in z.files}


def to_tensor(x: np.ndarray) -> torch.Tensor:
    """uint8 (N,28,28[,3]) -> float (N,3,28,28) in [0,1]; grayscale is replicated to 3 channels."""
    t = torch.from_numpy(x).float() / 255.0
    if t.ndim == 3:
        t = t.unsqueeze(-1).repeat(1, 1, 1, 3)
    return t.permute(0, 3, 1, 2).contiguous()


def gaussian_kernel(sigma: float, size: int = 7) -> torch.Tensor:
    ax = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    k1 = torch.exp(-(ax**2) / (2 * sigma**2))
    k1 = k1 / k1.sum()
    return (k1[:, None] * k1[None, :])[None, None].repeat(3, 1, 1, 1)


def corrupt(x: torch.Tensor, kind: str, severity: int, seed: int = 0) -> torch.Tensor:
    """Apply one corruption at severity 1..5 and clip to [0, 1]."""
    p = PARAMS[kind][severity - 1]
    if kind == "gaussian_noise":
        g = torch.Generator().manual_seed(seed)
        out = x + p * torch.randn(x.shape, generator=g)
    elif kind == "gaussian_blur":
        out = F.conv2d(F.pad(x, (3, 3, 3, 3), mode="reflect"), gaussian_kernel(p), groups=3)
    elif kind == "brightness":
        out = x + p
    elif kind == "contrast":
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        out = (x - m) * p + m
    elif kind == "pixelate":
        out = F.interpolate(F.interpolate(x, size=(int(p), int(p)), mode="area"), size=x.shape[-2:], mode="nearest")
    else:
        raise ValueError(kind)
    return out.clamp(0, 1)


def pooled_shift(x: torch.Tensor, kinds: list[str], severity: int, seed: int = 0) -> torch.Tensor:
    """Every image gets one randomly chosen corruption family at the given severity (a mixed shifted stream)."""
    g = np.random.default_rng(seed)
    pick = g.integers(0, len(kinds), size=len(x))
    out = x.clone()
    for i, k in enumerate(kinds):
        idx = np.where(pick == i)[0]
        if len(idx):
            out[idx] = corrupt(x[idx], k, severity, seed + i)
    return out


def noise_images(n: int, seed: int = 0) -> torch.Tensor:
    """Far-OOD control: uniform-noise images."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, 28, 28, generator=g)
