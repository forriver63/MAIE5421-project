"""Photometric and geometric attacks for robustness evaluation.

Migrated from validation_wam_20261007/run_validation.py (jpeg_attack /
noise_attack / rotate_attack / tensor_rgb / tensor_from_rgb).
"""
from __future__ import annotations

import io
import math

import numpy as np
import torch
from PIL import Image

from .wam.transforms import default_transform, unnormalize_img

IMAGE_SIZE = 512
CROP_SEED = 20261008  # matches the frozen historical sprint seed


def crop_dimensions(area: float, ratio: float) -> tuple[int, int]:
    width = min(IMAGE_SIZE, max(32, round(math.sqrt(area * IMAGE_SIZE ** 2 * ratio))))
    height = min(IMAGE_SIZE, max(32, round(area * IMAGE_SIZE ** 2 / width)))
    return width, height


def fixed_crops(image_index: int, payload_seed: int):
    """Deterministic fixed arbitrary-offset crops (evaluation protocol)."""
    for retained in (0.75, 0.50):
        for variant, ratio in enumerate((1.0, 4 / 3, 3 / 4)):
            width, height = crop_dimensions(retained, ratio)
            rng = np.random.default_rng(CROP_SEED + 50000 + image_index * 1009
                                        + payload_seed * 97 + variant * 13 + int(retained * 100))
            x0 = int(rng.integers(0, IMAGE_SIZE - width + 1))
            y0 = int(rng.integers(0, IMAGE_SIZE - height + 1))
            yield retained, variant, ratio, x0, y0, width, height


def tensor_rgb(tensor: torch.Tensor) -> np.ndarray:
    array = unnormalize_img(tensor.detach()).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return array


def tensor_from_rgb(array: np.ndarray) -> torch.Tensor:
    image = Image.fromarray(np.clip(np.rint(array * 255), 0, 255).astype(np.uint8), mode="RGB")
    return default_transform(image).unsqueeze(0)


def jpeg_attack(tensor: torch.Tensor, quality: int) -> torch.Tensor:
    image = Image.fromarray(np.clip(np.rint(tensor_rgb(tensor) * 255), 0, 255).astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=2)
    buffer.seek(0)
    return default_transform(Image.open(buffer).convert("RGB")).unsqueeze(0)


def noise_attack(tensor: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    rgb = tensor_rgb(tensor)
    return tensor_from_rgb(np.clip(rgb + rng.normal(0, sigma / 255, rgb.shape), 0, 1))


def rotate_attack(tensor: torch.Tensor, angle: float) -> torch.Tensor:
    image = Image.fromarray(np.clip(np.rint(tensor_rgb(tensor) * 255), 0, 255).astype(np.uint8), mode="RGB")
    rotated = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    return default_transform(rotated).unsqueeze(0)
