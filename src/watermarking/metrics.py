"""Quality metrics (PSNR / SSIM) for marked images.

Migrated from validation_wam_inner_ecc_20261008/run_sprint.py quality().
"""
from __future__ import annotations

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .attacks import tensor_rgb


def quality(original, marked) -> tuple[float, float]:
    a, b = tensor_rgb(original), tensor_rgb(marked)
    return (float(peak_signal_noise_ratio(a, b, data_range=1.0)),
            float(structural_similarity(a, b, channel_axis=2, data_range=1.0)))
