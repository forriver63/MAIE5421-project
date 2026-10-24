"""Blind multi-message detection using upstream WAM + DBSCAN clustering.

Migrated from validation_wam_inner_ecc_20261008/run_sprint.py detect_blind().
Returns one cluster per detected watermark with its raw 32-bit word, pixel mask,
and confidence. Packet-level (BCH/CRC) decoding is left to the caller.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from . import path

path.ensure_on_syspath()

from notebooks.inference_utils import multiwm_dbscan  # noqa: E402

from ..codec.bch import bits_to_int


def detect_blind(wam, image: torch.Tensor, epsilon: int = 1, min_samples: int = 500) -> dict:
    start = time.perf_counter()
    with torch.inference_mode():
        preds = wam.detect(image)["preds"]
    mask_probability = torch.sigmoid(preds[:, 0]).detach().cpu()
    bit_logits = preds[:, 1:].detach().cpu()
    centroids = {}
    labels = torch.full(mask_probability.shape[-2:], -1.0)
    error = None
    if bool((mask_probability > 0.5).any()):
        try:
            centroids, labels = multiwm_dbscan(
                bit_logits, mask_probability, epsilon=epsilon, min_samples=min_samples)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    clusters = []
    for label, bits in sorted(centroids.items(), key=lambda pair: int(pair[0])):
        binary = bits.detach().cpu().numpy().astype(np.uint8)
        word = bits_to_int(binary)
        pixel_mask = labels == float(label)
        bit_prob = torch.sigmoid(bit_logits[0])
        confidence = float((2 * (bit_prob[:, pixel_mask] - 0.5).abs()).mean()) if pixel_mask.any() else 0.0
        clusters.append({
            "label": int(label), "raw_word_hex": f"{word:08x}", "raw_word": word,
            "mask": pixel_mask, "confidence": confidence,
        })
    return {"clusters": clusters, "labels": labels, "mask_probability": mask_probability,
            "runtime_seconds": time.perf_counter() - start, "clustering_error": error}
