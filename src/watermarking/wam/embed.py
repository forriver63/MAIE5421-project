"""Localized multi-message embedding using upstream WAM.

Migrated from validation_wam_20261007/run_validation.py embed_messages().
"""
from __future__ import annotations

import time

import torch


def embed_messages(wam, image: torch.Tensor, messages: torch.Tensor, masks: torch.Tensor):
    watermarked = image.clone()
    start = time.perf_counter()
    with torch.inference_mode():
        for index in range(len(messages)):
            output = wam.embed(image, messages[index].unsqueeze(0).to(image.device))
            watermarked = output["imgs_w"] * masks[index] + watermarked * (1 - masks[index])
    return watermarked, time.perf_counter() - start
