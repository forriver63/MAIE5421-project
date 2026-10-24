"""Fixed non-overlapping region masks for localized WAM embedding.

Migrated from validation_wam_20261007/run_validation.py.
"""
from __future__ import annotations

import torch

IMAGE_SIZE = 512

# Eight fixed regions as (x0, y0, x1, y1) on a 512x512 canvas.
REGIONS = [
    (6, 35, 116, 240), (132, 35, 242, 240),
    (258, 35, 368, 240), (384, 35, 494, 240),
    (6, 272, 116, 477), (132, 272, 242, 477),
    (258, 272, 368, 477), (384, 272, 494, 477),
]
# Which region indices to use for each region count.
REGION_SELECTIONS = {2: [0, 7], 4: [0, 3, 4, 7], 8: list(range(8))}


def make_masks(count: int, device: torch.device) -> torch.Tensor:
    masks = torch.zeros((count, 1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32, device=device)
    for local_index, region_index in enumerate(REGION_SELECTIONS[count]):
        x0, y0, x1, y1 = REGIONS[region_index]
        masks[local_index, :, y0:y1, x0:x1] = 1
    return masks
