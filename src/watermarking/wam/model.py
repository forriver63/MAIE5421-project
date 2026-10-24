"""Model loading helpers wrapping upstream WAM."""
from __future__ import annotations

import os

import torch

from . import path

path.ensure_on_syspath()

from notebooks.inference_utils import (  # noqa: E402,F401
    load_model_from_checkpoint,
)

DEVICE = torch.device("cpu")

EXPECTED_COMMIT = "2c08af04d037d5667c02f6ddebbda9ff04581c3e"
EXPECTED_CHECKPOINT = "90ef232384e023bd63245eb0c131abd69d2afc7b8f17a71ccedceb542bf009e2"


def load_default_model(device: torch.device = DEVICE):
    """Load the official wam_mit checkpoint pinned in the project README.

    The upstream loader reads ``configs/*.yaml`` relative to the WAM checkout,
    so we temporarily chdir there and restore afterwards.
    """
    root = path.locate()
    cwd = os.getcwd()
    os.chdir(str(root))
    try:
        model = load_model_from_checkpoint(
            str(root / "checkpoints" / "params.json"),
            str(root / "checkpoints" / "wam_mit.pth"),
        )
    finally:
        os.chdir(cwd)
    return model.to(device).eval()
