"""Thin wrapper around the upstream ``watermark-anything`` model.

The upstream WAM source and checkpoint are external assets (pinned commit and
SHA-256). This subpackage locates them via the ``WAM_PATH`` environment variable
(or the default path) and exposes load / embed / detect helpers.
"""

from . import detect, embed, model

__all__ = ["model", "embed", "detect"]
