"""Locate the upstream watermark-anything checkout and expose it on sys.path."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Default checkout location (the historical sprint kept it under validation_wam_20261007).
_DEFAULT = Path(__file__).resolve().parents[3] / "validation_wam_20261007" / "watermark-anything"


def locate() -> Path:
    env = os.environ.get("WAM_PATH")
    path = (Path(env).expanduser() if env else _DEFAULT).resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"watermark-anything checkout not found at {path}; set WAM_PATH to its location")
    return path


def ensure_on_syspath() -> Path:
    path = locate()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path
