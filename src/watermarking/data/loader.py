"""Image dataset loading and manifest management.

Migrated from validation_wam_20261007/run_validation.py (load_images) and
validation_wam_inner_ecc_20261008/run_sprint.py (load_image).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image, ImageOps

from ..io import sha256
from ..wam.transforms import default_transform

IMAGE_SIZE = 512

IMAGE_FILES = {
    "astronaut": "astronaut.png",
    "coffee": "coffee.png",
    "chelsea": "chelsea.png",
    "rocket": "rocket.jpg",
    "hubble": "hubble_deep_field.jpg",
    "immunohistochemistry": "ihc.png",
    "retina": "retina.jpg",
    "motorcycle": "motorcycle_left.png",
    "camera": "camera.png",
    "coins": "coins.png",
}


def load_images(output_dir) -> list[dict]:
    """Prepare the 10 public scikit-image inputs and write a SHA-256 manifest.

    Prefer bundled files; the pinned scikit-image fetcher also resolves its cache
    and verifies downloaded source hashes when samples are not bundled.
    """
    import skimage.data

    data_dir = Path(skimage.data.__file__).parent
    output_dir = Path(output_dir)
    inputs = output_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    records = []
    for name, filename in IMAGE_FILES.items():
        source = data_dir / filename
        if not source.is_file():
            from skimage.data._fetchers import _fetch
            source = Path(_fetch(f"data/{filename}"))
        image = Image.open(source).convert("RGB")
        image = ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.LANCZOS)
        output = inputs / f"{name}.png"
        image.save(output)
        records.append({
            "name": name,
            "source": str(source),
            "source_sha256": sha256(source),
            "processed": str(output),
            "processed_sha256": sha256(output),
            "shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
            "provenance": "scikit-image 0.24.0 bundled public sample data",
        })
    (output_dir / "image_manifest.json").write_text(json.dumps(records, indent=2))
    return records


def load_image(record: dict, device: torch.device | None = None) -> torch.Tensor:
    tensor = default_transform(Image.open(record["processed"]).convert("RGB")).unsqueeze(0)
    return tensor.to(device) if device is not None else tensor
