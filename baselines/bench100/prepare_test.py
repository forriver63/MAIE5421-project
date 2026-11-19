"""Prepare only the pinned DIV2K validation images for the traditional baselines."""
import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from watermarking.bench.cache import file_hash, read_json, write_json


def prepare_test(archive, output):
    reference = ROOT / "baseline/image_manifest.json"
    records = [r for r in read_json(reference) if r["split"] == "test"]
    if [r["image_id"] for r in records] != [f"DIV2K_{i:04d}" for i in range(801, 901)]:
        raise ValueError("Official test IDs differ from the expected 0801–0900 range")

    with zipfile.ZipFile(archive) as z:
        for rec in records:
            image_id = rec["image_id"]
            source = output / rec["source"]
            processed = output / rec["processed"]
            member = f"DIV2K_valid_HR/{image_id[-4:]}.png"
            if source.exists():
                source_sha256 = file_hash(source)
            else:
                data = z.read(member)
                source_sha256 = hashlib.sha256(data).hexdigest()
                if source_sha256 == rec["source_sha256"]:
                    source.parent.mkdir(parents=True, exist_ok=True)
                    tmp = source.with_suffix(".tmp")
                    tmp.write_bytes(data)
                    tmp.replace(source)
            if source_sha256 != rec["source_sha256"]:
                raise ValueError(f"{image_id}: source SHA-256 mismatch")

            if processed.exists():
                processed_sha256 = file_hash(processed)
            else:
                processed.parent.mkdir(parents=True, exist_ok=True)
                tmp = processed.with_suffix(".tmp")
                # Same operations and default PNG encoder as bench.manifest.prepare.
                with Image.open(source) as im:
                    im = ImageOps.fit(ImageOps.exif_transpose(im).convert("RGB"), (512, 512),
                                      method=Image.Resampling.LANCZOS)
                    im.save(tmp, format="PNG")
                processed_sha256 = file_hash(tmp)
                if processed_sha256 == rec["processed_sha256"]:
                    tmp.replace(processed)
            if processed_sha256 != rec["processed_sha256"]:
                raise ValueError(f"{image_id}: processed PNG SHA-256 mismatch: {processed_sha256}")

    write_json(output / "images.json", records)
    write_json(output / "preparation.json", {
        "source_url": records[0]["source_url"],
        "archive_sha256": file_hash(archive),
        "reference_manifest_sha256": file_hash(reference),
        "test_images": len(records),
        "verified_source_sha256": len(records),
        "verified_processed_png_sha256": len(records),
        "calibration_images_used": 0,
        "pillow": Image.__version__,
        "zlib": Image.core.zlib_version,
    })
    print(f"Verified {len(records)} original and processed test image pairs: {output / 'images.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="Official DIV2K_valid_HR.zip")
    parser.add_argument("--output", required=True, type=Path, help="Local dataset directory")
    args = parser.parse_args()
    prepare_test(args.archive, args.output)
