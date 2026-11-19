"""Run the original RGB team32 implementations on the shared DIV2K test protocol."""
import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import os
import platform
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from watermarking.bench.attack_registry import ATTACKS, CORE, EXTENDED, apply_attack, plan_attack
from watermarking.bench.cache import file_hash, fingerprint, read_json, write_json
from watermarking.bench.manifest import load_manifest, payload_for
from watermarking.bench.scoring import quality, score_payload


def load_method(name):
    if name == "svd":
        # The legacy RGB adapter imports three helpers by the top-level name
        # "watermark". Bind that name only while the module is first loaded.
        core = importlib.import_module("baselines.svd.watermark")
        with patch.dict(sys.modules, {"watermark": core}):
            module = importlib.import_module("baselines.svd.rgb_adapter")
        if module.to_blocks is not core.to_blocks:
            raise ImportError("SVD adapter bound to an unexpected watermark module")
    else:
        module = importlib.import_module("baselines.dwt_dct.watermark")
    settings = read_json(ROOT / "baselines" / name / "configs/team32.json")
    config = module.Config(**settings)
    config.validate()
    return module, config


def case_rows(records, seeds):
    for rec in records:
        image_id = rec["image_id"]
        for seed in seeds:
            for attack in ATTACKS:
                yield dict(image_id=image_id, payload_seed=seed, negative=False,
                           attack=plan_attack(attack, image_id, seed))
        yield dict(image_id=image_id, payload_seed=-1, negative=True,
                   attack=plan_attack("clean", image_id, -1))


def manifest_records(path, limit):
    official = load_manifest(ROOT / "baseline/image_manifest.json", "test")
    records = load_manifest(path, "test")
    by_id = {r["image_id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate test image IDs")
    if [r["image_id"] for r in official] != [f"DIV2K_{i:04d}" for i in range(801, 901)]:
        raise ValueError("Official test scope changed")
    if limit is None and set(by_id) != {r["image_id"] for r in official}:
        raise ValueError("Test manifest differs from official test IDs")
    selected = []
    for ref in official[:limit]:
        rec = by_id[ref["image_id"]]
        for key in ("source_sha256", "processed_sha256", "shape"):
            if rec[key] != ref[key]:
                raise ValueError(f"{rec['image_id']} differs from official manifest: {key}")
        selected.append(rec)
    return selected


def provenance(method, protocol, records):
    bench = ROOT / "src/watermarking/bench"
    baseline = ROOT / "baselines" / method
    files = [Path(__file__), baseline / "watermark.py", baseline / "configs/team32.json",
             ROOT / "baseline/image_manifest.json", ROOT / "configs/protocol.example.json",
             *(bench / f"{name}.py" for name in ("cache", "attack_registry", "manifest", "scoring"))]
    if method == "svd":
        files.append(baseline / "rgb_adapter.py")
    hashes = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in files}
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "pillow", "scipy", "scikit-image")}
    env = dict(python=platform.python_version(), packages=versions, platform=platform.platform(),
               machine=platform.machine(), zlib=Image.core.zlib_version,
               jpeg=Image.core.jpeglib_version)
    params = dict(method=method, config=read_json(baseline / "configs/team32.json"),
                  protocol=protocol, conditions=list(ATTACKS), environment=env, files=hashes,
                  images=[{k: r[k] for k in ("image_id", "split", "source_sha256", "processed_sha256")}
                          for r in records])
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return dict(identity=fingerprint(params), checkout_commit=commit, **params)


def guard(directory, prov):
    path = directory / "provenance.json"
    if path.exists():
        if read_json(path)["identity"] != prov["identity"]:
            raise ValueError(f"Changed configuration, code, scope or environment. Choose a fresh {directory}")
    else:
        write_json(path, prov)


def rgb(path):
    with Image.open(path) as im:
        if im.mode != "RGB" or im.size != (512, 512):
            raise ValueError(f"Unexpected processed image mode/size in {path}")
        return np.array(im)


def bits_of(payload):
    return np.frombuffer(payload.to_bytes(4, "big"), dtype=np.uint8).copy()


def recover(module, attacked, settings):
    try:
        bits, _ = module.extract(attacked, settings)
    except module.SynchronizationError:
        if list(attacked.shape) == settings["image_shape"]:
            raise
        return None, "unsupported_geometry"
    if bits.shape != (32,) or not np.isin(bits, [0, 1]).all():
        raise ValueError("Extractor returned a malformed 32-bit message")
    return int.from_bytes(np.packbits(bits, bitorder="big").tobytes(), "big"), ""


def get_embedding(directory, module, config, cover, image_id, seed, payload, identity):
    stem = directory / f"s{seed}"
    png = stem.with_suffix(".png")
    meta = stem.with_suffix(".json")
    token = fingerprint([identity, image_id, seed, payload])
    if not meta.exists():
        bits = np.unpackbits(bits_of(payload), bitorder="big")
        marked, settings, _ = module.embed(cover, bits, config)
        tmp = png.with_suffix(".tmp")
        Image.fromarray(module.to_uint8(marked)).save(tmp, format="PNG")
        os.replace(tmp, png)
        write_json(meta, dict(token=token, png_sha256=file_hash(png), settings=settings,
                             key_sha256=fingerprint(settings), **quality(cover, rgb(png))))
    data = read_json(meta)
    if (data["token"] != token or data["png_sha256"] != file_hash(png)
            or data["key_sha256"] != fingerprint(data["settings"])):
        raise ValueError(f"Embedding or key does not match its cache record: {meta}")
    return rgb(png), data


def process_image(rec, method, module, config, seeds, cache, identity):
    image_id = rec["image_id"]
    directory = cache / image_id
    directory.mkdir(parents=True, exist_ok=True)
    error = directory / "execution.error.json"
    case = dict(image_id=image_id, stage="verify_cover")
    try:
        for key in ("source", "processed"):
            if file_hash(rec[key]) != rec[key + "_sha256"]:
                raise ValueError(f"{image_id}: {key} file hash mismatch")
        cover = rgb(rec["processed"])
        positives, embeddings, keys = [], [], []
        for seed in seeds:
            case = dict(image_id=image_id, payload_seed=seed, stage="embedding")
            payload = payload_for(image_id, seed)
            marked, meta = get_embedding(directory, module, config, cover, image_id, seed, payload, identity)
            settings = meta["settings"]
            base = dict(method=method, image_id=image_id, payload_seed=seed)
            embeddings.append(dict(**base, psnr=meta["psnr"], ssim=meta["ssim"],
                                   png_sha256=meta["png_sha256"], key_sha256=meta["key_sha256"]))
            keys.append(dict(**base, settings=settings, key_sha256=meta["key_sha256"]))
            if seed == 0:
                control_key = settings
            shard = directory / f"s{seed}.cases.json.gz"
            token = fingerprint([identity, image_id, seed, meta["png_sha256"], meta["key_sha256"]])
            if shard.exists():
                saved = read_json(shard)
                if saved["token"] != token or saved["sha256"] != fingerprint(saved["rows"]):
                    raise ValueError(f"Incompatible or damaged case cache: {shard}")
                positives.extend(saved["rows"])
                continue
            rows = []
            for condition in ATTACKS:
                case = dict(**base, condition=condition, stage="attack/extract/score")
                attacked = apply_attack(marked, plan_attack(condition, image_id, seed))
                received, reason = recover(module, attacked, settings)
                score = score_payload(dict(accepted=received is not None, payload=received), payload)
                rows.append(dict(**base, condition=condition, payload=f"{payload:032b}",
                                 recovered="" if received is None else f"{received:032b}",
                                 outcome=score["outcome"], ber=score["payload_ber"], reason=reason,
                                 attacked_sha256=hashlib.sha256(attacked.tobytes()).hexdigest(),
                                 height=attacked.shape[0], width=attacked.shape[1]))
            write_json(shard, dict(token=token, rows=rows, sha256=fingerprint(rows)))
            positives.extend(rows)
        case = dict(image_id=image_id, stage="clean_negative")
        received, reason = recover(module, apply_attack(cover, plan_attack("clean", image_id, -1)), control_key)
        negative = dict(method=method, image_id=image_id, payload_seed=-1, condition="clean",
                        recovered="" if received is None else f"{received:032b}", accepted=None,
                        reason=reason, key_sha256=fingerprint(control_key),
                        key_source="same_cover_seed0_positions" if method == "svd" else "fixed_team32_settings")
        error.unlink(missing_ok=True)
        return positives, embeddings, negative, keys
    except Exception:
        write_json(error, dict(case=case, traceback=traceback.format_exc()))
        raise


def write_csv(path, rows):
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Local source and processed paths")
    parser.add_argument("--output", type=Path, required=True, help="Portable tables and provenance")
    parser.add_argument("--cache", type=Path, required=True, help="Local embeddings, keys and checkpoints")
    parser.add_argument("--method", choices=("svd", "dwt_dct"), action="append")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="First N test images for smoke testing only")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and not 1 <= args.limit <= 100):
        parser.error("workers must be positive; limit must be 1 through 100")
    records = manifest_records(args.manifest, args.limit)
    protocol = read_json(ROOT / "configs/protocol.example.json")
    seeds = protocol["payload_seeds"]
    if (seeds != [0, 1, 2] or not protocol["png_roundtrip"] or len(CORE) != protocol["core_attack_instances"]
            or len(EXTENDED) != protocol["extended_attack_instances"]):
        raise ValueError("Shared protocol changed; review scope before running")
    for method in args.method or ("svd", "dwt_dct"):
        module, config = load_method(method)
        cache, out = args.cache / method, args.output / method
        prov = provenance(method, protocol, records)
        guard(cache, prov)
        guard(out, prov)
        write_json(out / "attack_plans.json.gz", list(case_rows(records, seeds)))
        cases, embeddings, negatives, keys = [], [], [], []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(process_image, r, method, module, config, seeds, cache, prov["identity"])
                    for r in records]
            for index, job in enumerate(jobs, 1):
                pos, emb, neg, saved_keys = job.result()
                cases.extend(pos)
                embeddings.extend(emb)
                negatives.append(neg)
                keys.extend(saved_keys)
                if index == 1 or index % 5 == 0 or index == len(records):
                    print(f"{method}: {index}/{len(records)} images", flush=True)
        write_csv(out / "cases.csv", cases)
        write_csv(out / "embeddings.csv", embeddings)
        write_csv(out / "negatives.csv", negatives)
        write_json(out / "extraction_settings.json.gz", keys)


if __name__ == "__main__":
    main()
