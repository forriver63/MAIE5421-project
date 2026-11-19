"""Check the first official image against the native APIs and serialized artifacts."""
import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

import run
from watermarking.bench.attack_registry import ATTACKS, apply_attack, plan_attack
from watermarking.bench.cache import file_hash, fingerprint, read_json, write_json
from watermarking.bench.manifest import payload_for
from watermarking.bench.scoring import quality, score_payload


def check(ok, context):
    if not ok:
        raise ValueError(f"Smoke verification failed: {context}")


def table(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Completed one-image smoke output")
    parser.add_argument("--cache", required=True, type=Path, help="Corresponding local PNG/key cache")
    args = parser.parse_args()
    rec, = run.manifest_records(args.manifest, 1)
    image_id = rec["image_id"]
    for k in ("source", "processed"):
        check(file_hash(rec[k]) == rec[k + "_sha256"], k)
    cover = run.rgb(rec["processed"])
    report = {}
    for method in ("svd", "dwt_dct"):
        out, cache = args.output / method, args.cache / method / image_id
        module, config = run.load_method(method)
        name = "svd.rgb_adapter" if method == "svd" else "dwt_dct.watermark"
        check(module.__name__ == "baselines." + name, "isolated module")
        rows = table(out / "cases.csv")
        positives = {(int(r["payload_seed"]), r["condition"]): r for r in rows}
        check(len(positives) == len(rows) == 3 * len(ATTACKS), "complete unique smoke cases")
        plans = read_json(out / "attack_plans.json.gz")
        plan_map = {(p["payload_seed"], p["attack"]["attack_id"]): p for p in plans}
        check(len(plan_map) == len(plans) == 3 * len(ATTACKS) + 1, "attack plan scope")
        for (seed, condition), planned in plan_map.items():
            check(planned["attack"] == plan_attack(condition, image_id, seed)
                  and planned["image_id"] == image_id and planned["negative"] == (seed == -1),
                  "shared attack plan")
        for seed in (0, 1, 2):
            expected = payload_for(image_id, seed)
            bits = [(expected >> i) & 1 for i in range(31, -1, -1)]
            native, key, _ = module.embed(cover, bits, config)
            png = cache / f"s{seed}.png"
            meta = read_json(png.with_suffix(".json"))
            with Image.open(png) as im:
                check(im.format == "PNG" and im.mode == "RGB", png)
                marked = np.array(im)
            check(np.array_equal(marked, np.clip(np.rint(native), 0, 255).astype(np.uint8)), png)
            check(meta["settings"] == key and meta["key_sha256"] == fingerprint(key), "restored key")
            if seed == 0:
                control_key = key
            q = quality(cover, marked)
            check(meta["png_sha256"] == file_hash(png) and all(meta[k] == v for k, v in q.items()),
                  "PNG integrity and quality")
            for condition in ATTACKS:
                row = positives[seed, condition]
                pixels = apply_attack(marked, plan_map[seed, condition]["attack"])
                check(row["attacked_sha256"] == hashlib.sha256(pixels.tobytes()).hexdigest(),
                      f"attacked pixels {seed}/{condition}")
                try:
                    decoded, _ = module.extract(pixels, meta["settings"])
                    received = int("".join(str(int(x)) for x in decoded), 2)
                except module.SynchronizationError:
                    check(pixels.shape != marked.shape and row["reason"] == "unsupported_geometry",
                          "geometry erasure")
                    received = None
                scores = score_payload(dict(accepted=received is not None, payload=received), expected)
                check(row["image_id"] == image_id and row["payload"] == f"{expected:032b}"
                      and row["recovered"] == ("" if received is None else f"{received:032b}")
                      and row["outcome"] == scores["outcome"]
                      and (float(row["ber"]) if row["ber"] else None) == scores["payload_ber"],
                      f"score {seed}/{condition}")
        neg, = table(out / "negatives.csv")
        control_bits, _ = module.extract(cover, control_key)
        check(neg["accepted"] == "" and neg["key_sha256"] == fingerprint(control_key)
              and neg["recovered"] == "".join(str(int(x)) for x in control_bits), "clean control")
        report[method] = dict(module=module.__name__, embeddings=3, positive_cases=len(rows),
                              negative_controls=1, status="passed")
    write_json(args.output / "smoke_verification.json", dict(image_id=image_id, methods=report))
    print(f"Smoke verification passed for {image_id}, both methods")


if __name__ == "__main__":
    main()
