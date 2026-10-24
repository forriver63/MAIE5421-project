"""Final robustness evaluation (refactored from validation_wam_final_robustness_20261008/run_robustness.py).

Frozen method: localized official WAM (8 regions) + BCH(31,16)+parity + CRC-5 +
spatial repetition [A,B,C,D,D,C,B,A]. Produces cases.csv / packets.csv /
negative_control.csv / search_gate.json under experiments/robustness/results/.
"""
from __future__ import annotations

import json
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from watermarking import attacks, io, masks as mask_mod, metrics
from watermarking import eval as ev
from watermarking.codec import outer, packet
from watermarking.data import loader
from watermarking.wam import embed as wam_embed
from watermarking.wam import model as wam_model

ROOT = Path(__file__).resolve().parents[1]  # experiments/
OUT = ROOT / "robustness" / "results"
FIGURES = ROOT / "robustness" / "figures"
LOGS = ROOT / "robustness" / "logs"
for folder in (OUT, FIGURES, LOGS):
    folder.mkdir(parents=True, exist_ok=True)

SEED = 20261008
DEVICE = torch.device("cpu")
ANGLES = (-10, -5, -2, -1, 1, 2, 5, 10)


def source_for(image_index: int, seed: int) -> tuple[int, int, int, int]:
    rng = np.random.default_rng(SEED + 70000 + image_index * 1009 + seed * 97)
    return tuple(int(value) for value in rng.integers(0, 256, 4, dtype=np.uint8))


def packet_messages(symbols: list[int]) -> torch.Tensor:
    return torch.tensor([packet.encode_bits(index, value) for index, value in enumerate(symbols)],
                        dtype=torch.float32)


def oracle_masks_rotated(masks: torch.Tensor, angle: float) -> torch.Tensor:
    arrays = []
    for i in range(len(masks)):
        image = Image.fromarray((masks[i, 0].cpu().numpy() * 255).astype(np.uint8), mode="L")
        image = image.rotate(angle, resample=Image.Resampling.NEAREST, expand=False, fillcolor=0)
        arrays.append(torch.from_numpy((np.asarray(image).copy() > 127).astype(np.float32))[None])
    return torch.stack(arrays)


def infer_case(wam, attacked, source, symbols, messages, oracle_masks):
    detection = ev.infer_packets(wam, attacked)
    decoded = ev.decode_global(detection)
    full_masks = mask_mod.make_masks(8, DEVICE)
    fractions = (oracle_masks.sum((1, 2, 3)) / full_masks.sum((1, 2, 3))).tolist()
    region_rows, metrics = ev.score(detection, list(symbols), messages, oracle_masks, fractions)
    exact = decoded["global_status"] == "decoded" and decoded["global_payload"] == source
    wrong = decoded["global_status"] == "decoded" and not exact
    corrected = sum(row["classification"] == "VALID" and (row["bch_corrected_bits"] or 0) > 0
                    for row in region_rows)
    matched = sum(row["localized"] for row in region_rows)
    bit_errors = sum(row["raw_hamming"] or 0 for row in region_rows)
    row = {"global_exact": bool(exact), "wrong_global_output": bool(wrong),
           "global_status": decoded["global_status"],
           "detected_clusters": metrics["detected_clusters"],
           "matched_packets": matched, "raw_bit_errors_matched": bit_errors,
           "raw_ber_matched": bit_errors / (32 * matched) if matched else None,
           "raw_exact_packets": sum(row["raw_exact"] for row in region_rows),
           "bch_exact_packets": sum(row["bch_exact_16"] for row in region_rows),
           "bch_corrected_exact_packets": corrected,
           "exact_crc_valid_packets": sum(row["classification"] == "VALID" for row in region_rows),
           "erased_packets": sum(row["classification"] == "ERASURE" for row in region_rows),
           "silent_wrong_packets": sum(row["classification"] == "SILENT WRONG" for row in region_rows),
           "detect_seconds": detection["runtime_seconds"],
           "unmatched_accepted_clusters": metrics["unmatched_accepted_clusters"],
           **{key: decoded[key] for key in ("accepted_clusters", "unique_valid_ids", "conflicting_ids",
                                            "erased_id_count", "outer_decode_seconds")}}
    return row, region_rows


def run_baseline(wam, manifest):
    cases, packet_rows, negatives = [], [], []
    masks = mask_mod.make_masks(8, DEVICE)
    for image_index, record in enumerate(manifest):
        original = loader.load_image(record, DEVICE)
        unmarked = ev.infer_packets(wam, original)
        neg_decoded = ev.decode_global(unmarked)
        negatives.append({"image": record["name"], "detected_clusters": len(unmarked["clusters"]),
                          "crc_valid_clusters": neg_decoded["accepted_clusters"],
                          "unique_valid_ids": neg_decoded["unique_valid_ids"],
                          "global_output": neg_decoded["global_status"] == "decoded",
                          "detect_seconds": unmarked["runtime_seconds"]})
        io.write_csv(OUT / "negative_control.csv", negatives)
        for seed in range(3):
            source = source_for(image_index, seed)
            symbols = outer.encode_repetition(source)
            messages = packet_messages(list(symbols))
            marked, embed_seconds = wam_embed.embed_messages(wam, original, messages, masks)
            psnr, ssim = metrics.quality(original, marked)
            case_id = f"{image_index:02d}_{seed}"
            variants = [("clean", marked, masks, None)]
            variants += [(f"jpeg{q}", attacks.jpeg_attack(marked, q), masks, None) for q in (75, 50)]
            variants += [(f"noise{s}", attacks.noise_attack(marked, s, SEED + image_index * 1009 + seed * 97 + s), masks, None)
                         for s in (2, 5)]
            variants += [(f"rotate{angle:+d}", attacks.rotate_attack(marked, angle),
                          oracle_masks_rotated(masks, angle), angle) for angle in ANGLES]
            crop = next(item for item in attacks.fixed_crops(image_index, seed) if item[0] == .75)
            _, crop_variant, _, x0, y0, width, height = crop
            cropped = marked[:, :, y0:y0 + height, x0:x0 + width]
            variants.append(("crop75_jpeg75", attacks.jpeg_attack(cropped, 75),
                             masks[:, :, y0:y0 + height, x0:x0 + width], None))
            for condition, attacked, oracle, angle in variants:
                measured, details = infer_case(wam, attacked, source, symbols, messages, oracle)
                row = {"case_id": case_id, "image": record["name"], "image_index": image_index,
                       "payload_seed": seed, "condition": condition, "attack_angle": angle,
                       "psnr_clean": psnr, "ssim_clean": ssim, "embed_seconds": embed_seconds,
                       **measured}
                cases.append(row)
                for detail in details:
                    packet_rows.append({"case_id": case_id, "condition": condition, **detail})
                io.log_event(LOGS / "events.jsonl", "case", id=case_id, condition=condition,
                             exact=row["global_exact"], valid=row["unique_valid_ids"],
                             silent=row["silent_wrong_packets"])
            io.write_csv(OUT / "cases.csv", cases)
            io.write_csv(OUT / "packets.csv", packet_rows)
    io.log_event(LOGS / "events.jsonl", "baseline_complete", cases=len(cases), negatives=len(negatives))
    return cases


def needs_search(cases):
    values = {}
    for magnitude in (1, 2):
        group = [row for row in cases if row["condition"] in (f"rotate+{magnitude}", f"rotate-{magnitude}")]
        assert len(group) == 60
        values[magnitude] = sum(row["global_exact"] for row in group) / len(group)
    pooled = (values[1] + values[2]) / 2
    decision = pooled < .9 or any(value < .8 for value in values.values())
    io.write_json(OUT / "search_gate.json", {"rate_by_magnitude": values, "pooled": pooled,
                                             "search_triggered": decision})
    return decision


def main():
    global OUT, FIGURES, LOGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "robustness",
                        help="run directory for results/, figures/ and logs/")
    parser.add_argument("--reuse", action="store_true",
                        help="explicitly reuse complete CSVs; does not run model inference")
    args = parser.parse_args()
    OUT, FIGURES, LOGS = [args.output_dir.resolve() / name for name in ("results", "figures", "logs")]
    for folder in (OUT, FIGURES, LOGS):
        folder.mkdir(parents=True, exist_ok=True)
    files = {"cases.csv": 420, "packets.csv": 3360, "negative_control.csv": 10}
    if not args.reuse and any((OUT / name).exists() for name in files):
        parser.error("Existing results found. Use a new --output-dir for fresh measurements, or --reuse for existing CSVs.")
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    io.log_event(LOGS / "events.jsonl", "start", seed=SEED, device=str(DEVICE))
    if args.reuse:
        tables = {}
        for name, expected in files.items():
            if not (OUT / name).is_file():
                parser.error(f"Missing {name}; --reuse requires a complete run")
            with (OUT / name).open() as stream:
                tables[name] = list(csv.DictReader(stream))
            if len(tables[name]) != expected:
                parser.error(f"Incomplete {name}: expected {expected} rows")
        cases = tables["cases.csv"]
        for row in cases:
            row["global_exact"] = row["global_exact"] == "True"
    else:
        manifest = loader.load_images(OUT)
        wam = wam_model.load_default_model(DEVICE)
        io.log_event(LOGS / "events.jsonl", "model_loaded",
                     parameters=sum(x.numel() for x in wam.parameters()))
        cases = run_baseline(wam, manifest)
    if needs_search(cases):
        raise RuntimeError("Rotation search gate triggered, but inverse-angle search is not implemented in this runner. Baseline CSVs are saved; do not claim the full protocol completed.")
    io.log_event(LOGS / "events.jsonl", "experiment_complete")
    print(json.dumps({"cases": len(cases), "results": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
