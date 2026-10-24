"""Inner-ECC clean validation (refactored from run_sprint.py).

Verifies that BCH(31,16)+parity protects the 16-bit useful payloads against the
small local bit errors observed in clean WAM decoding.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from watermarking import io, masks as mask_mod, metrics
from watermarking.codec import bch
from watermarking.data import loader
from watermarking.wam import detect as wam_detect
from watermarking.wam import embed as wam_embed
from watermarking.wam import model as wam_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "inner_ecc" / "results"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20261008
DEVICE = torch.device("cpu")


def useful_for(image_index: int, payload_seed: int, count: int) -> list[int]:
    rng = np.random.default_rng(SEED + image_index * 1009 + payload_seed * 97 + count)
    values = [int(x) for x in rng.integers(0, 1 << 16, count, dtype=np.uint16)]
    assert len(set(values)) == count
    return values


def protected_messages(useful: list[int]) -> torch.Tensor:
    return torch.tensor([bch.int_to_bits(bch.encode_int(value)) for value in useful], dtype=torch.float32)


def score_regions(detected, useful, masks, messages):
    count = len(useful)
    clusters = detected["clusters"]
    ground = F.interpolate(masks.detach().cpu(), size=detected["labels"].shape[-2:], mode="nearest")[:, 0] > 0.5
    overlaps = np.zeros((len(clusters), count), dtype=np.int32)
    for ci, cluster in enumerate(clusters):
        for ri in range(count):
            overlaps[ci, ri] = int((cluster["mask"] & ground[ri]).sum().item())
    paired = {}
    if len(clusters):
        rows, columns = linear_sum_assignment(-overlaps)
        for ci, ri in zip(rows, columns):
            if overlaps[ci, ri] > 0:
                paired[int(ri)] = int(ci)
    details = []
    for ri in range(count):
        row = {"region": ri, "localized": ri in paired, "raw_hamming": None,
               "raw_exact": False, "useful_exact": False, "silent_wrong_decode": False,
               "bch_status": "missing", "corrected_total_bits": None}
        if ri in paired:
            cluster = clusters[paired[ri]]
            truth_word = bch.bits_to_int(messages[ri].int().tolist())
            hamming = (cluster["raw_word"] ^ truth_word).bit_count()
            decoded = bch.decode_int(cluster["raw_word"])
            row.update({"raw_hamming": hamming, "raw_exact": hamming == 0,
                        "useful_exact": decoded["status"] == "valid" and decoded["payload"] == useful[ri],
                        "silent_wrong_decode": decoded["status"] == "valid" and decoded["payload"] != useful[ri],
                        "bch_status": decoded["status"], "corrected_total_bits": decoded["corrected_total_bits"]})
        details.append(row)
    return details


def run_clean(wam, manifest):
    rows, packets = [], []
    masks = mask_mod.make_masks(4, DEVICE)
    for image_index, record in enumerate(manifest):
        image = loader.load_image(record, DEVICE)
        for payload_seed in range(3):
            useful = useful_for(image_index, payload_seed, 4)
            messages = protected_messages(useful)
            marked, embed_seconds = wam_embed.embed_messages(wam, image, messages, masks)
            detection = wam_detect.detect_blind(wam, marked)
            details = score_regions(detection, useful, masks, messages)
            psnr, ssim = metrics.quality(image, marked)
            row = {"image": record["name"], "image_index": image_index, "payload_seed": payload_seed,
                   "regions": 4, "raw_exact_packets": sum(i["raw_exact"] for i in details),
                   "useful_exact_packets": sum(i["useful_exact"] for i in details),
                   "silent_wrong_decodes": sum(i["silent_wrong_decode"] for i in details),
                   "matched_packets": sum(i["localized"] for i in details),
                   "psnr": psnr, "ssim": ssim, "embed_seconds": embed_seconds,
                   "detect_seconds": detection["runtime_seconds"], "detected_clusters": len(detection["clusters"])}
            rows.append(row)
            for item in details:
                packets.append({"image": record["name"], "image_index": image_index, "payload_seed": payload_seed, **item})
            io.log_event(OUT / "events.jsonl", "clean_case", image=record["name"], seed=payload_seed,
                         raw_exact=row["raw_exact_packets"], protected_exact=row["useful_exact_packets"])
        io.write_csv(OUT / "clean_cases.csv", rows)
        io.write_csv(OUT / "clean_packets.csv", packets)
    return summarize_clean(rows, packets)


def summarize_clean(rows, packets):
    total = len(packets)
    protected = sum(row["useful_exact_packets"] for row in rows)
    raw = sum(row["raw_exact_packets"] for row in rows)
    silent = sum(row["silent_wrong_decodes"] for row in rows)
    summary = {"cases": len(rows), "packets": total,
               "raw_exact_packets": raw, "raw_exact_rate": raw / total,
               "useful_exact_packets": protected, "useful_exact_rate": protected / total,
               "silent_wrong_decodes": silent, "silent_wrong_rate": silent / total,
               "mean_psnr": float(np.mean([row["psnr"] for row in rows])),
               "mean_ssim": float(np.mean([row["ssim"] for row in rows]))}
    # Frozen Phase 2 reference in the original protocol, not an assumed pass.
    summary["reference_mean_psnr"] = 44.08383420582539
    summary["reference_mean_ssim"] = 0.9884136557579041
    summary["all_four_useful_exact_rate"] = sum(r["useful_exact_packets"] == 4 for r in rows) / len(rows)
    summary["psnr_change_db"] = summary["mean_psnr"] - summary["reference_mean_psnr"]
    summary["ssim_change"] = summary["mean_ssim"] - summary["reference_mean_ssim"]
    summary["gate"] = {"protected_packet_recovery_ge_98pct": summary["useful_exact_rate"] >= 0.98,
                       "all_four_images_ge_90pct": summary["all_four_useful_exact_rate"] >= 0.90,
                       "silent_wrong_le_1pct": summary["silent_wrong_rate"] <= 0.01,
                       "psnr_drop_le_0p5_db": summary["psnr_change_db"] >= -0.5,
                       "ssim_drop_le_0p005": summary["ssim_change"] >= -0.005}
    summary["gate"]["pass"] = all(summary["gate"].values())
    io.write_json(OUT / "clean_summary.json", summary)
    return summary


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    manifest = loader.load_images(OUT)
    wam = wam_model.load_default_model(DEVICE)
    summary = run_clean(wam, manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
