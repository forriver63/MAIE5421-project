"""Hierarchical outer-code comparison (refactored from run_final_comparison.py).

Paired blind-crop comparison of spatial repetition vs GF(256) MDS under the same
embedding budget, using the CRC/BCH packet codec and the shared eval functions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from watermarking import attacks, eval as ev, io, masks as mask_mod, metrics
from watermarking.codec import outer, packet
from watermarking.data import loader
from watermarking.wam import embed as wam_embed
from watermarking.wam import model as wam_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "hierarchical" / "results"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20261008
DEVICE = torch.device("cpu")


def source_for(image_index: int, seed: int) -> tuple[int, int, int, int]:
    rng = np.random.default_rng(SEED + 70000 + image_index * 1009 + seed * 97)
    return tuple(int(value) for value in rng.integers(0, 256, 4, dtype=np.uint8))


def packet_messages(symbols):
    return torch.tensor([packet.encode_bits(i, s) for i, s in enumerate(symbols)], dtype=torch.float32)


def run_comparison(wam, manifest):
    masks = mask_mod.make_masks(8, DEVICE)
    crop_rows, clean_rows, paired = [], [], []
    for image_index, record in enumerate(manifest):
        original = loader.load_image(record, DEVICE)
        for seed in range(3):
            source = source_for(image_index, seed)
            crop_windows = list(attacks.fixed_crops(image_index, seed))
            for method in ("repetition", "mds"):
                encode_fn = outer.encode_repetition if method == "repetition" else outer.encode_mds
                symbols = encode_fn(source)
                messages = packet_messages(list(symbols))
                marked, embed_seconds = wam_embed.embed_messages(wam, original, messages, masks)
                psnr, ssim = metrics.quality(original, marked)
                clean = ev.decode_global(ev.infer_packets(wam, marked), method=method)
                clean_exact = clean["global_status"] == "decoded" and clean["global_payload"] == source
                clean_rows.append({"method": method, "image": record["name"], "image_index": image_index,
                                   "payload_seed": seed, "global_exact": bool(clean_exact),
                                   "psnr": psnr, "ssim": ssim})
                for retained, variant, ratio, x0, y0, width, height in crop_windows:
                    cropped = marked[:, :, y0:y0 + height, x0:x0 + width]
                    detection = ev.infer_packets(wam, cropped)
                    decoded = ev.decode_global(detection, method=method)
                    oracle = masks[:, :, y0:y0 + height, x0:x0 + width]
                    fractions = (oracle.sum((1, 2, 3)) / masks.sum((1, 2, 3))).tolist()
                    details, scored = ev.score(detection, list(symbols), messages, oracle, fractions)
                    exact = decoded["global_status"] == "decoded" and decoded["global_payload"] == source
                    crop_rows.append({"method": method, "image": record["name"], "image_index": image_index,
                                      "payload_seed": seed, "retained_target": retained, "variant": variant,
                                      "global_exact": bool(exact), "unique_valid_ids": decoded["unique_valid_ids"],
                                      "silent_wrong_packets": sum(r["classification"] == "SILENT WRONG" for r in details),
                                      "unmatched_accepted_clusters": scored["unmatched_accepted_clusters"]})
                io.log_event(OUT / "events.jsonl", "paired_crop_case", method=method, image=record["name"],
                             seed=seed, exact_global=bool(clean_exact))
        io.write_csv(OUT / "clean_cases.csv", clean_rows)
        io.write_csv(OUT / "crop_cases.csv", crop_rows)
    index = {(r["method"], r["image_index"], r["payload_seed"], r["retained_target"], r["variant"]): r for r in crop_rows}
    for image_index in range(len(manifest)):
        for seed in range(3):
            for area in (.75, .5):
                for variant in range(3):
                    rep = index[("repetition", image_index, seed, area, variant)]
                    mds = index[("mds", image_index, seed, area, variant)]
                    paired.append({"image": rep["image"], "image_index": image_index, "payload_seed": seed,
                                   "retained_target": area, "variant": variant,
                                   "repetition_exact": rep["global_exact"], "mds_exact": mds["global_exact"]})
    io.write_csv(OUT / "paired_global_outcomes.csv", paired)
    return summarize_comparison(clean_rows, crop_rows, paired)


def summarize_comparison(clean_rows, crop_rows, paired):
    by_method = {}
    for method in ("repetition", "mds"):
        crops = [r for r in crop_rows if r["method"] == method]
        by_method[method] = {"crops": len(crops),
                             "exact_global": sum(r["global_exact"] for r in crops),
                             "rate": float(np.mean([r["global_exact"] for r in crops]))}
    combined_difference = float(np.mean([int(r["mds_exact"]) - int(r["repetition_exact"]) for r in paired]))
    differences = {str(area): float(np.mean([int(r["mds_exact"]) - int(r["repetition_exact"])
                                            for r in paired if r["retained_target"] == area]))
                   for area in (.75, .5)}
    quality = {method: {key: float(np.mean([r[key] for r in clean_rows if r["method"] == method]))
                        for key in ("psnr", "ssim")} for method in by_method}
    psnr_difference = quality["mds"]["psnr"] - quality["repetition"]["psnr"]
    ssim_difference = quality["mds"]["ssim"] - quality["repetition"]["ssim"]
    error_rates = {method: sum(r["silent_wrong_packets"] + r["unmatched_accepted_clusters"]
                               for r in crop_rows if r["method"] == method) / (by_method[method]["crops"] * 8)
                   for method in by_method}
    summary = {"paired_crops": len(paired), "by_method": by_method,
               "combined_paired_difference": combined_difference,
               "paired_differences_by_area": differences,
               "psnr_mds_minus_repetition_db": psnr_difference,
               "ssim_mds_minus_repetition": ssim_difference,
               "conservative_accepted_error_or_unmatched_rates": error_rates,
               "scope": "outer comparison only; historical phase 1/2 prerequisites are not rerun",
               "gate": {"combined_gain_ge_10pp": combined_difference >= .10,
                        "nonnegative_gain_each_area": all(v >= 0 for v in differences.values()),
                        "psnr_difference_le_0p5db": abs(psnr_difference) <= .5,
                        "ssim_difference_le_0p005": abs(ssim_difference) <= .005,
                        "silent_or_unmatched_accepted_le_0p5pct_each_method": all(v <= .005 for v in error_rates.values())}}
    summary["gate"]["pass"] = all(summary["gate"].values())
    io.write_json(OUT / "final_comparison_summary.json", summary)
    return summary


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    manifest = loader.load_images(OUT)
    wam = wam_model.load_default_model(DEVICE)
    summary = run_comparison(wam, manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
