"""Official-WAM staged baseline screening (refactored from run_validation.py).

Loads the 10 public images, writes the SHA-256 manifest, and runs the staged
clean 2/4/8-region gates, stopping at the first failed gate. Uses the early
gt-matched detector (detect_messages) preserved inline for historical scoring.
Only the clean screening stage is implemented here, not the original crop gates.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from watermarking import attacks, io, masks as mask_mod
from watermarking.data import loader
from watermarking.wam import embed as wam_embed
from watermarking.wam import model as wam_model
from watermarking.wam.detect import path as wam_path
from watermarking.wam.transforms import default_transform

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "baseline" / "results"
VIZ = ROOT / "baseline" / "visualizations"
for folder in (OUT, VIZ):
    folder.mkdir(parents=True, exist_ok=True)

SEED = 20261007
DEVICE = torch.device("cpu")
IMAGE_SIZE = 512

wam_path.ensure_on_syspath()
from notebooks.inference_utils import multiwm_dbscan  # noqa: E402


def make_masks(count: int, device: torch.device) -> torch.Tensor:
    return mask_mod.make_masks(count, device)


def messages_for(image_index: int, payload_seed: int, count: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED + image_index * 1009 + payload_seed * 97 + count)
    messages = torch.randint(0, 2, (count, 32), generator=generator).float()
    if len({tuple(row.tolist()) for row in messages}) != count:
        raise RuntimeError("Unexpected duplicate message in fixed screening set")
    return messages


def detect_messages(wam, image, gt_messages, gt_masks=None):
    start = time.perf_counter()
    with torch.inference_mode():
        preds = wam.detect(image)["preds"]
    mask_prob = torch.sigmoid(preds[:, 0])
    bit_logits = preds[:, 1:]
    detected_pixels = int((mask_prob > 0.5).sum().item())
    centroids = {}
    labels = torch.full(mask_prob.shape[-2:], -1.0)
    clustering_error = None
    if detected_pixels:
        try:
            centroids, labels = multiwm_dbscan(bit_logits, mask_prob, epsilon=1, min_samples=500)
        except Exception as error:
            clustering_error = f"{type(error).__name__}: {error}"
    cluster_items = sorted(centroids.items(), key=lambda item: int(item[0]))
    predictions = torch.stack([item[1].float() for item in cluster_items]) if cluster_items else torch.empty((0, 32))
    truths = gt_messages.detach().cpu().float()
    matched = []
    if len(predictions) and len(truths):
        hamming = (predictions[:, None, :] != truths[None, :, :]).sum(2).numpy()
        rows, columns = linear_sum_assignment(hamming)
        for row, column in zip(rows, columns):
            label_value = cluster_items[row][0]
            pixel_mask = labels == float(label_value)
            bit_prob = torch.sigmoid(bit_logits[0].detach().cpu())
            confidence = float((2 * (bit_prob[:, pixel_mask] - 0.5).abs()).mean()) if pixel_mask.any() else 0.0
            matched.append({"cluster": int(label_value), "truth": int(column),
                            "hamming": int(hamming[row, column]),
                            "bit_accuracy": float(1 - hamming[row, column] / 32), "confidence": confidence})
    exact = sum(item["hamming"] == 0 for item in matched)
    return {"detected_clusters": len(predictions), "exact_packets": exact,
            "packet_recall": exact / len(truths),
            "mean_matched_bit_accuracy": float(np.mean([item["bit_accuracy"] for item in matched])) if matched else 0.0,
            "detected_pixels": detected_pixels, "runtime_seconds": time.perf_counter() - start,
            "clustering_error": clustering_error}


def summarize_clean(rows, count):
    selected = [row for row in rows if row["region_count"] == count]
    packets = sum(row["exact_packets"] for row in selected)
    total = len(selected) * count
    return {"region_count": count, "cases": len(selected), "exact_packet_recall": packets / total,
            "at_least_four_exact_rate": sum(row["exact_packets"] >= 4 for row in selected) / len(selected),
            "all_exact_image_rate": sum(row["exact_packets"] == count for row in selected) / len(selected),
            "mean_bit_accuracy": float(np.mean([row["mean_matched_bit_accuracy"] for row in selected])),
            "mean_clusters": float(np.mean([row["detected_clusters"] for row in selected])),
            "mean_psnr": float(np.mean([row["psnr"] for row in selected])),
            "mean_ssim": float(np.mean([row["ssim"] for row in selected]))}


def run_clean(wam, manifest):
    rows = []
    summaries = []
    stop = None
    for count in (2, 4, 8):
        masks = make_masks(count, DEVICE)
        for image_index, record in enumerate(manifest):
            image = loader.load_image(record, DEVICE)
            for payload_seed in range(3):
                messages = messages_for(image_index, payload_seed, count)
                watermarked, embed_seconds = wam_embed.embed_messages(wam, image, messages, masks)
                detected = detect_messages(wam, watermarked, messages, masks)
                orig_np = attacks.tensor_rgb(image)
                wm_np = attacks.tensor_rgb(watermarked)
                psnr = float(peak_signal_noise_ratio(orig_np, wm_np, data_range=1.0))
                ssim = float(structural_similarity(orig_np, wm_np, channel_axis=2, data_range=1.0))
                rows.append({"phase": "clean", "image": record["name"], "image_index": image_index,
                             "payload_seed": payload_seed, "region_count": count, "psnr": psnr, "ssim": ssim,
                             "embed_seconds": embed_seconds, "detect_seconds": detected["runtime_seconds"],
                             "exact_packets": detected["exact_packets"], "packet_recall": detected["packet_recall"],
                             "mean_matched_bit_accuracy": detected["mean_matched_bit_accuracy"],
                             "detected_clusters": detected["detected_clusters"], "detected_pixels": detected["detected_pixels"],
                             "clustering_error": detected["clustering_error"]})
                io.log_event(OUT / "events.jsonl", "clean_case", image=record["name"], seed=payload_seed,
                             region_count=count, exact_packets=detected["exact_packets"])
        summary = summarize_clean(rows, count)
        summaries.append(summary)
        io.write_csv(OUT / "clean_cases.csv", rows)
        io.write_json(OUT / "clean_summary.json", summaries)
        if count == 2 and not (summary["exact_packet_recall"] >= 0.95 and summary["all_exact_image_rate"] >= 0.90):
            stop = "Clean two-region gate failed"
        elif count == 4 and not (summary["exact_packet_recall"] >= 0.90 and summary["all_exact_image_rate"] >= 0.80):
            stop = "Clean four-region gate failed"
        elif count == 8 and not (summary["exact_packet_recall"] >= 0.85
                                and summary["at_least_four_exact_rate"] >= 0.90):
            stop = "Clean eight-region gate failed"
        if stop:
            break
    return rows, summaries, stop


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    manifest = loader.load_images(OUT)
    wam = wam_model.load_default_model(DEVICE)
    io.log_event(OUT / "events.jsonl", "model_loaded", parameters=sum(p.numel() for p in wam.parameters()))
    _, summaries, stop = run_clean(wam, manifest)
    result = {"decision": "CLEAN_PASS" if stop is None else "NO-GO",
              "scope": "clean screening only; crop/coding gates are not run",
              "stop_reason": stop, "clean_summary": summaries}
    io.write_json(OUT / "final_result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
