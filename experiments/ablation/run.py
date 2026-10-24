"""Ablation runner (extensible).

This compares outer codes (repetition vs MDS), not removal of BCH/CRC components.
To add a method, update both METHODS and watermarking.eval.decode_global.
To add an attack, update both the variants and requested condition list.
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
OUT = ROOT / "ablation" / "results"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20261008
DEVICE = torch.device("cpu")

# Registered outer codes; decoding is dispatched by name in eval.decode_global.
METHODS = {
    "repetition": (outer.encode_repetition, outer.decode_repetition),
    "mds": (outer.encode_mds, outer.decode_mds),
}


def source_for(image_index: int, seed: int) -> tuple[int, int, int, int]:
    rng = np.random.default_rng(SEED + 70000 + image_index * 1009 + seed * 97)
    return tuple(int(v) for v in rng.integers(0, 256, 4, dtype=np.uint8))


def packet_messages(symbols):
    return torch.tensor([packet.encode_bits(i, s) for i, s in enumerate(symbols)],
                        dtype=torch.float32)


def run_ablation(wam, manifest, conditions=("clean", "jpeg75")):
    masks = mask_mod.make_masks(8, DEVICE)
    rows = []
    for image_index, record in enumerate(manifest):
        original = loader.load_image(record, DEVICE)
        for seed in range(3):
            source = source_for(image_index, seed)
            for method, (encode_fn, _) in METHODS.items():
                symbols = encode_fn(source)
                messages = packet_messages(list(symbols))
                marked, _ = wam_embed.embed_messages(wam, original, messages, masks)
                variants = [("clean", marked, masks)]
                variants += [(f"jpeg{q}", attacks.jpeg_attack(marked, q), masks) for q in (75, 50)]
                for condition, attacked, oracle in variants:
                    if condition not in conditions:
                        continue
                    detection = ev.infer_packets(wam, attacked)
                    decoded = ev.decode_global(detection, method=method)
                    exact = decoded["global_status"] == "decoded" and decoded["global_payload"] == source
                    rows.append({"image": record["name"], "image_index": image_index,
                                 "payload_seed": seed, "method": method, "condition": condition,
                                 "global_exact": bool(exact), "status": decoded["global_status"],
                                 "valid_ids": decoded["unique_valid_ids"]})
                    io.log_event(OUT / "events.jsonl", "ablation_case", method=method,
                                 image=record["name"], seed=seed, condition=condition, exact=bool(exact))
    io.write_csv(OUT / "ablation_cases.csv", rows)
    summary = []
    for method in METHODS:
        for condition in conditions:
            group = [r for r in rows if r["method"] == method and r["condition"] == condition]
            summary.append({"method": method, "condition": condition, "cases": len(group),
                            "exact_rate": float(np.mean([r["global_exact"] for r in group])) if group else None,
                            "mean_valid_ids": float(np.mean([r["valid_ids"] for r in group])) if group else None})
    io.write_json(OUT / "ablation_summary.json", summary)
    return summary


def main():
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    manifest = loader.load_images(OUT)
    wam = wam_model.load_default_model(DEVICE)
    summary = run_ablation(wam, manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
