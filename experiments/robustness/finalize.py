"""Aggregate robustness evidence and write summary + figure + decision.

Refactored from validation_wam_final_robustness_20261008/finalize.py.
"""
from __future__ import annotations

import csv
import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from watermarking import io

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "robustness" / "results"
FIGURES = ROOT / "robustness" / "figures"


def read_csv(name):
    with (RESULTS / name).open() as stream:
        return list(csv.DictReader(stream))


def num(value):
    return float(value or 0)


def mean(rows, key):
    return float(np.mean([num(row[key]) for row in rows])) if rows else None


def pct(n, d):
    return f"{100 * n / d:.1f}%" if d else "n/a"


def main():
    global RESULTS, FIGURES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "robustness",
                        help="run directory containing results/; figures/ is written here")
    args = parser.parse_args()
    RESULTS = args.output_dir.resolve() / "results"
    FIGURES = args.output_dir.resolve() / "figures"
    FIGURES.mkdir(parents=True, exist_ok=True)
    cases = read_csv("cases.csv")
    packets = read_csv("packets.csv")
    negatives = read_csv("negative_control.csv")
    assert len(cases) == 420 and len(packets) == 420 * 8 and len(negatives) == 10
    conditions = ["clean", "jpeg75", "jpeg50", "noise2", "noise5",
                  "rotate-1", "rotate+1", "rotate-2", "rotate+2",
                  "rotate-5", "rotate+5", "rotate-10", "rotate+10", "crop75_jpeg75"]
    summaries = []
    for condition in conditions:
        group = [x for x in cases if x["condition"] == condition]
        assert len(group) == 30
        associated = [x for x in packets if x["condition"] == condition]
        matched = sum(int(x["matched_packets"]) for x in group)
        errors = sum(int(x["raw_bit_errors_matched"]) for x in group)
        exact = sum(x["global_exact"] == "True" for x in group)
        summaries.append({
            "condition": condition, "n": len(group), "global_exact": exact,
            "global_exact_rate": exact / len(group),
            "wrong_global_outputs": sum(x["wrong_global_output"] == "True" for x in group),
            "mean_clusters": mean(group, "detected_clusters"),
            "mean_unique_valid_ids": mean(group, "unique_valid_ids"),
            "packet_erasure_rate": sum(int(x["erased_packets"]) for x in group) / (8 * len(group)),
            "matched_packets": matched,
            "raw_ber_matched": errors / (32 * matched) if matched else None,
            "bch_corrected_exact_packets": sum(int(x["bch_corrected_exact_packets"]) for x in group),
            "bch_exact_packets": sum(int(x["bch_exact_packets"]) for x in group),
            "crc_valid_exact_packets": sum(int(x["exact_crc_valid_packets"]) for x in group),
            "silent_wrong_packets": sum(int(x["silent_wrong_packets"]) for x in group),
            "unmatched_accepted_clusters": sum(int(x["unmatched_accepted_clusters"]) for x in group),
            "mean_detect_seconds": mean(group, "detect_seconds"),
            "raw_exact_packets": sum(x["raw_exact"] == "True" for x in associated),
        })
    gate = json.loads((RESULTS / "search_gate.json").read_text())
    summary = {"conditions": summaries,
               "negative_control": {"images": len(negatives),
                                    "false_clusters": sum(int(x["detected_clusters"]) for x in negatives),
                                    "crc_valid_false_clusters": sum(int(x["crc_valid_clusters"]) for x in negatives),
                                    "false_global_outputs": sum(x["global_output"] == "True" for x in negatives)},
               "search_gate": gate,
               "search": {"cases": 0, "exact": 0, "wrong_global_outputs": 0, "mean_wall_seconds": None}}
    io.write_json(RESULTS / "summary.json", summary)

    by = {x["condition"]: x for x in summaries}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    severity = ["clean", "jpeg75", "jpeg50", "noise2", "noise5", "crop75_jpeg75"]
    labels = ["Clean", "JPEG 75", "JPEG 50", "Noise σ2", "Noise σ5", "75% crop\n+ JPEG 75"]
    axes[0].bar(labels, [by[x]["global_exact_rate"] for x in severity], color="#3867a6")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Exact 32-bit global recovery")
    axes[0].set_title("Photometric and combined attacks")
    for i, x in enumerate(severity):
        axes[0].text(i, by[x]["global_exact_rate"] + .012, f'{by[x]["global_exact"]}/30', ha="center", fontsize=8)
    angles = [-10, -5, -2, -1, 1, 2, 5, 10]
    axes[1].plot(angles, [by[f"rotate{x:+d}"]["global_exact_rate"] for x in angles], "o-", color="#b04d45")
    axes[1].axhline(.9, ls="--", lw=1, color="gray")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(angles)
    axes[1].set_xlabel("Rotation angle (degrees)")
    axes[1].set_ylabel("Exact 32-bit global recovery")
    axes[1].set_title("No correction, blind decoder")
    fig.tight_layout()
    fig.savefig(FIGURES / "global_recovery_and_rotation.png", dpi=180)
    plt.close(fig)

    small = [by[f"rotate{x:+d}"] for x in (-2, -1, 1, 2)]
    jpeg_good = min(by["jpeg75"]["global_exact_rate"], by["jpeg50"]["global_exact_rate"]) >= .9
    noise_good = min(by["noise2"]["global_exact_rate"], by["noise5"]["global_exact_rate"]) >= .9
    rotation_good = min(x["global_exact_rate"] for x in small) >= .8
    if jpeg_good and noise_good and rotation_good and not gate["search_triggered"]:
        decision = "FINALIZED"
    else:
        decision = "LIMITED"
    io.write_json(RESULTS / "decision.json", {"decision": decision})
    print(json.dumps({"decision": decision, "summary": str(RESULTS / "summary.json"),
                      "figure": str(FIGURES / "global_recovery_and_rotation.png")}, indent=2))


if __name__ == "__main__":
    main()
