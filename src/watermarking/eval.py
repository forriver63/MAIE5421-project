"""Per-packet inference and scoring shared by experiments.

Migrated from run_phase1.py (infer_packets / score) and run_robustness.py
(decode_global), with the BCH sub-result read from ``packet["bch"]`` (identical
to the old ``cluster["decoded"]`` produced by decode_int on the same word).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .codec.outer import collect_accepted_packets, decode_mds, decode_repetition
from .codec.packet import decode_packet, encode_packet, packet16
from .wam.detect import detect_blind


def infer_packets(wam, marked: torch.Tensor) -> dict:
    detection = detect_blind(wam, marked)
    for cluster in detection["clusters"]:
        cluster["packet"] = decode_packet(cluster["raw_word"])
    return detection


def score(detection: dict, symbols: list[int], messages: torch.Tensor,
          oracle_masks: torch.Tensor, physical_fractions: list[float]) -> tuple[list[dict], dict]:
    """Oracle overlap is for evaluation only, applied after blind inference."""
    count = len(symbols)
    clusters = detection["clusters"]
    ground = F.interpolate(oracle_masks.detach().cpu(), size=detection["labels"].shape[-2:], mode="nearest")[:, 0] > .5
    overlap = np.zeros((len(clusters), count), dtype=np.int32)
    for ci, cluster in enumerate(clusters):
        for region in range(count):
            overlap[ci, region] = int((cluster["mask"] & ground[region]).sum())
    assigned = {}
    if len(clusters):
        rows, columns = linear_sum_assignment(-overlap)
        for ci, region in zip(rows, columns):
            if overlap[ci, region] > 0:
                assigned[int(region)] = int(ci)
    region_rows = []
    for region in range(count):
        item = {"region": region, "physical_retention": physical_fractions[region],
                "localized": region in assigned, "raw_hamming": None,
                "raw_exact": False, "bch_exact_16": False,
                "bch_status": "missing", "bch_corrected_bits": None,
                "crc_status": "missing", "accepted_packet_id": None,
                "accepted_symbol": None, "classification": "ERASURE",
                "confidence": None, "cluster_label": None, "overlap_pixels_256": 0}
        if region in assigned:
            cluster = clusters[assigned[region]]
            packet = cluster["packet"]
            expected = encode_packet(region, symbols[region])
            hamming = (cluster["raw_word"] ^ expected).bit_count()
            bch = packet["bch"]
            bch_exact = bch["status"] == "valid" and bch["payload"] == packet16(region, symbols[region])
            classification = "ERASURE"
            if packet["status"] == "ACCEPTED":
                classification = "VALID" if (packet["packet_id"], packet["symbol"]) == (region, symbols[region]) else "SILENT WRONG"
            item.update({
                "raw_hamming": hamming, "raw_exact": hamming == 0,
                "bch_exact_16": bch_exact, "bch_status": bch["status"],
                "bch_corrected_bits": bch["corrected_total_bits"],
                "crc_status": packet["reason"] or "accepted",
                "accepted_packet_id": packet["packet_id"], "accepted_symbol": packet["symbol"],
                "classification": classification, "confidence": cluster["confidence"],
                "cluster_label": cluster["label"], "overlap_pixels_256": int(overlap[assigned[region], region]),
            })
        region_rows.append(item)
    assigned_cluster_indices = set(assigned.values())
    unmatched = [cluster for ci, cluster in enumerate(clusters) if ci not in assigned_cluster_indices]
    metrics = {"detected_clusters": len(clusters), "matched_clusters": len(assigned),
               "unmatched_clusters": len(unmatched),
               "unmatched_accepted_clusters": sum(cluster["packet"]["status"] == "ACCEPTED" for cluster in unmatched),
               "clustering_error": detection["clustering_error"]}
    return region_rows, metrics


def decode_global(detection: dict, method: str = "repetition") -> dict:
    if method not in ("repetition", "mds"):
        raise ValueError(f"Unknown outer code: {method}")
    start = time.perf_counter()
    packets = [cluster["packet"] for cluster in detection["clusters"]]
    unique, counts = collect_accepted_packets(packets)
    decoded = decode_repetition(unique) if method == "repetition" else decode_mds(unique)
    return {**counts, "global_status": decoded["status"],
            "global_payload": decoded["payload"],
            "outer_decode_seconds": time.perf_counter() - start}
