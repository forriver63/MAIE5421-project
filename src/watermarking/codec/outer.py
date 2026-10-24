"""Fixed (8,4) GF(256) Vandermonde MDS and spatial repetition control.

Migrated verbatim from validation_wam_hierarchical_final_20261008/outer_codes.py
to preserve bit-identical behavior.
"""
from __future__ import annotations

import functools
from itertools import combinations

PRIMITIVE_POLYNOMIAL = 0x11D
REPETITION_SOURCE_BY_REGION = (0, 1, 2, 3, 3, 2, 1, 0)


def gf_mul(a: int, b: int) -> int:
    out = 0
    while b:
        if b & 1:
            out ^= a
        b >>= 1
        a <<= 1
        if a & 0x100:
            a ^= PRIMITIVE_POLYNOMIAL
    return out


def gf_pow(a: int, exponent: int) -> int:
    out = 1
    for _ in range(exponent):
        out = gf_mul(out, a)
    return out


def gf_inverse(a: int) -> int:
    if a == 0:
        raise ZeroDivisionError("zero has no GF(256) inverse")
    return gf_pow(a, 254)


MDS_ROWS = tuple(tuple(gf_pow(point, degree) for degree in range(4)) for point in range(1, 9))


def encode_mds(source: tuple[int, int, int, int]) -> tuple[int, ...]:
    assert len(source) == 4 and all(0 <= item < 256 for item in source)
    return tuple(functools.reduce(
        int.__xor__, (gf_mul(coefficient, symbol) for coefficient, symbol in zip(row, source)), 0)
        for row in MDS_ROWS)


def encode_repetition(source: tuple[int, int, int, int]) -> tuple[int, ...]:
    assert len(source) == 4 and all(0 <= item < 256 for item in source)
    return tuple(source[index] for index in REPETITION_SOURCE_BY_REGION)


def solve_four(rows: list[tuple[int, int, int, int]], values: list[int]) -> tuple[int, ...] | None:
    if len(rows) != 4 or len(values) != 4:
        raise ValueError("need exactly four rows and values")
    matrix = [list(row) + [value] for row, value in zip(rows, values)]
    for column in range(4):
        pivot = next((row for row in range(column, 4) if matrix[row][column]), None)
        if pivot is None:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        inverse = gf_inverse(matrix[column][column])
        matrix[column] = [gf_mul(inverse, value) for value in matrix[column]]
        for other in range(4):
            if other != column and matrix[other][column]:
                factor = matrix[other][column]
                matrix[other] = [value ^ gf_mul(factor, reference)
                                 for value, reference in zip(matrix[other], matrix[column])]
    return tuple(matrix[index][4] for index in range(4))


def collect_accepted_packets(decoded_packets: list[dict]) -> tuple[dict[int, int], dict]:
    """Collect CRC-valid predictions without using source data or spatial masks."""
    candidates = {}
    for packet in decoded_packets:
        if packet["status"] == "ACCEPTED":
            candidates.setdefault(packet["packet_id"], set()).add(packet["symbol"])
    conflicts = [packet_id for packet_id, values in candidates.items() if len(values) > 1]
    unique = {packet_id: next(iter(values)) for packet_id, values in candidates.items() if len(values) == 1}
    return unique, {"accepted_clusters": sum(packet["status"] == "ACCEPTED" for packet in decoded_packets),
                    "unique_valid_ids": len(unique), "conflicting_ids": sorted(conflicts),
                    "erased_id_count": 8 - len(unique)}


def decode_mds(unique: dict[int, int]) -> dict:
    if len(unique) < 4:
        return {"status": "insufficient_packets", "payload": None}
    indices = sorted(unique)
    candidate = solve_four([MDS_ROWS[index] for index in indices[:4]],
                           [unique[index] for index in indices[:4]])
    if candidate is None:
        return {"status": "singular_subset", "payload": None}
    regenerated = encode_mds(candidate)
    if any(regenerated[index] != unique[index] for index in indices):
        return {"status": "inconsistent_packets", "payload": None}
    return {"status": "decoded", "payload": candidate}


def decode_repetition(unique: dict[int, int]) -> dict:
    recovered = []
    for source_index in range(4):
        values = {unique[region] for region in range(8)
                  if region in unique and REPETITION_SOURCE_BY_REGION[region] == source_index}
        if len(values) == 0:
            return {"status": "missing_source_symbol", "payload": None}
        if len(values) > 1:
            return {"status": "repetition_conflict", "payload": None}
        recovered.append(next(iter(values)))
    return {"status": "decoded", "payload": tuple(recovered)}


def exhaustive_mds_verification(test_sources: list[tuple[int, int, int, int]]) -> dict:
    subsets = list(combinations(range(8), 4))
    assert len(subsets) == 70
    for subset in subsets:
        rows = [MDS_ROWS[index] for index in subset]
        assert solve_four(rows, [0, 0, 0, 0]) is not None
        for source in test_sources:
            coded = encode_mds(source)
            recovered = solve_four(rows, [coded[index] for index in subset])
            assert recovered == source, (subset, source, recovered)
    return {"subsets": len(subsets), "test_sources": len(test_sources),
            "recoveries": len(subsets) * len(test_sources),
            "every_four_row_subset_invertible": True,
            "all_exact": True,
            "gf_primitive_polynomial_hex": hex(PRIMITIVE_POLYNOMIAL),
            "matrix_rows": [list(row) for row in MDS_ROWS]}
