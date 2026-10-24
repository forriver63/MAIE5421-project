"""CRC-5 error-detection check, polynomial x^5 + x^2 + 1 (0x25).

Migrated verbatim from validation_wam_hierarchical_final_20261008/packet_codec.py
to preserve bit-identical behavior.
"""
from __future__ import annotations

CRC_POLY = 0x25  # x^5 + x^2 + 1, including the x^5 term.


def crc5(data11: int) -> int:
    if not 0 <= data11 < (1 << 11):
        raise ValueError("CRC input must be 11 bits")
    remainder = data11 << 5  # five zero augmentation bits, init=0.
    for bit in range(15, 4, -1):
        if remainder & (1 << bit):
            remainder ^= CRC_POLY << (bit - 5)
    assert 0 <= remainder < 32
    return remainder
