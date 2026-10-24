"""CRC-5 + previously verified extended BCH(31,16) WAM packet codec.

Migrated verbatim from validation_wam_hierarchical_final_20261008/packet_codec.py
to preserve bit-identical behavior.
"""
from __future__ import annotations

from .bch import bits_to_int, decode_int, encode_int, int_to_bits
from .crc import crc5


def packet16(packet_id: int, symbol: int) -> int:
    if not 0 <= packet_id < 8 or not 0 <= symbol < 256:
        raise ValueError("packet ID must be 3 bits and symbol 8 bits")
    data11 = (packet_id << 8) | symbol
    return (data11 << 5) | crc5(data11)


def encode_packet(packet_id: int, symbol: int) -> int:
    return encode_int(packet16(packet_id, symbol))


def decode_packet(word32: int) -> dict:
    bch = decode_int(word32)
    if bch["status"] != "valid":
        return {"status": "ERASURE", "reason": "bch_" + bch["status"],
                "packet_id": None, "symbol": None, "bch": bch}
    packet = bch["payload"]
    data11, check = packet >> 5, packet & 31
    if crc5(data11) != check:
        return {"status": "ERASURE", "reason": "crc_reject",
                "packet_id": None, "symbol": None, "bch": bch}
    return {"status": "ACCEPTED", "reason": None,
            "packet_id": data11 >> 8, "symbol": data11 & 255, "bch": bch}


def encode_bits(packet_id: int, symbol: int) -> list[int]:
    return int_to_bits(encode_packet(packet_id, symbol))
