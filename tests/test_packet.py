"""Bit-identical packet codec verification (migrated from test_packet_codec.py)."""
import math
from itertools import combinations

from watermarking.codec.packet import decode_packet, encode_packet, packet16
from watermarking.codec.crc import crc5


def test_zero_error_roundtrip_all_inputs():
    for packet_id in range(8):
        for symbol in range(256):
            word = encode_packet(packet_id, symbol)
            result = decode_packet(word)
            assert result["status"] == "ACCEPTED"
            assert (result["packet_id"], result["symbol"]) == (packet_id, symbol)
            assert packet16(packet_id, symbol) >> 5 == (packet_id << 8) | symbol
    assert crc5(0) == 0


def test_exhaustive_bit_error_patterns():
    for weight in range(7):
        exact = erasure = silent_wrong = 0
        for positions in combinations(range(32), weight):
            error = sum(1 << bit for bit in positions)
            result = decode_packet(error)
            if result["status"] == "ERASURE":
                erasure += 1
            elif (result["packet_id"], result["symbol"]) == (0, 0):
                exact += 1
            else:
                silent_wrong += 1
        assert exact + erasure + silent_wrong == math.comb(32, weight)
        if weight <= 3:
            assert exact == math.comb(32, weight)
        if weight == 4:
            assert erasure == math.comb(32, weight)
