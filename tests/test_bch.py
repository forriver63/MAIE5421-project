"""Bit-identical BCH codec verification (migrated from test_bch31.py)."""
from itertools import combinations

from watermarking.codec.bch import COSETS, GENERATOR, bits_to_int, decode_int, encode_int, int_to_bits


def test_cosets_and_distance():
    assert COSETS == [[1, 2, 4, 8, 16], [3, 6, 12, 24, 17], [5, 10, 20, 9, 18]]
    min_distance_31 = min((encode_int(message) >> 1).bit_count() for message in range(1, 1 << 16))
    min_distance = min(encode_int(message).bit_count() for message in range(1, 1 << 16))
    assert min_distance_31 == 7
    assert min_distance == 8


def test_all_messages_roundtrip():
    for message in range(1 << 16):
        encoded = encode_int(message)
        assert bits_to_int(int_to_bits(encoded)) == encoded
        decoded = decode_int(encoded)
        assert decoded["status"] == "valid" and decoded["payload"] == message
        assert decoded["corrected_total_bits"] == 0


def test_exhaustive_error_patterns_on_zero_codeword():
    for weight in (1, 2, 3, 4):
        for positions in combinations(range(32), weight):
            error = sum(1 << position for position in positions)
            decoded = decode_int(error)
            if weight <= 3:
                assert decoded["status"] == "valid" and decoded["payload"] == 0
                assert decoded["corrected_total_bits"] == weight
            else:
                assert decoded["status"] != "valid"
