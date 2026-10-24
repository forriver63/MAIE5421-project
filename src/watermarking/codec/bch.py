"""Primitive narrow-sense binary BCH(31, 16, 7), extended by one parity bit.

The generator is constructed from roots alpha^1,...,alpha^6 in GF(32), with
primitive field polynomial x^5 + x^2 + 1. A syndrome table corrects up to
three bit flips. The extra parity bit makes the minimum distance eight:
all patterns of at most three flips are corrected; four are detected.

Migrated verbatim from validation_wam_inner_ecc_20261008/bch31.py to preserve
bit-identical behavior.
"""
from __future__ import annotations

from itertools import combinations

N = 31
K = 16
T = 3
FIELD_POLY = 0x25


def gf_mul(a: int, b: int) -> int:
    out = 0
    while b:
        if b & 1:
            out ^= a
        b >>= 1
        a <<= 1
        if a & 0x20:
            a ^= FIELD_POLY
    return out


def gf_pow(a: int, exponent: int) -> int:
    out = 1
    for _ in range(exponent):
        out = gf_mul(out, a)
    return out


def polynomial_mul_binary(a: int, b: int) -> int:
    out = 0
    while b:
        if b & 1:
            out ^= a
        a <<= 1
        b >>= 1
    return out


def generator_polynomial() -> tuple[int, list[list[int]]]:
    visited = set()
    result = 1
    cosets = []
    for exponent in range(1, 7):
        if exponent in visited:
            continue
        coset = []
        current = exponent
        while current not in coset:
            coset.append(current)
            visited.add(current)
            current = (2 * current) % N
        assert current == exponent
        coefficients = [1]
        for power in coset:
            root = gf_pow(2, power)
            updated = [0] * (len(coefficients) + 1)
            for i, coefficient in enumerate(coefficients):
                updated[i] ^= gf_mul(coefficient, root)
                updated[i + 1] ^= coefficient
            coefficients = updated
        assert all(coefficient in (0, 1) for coefficient in coefficients)
        minimal = sum(coefficient << i for i, coefficient in enumerate(coefficients))
        result = polynomial_mul_binary(result, minimal)
        cosets.append(coset)
    assert result.bit_length() - 1 == N - K
    return result, cosets


GENERATOR, COSETS = generator_polynomial()


def remainder(value: int) -> int:
    while value.bit_length() >= GENERATOR.bit_length():
        value ^= GENERATOR << (value.bit_length() - GENERATOR.bit_length())
    return value


assert remainder((1 << N) | 1) == 0


def encode_int(useful: int) -> int:
    if not 0 <= useful < (1 << K):
        raise ValueError("useful payload must be a 16-bit integer")
    shifted = useful << (N - K)
    code31 = shifted | remainder(shifted)
    return (code31 << 1) | (code31.bit_count() & 1)


def int_to_bits(value: int) -> list[int]:
    return [(value >> bit) & 1 for bit in range(31, -1, -1)]


def bits_to_int(bits) -> int:
    if len(bits) != 32:
        raise ValueError("expected exactly 32 bits")
    out = 0
    for bit in bits:
        if int(bit) not in (0, 1):
            raise ValueError("non-binary input")
        out = (out << 1) | int(bit)
    return out


def build_syndrome_table() -> dict[int, int]:
    table = {}
    for weight in range(T + 1):
        for positions in combinations(range(N), weight):
            error = sum(1 << position for position in positions)
            syndrome = remainder(error)
            if syndrome in table:
                raise AssertionError("ambiguous <=3-bit BCH syndrome")
            table[syndrome] = error
    assert len(table) == 1 + 31 + 465 + 4495
    return table


SYNDROME_TABLE = build_syndrome_table()


def decode_int(received32: int) -> dict:
    if not 0 <= received32 < (1 << 32):
        raise ValueError("received message must be 32 bits")
    received31, received_parity = received32 >> 1, received32 & 1
    syndrome = remainder(received31)
    error = SYNDROME_TABLE.get(syndrome)
    if error is None:
        return {"status": "uncorrectable_syndrome", "payload": None,
                "corrected_bch_bits": None, "corrected_parity_bit": None,
                "corrected_total_bits": None, "syndrome": syndrome}
    corrected31 = received31 ^ error
    parity_error = (corrected31.bit_count() & 1) != received_parity
    corrected_total = error.bit_count() + int(parity_error)
    if corrected_total > T:
        return {"status": "parity_reject", "payload": None,
                "corrected_bch_bits": error.bit_count(),
                "corrected_parity_bit": bool(parity_error),
                "corrected_total_bits": corrected_total, "syndrome": syndrome}
    assert remainder(corrected31) == 0
    return {"status": "valid", "payload": corrected31 >> (N - K),
            "corrected_bch_bits": error.bit_count(),
            "corrected_parity_bit": bool(parity_error),
            "corrected_total_bits": corrected_total, "syndrome": syndrome}
