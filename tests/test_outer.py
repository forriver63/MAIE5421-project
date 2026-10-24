"""Bit-identical outer code verification (migrated from test_outer_codes.py)."""
import random

from watermarking.codec.outer import (
    collect_accepted_packets,
    decode_mds,
    decode_repetition,
    encode_mds,
    encode_repetition,
    exhaustive_mds_verification,
    REPETITION_SOURCE_BY_REGION,
)


def _sources():
    rng = random.Random(20261008)
    sources = [(0, 0, 0, 0), (255, 255, 255, 255), (1, 2, 3, 4)]
    sources += [tuple(rng.randrange(256) for _ in range(4)) for _ in range(256)]
    return sources


def test_exhaustive_mds_verification():
    result = exhaustive_mds_verification(_sources())
    assert result["subsets"] == 70 and result["all_exact"]


def test_encode_decode_roundtrip():
    for source in _sources():
        mds = encode_mds(source)
        repetition = encode_repetition(source)
        assert decode_mds(dict(enumerate(mds))) == {"status": "decoded", "payload": source}
        assert decode_repetition(dict(enumerate(repetition))) == {"status": "decoded", "payload": source}


def test_repetition_conflict_and_collect():
    source = (11, 22, 33, 44)
    repeated = encode_repetition(source)
    assert decode_repetition({0: repeated[0], 7: repeated[7], 1: repeated[1], 6: repeated[6],
                              2: repeated[2], 5: repeated[5], 3: repeated[3], 4: repeated[4]})["payload"] == source
    assert decode_repetition({0: 11, 7: 12, 1: 22, 2: 33, 3: 44})["status"] == "repetition_conflict"
    assert REPETITION_SOURCE_BY_REGION == (0, 1, 2, 3, 3, 2, 1, 0)
    unique, info = collect_accepted_packets([
        {"status": "ACCEPTED", "packet_id": 0, "symbol": 11},
        {"status": "ACCEPTED", "packet_id": 0, "symbol": 12},
        {"status": "ACCEPTED", "packet_id": 1, "symbol": 22},
        {"status": "ERASURE", "packet_id": None, "symbol": None},
    ])
    assert unique == {1: 22} and info["conflicting_ids"] == [0]
