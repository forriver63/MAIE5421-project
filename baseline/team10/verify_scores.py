"""Independently verify the shared 32-bit scores without neural inference."""
from pathlib import Path
import csv, gzip, json
from collections import defaultdict
import numpy as np
import bchlib


def verify():
    root = Path(__file__).resolve().parent
    bch = bchlib.BCH(5, 137)
    counts = defaultdict(lambda: [0, 0])
    totals = defaultdict(lambda: [0, 0, 0, 0, 0., 0.])
    keys = set()
    variants = {'full_png', 'hidden045_bits', 'redmark_A', 'redmark_B',
                'rosteals_A', 'rosteals_B', 'sepmark_B'}
    with gzip.open(root / 'cases.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            v, i, seed = r['variant'], r['image_index'], r['seed']
            assert v in variants and i in range(10) and seed in range(3)
            key = (v, i, seed, r['condition'])
            assert key not in keys
            keys.add(key)
            payload = np.random.default_rng(20261008 + 70000 + i*1009 + seed*97).integers(0, 256, 4, dtype=np.uint8)
            truth = np.unpackbits(payload)
            expected = int.from_bytes(payload.tobytes(), 'big')
            if v == 'full_png':
                assert r['expected_payload'] == expected
                accepted = r['accepted']
                exact = accepted and r['recovered_payload'] == expected
            elif v == 'hidden045_bits':
                assert r['expected_payload'] == expected
                bits = np.asarray(r['raw']['logits']) > 0
                assert bits.shape == (48,)
                accepted = True
                exact = np.array_equal(bits[16:], truth)
                marker = np.unpackbits(np.array([0xA5, 0xC3], dtype=np.uint8))
                assert r['marker_gated_exact'] == bool(exact and np.count_nonzero(bits[:16] != marker) <= 1)
            elif v.startswith('rosteals'):
                pad = np.zeros(24, dtype=np.uint8) if v.endswith('_A') else np.random.default_rng([20261008, i, seed, 1]).integers(0, 2, 24)
                truth56 = np.r_[truth, pad].astype(np.uint8)
                np.testing.assert_array_equal(r['expected_bits56'], truth56)
                data = bytearray(np.packbits(truth56).tobytes())
                packet = np.r_[np.unpackbits(np.frombuffer(data + bch.encode(data), dtype=np.uint8)), np.zeros(4, dtype=np.uint8)]
                np.testing.assert_array_equal(r['packet100'], packet)
                scores = np.asarray(r['scores'])
                assert scores.shape == (100,) and r['threshold'] == 0
                received = np.packbits(scores[:96] > 0).tobytes()
                data, ecc = bytearray(received[:7]), bytearray(received[7:])
                flips = bch.decode(data, ecc)
                assert flips == r['corrected_flips']
                accepted = flips >= 0
                if accepted:
                    bch.correct(data, ecc)
                exact = accepted and data[:4] == payload.tobytes()
            else:
                capacity = 1024 if v.startswith('redmark') else 128
                pad = np.zeros(capacity-32, dtype=np.uint8) if v.endswith('_A') else np.random.default_rng([20261008, i, seed, 1]).integers(0, 2, capacity-32)
                np.testing.assert_array_equal(r['expected_bits'], np.r_[truth, pad])
                scores = np.asarray(r['scores'])
                threshold = .5 if v.startswith('redmark') else 0
                assert scores.shape == (capacity,) and r['threshold'] == threshold
                exact = np.array_equal(scores[:32] > threshold, truth)
                accepted = True
            assert r['exact'] == bool(exact)
            assert r['wrong'] == bool(accepted and not exact)
            assert r['erasure'] == (not accepted)
            c = counts[v, r['condition']]
            c[0] += int(exact)
            c[1] += 1
            t = totals[v]
            for k, value in enumerate([int(exact), 1, int(r['wrong']), int(r['erasure']), r['psnr'], r['ssim']]):
                t[k] += value
    assert len(keys) == 2940 and len(counts) == 98
    with (root / 'conditions.csv').open(newline='') as f:
        table = list(csv.DictReader(f))
    assert len(table) == 98
    for row in table:
        exact, total = counts[row['variant'], row['condition']]
        assert [exact, total] == [int(row['exact']), int(row['cases'])]
        assert total == 30 and abs(exact/total-float(row['recovery'])) < 1e-12
    with (root / 'overall.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            a, n, w, e, p, s = totals[row['variant']]
            assert [a, n] == [int(row['exact']), int(row['cases'])]
            for field, value in [('recovery', a/n), ('wrong', w/n), ('erasure', e/n), ('psnr', p/n), ('ssim', s/n)]:
                assert abs(value-float(row[field])) < 1e-9, (row['variant'], field)
    print('Verified 2940 unique cases, deterministic messages, all 98 conditions and overall metrics.')


if __name__ == '__main__':
    verify()
