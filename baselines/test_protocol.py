"""Source-level parity checks without installing torch or loading WAM weights."""
import ast
import math
from pathlib import Path
import unittest
import numpy as np
import shared_protocol as p

REPO = Path(__file__).resolve().parents[1]


def functions_from_source(path, names, globals_):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(nodes) != len(names):
        raise AssertionError('Mainline source changed; review protocol')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), globals_)
    return globals_


class ProtocolTests(unittest.TestCase):
    def test_mainline_payloads(self):
        main = functions_from_source(REPO/'experiments/robustness/run.py', {'source_for'},
                                     {'np': np, 'SEED': 20261008})
        for i in range(10):
            for seed in range(3):
                self.assertEqual(tuple(p.source_bytes(i, seed)), main['source_for'](i, seed))

    def test_mainline_all_crop_coordinates(self):
        main = functions_from_source(REPO/'src/watermarking/attacks.py', {'crop_dimensions', 'fixed_crops'},
                                     {'np': np, 'math': math, 'IMAGE_SIZE': 512, 'CROP_SEED': 20261008})
        for i in range(10):
            for seed in range(3):
                item = next(main['fixed_crops'](i, seed))
                self.assertEqual(p.crop_box(i, seed), item[3:])

    def test_all_conditions_and_metrics(self):
        a = np.full((512, 512, 3), 120, dtype=np.uint8)
        variants = list(p.attacks(a, 0, 0))
        self.assertEqual(len(variants), 14)
        self.assertEqual(variants[0][0], 'clean')
        self.assertEqual(variants[-1][1].shape, (444, 443, 3))
        self.assertEqual(p.message_metrics(np.zeros(32), np.ones(32))['payload_ber'], 1)
        self.assertIsNone(p.message_metrics(np.zeros(32), None)['payload_ber'])
        self.assertFalse(p.message_metrics(np.zeros(32), None)['global_exact'])
        self.assertAlmostEqual(p.quality(a, a)['ssim'], 1)


if __name__ == '__main__':
    unittest.main()
