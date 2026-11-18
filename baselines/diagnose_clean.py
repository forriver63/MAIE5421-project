"""Locate clean errors without changing production algorithms or parameters.

Run from the repository: python baselines/diagnose_clean.py --method svd
Raw-float decoding is a mathematical diagnostic: it permits out-of-range values.
Clipped and PNG stages use the production extractor. No parameters are tuned.
"""
import argparse
import csv
from datetime import datetime
import importlib
import io
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image


def raw_decode(module, method, image, settings):
    y = np.asarray(image, float) @ np.array([.299, .587, .114])
    cfg = module.Config(**settings['config'])
    if method == 'svd':
        blocks = module.to_blocks(y, cfg.block_size)
        norms = np.linalg.norm(blocks[settings['positions']], ord=2, axis=(-2, -1))
        return (np.ceil(norms/cfg.delta).astype(np.int64) % 2).astype(np.uint8)
    _, _, coefficients, _ = module.analyze_plane(y, cfg.subband)
    n = settings['payload_length']
    positions = module.block_positions(len(coefficients), n, cfg.repetitions)
    u, v = module.MID_POSITIONS.T
    values = coefficients[positions[:, None], u[None, :], v[None, :]]
    centered = values-values.mean(axis=1, keepdims=True)
    pn = module.pn_sequences(cfg.pn_seed)
    denom = np.linalg.norm(centered, axis=1)*np.linalg.norm(pn[0])
    scores = np.divide(centered @ pn.T, denom[:, None],
                       out=np.zeros((len(positions), 2)), where=denom[:, None] > 1e-12)
    scores = scores.reshape(cfg.repetitions, n, 2).mean(axis=0)
    return (scores[:, 1] >= scores[:, 0]).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['svd', 'dwt_dct'], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent/args.method
    sys.path.insert(0, str(root))
    module = importlib.import_module('rgb_adapter' if args.method == 'svd' else 'watermark')
    protocol = importlib.import_module('protocol')
    prepare = importlib.import_module('prepare_data')
    manifest = json.loads((root/'data/manifest.json').read_text())
    config = json.loads((root/'configs/team32.json').read_text())
    rows = []
    for rec in manifest['images']:
        path = root/'data'/rec['file']
        if prepare.sha256(path) != rec['sha256']:
            raise ValueError(f'Input hash mismatch: {path}')
        with Image.open(path) as im:
            original = np.array(im.convert('RGB'))
        for seed in range(3):
            truth = protocol.source_bits(rec['image_index'], seed)
            marked, settings, diagnostic = module.embed(original, truth, module.Config(**config))
            clipped = np.clip(marked, 0, 255)
            quantized = module.to_uint8(marked)
            buffer = io.BytesIO()
            Image.fromarray(quantized).save(buffer, format='PNG')
            buffer.seek(0)
            with Image.open(buffer) as im:
                reloaded = np.array(im)
            np.testing.assert_array_equal(quantized, reloaded)
            row = {'image': rec['name'], 'payload_seed': seed,
                   'payload_hex': bytes(protocol.source_bytes(rec['image_index'], seed)).hex()}
            for stage, image in [('raw_float', marked), ('clipped_float', clipped), ('png_uint8', reloaded)]:
                decoded = raw_decode(module, args.method, image, settings)
                if stage != 'raw_float':
                    actual, _ = module.extract(image, settings)
                    np.testing.assert_array_equal(decoded, actual)
                row[stage+'_errors'] = int(np.count_nonzero(decoded != truth))
                row[stage+'_wrong_indices'] = ' '.join(map(str, np.flatnonzero(decoded != truth)))
            row['channel_values_outside_range_gt_1e_8'] = int(((marked < -1e-8) | (marked > 255+1e-8)).sum())
            row['diagnostic'] = json.dumps(diagnostic)
            rows.append(row)
    out = root/'outputs'/datetime.now().strftime('clean_diagnostic_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    with (out/'cases.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {'method': args.method, 'config': config, 'cases': len(rows), 'stages': {}}
    print(f'\nMethod: {args.method}; {len(rows)} messages, 32 bits each')
    for stage in ['raw_float', 'clipped_float', 'png_uint8']:
        errors = sum(r[stage+'_errors'] for r in rows)
        exact = sum(r[stage+'_errors'] == 0 for r in rows)
        summary['stages'][stage] = {'exact': exact, 'errors': errors, 'total_bits': len(rows)*32}
        print(f'{stage:16s} exact={exact}/{len(rows)} errors={errors}/{len(rows)*32}')
    summary['note'] = 'Raw floats may be invalid display pixels. Errors can be corrected as well as introduced between stages; inspect per-bit indices. No cause is established by stage counts alone.'
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Saved:', out)


if __name__ == '__main__':
    main()
