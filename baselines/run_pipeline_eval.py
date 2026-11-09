"""One RGB-PNG evaluation entry for svd, dwt_dct, and the real WAM pipeline.

One method per process avoids same-named legacy module collisions.
WAM needs the main repository environment, upstream checkout and checkpoint.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import importlib
import json
import csv
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
from PIL import Image, __version__ as pillow_version
from shared_protocol import attacks, source_bits, source_bytes, quality, message_metrics, TEAM_COMMIT

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=['svd', 'dwt_dct', 'wam'])
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    # Always validate identical pinned source images, even on a new branch commit.
    method_dir = ROOT/('svd' if args.method == 'wam' else args.method)
    sys.path.insert(0, str(method_dir))
    prep = importlib.import_module('prepare_data')
    prep.prepare()
    manifest = json.loads((method_dir/'data/manifest.json').read_text())
    source_files = [Path(__file__), ROOT/'shared_protocol.py', method_dir/'prepare_data.py']
    if args.method == 'wam':
        sys.path.insert(0, str(REPO/'src'))
        from watermarking.pipeline import embed_image, detect_image
        from watermarking.wam.model import load_default_model
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(20261008)
        model = load_default_model()
        config = {'pipeline': 'WAM+BCH+CRC+8-region repetition', 'useful_bits': 32, 'encoded_bits': 256}
        source_files += list((REPO/'src/watermarking').rglob('*.py'))
        def encode(image, bits):
            payload = int.from_bytes(bytes(np.packbits(bits)), 'big')
            return np.array(embed_image(Image.fromarray(image), payload, wam=model)), {}
        def decode(image, key):
            result = detect_image(Image.fromarray(image), wam=model)
            if result['status'] != 'decoded' or result['payload'] is None:
                return None, 'no_output'
            value = int(result['payload']).to_bytes(4, 'big')
            return np.unpackbits(np.frombuffer(value, dtype=np.uint8)), 'decoded'
    else:
        module = importlib.import_module('rgb_adapter' if args.method == 'svd' else 'watermark')
        config = json.loads((method_dir/'configs/team32.json').read_text())
        source_files += list(method_dir.glob('*.py'))+[method_dir/'configs/team32.json']
        def encode(image, bits):
            floating, key, _ = module.embed(image, bits, module.Config(**config))
            return module.to_uint8(floating), key
        def decode(image, key):
            try:
                bits, _ = module.extract(image, key)
                return bits, 'decoded'
            except module.SynchronizationError:
                return None, 'unsupported_geometry'
    out = args.out or ROOT/'outputs'/f'{args.method}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True, exist_ok=False)
    metadata = {'protocol_id': 'rgb_png32_v1', 'method': args.method, 'config': config,
                'git_head': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
                'dataset_commit': TEAM_COMMIT, 'dataset': manifest,
                'source_hashes': {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                'python': platform.python_version(), 'numpy': np.__version__, 'Pillow': pillow_version,
                'quality': 'RGB after PNG roundtrip; shared uniform7 SSIM/sample covariance',
                'ber': 'effective32-bit decoded messages only; no-output counts separately',
                'limitations': ['not matched distortion or redundancy budget', 'no calibrated watermark-presence test for traditional methods',
                               'not interchangeable with historical float-clean WAM results'], 'finished': False}
    if args.method == 'wam':
        metadata['torch'] = torch.__version__
    save_json(out/'run.json', metadata)
    rows, quality_rows = [], []
    for record in manifest['images']:
        i, name = record['image_index'], record['name']
        with Image.open(method_dir/'data'/record['file']) as im:
            original = np.array(im.convert('RGB'))
        for seed in range(3):
            truth = source_bits(i, seed)
            start = time.perf_counter()
            marked, key = encode(original, truth)
            embed_seconds = time.perf_counter()-start
            case = f'{i:02d}_{seed}'
            folder = out/'images'/case
            folder.mkdir(parents=True)
            Image.fromarray(marked).save(folder/'marked.png')
            save_json(folder/'settings.json', key)
            with Image.open(folder/'marked.png') as im:
                saved = np.array(im.convert('RGB'))
            q = quality(original, saved)
            quality_rows.append({'case_id': case, 'image': name, 'payload_seed': seed, **q,
                                 'embed_seconds': embed_seconds})
            for condition, received, params in attacks(saved, i, seed):
                start = time.perf_counter()
                bits, status = decode(received, key)
                seconds = time.perf_counter()-start
                metric = message_metrics(truth, bits)
                rows.append({'method': args.method, 'case_id': case, 'image': name, 'payload_seed': seed,
                             'payload_hex': bytes(source_bytes(i, seed)).hex(), 'condition': condition,
                             'status': status, 'has_output': bits is not None,
                             'wrong_global_output': bits is not None and not metric['global_exact'], **metric,
                             'decoded_hex': bytes(np.packbits(bits)).hex() if bits is not None else '',
                             'psnr_clean': q['psnr_db'], 'ssim_clean': q['ssim'],
                             'embed_seconds': embed_seconds, 'decode_seconds': seconds,
                             'attack_parameters': json.dumps(params)})
        print(f'{args.method}: {name} complete', flush=True)
        save_csv(out/'cases.csv', rows)
        save_csv(out/'quality.csv', quality_rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row['condition']].append(row)
    summary = []
    for condition, group in groups.items():
        valid = [r for r in group if r['has_output']]
        denominator = 32*len(valid)
        errors = sum(r['bit_errors'] for r in valid)
        summary.append({'condition': condition, 'cases': len(group),
                        'exact_cases': sum(r['global_exact'] for r in group),
                        'exact_recovery_rate': sum(r['global_exact'] for r in group)/len(group),
                        'wrong_output_cases': sum(r['wrong_global_output'] for r in group),
                        'no_output_cases': len(group)-len(valid), 'ber_denominator_bits': denominator,
                        'bit_errors_decoded': errors, 'payload_ber_decoded': errors/denominator if denominator else None})
    save_csv(out/'summary.csv', summary)
    save_json(out/'summary.json', {'conditions': summary, 'mean_psnr_db': float(np.mean([q['psnr_db'] for q in quality_rows])),
                                  'mean_ssim': float(np.mean([q['ssim'] for q in quality_rows]))})
    metadata['finished'] = True
    save_json(out/'run.json', metadata)
    print('Saved:', out.resolve())


if __name__ == '__main__':
    main()
