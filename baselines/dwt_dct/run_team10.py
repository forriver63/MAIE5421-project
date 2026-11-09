"""Run the ten-image development check. PyCharm Run works without arguments.

Each run gets a fresh output directory. No WAM model or torch installation needed.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image, __version__ as pillow_version

from prepare_data import ROOT, prepare, sha256
from protocol import TEAM_COMMIT, attacks, message_metrics, quality, source_bits, source_bytes
from watermark import Config, SynchronizationError, embed, extract, to_uint8


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bit_image(bits):
    im = Image.fromarray(np.asarray(bits, dtype=np.uint8).reshape(4, 8)*255)
    return im.resize((256, 128), Image.Resampling.NEAREST)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/team32.json')
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--subband', choices=('HL2', 'HH2'))
    parser.add_argument('--repetitions', type=int, help='Optional repetition adapter; paper uses 1')
    args = parser.parse_args()
    cfg_dict = json.loads(args.config.read_text(encoding='utf-8'))
    for name in ('alpha', 'subband', 'repetitions'):
        if getattr(args, name) is not None:
            cfg_dict[name] = getattr(args, name)
    cfg = Config(**cfg_dict)
    cfg.validate()
    if cfg.color_mode != 'rgb_y':
        parser.error('Team10 RGB runner requires rgb_y; use demo_paper.py for gray core.')
    if not (ROOT/'data/manifest.json').exists():
        prepare()
    manifest = json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    if manifest['commit'] != TEAM_COMMIT:
        raise ValueError('Dataset manifest uses a different team source revision')
    for record in manifest['images']:
        if sha256(ROOT/'data'/record['file']) != record['sha256']:
            raise ValueError(f"Input hash mismatch: {record['name']}")
    out = args.out or ROOT/'outputs'/datetime.now().strftime('team10_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    source_hashes = {p.name: sha256(p) for p in ROOT.glob('*.py')}
    run_metadata = {
        'method': 'Al-Haj 2007 core + explicit RGB-Y/32-bit adapter',
        'status': 'development screening, not paper-table or matched-quality WAM comparison',
        'team_commit': TEAM_COMMIT, 'config': asdict(cfg), 'payload_bits': 32,
        'embedded_bit_instances': 32*cfg.repetitions, 'images': 10, 'payload_seeds': [0,1,2],
        'protocol_id': 'rgb_png32_v1', 'shared_protocol_sha256': sha256(ROOT.parent/'shared_protocol.py'), 'source_sha256': source_hashes, 'dataset': manifest,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'Pillow': pillow_version},
        'clean_policy': 'all measured attacks start from saved/reloaded 8-bit PNG; clipped float clean is diagnostic only',
        'ssim_settings': 'RGB, uniform7x7, sample covariance, K1=.01,K2=.03,range255,valid interior; NumPy implementation',
        'ber_policy': 'effective32-bit BER on decoded cases; denominator and extraction failures reported separately',
        'geometry_policy': 'rotation decoded without angle; changed dimensions rejected; no oracle crop alignment',
        'differences_from_team_historical': ['PNG round trip for all attacks vs float clean/tensor source',
                                           'NumPy float64 vs torch float32 normalization',
                                           'Pillow/NumPy versions recorded, not historical pins',
                                           'no WAM localization/packet metrics or calibrated presence detection'],
        'finished': False,
    }
    write_json(out/'run.json', run_metadata)
    cases, qualities, by_condition = [], [], defaultdict(list)
    for record in manifest['images']:
        i, name = record['image_index'], record['name']
        with Image.open(ROOT/'data'/record['file']) as im:
            image = np.array(im)
        for seed in range(3):
            case_id = f'{i:02d}_{seed}'
            case_dir = out/'cases'/case_id
            case_dir.mkdir(parents=True)
            truth = source_bits(i, seed)
            payload_hex = bytes(source_bytes(i, seed)).hex()
            start = time.perf_counter()
            floating, settings, diagnostic = embed(image, truth, cfg)
            embed_seconds = time.perf_counter()-start
            float_bits, _ = extract(np.clip(floating, 0, 255), settings)
            Image.fromarray(to_uint8(floating)).save(case_dir/'marked.png')
            write_json(case_dir/'settings.json', settings)
            with Image.open(case_dir/'marked.png') as im:
                saved = np.array(im)
            disk_settings = json.loads((case_dir/'settings.json').read_text())
            quality_row = {'case_id': case_id, 'image': name, 'payload_seed': seed,
                           'payload_hex': payload_hex, **quality(image, saved),
                           'float_clean_bit_errors': int(np.count_nonzero(float_bits != truth)),
                           'embed_seconds': embed_seconds, **diagnostic}
            qualities.append(quality_row)
            # Truth is stored in evaluation records only, never in extraction settings.
            write_json(case_dir/'evaluation_truth.json', {'payload_hex': payload_hex,
                                                         'bits': truth.tolist()})
            for condition, attacked, params in attacks(saved, i, seed):
                start = time.perf_counter()
                try:
                    recovered, info = extract(attacked, disk_settings)
                    status, reason = 'decoded', ''
                except SynchronizationError as error:
                    recovered, info = None, {}
                    status, reason = 'unsupported_geometry', str(error)
                elapsed = time.perf_counter()-start
                row = {'case_id': case_id, 'image': name, 'image_index': i, 'payload_seed': seed,
                       'payload_hex': payload_hex, 'condition': condition,
                       'alpha': cfg.alpha, 'subband': cfg.subband, 'repetitions': cfg.repetitions,
                       'useful_bits': 32, 'embedded_bit_instances': 32*cfg.repetitions,
                       'status': status, 'reason': reason, **message_metrics(truth, recovered),
                       'decoded_hex': bytes(np.packbits(recovered)).hex() if recovered is not None else '',
                       'psnr_clean': quality_row['psnr_db'], 'ssim_clean': quality_row['ssim'],
                       'decode_seconds': elapsed, 'mean_absolute_margin': info.get('mean_absolute_margin'),
                       'attack_parameters': json.dumps(params)}
                cases.append(row)
                by_condition[condition].append(row)
                if i == 0 and seed == 0:
                    samples = out/'example_astronaut'
                    samples.mkdir(exist_ok=True)
                    Image.fromarray(attacked).save(samples/f'{condition}.png')
                    if recovered is not None:
                        bit_image(recovered).save(samples/f'bits_{condition}.png')
            if i == 0 and seed == 0:
                Image.fromarray(image).save(samples/'original.png')
                Image.fromarray(to_uint8(np.abs(saved.astype(float)-image)*20)).save(samples/'difference_x20.png')
                bit_image(truth).save(samples/'bits_original.png')
            write_csv(out/'cases.csv', cases)
            write_csv(out/'quality.csv', qualities)
        clean = [r for r in cases if r['image'] == name and r['condition'] == 'clean']
        print(f"{name:24s} clean exact={sum(r['global_exact'] for r in clean)}/3", flush=True)
    summary = []
    for condition, rows in by_condition.items():
        decoded = [r for r in rows if r['status'] == 'decoded']
        errors = sum(r['bit_errors'] for r in decoded)
        denominator = len(decoded)*32
        summary.append({'condition': condition, 'cases': len(rows),
                        'exact_cases': sum(r['global_exact'] for r in rows),
                        'exact_recovery_rate': sum(r['global_exact'] for r in rows)/len(rows),
                        'decoded_cases': len(decoded), 'extraction_failures': len(rows)-len(decoded),
                        'bit_errors_decoded': errors, 'ber_denominator_bits': denominator,
                        'payload_ber_decoded': errors/denominator if denominator else None})
    write_csv(out/'summary.csv', summary)
    summary_json = {'conditions': summary,
                    'mean_psnr_db': float(np.mean([q['psnr_db'] for q in qualities])),
                    'mean_ssim': float(np.mean([q['ssim'] for q in qualities])),
                    'float_clean_exact_cases': sum(q['float_clean_bit_errors'] == 0 for q in qualities),
                    'total_cases': len(cases)}
    write_json(out/'summary.json', summary_json)
    run_metadata['finished'] = True
    write_json(out/'run.json', run_metadata)
    lines = ['# Ten-image development check', '',
             f'Config: `{asdict(cfg)}`', '',
             f"RGB mean PSNR: {summary_json['mean_psnr_db']:.4f} dB; SSIM: {summary_json['mean_ssim']:.6f}.", '',
             '| Condition | Exact / 30 | Effective-bit BER (decoded only) | Extraction failures |',
             '|---|---:|---:|---:|']
    for row in summary:
        ber = 'N/A' if row['payload_ber_decoded'] is None else f"{row['payload_ber_decoded']:.6f}"
        lines.append(f"| {row['condition']} | {row['exact_cases']}/{row['cases']} | {ber} | {row['extraction_failures']} |")
    lines += ['', 'Crop dimensions cannot be synchronized by this implementation: recorded as extraction failure, '
              'not BER=0. No angle/crop coordinates are passed to the decoder.', '',
              'This is a 32-bit/RGB adapter check with explicitly chosen parameters. No WAM comparison is claimed; '
              'payload budgets, output quantization and image quality must be agreed with the evaluation owner.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(summary_json, indent=2))
    print('Saved to:', out.resolve())


if __name__ == '__main__':
    main()
