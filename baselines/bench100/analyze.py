"""Validate and summarize the two complete DIV2K test runs from their exported CSVs."""
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from watermarking.bench.attack_registry import ATTACKS, CORE, EXTENDED, plan_attack
from watermarking.bench.cache import file_hash, fingerprint, read_json
from watermarking.bench.manifest import payload_for
from watermarking.bench.scoring import score_payload
from watermarking.bench.statistics import paired_image_bootstrap

METHODS = ('svd', 'dwt_dct')
FAMILIES = {
    'clean': ('clean',),
    'jpeg': tuple(a for a in CORE if a.startswith('jpeg')),
    'noise': tuple(a for a in CORE if a.startswith('noise')),
    'rotation': tuple(a for a in CORE if a.startswith('rotate')),
    'crop': tuple(a for a in CORE if a.startswith('crop') and a != 'crop75_jpeg75'),
    'combined': ('crop75_jpeg75',),
}
RATES = ('exact_rate', 'wrong_rate', 'erasure_rate', 'output_coverage', 'conditional_payload_ber')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_defined(values):
    values = [x for x in values if x is not None]
    return fmean(values) if values else None


def check_keys(rows, fields, expected, name):
    keys = Counter(tuple(r[c] for c in fields) for r in rows)
    missing = sorted(expected - keys.keys())
    extra = sorted(keys.keys() - expected)
    repeats = sum(n - 1 for n in keys.values())
    if missing or extra or repeats:
        raise ValueError(f'{name}: missing={len(missing)} {missing[:3]}, unexpected={len(extra)} '
                         f'{extra[:3]}, duplicates={repeats}')


def check_bits(bits):
    if len(bits) != 32 or set(bits) - {'0', '1'}:
        raise ValueError(f'Expected 32 binary digits, got {bits!r}')
    return int(bits, 2)


def check_plans(folder, image_ids, seeds):
    plans = read_json(folder / 'attack_plans.json.gz')
    expected = {(i, s, a, False) for i in image_ids for s in seeds for a in ATTACKS}
    expected.update((i, -1, 'clean', True) for i in image_ids)
    for p in plans:
        i, s, a = p['image_id'], p['payload_seed'], p['attack']['attack_id']
        key = (i, s, a, p['negative'])
        if key not in expected or p['attack'] != plan_attack(a, i, s):
            raise ValueError(f'Duplicate, unexpected or mismatched attack plan {key}')
        expected.remove(key)
    if expected:
        raise ValueError(f'{len(expected)} missing attack plans')
    return len(plans)


def validate(folder, manifest, config, record):
    reference = {r['image_id']: r for r in manifest if r['split'] == 'test'}
    image_ids = sorted(reference)
    seeds = config['payload_seeds']
    prov = read_json(folder / 'provenance.json')
    params = {k: v for k, v in prov.items() if k not in ('identity', 'checkout_commit')}
    if (prov['method'] != folder.name or prov['identity'] != fingerprint(params)
            or prov['conditions'] != list(ATTACKS) or prov['protocol'] != config):
        raise ValueError('Inconsistent benchmark provenance')
    if prov['images'] != [{k: reference[i][k] for k in ('image_id', 'split', 'source_sha256', 'processed_sha256')}
                          for i in image_ids]:
        raise ValueError('Image IDs, split, or hashes differ from the repository manifest')
    record.update(identity=prov['identity'], checkout_commit=prov['checkout_commit'])
    paths = [folder / f'{name}.csv' for name in ('cases', 'embeddings', 'negatives')]
    cases, embeds, negatives = map(read_csv, paths)
    record.update(rows=len(cases), embeddings=len(embeds), negatives=len(negatives),
                  csv_sha256={p.name: file_hash(p) for p in paths})
    if any(r['method'] != folder.name for rows in (cases, embeds, negatives) for r in rows):
        raise ValueError('Mixed method labels')
    pairs = {(i, str(s)) for i in image_ids for s in seeds}
    check_keys(cases, ('image_id', 'payload_seed', 'condition'),
               {(i, s, a) for i, s in pairs for a in ATTACKS}, 'positive cases')
    check_keys(embeds, ('image_id', 'payload_seed'), pairs, 'embeddings')
    check_keys(negatives, ('image_id',), {(i,) for i in image_ids}, 'clean negatives')
    record['verified_attack_plans'] = check_plans(folder, image_ids, seeds)
    settings = read_json(folder / 'extraction_settings.json.gz')
    keys = {(r['image_id'], str(r['payload_seed'])): fingerprint(r['settings']) for r in settings}
    if len(keys) != len(settings) or set(keys) != pairs or any(
            keys[r['image_id'], r['payload_seed']] != r['key_sha256'] for r in embeds):
        raise ValueError('Extraction key archive differs from the embedding table')
    seed0_keys = {r['image_id']: r['key_sha256'] for r in embeds if r['payload_seed'] == '0'}
    for r in cases:
        i, seed, attack = r['image_id'], int(r['payload_seed']), r['condition']
        expected = payload_for(i, seed)
        decoded = check_bits(r['recovered']) if r['recovered'] else None
        score = score_payload(dict(accepted=decoded is not None, payload=decoded), expected)
        ber = float(r['ber']) if r['ber'] else None
        if (expected != check_bits(r['payload']) or r['outcome'] != score['outcome']
                or ber != score['payload_ber']):
            raise ValueError(f'{i}/{seed}/{attack}: inconsistent expected bits, outcome or BER')
        if decoded is None and (r['reason'] != 'unsupported_geometry' or not attack.startswith('crop')):
            raise ValueError(f'{i}/{seed}/{attack}: unexpected erasure reason {r["reason"]!r}')
        r['ber'] = ber
    for r in embeds:
        psnr = float(r['psnr']) if r['psnr'] else None  # quality() writes None for infinite PSNR.
        ssim = float(r['ssim'])
        if (psnr is not None and not math.isfinite(psnr)) or not (-1 <= ssim <= 1):
            raise ValueError(f'Invalid quality in {r}')
        r.update(psnr=psnr, ssim=ssim)
    for r in negatives:
        if r['recovered']:
            check_bits(r['recovered'])
        if r['accepted'] or not r['key_source']:
            raise ValueError(f'Clean negative must document its key/settings and omit detector acceptance: {r}')
        if r['condition'] != 'clean' or r['payload_seed'] != '-1':
            raise ValueError(f'Unexpected negative condition/seed: {r}')
        if not r['key_sha256'] or r['key_sha256'] != seed0_keys[r['image_id']]:
            raise ValueError(f'Negative key does not match the seed-0 embedding for {r["image_id"]}')
    record.update(unique_positive_cases=len(cases), per_condition=dict(Counter(r['condition'] for r in cases)),
                  unique_embeddings=len(embeds), unique_negative_images=len(negatives),
                  negative_bit_outputs=sum(bool(r['recovered']) for r in negatives), missing=0, duplicates=0)
    return cases, embeds, negatives


def rates(rows):
    n = len(rows)
    counts = Counter(r['outcome'] for r in rows)
    known = [r['ber'] for r in rows if r['ber'] is not None]
    return dict(n=n, exact=counts['exact'], wrong=counts['wrong'], erasure=counts['erasure'],
                returned=len(known), exact_rate=counts['exact'] / n, wrong_rate=counts['wrong'] / n,
                erasure_rate=counts['erasure'] / n, output_coverage=len(known) / n,
                conditional_payload_ber=mean_defined(known))


def average_rates(records):
    return {k: mean_defined(r[k] for r in records) for k in RATES}


def summarize(method, data, image_ids):
    cases, embeddings, neg = data
    by_attack = defaultdict(list)
    by_image = defaultdict(list)
    for r in cases:
        by_attack[r['condition']].append(r)
        by_image[r['image_id'], r['condition']].append(r)
    conditions = [dict(method=method, tier='core' if a in CORE else 'extended',
                       condition=a, **rates(by_attack[a])) for a in ATTACKS]
    by_family = defaultdict(list)
    image_core = {}
    for i in image_ids:
        family_rates = []
        for family, attacks in FAMILIES.items():
            result = rates([r for a in attacks for r in by_image[i, a]])
            family_rates.append(result)
            by_family[family].append(result)
        image_core[i] = average_rates(family_rates)
    family_rows = [dict(method=method, family=f, n_conditions=len(attacks),
                        n_cases=sum(len(by_attack[a]) for a in attacks), **average_rates(by_family[f]))
                   for f, attacks in FAMILIES.items()]
    core = average_rates(list(image_core.values()))
    extended = rates([r for a in EXTENDED for r in by_attack[a]])
    quality = dict(psnr_db=mean_defined(r['psnr'] for r in embeddings),
                   ssim=fmean(r['ssim'] for r in embeddings),
                   quality_n=len(embeddings), finite_psnr_n=sum(r['psnr'] is not None for r in embeddings))
    summary = dict(method=method, **quality, **{'core_' + k: v for k, v in core.items()},
                   **{'extended_' + k: extended[k] for k in RATES},
                   negative_images=len(neg), negative_bit_outputs=sum(bool(r['recovered']) for r in neg))
    return conditions, family_rows, summary, image_core


def percent(x):
    return f'{100 * x:.2f}%' if x is not None else '—'


def report(path, methods, conditions, ci):
    lines = ['# DIV2K 100 图传统水印结果', '',
             '| 评估指标 | SVD | DWT-DCT |', '|---|---:|---:|']
    clean = [next(r for r in conditions[m] if r['condition'] == 'clean') for m in METHODS]
    def row(label, values):
        lines.append('| ' + ' | '.join([label, *values]) + ' |')

    row('Clean 完整恢复 / 300', [str(r['exact']) for r in clean])
    for label, key in [('核心完整恢复率', 'core_exact_rate'), ('核心错误消息率', 'core_wrong_rate'),
                       ('核心擦除率', 'core_erasure_rate'), ('核心有效输出覆盖率', 'core_output_coverage'),
                       ('核心有输出时的消息 BER¹', 'core_conditional_payload_ber'),
                       ('扩展完整恢复率', 'extended_exact_rate'), ('扩展错误消息率', 'extended_wrong_rate'),
                       ('扩展擦除率', 'extended_erasure_rate'), ('扩展有效输出覆盖率', 'extended_output_coverage'),
                       ('扩展有输出时的消息 BER', 'extended_conditional_payload_ber')]:
        row(label, [percent(s[key]) for s in methods])
    row('PNG 往返后平均 PSNR（dB）', [f'{s["psnr_db"]:.3f}' for s in methods])
    row('PNG 往返后平均 SSIM', [f'{s["ssim"]:.5f}' for s in methods])
    row('未嵌入对照返回 32 位 / 100', [str(s['negative_bit_outputs']) for s in methods])
    lines.extend(['', '## 28 个条件', '',
                  '| 条件 | SVD 完整恢复 / 300 | DWT-DCT 完整恢复 / 300 | SVD 覆盖率 | DWT-DCT 覆盖率 |',
                  '|---|---:|---:|---:|---:|'])
    for left, right in zip(*(conditions[m] for m in METHODS)):
        row(left['condition'], [str(left['exact']), str(right['exact']),
                                percent(left['output_coverage']), percent(right['output_coverage'])])
    diff, lo, hi = [100 * ci[k] for k in ('difference', 'lower', 'upper')]
    lines.extend(['', '## 统计口径', '',
        '核心 20 个条件先在每张图内按攻击类别平均三个消息和各档强度，再等权平均 clean、JPEG、噪声、旋转、裁剪、裁剪加 JPEG 六类，最后平均 100 张图。扩展统计取其余 8 个条件，分母为 2,400 个样本。',
        '', '¹ BER 衡量已返回的完整 32 位有效消息。逐条件以有输出的消息为分母，擦除保留在完整恢复率和覆盖率的分母中，BER 留空。核心 BER 按主线的分层平均顺序计算，跳过整组均无输出的 BER。WAM 局部载体 BER 的统计单位是定位到的编码包。',
        '', f'SVD 减去 DWT-DCT 的核心完整恢复率差为 {diff:+.2f} 个百分点，95% 配对区间为 [{lo:+.2f}, {hi:+.2f}]。共享 bootstrap 函数按图片对 100 个差值有放回重采样，重复 2,000 次，种子为 20261106。',
        '', '每张未嵌入图只做一次 clean 提取。SVD 使用同一图 seed 0 嵌入保存的块位置 key，DWT-DCT 使用相应固定提取设置。两个提取器均没有独立的存在性判决。表中对照数字表示返回比特串的数量，检测接受率和假阳性率无定义。',
        '', '详细计数和条件 BER 见 [逐条件明细](tables/conditions.csv)，各攻击类别的均值见 [核心分类汇总](tables/core_families.csv)，完整性和原始 CSV 的 SHA-256 见 [核验记录](integrity.json)。'])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Runner output containing svd/ and dwt_dct/')
    parser.add_argument('--cache', type=Path, help='Scan original checkpoint directory for execution errors')
    args = parser.parse_args()
    root = args.output.resolve()
    config = read_json(REPO / 'configs/protocol.example.json')
    manifest = read_json(REPO / 'baseline/image_manifest.json')
    ids = sorted(r['image_id'] for r in manifest if r['split'] == 'test')
    seeds = config['payload_seeds']
    if ids != [f'DIV2K_{i:04d}' for i in range(801, 901)] or seeds != [0, 1, 2]:
        raise ValueError('Unexpected test manifest or payload seeds')
    if sorted(CORE) != sorted(a for attacks in FAMILIES.values() for a in attacks):
        raise ValueError('Unexpected core attack family membership')
    integrity = dict(status='incomplete', expected_per_method=dict(images=len(ids), positives=8400,
                     embeddings=300, per_condition=300, negatives=100), methods={})
    records = {}
    errors = [str(p.relative_to(root)) for p in sorted(root.rglob('*.error.json'))]
    if args.cache:
        if not args.cache.is_dir():
            raise ValueError(f'Checkpoint directory missing: {args.cache}')
        errors.extend('cache/' + p.relative_to(args.cache).as_posix()
                      for p in sorted(args.cache.rglob('*.error.json')))
    integrity.update(checkpoint_error_scan=bool(args.cache), analysis_sha256=file_hash(Path(__file__)))
    integrity['execution_errors'] = errors
    for method in METHODS:
        check = integrity['methods'][method] = {}
        try:
            records[method] = validate(root / method, manifest, config, check)
        except (ValueError, KeyError, OSError) as exc:
            check['error'] = str(exc)
    complete = len(records) == len(METHODS) and not errors
    if complete:
        integrity['status'] = 'complete'
    (root / 'integrity.json').write_text(json.dumps(integrity, indent=2) + '\n', encoding='utf-8')
    if not complete:
        raise ValueError(f'Incomplete run; see {root / "integrity.json"}')
    tables = root / 'tables'
    tables.mkdir(exist_ok=True)
    conditions, families, summary, images = {}, [], [], {}
    for m in METHODS:
        conditions[m], f, s, images[m] = summarize(m, records[m], ids)
        families.extend(f)
        summary.append(s)
    write_csv(tables / 'conditions.csv', [r for m in METHODS for r in conditions[m]])
    comparison = []
    for pair in zip(*(conditions[m] for m in METHODS)):
        row = {k: pair[0][k] for k in ('tier', 'condition')}
        for result in pair:
            row.update({result['method'] + '_' + k: v for k, v in result.items()
                        if k not in ('tier', 'condition', 'method')})
        comparison.append(row)
    write_csv(tables / 'comparison.csv', comparison)
    write_csv(tables / 'core_families.csv', families)
    write_csv(tables / 'overview.csv', summary)
    ci = paired_image_bootstrap([images['svd'][i]['exact_rate'] - images['dwt_dct'][i]['exact_rate']
                                for i in ids], config['bootstrap_replicates'], config['bootstrap_seed'])
    write_csv(tables / 'paired_core_exact.csv',
              [dict(comparison='svd_minus_dwt_dct', **ci, replicates=config['bootstrap_replicates'],
                    seed=config['bootstrap_seed'])])
    report(root / 'REPORT_ZH.md', summary, conditions, ci)
    print(f'Validated {len(records)} complete methods. Report: {root / "REPORT_ZH.md"}')


if __name__ == '__main__':
    main()
