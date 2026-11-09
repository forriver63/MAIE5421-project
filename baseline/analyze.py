"""Compare complete system runs: --run full=.../full --run hidden=.../hidden."""
import argparse
from pathlib import Path
import pandas as pd
from watermarking.bench.cache import read_json, fingerprint
from watermarking.bench.manifest import payload_for
from watermarking.bench.scoring import score_payload
from watermarking.bench.attack_registry import CORE
from watermarking.bench.statistics import paired_image_bootstrap


def family(attack):
    if attack == 'crop75_jpeg75':
        return 'combined'
    if attack.startswith('rotate'):
        return 'rotation'
    return next((x for x in ['clean', 'jpeg', 'noise', 'crop'] if attack.startswith(x)), 'extended')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='Unique label=method directory')
    parser.add_argument('--reference', default='full')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows, negatives, seen = [], [], set()
    for specification in args.run:
        label, folder = specification.split('=', 1)
        if label in seen:
            raise ValueError('Repeated label')
        seen.add(label)
        folder = Path(folder)
        provenance = read_json(folder / 'provenance.json')
        identity = provenance.pop('identity')
        assert identity == fingerprint(provenance)
        count = 0
        negative_count = 0
        for path in (folder / 'cases').glob('*.json.gz'):
            case = read_json(path)
            result = next(r for r in case['results'] if r['strategy'] ==
                          ('A0_full' if case['method'] == 'full' else case['method']))
            if case['negative']:
                negatives.append(dict(variant=label, image_id=case['image_id'], accepted=result['accepted']))
                negative_count += 1
                continue
            expected = payload_for(case['image_id'], case['payload_seed'])
            assert case['expected_payload'] == expected
            score = score_payload(result, expected)
            assert all(result[k] == v for k, v in score.items())
            rows.append(dict(variant=label, image_id=case['image_id'], case_id=case['case_id'],
                             attack_id=case['attack']['attack_id'], **score, **case['quality']))
            count += 1
        assert count == 8400 and negative_count == 100, 'Requires complete positive and negative runs'
        assert not list(folder.rglob('*.error.json'))
    data = pd.DataFrame(rows)
    assert not data.duplicated(['variant', 'case_id']).any()
    assert data.groupby(['variant', 'attack_id']).size().eq(300).all()
    data['family'] = data.attack_id.map(family)
    metrics = ['exact', 'wrong', 'erasure', 'raw_payload_ber', 'psnr', 'ssim']
    perimage = data[data.attack_id.isin(CORE)].groupby(['variant', 'image_id', 'family'])[metrics].mean().groupby(['variant', 'image_id']).mean()
    summary = perimage.groupby('variant').mean().reset_index()
    differences = []
    reference = perimage.loc[args.reference, 'exact'].sort_index()
    for label in seen - {args.reference}:
        comparison = perimage.loc[label, 'exact'].sort_index()
        assert reference.index.equals(comparison.index) and len(reference) == 100
        differences.append(dict(variant=label, **paired_image_bootstrap((reference-comparison).to_numpy(), 2000, 20261106)))
    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / 'core_macro_summary.csv', index=False)
    pd.DataFrame(differences).to_csv(args.out / 'paired_differences.csv', index=False)
    data.groupby(['variant', 'attack_id'])[metrics].mean().to_csv(args.out / 'attack_summary.csv')
    pd.DataFrame(negatives).groupby('variant').accepted.agg(['size', 'sum']).to_csv(args.out / 'negative_summary.csv')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
