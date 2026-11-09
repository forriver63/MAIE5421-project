import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .cache import read_json, write_json
from .statistics import paired_image_bootstrap
from .validation import validate_run


def finalize(run_dir, out, allow_partial=False):
    run_dir, out = Path(run_dir), Path(out)
    validate_run(run_dir, allow_partial)
    out.mkdir(parents=True, exist_ok=True)
    rows, clusters, regions, negatives, missing = [], [], [], [], []
    for model_dir in sorted(run_dir.iterdir()):
        if not (model_dir / 'manifest').is_dir():
            continue
        for manifest in sorted((model_dir / 'manifest').glob('*.json')):
            for planned in read_json(manifest):
                path = model_dir / 'cases' / f"{planned['case_id']}.json.gz"
                if not path.exists():
                    missing.append(str(path))
                    continue
                c = read_json(path)
                base = dict(method=c['method'], image_id=c['image_id'], case_id=c['case_id'],
                            payload_seed=c['payload_seed'], attack_id=c['attack']['attack_id'],
                            split=c['split'], extract_seconds=c['extract_seconds'])
                for r in c['results']:
                    row = dict(**base, strategy=r['strategy'], accepted=r['accepted'],
                               status=r['status'], payload=r.get('payload'))
                    if c['negative']:
                        row['local_packets_passed'] = sum(p['status'] == 'ACCEPTED' for p in r.get('packets', []))
                        row['local_packets_selected'] = sum(p.get('selected_for_layout', p['status'] == 'ACCEPTED') for p in r.get('packets', []))
                        negatives.append(row)
                        continue
                    for key in ('outcome', 'exact', 'wrong', 'erasure', 'payload_ber', 'raw_payload_exact',
                                'raw_payload_ber', 'nc', 'wrong_accepted_clusters', 'unmatched_accepted_clusters'):
                        row[key] = r.get(key)
                    row.update(c['quality'])
                    row['embed_seconds'] = c['embed_seconds']
                    rows.append(row)
                    for i, packet in enumerate(r.get('packets', [])):
                        clusters.append(dict(**base, strategy=r['strategy'], cluster_index=i,
                            status=packet['status'], reason=packet['reason'],
                            passes_checks=packet.get('passes_checks', packet['status'] == 'ACCEPTED'),
                            selected_for_layout=packet.get('selected_for_layout', packet['status'] == 'ACCEPTED'),
                            packet_id=packet['packet_id'], symbol=packet['symbol']))
                    for region in r.get('regions', []):
                        regions.append(dict(**base, strategy=r['strategy'], **region))
    errors = list(run_dir.rglob('*.error.json'))
    if (missing or errors) and not allow_partial:
        raise RuntimeError(f'Incomplete: {len(missing)} missing cases, {len(errors)} execution errors')
    if not rows:
        raise ValueError('No positive results')
    results = out / 'results'
    results.mkdir(exist_ok=True)
    tables = out / 'tables'
    tables.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    if df.duplicated(['method', 'strategy', 'case_id']).any():
        raise ValueError('Duplicate result rows')
    assert ((df.exact + df.wrong + df.erasure) == 1).all()
    df.to_csv(results / 'cases.csv', index=False)
    pd.DataFrame(clusters).to_csv(results / 'clusters.csv', index=False)
    pd.DataFrame(regions).to_csv(results / 'regions.csv', index=False)
    pd.DataFrame(negatives).to_csv(results / 'negatives.csv', index=False)
    # Layout is an analysis dimension, not a source of independent images.
    main = df.copy()
    main['group'] = main.strategy.where(~main.strategy.str.startswith('A3_single'), 'A3_single_mean')
    metrics = ['exact', 'wrong', 'erasure', 'payload_ber', 'raw_payload_exact', 'raw_payload_ber',
               'psnr', 'ssim', 'extract_seconds', 'wrong_accepted_clusters', 'unmatched_accepted_clusters']
    grouped = main.groupby(['method', 'group', 'image_id', 'case_id', 'attack_id'], as_index=False)[metrics].mean()
    summary = grouped.groupby(['method', 'group', 'attack_id']).agg(
        n=('case_id', 'size'), images=('image_id', 'nunique'), exact_equivalent_count=('exact', 'sum'),
        exact_rate=('exact', 'mean'), wrong_rate=('wrong', 'mean'), erasure_rate=('erasure', 'mean'),
        conditional_ber=('payload_ber', 'mean'), raw_exact_rate=('raw_payload_exact', 'mean'),
        raw_ber=('raw_payload_ber', 'mean'), mean_psnr=('psnr', 'mean'), mean_ssim=('ssim', 'mean'),
        extract_seconds=('extract_seconds', 'median')).reset_index()
    coverage = main.groupby(['method', 'group', 'attack_id']).agg(
        accepted_case_layouts=('accepted', 'sum'), case_layouts=('accepted', 'size'),
        conditional_ber=('payload_ber', 'mean')).reset_index()
    summary = summary.drop(columns='conditional_ber').merge(coverage, on=['method', 'group', 'attack_id'])
    summary['output_coverage'] = summary.accepted_case_layouts / summary.case_layouts
    summary.to_csv(tables / 'attack_summary.csv', index=False)
    if negatives:
        neg = pd.DataFrame(negatives)
        ns = neg.groupby(['method', 'strategy']).agg(n=('image_id', 'size'),
            images=('image_id', 'nunique'), accepted_count=('accepted', 'sum'),
            local_packets_passed=('local_packets_passed', 'sum'),
            local_packets_selected=('local_packets_selected', 'sum')).reset_index()
        ns['accepted_rate'] = ns.accepted_count / ns.n
        reuse = ns[(ns.method == 'full') & (ns.strategy == 'A1_bch_off')].copy()
        reuse['method'], reuse['strategy'] = 'e1_reused_full_negatives', 'E1_no_bch'
        pd.concat([ns, reuse]).to_csv(tables / 'negative_summary.csv', index=False)
    def family(name):
        for prefix, group in [('jpeg', 'JPEG'), ('noise', 'noise'), ('rotate', 'rotation')]:
            if name.startswith(prefix):
                return group
        if name == 'crop75_jpeg75':
            return 'combined'
        if name.startswith('crop'):
            return 'crop'
        return 'clean' if name == 'clean' else 'extended'
    summary['family'] = summary.attack_id.map(family)
    family_means = summary[summary.family != 'extended'].groupby(['method', 'group', 'family'])[
        ['exact_rate', 'wrong_rate', 'erasure_rate']].mean().reset_index()
    family_means.to_csv(tables / 'core_family_summary.csv', index=False)
    macro = family_means.groupby(['method', 'group']).agg(
        represented_families=('family', 'nunique'), core_macro_exact=('exact_rate', 'mean'),
        core_macro_wrong=('wrong_rate', 'mean'), core_macro_erasure=('erasure_rate', 'mean')).reset_index()
    macro.to_csv(tables / 'core_macro_summary.csv', index=False)
    layout = df[df.strategy.str.startswith('A3_single')].groupby(['strategy', 'attack_id']).agg(
        n=('exact', 'size'), exact_rate=('exact', 'mean')).reset_index()
    layout.to_csv(tables / 'single_layouts.csv', index=False)
    layout.groupby('attack_id').exact_rate.agg(['mean', 'min', 'max']).to_csv(tables / 'single_layout_range.csv')
    # Bootstrap image-level paired differences, averaging all messages and layout instances first.
    ci_rows = []
    by_image = grouped.groupby(['method', 'group', 'attack_id', 'image_id']).exact.mean().reset_index()
    base = by_image[(by_image.method == 'full') & (by_image.group == 'A0_full')]
    for (method, group, attack), subset in by_image.groupby(['method', 'group', 'attack_id']):
        if method == 'full' and group == 'A0_full':
            continue
        paired = base[base.attack_id == attack].merge(subset, on='image_id', suffixes=('_full', '_variant'))
        if not paired.empty:
            ci_rows.append(dict(method=method, group=group, attack_id=attack, **paired_image_bootstrap(
                paired.exact_full - paired.exact_variant)))
    pd.DataFrame(ci_rows).to_csv(tables / 'paired_differences.csv', index=False)
    figdir = out / 'figures'
    figdir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for (method, group), sub in summary.groupby(['method', 'group']):
        label = f'{method}/{group}'
        families = [('JPEG quality', [('jpeg50', 50), ('jpeg75', 75)]),
                    ('Retained crop area', [('crop50_' + str(i), .5) for i in range(3)] +
                     [('crop75_' + str(i), .75) for i in range(3)]),
                    ('Rotation (degrees)', [(f'rotate{a:+}', a) for a in (-10, -5, -2, -1, 1, 2, 5, 10)])]
        for ax, (title, keys) in zip(axes, families):
            samples = [(x, float(sub[sub.attack_id == name].exact_rate.iloc[0]))
                       for name, x in keys if name in set(sub.attack_id)]
            if samples:
                s = pd.DataFrame(samples, columns=['x', 'y']).groupby('x').y.mean()
                ax.plot(s.index, s.values, marker='o', label=label)
            ax.set(xlabel=title, ylabel='Exact 32-bit recovery', ylim=(-.03, 1.03))
            ax.grid(alpha=.25)
    axes[-1].legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(figdir / 'robustness_curves.png', dpi=180)
    plt.close(fig)
    # CRC must be assessed using both incorrect acceptance and rejection.
    crc = summary[(summary.method == 'full') & summary.group.isin(['A0_full', 'A2_crc_off'])]
    crc.to_csv(tables / 'crc_tradeoff.csv', index=False)
    if not crc.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        for group, sub in crc.groupby('group'):
            x = np.arange(len(sub))
            for ax, metric in zip(axes, ['wrong_rate', 'erasure_rate']):
                ax.plot(x, sub[metric], marker='.', label=group)
                ax.set_xticks(x, sub.attack_id, rotation=90, fontsize=6)
                ax.set_ylabel(metric)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figdir / 'crc_tradeoff.png', dpi=180)
        plt.close(fig)
    completeness = dict(status='complete_for_declared_manifests' if not missing and not errors else 'partial',
                        missing_cases=len(missing), execution_errors=len(errors), positive_result_rows=len(df),
                        note='This status checks declared manifests, not completion of every planned project stage.')
    write_json(out / 'summary.json', dict(**completeness, attack_summaries=summary.replace({np.nan: None}).to_dict('records')))
    print(completeness)
    return df, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--allow-partial', action='store_true')
    a = p.parse_args()
    finalize(a.run_dir, a.out, a.allow_partial)


if __name__ == '__main__':
    main()
