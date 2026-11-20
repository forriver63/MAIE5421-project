"""Recompute the five reported metrics from the released case records."""
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parent
    cases = pd.read_csv(root / 'cases.csv.gz', dtype=str)
    keys = ['variant', 'condition']
    outcomes = ['exact', 'wrong', 'erasure']
    metrics = outcomes + ['psnr', 'ssim']
    assert not cases.duplicated(keys + ['image_id', 'payload_seed']).any()
    for column in outcomes:
        cases[column] = cases[column].str.lower().map({'true': 1, 'false': 0, '1': 1, '0': 0})
    assert cases[outcomes].notna().all().all() and cases[outcomes].sum(axis=1).eq(1).all()
    cases[['psnr', 'ssim']] = cases[['psnr', 'ssim']].astype(float)
    grouped = cases.groupby(keys)
    assert grouped.size().eq(300).all() and grouped.image_id.nunique().eq(100).all()
    actual = grouped[metrics].mean().sort_index()
    published = pd.read_csv(root / 'attack_summary.csv').set_index(keys).sort_index()
    assert actual.index.equals(published.index)
    np.testing.assert_allclose(actual, published[metrics], rtol=0, atol=1e-10)
    for file, expected in [('core20_metrics.csv', 160), ('hidden_gated_supplement.csv', 20)]:
        table = pd.read_csv(root / file).set_index(keys)
        assert len(table) == expected and table.index.is_unique and table.n.eq(300).all()
        columns = ['exact_rate', 'wrong_rate', 'erasure_rate', 'psnr_clean_db', 'ssim_clean']
        np.testing.assert_allclose(table[columns], actual.loc[table.index], rtol=0, atol=1e-10)
        np.testing.assert_array_equal(table[[f'{c}_count' for c in outcomes]], grouped[outcomes].sum().loc[table.index])
    print(f'Verified {len(cases):,} case records and both core-condition tables.')


if __name__ == '__main__':
    main()
