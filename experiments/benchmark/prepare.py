import argparse
from pathlib import Path
from watermarking.bench.manifest import prepare
from watermarking.bench.attack_registry import ATTACKS, CORE
from watermarking.bench.cache import read_json

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    c = read_json(a.config)
    if a.dry_run:
        print(dict(images=c['expected_test_images'], main_per_model=100 * 3 * len(ATTACKS),
                   e1=100 * 3 * len(CORE), main_models=['full', 'hidden'], decoder_strategies=19,
                   manifest_exists=Path(c['manifest']).exists()))
    else:
        print('Prepared', len(prepare(c['root'])), 'images')
