import argparse
import os
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
    c['root'] = str((Path(a.config).resolve().parent / os.environ.get('WM_ROOT', c['root'])).resolve())
    c['manifest'] = str((Path(c['root']) / c['manifest']).resolve())
    if a.dry_run:
        count = c['expected_test_images'] * len(c['payload_seeds'])
        print(dict(images=c['expected_test_images'], main_per_model=count * len(ATTACKS),
                   e1=count * len(CORE), main_models=['full', 'hidden'], decoder_strategies=19,
                   manifest_exists=Path(c['manifest']).exists()))
    else:
        print('Prepared', len(prepare(c['root'])), 'images')
