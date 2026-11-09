"""Shared resumable inference runner. No ground truth is passed to extract()."""
import argparse
import io
import hashlib
import os
import subprocess
import shutil
import time
import traceback
import psutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .cache import file_hash, fingerprint, load_cache, read_json, write_json
from .manifest import load_manifest, payload_for, seed_for
from .attack_registry import ATTACKS, CORE, apply_attack, plan_attack, transform_masks
from .decode_variants import STRATEGIES, decode_words
from .scoring import quality, score_payload, score_regions


def code_digest():
    base = Path(__file__).resolve().parents[1]
    return fingerprint({str(p.relative_to(base)): file_hash(p) for p in sorted(base.rglob('*.py'))})


def provenance(config, method):
    root = Path(config['root'])
    weights = list((root / 'models/hidden').glob('*.pt')) if method == 'hidden' else [
        root / 'third_party/watermark-anything/checkpoints/wam_mit.pth']
    executable = shutil.which('git')
    try:
        commit = subprocess.check_output([executable, '-C',
            str(root / 'third_party/watermark-anything'), 'rev-parse', 'HEAD'], text=True).strip() if executable else 'unknown'
    except (OSError, subprocess.CalledProcessError):
        commit = 'unknown'
    return dict(code_digest=code_digest(), weights={str(p): file_hash(p) for p in weights},
                torch=torch.__version__, method=method, config=config,
                upstream_commit=commit)


def _sync(device):
    if device.startswith('cuda'):
        torch.cuda.synchronize()


def run(config, run_dir, method, split, *, limit=None, seeds=None, attacks=None,
        negatives=False, max_cases=None):
    run_dir, root = Path(run_dir), Path(config['root'])
    source_dir = Path(__file__).resolve().parents[2]
    if run_dir.resolve() == source_dir or source_dir in run_dir.resolve().parents:
        raise ValueError('Benchmark output must be outside the Python source directory')
    torch.set_num_threads(config['torch_threads'])
    torch.manual_seed(20261106)
    records = load_manifest(config['manifest'], split)
    records = records[:limit] if limit else records
    seeds = config['payload_seeds'] if seeds is None else seeds
    attacks = tuple(attacks or (CORE if method == 'e1' else ATTACKS))
    if negatives:
        seeds, attacks = [-1], ('clean',)
    if not records or not attacks or any(a not in ATTACKS for a in attacks):
        raise ValueError('Empty manifest/unknown attack')
    prov = provenance(config, method)
    identity = fingerprint(prov)
    model_dir = run_dir / method
    guard = model_dir / 'provenance.json'
    if guard.exists() and read_json(guard)['identity'] != identity:
        raise ValueError('Run identity changed: create a new run directory')
    write_json(guard, dict(identity=identity, **prov))
    write_json(run_dir / 'config.resolved.json', config)
    plans = []
    for rec in records:
        for seed in seeds:
            for attack in attacks:
                plan = plan_attack(attack, rec['image_id'], seed)
                plans.append(dict(case_id=f"{rec['image_id']}_s{seed}_{attack}", image_id=rec['image_id'],
                                  split=split, payload_seed=seed, attack=plan, negative=negatives))
    scope = f'{split}_{"negative" if negatives else "positive"}'
    manifest_path = model_dir / 'manifest' / f'{scope}.json'
    if manifest_path.exists() and fingerprint(read_json(manifest_path)) != fingerprint(plans):
        raise ValueError('Manifest scope changed; choose another run directory')
    write_json(manifest_path, plans)
    by_id = {r['image_id']: r for r in records}
    adapter = None
    completed, fresh = 0, 0
    start = time.perf_counter()
    embeddings = {}
    for planned in plans:
        rec = by_id[planned['image_id']]
        key = fingerprint(dict(identity=identity, plan=planned, processed_sha256=rec['processed_sha256']))
        outpath = model_dir / 'cases' / f"{planned['case_id']}.json.gz"
        cached = load_cache(outpath, key)
        if cached is not None:
            completed += 1
            continue
        if max_cases is not None and fresh >= max_cases:
            break
        if adapter is None:
            load_start = time.perf_counter()
            if method == 'hidden':
                from .adapters.hidden import HiddenAdapter
                adapter = HiddenAdapter(root / 'models/hidden', config['hidden_strength'], config['device'],
                                        config.get('hidden_protocol', 'native512'))
            else:
                from .adapters.wam import WamAdapter
                adapter = WamAdapter('full' if method == 'full' else 'e1', config['device'])
            write_json(model_dir / 'load_timing.json', dict(seconds=time.perf_counter() - load_start))
            warm_start = time.perf_counter()
            for _ in range(3):
                adapter.extract(np.full((512, 512, 3), 128, dtype=np.uint8))
            _sync(config['device'])
            write_json(model_dir / 'warmup.json', dict(iterations=3, image='synthetic uniform RGB128',
                       seconds=time.perf_counter() - warm_start))
        try:
            cover = np.array(Image.open(rec['processed']).convert('RGB'))
            if file_hash(rec['processed']) != rec['processed_sha256']:
                raise ValueError('Processed image hash changed')
            seed = planned['payload_seed']
            expected = None if negatives else payload_for(rec['image_id'], seed)
            embed_key = f"{rec['image_id']}_s{seed}"
            if negatives:
                marked, metadata, q, embed_seconds = cover, {}, {}, None
            elif embed_key in embeddings:
                marked, metadata, q, embed_seconds = embeddings[embed_key]
            else:
                png = model_dir / 'embeddings' / f'{embed_key}.png'
                meta_path = png.with_suffix('.json')
                if png.exists() and meta_path.exists():
                    meta = read_json(meta_path)
                    if (meta['identity'] != identity or meta['png_sha256'] != file_hash(png)
                            or meta.get('processed_sha256') != rec['processed_sha256'] or meta.get('payload') != expected):
                        raise ValueError('Embedding cache mismatch')
                    marked = np.array(Image.open(png).convert('RGB'))
                    metadata, q, embed_seconds = meta['metadata'], meta['quality'], meta['embed_seconds']
                else:
                    _sync(config['device'])
                    t = time.perf_counter()
                    encoded = adapter.embed(cover, expected, public_seed=seed_for(
                        f'maie5421-padding-v1|{rec["image_id"]}|{seed}|{method}'))
                    _sync(config['device'])
                    embed_seconds = time.perf_counter() - t
                    png.parent.mkdir(parents=True, exist_ok=True)
                    b = io.BytesIO()
                    Image.fromarray(encoded.image).save(b, format='PNG')
                    png.write_bytes(b.getvalue())
                    marked = np.array(Image.open(png).convert('RGB'))
                    metadata, q = encoded.metadata, quality(cover, marked)
                    write_json(meta_path, dict(identity=identity, png_sha256=file_hash(png),
                               metadata=metadata, quality=q, embed_seconds=embed_seconds,
                               processed_sha256=rec['processed_sha256'], payload=expected))
                embeddings = {embed_key: (marked, metadata, q, embed_seconds)}
            attacked = apply_attack(marked, planned['attack'])
            _sync(config['device'])
            t = time.perf_counter()
            raw = adapter.extract(attacked)
            _sync(config['device'])
            extract_seconds = time.perf_counter() - t
            labels = raw.pop('labels', None)
            label_path = None
            if labels is not None:
                label_file = model_dir / 'raw_masks' / f"{planned['case_id']}.npz"
                label_file.parent.mkdir(parents=True, exist_ok=True)
                tmp_label = label_file.with_suffix('.tmp')
                with open(tmp_label, 'wb') as f:
                    np.savez_compressed(f, labels=labels)
                os.replace(tmp_label, label_file)
                label_path = str(label_file)
            codec_start = time.perf_counter()
            results = []
            if method == 'hidden':
                r = dict(strategy=method, **raw)
                if expected is not None:
                    r.update(score_payload(r, expected))
                results.append(r)
            else:
                clusters = raw['clusters']
                words = [c['raw_word'] for c in clusters]
                strategies = STRATEGIES if method == 'full' else ('E1_no_bch',)
                masks = None if negatives else transform_masks(adapter.oracle_masks(), planned['attack'], labels.shape)
                for strategy in strategies:
                    r = decode_words(words, strategy)
                    if expected is not None:
                        r.update(score_payload(r, expected))
                        regions, assigned = score_regions(clusters, labels, metadata['words'],
                                                          metadata['symbols'], masks, r['packets'], bch_encoded=method == 'full')
                        r['regions'] = regions
                        r['unmatched_accepted_clusters'] = sum(
                            i not in assigned and p.get('selected_for_layout', p['status'] == 'ACCEPTED') for i, p in enumerate(r['packets']))
                        r['wrong_accepted_clusters'] = sum(p.get('selected_for_layout', p['status'] == 'ACCEPTED') and
                            p['symbol'] != metadata['symbols'][p['packet_id']] for p in r['packets'])
                    results.append(r)
            value = dict(fingerprint=key, **planned, method=method, expected_payload=expected,
                         raw=raw, raw_masks_path=label_path, results=results, quality=q,
                         embed_seconds=embed_seconds, extract_seconds=extract_seconds,
                         attacked_sha256=hashlib.sha256(attacked.tobytes()).hexdigest(),
                         attacked_shape=list(attacked.shape), codec_scoring_seconds=time.perf_counter() - codec_start,
                         process_peak_working_set_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', None))
            write_json(outpath, value)
            error_path = outpath.with_name(outpath.name + '.error.json')
            if error_path.exists():
                error_path.unlink()
            completed += 1
            fresh += 1
            write_json(model_dir / f'progress_{scope}.json', dict(completed=completed, expected=len(plans),
                       fresh=fresh, elapsed_seconds=time.perf_counter() - start,
                       status='complete' if completed == len(plans) else 'running'))
            if fresh % 10 == 0 or fresh == 1:
                print(f'{method} {scope} {completed}/{len(plans)} last_extract={extract_seconds:.3f}s', flush=True)
        except Exception:
            write_json(outpath.with_name(outpath.name + '.error.json'), dict(case=planned, error=traceback.format_exc()))
            raise
    return dict(method=method, completed=completed, expected=len(plans), fresh=fresh)


def main(default_method=None, stage_mode=False):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--method', choices=['full', 'hidden', 'e1'], default=default_method or 'hidden')
    p.add_argument('--split', choices=['calibration', 'test'], default='test')
    p.add_argument('--stage', choices=['decode', 'encode'])
    p.add_argument('--limit', type=int)
    p.add_argument('--seed-count', type=int, choices=[1, 2, 3], default=3)
    p.add_argument('--attacks', nargs='+', choices=ATTACKS)
    p.add_argument('--negative', action='store_true')
    p.add_argument('--max-cases', type=int)
    p.add_argument('--resume', action='store_true', help='Exact fingerprint matches always resume safely')
    a = p.parse_args()
    config = read_json(a.config)
    config_path = Path(a.config).resolve()
    config['root'] = str((config_path.parent / os.environ.get('WM_ROOT', config['root'])).resolve())
    config['manifest'] = str((Path(config['root']) / config['manifest']).resolve())
    method = ('full' if a.stage == 'decode' else 'e1') if a.stage else a.method
    print(run(config, a.run_dir, method, a.split, limit=a.limit,
              seeds=list(range(a.seed_count)), attacks=a.attacks, negatives=a.negative, max_cases=a.max_cases))


if __name__ == '__main__':
    main()
