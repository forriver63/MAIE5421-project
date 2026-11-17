"""Validate artifact identity and the complete frozen experiment contract."""
from pathlib import Path
from .cache import fingerprint, read_json, file_hash
from .decode_variants import STRATEGIES
from .manifest import payload_for, load_manifest
from .attack_registry import ATTACKS, CORE, plan_attack


def strategies(method):
    if method == 'full':
        return list(STRATEGIES)
    if method == 'hidden':
        return [method]
    if method == 'e1':
        return ['E1_no_bch']
    raise ValueError(f'Unknown method: {method}')


def validate_run(run_dir, allow_partial=False, require_formal=False):
    root = Path(run_dir)
    frozen = read_json(root / 'freeze.json') if (root / 'freeze.json').exists() else None
    if require_formal and frozen is None:
        raise ValueError('Formal report requires freeze.json')
    missing, counts = [], {}
    models = frozen['expected'] if frozen else {
        p.name: None for p in root.iterdir() if (p / 'manifest').is_dir()}
    for method in models:
        folder = root / method
        prov = read_json(folder / 'provenance.json')
        identity = prov.pop('identity')
        if fingerprint(prov) != identity:
            raise ValueError('Invalid provenance identity')
        if frozen and frozen['model_identities'][method] != identity:
            raise ValueError('Frozen model identity mismatch')
        records = {r['image_id']: r for r in read_json(prov['config']['manifest'])}
        for record in records.values():
            processed = Path(record['processed'])
            if not processed.is_absolute():
                processed = Path(prov['config']['manifest']).resolve().parent / processed
            if file_hash(processed) != record['processed_sha256']:
                raise ValueError('Processed image hash changed')
        all_ids = set()
        scopes = {}
        for manifest in (folder / 'manifest').glob('*.json'):
            plans = read_json(manifest)
            scopes[manifest.stem] = len(plans)
            if frozen:
                negative = manifest.stem == 'test_negative'
                expected = [dict(case_id=f"{r['image_id']}_s{s}_{a}", image_id=r['image_id'], split='test',
                    payload_seed=s, attack=plan_attack(a, r['image_id'], s), negative=negative)
                    for r in load_manifest(prov['config']['manifest'], 'test')
                    for s in ([-1] if negative else prov['config']['payload_seeds'])
                    for a in (['clean'] if negative else CORE if method == 'e1' else ATTACKS)]
                if plans != expected:
                    raise ValueError('Formal manifest differs from protocol')
            for plan in plans:
                cid = plan['case_id']
                if cid in all_ids:
                    raise ValueError('Duplicate case in manifests')
                all_ids.add(cid)
                path = folder / 'cases' / f'{cid}.json.gz'
                if not path.exists():
                    missing.append(str(path))
                    continue
                case = read_json(path)
                rec = records[plan['image_id']]
                key = fingerprint(dict(identity=identity, plan=plan, processed_sha256=rec['processed_sha256']))
                if case['fingerprint'] != key or case['method'] != method or any(case[k] != v for k, v in plan.items()):
                    raise ValueError('Case identity/plan mismatch')
                if rec['split'] != plan['split']:
                    raise ValueError('Split mismatch')
                if case['expected_payload'] != (None if plan['negative'] else payload_for(plan['image_id'], plan['payload_seed'])):
                    raise ValueError('Payload mismatch')
                if [r['strategy'] for r in case['results']] != strategies(method):
                    raise ValueError('Missing/duplicate/unexpected strategies')
                for r in case['results']:
                    if r['accepted'] != (r['payload'] is not None):
                        raise ValueError('Invalid acceptance')
                    if not plan['negative'] and (r['exact'] + r['wrong'] + r['erasure'] != 1):
                        raise ValueError('Invalid outcome')
        actual = {p.name.removesuffix('.json.gz') for p in (folder / 'cases').glob('*.json.gz')}
        if actual - all_ids:
            raise ValueError('Unexpected case files')
        if frozen and scopes != frozen['expected'][method]:
            raise ValueError('Missing or unexpected formal manifest scope')
        counts[method] = scopes
    errors = list(root.rglob('*.error.json'))
    if (missing or errors) and not allow_partial:
        raise ValueError(f'Incomplete run: {len(missing)} missing, {len(errors)} errors')
    return dict(missing=len(missing), errors=len(errors), counts=counts)
