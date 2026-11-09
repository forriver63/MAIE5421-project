"""Regression tests for review findings, using isolated artifacts only."""
import pytest
from watermarking.bench.cache import read_json, write_json, fingerprint, file_hash
from watermarking.bench.validation import validate_run
from watermarking.bench.manifest import payload_for
from watermarking.bench.attack_registry import plan_attack
from watermarking.bench.decode_variants import decode_words, SINGLE_LAYOUTS
from watermarking.codec.packet import encode_packet
from watermarking.codec.outer import encode_repetition

@pytest.fixture
def fixture_run(tmp_path):
    """Build a valid tiny run without private datasets or an adjacent workspace."""
    image = tmp_path / 'image.bin'
    image.write_bytes(b'fixture')
    manifest = tmp_path / 'images.json'
    write_json(manifest, [dict(image_id='fixture', split='calibration', processed=str(image),
                               processed_sha256=file_hash(image))])
    prov = dict(config=dict(manifest=str(manifest)), method='full')
    identity = fingerprint(prov)
    plan = dict(case_id='fixture_s0_clean', image_id='fixture', split='calibration',
                payload_seed=0, attack=plan_attack('clean', 'fixture', 0), negative=False)
    case = dict(fingerprint=fingerprint(dict(identity=identity, plan=plan,
                    processed_sha256=file_hash(image))), **plan, method='full',
                expected_payload=payload_for('fixture', 0), results=[])
    from watermarking.bench.decode_variants import STRATEGIES
    case['results'] = [dict(strategy=s, accepted=False, payload=None, exact=0, wrong=0, erasure=1)
                       for s in STRATEGIES]
    source = tmp_path / 'source/full'
    write_json(source / 'provenance.json', dict(identity=identity, **prov))
    write_json(source / 'manifest/calibration_positive.json', [plan])
    write_json(source / 'cases/fixture_s0_clean.json.gz', case)
    assert validate_run(source.parent)['missing'] == 0
    return source


@pytest.mark.parametrize('damage', ['fingerprint', 'strategy', 'duplicate', 'extra'])
def test_reject_invalid_cached_artifacts(tmp_path, damage, fixture_run):
    source = fixture_run
    prov = read_json(source / 'provenance.json')
    plan = read_json(source / 'manifest/calibration_positive.json')[0]
    c = read_json(source / 'cases' / (plan['case_id'] + '.json.gz'))
    folder = tmp_path / 'full'
    write_json(folder / 'provenance.json', prov)
    write_json(folder / 'manifest/calibration_positive.json', [plan, plan] if damage == 'duplicate' else [plan])
    if damage == 'fingerprint':
        c['fingerprint'] = 'wrong'
    if damage == 'strategy':
        c['results'].pop()
    write_json(folder / 'cases' / (plan['case_id'] + '.json.gz'), c)
    if damage == 'extra':
        write_json(folder / 'cases/extra.json.gz', c)
    with pytest.raises(ValueError):
        validate_run(tmp_path)


def test_layout_acceptance_distinguishes_excluded_copies():
    words = [encode_packet(i, s) for i, s in enumerate(encode_repetition((1, 2, 3, 4)))]
    result = decode_words(words, 'A3_single_00')
    assert result['accepted']
    assert sum(p['passes_checks'] for p in result['packets']) == 8
    assert {p['packet_id'] for p in result['packets'] if p['selected_for_layout']} == set(SINGLE_LAYOUTS[0])


def test_frozen_identity_rejected_without_mutation(tmp_path, fixture_run):
    root = fixture_run.parent
    write_json(root / 'freeze.json', dict(expected={'full': {'calibration_positive': 1}},
                                         model_identities={'full': 'changed'}))
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='Frozen model identity'):
        validate_run(root)
    assert before == {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_missing_negative_scope_rejected(tmp_path, fixture_run):
    source = fixture_run
    prov = read_json(source / 'provenance.json')
    identity = prov['identity']
    # A frozen run must include both required scopes even before examining cases.
    write_json(tmp_path / 'full/provenance.json', prov)
    (tmp_path / 'full/manifest').mkdir()
    write_json(tmp_path / 'freeze.json', dict(expected={'full': {'test_positive': 8400, 'test_negative': 100}},
        model_identities={'full': identity}))
    with pytest.raises(ValueError, match='scope'):
        validate_run(tmp_path, require_formal=True)


def test_validator_resolves_relative_processed_image(tmp_path, fixture_run):
    manifest = tmp_path / 'images.json'
    records = read_json(manifest)
    records[0]['processed'] = 'image.bin'
    write_json(manifest, records)
    assert validate_run(fixture_run.parent)['missing'] == 0
