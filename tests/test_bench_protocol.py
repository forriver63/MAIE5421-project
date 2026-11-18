import inspect
import random
import numpy as np
import pytest
from watermarking.bench.cache import load_cache, write_json
from watermarking.bench.decode_variants import SINGLE_LAYOUTS, STRATEGIES, decode_packet_variant, decode_words
from watermarking.bench.attack_registry import ATTACKS, CORE, plan_attack, apply_attack
from watermarking.bench.scoring import score_payload, score_regions
from watermarking.bench.adapters.hidden import bits_of, parse_logits
from watermarking.codec.packet import decode_packet, encode_packet, packet16
from watermarking.codec.outer import encode_repetition
from watermarking.codec.bch import encode_int


def test_full_equivalence_and_zero_payload():
    rng = random.Random(17)
    for _ in range(1000):
        w = rng.getrandbits(32)
        assert decode_packet_variant(w) == decode_packet(w)
    words = [encode_packet(i, s) for i, s in enumerate(encode_repetition((0, 0, 0, 0)))]
    assert decode_words(words)['payload'] == 0
    assert score_payload(decode_words(words), 0)['outcome'] == 'exact'


def test_systematic_and_bypass(monkeypatch):
    from watermarking.bench import decode_variants as dv
    assert all(encode_int(i) >> 16 == i for i in range(65536))
    monkeypatch.setattr(dv, 'decode_int', lambda _: (_ for _ in ()).throw(AssertionError('BCH called')))
    assert dv.decode_packet_variant(encode_packet(3, 7), bch=False)['symbol'] == 7


def test_crc_off_keeps_bch_reject_and_can_accept_silent_wrong():
    # Extended BCH detects all weight-4 errors; CRC-off cannot override this gate.
    assert decode_packet_variant(15, crc=False)['status'] == 'ERASURE'
    # Valid BCH codeword containing an invalid CRC is accepted only with CRC disabled.
    w = encode_int(packet16(2, 91) ^ 1)
    assert decode_packet_variant(w)['reason'] == 'crc_reject'
    assert decode_packet_variant(w, crc=False)['status'] == 'ACCEPTED'


def test_layouts_and_predicted_id_filtering():
    assert len(set(SINGLE_LAYOUTS)) == 16
    assert all(len(s) == 4 for s in SINGLE_LAYOUTS)
    words = [encode_packet(i, s) for i, s in enumerate(encode_repetition((11, 22, 33, 44)))]
    assert all(decode_words(words, s)['accepted'] for s in STRATEGIES)
    for i, layout in enumerate(SINGLE_LAYOUTS):
        selected_words = [words[j] for j in layout]
        assert decode_words(selected_words, f'A3_single_{i:02}')['accepted']
        assert not decode_words(selected_words[:-1], f'A3_single_{i:02}')['accepted']
    assert 'expected' not in inspect.signature(decode_words).parameters
    assert decode_words(words + [encode_packet(0, 99)])['accepted']  # other copy may recover A
    assert decode_words([encode_packet(0, 11), encode_packet(7, 99)] + words[1:7])['status'] == 'repetition_conflict'


def test_outcomes_and_missing_ber():
    r = dict(payload=None, accepted=False)
    a, b = score_payload(r, 0), score_payload(r, 15)
    assert r == dict(payload=None, accepted=False)
    assert a['payload_ber'] is None and b['erasure'] == 1
    for result in [dict(payload=0, accepted=True), dict(payload=1, accepted=True), r]:
        s = score_payload(result, 0)
        assert s['exact'] + s['wrong'] + s['erasure'] == 1


def test_carrier_scoring_uses_actual_padding():
    words = [packet16(i, 0) << 16 | 0xaacc for i in range(8)]
    labels = np.tile(np.arange(8, dtype=np.int16), (8, 1))
    masks = np.stack([labels == i for i in range(8)])
    clusters = [dict(label=i, raw_word=w) for i, w in enumerate(words)]
    packets = [decode_packet_variant(w, bch=False) for w in words]
    regions, _ = score_regions(clusters, labels, words, [0] * 8, masks, packets)
    assert all(r['raw_hamming'] == 0 for r in regions)
    altered, _ = score_regions(clusters, labels, [w ^ 1 for w in words], [0] * 8, masks, packets)
    assert all(r['raw_hamming'] == 1 for r in altered)


def test_attack_registry_determinism_and_scale():
    assert len(CORE) == 20 and len(ATTACKS) == 28
    x = np.full((512, 512, 3), 128, dtype=np.uint8)
    for attack in ATTACKS:
        plan = plan_attack(attack, 'same_image', 2)
        assert plan == plan_attack(attack, 'same_image', 2)
        a, b = apply_attack(x, plan), apply_attack(x, plan)
        assert a.dtype == np.uint8 and np.array_equal(a, b)
    p = plan_attack('noise5', 'same_image', 2)
    assert 4.8 < apply_attack(x, p).astype(float).std() < 5.2
    p = plan_attack('crop75_0', 'a', 0)
    q = plan_attack('crop75_jpeg75', 'a', 0)
    assert all(p[k] == q[k] for k in ['seed', 'x', 'y', 'width', 'height'])
    assert abs(p['actual_area'] - .75) < .003
    with pytest.raises(ValueError):
        plan_attack('typo', 'a', 0)


def test_hidden_marker_boundaries_and_msb():
    bits = np.array(bits_of(0xA5C3, 16) + bits_of(0x12345678, 32), dtype=np.float32)
    assert set(2 * bits - 1) == {-1, 1}
    logits = 2 * bits - 1
    assert parse_logits(logits)['payload'] == 0x12345678
    logits[0] *= -1
    assert parse_logits(logits)['accepted']
    logits[1] *= -1
    assert not parse_logits(logits)['accepted']
    assert parse_logits(logits)['raw_payload'] == 0x12345678


def test_cache_resume_and_fingerprint_mismatch(tmp_path):
    p = tmp_path / 'case.json.gz'
    assert load_cache(p, 'one') is None
    write_json(p, dict(fingerprint='one', results=[]))
    assert load_cache(p, 'one')['results'] == []
    with pytest.raises(ValueError, match='fingerprint'):
        load_cache(p, 'two')
    p.write_bytes(b'truncated')
    with pytest.raises(Exception):
        load_cache(p, 'one')


def test_exact_binary_graph_matches_pixel_dbscan():
    import torch
    from sklearn.cluster import DBSCAN
    from watermarking.bench.clustering import binary_dbscan
    rng = np.random.default_rng(20261106)
    for _ in range(12):
        # Duplicates, isolated points, connected cores and ambiguous border cases.
        words = rng.integers(0, 64, size=400, dtype=np.uint32)
        bits = ((words[:, None] >> np.arange(31, -1, -1)) & 1).astype(np.float32)
        logits = torch.from_numpy((bits * 2 - 1).T.reshape(1, 32, 20, 20))
        masks = torch.ones((1, 20, 20))
        expected = DBSCAN(eps=1, min_samples=40).fit(bits).labels_.reshape(20, 20)
        centroids, actual = binary_dbscan(logits, masks, min_samples=40)
        assert np.array_equal(actual.numpy(), expected)
        for label, centroid in centroids.items():
            assert np.array_equal(centroid.numpy(), bits[expected.reshape(-1) == label].mean(axis=0) > .5)


def test_runner_resume_manifest_and_no_truth_in_extract(tmp_path, monkeypatch):
    from PIL import Image
    from watermarking.bench import runner
    from watermarking.bench.adapters import hidden
    from watermarking.bench.contracts import EncodeResult
    from watermarking.bench.cache import file_hash
    image = tmp_path / 'image.png'
    Image.fromarray(np.full((512, 512, 3), 128, np.uint8)).save(image)
    manifest = tmp_path / 'images.json'
    write_json(manifest, [dict(image_id='dummy', split='calibration', processed=str(image),
                               processed_sha256=file_hash(image))])
    config = dict(root=str(tmp_path), manifest=str(manifest), device='cpu', torch_threads=1,
                  hidden_strength=.45, payload_seeds=[0])
    calls = []
    class Fake:
        def __init__(self, *args):
            pass
        def embed(self, rgb, payload, *, public_seed):
            return EncodeResult(rgb)
        def extract(self, rgb):
            calls.append(rgb.shape)
            return dict(payload=0, raw_payload=0, accepted=True, status='decoded')
    monkeypatch.setattr(hidden, 'HiddenAdapter', Fake)
    monkeypatch.setattr(runner, 'provenance', lambda c, m: dict(config=c, method=m))
    kwargs = dict(attacks=['clean', 'noise5', 'crop75_0'])
    first = runner.run(config, tmp_path / 'run', 'hidden', 'calibration', **kwargs)
    first_calls = list(calls)
    calls.clear()
    second = runner.run(config, tmp_path / 'run', 'hidden', 'calibration', **kwargs)
    assert first['fresh'] == 3 and second['fresh'] == 0 and not calls
    assert first_calls[-1][0] < 512  # Crop pixels only, no coordinate or true message argument.
    with pytest.raises(ValueError, match='identity'):
        runner.run(dict(config, hidden_strength=.6), tmp_path / 'run', 'hidden', 'calibration', **kwargs)


def test_bootstrap_samples_images():
    from watermarking.bench.statistics import paired_image_bootstrap
    result = paired_image_bootstrap([0., 1.])
    assert result['difference'] == .5 and result['images'] == 2
    assert result['lower'] == 0 and result['upper'] == 1
