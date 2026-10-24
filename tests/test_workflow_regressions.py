"""Regression checks with synthetic measurements; no WAM weights/inference.

These integration imports need the upstream source. Codec tests stay usable
without it; skip this file when the optional checkout has not been prepared.
"""
import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path(os.environ.get('WAM_PATH', ROOT / 'validation_wam_20261007/watermark-anything'))
pytestmark = pytest.mark.skipif(not UPSTREAM.is_dir(), reason='requires upstream WAM source, no weights')


def runner(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'experiments' / name / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bch_gate_rejects_quality_loss_and_incomplete_images(tmp_path, monkeypatch):
    module = runner('inner_ecc')
    monkeypatch.setattr(module, 'OUT', tmp_path)
    rows = [{'useful_exact_packets': 4, 'raw_exact_packets': 4,
             'silent_wrong_decodes': 0, 'psnr': 43.0, 'ssim': .98} for _ in range(30)]
    summary = module.summarize_clean(rows, [None] * 120)
    assert summary['gate']['protected_packet_recovery_ge_98pct']
    assert not summary['gate']['psnr_drop_le_0p5_db']
    assert not summary['gate']['ssim_drop_le_0p005']
    assert not summary['gate']['pass']
    for row in rows:
        row.update(psnr=44.1, ssim=.989)
    assert module.summarize_clean(rows, [None] * 120)['gate']['pass']
    for row in rows[:4]:
        row['useful_exact_packets'] = 3
    assert not module.summarize_clean(rows, [None] * 120)['gate']['all_four_images_ge_90pct']


def test_mds_gain_alone_cannot_pass_gate(tmp_path, monkeypatch):
    module = runner('hierarchical')
    monkeypatch.setattr(module, 'OUT', tmp_path)
    clean = [{'method': method, 'psnr': 41., 'ssim': .98} for method in ('repetition', 'mds')]
    crops = [{'method': method, 'global_exact': True, 'silent_wrong_packets': 0,
              'unmatched_accepted_clusters': 0} for method in ('repetition', 'mds')]
    paired = [{'retained_target': .75, 'mds_exact': False, 'repetition_exact': True}]
    paired += [{'retained_target': .5, 'mds_exact': True, 'repetition_exact': False}] * 3
    summary = module.summarize_comparison(clean, crops, paired)
    assert summary['gate']['combined_gain_ge_10pp']
    assert not summary['gate']['nonnegative_gain_each_area']
    assert not summary['gate']['pass']
    paired[0].update(mds_exact=True, repetition_exact=False)
    assert module.summarize_comparison(clean, crops, paired)['gate']['pass']
    crops[1]['unmatched_accepted_clusters'] = 1
    assert not module.summarize_comparison(clean, crops, paired)['gate']['pass']
    crops[1]['unmatched_accepted_clusters'] = 0
    clean[1].update(psnr=39., ssim=.97)
    gate = module.summarize_comparison(clean, crops, paired)['gate']
    assert not gate['psnr_difference_le_0p5db'] and not gate['ssim_difference_le_0p005']


def test_embed_fits_non_square_image_and_detect_keeps_crop(monkeypatch):
    from PIL import Image
    from watermarking import pipeline

    class FakeWam:
        def embed(self, image, messages):
            assert image.shape == (1, 3, 512, 512)
            return {'imgs_w': image}

    marked = pipeline.embed_image(Image.new('RGB', (640, 400)), 0, wam=FakeWam())
    assert marked.size == (512, 512)
    observed = []
    monkeypatch.setattr(pipeline, 'infer_packets', lambda wam, tensor: observed.append(tuple(tensor.shape)) or {'clusters': []})
    monkeypatch.setattr(pipeline, 'decode_global', lambda detection: {
        'global_payload': (0, 0, 0, 0), 'global_status': 'decoded',
        'unique_valid_ids': 4, 'conflicting_ids': []})
    result = pipeline.detect_image(marked.crop((0, 0, 300, 250)), wam=FakeWam())
    assert observed == [(1, 3, 250, 300)]
    assert result['payload'] == 0


def test_relative_wam_path_is_resolved(monkeypatch, tmp_path):
    from watermarking.wam.path import locate
    (tmp_path / 'upstream').mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('WAM_PATH', 'upstream')
    assert locate() == tmp_path / 'upstream'


def test_unknown_outer_method_is_rejected():
    from watermarking.eval import decode_global
    with pytest.raises(ValueError, match='Unknown outer code'):
        decode_global({'clusters': []}, method='typo')


def test_json_cli_separates_upstream_diagnostics(tmp_path, monkeypatch, capsys):
    import json
    from PIL import Image
    from watermarking import cli
    path = tmp_path / 'input.png'
    Image.new('RGB', (10, 10)).save(path)

    def fake_detect(image):
        print('Model loaded successfully')
        return {'status': 'decoded', 'payload': 123}

    monkeypatch.setattr(cli, 'detect_image', fake_detect)
    assert cli.main(['detect', str(path), '--json']) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)['payload'] == 123
    assert 'Model loaded successfully' in captured.err


def test_crop_retention_is_not_reported_as_one(monkeypatch):
    module = runner('robustness')
    masks = module.mask_mod.make_masks(8, module.DEVICE)
    oracle = masks[:, :, :256, :]
    observed = []
    monkeypatch.setattr(module.ev, 'infer_packets', lambda *args: {'runtime_seconds': 0.})
    monkeypatch.setattr(module.ev, 'decode_global', lambda *args: {
        'global_status': 'missing', 'global_payload': None, 'accepted_clusters': 0,
        'unique_valid_ids': 0, 'conflicting_ids': [], 'erased_id_count': 8,
        'outer_decode_seconds': 0.})
    def fake_score(detection, symbols, messages, masks, fractions):
        observed.extend(fractions)
        return [], {'detected_clusters': 0, 'unmatched_accepted_clusters': 0}
    monkeypatch.setattr(module.ev, 'score', fake_score)
    module.infer_case(None, None, (1, 2, 3, 4), [], None, oracle)
    assert observed == [1., 1., 1., 1., 0., 0., 0., 0.]


def test_dataset_uses_fetcher_when_sample_not_bundled(tmp_path, monkeypatch):
    from PIL import Image
    import skimage.data
    from skimage.data import _fetchers
    from watermarking.data import loader
    source = tmp_path / 'source.png'
    Image.new('RGB', (640, 400)).save(source)
    monkeypatch.setattr(skimage.data, '__file__', str(tmp_path / 'missing/__init__.py'))
    monkeypatch.setattr(loader, 'IMAGE_FILES', {'example': 'example.png'})
    requests = []
    monkeypatch.setattr(_fetchers, '_fetch', lambda name: requests.append(name) or str(source))
    records = loader.load_images(tmp_path / 'prepared')
    assert requests == ['data/example.png']
    assert Image.open(records[0]['processed']).size == (512, 512)
    assert records[0]['source_sha256'] == loader.sha256(source)
