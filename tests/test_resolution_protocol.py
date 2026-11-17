import numpy as np
import torch
from watermarking.bench.adapters.hidden import HiddenAdapter


class Encoder(torch.nn.Module):
    def forward(self, x, message):
        return x + .1


def stub(protocol, strength):
    adapter = HiddenAdapter.__new__(HiddenAdapter)
    adapter.protocol, adapter.strength = protocol, strength
    adapter.device = torch.device('cpu')
    adapter.encoder = Encoder()
    adapter.mean = torch.tensor([.485, .456, .406])[None, :, None, None]
    adapter.std = torch.tensor([.229, .224, .225])[None, :, None, None]
    return adapter


def test_zero_strength_preserves_every_pixel():
    rgb = np.random.default_rng(2).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    assert np.array_equal(stub('resize256', 0).embed(rgb, 42).image, rgb)


def test_residual_has_no_normalization_offset():
    rgb = np.full((512, 512, 3), 100, dtype=np.uint8)
    out = stub('resize256', 1).embed(rgb, 42).image
    expected = np.round(100 + .1 * np.array([.229, .224, .225]) * 255).astype(np.uint8)
    assert np.all(out == expected)


def test_native_path_preserves_original_formula():
    rgb = np.random.default_rng(4).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    a = stub('native512', .45)
    x = (a._tensor(rgb) - a.mean) / a.std
    expected = (((x + .45 * ((x + .1) - x)) * a.std + a.mean).clamp(0, 1)[0]
                .permute(1, 2, 0) * 255).round().byte().numpy()
    assert np.array_equal(a.embed(rgb, 0).image, expected)
