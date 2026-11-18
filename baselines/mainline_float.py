"""Bridge the traditional float RGB arrays to the mainline robustness API."""
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from watermarking import attacks
from watermarking.metrics import quality
from watermarking.wam.transforms import default_transform, unnormalize_img
from watermark_anything.data.transforms import normalize_img


def tensors(original, marked):
    source = default_transform(Image.fromarray(original)).unsqueeze(0)
    rgb = torch.from_numpy(marked).permute(2, 0, 1).to(torch.float32) / 255
    return source, normalize_img(rgb).unsqueeze(0)


def decode_array(tensor):
    return unnormalize_img(tensor)[0].permute(1, 2, 0).numpy() * 255


def variants(marked, image_index, seed):
    yield 'clean', marked, {}
    for q in (75, 50):
        yield f'jpeg{q}', attacks.jpeg_attack(marked, q), {'quality': q, 'subsampling': 2}
    for sigma in (2, 5):
        rng_seed = 20261008 + image_index * 1009 + seed * 97 + sigma
        yield f'noise{sigma}', attacks.noise_attack(marked, sigma, rng_seed), {'sigma': sigma, 'seed': rng_seed}
    for angle in (-10, -5, -2, -1, 1, 2, 5, 10):
        yield f'rotate{angle:+d}', attacks.rotate_attack(marked, angle), {'angle': angle}
    _, _, _, x, y, w, h = next(attacks.fixed_crops(image_index, seed))
    crop = marked[:, :, y:y+h, x:x+w]
    yield 'crop75_jpeg75', attacks.jpeg_attack(crop, 75), {'crop_xywh': [x, y, w, h], 'quality': 75}
