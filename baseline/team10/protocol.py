"""Shared teammate attack sequence, preserved verbatim in AST form."""
from watermarking import attacks
from experiments.robustness.run import SEED, ANGLES, source_for

def conditions(marked, index, seed):
    """每次返回一个条件名和一张待解码的图；每种条件都从同一张水印图开始。"""
    yield ('clean', marked)
    for quality in (75, 50):
        yield (f'jpeg{quality}', attacks.jpeg_attack(marked, quality))
    for sigma in (2, 5):
        noise_seed = SEED + index * 1009 + seed * 97 + sigma
        yield (f'noise{sigma}', attacks.noise_attack(marked, sigma, noise_seed))
    for angle in ANGLES:
        yield (f'rotate{angle:+d}', attacks.rotate_attack(marked, angle))
    crop = next((c for c in attacks.fixed_crops(index, seed) if c[0] == 0.75))
    (_, _, _, x, y, width, height) = crop
    cropped = marked[:, :, y:y + height, x:x + width]
    yield ('crop75_jpeg75', attacks.jpeg_attack(cropped, 75))
