"""All attacks consume/return RGB uint8; model normalization is adapter-local."""
import io
import math
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter
from .manifest import seed_for

CORE = ('clean', 'jpeg75', 'jpeg50', 'noise2', 'noise5') + tuple(
    f'rotate{a:+}' for a in (-10, -5, -2, -1, 1, 2, 5, 10)) + tuple(
    f'crop{area}_{v}' for area in (75, 50) for v in range(3)) + ('crop75_jpeg75',)
EXTENDED = ('salt_pepper', 'resize0.5', 'resize1.5', 'median3', 'gaussian5', 'equalize_y', 'gamma0.7', 'gamma1.5')
ATTACKS = CORE + EXTENDED


def plan_attack(attack_id, image_id, payload_seed):
    if attack_id not in ATTACKS:
        raise ValueError(f'Unknown attack: {attack_id}')
    # Combined attack shares exactly the crop75_0 coordinates.
    seed_id = 'crop75_0' if attack_id == 'crop75_jpeg75' else attack_id
    seed = seed_for(f'maie5421-attack-v1|{image_id}|{payload_seed}|{seed_id}')
    p = dict(attack_id=attack_id, seed=seed)
    if attack_id.startswith('crop'):
        area = int(attack_id[4:6]) / 100
        v = 0 if attack_id == 'crop75_jpeg75' else int(attack_id[-1])
        ratio = (1., 4 / 3, 3 / 4)[v]
        width = min(512, max(32, round(math.sqrt(area * 512**2 * ratio))))
        height = min(512, max(32, round(area * 512**2 / width)))
        rng = np.random.default_rng(seed)
        p.update(x=int(rng.integers(0, 513 - width)), y=int(rng.integers(0, 513 - height)),
                 width=width, height=height, target_area=area, ratio=ratio,
                 actual_area=width * height / 512**2)
    return p


def _u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def apply_attack(rgb, plan):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('Attack input must be RGB uint8')
    name = plan['attack_id']
    if name not in ATTACKS:
        raise ValueError(name)
    im = Image.fromarray(rgb)
    if name.startswith('crop'):
        x, y, w, h = (plan[k] for k in ('x', 'y', 'width', 'height'))
        im = im.crop((x, y, x + w, y + h))
    if name.startswith('jpeg') or name == 'crop75_jpeg75':
        b = io.BytesIO()
        im.save(b, format='JPEG', quality=50 if name == 'jpeg50' else 75, subsampling=2)
        b.seek(0)
        return np.array(Image.open(b).convert('RGB'))
    if name.startswith('rotate'):
        return np.array(im.rotate(float(name[6:]), Image.Resampling.BICUBIC,
                                  expand=False, fillcolor=(0, 0, 0)))
    if name.startswith('noise'):
        return _u8(rgb.astype(float) + np.random.default_rng(plan['seed']).normal(
            0, float(name[5:]), rgb.shape))
    if name == 'salt_pepper':
        out = rgb.copy()
        r = np.random.default_rng(plan['seed']).random(rgb.shape[:2])
        out[r < .005] = 0
        out[(r >= .005) & (r < .01)] = 255
        return out
    if name.startswith('resize'):
        n = round(512 * float(name[6:]))
        return np.array(im.resize((n, n), Image.Resampling.BICUBIC).resize((512, 512), Image.Resampling.BICUBIC))
    if name == 'median3':
        return median_filter(rgb, size=(3, 3, 1), mode='reflect')
    if name == 'gaussian5':
        return _u8(gaussian_filter(rgb.astype(float), sigma=(1, 1, 0), radius=(2, 2, 0), mode='reflect'))
    if name == 'equalize_y':
        y, cb, cr = im.convert('YCbCr').split()
        from PIL import ImageOps
        return np.array(Image.merge('YCbCr', (ImageOps.equalize(y), cb, cr)).convert('RGB'))
    if name.startswith('gamma'):
        return _u8(255 * (rgb.astype(float) / 255)**float(name[5:]))
    return np.array(im).copy()


def transform_masks(masks, plan, target_shape):
    """Oracle-only geometry; never an input to adapter.extract or decode_words."""
    from PIL import Image
    result = []
    for m in masks:
        im = Image.fromarray(m.astype(np.uint8) * 255)
        name = plan['attack_id']
        if name.startswith('crop'):
            x, y, w, h = (plan[k] for k in ('x', 'y', 'width', 'height'))
            im = im.crop((x, y, x + w, y + h))
        elif name.startswith('rotate'):
            im = im.rotate(float(name[6:]), Image.Resampling.NEAREST, expand=False, fillcolor=0)
        elif name.startswith('resize'):
            n = round(512 * float(name[6:]))
            im = im.resize((n, n), Image.Resampling.NEAREST).resize((512, 512), Image.Resampling.NEAREST)
        result.append(np.array(im.resize((target_shape[1], target_shape[0]), Image.Resampling.NEAREST)) > 127)
    return np.stack(result)
