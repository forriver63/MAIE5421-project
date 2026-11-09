"""Small dependency-free adapter for the inspected team protocol at e437788.

Formulas/settings are aligned, but this runner starts all attacks from an 8-bit
PNG round trip. Team historical measurements start from the floating WAM tensor.
This adapter is not a claim of bit-identical PyTorch normalization or JPEG codecs.
"""
import io
import math
import numpy as np
from PIL import Image
from watermark import to_uint8

TEAM_COMMIT = 'e43778855daee459079feb6394150355d66d5744'
TEAM_URL = 'https://github.com/forriver63/MAIE5421-project'
IMAGE_NAMES = ['astronaut', 'coffee', 'chelsea', 'rocket', 'hubble',
               'immunohistochemistry', 'retina', 'motorcycle', 'camera', 'coins']
SEED = 20261008


def source_bytes(image_index, payload_seed):
    rng = np.random.default_rng(SEED+70000+image_index*1009+payload_seed*97)
    return rng.integers(0, 256, 4, dtype=np.uint8)


def source_bits(image_index, payload_seed):
    return np.unpackbits(source_bytes(image_index, payload_seed), bitorder='big')


def crop_box(image_index, payload_seed):
    # First variant from team fixed_crops(): retained=.75, ratio=1, variant=0.
    w = min(512, max(32, round(math.sqrt(0.75*512**2))))
    h = min(512, max(32, round(0.75*512**2/w)))
    rng = np.random.default_rng(SEED+50000+image_index*1009+payload_seed*97+75)
    x, y = int(rng.integers(0, 512-w+1)), int(rng.integers(0, 512-h+1))
    return x, y, w, h


def jpeg(a, quality):
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format='JPEG', quality=quality, subsampling=2)
    buf.seek(0)
    with Image.open(buf) as im:
        return np.array(im.convert('RGB'))


def attacks(a, image_index, payload_seed):
    """14 conditions. Ground-truth crop box is logged, never passed to extract."""
    yield 'clean_png', a, {}
    for q in (75, 50):
        yield f'jpeg{q}', jpeg(a, q), {'quality': q, 'subsampling': 2}
    for sigma in (2, 5):
        seed = SEED+image_index*1009+payload_seed*97+sigma
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, sigma/255, a.shape)
        noisy = to_uint8(np.clip(a.astype(np.float64)/255+noise, 0, 1)*255)
        yield f'noise{sigma}', noisy, {'sigma_8bit': sigma, 'seed': seed}
    for angle in (-10, -5, -2, -1, 1, 2, 5, 10):
        im = Image.fromarray(a).rotate(angle, resample=Image.Resampling.BICUBIC,
                                      expand=False, fillcolor=(0, 0, 0))
        yield f'rotate{angle:+d}', np.array(im), {'angle': angle, 'expand': False,
                                                'resampling': 'bicubic', 'fill': [0, 0, 0]}
    x, y, w, h = crop_box(image_index, payload_seed)
    yield 'crop75_jpeg75', jpeg(a[y:y+h, x:x+w], 75), {'crop_xywh': [x,y,w,h],
                                                    'quality': 75, 'retained_area': w*h/512**2}


def quality(original, marked):
    """RGB PSNR and default-style skimage SSIM, reimplemented with NumPy.

    data_range=255, uniform 7x7 window, sample covariance (49/48), K1=.01,
    K2=.03; valid spatial region and equal averaging of RGB channels. This
    matches the explicit mathematical settings of skimage's default SSIM.
    """
    a, b = np.asarray(original, float), np.asarray(marked, float)
    if a.shape != b.shape or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError('quality expects same-shaped RGB images')
    if min(a.shape[:2]) < 7:
        raise ValueError('SSIM requires at least 7x7 pixels')

    def mean7(x):
        sums = np.pad(x, ((1, 0), (1, 0), (0, 0))).cumsum(0).cumsum(1)
        return (sums[7:,7:]-sums[:-7,7:]-sums[7:,:-7]+sums[:-7,:-7])/49

    ma, mb = mean7(a), mean7(b)
    va = (mean7(a*a)-ma*ma)*49/48
    vb = (mean7(b*b)-mb*mb)*49/48
    cov = (mean7(a*b)-ma*mb)*49/48
    c1, c2 = (0.01*255)**2, (0.03*255)**2
    ssim = np.mean(((2*ma*mb+c1)*(2*cov+c2))/((ma*ma+mb*mb+c1)*(va+vb+c2)))
    mse = float(np.mean((a-b)**2))
    return {'psnr_db': float(10*np.log10(255**2/mse)) if mse else None,
            'psnr_infinite': mse == 0, 'ssim': float(ssim)}


def message_metrics(truth, recovered):
    if recovered is None:
        return {'bit_errors': None, 'payload_ber': None, 'global_exact': False,
                'normalized_correlation': None}
    a, b = np.asarray(truth, float), np.asarray(recovered, float)
    if a.shape != b.shape:
        raise ValueError('Payload length mismatch')
    errors = int(np.count_nonzero(a != b))
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    return {'bit_errors': errors, 'payload_ber': errors/len(a), 'global_exact': errors == 0,
            'normalized_correlation': float(a @ b / denom) if denom else None}
