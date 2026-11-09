"""Zhang et al. (2017), section 4; spatial approximation from equation (10).

All host-image calculations use float64 on the 0..255 scale.
No host image or true watermark is passed to extract().
"""
import numpy as np


def binary_square(watermark):
    w = np.asarray(watermark)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or w.size == 0:
        raise ValueError('Arnold transform requires a nonempty square watermark.')
    if not np.isin(w, [0, 1]).all():
        raise ValueError('Watermark values must be 0 or 1, not 0 or 255.')
    return w.astype(np.uint8)


def arnold(watermark, iterations=12, inverse=False):
    """Equation (2). x=column, y=row; inverse matrix is [[2,-1],[-1,1]]."""
    w = binary_square(watermark).copy()
    if not isinstance(iterations, int) or iterations < 0:
        raise ValueError('iterations must be a nonnegative integer.')
    n = w.shape[0]
    y, x = np.indices(w.shape)
    if inverse:
        xx, yy = (2*x-y) % n, (-x+y) % n
    else:
        xx, yy = (x+y) % n, (x+2*y) % n
    for _ in range(iterations):
        out = np.empty_like(w)
        out[yy, xx] = w
        w = out
    return w


def to_blocks(image, block_size=4):
    """Row-major block order; preserves the location represented by each index."""
    a = np.asarray(image, dtype=np.float64)
    m = block_size
    if not isinstance(m, int) or m < 1:
        raise ValueError('block_size must be a positive integer.')
    if a.ndim != 2 or a.size == 0 or not np.isfinite(a).all():
        raise ValueError('Expected a nonempty finite 2D grayscale image.')
    h, w = a.shape
    if h % m or w % m:
        raise ValueError('Image dimensions must be divisible by block_size.')
    return a.reshape(h//m, m, w//m, m).transpose(0, 2, 1, 3).reshape(-1, m, m).copy()


def from_blocks(blocks, shape):
    h, w = shape
    m = blocks.shape[-1]
    return blocks.reshape(h//m, w//m, m, m).transpose(0, 2, 1, 3).reshape(h, w)


def target_values(singular_values, bits, delta):
    """Equations (12)-(13), including the paper's tie rule (choose c1)."""
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError('delta must be positive and finite.')
    s = np.asarray(singular_values, dtype=np.float64)
    b = np.asarray(bits)
    if not np.isin(b, [0, 1]).all():
        raise ValueError('bits must be binary.')
    k1 = np.floor(np.ceil(s / delta) / 2)
    t1 = np.where(b == 1, 0.5*delta, -0.5*delta)
    t2 = np.where(b == 1, -1.5*delta, 1.5*delta)
    c1, c2 = 2*k1*delta+t1, 2*k1*delta+t2
    return np.where(np.abs(s-c2) < np.abs(s-c1), c2, c1)


def embed(image, watermark, block_size=4, delta=12.0, iterations=12):
    """Return floating image, extraction key, and approximation diagnostics.

    Spatial update: A_w = A + (target-lambda)/m, using equations (8) and (10).
    This is approximate for nonconstant blocks. No corrective iterations are added.
    """
    a = np.asarray(image, dtype=np.float64)
    if (a < 0).any() or (a > 255).any():
        raise ValueError('Input image must use the 0..255 scale.')
    w = binary_square(watermark)
    blocks = to_blocks(a, block_size)
    if w.size > len(blocks):
        raise ValueError('One bit per selected block: insufficient image capacity.')
    # Matrix 2-norm = largest singular value, NOT the Frobenius norm.
    # NumPy may compute this using SVD internally; do not claim an SVD-free runtime.
    s = np.linalg.norm(blocks, ord=2, axis=(-2, -1))
    # Stable order resolves equal values by original row-major block index.
    positions = np.argsort(-s, kind='stable')[:w.size]
    bits = arnold(w, iterations).ravel()
    targets = target_values(s[positions], bits, delta)
    blocks[positions] += ((targets-s[positions])/block_size)[:, None, None]
    actual = np.linalg.norm(blocks[positions], ord=2, axis=(-2, -1))
    marked = from_blocks(blocks, a.shape)
    key = {
        'method': 'zhang2017_spatial_eq10',
        'image_shape': list(a.shape), 'watermark_shape': list(w.shape),
        'block_size': block_size, 'delta': float(delta),
        'arnold_iterations': iterations, 'positions': positions.tolist(),
        'coordinate_convention': 'x=column,y=row; row-major flattening',
    }
    diagnostics = {
        'mean_abs_target_error_float': float(np.mean(np.abs(actual-targets))),
        'max_abs_target_error_float': float(np.max(np.abs(actual-targets))),
        'out_of_range_pixels_before_clipping': int(((marked < 0) | (marked > 255)).sum()),
    }
    return marked, key, diagnostics


def extract(image, key):
    """Equation (14), then inverse Arnold; only image and side-information key."""
    if list(np.shape(image)) != key['image_shape']:
        raise ValueError('Image size changed. Geometric synchronization is not implemented.')
    blocks = to_blocks(image, key['block_size'])
    positions = np.asarray(key['positions'], dtype=np.int64)
    shape = tuple(key['watermark_shape'])
    if len(positions) != int(np.prod(shape)) or len(np.unique(positions)) != len(positions):
        raise ValueError('Invalid key positions.')
    if (positions < 0).any() or (positions >= len(blocks)).any():
        raise ValueError('Key position out of range.')
    delta = key['delta']
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError('Invalid delta.')
    s = np.linalg.norm(blocks[positions], ord=2, axis=(-2, -1))
    bits = (np.ceil(s/delta).astype(np.int64) % 2).astype(np.uint8)
    return arnold(bits.reshape(shape), key['arnold_iterations'], inverse=True)


def to_uint8(image):
    """Explicit engineering convention: clip, round half up, then uint8."""
    return np.floor(np.clip(image, 0, 255)+0.5).astype(np.uint8)
