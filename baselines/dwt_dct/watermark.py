"""Al-Haj (2007) DWT-DCT core, with explicit engineering choices.

Haar DWT along HL1->HL2 or HH1->HH2 (NOT an LL1 second level).
Orthonormal 4x4 DCT-II; additive PN modulation; correlation decoding.
RGB-Y is a documented project adapter. No training, QIM, SVD or ECC.
"""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class Config:
    alpha: float = 20.0  # selected starting value; paper does not specify it
    pn_seed: int = 2007  # key seed, distinct from the team's payload seed
    subband: str = 'HL2'
    color_mode: str = 'rgb_y'
    repetitions: int = 1  # >1 is an OPTIONAL adaptation, not in the paper

    def validate(self):
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError('alpha must be positive and finite')
        if self.subband not in ('HL2', 'HH2'):
            raise ValueError('subband must be HL2 or HH2')
        if self.color_mode not in ('gray', 'rgb_y'):
            raise ValueError('color_mode must be gray or rgb_y')
        if not isinstance(self.repetitions, int) or self.repetitions < 1:
            raise ValueError('repetitions must be a positive integer')
        if not isinstance(self.pn_seed, int) or self.pn_seed < 0:
            raise ValueError('pn_seed must be a nonnegative integer')


class SynchronizationError(ValueError):
    """Input geometry no longer matches the agreed block grid."""


# Explicit mid-band choice, because the paper does not give an exact mask.
MID_POSITIONS = np.array([(u, v) for u in range(4) for v in range(4)
                          if u+v in (2, 3)], dtype=int)
Y_WEIGHTS = np.array([0.299, 0.587, 0.114])


def haar_dwt2(a):
    """Orthonormal Haar. HL=high-pass in x, low-pass in y (upper-right)."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or any(d % 2 for d in a.shape):
        raise ValueError('DWT requires a 2D array with even dimensions')
    lo = (a[:, 0::2]+a[:, 1::2])/np.sqrt(2)
    hi = (a[:, 0::2]-a[:, 1::2])/np.sqrt(2)
    return {
        'LL': (lo[0::2]+lo[1::2])/np.sqrt(2),
        'HL': (hi[0::2]+hi[1::2])/np.sqrt(2),
        'LH': (lo[0::2]-lo[1::2])/np.sqrt(2),
        'HH': (hi[0::2]-hi[1::2])/np.sqrt(2),
    }


def haar_idwt2(bands):
    ll, hl, lh, hh = [bands[k] for k in ('LL', 'HL', 'LH', 'HH')]
    h, w = ll.shape
    lo, hi = np.empty((2*h, w)), np.empty((2*h, w))
    lo[0::2], lo[1::2] = (ll+lh)/np.sqrt(2), (ll-lh)/np.sqrt(2)
    hi[0::2], hi[1::2] = (hl+hh)/np.sqrt(2), (hl-hh)/np.sqrt(2)
    out = np.empty((2*h, 2*w))
    out[:, 0::2], out[:, 1::2] = (lo+hi)/np.sqrt(2), (lo-hi)/np.sqrt(2)
    return out


def dct_matrix():
    k, n = np.indices((4, 4))
    c = np.sqrt(2/4)*np.cos(np.pi*(2*n+1)*k/8)
    c[0] /= np.sqrt(2)
    return c


C = dct_matrix()


def dct4(blocks):
    return C @ blocks @ C.T


def idct4(coefficients):
    return C.T @ coefficients @ C


def split_blocks(a):
    h, w = a.shape
    return a.reshape(h//4, 4, w//4, 4).transpose(0, 2, 1, 3).reshape(-1, 4, 4).copy()


def join_blocks(blocks, shape):
    h, w = shape
    return blocks.reshape(h//4, w//4, 4, 4).transpose(0, 2, 1, 3).reshape(h, w)


def pn_sequences(seed):
    """Deterministic zero-mean, orthogonal Gaussian PN pair, each RMS=1.

    Gram-Schmidt enforces the paper's 'uncorrelated' requirement. The Gaussian
    family, normalization and seed are documented choices, not supplied by paper.
    """
    n = len(MID_POSITIONS)
    p = np.random.default_rng(seed).normal(size=(2, n))
    p -= p.mean(axis=1, keepdims=True)
    p[0] /= np.linalg.norm(p[0])
    p[1] -= np.dot(p[1], p[0])*p[0]
    p[1] /= np.linalg.norm(p[1])
    return p*np.sqrt(n)


def block_positions(capacity, bits, repetitions):
    used = bits*repetitions
    if used > capacity or bits < 1:
        raise ValueError(f'Capacity={capacity} blocks; requested {bits} bits x {repetitions}')
    # Image-independent spatial sampling; full-capacity use reduces to row order.
    return np.floor((np.arange(used)+0.5)*capacity/used).astype(int)


def image_plane(image, mode, allow_out_of_range=False):
    a = np.asarray(image, dtype=np.float64)
    if a.size == 0 or not np.isfinite(a).all() or (not allow_out_of_range and ((a < 0).any() or (a > 255).any())):
        raise ValueError('Image must contain finite values in 0..255')
    if mode == 'gray' and a.ndim == 2:
        return a, a
    if mode == 'rgb_y' and a.ndim == 3 and a.shape[2] == 3:
        return a, a @ Y_WEIGHTS
    raise ValueError(f'Expected {"HxW grayscale" if mode == "gray" else "HxWx3 RGB"}')


def analyze_plane(plane, subband):
    if any(d % 16 for d in plane.shape):
        raise SynchronizationError('Height/width must be multiples of 16; no automatic resizing')
    branch = subband[:2]
    first = haar_dwt2(plane)
    second = haar_dwt2(first[branch])
    selected = second[branch]
    return first, second, dct4(split_blocks(selected)), selected.shape


def to_uint8(a):
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def embed(image, payload_bits, config=None):
    """Return (floating marked image, extraction settings, diagnostics).

    Call to_uint8 then save PNG for a disk experiment. Settings contain no host,
    message, predicted bits, attack coordinates or data-dependent positions.
    """
    cfg = config or Config()
    cfg.validate()
    bits = np.asarray(payload_bits)
    if bits.ndim != 1 or not np.isin(bits, [0, 1]).all():
        raise ValueError('payload_bits must be a 1D binary array')
    bits = bits.astype(np.uint8)
    a, plane = image_plane(image, cfg.color_mode)
    first, second, coefficients, shape = analyze_plane(plane, cfg.subband)
    positions = block_positions(len(coefficients), len(bits), cfg.repetitions)
    symbols = np.tile(bits, cfg.repetitions)
    pn = pn_sequences(cfg.pn_seed)
    u, v = MID_POSITIONS.T
    # Equations (3)-(4): X' = X + alpha * PN_bit. No forced-correct decoding.
    coefficients[positions[:, None], u[None, :], v[None, :]] += cfg.alpha*pn[symbols]
    branch = cfg.subband[:2]
    second[branch] = join_blocks(idct4(coefficients), shape)
    first[branch] = haar_idwt2(second)
    marked_y = haar_idwt2(first)
    # Adding the luminance residual equally to R/G/B preserves linear chroma
    # before clipping. It avoids integer YCbCr round trips on unchanged pixels.
    marked = a+(marked_y-plane)[..., None] if cfg.color_mode == 'rgb_y' else marked_y
    settings = {'method': 'alhaj2007_haar_dct4_v1', 'image_shape': list(a.shape),
                'payload_length': len(bits), 'config': asdict(cfg),
                'mid_positions': MID_POSITIONS.tolist(),
                'layout': 'floor((j+0.5)*capacity/(bits*repetitions)); tile bits'}
    diagnostic = {'capacity_bits': len(coefficients), 'useful_bits': len(bits),
                  'embedded_bit_instances': len(symbols),
                  'clipped_channel_values': int(((marked < 0) | (marked > 255)).sum())}
    return marked, settings, diagnostic


def extract(image, settings, *, allow_out_of_range=False):
    """Blind message decoding: only received image and agreed settings."""
    if settings.get('method') != 'alhaj2007_haar_dct4_v1':
        raise ValueError('Unknown method/settings version')
    if list(np.shape(image)) != settings['image_shape']:
        raise SynchronizationError('Image size changed; blind crop synchronization is not implemented')
    if settings['mid_positions'] != MID_POSITIONS.tolist():
        raise ValueError('Incompatible mid-band settings')
    cfg = Config(**settings['config'])
    cfg.validate()
    _, plane = image_plane(image, cfg.color_mode, allow_out_of_range=allow_out_of_range)
    _, _, coefficients, _ = analyze_plane(plane, cfg.subband)
    n = settings['payload_length']
    positions = block_positions(len(coefficients), n, cfg.repetitions)
    u, v = MID_POSITIONS.T
    values = coefficients[positions[:, None], u[None, :], v[None, :]]
    centered = values-values.mean(axis=1, keepdims=True)
    pn = pn_sequences(cfg.pn_seed)
    denominator = np.linalg.norm(centered, axis=1)*np.linalg.norm(pn[0])
    scores = np.divide(centered @ pn.T, denominator[:, None],
                       out=np.zeros((len(positions), 2)), where=denominator[:, None] > 1e-12)
    # Paper: choose 0 if corr0 > corr1, otherwise 1 (includes ties).
    if cfg.repetitions == 1:
        aggregate = scores
    else:
        aggregate = scores.reshape(cfg.repetitions, n, 2).mean(axis=0)
    bits = (aggregate[:, 1] >= aggregate[:, 0]).astype(np.uint8)
    return bits, {'correlation_scores': aggregate.tolist(),
                  'mean_absolute_margin': float(np.mean(np.abs(aggregate[:, 1]-aggregate[:, 0])))}
