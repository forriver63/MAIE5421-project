"""RGB/32-bit project adapter for Zhang2017. Gray paper core is watermark.py.

Keep spectral-norm ranking, quantization targets and Eq10 spatial approximation.
Arbitrary bit vectors use no Arnold transform (the paper's square-logo transform).
Selected image-dependent positions remain explicit extraction side information.
"""
from dataclasses import dataclass, asdict
import numpy as np
from watermark import to_blocks, from_blocks, target_values


@dataclass(frozen=True)
class Config:
    delta: float = 12.0
    block_size: int = 4
    color_mode: str = 'rgb_y'

    def validate(self):
        if not np.isfinite(self.delta) or self.delta <= 0:
            raise ValueError('delta must be positive and finite')
        if not isinstance(self.block_size, int) or self.block_size < 1:
            raise ValueError('block_size must be a positive integer')
        if self.color_mode != 'rgb_y':
            raise ValueError('This adapter accepts RGB; use watermark.py for gray')


class SynchronizationError(ValueError):
    pass


def plane(image):
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 3 or a.shape[2] != 3 or a.size == 0:
        raise ValueError('Expected HxWx3 RGB')
    if not np.isfinite(a).all() or (a < 0).any() or (a > 255).any():
        raise ValueError('Expected finite 0..255 RGB values')
    return a, a @ np.array([.299, .587, .114])


def to_uint8(image):
    # Same serialization as the DWT-DCT team32 adapter.
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def embed(image, payload_bits, config=None):
    cfg = config or Config()
    cfg.validate()
    bits = np.asarray(payload_bits)
    if bits.ndim != 1 or bits.size == 0 or not np.isin(bits, [0, 1]).all():
        raise ValueError('Expected nonempty binary vector')
    a, y = plane(image)
    blocks = to_blocks(y, cfg.block_size)
    if len(bits) > len(blocks):
        raise ValueError('Insufficient capacity')
    norms = np.linalg.norm(blocks, ord=2, axis=(-2, -1))
    positions = np.argsort(-norms, kind='stable')[:len(bits)]
    targets = target_values(norms[positions], bits, cfg.delta)
    blocks[positions] += ((targets-norms[positions])/cfg.block_size)[:, None, None]
    actual = np.linalg.norm(blocks[positions], ord=2, axis=(-2, -1))
    marked_y = from_blocks(blocks, y.shape)
    marked = a+(marked_y-y)[..., None]
    settings = {'method': 'zhang2017_rgb32_eq10_v1', 'image_shape': list(a.shape),
                'payload_length': len(bits), 'config': asdict(cfg),
                'positions': positions.tolist(), 'arnold_iterations': 0,
                'position_policy': 'descending original spectral norm; stable row-major ties'}
    diagnostic = {'capacity_bits': len(blocks), 'useful_bits': len(bits),
                  'embedded_bit_instances': len(bits),
                  'clipped_channel_values': int(((marked < 0) | (marked > 255)).sum()),
                  'mean_abs_target_error_float': float(np.mean(np.abs(actual-targets))),
                  'max_abs_target_error_float': float(np.max(np.abs(actual-targets)))}
    return marked, settings, diagnostic


def extract(image, settings):
    if settings.get('method') != 'zhang2017_rgb32_eq10_v1':
        raise ValueError('Incompatible extraction settings')
    if list(np.shape(image)) != settings['image_shape']:
        raise SynchronizationError('Image size changed; blind crop synchronization is not implemented')
    cfg = Config(**settings['config'])
    cfg.validate()
    _, y = plane(image)
    blocks = to_blocks(y, cfg.block_size)
    positions = np.asarray(settings['positions'], dtype=int)
    if (positions.ndim != 1 or len(positions) != settings['payload_length'] or
            len(np.unique(positions)) != len(positions) or
            (positions < 0).any() or (positions >= len(blocks)).any()):
        raise ValueError('Invalid selected block positions')
    norms = np.linalg.norm(blocks[positions], ord=2, axis=(-2, -1))
    bits = (np.ceil(norms/cfg.delta).astype(np.int64) % 2).astype(np.uint8)
    return bits, {}
