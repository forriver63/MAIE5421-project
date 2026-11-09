"""Meta Revisited HiDDeN / JND; float32 input and MSB-first bits."""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from ..contracts import EncodeResult


def bits_of(value, width):
    return [(value >> i) & 1 for i in range(width - 1, -1, -1)]


def parse_logits(logits):
    bits = np.asarray(logits) > 0
    if bits.shape != (48,):
        raise ValueError(f'Expected 48 logits; got {bits.shape}')
    marker_errors = int(np.count_nonzero(bits[:16] != bits_of(0xA5C3, 16)))
    raw = int(''.join(str(int(b)) for b in bits[16:]), 2)
    accepted = marker_errors <= 1
    return dict(payload=raw if accepted else None, raw_payload=raw, accepted=accepted,
                status='decoded' if accepted else 'marker_reject', marker_errors=marker_errors,
                logits=np.asarray(logits).tolist())


class HiddenAdapter:
    def __init__(self, model_dir, strength=.45, device='cpu', protocol='native512'):
        if protocol not in ('native512', 'resize256'):
            raise ValueError(f'Unknown HiDDeN protocol: {protocol}')
        self.protocol = protocol
        self.device = torch.device(device)
        self.strength = strength
        self.encoder = torch.jit.load(str(Path(model_dir) / 'encoder_with_jnd_48b.torchscript.pt'), map_location=self.device).eval()
        self.decoder = torch.jit.load(str(Path(model_dir) / 'decoder_48b.torchscript.pt'), map_location=self.device).eval()
        self.mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]

    def _tensor(self, rgb):
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None].to(self.device) / 255

    @torch.inference_mode()
    def embed(self, rgb, payload, *, public_seed=0):
        pixels = self._tensor(rgb)
        low = F.interpolate(pixels, size=(256, 256), mode='bilinear', align_corners=False,
                            antialias=True) if self.protocol == 'resize256' else pixels
        x = (low - self.mean) / self.std
        message = torch.tensor(bits_of(0xA5C3, 16) + bits_of(payload, 32),
                               dtype=torch.float32, device=self.device)[None] * 2 - 1
        aux = self.encoder(x, message)
        if self.protocol == 'resize256':
            residual = self.strength * (aux - x) * self.std
            residual = F.interpolate(residual, size=pixels.shape[-2:], mode='bilinear',
                                     align_corners=False, antialias=True)
            out = (pixels + residual).clamp(0, 1)
        else:
            out = ((x + self.strength * (aux - x)) * self.std + self.mean).clamp(0, 1)
        image = (out[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()
        return EncodeResult(image, dict(native_bits=48, useful_bits=32, marker_hex='a5c3',
                                        lambda_w=self.strength))

    @torch.inference_mode()
    def extract(self, rgb):
        pixels = self._tensor(rgb)
        if self.protocol == 'resize256':
            pixels = F.interpolate(pixels, size=(256, 256), mode='bilinear',
                                   align_corners=False, antialias=True)
        x = (pixels - self.mean) / self.std
        logits = self.decoder(x).flatten().cpu().numpy()
        return parse_logits(logits)
