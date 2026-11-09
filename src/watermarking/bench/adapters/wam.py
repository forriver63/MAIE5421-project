import numpy as np
import time
import torch
from PIL import Image
from ...codec.packet import encode_packet, packet16
from ...codec.outer import encode_repetition
from ...codec.bch import int_to_bits
from ...masks import make_masks
from ...wam.model import load_default_model
from ...wam.transforms import default_transform, unnormalize_img
from ...wam.embed import embed_messages
from ...wam.detect import detect_blind
from ..contracts import EncodeResult
from ..clustering import binary_dbscan


class WamAdapter:
    def __init__(self, variant='full', device='cpu'):
        if variant not in ('full', 'e1'):
            raise ValueError(variant)
        self.device = torch.device(device)
        self.variant = variant
        self.model = load_default_model(self.device)
        self.masks = make_masks(8, self.device)

    @torch.inference_mode()
    def embed(self, rgb, payload, *, public_seed=0):
        source = tuple(payload.to_bytes(4, 'big'))
        symbols = encode_repetition(source)
        rng = np.random.default_rng(public_seed)
        words = [encode_packet(i, s) if self.variant == 'full' else
                 (packet16(i, s) << 16) | int(rng.integers(0, 65536))
                 for i, s in enumerate(symbols)]
        messages = torch.tensor([int_to_bits(w) for w in words], dtype=torch.float32, device=self.device)
        x = default_transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        y, _ = embed_messages(self.model, x, messages, self.masks)
        image = (unnormalize_img(y)[0].clamp(0, 1).permute(1, 2, 0) * 255).round().byte().cpu().numpy()
        return EncodeResult(image, dict(words=words, symbols=list(symbols), native_bits=256,
                                        useful_bits=32, variant=self.variant))

    @torch.inference_mode()
    def extract(self, rgb):
        x = default_transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        cluster_time = [0.0]
        def measured_cluster(*args, **kwargs):
            start = time.perf_counter()
            result = binary_dbscan(*args, **kwargs)
            cluster_time[0] += time.perf_counter() - start
            return result
        detection = detect_blind(self.model, x, clusterer=measured_cluster)
        if detection['clustering_error']:
            raise RuntimeError(detection['clustering_error'])
        clusters = [{k: v for k, v in c.items() if k != 'mask'} for c in detection['clusters']]
        return dict(clusters=clusters, labels=detection['labels'].numpy().astype(np.int16),
                    detection_seconds=detection['runtime_seconds'], clustering_error=None,
                    clustering_seconds=cluster_time[0])

    def oracle_masks(self):
        return self.masks[:, 0].cpu().numpy().astype(bool)
