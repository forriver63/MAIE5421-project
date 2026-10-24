"""High-level embed/detect API shared by the CLI and the web service.

A 32-bit payload is interpreted as four bytes A,B,C,D (big-endian); the frozen
pipeline is: repetition -> packet (ID+data+CRC5) -> BCH -> 8 localized WAM words.
"""
from __future__ import annotations

import torch
from PIL import Image, ImageOps

from .attacks import tensor_rgb
from .codec import outer, packet
from .eval import decode_global, infer_packets
from .masks import make_masks
from .wam import embed as wam_embed
from .wam import model as wam_model
from .wam.transforms import default_transform

_REGION_COUNT = 8


def payload_to_source(payload: int) -> tuple[int, int, int, int]:
    if not 0 <= payload < (1 << 32):
        raise ValueError("payload must be a 32-bit unsigned integer")
    return tuple((payload >> (8 * i)) & 0xFF for i in range(3, -1, -1))


def source_to_payload(source) -> int:
    return sum(int(b) << (8 * (3 - i)) for i, b in enumerate(source))


def _messages_for(source):
    symbols = outer.encode_repetition(source)
    messages = torch.tensor([packet.encode_bits(i, s) for i, s in enumerate(symbols)],
                            dtype=torch.float32)
    return symbols, messages


def embed_image(image: Image.Image, payload: int, wam=None) -> Image.Image:
    wam = wam if wam is not None else wam_model.load_default_model()
    source = payload_to_source(payload)
    symbols, messages = _messages_for(source)
    # The validated masks are defined on a fixed 512x512 canvas.
    image = ImageOps.fit(image.convert("RGB"), (512, 512),
                         method=Image.Resampling.LANCZOS)
    tensor = default_transform(image).unsqueeze(0)
    masks = make_masks(_REGION_COUNT, tensor.device)
    watermarked, _ = wam_embed.embed_messages(wam, tensor, messages, masks)
    array = (tensor_rgb(watermarked) * 255).round().clip(0, 255).astype("uint8")
    return Image.fromarray(array, mode="RGB")


def detect_image(image: Image.Image, wam=None) -> dict:
    wam = wam if wam is not None else wam_model.load_default_model()
    tensor = default_transform(image.convert("RGB")).unsqueeze(0)
    detection = infer_packets(wam, tensor)
    decoded = decode_global(detection)
    payload = source_to_payload(decoded["global_payload"]) if decoded["global_payload"] else None
    return {"status": decoded["global_status"], "payload": payload,
            "valid_ids": decoded["unique_valid_ids"],
            "detected_clusters": len(detection["clusters"]),
            "conflicting_ids": decoded["conflicting_ids"]}
