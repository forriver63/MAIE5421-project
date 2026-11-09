from dataclasses import dataclass, field
from typing import Protocol
import numpy as np


@dataclass
class EncodeResult:
    image: np.ndarray
    metadata: dict = field(default_factory=dict)


@dataclass
class DecodeResult:
    payload: int | None
    raw_payload: int | None
    accepted: bool
    status: str
    diagnostics: dict = field(default_factory=dict)


class ModelAdapter(Protocol):
    def embed(self, rgb: np.ndarray, payload: int, *, public_seed: int) -> EncodeResult: ...
    def extract(self, rgb: np.ndarray) -> dict: ...
