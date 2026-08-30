"""Embedding provider seam from docs/evaluation-spec.md §5."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np

_WORD_RE = re.compile(r"[\w-]+", flags=re.UNICODE)


@runtime_checkable
class EmbeddingsProvider(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class FakeEmbeddings:
    """Stable signed feature-hashed bag-of-words embeddings."""

    def __init__(self, dim: int = 64) -> None:
        if dim <= 0:
            raise ValueError("embedding dimension must be positive")
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            for token in _WORD_RE.findall(text.casefold()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                raw = int.from_bytes(digest, byteorder="big", signed=False)
                column = raw % self.dim
                sign = 1.0 if raw & 1 else -1.0
                vectors[row, column] += sign
            norm = np.linalg.norm(vectors[row])
            if norm:
                vectors[row] /= norm
        return vectors
