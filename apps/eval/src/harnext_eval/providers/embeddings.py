"""Embedding provider seam from docs/evaluation-spec.md §5."""

from __future__ import annotations

import hashlib
import importlib
import re
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

_WORD_RE = re.compile(r"[\w-]+", flags=re.UNICODE)
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]+[-_:]\d+\b")


def _features(text: str) -> list[tuple[str, float]]:
    """Return words, token n-grams, and identifier character n-grams."""

    tokens = _WORD_RE.findall(text.casefold())
    features = [(f"w:{token}", 1.0) for token in tokens]
    for size, weight in ((2, 1.35), (3, 1.15)):
        features.extend(
            (f"t{size}:{' '.join(tokens[start : start + size])}", weight)
            for start in range(len(tokens) - size + 1)
        )
    for identifier in _IDENTIFIER_RE.findall(text.casefold()):
        compact = re.sub(r"[-_:]", "", identifier)
        # IDs carry the join intent in E2 queries; give the full identifier
        # enough mass to survive noisy raw-event payloads and a 64-d hash.
        features.append((f"id:{identifier}", 12.0))
        features.extend(
            (f"id4:{compact[start : start + 4]}", 1.5)
            for start in range(len(compact) - 3)
        )
    return features


@runtime_checkable
class EmbeddingsProvider(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class FakeEmbeddings:
    """Stable signed feature hashing over words and lexical n-grams."""

    def __init__(self, dim: int = 64) -> None:
        if dim <= 0:
            raise ValueError("embedding dimension must be positive")
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            for feature, weight in _features(text):
                digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
                raw = int.from_bytes(digest, byteorder="big", signed=False)
                column = raw % self.dim
                sign = 1.0 if raw & 1 else -1.0
                vectors[row, column] += sign * weight
            norm = np.linalg.norm(vectors[row])
            if norm:
                vectors[row] /= norm
        return vectors


def _normalised_matrix(rows: Sequence[Sequence[float]], expected_rows: int) -> np.ndarray:
    """Validate a remote embedding response and return cosine-ready vectors."""

    vectors = np.asarray(rows, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != expected_rows:
        raise RuntimeError(
            "embedding provider returned an invalid matrix shape: "
            f"expected {expected_rows} rows, got {vectors.shape}"
        )
    if not np.isfinite(vectors).all():
        raise RuntimeError("embedding provider returned non-finite values")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.divide(vectors, norms, out=vectors, where=norms != 0)
    return vectors


class VoyageEmbeddings:
    """Pinned Voyage adapter whose optional SDK is imported only on use."""

    provider_id = "voyage"

    def __init__(
        self,
        model: str,
        revision: str,
        *,
        api_key: str | None = None,
    ) -> None:
        if not model.strip() or not revision.strip():
            raise ValueError("Voyage embeddings require non-empty model and revision")
        self.model_id = model
        self.model_revision = revision
        self.api_key = api_key

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float64)
        try:
            voyageai = importlib.import_module("voyageai")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "VoyageEmbeddings requires the optional 'voyageai' SDK; "
                "install it to use this provider"
            ) from exc
        client = voyageai.Client(api_key=self.api_key)
        response = client.embed(texts, model=self.model_id)
        return _normalised_matrix(response.embeddings, len(texts))


class OpenAIEmbeddings:
    """Pinned OpenAI adapter whose optional SDK is imported only on use."""

    provider_id = "openai"

    def __init__(
        self,
        model: str,
        revision: str,
        *,
        api_key: str | None = None,
    ) -> None:
        if not model.strip() or not revision.strip():
            raise ValueError("OpenAI embeddings require non-empty model and revision")
        self.model_id = model
        self.model_revision = revision
        self.api_key = api_key

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float64)
        try:
            openai = importlib.import_module("openai")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "OpenAIEmbeddings requires the optional 'openai' SDK; "
                "install it to use this provider"
            ) from exc
        client = openai.OpenAI(api_key=self.api_key)
        response = client.embeddings.create(input=texts, model=self.model_id)
        data: list[Any] = sorted(
            response.data,
            key=lambda item: int(getattr(item, "index", 0)),
        )
        return _normalised_matrix(
            [item.embedding for item in data],
            len(texts),
        )
