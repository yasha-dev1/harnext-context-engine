"""Persisted lexical/dense/hybrid retrieval and real cross-encoder reranking."""

from __future__ import annotations

import os
import re
from pathlib import Path

import httpx
import numpy as np
from rank_bm25 import BM25Okapi

from .config import RetrievalConfig, digest


def tokens(text: str) -> list[str]:
    return re.findall(r"[\w:-]+", text.casefold())


def model_fingerprint(model) -> dict:
    directory = Path(model.model._model_dir)
    hashes = {
        str(path.relative_to(directory)): digest(path.read_bytes())
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(directory).parts)
    }
    if not hashes:
        raise ValueError("neural model files could not be fingerprinted")
    return {"content_sha256": digest(hashes), "files": hashes}


class Neural:
    def __init__(self, config: RetrievalConfig, *, embed: bool, rerank: bool):
        self.config = config
        self.encoder = None
        self.reranker = None
        self.pins = {}
        cfg = config.embeddings
        if embed and cfg.provider == "fastembed":
            from fastembed import TextEmbedding

            self.encoder = TextEmbedding(
                model_name=cfg.model, cache_dir=cfg.cache_dir, threads=cfg.threads
            )
            self.pins["embedding"] = model_fingerprint(self.encoder)
            if (
                cfg.revision != "content-hash-at-build"
                and cfg.revision != self.pins["embedding"]["content_sha256"]
            ):
                raise ValueError("embedding content does not match the configured revision hash")
        elif embed:
            if cfg.revision == "content-hash-at-build":
                raise ValueError("remote embedding deployments require an explicit revision label")
            self.pins["embedding"] = {
                "provider": cfg.provider,
                "model": cfg.model,
                "revision": cfg.revision,
                "base_url": cfg.base_url,
            }
        if rerank:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self.reranker = TextCrossEncoder(
                config.reranker_model, cache_dir=cfg.cache_dir, threads=cfg.threads
            )
            self.pins["reranker"] = model_fingerprint(self.reranker)

    def embed(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, 0))
        if self.encoder is not None:
            fn = self.encoder.query_embed if query else self.encoder.passage_embed
            rows = list(fn(texts))
        else:
            cfg = self.config.embeddings
            key = os.environ.get(cfg.api_key_env or "")
            if not key:
                raise ValueError("configured embedding credential is unavailable")
            if cfg.base_url is None:
                raise ValueError("embedding base_url is required")
            response = httpx.post(
                f"{cfg.base_url.rstrip('/')}/embeddings",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": cfg.model, "input": texts},
                timeout=cfg.timeout_s,
            )
            response.raise_for_status()
            records = sorted(response.json()["data"], key=lambda item: item["index"])
            if [r["index"] for r in records] != list(range(len(texts))):
                raise ValueError("embedding response indices differ from requested documents")
            rows = [row["embedding"] for row in records]
        matrix = np.asarray(rows, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts) or not np.isfinite(matrix).all():
            raise ValueError("invalid embedding response")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("zero embedding vector")
        return matrix / norms


def chunks(documents: dict[str, str], config: RetrievalConfig) -> list[dict]:
    result = []
    for path, text in sorted(documents.items()):
        data = text.encode()
        size = len(data) if config.chunking == "document" else config.chunk_bytes
        start = 0
        while start < len(data):
            body = data[start : start + max(1, size)].decode(errors="ignore")
            end = start + len(body.encode())
            if end <= start:
                raise ValueError("chunk cap cannot accommodate a UTF-8 character")
            result.append(
                {
                    "id": digest([path, start, end, body]),
                    "path": path,
                    "start": start,
                    "end": end,
                    "text": body,
                }
            )
            if end == len(data):
                break
            start = max(start + 1, end - config.overlap_bytes)
            while start < len(data) and data[start] & 0xC0 == 0x80:
                start += 1
    return result


class SearchIndex:
    def __init__(self, config: RetrievalConfig, records: list[dict], *, vectors=None, pins=None):
        self.config, self.records = config, records
        self.neural = Neural(
            config,
            embed=config.method in {"dense", "hybrid"},
            rerank=config.reranker == "cross_encoder",
        )
        if pins is not None and pins != self.neural.pins:
            raise ValueError("model bytes/deployment pins differ from the built snapshot")
        corpus = [tokens(record["text"]) or ["__empty__"] for record in records]
        self.lexical = BM25Okapi(corpus, k1=config.bm25_k1, b=config.bm25_b) if corpus else None
        self.vectors = (
            np.asarray(vectors, dtype=float)
            if vectors is not None
            else self.neural.embed([r["text"] for r in records])
            if config.method in {"dense", "hybrid"}
            else None
        )
        if self.vectors is not None and len(self.vectors) != len(records):
            raise ValueError("persisted vector/chunk count mismatch")

    def search(self, query: str) -> list[dict]:
        if not query.strip() or not self.records:
            return []
        cfg = self.config
        if self.lexical is None:
            return []
        lexical = self.lexical.get_scores(tokens(query))
        lex = sorted(range(len(self.records)), key=lambda i: (-lexical[i], self.records[i]["id"]))
        dense = []
        cosine: list[float] = []
        if cfg.method in {"dense", "hybrid"}:
            vector = self.neural.embed([query], query=True)[0]
            cosine = list(map(float, self.vectors @ vector))
            dense = sorted(
                range(len(self.records)), key=lambda i: (-cosine[i], self.records[i]["id"])
            )
        scores: dict[int, float]
        if cfg.method == "literal":
            scores = {
                i: float(self.records[i]["text"].casefold().count(query.casefold())) for i in lex
            }
            chosen = sorted(
                (i for i in lex if scores[i] > 0), key=lambda i: (-scores[i], self.records[i]["id"])
            )[: cfg.candidates]
        elif cfg.method == "bm25":
            chosen, scores = lex[: cfg.candidates], dict(enumerate(map(float, lexical)))
        elif cfg.method == "dense":
            chosen, scores = dense[: cfg.candidates], dict(enumerate(cosine))
        else:
            scores = {}
            for ranking in (lex[: cfg.candidates], dense[: cfg.candidates]):
                for rank, index in enumerate(ranking, 1):
                    scores[index] = scores.get(index, 0.0) + 1 / (cfg.rrf_k + rank)
            chosen = sorted(scores, key=lambda i: (-scores[i], self.records[i]["id"]))[
                : cfg.candidates
            ]
        if self.neural.reranker is not None and chosen:
            reranked = list(
                self.neural.reranker.rerank(query, [self.records[i]["text"] for i in chosen])
            )
            if len(reranked) != len(chosen) or not np.isfinite(reranked).all():
                raise ValueError("invalid reranker response")
            scores = dict(zip(chosen, map(float, reranked), strict=True))
            chosen.sort(key=lambda i: (-scores[i], self.records[i]["id"]))
        return [{**self.records[i], "score": float(scores[i])} for i in chosen[: cfg.top_k]]

    def serializable(self):
        return {
            "records": self.records,
            "vectors": self.vectors.tolist() if self.vectors is not None else None,
            "pins": self.neural.pins,
            "config": self.config.model_dump(mode="json"),
        }
