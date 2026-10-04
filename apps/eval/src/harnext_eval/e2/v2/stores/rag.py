"""R-hyb: hybrid retrieval over the event stream with the blueprint parameters (D08–D12).

Token chunks (≤512 tokens of the embedding model's own tokenizer, one event per chunk
unless longer, no same-entity bundling), BM25 k1=1.2 b=0.75, dense bge-small via local
fastembed, RRF k=60 over 40 candidates per ranker, cross-encoder rerank of the fused 40
→ top 10. Model pinning is reused from `harnext_builder.strategies.retrieval`.

One cumulative, append-only index per arm, sharded by month (`index/<YYYY-MM>.*`):
`chunks.jsonl` (chunk records), `emb.f32` (raw float32 rows, memory-mapped by `search`)
and `lex.npz` (byte offsets of the chunk lines, chunk ids, and the BM25 postings of the
shard: per-chunk term ids + counts over a shard-local vocabulary). A shard is written
once, atomically, so the build holds one month in memory and a killed build redoes at
most one month. The stream is time-ordered, so checkpoint X = the shards ≤ X.

`export/<YYYY-MM>/` is therefore tiny: `meta.json` plus HARD LINKS to the shards ≤ X
(`shards/`), no copy. The directory mounted into a trial holds nothing from after the
checkpoint — neither text, vectors nor vocabulary — and `search` derives every BM25
statistic (N, document frequencies, average length) from those shards alone.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from bisect import bisect_left
from collections import Counter
from collections.abc import Iterable
from functools import lru_cache
from itertools import groupby
from pathlib import Path
from typing import Any, cast

import numpy as np
from harnext_builder.strategies.config import EncoderConfig, RetrievalConfig
from harnext_builder.strategies.retrieval import Neural, tokens
from scipy import sparse

from harnext_eval.e2.v2.stores.base import ROOT, lock, month, next_month, save_json, unlock

MODELS = ROOT.parents[1] / ".harnext/models"
EMBED_SLICE = 2048  # chunks handed to the encoder at once (sorted by length inside)
RERANK_BATCH = 8  # same reason: 40 pairs of ≤512 tokens in one batch cost ~2 GB of buffers
EMBED_BATCH = 16  # ONNX batch: attention memory grows with batch × longest² — keep small
MAX_TOKENS, SPECIAL_TOKENS, OVERLAP_TOKENS, HEADER_TOKENS = 512, 2, 32, 96


def retrieval_config(top_k: int = 10) -> RetrievalConfig:
    return RetrievalConfig.model_validate(
        {
            "method": "hybrid",
            "chunking": "fixed",  # informational: chunking is by tokens, below
            "candidates": 40,
            "top_k": top_k,
            "rrf_k": 60,
            "bm25_k1": 1.2,
            "bm25_b": 0.75,
            "reranker": "cross_encoder",
            "embeddings": EncoderConfig(
                cache_dir=str(MODELS), threads=max(2, (os.cpu_count() or 4) // 2)
            ).model_dump(),
        }
    )


class LocalNeural(Neural):
    """The repo's pinned fastembed encoder + cross-encoder, plus the encoder's tokenizer
    (truncation off) as the unit of chunk length."""

    def __init__(self, config: RetrievalConfig):
        from tokenizers import Tokenizer

        super().__init__(config, embed=True, rerank=True)
        assert self.encoder is not None and self.reranker is not None
        encoder, reranker = cast(Any, self.encoder.model), cast(Any, self.reranker.model)
        self.tokenizer = Tokenizer.from_str(encoder.tokenizer.to_str())
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.pins["embedding"] |= {
            "model": config.embeddings.model,
            "revision": Path(encoder._model_dir).name,
        }
        self.pins["reranker"] |= {
            "model": config.reranker_model,
            "revision": Path(reranker._model_dir).name,
        }

    def spans(self, text: str) -> list[tuple[int, int]]:
        return self.tokenizer.encode(text, add_special_tokens=False).offsets

    def passages(self, texts: list[str]) -> np.ndarray:
        """Normalised float32 passage vectors. Texts go to the encoder sorted by length in
        small batches: a batch is padded to its longest member, so mixed lengths at
        fastembed's default batch of 256 cost gigabytes of attention buffers and most of
        the time is spent on padding."""
        assert self.encoder is not None
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        rows = self.encoder.passage_embed([texts[i] for i in order], batch_size=EMBED_BATCH)
        matrix = np.asarray(list(rows), dtype=np.float64)
        if matrix.shape[0] != len(texts) or not np.isfinite(matrix).all():
            raise ValueError("invalid embedding response")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("zero embedding vector")
        out = np.empty(matrix.shape, dtype=np.float32)
        out[order] = matrix / norms
        return out


class _Overlap:
    def rerank(self, query: str, texts: list[str], batch_size: int = 0) -> list[float]:
        wanted = set(tokens(query))
        return [len(wanted & set(tokens(text))) / (len(wanted) or 1) for text in texts]


class HashNeural:
    """Offline stand-in (`--builder fake`, tests): hashed bag-of-words vectors, token-overlap
    reranker, whitespace tokens. Same interface as `LocalNeural`, no model files."""

    pins = {"embedding": {"model": "fake-hash-64"}, "reranker": {"model": "fake-overlap"}}
    reranker = _Overlap()

    def embed(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        matrix = np.full((len(texts), 64), 1e-6)
        for row, text in enumerate(texts):
            for token in tokens(text):
                matrix[row, int(hashlib.sha256(token.encode()).hexdigest(), 16) % 64] += 1
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    def passages(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts).astype(np.float32)

    def spans(self, text: str) -> list[tuple[int, int]]:
        import re

        return [m.span() for m in re.finditer(r"\S+", text)]


def chunk(event: dict, neural) -> list[dict]:
    """One chunk per event; an event longer than the budget is cut on token boundaries.
    Every chunk repeats a one-line header so a part still names its event."""
    entities = " ".join(event.get("entities") or [])
    header = f"[{event['time']}] {event['type']} {entities} — {event.get('title') or ''} ({event.get('actor') or 'unknown'})"
    head_spans = neural.spans(header)[:HEADER_TOKENS]
    header = header[: head_spans[-1][1]] if head_spans else ""
    budget = MAX_TOKENS - SPECIAL_TOKENS - len(head_spans) - 1
    text = event.get("text") or ""
    spans = neural.spans(text)
    cuts, start = [], 0
    while start < len(spans) or not cuts:
        end = min(start + budget, len(spans))
        cuts.append((spans[start][0] if spans else 0, spans[end - 1][1] if spans else 0))
        if end >= len(spans):
            break
        start = end - OVERLAP_TOKENS
    return [
        {
            "id": f"{event['id']}#{part}",
            "event_id": event["id"],
            "part": part,
            "parts": len(cuts),
            "time": event["time"],
            "type": event["type"],
            "entities": event.get("entities") or [],
            "title": event.get("title"),
            "text": f"{header}\n{text[a:b]}",
        }
        for part, (a, b) in enumerate(cuts)
    ]


class SparseBM25:
    """`rank_bm25.BM25Okapi` scores (same idf, epsilon floor, k1, b) from a sparse
    term×chunk matrix: memory and query time stay bounded at ~10⁶ chunks, where the
    pure-Python original keeps one dict per chunk and loops over all of them per query."""

    def __init__(self, corpus: Iterable[list[str]], k1: float, b: float, epsilon: float = 0.25):
        vocabulary: dict[str, int] = {}
        terms, counts, indptr, lengths = [], [], [0], []
        for words in corpus:
            lengths.append(len(words))
            for word, count in Counter(words).items():
                terms.append(vocabulary.setdefault(word, len(vocabulary)))
                counts.append(count)
            indptr.append(len(terms))
        self._finish(
            vocabulary,
            np.asarray(terms, dtype=np.int32),
            np.asarray(counts, dtype=np.uint16),
            np.asarray(indptr, dtype=np.int64),
            np.asarray(lengths, dtype=np.float64),
            k1, b, epsilon,
        )  # fmt: skip

    @classmethod
    def from_postings(cls, vocabulary, terms, counts, indptr, lengths, k1, b, epsilon=0.25):
        """From per-chunk postings (chunk `d` holds `terms[indptr[d]:indptr[d+1]]`)."""
        self = cls.__new__(cls)
        self._finish(vocabulary, terms, counts, indptr, np.asarray(lengths, np.float64), k1, b, epsilon)  # fmt: skip
        return self

    def _finish(self, vocabulary, terms, counts, indptr, lengths, k1, b, epsilon):
        shape = (len(vocabulary), len(lengths))
        by_chunk = sparse.csc_matrix((counts, terms, indptr), shape=shape)
        self.tf = by_chunk.tocsr()  # term rows; counts stay uint16, 6 bytes per posting
        del by_chunk
        self.vocabulary, self.k1 = vocabulary, k1
        self.documents, self.average_length = len(lengths), float(lengths.mean())
        self.norm = k1 * (1 - b + b * lengths / lengths.mean())
        self.containing = np.diff(self.tf.indptr)
        idf = np.log(len(lengths) - self.containing + 0.5) - np.log(self.containing + 0.5)
        self.idf = np.where(idf < 0, epsilon * idf.mean(), idf)

    def statistics(self) -> dict:
        """Everything the scores depend on besides the postings — for the as-of test."""
        return {
            "documents": self.documents,
            "average_length": self.average_length,
            "df": {word: int(self.containing[term]) for word, term in self.vocabulary.items()},
            "idf": {word: float(self.idf[term]) for word, term in self.vocabulary.items()},
        }

    def get_scores(self, query: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.norm))
        for word in query:  # repeated query words count repeatedly, as in BM25Okapi
            term = self.vocabulary.get(word)
            if term is None:
                continue
            start, end = self.tf.indptr[term], self.tf.indptr[term + 1]
            chunks = self.tf.indices[start:end]
            freq = self.tf.data[start:end].astype(np.float64)
            scores[chunks] += self.idf[term] * freq * (self.k1 + 1) / (freq + self.norm[chunks])
        return scores


def id_rank(ids: list[str]) -> np.ndarray:
    """Position of every chunk id in sorted order: the tie-break key, as integers."""
    rank = np.empty(len(ids), dtype=np.int64)
    rank[sorted(range(len(ids)), key=ids.__getitem__)] = np.arange(len(ids))
    return rank


def best(scores: np.ndarray, rank: np.ndarray, n: int) -> list[int]:
    """Indices of the `n` highest scores, ties broken by chunk id (`rank`) — `SearchIndex`'s
    ordering without sorting the whole corpus in Python."""
    if len(scores) > n:
        floor = np.partition(scores, -n)[-n]
        pool = np.flatnonzero(scores >= floor)
    else:
        pool = np.arange(len(scores))
    return pool[np.lexsort((rank[pool], -scores[pool]))][:n].tolist()


DENSE_BLOCK = 65536  # rows per matrix-vector product: bounds what a query pages in at once


class Index:
    """Hybrid search over an exported checkpoint: the ranking of
    `harnext_builder.strategies.retrieval.SearchIndex` (RRF over the top `candidates` of
    BM25 and cosine, cross-encoder rerank of the fused list, ties by chunk id). Only the
    export's own shards are read: postings into one sparse matrix, vectors memory-mapped,
    chunk records fetched by byte offset for the fused candidates alone."""

    def __init__(self, config: RetrievalConfig, root: Path, meta: dict, neural):
        self.config, self.neural, self.root = config, neural, root
        shards = [s for s in meta["shards"] if s["chunks"]]
        self.names = [s["month"] for s in shards]
        self.starts = np.cumsum([0] + [s["chunks"] for s in shards])
        total, postings = int(self.starts[-1]), sum(s["postings"] for s in shards)
        vocabulary: dict[str, int] = {}
        terms = np.empty(postings, dtype=np.int32)
        counts = np.empty(postings, dtype=np.uint16)
        indptr = np.zeros(total + 1, dtype=np.int64)
        lengths = np.empty(total, dtype=np.float64)
        self.ids: list[str] = []
        self.offsets, self.vectors, at = [], [], 0
        for shard, first in zip(shards, self.starts, strict=False):
            size = shard["chunks"]
            with np.load(root / "shards" / f"{shard['month']}.lex.npz") as lex:
                words = json.loads(lex["words"].tobytes())
                to_global = np.fromiter(
                    (vocabulary.setdefault(w, len(vocabulary)) for w in words), np.int32, len(words)
                )
                local = lex["terms"]
                terms[at : at + len(local)] = to_global[local]
                counts[at : at + len(local)] = lex["counts"]
                indptr[first + 1 : first + size + 1] = lex["indptr"][1:] + at
                lengths[first : first + size] = lex["lengths"]
                self.offsets.append(lex["offsets"])
                self.ids += lex["ids"].tobytes().decode().split("\n")
                at += len(local)
            vectors = np.memmap(root / "shards" / f"{shard['month']}.emb.f32", np.float32, "r")
            self.vectors.append(vectors.reshape(size, -1))
        if at != postings or len(self.ids) != total:
            raise ValueError("persisted shard sizes differ from meta.json")
        self.rank = id_rank(self.ids)
        self.order = np.argsort(self.rank)  # chunk indices in id order, for `event`
        self.lexical = (
            SparseBM25.from_postings(vocabulary, terms, counts, indptr, lengths,
                                     config.bm25_k1, config.bm25_b) if total else None
        )  # fmt: skip

    def __len__(self) -> int:
        return len(self.ids)

    def record(self, index: int) -> dict:
        shard = int(np.searchsorted(self.starts, index, side="right")) - 1
        offsets, local = self.offsets[shard], index - int(self.starts[shard])
        with (self.root / "shards" / f"{self.names[shard]}.chunks.jsonl").open("rb") as handle:
            handle.seek(int(offsets[local]))
            return json.loads(handle.read(int(offsets[local + 1] - offsets[local])))

    def dense(self, vector: np.ndarray) -> np.ndarray:
        scores = np.empty(len(self.ids), dtype=np.float64)
        for shard, first in zip(self.vectors, self.starts, strict=False):
            for at in range(0, len(shard), DENSE_BLOCK):  # float32 products, block by block
                block = shard[at : at + DENSE_BLOCK]
                scores[first + at : first + at + len(block)] = block @ vector
        return scores

    def event(self, key: str) -> list[dict]:
        """The chunks of one event (or the one chunk with that id), in part order."""
        ordered = self.order
        at = bisect_left(ordered, key, key=lambda i: self.ids[i])
        found = []
        while at < len(ordered) and (
            (name := self.ids[ordered[at]]) == key or name.startswith(f"{key}#")
        ):
            found.append(self.record(int(ordered[at])))
            at += 1
        return sorted(
            (r for r in found if key in (r["id"], r["event_id"])), key=lambda r: r["part"]
        )

    def search(self, query: str, k: int | None = None) -> list[dict]:
        if not query.strip() or self.lexical is None:
            return []
        cfg = self.config
        lexical = best(self.lexical.get_scores(tokens(query)), self.rank, cfg.candidates)
        vector = self.neural.embed([query], query=True)[0].astype(np.float32)
        dense = best(self.dense(vector), self.rank, cfg.candidates)
        fused: dict[int, float] = {}
        for ranking in (lexical, dense):
            for rank, index in enumerate(ranking, 1):
                fused[index] = fused.get(index, 0.0) + 1 / (cfg.rrf_k + rank)
        chosen = sorted(fused, key=lambda i: (-fused[i], self.rank[i]))[: cfg.candidates]
        records = [self.record(i) for i in chosen]
        scores = list(
            self.neural.reranker.rerank(
                query, [r["text"] for r in records], batch_size=RERANK_BATCH
            )
        )
        if len(scores) != len(chosen) or not np.isfinite(scores).all():
            raise ValueError("invalid reranker response")
        order = sorted(range(len(chosen)), key=lambda n: (-scores[n], self.rank[chosen[n]]))
        return [{**records[n], "score": float(scores[n])} for n in order[: k or cfg.top_k]]


def neural_for(builder: str):
    return HashNeural() if builder == "fake" else LocalNeural(retrieval_config())


def is_export(index_dir: Path) -> bool:
    meta = index_dir / "meta.json"
    return meta.exists() and "shards" in json.loads(meta.read_text())


@lru_cache(maxsize=2)
def load(index_dir: str) -> Index:
    root = Path(index_dir)
    meta = json.loads((root / "meta.json").read_text())
    neural = neural_for(meta["builder"])
    if {k: v.get("content_sha256") for k, v in neural.pins.items()} != {
        k: v.get("content_sha256") for k, v in meta["models"].items()
    }:
        raise ValueError("model files differ from the ones the index was built with")
    return Index(retrieval_config(), root, meta, neural)


def search(index_dir: str | Path, query: str, k: int = 10) -> list[dict]:
    """Top-`k` chunks (id, event_id, time, type, entities, title, text, score) for the MCP server."""
    return load(str(Path(index_dir).resolve())).search(query, k)


def read_event(index_dir: str | Path, key: str) -> list[dict]:
    """The whole event `key` (an event id or a chunk id) for the MCP server's `read_event`."""
    chunks = load(str(Path(index_dir).resolve())).event(key)
    if not chunks:
        raise ValueError("no such event in this checkpoint")
    return [{"id": key, "text": "\n".join(str(c.get("text", "")) for c in chunks)}]


SHARD_FILES = ("chunks.jsonl", "lex.npz", "emb.f32")  # emb.f32 last: the shard's commit point


def write_shard(index: Path, ym: str, records: list[dict], vectors: np.ndarray) -> dict:
    """The three files of one month, each renamed into place; returns their sizes and hashes."""
    lines = [(json.dumps(r, ensure_ascii=False) + "\n").encode() for r in records]
    words: dict[str, int] = {}
    terms, counts, indptr, lengths = [], [], [0], []
    for record in records:
        found = tokens(record["text"]) or ["__empty__"]
        lengths.append(len(found))
        for word, count in Counter(found).items():
            terms.append(words.setdefault(word, len(words)))
            counts.append(count)
        indptr.append(len(terms))
    if counts and max(counts) > np.iinfo(np.uint16).max:
        raise ValueError(f"R-hyb: a term count in {ym} does not fit uint16")
    if any("\n" in r["id"] for r in records):
        raise ValueError(f"R-hyb: a chunk id in {ym} holds a newline")
    blob = lambda text: np.frombuffer(text.encode(), dtype=np.uint8)  # noqa: E731
    with (index / f"{ym}.lex.tmp.npz").open("wb") as handle:
        np.savez(
            handle,
            offsets=np.cumsum([0] + [len(line) for line in lines], dtype=np.uint64),
            ids=blob("\n".join(r["id"] for r in records)),
            words=blob(json.dumps(list(words), ensure_ascii=False)),
            terms=np.asarray(terms, dtype=np.uint32),
            counts=np.asarray(counts, dtype=np.uint16),
            indptr=np.asarray(indptr, dtype=np.int64),
            lengths=np.asarray(lengths, dtype=np.uint32),
        )
    (index / f"{ym}.chunks.jsonl.tmp").write_bytes(b"".join(lines))
    (index / f"{ym}.emb.f32.tmp").write_bytes(np.ascontiguousarray(vectors, np.float32).tobytes())
    temporary = {"chunks.jsonl": "chunks.jsonl.tmp", "lex.npz": "lex.tmp.npz", "emb.f32": "emb.f32.tmp"}  # fmt: skip
    digests = {}
    for name in SHARD_FILES:
        digests[name] = hashlib.sha256((index / f"{ym}.{temporary[name]}").read_bytes()).hexdigest()
        os.replace(index / f"{ym}.{temporary[name]}", index / f"{ym}.{name}")
    return {"postings": len(terms), "sha256": digests,
            "bytes": sum((index / f"{ym}.{name}").stat().st_size for name in SHARD_FILES)}  # fmt: skip


class RagStore:
    arm = "R-hyb"

    def __init__(self, *, builder: str = "claude", neural=None, **_):
        self.builder = "fake" if builder == "fake" else "local"
        self._neural = neural

    @property
    def neural(self):
        self._neural = self._neural or neural_for(self.builder)
        return self._neural

    def build(self, events: Iterable[dict], out: Path, *, until: str | None = None) -> None:
        index, path = out / "index", out / "build-state.json"
        index.mkdir(parents=True, exist_ok=True)
        state = json.loads(path.read_text()) if path.exists() else {"builder": self.builder, "format": 2, "checkpoints": {}, "seconds": 0.0}  # fmt: skip
        if state["builder"] != self.builder or state.get("format") != 2:
            raise ValueError(f"R-hyb index at {out} was built with {state['builder']} "
                             f"(format {state.get('format', 1)}), now {self.builder} (format 2)")  # fmt: skip
        previous = None
        for ym, group in groupby(events, key=month):
            if until and ym > until:
                break
            if previous and ym < previous:
                raise ValueError(f"R-hyb: the stream is not ordered by time at {ym}")
            gap = next_month(previous) if previous else ym
            while gap < ym:  # a month without events is still a checkpoint
                state["checkpoints"].setdefault(gap, {"events": 0, "chunks": 0, "postings": 0})
                gap = next_month(gap)
            previous = ym
            done = state["checkpoints"].get(ym)
            if done and (not done["chunks"] or (index / f"{ym}.emb.f32").exists()):
                if (count := sum(1 for _ in group)) != done["events"]:
                    raise RuntimeError(  # the complete stream, no more and no less
                        f"R-hyb: shard {ym} holds {done['events']} events, "
                        f"the stream has {count}; delete index/{ym}.* to rebuild it"
                    )
                continue
            started = time.monotonic()
            records, events_in_month = [], 0
            for event in group:  # one month in memory, never the stream
                records += chunk(event, self.neural)
                events_in_month += 1
            texts = [r["text"] for r in records]
            vectors = np.concatenate(
                [self.neural.passages(texts[n : n + EMBED_SLICE]) for n in range(0, len(texts), EMBED_SLICE)]
            )  # fmt: skip
            shard = write_shard(index, ym, records, vectors)
            seconds = round(time.monotonic() - started, 2)
            state["checkpoints"][ym] = {"events": events_in_month, "chunks": len(records),
                                        "dim": int(vectors.shape[1]), "seconds": seconds, **shard}  # fmt: skip
            state["seconds"] += seconds
            save_json(path, state)
            print(f"R-hyb {ym}: {events_in_month} events, {len(records)} chunks, {seconds:.0f} s "
                  f"({len(records) / max(seconds, 1e-9):.1f} chunks/s)", flush=True)  # fmt: skip
        save_json(path, state)

    def export(self, out: Path, checkpoint: str) -> Path:
        state = json.loads((out / "build-state.json").read_text())
        if checkpoint not in state["checkpoints"]:
            raise ValueError(f"R-hyb: checkpoint {checkpoint} is not built")
        shards = [
            {"month": ym, **{k: n[k] for k in ("events", "chunks", "postings", "sha256")}}
            for ym, n in sorted(state["checkpoints"].items())
            if ym <= checkpoint and n["chunks"]
        ]
        target = out / "export" / checkpoint
        if target.exists():
            unlock(target)
            shutil.rmtree(target)
        (target / "shards").mkdir(parents=True)
        for shard in shards:  # hard links: the checkpoint's view of the shared index, no copy
            for name in SHARD_FILES:
                source = out / "index" / f"{shard['month']}.{name}"
                try:
                    os.link(source, target / "shards" / source.name)
                except OSError:  # another filesystem
                    shutil.copy2(source, target / "shards" / source.name)
        meta = {
            "arm": self.arm,
            "checkpoint": checkpoint,
            "builder": self.builder,
            "models": self.neural.pins,
            "retrieval": retrieval_config().model_dump(mode="json"),
            "chunking": {
                "unit": "embedding-model tokens",
                "max_tokens": MAX_TOKENS,
                "overlap_tokens": OVERLAP_TOKENS,
                "bundling": "none",
            },  # fmt: skip
            "events": sum(s["events"] for s in shards),
            "chunks": sum(s["chunks"] for s in shards),
            "shards": shards,
        }
        save_json(target / "meta.json", meta)
        months = [state["checkpoints"][s["month"]] for s in shards]
        save_json(
            out / "export" / f"{checkpoint}.manifest.json",
            {
                "arm": self.arm,
                "checkpoint": checkpoint,
                # the shard files' own hashes, taken when each was written: nothing is re-read
                "content_sha256": hashlib.sha256(
                    json.dumps(
                        [[s["month"], s["chunks"], s["sha256"]] for s in shards], sort_keys=True
                    ).encode()
                ).hexdigest(),  # fmt: skip
                "files": 1 + len(SHARD_FILES) * len(shards),
                "bytes": (target / "meta.json").stat().st_size + sum(m["bytes"] for m in months),
                "build": {k: meta[k] for k in ("builder", "models", "retrieval", "chunking")},
            },
        )
        lock(target)
        return target
