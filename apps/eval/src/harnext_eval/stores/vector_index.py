"""Persistent vector retrieval for S4/S5 in docs/evaluation-spec.md §7 E3."""

from __future__ import annotations

import io
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from harnext_eval.providers.embeddings import EmbeddingsProvider, FakeEmbeddings
from harnext_eval.stores.base import StoreHandle
from harnext_eval.types import SnapshotRef


@dataclass(frozen=True, slots=True)
class SearchHit:
    item_id: str
    score: float
    metadata: dict[str, Any]


class VectorIndex:
    """A small exact-cosine index persisted as safe NumPy and JSON files."""

    def __init__(self, directory: str | Path, provider: EmbeddingsProvider | None = None) -> None:
        self.directory = Path(directory)
        self.provider = provider or FakeEmbeddings()
        self._vectors: np.ndarray | None = None
        self._ids: list[str] | None = None
        self._documents: list[str] | None = None
        self._records: list[dict[str, Any]] | None = None

    @property
    def vectors_path(self) -> Path:
        return self.directory / "embeddings.npy"

    @property
    def ids_path(self) -> Path:
        return self.directory / "ids.json"

    def build(
        self,
        ids: list[str],
        documents: list[str],
        *,
        records: list[dict[str, Any]] | None = None,
        indexed_event_count: int | None = None,
    ) -> None:
        if len(ids) != len(documents):
            raise ValueError("ids and documents must have the same length")
        if len(set(ids)) != len(ids):
            raise ValueError("vector index ids must be unique")
        if records is None:
            records = [{} for _ in ids]
        if len(records) != len(ids):
            raise ValueError("records and ids must have the same length")

        vectors = np.asarray(self.provider.embed(documents), dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[0] != len(ids):
            raise ValueError("embedding provider returned an invalid matrix shape")
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.vectors_path.open("wb") as destination:
            np.save(destination, vectors, allow_pickle=False)
        self.ids_path.write_text(json.dumps(ids, indent=2) + "\n", encoding="utf-8")
        (self.directory / "documents.json").write_text(
            json.dumps(documents, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (self.directory / "records.json").write_text(
            json.dumps(records, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )
        metadata = {
            "dimension": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
            "document_count": len(ids),
            "indexed_event_count": indexed_event_count
            if indexed_event_count is not None
            else len(ids),
        }
        (self.directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._vectors = vectors
        self._ids = list(ids)
        self._documents = list(documents)
        self._records = list(records)

    def _load(self) -> None:
        if self._vectors is not None:
            return
        with self.vectors_path.open("rb") as source:
            self._vectors = np.load(source, allow_pickle=False)
        self._ids = json.loads(self.ids_path.read_text(encoding="utf-8"))
        self._documents = json.loads(
            (self.directory / "documents.json").read_text(encoding="utf-8")
        )
        loaded_ids = self._ids
        assert loaded_ids is not None
        records_path = self.directory / "records.json"
        self._records = (
            json.loads(records_path.read_text(encoding="utf-8"))
            if records_path.exists()
            else [{} for _ in loaded_ids]
        )

    @property
    def count(self) -> int:
        self._load()
        assert self._ids is not None
        return len(self._ids)

    def search_hits(self, query: str, *, top_k: int = 10) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self._load()
        assert self._vectors is not None
        assert self._ids is not None
        assert self._documents is not None
        assert self._records is not None
        ids = self._ids
        documents = self._documents
        records = self._records
        if not ids:
            return []
        query_vector = np.asarray(self.provider.embed([query]), dtype=np.float64)
        if query_vector.shape != (1, self._vectors.shape[1]):
            raise ValueError("query embedding dimension differs from persisted index")
        scores = self._vectors @ query_vector[0]
        query_folded = query.casefold().strip()
        ranked: list[tuple[float, int]] = []
        for index, score in enumerate(scores):
            exact_bonus = 2.0 if ids[index].casefold() == query_folded else 0.0
            content_bonus = (
                1.0 if query_folded and query_folded in documents[index].casefold() else 0.0
            )
            ranked.append((float(score) + exact_bonus + content_bonus, index))
        ranked.sort(key=lambda item: (-item[0], ids[item[1]]))
        return [
            SearchHit(
                item_id=ids[index],
                score=score,
                metadata=dict(records[index]),
            )
            for score, index in ranked[:top_k]
        ]

    def search(self, query: str, *, top_k: int = 10) -> list[str]:
        """Return ranked item IDs; exact-ID queries are deterministic and boosted."""

        return [hit.item_id for hit in self.search_hits(query, top_k=top_k)]

    @classmethod
    def from_store(
        cls,
        store: StoreHandle,
        *,
        provider: EmbeddingsProvider | None = None,
        ref: SnapshotRef | str | None = None,
        relpath: str = "_vector",
    ) -> VectorIndex:
        """Load the current index or an immutable index from a git snapshot."""

        if ref is None:
            selected_provider = provider
            if selected_provider is None:
                from harnext_eval.stores.layouts import configured_embeddings

                selected_provider = configured_embeddings(store)
            if selected_provider is None:
                metadata_path = store.worktree / relpath / "metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                selected_provider = FakeEmbeddings(dim=int(metadata.get("dimension", 64)))
            return cls(store.worktree / relpath, selected_provider)
        sha = ref.sha if isinstance(ref, SnapshotRef) else ref
        blobs: dict[str, bytes] = {}
        for filename in (
            "embeddings.npy",
            "ids.json",
            "documents.json",
            "records.json",
            "metadata.json",
        ):
            result = subprocess.run(
                ["git", "-C", str(store.worktree), "show", f"{sha}:{relpath}/{filename}"],
                capture_output=True,
                check=False,
            )
            if result.returncode == 0:
                blobs[filename] = result.stdout
        required = {"embeddings.npy", "ids.json", "documents.json"}
        if not required.issubset(blobs):
            missing = ", ".join(sorted(required - blobs.keys()))
            raise FileNotFoundError(f"snapshot {sha} has no complete vector index: {missing}")

        metadata = json.loads(blobs.get("metadata.json", b"{}").decode())
        selected_provider = provider
        if selected_provider is None:
            from harnext_eval.stores.layouts import configured_embeddings

            selected_provider = configured_embeddings(store)
        selected_provider = selected_provider or FakeEmbeddings(
            dim=int(metadata.get("dimension", 64))
        )
        index = cls(store.worktree / relpath, selected_provider)
        index._vectors = np.load(io.BytesIO(blobs["embeddings.npy"]), allow_pickle=False)
        index._ids = json.loads(blobs["ids.json"].decode())
        index._documents = json.loads(blobs["documents.json"].decode())
        loaded_ids = index._ids
        assert loaded_ids is not None
        index._records = json.loads(blobs.get("records.json", b"[]").decode()) or [
            {} for _ in loaded_ids
        ]
        return index


def search_store(
    store: StoreHandle,
    query: str,
    *,
    top_k: int = 10,
    provider: EmbeddingsProvider | None = None,
    ref: SnapshotRef | str | None = None,
) -> list[str]:
    """Convenience top-k API shared by vector and hybrid store consumers."""

    return VectorIndex.from_store(store, provider=provider, ref=ref).search(query, top_k=top_k)
