"""search_context — lightweight keyword retrieval over a context snapshot, no LLM.

`context_research` sends an internal read agent through the store and returns a synthesized
answer: thorough, but it costs a model run per question. `search_context` lets the calling agent
look things up itself: the snapshot's markdown files are cut into heading-sized chunks and ranked
with BM25 (Okapi, k1=1.2, b=0.75); each hit carries its file path and line range, so the caller
can cite it or narrow the next query. Identifiers such as `KAFKA-20592`, `KIP-1331` or
`RemoteLogManager` are single tokens and match exactly.
"""

from __future__ import annotations

import math
import re
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from harnext_builder.agentfs.store import OrgFsStore

MAX_CHUNK_CHARS = 1500
MAX_CHUNK_LINES = 60
SNIPPET_CHARS = 1200
DEFAULT_K, MAX_K = 8, 20
K1, B = 1.2, 0.75
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
_HEADING = re.compile(r"^#{1,6} ")


def tokens(text: str) -> list[str]:
    """Lower-cased words; a compound (`kafka-20592`, `log.retention.ms`) also yields its parts."""
    out = []
    for word in _TOKEN.findall(text.lower()):
        out.append(word)
        if any(c in word for c in "-_."):
            out.extend(p for p in re.split(r"[-_.]", word) if p)
    return out


@dataclass(frozen=True)
class Chunk:
    path: str
    start: int  # 1-based, inclusive
    end: int
    text: str


def chunks(rel: str, text: str) -> list[Chunk]:
    """Heading-delimited sections, split further at MAX_CHUNK_LINES / MAX_CHUNK_CHARS."""
    lines = text.splitlines()
    out, begin, size = [], 0, 0
    for i, line in enumerate(lines):
        new_section = _HEADING.match(line) and i > begin
        too_big = i - begin >= MAX_CHUNK_LINES or size + len(line) > MAX_CHUNK_CHARS
        if i > begin and (new_section or too_big):
            out.append(Chunk(rel, begin + 1, i, "\n".join(lines[begin:i])))
            begin, size = i, 0
        size += len(line) + 1
    if begin < len(lines):
        out.append(Chunk(rel, begin + 1, len(lines), "\n".join(lines[begin:])))
    return [c for c in out if c.text.strip()]


class Index:
    def __init__(self, root: Path):
        started = time.monotonic()
        self.chunks: list[Chunk] = []
        for path in sorted(root.rglob("*.md")):
            if any(part.startswith(".") for part in path.relative_to(root).parts):
                continue
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            self.chunks.extend(chunks(str(path.relative_to(root)), text))
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.lengths = []
        for n, chunk in enumerate(self.chunks):
            counts = Counter(tokens(chunk.path + "\n" + chunk.text))
            self.lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                self.postings.setdefault(term, []).append((n, tf))
        self.avg = sum(self.lengths) / max(len(self.lengths), 1)
        self.seconds = round(time.monotonic() - started, 2)

    def search(self, query: str, k: int) -> list[tuple[float, Chunk]]:
        scores: Counter[int] = Counter()
        total = len(self.chunks)
        for term in set(tokens(query)):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = math.log(1 + (total - len(posting) + 0.5) / (len(posting) + 0.5))
            for n, tf in posting:
                norm = tf + K1 * (1 - B + B * self.lengths[n] / self.avg)
                scores[n] += idf * tf * (K1 + 1) / norm
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [(round(score, 3), self.chunks[n]) for n, score in ranked]


_BUILD = threading.Lock()


@lru_cache(maxsize=4)
def _cached(root: str, _stamp: float) -> Index:
    return Index(Path(root))


def _index(root: str, stamp: float) -> Index:
    """One build per directory: a caller arriving mid-build waits for it instead of repeating it."""
    with _BUILD:
        return _cached(root, stamp)


def search_dir(root: str | Path, query: str, k: int = DEFAULT_K) -> dict[str, Any]:
    """Rank the snapshot's chunks for `query`. Shared by the product (over a materialized
    snapshot) and the E2 evaluation (over a checkpoint directory). The index is built once per
    directory and reused while the directory is unchanged."""
    started = time.monotonic()
    root = Path(root).resolve()
    index = _index(str(root), root.stat().st_mtime)
    k = max(1, min(int(k), MAX_K))
    hits = [
        {"path": c.path, "lines": f"{c.start}-{c.end}", "score": score,
         "text": c.text[:SNIPPET_CHARS] + ("…" if len(c.text) > SNIPPET_CHARS else "")}
        for score, c in index.search(query, k)
    ]  # fmt: skip
    return {
        "hits": hits,
        "chunks_indexed": len(index.chunks),
        "index_seconds": index.seconds,
        "seconds": round(time.monotonic() - started, 3),
    }


async def search(store: OrgFsStore, org_id: str, query: str, k: int = DEFAULT_K) -> dict[str, Any]:
    snap = await store.latest_snapshot(org_id)
    if snap is None:
        return {"hits": [], "snapshot_id": None}
    files = await store.list_files(org_id, snapshot_id=snap.id)
    with tempfile.TemporaryDirectory(prefix=f"search-{org_id}-") as tmp:
        for rel in files:
            content = await store.read_file(org_id, rel, snapshot_id=snap.id)
            if content is None:
                continue
            dst = Path(tmp) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(content)
        out = search_dir(tmp, query, k)
    return {"hits": out["hits"], "snapshot_id": snap.id}
