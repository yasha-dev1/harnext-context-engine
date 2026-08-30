"""S5 curated hybrid vector store from docs/evaluation-spec.md §7 E3."""

from __future__ import annotations

import json
from collections import defaultdict

from harnext_eval.stores.base import StoreHandle
from harnext_eval.stores.build_s3 import fold_s3
from harnext_eval.stores.layouts import delivered_event_ids, runtime_for
from harnext_eval.stores.vector_index import VectorIndex
from harnext_eval.types import EvalEvent


def _file_documents(
    store: StoreHandle, event_ids: list[str]
) -> tuple[list[str], list[str], list[dict[str, object]]]:
    """Chunk durable store files into file records and cited-event snippets."""

    ids: list[str] = []
    documents: list[str] = []
    records: list[dict[str, object]] = []
    snippets: dict[str, list[str]] = defaultdict(list)
    candidates = [
        path
        for path in store.worktree.rglob("*")
        if path.is_file() and ".git" not in path.parts and "_vector" not in path.parts
    ]
    for path in sorted(candidates):
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relpath = path.relative_to(store.worktree).as_posix()
        file_id = f"file:{relpath}"
        ids.append(file_id)
        documents.append(f"{relpath}\n{content}")
        records.append({"kind": "file", "path": relpath})
        for line in content.splitlines():
            for event_id in event_ids:
                if event_id in line:
                    snippets[event_id].append(f"{relpath}: {line}")

    for event_id in event_ids:
        ids.append(event_id)
        documents.append("\n".join(snippets.get(event_id, [event_id])))
        records.append(
            {
                "event_id": event_id,
                "kind": "event-citations",
                "source_files": sorted(
                    {snippet.split(": ", 1)[0] for snippet in snippets.get(event_id, [])}
                ),
            }
        )
    return ids, documents, records


def fold_s5(store: StoreHandle, events: list[EvalEvent], lane: str) -> None:
    """Run S3 curation, then rebuild embeddings over its durable files."""

    fold_s3(store, events, lane)
    event_ids = delivered_event_ids(store)
    ids, documents, records = _file_documents(store, event_ids)
    index_dir = store.worktree / "_vector"
    runtime = runtime_for(store)
    VectorIndex(index_dir, runtime.embeddings).build(
        ids,
        documents,
        records=records,
        indexed_event_count=len(event_ids),
    )
    last_event = max(events, key=lambda event: (event.time, event.id))
    with (index_dir / "snapshot_counts.jsonl").open("a", encoding="utf-8") as destination:
        destination.write(
            json.dumps(
                {
                    "T_last_event": last_event.time.isoformat(),
                    "indexed_events": len(event_ids),
                    "lane": lane,
                    "last_event_id": last_event.id,
                },
                sort_keys=True,
            )
            + "\n"
        )


fold = fold_s5

