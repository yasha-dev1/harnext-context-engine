"""S4 raw-event vector store from docs/evaluation-spec.md §7 E3."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from harnext_eval.stores.base import StoreHandle
from harnext_eval.stores.layouts import (
    delivered_event_ids,
    record_input_metadata,
    runtime_for,
    unseen_events,
)
from harnext_eval.stores.vector_index import VectorIndex
from harnext_eval.types import EvalEvent


def _remove_context_seed(store: StoreHandle) -> None:
    for relpath in ("CLAUDE.md", "INDEX.md", "_meta/schema.md", "_meta/superseded.md"):
        path = store.worktree / relpath
        if path.exists():
            path.unlink()
    for dirname in ("entities", "topics"):
        path = store.worktree / dirname
        if path.exists():
            shutil.rmtree(path)


def _existing(index_dir: Path) -> tuple[list[str], list[str], list[dict[str, object]]]:
    directory = index_dir
    ids_path = directory / "ids.json"
    if not ids_path.exists():
        return [], [], []
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    documents = json.loads((directory / "documents.json").read_text(encoding="utf-8"))
    records_path = directory / "records.json"
    records = (
        json.loads(records_path.read_text(encoding="utf-8"))
        if records_path.exists()
        else [{} for _ in ids]
    )
    return ids, documents, records


def fold_s4(store: StoreHandle, events: list[EvalEvent], lane: str) -> None:
    """Embed raw events, with an indexed-event count recorded for every fold."""

    accepted = unseen_events(store, events)
    _remove_context_seed(store)
    index_dir = store.worktree / "_vector"
    ids, documents, records = _existing(index_dir)
    for event in accepted:
        ids.append(event.id)
        documents.append(event.model_dump_json())
        records.append(
            {
                "event_id": event.id,
                "source": event.source,
                "subject": event.subject,
                "time": event.time.isoformat(),
                "type": event.type,
            }
        )
    record_input_metadata(store, accepted)
    runtime = runtime_for(store)
    VectorIndex(index_dir, runtime.embeddings).build(
        ids,
        documents,
        records=records,
        indexed_event_count=len(delivered_event_ids(store)),
        chunking="one-raw-event-per-document-v1",
    )
    last_event = max(events, key=lambda event: (event.time, event.id))
    folds = index_dir / "snapshot_counts.jsonl"
    with folds.open("a", encoding="utf-8") as destination:
        destination.write(
            json.dumps(
                {
                    "T_last_event": last_event.time.isoformat(),
                    "indexed_events": len(ids),
                    "lane": lane,
                    "last_event_id": last_event.id,
                },
                sort_keys=True,
            )
            + "\n"
        )


fold = fold_s4
