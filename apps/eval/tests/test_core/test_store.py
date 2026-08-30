"""Snapshot-store tests for docs/evaluation-spec.md §3.3."""

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from harnext_eval.stores.base import StoreHandle, register_layout
from harnext_eval.types import EvalEvent


def _event(event_id: str, at: datetime) -> EvalEvent:
    return EvalEvent(
        id=event_id,
        source="jira:test",
        type="issue.transition",
        subject="issue:HNX-1",
        time=at,
        mgtenant="test",
        baseline_keys=["component:builder"],
        data={"field": "status", "to": event_id},
    )


def test_snapshot_read_and_materialise(tmp_path: Path) -> None:
    def fold_test(store: StoreHandle, events: list[EvalEvent], lane: str) -> None:
        event = events[-1]
        store.write("entities/HNX-1.md", f"{lane}:{event.id}\n")

    register_layout("TEST", fold_test)
    store = StoreHandle("TEST", "acme", tmp_path / "store")
    start = datetime(2026, 1, 1, tzinfo=UTC)
    first = store.fold([_event("event-1", start)], "batch")
    second = store.fold([_event("event-2", start + timedelta(hours=1))], "fast")

    assert store.snapshot(start + timedelta(minutes=30)) == first
    assert store.snapshot(start + timedelta(hours=2)) == second
    assert store.read(first, "entities/HNX-1.md") == "batch:event-1\n"
    assert "entities/HNX-1.md" in store.list_files(second)

    checkout = store.materialise(first)
    try:
        assert (checkout / "entities/HNX-1.md").read_text() == "batch:event-1\n"
    finally:
        shutil.rmtree(checkout)


def test_snapshot_before_first_fold_is_missing(tmp_path: Path) -> None:
    store = StoreHandle("UNREGISTERED", "acme", tmp_path / "store")
    with pytest.raises(LookupError):
        store.snapshot(datetime(2026, 1, 1, tzinfo=UTC))
