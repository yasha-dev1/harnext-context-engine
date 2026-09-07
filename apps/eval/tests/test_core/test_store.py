"""Snapshot-store tests for docs/evaluation-spec.md §3.3."""

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from harnext_eval.stores.base import SnapshotLedgerError, StoreHandle, register_layout
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


def test_snapshot_uses_cumulative_high_water_for_late_older_fold(tmp_path: Path) -> None:
    def fold_test(store: StoreHandle, events: list[EvalEvent], lane: str) -> None:
        del lane
        store.write(f"events/{events[-1].id}.md", events[-1].id)

    register_layout("HIGHWATER", fold_test)
    store = StoreHandle("HIGHWATER", "acme", tmp_path / "store")
    start = datetime(2026, 1, 1, tzinfo=UTC)
    safe = store.fold([_event("safe", start)], "fast")
    future = store.fold([_event("future", start + timedelta(seconds=10))], "fast")
    late_old = store.fold([_event("late-old", start + timedelta(seconds=5))], "batch")

    assert late_old.T_last_event == future.T_last_event
    assert late_old.last_event_id == "future"
    assert store.snapshot(start + timedelta(seconds=7)) == safe
    assert store.delivered_event_ids(late_old) == ("safe", "future", "late-old")
    rows = store.delivery_records(late_old)
    assert rows[-1].fold_max_event_time == start + timedelta(seconds=5)
    assert rows[-1].snapshot_T_last_event == start + timedelta(seconds=10)


def test_snapshot_fails_closed_when_delivery_ledger_is_tampered(tmp_path: Path) -> None:
    def fold_test(store: StoreHandle, events: list[EvalEvent], lane: str) -> None:
        del lane
        store.write(f"events/{events[-1].id}.md", events[-1].id)

    register_layout("LEDGER", fold_test)
    store = StoreHandle("LEDGER", "acme", tmp_path / "store")
    at = datetime(2026, 1, 1, tzinfo=UTC)
    store.fold([_event("event-1", at)], "batch")
    row = json.loads(store.delivered_jsonl.read_text(encoding="utf-8"))
    row["event_time"] = (at + timedelta(days=1)).isoformat()
    store.delivered_jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(SnapshotLedgerError, match="max event time|high-water"):
        store.snapshot(at + timedelta(days=2))
