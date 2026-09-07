"""Leakage gate pass/fail tests for docs/evaluation-spec.md §4.2."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from harnext_eval.replay.gate import leakage_gate
from harnext_eval.stores.base import StoreHandle, register_layout
from harnext_eval.types import EvalEvent, Probe, SnapshotRef, Task

_T = datetime(2026, 2, 1, tzinfo=UTC)


def _event(event_id: str, offset: int, body: str) -> EvalEvent:
    return EvalEvent(
        id=event_id,
        source="jira:test",
        type="issue.comment",
        subject="issue:HNX-1",
        time=_T + timedelta(seconds=offset),
        mgtenant="test",
        data={"body": body},
    )


def _probe(question: str, *, source_event_ids: list[str] | None = None) -> Probe:
    return Probe(
        probe_id="probe-1",
        family="extraction",
        entity="HNX-1",
        T=_T,
        question=question,
        gold="alpha",
        gold_type="exact",
        source_event_ids=source_event_ids or [],
    )


def _store(tmp_path: Path, events: list[EvalEvent]) -> StoreHandle:
    def fold_gate(store: StoreHandle, folded: list[EvalEvent], lane: str) -> None:
        del lane
        for event in folded:
            store.write(f"events/{event.id}.md", event.model_dump_json())

    register_layout("GATE", fold_gate)
    store = StoreHandle("GATE", "acme", tmp_path / "store")
    for event in events:
        store.fold([event], "batch")
    return store


def test_gate_resolves_store_ledger_and_does_not_require_probe_action_time(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out" / "gate.csv"
    pre = _event("pre", -1, "alpha status")
    post = _event("post", 1, "future codename zephyr")
    store = _store(tmp_path, [pre, post])

    assert leakage_gate(
        _probe("alpha status", source_event_ids=["pre"]),
        store=store,
        T=_T,
        all_events=[pre, post],
        material="alpha status from pre",
        out_csv=output,
    )
    with output.open(newline="", encoding="utf-8") as source:
        row = next(csv.DictReader(source))
    assert row["sha"] == store.snapshot(_T).sha
    assert row["last_event_id"] == "pre"
    assert row["result"] == "PASS"
    assert row["reasons"] == ""


def test_gate_rejects_selected_future_commit_and_post_t_source(tmp_path: Path) -> None:
    output = tmp_path / "gate.csv"
    pre = _event("pre", -1, "alpha status")
    post = _event("post", 1, "future codename zephyr")
    store = _store(tmp_path, [pre, post])
    future_ref = store.delivery_records()[-1]
    selected_future = SnapshotRef(
        sha=future_ref.sha,
        T_last_event=future_ref.snapshot_T_last_event,
        last_event_id="post",
        lane="batch",
    )

    assert not leakage_gate(
        _probe("codename zephyr", source_event_ids=["post"]),
        selected_future,
        store=store,
        T=_T,
        all_events=[pre, post],
        material="post reveals codename zephyr",
        out_csv=output,
    )
    with output.open(newline="", encoding="utf-8") as source:
        row = next(csv.DictReader(source))
    assert "snapshot_not_last_safe_commit" in row["reasons"]
    assert "question_token_only_post_T" in row["reasons"]
    assert "source_event_after_T:post" in row["reasons"]
    assert "post_T_event_in_material:post" in row["reasons"]


def test_gate_fails_closed_for_unmapped_legacy_delivery_log(tmp_path: Path) -> None:
    output = tmp_path / "gate.csv"
    pre = _event("pre", -1, "alpha status")
    snapshot = SnapshotRef(
        sha="unmapped-sha",
        T_last_event=pre.time,
        last_event_id="pre",
        lane="batch",
    )

    assert not leakage_gate(
        _probe("alpha status"),
        snapshot,
        [pre],
        all_events=[pre],
        out_csv=output,
    )
    with output.open(newline="", encoding="utf-8") as source:
        row = next(csv.DictReader(source))
    assert "delivery_ledger_unverifiable" in row["reasons"]


def test_legacy_snapshot_call_uses_registered_store_not_filtered_argument(
    tmp_path: Path,
) -> None:
    output = tmp_path / "gate.csv"
    pre = _event("registered-pre", -1, "alpha status")
    post = _event("registered-post", 1, "future material")
    store = _store(tmp_path, [pre, post])
    ref = store.snapshot(_T)

    assert leakage_gate(
        _probe("alpha status", source_event_ids=["registered-pre"]),
        ref,
        [],
        all_events=[pre, post],
        out_csv=output,
    )
    with output.open(newline="", encoding="utf-8") as source:
        row = next(csv.DictReader(source))
    assert row["result"] == "PASS"
    assert row["sha"] == ref.sha


def test_gate_fails_when_store_ledger_is_contaminated_after_snapshot(tmp_path: Path) -> None:
    output = tmp_path / "gate.csv"
    pre = _event("tamper-pre", -1, "alpha status")
    store = _store(tmp_path, [pre])
    row = json.loads(store.delivered_jsonl.read_text(encoding="utf-8"))
    row["event_time"] = (_T + timedelta(seconds=1)).isoformat()
    store.delivered_jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert not leakage_gate(
        _probe("alpha status"),
        store=store,
        T=_T,
        all_events=[pre],
        out_csv=output,
    )
    with output.open(newline="", encoding="utf-8") as source:
        gate_row = next(csv.DictReader(source))
    assert "delivery_ledger_unverifiable" in gate_row["reasons"]


def test_task_gate_recursively_detects_structured_gold_in_envelope(tmp_path: Path) -> None:
    output = tmp_path / "gate.csv"
    pre = _event("pre", -1, "trigger")
    store = _store(tmp_path, [pre])
    task = Task(
        task_id="task-1",
        corpus="R",
        T=_T,
        trigger_event_id="pre",
        entity="HNX-1",
        kind="fast",
        gold={
            "action": {
                "time": (_T + timedelta(minutes=1)).isoformat(),
                "text": "approve release",
                "reviewers": ["alice"],
            }
        },
        gold_coverage={"text": True},
    )

    assert not leakage_gate(
        task,
        store=store,
        T=_T,
        all_events=[pre],
        envelope="Draft the action: approve release for the team.",
        out_csv=output,
    )
    with output.open(newline="", encoding="utf-8") as source:
        row = next(csv.DictReader(source))
    assert "gold_action_in_envelope:approve release" in row["reasons"]
