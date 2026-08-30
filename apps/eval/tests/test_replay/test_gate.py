"""Leakage gate pass/fail tests for docs/evaluation-spec.md §4.2."""

from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path

from harnext_eval.replay.gate import leakage_gate
from harnext_eval.types import EvalEvent, Probe, SnapshotRef

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


def _probe(question: str) -> Probe:
    return Probe(
        probe_id="probe-1",
        family="extraction",
        entity="HNX-1",
        T=_T,
        question=question,
        gold="alpha",
        gold_type="exact",
    )


def _snapshot() -> SnapshotRef:
    return SnapshotRef(
        sha="snapshot-sha",
        T_last_event=_T,
        last_event_id="pre",
        lane="batch",
    )


def test_gate_writes_pass_and_fail_rows(tmp_path: Path) -> None:
    output = tmp_path / "out" / "gate.csv"
    pre = _event("pre", -1, "alpha status")
    post = _event("post", 1, "future codename zephyr")

    assert leakage_gate(
        _probe("alpha status"),
        _snapshot(),
        [pre],
        all_events=[pre, post],
        gold_action_time=_T + timedelta(hours=1),
        out_csv=output,
    )
    assert not leakage_gate(
        _probe("codename zephyr"),
        _snapshot(),
        [pre, post],
        gold_action_time=_T,
        gold_action="approve release",
        envelope="The leaked next action is: approve release.",
        out_csv=output,
    )

    with output.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert [row["result"] for row in rows] == ["PASS", "FAIL"]
    assert "delivered_event_after_T" in rows[1]["reasons"]
    assert "question_token_only_post_T" in rows[1]["reasons"]
    assert "gold_action_not_after_T" in rows[1]["reasons"]
    assert "gold_action_in_envelope" in rows[1]["reasons"]
