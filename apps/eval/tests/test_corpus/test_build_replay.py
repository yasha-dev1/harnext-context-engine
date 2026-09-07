"""Replay ordering and hash checks for docs/evaluation-spec.md §3.3."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from harnext_eval.corpus.build_replay import build_replay, main, read_replay
from harnext_eval.types import EvalEvent


def _event(event_id: str, minute: int, *, tenant: str = "wrong") -> EvalEvent:
    return EvalEvent(
        id=event_id,
        source="jira:KAFKA",
        type="org.apache.jira.issue.comment",
        subject="issue:KAFKA-1",
        time=datetime(2026, 1, 1, 0, minute, tzinfo=UTC),
        mgtenant=tenant,
        baseline_keys=[],
        data={
            "issue_key": "KAFKA-1",
            "components": ["Streams"],
            "author_email": "alice@apache.org",
            "body": f"event {event_id}",
        },
    )


def test_merge_orders_assigns_keys_and_writes_stable_sha256(tmp_path: Path) -> None:
    late = _event("late", 2)
    early = _event("early", 1)
    first = build_replay([[late], [early]], tmp_path / "first.jsonl", mgtenant="kafka")
    second = build_replay([[early], [late]], tmp_path / "second.jsonl", mgtenant="kafka")

    assert first.sha256 == second.sha256
    assert first.path.read_bytes() == second.path.read_bytes()
    assert hashlib.sha256(first.path.read_bytes()).hexdigest() == first.sha256
    assert first.sha256_path.read_text() == f"{first.sha256}  first.jsonl\n"
    events = list(read_replay(first.path))
    assert [event.id for event in events] == ["early", "late"]
    assert all(event.mgtenant == "kafka" for event in events)
    assert events[0].baseline_keys[0].startswith("contributor:")
    assert "component:streams" in events[0].baseline_keys


def test_module_cli_merges_existing_evalevent_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(_event("one", 1).model_dump_json() + "\n", encoding="utf-8")
    output = tmp_path / "replay.jsonl"

    assert main(["--input", str(source), "--output", str(output), "--tenant", "kafka"]) == 0
    assert output.exists()
    assert output.with_suffix(".jsonl.sha256").exists()
    assert [event.id for event in read_replay(output)] == ["one"]

