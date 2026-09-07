"""Round-trip tests for the shared records in docs/evaluation-spec.md §5."""

from datetime import UTC, datetime

import pytest
from harnext_eval.types import (
    Answer,
    EvalEvent,
    GradeResult,
    Probe,
    RouterRecord,
    RunManifest,
    SnapshotRef,
    Task,
)
from pydantic import BaseModel

NOW = datetime(2026, 5, 3, 10, 12, 44, tzinfo=UTC)


@pytest.mark.parametrize(
    "record",
    [
        EvalEvent(
            id="event-1",
            source="jira:test",
            type="issue.transition",
            subject="issue:HNX-1",
            time=NOW,
            mgtenant="test",
            baseline_keys=["component:builder"],
            intended_send_ts=NOW,
            data={"field": "status", "to": "Done"},
        ),
        Probe(
            probe_id="probe-1",
            family="extraction",
            entity="HNX-1",
            T=NOW,
            question="What is the status of HNX-1?",
            gold="Done",
            gold_type="exact",
            superseded_values=["Open"],
            source_event_ids=["event-1"],
        ),
        Task(
            task_id="task-1",
            corpus="synthetic",
            T=NOW,
            trigger_event_id="event-1",
            entity="HNX-1",
            kind="fast",
            gold={"people": ["user-1"]},
            gold_coverage={"people": True},
        ),
        RouterRecord(
            event_id="event-1",
            t=NOW,
            score=0.8,
            lane="fast",
            policy="R0",
            budget_pct=2,
            baseline_key_used="component:builder",
            features_fired={"vote": False},
        ),
        Answer(
            probe_id="probe-1",
            arm="A1",
            text="Done",
            cited_ids=["event-1"],
            tokens_read=12,
            tool_calls=0,
            latency_s=0.01,
        ),
        GradeResult(item_id="probe-1", metric="exact", value=1, details={}),
        SnapshotRef(sha="abc", T_last_event=NOW, last_event_id="event-1", lane="batch"),
        RunManifest(
            run_id="run-1",
            created_at=NOW,
            config_hash="config",
            replay_hash="replay",
            probe_hash="probes",
            git_sha="git",
            model_ids={"reader": "fake"},
            prices={},
            seeds=[1],
            prereg_ref=None,
        ),
    ],
)
def test_json_round_trip(record: BaseModel) -> None:
    restored = type(record).model_validate_json(record.model_dump_json())
    assert restored == record
