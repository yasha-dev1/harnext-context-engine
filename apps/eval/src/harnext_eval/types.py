"""Shared evaluation records from docs/evaluation-spec.md §4 and §5."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from harnext_shared.envelope import CloudEvent
from pydantic import BaseModel, ConfigDict, Field


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvalEvent(CloudEvent):
    baseline_keys: list[str] = Field(default_factory=list)
    intended_send_ts: datetime | None = None


class Probe(StrictRecord):
    probe_id: str
    family: Literal[
        "extraction", "temporal", "update", "multisource", "code_location", "abstention"
    ]
    entity: str
    T: datetime
    question: str
    gold: Any
    gold_type: Literal["exact", "links", "files"]
    superseded_values: list[str] = Field(default_factory=list)
    source_event_ids: list[str] = Field(default_factory=list)


class Task(StrictRecord):
    task_id: str
    corpus: str
    T: datetime
    trigger_event_id: str
    entity: str
    kind: Literal["fast", "batch"]
    gold: dict[str, Any]
    gold_coverage: dict[str, bool]


class RouterRecord(StrictRecord):
    event_id: str
    t: datetime
    score: float
    lane: str
    policy: str
    budget_pct: float
    baseline_key_used: str | None
    features_fired: dict[str, Any]


class Answer(StrictRecord):
    probe_id: str
    arm: str
    text: str
    cited_ids: list[str]
    tokens_read: int
    tool_calls: int
    latency_s: float


class GradeResult(StrictRecord):
    item_id: str
    metric: str
    value: float
    details: dict[str, Any]


class SnapshotRef(StrictRecord):
    sha: str
    T_last_event: datetime
    last_event_id: str
    lane: str


class RunManifest(StrictRecord):
    run_id: str
    created_at: datetime
    config_hash: str
    replay_hash: str
    probe_hash: str
    git_sha: str
    model_ids: dict[str, str]
    prices: dict[str, float]
    seeds: list[int]
    prereg_ref: str | None
