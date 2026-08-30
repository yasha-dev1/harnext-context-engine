"""Event-clock replay tests for docs/evaluation-spec.md §3.3 and PLAN.md §6."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from harnext_eval.config import EngineConfig, load_config
from harnext_eval.corpus.synthetic import generate_synthetic_events
from harnext_eval.replay.driver import run_pipeline
from harnext_eval.stores.base import StoreHandle
from harnext_eval.types import EvalEvent, SnapshotRef

_START = datetime(2026, 1, 1, tzinfo=UTC)


class StubStore:
    def __init__(self) -> None:
        self.folds: list[tuple[list[EvalEvent], str]] = []

    def fold(self, events: list[EvalEvent], lane: str) -> SnapshotRef:
        self.folds.append((list(events), lane))
        last = max(events, key=lambda event: (event.time, event.id))
        return SnapshotRef(
            sha=f"sha-{len(self.folds)}",
            T_last_event=last.time,
            last_event_id=last.id,
            lane=lane,
        )


class ScoredPolicy:
    name = "scored-test"

    def rules(self, event: EvalEvent) -> str | None:
        del event
        return None

    def score(self, event: EvalEvent) -> float:
        return float((event.data or {})["score"])


def _cfg(*, gap: float, cap: int, age: float) -> EngineConfig:
    path = Path(__file__).parents[2] / "configs" / "baseline-minimal.yaml"
    engine = load_config(path).engine
    return engine.model_copy(
        update={
            "window": engine.window.model_copy(
                update={"gap_s": gap, "max_events": cap, "max_age_s": age}
            )
        }
    )


def _event(event_id: str, seconds: float, *, subject: str = "issue:HNX-1") -> EvalEvent:
    return EvalEvent(
        id=event_id,
        source="jira:test",
        type="issue.comment",
        subject=subject,
        time=_START + timedelta(seconds=seconds),
        mgtenant="test",
        baseline_keys=["component:builder"],
        data={"body": event_id},
    )


def _run(events: list[EvalEvent], cfg: EngineConfig) -> tuple[StubStore, object]:
    store = StubStore()
    stats = run_pipeline(events, cfg, cast(StoreHandle, store))
    return store, stats


def test_driver_runs_synthetic_corpus_and_records_every_fold() -> None:
    events = generate_synthetic_events(seed=9, event_count=30, days=1, entity_count=4)
    store = StubStore()
    decisions = []

    stats = run_pipeline(
        reversed(events),
        _cfg(gap=30, cap=20, age=120),
        cast(StoreHandle, store),
        on_decision=decisions.append,
    )

    assert stats.events == 30
    assert len(decisions) == 30
    assert len(stats.snapshots) == len(store.folds) == sum(stats.folds_per_lane.values())
    assert [record.t for record in decisions] == sorted(record.t for record in decisions)


def test_windows_close_on_event_clock_gap_cap_and_max_age() -> None:
    gap_store, gap_stats = _run(
        [_event("a", 0), _event("advance", 6, subject="issue:OTHER")],
        _cfg(gap=5, cap=20, age=100),
    )
    assert [event.id for event in gap_store.folds[0][0]] == ["a"]
    assert gap_stats.windows_closed_by_reason["gap"] == 1  # type: ignore[attr-defined]

    cap_store, cap_stats = _run(
        [_event("a", 0), _event("b", 1)], _cfg(gap=100, cap=2, age=100)
    )
    assert [event.id for event in cap_store.folds[0][0]] == ["a", "b"]
    assert cap_stats.windows_closed_by_reason["cap"] == 1  # type: ignore[attr-defined]

    age_store, age_stats = _run(
        [
            _event("a", 0),
            _event("b", 2),
            _event("advance", 5, subject="issue:OTHER"),
        ],
        _cfg(gap=100, cap=20, age=5),
    )
    assert [event.id for event in age_store.folds[0][0]] == ["a", "b"]
    assert age_stats.windows_closed_by_reason["max_age"] == 1  # type: ignore[attr-defined]


def test_cutoff_is_inclusive_and_flushes_open_windows() -> None:
    store = StubStore()
    stats = run_pipeline(
        [_event("after", 2), _event("at", 1), _event("before", 0)],
        _cfg(gap=100, cap=20, age=100),
        cast(StoreHandle, store),
        cutoff=_START + timedelta(seconds=1),
    )

    assert stats.events == 2
    assert [[event.id for event in events] for events, _ in store.folds] == [["before", "at"]]
    assert stats.windows_closed_by_reason == {"cutoff": 1}
    assert stats.folds_per_lane == {"fast": 0, "batch": 1}


def test_router_uses_rules_floor_and_budgeted_policy_scores() -> None:
    cfg = _cfg(gap=100, cap=20, age=100)
    rule_event = _event("vote", 0)
    assert rule_event.data is not None
    rule_event.data["body"] = "[VOTE] release HNX-1"
    rule_store = StubStore()
    records = []
    run_pipeline(
        [rule_event],
        cfg,
        cast(StoreHandle, rule_store),
        on_decision=records.append,
    )
    assert records[0].lane == "fast"
    assert rule_store.folds[0][1] == "fast"

    router = cfg.router.model_copy(
        update={
            "rules": cfg.router.rules.model_copy(update={"enabled": False}),
            "deviation": cfg.router.deviation.model_copy(update={"enabled": True}),
            "budget_pct": 50,
        }
    )
    scored_cfg = cfg.model_copy(update={"router": router})
    low = _event("low", 1, subject="issue:LOW")
    high = _event("high", 2, subject="issue:HIGH")
    assert low.data is not None and high.data is not None
    low.data["score"] = 0.1
    high.data["score"] = 0.9
    scored_store = StubStore()
    scored_records = []
    run_pipeline(
        [low, high],
        scored_cfg,
        cast(StoreHandle, scored_store),
        on_decision=scored_records.append,
        policy=ScoredPolicy(),
    )
    assert {record.event_id: record.lane for record in scored_records} == {
        "low": "batch",
        "high": "fast",
    }
