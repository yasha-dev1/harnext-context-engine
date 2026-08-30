"""In-process, event-clock replay driver for docs/evaluation-spec.md §3.3 and §5."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from harnext_classifier.rules import rules_match

from harnext_eval.config import EngineConfig
from harnext_eval.stores.base import StoreHandle
from harnext_eval.types import EvalEvent, RouterRecord, SnapshotRef


@runtime_checkable
class RouterPolicy(Protocol):
    """Policy seam used by E1 scorers without coupling replay to E1."""

    def rules(self, event: EvalEvent) -> str | None:
        """Return the id of a deterministic rule that promotes ``event``."""

        ...

    def score(self, event: EvalEvent) -> float:
        """Return a deviation score, where a larger value is more urgent."""

        ...


@dataclass(frozen=True)
class DriverStats:
    """Replay totals, including every snapshot returned by a store fold."""

    events: int
    folds_per_lane: dict[str, int]
    windows_closed_by_reason: dict[str, int]
    snapshots: tuple[SnapshotRef, ...] = ()

    @property
    def folds(self) -> dict[str, int]:
        """Backward-compatible short name for lane fold counts."""

        return self.folds_per_lane

    @property
    def windows_closed(self) -> dict[str, int]:
        """Backward-compatible short name for window close counts."""

        return self.windows_closed_by_reason


@dataclass
class _Window:
    events: list[EvalEvent] = field(default_factory=list)

    @property
    def first_time(self) -> datetime:
        return self.events[0].time

    @property
    def last_time(self) -> datetime:
        return self.events[-1].time


@dataclass(frozen=True)
class _ScoredEvent:
    event: EvalEvent
    rule: str | None
    score: float
    deviation_candidate: bool


class RulesOnlyPolicy:
    """The config-driven R1 floor used when no E1 policy is supplied."""

    name = "rules-only"

    def rules(self, event: EvalEvent) -> str | None:
        classifier_rule = rules_match(event)
        if classifier_rule is not None:
            return classifier_rule

        flattened = _flatten(event.data)
        priority_values = {
            str(value).casefold()
            for key, value in flattened.items()
            if key.rsplit(".", 1)[-1].casefold() in {"priority", "severity", "urgency", "to"}
        }
        if priority_values & {"blocker", "critical", "p0"}:
            return "rule:declared-priority"

        text = " ".join(str(value) for value in flattened.values())
        if re.search(r"(?i)(?:\[vote\]|\bcve(?:-\d{4}-\d+)?\b|\bblocker\b)", text):
            return "rule:declared-text"
        if re.search(r"(?i)\b(?:on[- ]call|pagerduty|page triggered)\b", text):
            return "rule:on-call"
        return None

    def score(self, event: EvalEvent) -> float:
        del event
        return 0.0


def run_pipeline(
    events: Iterable[EvalEvent],
    cfg: EngineConfig,
    store: StoreHandle,
    *,
    cutoff: datetime | None = None,
    on_decision: Callable[[RouterRecord], None] | None = None,
    policy: RouterPolicy | None = None,
) -> DriverStats:
    """Route and fold events in deterministic event-time order.

    Deviation admission is the top configured percentage of rule-negative
    events. Rules remain an unconditional floor. A cutoff is inclusive: an
    event at ``cutoff`` is processed and the first event after it is not.
    """

    router_policy = policy or RulesOnlyPolicy()
    ordered = sorted(events, key=lambda event: (event.time, event.id))
    selected: list[EvalEvent] = []
    stopped_at_cutoff = False
    for event in ordered:
        if cutoff is not None and event.time > cutoff:
            stopped_at_cutoff = True
            break
        selected.append(event)

    scored = _score_events(selected, cfg, router_policy)
    windows: dict[tuple[str, str], _Window] = {}
    folds: Counter[str] = Counter()
    close_reasons: Counter[str] = Counter()
    snapshots: list[SnapshotRef] = []
    confirmations: Counter[str] = Counter()
    admitted_situations: set[str] = set()

    def fold(batch: list[EvalEvent], lane: str) -> None:
        ref = store.fold(batch, lane)
        snapshots.append(ref)
        folds[lane] += 1

    def close(key: tuple[str, str], reason: str) -> None:
        window = windows.pop(key)
        fold(window.events, "batch")
        close_reasons[reason] += 1

    for item in scored:
        event = item.event
        for key, reason in _due_windows(windows, event.time, cfg):
            close(key, reason)

        lane, features = _route(
            item,
            cfg,
            confirmations=confirmations,
            admitted_situations=admitted_situations,
        )
        record = RouterRecord(
            event_id=event.id,
            t=event.time,
            score=item.score,
            lane=lane,
            policy=str(getattr(router_policy, "name", type(router_policy).__name__)),
            budget_pct=cfg.router.budget_pct,
            baseline_key_used=event.baseline_keys[0] if event.baseline_keys else None,
            features_fired=features,
        )
        if on_decision is not None:
            on_decision(record)

        if lane == "fast":
            fold([event], "fast")
            continue

        key = (event.mgtenant, event.subject)
        window = windows.setdefault(key, _Window())
        window.events.append(event)
        if len(window.events) >= cfg.window.max_events:
            close(key, "cap")

    if cutoff is not None:
        for key, reason in _due_windows(windows, cutoff, cfg):
            close(key, reason)

    flush_reason = "cutoff" if stopped_at_cutoff else "flush"
    for key in sorted(windows, key=lambda value: (windows[value].first_time, value)):
        close(key, flush_reason)

    return DriverStats(
        events=len(selected),
        folds_per_lane={"fast": folds["fast"], "batch": folds["batch"]},
        windows_closed_by_reason=dict(close_reasons),
        snapshots=tuple(snapshots),
    )


def _score_events(
    events: list[EvalEvent], cfg: EngineConfig, policy: RouterPolicy
) -> list[_ScoredEvent]:
    raw: list[tuple[EvalEvent, str | None, float]] = []
    for event in events:
        rule = policy.rules(event) if cfg.router.rules.enabled else None
        score = float(policy.score(event))
        if not math.isfinite(score):
            raise ValueError(f"router policy returned a non-finite score for {event.id}")
        raw.append((event, rule, score))

    candidate_indexes: set[int] = set()
    if cfg.router.deviation.enabled:
        eligible = [
            (index, score, event)
            for index, (event, rule, score) in enumerate(raw)
            if rule is None and score >= cfg.router.guards.absolute_floor
        ]
        requested = math.ceil(len([row for row in raw if row[1] is None]) * cfg.router.budget_pct / 100)
        ranked = sorted(eligible, key=lambda row: (-row[1], row[2].time, row[2].id))
        candidate_indexes = {index for index, _, _ in ranked[:requested]}

    return [
        _ScoredEvent(event, rule, score, index in candidate_indexes)
        for index, (event, rule, score) in enumerate(raw)
    ]


def _route(
    item: _ScoredEvent,
    cfg: EngineConfig,
    *,
    confirmations: Counter[str],
    admitted_situations: set[str],
) -> tuple[str, dict[str, Any]]:
    features: dict[str, Any] = {}
    fast = item.rule is not None
    if item.rule is not None:
        features["rule"] = item.rule
    elif item.deviation_candidate:
        key = item.event.baseline_keys[0] if item.event.baseline_keys else item.event.subject
        confirmations[key] += 1
        confirmed = not cfg.router.guards.multi_window or confirmations[key] >= 2
        dedup_key = f"{item.event.mgtenant}:{item.event.subject}"
        unique = not cfg.router.guards.situation_dedup or dedup_key not in admitted_situations
        fast = confirmed and unique
        features.update(
            {
                "deviation": True,
                "absolute_floor": cfg.router.guards.absolute_floor,
                "confirmed": confirmed,
                "situation_unique": unique,
            }
        )
        if fast:
            admitted_situations.add(dedup_key)

    if cfg.lane_design == "single":
        if fast:
            features["single_lane_demoted"] = True
        fast = False
    return ("fast" if fast else "batch"), features


def _due_windows(
    windows: dict[tuple[str, str], _Window], now: datetime, cfg: EngineConfig
) -> list[tuple[tuple[str, str], str]]:
    due: list[tuple[datetime, tuple[str, str], str]] = []
    for key, window in windows.items():
        gap_at = window.last_time + timedelta(seconds=cfg.window.gap_s)
        age_at = window.first_time + timedelta(seconds=cfg.window.max_age_s)
        deadlines = [(gap_at, "gap"), (age_at, "max_age")]
        eligible = [deadline for deadline in deadlines if deadline[0] <= now]
        if eligible:
            deadline, reason = min(eligible, key=lambda value: (value[0], value[1]))
            due.append((deadline, key, reason))
    return [(key, reason) for _, key, reason in sorted(due, key=lambda value: value[:2])]


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(child, path))
        return flattened
    if isinstance(value, list):
        return {f"{prefix}.{index}": child for index, child in enumerate(value)}
    return {prefix: value}
