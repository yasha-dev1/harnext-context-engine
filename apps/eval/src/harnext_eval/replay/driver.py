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

    @property
    def baseline_key_used(self) -> str | None:
        """Return the key selected while scoring the current event."""

        ...

    @property
    def features_fired(self) -> dict[str, Any]:
        """Return policy-owned rule, scorer, and guard evidence."""

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


@dataclass
class CausalBudgetAdmission:
    """Prefix-only deviation budget; rules are an explicit, audited floor.

    Capacity is ``floor(rule_negative_events_seen * budget_pct / 100)``. A
    candidate can consume only capacity already earned by the current prefix,
    so appending future events cannot change an earlier decision.
    """

    budget_pct: float
    score_floor: float = -math.inf
    rule_negative_seen: int = 0
    deviations_admitted: int = 0

    def consider(
        self, *, rule: str | None, score: float, policy_eligible: bool = True
    ) -> tuple[bool, dict[str, Any]]:
        """Return whether the current event may attempt deviation admission."""

        if rule is not None:
            return False, {
                "rules_floor_budget_exempt": True,
                "deviation_seen": self.rule_negative_seen,
                "deviation_capacity": math.floor(
                    self.rule_negative_seen * self.budget_pct / 100.0
                ),
                "deviations_admitted": self.deviations_admitted,
            }
        self.rule_negative_seen += 1
        capacity = math.floor(self.rule_negative_seen * self.budget_pct / 100.0)
        eligible = policy_eligible and score >= self.score_floor
        candidate = eligible and self.deviations_admitted < capacity
        return candidate, {
            "rules_floor_budget_exempt": False,
            "deviation_seen": self.rule_negative_seen,
            "deviation_capacity": capacity,
            "deviations_admitted": self.deviations_admitted,
            "deviation_score_floor": self.score_floor,
            "policy_eligible": policy_eligible,
            "deviation_budget_eligible": eligible,
            "deviation_budget_available": self.deviations_admitted < capacity,
        }

    def record(self, admitted: bool) -> dict[str, int]:
        """Commit the current candidate outcome and return post-decision audit counts."""

        if admitted:
            self.deviations_admitted += 1
        return {
            "deviation_seen": self.rule_negative_seen,
            "deviation_capacity": math.floor(
                self.rule_negative_seen * self.budget_pct / 100.0
            ),
            "deviations_admitted": self.deviations_admitted,
        }


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
    admission: CausalBudgetAdmission | None = None,
) -> DriverStats:
    """Route and fold events in deterministic event-time order.

    Deviation admission is budgeted from the event prefix only. Rules remain an
    unconditional, explicitly audited floor outside that deviation budget. A
    cutoff is inclusive: an event at ``cutoff`` is processed and the first
    event after it is not.
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

    budget = admission or CausalBudgetAdmission(
        budget_pct=cfg.router.budget_pct,
        # ``absolute_floor`` is a volume guard interpreted by R5 itself, not a
        # score threshold. Callers may inject a controller with a threshold
        # frozen on prior data without changing the RouterPolicy seam.
        score_floor=-math.inf,
    )
    windows: dict[tuple[str, str], _Window] = {}
    folds: Counter[str] = Counter()
    close_reasons: Counter[str] = Counter()
    snapshots: list[SnapshotRef] = []
    def fold(batch: list[EvalEvent], lane: str) -> None:
        ref = store.fold(batch, lane)
        snapshots.append(ref)
        folds[lane] += 1

    def close(key: tuple[str, str], reason: str) -> None:
        window = windows.pop(key)
        fold(window.events, "batch")
        close_reasons[reason] += 1

    for event in selected:
        for key, reason in _due_windows(windows, event.time, cfg):
            close(key, reason)

        rule = router_policy.rules(event) if cfg.router.rules.enabled else None
        score = float(router_policy.score(event))
        if not math.isfinite(score):
            raise ValueError(f"router policy returned a non-finite score for {event.id}")
        policy_features = dict(getattr(router_policy, "features_fired", {}) or {})
        selected_key = getattr(router_policy, "baseline_key_used", None)
        if selected_key is None:
            selected_key = event.baseline_keys[0] if event.baseline_keys else None
        policy_eligible = bool(policy_features.get("eligible", True))
        deviation_candidate = False
        budget_features: dict[str, Any] = {
            "deviation_budget_enabled": False,
            "rules_floor_budget_exempt": rule is not None,
        }
        if cfg.router.deviation.enabled:
            deviation_candidate, budget_features = budget.consider(
                rule=rule,
                score=score,
                policy_eligible=policy_eligible,
            )
            budget_features["deviation_budget_enabled"] = True
        item = _ScoredEvent(event, rule, score, deviation_candidate)

        lane, features = _route(item, cfg)
        if cfg.router.deviation.enabled:
            budget_features.update(
                budget.record(lane == "fast" and rule is None and deviation_candidate)
            )
        guard_outcomes = {
            name: policy_features[name]
            for name in (
                "volume_guard",
                "multi_window_confirmed",
                "eligible",
                "guard",
            )
            if name in policy_features
        }
        features = {
            **policy_features,
            **budget_features,
            **features,
            "rules_id": rule,
            "guard_outcomes": guard_outcomes,
            "selected_baseline_key": selected_key,
        }
        record = RouterRecord(
            event_id=event.id,
            t=event.time,
            score=score,
            lane=lane,
            policy=str(getattr(router_policy, "name", type(router_policy).__name__)),
            budget_pct=cfg.router.budget_pct,
            baseline_key_used=selected_key,
            features_fired=features,
        )
        if on_decision is not None:
            on_decision(record)

        if lane == "fast":
            key = (event.mgtenant, event.subject)
            if key in windows:
                close(key, "fast")
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


def _route(
    item: _ScoredEvent,
    cfg: EngineConfig,
) -> tuple[str, dict[str, Any]]:
    features: dict[str, Any] = {}
    fast = item.rule is not None or item.deviation_candidate
    if item.rule is not None:
        features["rule"] = item.rule
    elif item.deviation_candidate:
        features["deviation"] = True

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
