"""Incremental temporal-firewall features for docs/evaluation-spec.md §7 E1."""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from harnext_eval.types import EvalEvent

FEATURE_NAMES = (
    "log_gap_s",
    "count_5m_ratio",
    "count_1h_ratio",
    "type_mix_js",
    "actor_novelty",
    "subject_novelty",
    "priority",
    "log_money",
    "time_of_day_bucket",
)


@dataclass(frozen=True)
class FeatureVector:
    event_id: str
    t: datetime
    baseline_key: str
    values: dict[str, float]
    context: dict[str, float]

    def as_array(self) -> np.ndarray:
        return np.asarray([self.values[name] for name in FEATURE_NAMES], dtype=float)


@dataclass
class _KeyState:
    last_time: datetime | None
    first_bucket: datetime | None
    recent: deque[tuple[datetime, str]]
    bucket_counts: dict[datetime, int]
    bucket_types: dict[datetime, Counter[str]]
    gaps: deque[tuple[datetime, float]]
    actors: set[str]
    subjects: set[str]


def _new_state() -> _KeyState:
    return _KeyState(None, None, deque(), {}, {}, deque(), set(), set())


def _bucket(t: datetime) -> datetime:
    value = t.astimezone(UTC)
    return value.replace(minute=(value.minute // 5) * 5, second=0, microsecond=0)


def _flatten(data: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(data, dict):
        for key, value in data.items():
            yield str(key).casefold(), value
            yield from _flatten(value)
    elif isinstance(data, list):
        for value in data:
            yield from _flatten(value)


def _first(data: dict[str, Any], names: set[str]) -> Any:
    for key, value in _flatten(data):
        if key in names and not isinstance(value, (dict, list)):
            return value
    return None


def _priority(data: dict[str, Any]) -> float:
    raw = str(_first(data, {"priority", "severity"}) or "").casefold()
    return float(
        {"trivial": 0, "minor": 1, "normal": 2, "major": 3, "critical": 4, "blocker": 5}.get(raw, 0)
    )


def _money(data: dict[str, Any]) -> float:
    raw = _first(data, {"amount", "money", "value", "total", "dispute_amount"})
    if isinstance(raw, dict):
        raw = raw.get("amount", 0)
    try:
        return math.log1p(abs(float(raw or 0)))
    except (TypeError, ValueError):
        return 0.0


def _actor(data: dict[str, Any]) -> str:
    value = _first(data, {"actor", "author", "from", "user", "sender"})
    return str(value or "")


def _js(left: Counter[str], right: Counter[str]) -> float:
    names = sorted(set(left) | set(right))
    if not names or not sum(left.values()) or not sum(right.values()):
        return 0.0
    p = np.asarray([left[name] for name in names], dtype=float)
    q = np.asarray([right[name] for name in names], dtype=float)
    p /= p.sum()
    q /= q.sum()
    middle = (p + q) / 2

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return (kl(p, middle) + kl(q, middle)) / 2


def _counter_median(distribution: Counter[float], total: int) -> float:
    """Median of a compact value->frequency distribution."""

    if total <= 0:
        return 0.0

    def value_at(position: int) -> float:
        seen = 0
        for value in sorted(distribution):
            seen += distribution[value]
            if seen > position:
                return float(value)
        return 0.0

    lower = value_at((total - 1) // 2)
    upper = value_at(total // 2)
    return (lower + upper) / 2.0


class CausalFeatureExtractor:
    """Stateful feature fold; `update` rejects out-of-order event-time input."""

    def __init__(
        self, *, history: timedelta = timedelta(weeks=4), global_only: bool = False
    ) -> None:
        self.history = history
        self.global_only = global_only
        self._states: defaultdict[str, _KeyState] = defaultdict(_new_state)
        self._last_event_time: datetime | None = None

    def update(self, event: EvalEvent) -> list[FeatureVector]:
        if self._last_event_time is not None and event.time < self._last_event_time:
            raise ValueError("events must be supplied in non-decreasing event-time order")
        self._last_event_time = event.time
        keys = ["__global__"] if self.global_only else sorted(set(event.baseline_keys)) or ["__global__"]
        vectors = [self._update_key(key, event) for key in keys]
        return vectors

    def _update_key(self, key: str, event: EvalEvent) -> FeatureVector:
        state = self._states[key]
        now = event.time
        current_bucket = _bucket(now)
        if state.first_bucket is None:
            state.first_bucket = current_bucket
        history_start = current_bucket - self.history
        expired = [stamp for stamp in state.bucket_counts if stamp < history_start]
        for stamp in expired:
            state.bucket_counts.pop(stamp, None)
            state.bucket_types.pop(stamp, None)
        while state.recent and state.recent[0][0] < now - timedelta(hours=1):
            state.recent.popleft()

        effective_start = max(state.first_bucket, history_start)
        completed_bucket_count = max(
            0, int((current_bucket - effective_start).total_seconds() // (5 * 60))
        )
        completed_counts = [
            count
            for stamp, count in state.bucket_counts.items()
            if effective_start <= stamp < current_bucket
        ]
        count_distribution: Counter[float] = Counter(float(value) for value in completed_counts)
        count_distribution[0.0] += completed_bucket_count - len(completed_counts)
        median_5m = _counter_median(count_distribution, completed_bucket_count)
        deviation_distribution: Counter[float] = Counter()
        for value, frequency in count_distribution.items():
            deviation_distribution[abs(value - median_5m)] += frequency
        mad_5m = _counter_median(deviation_distribution, completed_bucket_count)
        hourly_by_bucket: Counter[datetime] = Counter()
        for stamp, count in state.bucket_counts.items():
            for offset in range(12):
                hour_stamp = stamp + timedelta(minutes=5 * offset)
                if effective_start <= hour_stamp < current_bucket:
                    hourly_by_bucket[hour_stamp] += count
        hourly_distribution: Counter[float] = Counter(
            float(value) for value in hourly_by_bucket.values()
        )
        hourly_distribution[0.0] += completed_bucket_count - len(hourly_by_bucket)
        median_1h = _counter_median(hourly_distribution, completed_bucket_count)
        hourly_deviations: Counter[float] = Counter()
        for value, frequency in hourly_distribution.items():
            hourly_deviations[abs(value - median_1h)] += frequency
        mad_1h = _counter_median(hourly_deviations, completed_bucket_count)
        current_5m = state.bucket_counts.get(current_bucket, 0) + 1
        current_1h = sum(1 for stamp, _ in state.recent if stamp > now - timedelta(hours=1)) + 1
        baseline_5m = max(median_5m, 1.0)
        baseline_1h = max(median_1h, 1.0)

        recent_types = Counter(
            kind for stamp, kind in state.recent if stamp > now - timedelta(hours=1)
        )
        recent_types[event.type] += 1
        baseline_types: Counter[str] = Counter()
        for stamp, kinds in state.bucket_types.items():
            if history_start <= stamp < current_bucket:
                baseline_types.update(kinds)
        data = event.data or {}
        actor = _actor(data)
        gap = (
            (now - state.last_time).total_seconds()
            if state.last_time
            else self.history.total_seconds()
        )
        log_gap = math.log1p(max(gap, 0.0))
        while state.gaps and state.gaps[0][0] < now - self.history:
            state.gaps.popleft()
        historic_gaps = np.asarray([value for _, value in state.gaps], dtype=float)
        median_gap = float(np.median(historic_gaps)) if len(historic_gaps) else log_gap
        mad_gap = (
            float(np.median(np.abs(historic_gaps - median_gap))) if len(historic_gaps) else 0.0
        )
        values = {
            "log_gap_s": log_gap,
            "count_5m_ratio": current_5m / baseline_5m,
            "count_1h_ratio": current_1h / baseline_1h,
            "type_mix_js": _js(recent_types, baseline_types),
            "actor_novelty": float(bool(actor) and actor not in state.actors),
            "subject_novelty": float(event.subject not in state.subjects),
            "priority": _priority(data),
            "log_money": _money(data),
            "time_of_day_bucket": float(now.astimezone(UTC).hour // 4),
        }
        state.last_time = now
        state.recent.append((now, event.type))
        state.bucket_counts[current_bucket] = current_5m
        state.bucket_types.setdefault(current_bucket, Counter())[event.type] += 1
        state.gaps.append((now, log_gap))
        if actor:
            state.actors.add(actor)
        state.subjects.add(event.subject)
        return FeatureVector(
            event_id=event.id,
            t=event.time,
            baseline_key=key,
            values=values,
            context={
                "count_5m": float(current_5m),
                "count_1h": float(current_1h),
                "baseline_median_5m": median_5m,
                "baseline_mad_5m": mad_5m,
                "baseline_median_1h": median_1h,
                "baseline_mad_1h": mad_1h,
                "robust_z_5m": (current_5m - median_5m) / max(1.4826 * mad_5m, 1.0),
                "robust_z_1h": (current_1h - median_1h) / max(1.4826 * mad_1h, 1.0),
                "baseline_median_log_gap": median_gap,
                "baseline_mad_log_gap": mad_gap,
                "window_epoch_5m": current_bucket.timestamp(),
            },
        )

    def transform(self, events: Iterable[EvalEvent]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for event in events:
            for vector in self.update(event):
                rows.append(
                    {
                        "event_id": vector.event_id,
                        "t": vector.t,
                        "baseline_key": vector.baseline_key,
                        **vector.values,
                        **vector.context,
                    }
                )
        return pd.DataFrame(rows)


def extract_features(events: Iterable[EvalEvent]) -> pd.DataFrame:
    """Convenience deterministic fold over an already event-time-ordered replay."""

    return CausalFeatureExtractor().transform(events)
