"""Workload fitting and schedule generation for docs/evaluation-spec.md §7 E6.

The finite-size burstiness estimator is the Kim--Jo :math:`A_n(r)` correction
(Phys. Rev. E 94, 032311, equation 22).  Schedules carry wall-clock intended
send timestamps so the runner can detect coordinated omission.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal

import numpy as np

from harnext_eval.types import EvalEvent

WorkloadShape = Literal[
    "steady",
    "pareto_on_off",
    "poisson",
    "benign_flash",
    "anomalous_burst",
    "zipf_hot",
]


@dataclass(frozen=True)
class EntityInterArrivalFit:
    """Observed inter-arrival summary for one replay entity."""

    entity: str
    count: int
    rate_hz: float
    mean_s: float
    std_s: float
    quantiles_s: tuple[float, float, float]
    burstiness_b: float
    memory_m: float
    inter_arrivals_s: tuple[float, ...] = field(repr=False)


@dataclass(frozen=True)
class WorkloadFit:
    """Parameters needed to reproduce the temporal and popularity workload."""

    entity_fits: dict[str, EntityInterArrivalFit]
    entities: tuple[str, ...]
    popularity: tuple[float, ...]
    zipf_exponent: float
    mean_rate_hz: float
    burstiness_b: float
    memory_m: float
    type_mix: dict[str, float]
    replay_events: tuple[EvalEvent, ...] = field(repr=False)


@dataclass(frozen=True)
class ScheduleEntry:
    """One timestamped event or an out-of-band worker control marker."""

    intended_send_ts: datetime
    entity: str
    event: EvalEvent | None
    urgent: bool = False
    shape: str = "steady"
    marker: Literal["worker_kill"] | None = None

    @property
    def event_id(self) -> str:
        return self.event.id if self.event is not None else f"marker:{self.marker}"


@dataclass(frozen=True)
class CalibrationResult:
    """Result of the ON-tail-index bisection."""

    tail_index: float
    realised_b: float
    target_b: float
    converged: bool
    iterations: int
    schedule: tuple[ScheduleEntry, ...]


def kim_jo_burstiness(inter_arrivals_s: Sequence[float]) -> float:
    """Return finite-size-corrected burstiness in ``[-1, 1]``.

    ``n`` in Kim and Jo is the number of events, hence one more than the
    number of open-boundary inter-arrival observations supplied here.  The
    population standard deviation is used to match the paper's definition.
    """

    values = np.asarray(inter_arrivals_s, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or np.any(values < 0):
        raise ValueError("inter-arrivals must contain finite non-negative values")
    mean = float(values.mean())
    if mean == 0:
        return 1.0
    n_events = int(values.size + 1)
    if n_events < 3:
        return -1.0 if float(values.std()) == 0 else 0.0
    r = float(values.std(ddof=0) / mean)
    root_plus = math.sqrt(n_events + 1)
    root_minus = math.sqrt(n_events - 1)
    denominator = (root_plus - 2.0) * r + root_minus
    if denominator == 0:
        return 0.0
    corrected = (root_plus * r - root_minus) / denominator
    return float(np.clip(corrected, -1.0, 1.0))


def memory_coefficient(inter_arrivals_s: Sequence[float]) -> float:
    """Return the Goh--Barabási adjacent inter-arrival memory coefficient M."""

    values = np.asarray(inter_arrivals_s, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return 0.0
    left = values[:-1]
    right = values[1:]
    left_std = float(left.std(ddof=0))
    right_std = float(right.std(ddof=0))
    if left_std == 0 or right_std == 0:
        return 0.0
    return float(np.clip(np.mean((left - left.mean()) * (right - right.mean())) / (
        left_std * right_std
    ), -1.0, 1.0))


def _entity(event: EvalEvent) -> str:
    return event.subject or (event.baseline_keys[0] if event.baseline_keys else event.id)


def _zipf_exponent(counts: Sequence[int]) -> float:
    positive = np.asarray(sorted((count for count in counts if count > 0), reverse=True))
    if positive.size < 2 or np.all(positive == positive[0]):
        return 0.0
    ranks = np.arange(1, positive.size + 1, dtype=float)
    slope, _ = np.polyfit(np.log(ranks), np.log(positive), 1)
    return max(0.0, float(-slope))


def fit_workload(events: Iterable[EvalEvent]) -> WorkloadFit:
    """Fit inter-arrivals, Zipf popularity, B, M, and type mix from a replay."""

    replay = tuple(sorted(events, key=lambda event: event.time))
    if len(replay) < 2:
        raise ValueError("at least two replay events are required")
    by_entity: dict[str, list[datetime]] = defaultdict(list)
    type_counts: Counter[str] = Counter()
    for event in replay:
        by_entity[_entity(event)].append(event.time)
        type_counts[event.type] += 1

    duration_s = max((replay[-1].time - replay[0].time).total_seconds(), 1e-9)
    entity_fits: dict[str, EntityInterArrivalFit] = {}
    all_intervals: list[float] = []
    for entity, timestamps in by_entity.items():
        intervals = [
            (right - left).total_seconds()
            for left, right in zip(timestamps, timestamps[1:], strict=False)
        ]
        all_intervals.extend(intervals)
        array = np.asarray(intervals, dtype=float)
        if array.size:
            observed_quantiles = np.quantile(array, [0.5, 0.9, 0.99])
            quantiles = (
                float(observed_quantiles[0]),
                float(observed_quantiles[1]),
                float(observed_quantiles[2]),
            )
            mean_s = float(array.mean())
            std_s = float(array.std(ddof=0))
            burstiness = kim_jo_burstiness(intervals)
            memory = memory_coefficient(intervals)
        else:
            quantiles = (duration_s, duration_s, duration_s)
            mean_s = duration_s
            std_s = 0.0
            burstiness = -1.0
            memory = 0.0
        entity_fits[entity] = EntityInterArrivalFit(
            entity=entity,
            count=len(timestamps),
            rate_hz=len(timestamps) / duration_s,
            mean_s=mean_s,
            std_s=std_s,
            quantiles_s=quantiles,
            burstiness_b=burstiness,
            memory_m=memory,
            inter_arrivals_s=tuple(intervals),
        )

    ranked = sorted(entity_fits, key=lambda name: (-entity_fits[name].count, name))
    counts = [entity_fits[name].count for name in ranked]
    total = sum(counts)
    return WorkloadFit(
        entity_fits=entity_fits,
        entities=tuple(ranked),
        popularity=tuple(count / total for count in counts),
        zipf_exponent=_zipf_exponent(counts),
        mean_rate_hz=len(replay) / duration_s,
        burstiness_b=kim_jo_burstiness(all_intervals),
        memory_m=memory_coefficient(all_intervals),
        type_mix={name: count / len(replay) for name, count in sorted(type_counts.items())},
        replay_events=replay,
    )


def _start_time(start: datetime | None) -> datetime:
    value = start or datetime.now(UTC) + timedelta(milliseconds=50)
    if value.tzinfo is None:
        raise ValueError("schedule start must be timezone-aware")
    return value


def _zipf_weights(entity_count: int, exponent: float) -> list[float]:
    if entity_count <= 0:
        return []
    raw = [rank ** (-max(exponent, 0.0)) for rank in range(1, entity_count + 1)]
    total = sum(raw)
    return [value / total for value in raw]


def _clone_event(
    fit: WorkloadFit,
    entity: str,
    intended: datetime,
    index: int,
    rng: random.Random,
    *,
    shape: str,
    urgent: bool = False,
    forced_type: str | None = None,
) -> EvalEvent:
    candidates = fit.replay_events
    if forced_type is not None:
        typed = tuple(event for event in candidates if event.type == forced_type)
        if typed:
            candidates = typed
    template = candidates[rng.randrange(len(candidates))]
    data = dict(template.data or {})
    if "issue_key" in data:
        data["issue_key"] = entity.removeprefix("issue:")
    data.update({"e6_shape": shape, "e6_urgent": urgent})
    digest = hashlib.sha256(
        f"e6:{shape}:{intended.isoformat()}:{entity}:{index}:{template.id}".encode()
    ).hexdigest()[:24]
    return template.model_copy(
        update={
            "id": digest,
            "subject": entity,
            "time": intended,
            "intended_send_ts": intended,
            "data": data,
        }
    )


def _entries_from_times(
    fit: WorkloadFit,
    timestamps: Sequence[datetime],
    *,
    seed: int,
    shape: str,
    exponent: float | None = None,
    anomalous: bool = False,
) -> list[ScheduleEntry]:
    rng = random.Random(seed)
    weights = _zipf_weights(len(fit.entities), fit.zipf_exponent if exponent is None else exponent)
    hot = set(fit.entities[:3])
    shifted_type = max(fit.type_mix, key=lambda name: fit.type_mix[name])
    entries: list[ScheduleEntry] = []
    for index, intended in enumerate(timestamps):
        entity = rng.choices(fit.entities, weights=weights, k=1)[0]
        hot_shift = anomalous and entity in hot and rng.random() < 0.8
        # Preserve a small baseline urgent population on cold entities so the
        # fairness score remains identifiable during a hot-entity burst.
        urgent = hot_shift or (anomalous and entity not in hot and rng.random() < 0.05)
        forced_type = shifted_type if hot_shift else None
        event = _clone_event(
            fit,
            entity,
            intended,
            index,
            rng,
            shape=shape,
            urgent=urgent,
            forced_type=forced_type,
        )
        entries.append(
            ScheduleEntry(
                intended_send_ts=intended,
                entity=entity,
                event=event,
                urgent=urgent,
                shape=shape,
            )
        )
    return entries


def generate_steady_schedule(
    fit: WorkloadFit,
    *,
    rate_multiplier: float = 1.0,
    duration_s: float,
    start: datetime | None = None,
    seed: int = 1,
    rate_hz: float | None = None,
) -> list[ScheduleEntry]:
    """Generate an evenly paced open-loop schedule at ``k ×`` fitted mean."""

    if rate_multiplier <= 0 or duration_s <= 0:
        raise ValueError("rate_multiplier and duration_s must be positive")
    rate = (fit.mean_rate_hz if rate_hz is None else rate_hz) * rate_multiplier
    if rate <= 0:
        raise ValueError("schedule rate must be positive")
    begin = _start_time(start)
    count = max(1, int(math.ceil(rate * duration_s)))
    timestamps = [begin + timedelta(seconds=index / rate) for index in range(count)]
    return _entries_from_times(fit, timestamps, seed=seed, shape="steady")


def generate_poisson_schedule(
    fit: WorkloadFit,
    *,
    rate_multiplier: float = 1.0,
    duration_s: float,
    start: datetime | None = None,
    seed: int = 1,
    rate_hz: float | None = None,
) -> list[ScheduleEntry]:
    """Generate the B→0 Poisson control at the requested mean rate."""

    if rate_multiplier <= 0 or duration_s <= 0:
        raise ValueError("rate_multiplier and duration_s must be positive")
    rate = (fit.mean_rate_hz if rate_hz is None else rate_hz) * rate_multiplier
    rng = random.Random(seed)
    begin = _start_time(start)
    elapsed = 0.0
    timestamps: list[datetime] = []
    while True:
        elapsed += rng.expovariate(rate)
        if elapsed >= duration_s:
            break
        timestamps.append(begin + timedelta(seconds=elapsed))
    if not timestamps:
        timestamps.append(begin)
    return _entries_from_times(fit, timestamps, seed=seed + 1, shape="poisson")


def generate_pareto_on_off(
    fit: WorkloadFit,
    *,
    tail_index: float,
    duration_s: float,
    rate_multiplier: float = 1.0,
    start: datetime | None = None,
    seed: int = 1,
    rate_hz: float | None = None,
) -> list[ScheduleEntry]:
    """Generate independent Pareto ON/OFF activity for every entity."""

    if tail_index <= 1.0:
        raise ValueError("Pareto tail_index must be greater than one")
    if duration_s <= 0 or rate_multiplier <= 0:
        raise ValueError("duration_s and rate_multiplier must be positive")
    begin = _start_time(start)
    aggregate_rate = (fit.mean_rate_hz if rate_hz is None else rate_hz) * rate_multiplier
    popularity = fit.popularity or tuple([1 / len(fit.entities)] * len(fit.entities))
    entries: list[ScheduleEntry] = []
    for entity_index, (entity, share) in enumerate(zip(fit.entities, popularity, strict=True)):
        rng = random.Random(seed * 10_007 + entity_index)
        entity_rate = max(aggregate_rate * share, 1 / duration_s)
        burst_factor = 1.0 + min(19.0, 4.0 / (tail_index - 1.0))
        on_rate = max(entity_rate * burst_factor, 2.0 / duration_s)
        minimum_on = max(2.0 / on_rate, min(duration_s / 20.0, 0.25))
        elapsed = rng.random() * minimum_on
        local_index = 0
        while elapsed < duration_s:
            on_duration = minimum_on * rng.paretovariate(tail_index)
            on_end = min(duration_s, elapsed + on_duration)
            gap = 1.0 / on_rate
            event_time = elapsed
            while event_time < on_end:
                intended = begin + timedelta(seconds=event_time)
                event = _clone_event(
                    fit,
                    entity,
                    intended,
                    local_index,
                    rng,
                    shape="pareto_on_off",
                )
                entries.append(
                    ScheduleEntry(intended, entity, event, shape="pareto_on_off")
                )
                local_index += 1
                event_time += gap
            # A proportional OFF period keeps the long-run mean approximately
            # fixed.  Its ratio converges to zero as alpha grows, giving the
            # bisection a Poisson-like lower endpoint as well as a bursty one.
            off_scale = max(on_duration * (burst_factor - 1.0), gap / 10.0)
            elapsed = on_end + rng.expovariate(1.0 / off_scale)
    entries.sort(key=lambda entry: (entry.intended_send_ts, entry.entity, entry.event_id))
    return entries


def schedule_burstiness(schedule: Sequence[ScheduleEntry]) -> float:
    """Estimate B from per-entity intervals in a generated schedule."""

    by_entity: dict[str, list[datetime]] = defaultdict(list)
    for entry in schedule:
        if entry.event is not None:
            by_entity[entry.entity].append(entry.intended_send_ts)
    intervals: list[float] = []
    for timestamps in by_entity.values():
        intervals.extend(
            (right - left).total_seconds()
            for left, right in zip(timestamps, timestamps[1:], strict=False)
        )
    return kim_jo_burstiness(intervals) if intervals else -1.0


def calibrate_pareto_on_off(
    fit: WorkloadFit,
    *,
    target_b: float,
    duration_s: float,
    rate_multiplier: float = 1.0,
    start: datetime | None = None,
    seed: int = 1,
    rate_hz: float | None = None,
    tolerance: float = 0.05,
    max_iterations: int = 18,
    tail_bounds: tuple[float, float] = (1.05, 50.0),
) -> CalibrationResult:
    """Bisect the ON-period Pareto tail index until realised B matches target."""

    if not -1.0 <= target_b <= 1.0:
        raise ValueError("target_b must be in [-1, 1]")
    low, high = tail_bounds
    if low <= 1 or high <= low:
        raise ValueError("tail_bounds must satisfy 1 < low < high")
    best: tuple[float, float, list[ScheduleEntry]] | None = None
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        alpha = (low + high) / 2.0
        schedule = generate_pareto_on_off(
            fit,
            tail_index=alpha,
            duration_s=duration_s,
            rate_multiplier=rate_multiplier,
            start=start,
            seed=seed,
            rate_hz=rate_hz,
        )
        realised = schedule_burstiness(schedule)
        if best is None or abs(realised - target_b) < abs(best[1] - target_b):
            best = (alpha, realised, schedule)
        if abs(realised - target_b) <= tolerance:
            break
        # Smaller alpha produces longer ON runs and therefore more burstiness.
        if realised < target_b:
            high = alpha
        else:
            low = alpha
    assert best is not None
    return CalibrationResult(
        tail_index=best[0],
        realised_b=best[1],
        target_b=target_b,
        converged=abs(best[1] - target_b) <= tolerance,
        iterations=iterations,
        schedule=tuple(best[2]),
    )


def _piecewise_schedule(
    fit: WorkloadFit,
    *,
    shape: Literal["benign_flash", "anomalous_burst", "zipf_hot"],
    rate_hz: float,
    duration_s: float,
    burst_start_s: float,
    burst_duration_s: float,
    start: datetime,
    seed: int,
) -> list[ScheduleEntry]:
    if not 0 <= burst_start_s < duration_s:
        raise ValueError("burst_start_s must fall inside the schedule")
    burst_end_s = min(duration_s, burst_start_s + burst_duration_s)
    segments = (
        (0.0, burst_start_s, rate_hz, False),
        (burst_start_s, burst_end_s, rate_hz * 5.0, True),
        (burst_end_s, duration_s, rate_hz, False),
    )
    entries: list[ScheduleEntry] = []
    for segment_index, (left, right, segment_rate, in_burst) in enumerate(segments):
        if right <= left:
            continue
        count = max(1, int(math.ceil(segment_rate * (right - left))))
        timestamps = [
            start + timedelta(seconds=left + index / segment_rate) for index in range(count)
        ]
        anomalous = shape == "anomalous_burst" and in_burst
        exponent = fit.zipf_exponent + 1.5 if shape == "zipf_hot" and in_burst else None
        part = _entries_from_times(
            fit,
            timestamps,
            seed=seed + segment_index,
            shape=shape,
            exponent=exponent,
            anomalous=anomalous,
        )
        entries.extend(part)
    entries.sort(key=lambda entry: (entry.intended_send_ts, entry.entity, entry.event_id))
    return entries


def generate_schedule(
    fit: WorkloadFit,
    *,
    shape: WorkloadShape,
    duration_s: float,
    rate_multiplier: float = 1.0,
    start: datetime | None = None,
    seed: int = 1,
    rate_hz: float | None = None,
    target_b: float | None = None,
    burst_start_s: float | None = None,
    burst_duration_s: float = 600.0,
    worker_kill: bool = False,
) -> list[ScheduleEntry]:
    """Generate any E6 workload shape, optionally with a peak kill marker."""

    begin = _start_time(start)
    base_rate = (fit.mean_rate_hz if rate_hz is None else rate_hz) * rate_multiplier
    if shape == "steady":
        entries = generate_steady_schedule(
            fit, duration_s=duration_s, start=begin, seed=seed, rate_hz=base_rate
        )
    elif shape == "poisson":
        entries = generate_poisson_schedule(
            fit, duration_s=duration_s, start=begin, seed=seed, rate_hz=base_rate
        )
    elif shape == "pareto_on_off":
        calibration = calibrate_pareto_on_off(
            fit,
            target_b=fit.burstiness_b if target_b is None else target_b,
            duration_s=duration_s,
            start=begin,
            seed=seed,
            rate_hz=base_rate,
        )
        entries = list(calibration.schedule)
    else:
        if burst_start_s is None:
            burst_start_s = max(0.0, (duration_s - min(burst_duration_s, duration_s)) / 2.0)
        entries = _piecewise_schedule(
            fit,
            shape=shape,
            rate_hz=base_rate,
            duration_s=duration_s,
            burst_start_s=burst_start_s,
            burst_duration_s=burst_duration_s,
            start=begin,
            seed=seed,
        )
    if worker_kill:
        peak = (burst_start_s or duration_s / 2.0) + min(burst_duration_s, duration_s) / 2.0
        entries.append(
            ScheduleEntry(
                intended_send_ts=begin + timedelta(seconds=min(peak, duration_s)),
                entity="__control__",
                event=None,
                shape=shape,
                marker="worker_kill",
            )
        )
        entries.sort(key=lambda entry: (entry.intended_send_ts, entry.marker is None))
    return entries


# Readable aliases used by experiment notebooks and downstream integrations.
estimate_burstiness = kim_jo_burstiness
fit_from_replay = fit_workload


__all__ = [
    "CalibrationResult",
    "EntityInterArrivalFit",
    "ScheduleEntry",
    "WorkloadFit",
    "WorkloadShape",
    "calibrate_pareto_on_off",
    "estimate_burstiness",
    "fit_from_replay",
    "fit_workload",
    "generate_pareto_on_off",
    "generate_poisson_schedule",
    "generate_schedule",
    "generate_steady_schedule",
    "kim_jo_burstiness",
    "memory_coefficient",
    "schedule_burstiness",
]
