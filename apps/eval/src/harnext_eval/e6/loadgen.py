"""Workload fitting and frozen schedule generation for evaluation spec §7 E6.

Urgency is construction gold and is deliberately held in :class:`ScheduleEntry`
sidecars. It is never copied into an ``EvalEvent`` payload, so a router sees
only information available at the event timestamp.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

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

_DEFAULT_START = datetime(2035, 1, 1, tzinfo=UTC)


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
class SituationSpec:
    """Independent construction-gold situation injected into every workload arm."""

    situation_id: str
    entity: str
    onset_fraction: float
    archetype: str
    cost_weight: float = 1.0
    pulses: int = 3

    def __post_init__(self) -> None:
        if not 0 <= self.onset_fraction <= 1:
            raise ValueError("situation onset_fraction must be in [0, 1]")
        if self.cost_weight <= 0 or self.pulses <= 0:
            raise ValueError("situation cost_weight and pulses must be positive")


@dataclass(frozen=True)
class ScheduleEntry:
    """One timestamped event plus out-of-band gold or a worker marker."""

    intended_send_ts: datetime
    entity: str
    event: EvalEvent | None
    urgent: bool = False
    situation_id: str | None = None
    cost_weight: float = 0.0
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


@dataclass(frozen=True)
class SchedulePlan:
    """Frozen schedule and all generation metadata needed to reproduce it."""

    entries: tuple[ScheduleEntry, ...]
    schedule_id: str
    shape: str
    seed: int
    rate_hz: float
    duration_s: float
    burst_start_s: float
    burst_end_s: float
    entity_cardinality: int
    target_b: float | None
    realised_b: float
    tail_index: float | None = None
    calibration_iterations: int = 0
    calibration_converged: bool = True


def kim_jo_burstiness(inter_arrivals_s: Sequence[float]) -> float:
    """Return finite-size-corrected burstiness in ``[-1, 1]``."""

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
    ratio = float(values.std(ddof=0) / mean)
    root_plus = math.sqrt(n_events + 1)
    root_minus = math.sqrt(n_events - 1)
    denominator = (root_plus - 2.0) * ratio + root_minus
    if denominator == 0:
        return 0.0
    corrected = (root_plus * ratio - root_minus) / denominator
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
    covariance = np.mean((left - left.mean()) * (right - right.mean()))
    return float(np.clip(covariance / (left_std * right_std), -1.0, 1.0))


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

    replay = tuple(sorted(events, key=lambda event: (event.time, event.id)))
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
            quantiles = tuple(float(value) for value in observed_quantiles)
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
            quantiles_s=(quantiles[0], quantiles[1], quantiles[2]),
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
    value = start or _DEFAULT_START
    if value.tzinfo is None:
        raise ValueError("schedule start must be timezone-aware")
    return value


def _zipf_weights(entity_count: int, exponent: float) -> list[float]:
    if entity_count <= 0:
        return []
    raw = [rank ** (-max(exponent, 0.0)) for rank in range(1, entity_count + 1)]
    total = sum(raw)
    return [value / total for value in raw]


def _entity_population(fit: WorkloadFit, cardinality: int | None) -> tuple[str, ...]:
    requested = len(fit.entities) if cardinality is None else cardinality
    if requested <= 0:
        raise ValueError("entity cardinality must be positive")
    if requested <= len(fit.entities):
        return fit.entities[:requested]
    expanded = list(fit.entities)
    for index in range(len(expanded), requested):
        expanded.append(f"{fit.entities[index % len(fit.entities)]}:replica:{index}")
    return tuple(expanded)


def _clone_event(
    fit: WorkloadFit,
    entity: str,
    intended: datetime,
    index: int,
    rng: random.Random,
    *,
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
    digest = hashlib.sha256(
        f"e6:{intended.isoformat()}:{entity}:{index}:{template.id}".encode()
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
    entity_cardinality: int | None = None,
) -> list[ScheduleEntry]:
    rng = random.Random(seed)
    entities = _entity_population(fit, entity_cardinality)
    weights = _zipf_weights(
        len(entities), fit.zipf_exponent if exponent is None else exponent
    )
    hot = set(entities[:3])
    shifted_type = min(fit.type_mix, key=lambda name: fit.type_mix[name])
    entries: list[ScheduleEntry] = []
    for index, intended in enumerate(timestamps):
        entity = rng.choices(entities, weights=weights, k=1)[0]
        hot_shift = anomalous and entity in hot and rng.random() < 0.8
        event = _clone_event(
            fit,
            entity,
            intended,
            index,
            rng,
            forced_type=shifted_type if hot_shift else None,
        )
        entries.append(ScheduleEntry(intended, entity, event, shape=shape))
    return entries


def generate_steady_schedule(
    fit: WorkloadFit,
    *,
    rate_multiplier: float = 1.0,
    duration_s: float,
    start: datetime | None = None,
    seed: int = 1,
    rate_hz: float | None = None,
    entity_cardinality: int | None = None,
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
    return _entries_from_times(
        fit,
        timestamps,
        seed=seed,
        shape="steady",
        entity_cardinality=entity_cardinality,
    )


def generate_poisson_schedule(
    fit: WorkloadFit,
    *,
    rate_multiplier: float = 1.0,
    duration_s: float,
    start: datetime | None = None,
    seed: int = 1,
    rate_hz: float | None = None,
    entity_cardinality: int | None = None,
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
    return _entries_from_times(
        fit,
        timestamps,
        seed=seed + 1,
        shape="poisson",
        entity_cardinality=entity_cardinality,
    )


def generate_pareto_on_off(
    fit: WorkloadFit,
    *,
    tail_index: float,
    duration_s: float,
    rate_multiplier: float = 1.0,
    start: datetime | None = None,
    seed: int = 1,
    rate_hz: float | None = None,
    entity_cardinality: int | None = None,
) -> list[ScheduleEntry]:
    """Generate independent Pareto ON/OFF activity for every entity."""

    if tail_index <= 1.0:
        raise ValueError("Pareto tail_index must be greater than one")
    if duration_s <= 0 or rate_multiplier <= 0:
        raise ValueError("duration_s and rate_multiplier must be positive")
    begin = _start_time(start)
    aggregate_rate = (fit.mean_rate_hz if rate_hz is None else rate_hz) * rate_multiplier
    entities = _entity_population(fit, entity_cardinality)
    popularity = _zipf_weights(len(entities), fit.zipf_exponent)
    entries: list[ScheduleEntry] = []
    for entity_index, (entity, share) in enumerate(zip(entities, popularity, strict=True)):
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
                event = _clone_event(fit, entity, intended, local_index, rng)
                entries.append(ScheduleEntry(intended, entity, event, shape="pareto_on_off"))
                local_index += 1
                event_time += gap
            off_scale = max(on_duration * (burst_factor - 1.0), gap / 10.0)
            elapsed = on_end + rng.expovariate(1.0 / off_scale)
    entries.sort(key=lambda entry: (entry.intended_send_ts, entry.entity, entry.event_id))
    return entries


def schedule_burstiness(schedule: Sequence[ScheduleEntry]) -> float:
    """Estimate B from non-gold per-entity intervals in a generated schedule."""

    by_entity: dict[str, list[datetime]] = defaultdict(list)
    for entry in schedule:
        if entry.event is not None and not entry.urgent:
            by_entity[entry.entity].append(entry.intended_send_ts)
    scores: list[tuple[float, int]] = []
    for timestamps in by_entity.values():
        intervals = [
            (right - left).total_seconds()
            for left, right in zip(timestamps, timestamps[1:], strict=False)
        ]
        if len(intervals) >= 2:
            scores.append((kim_jo_burstiness(intervals), len(intervals)))
    if not scores:
        return -1.0
    return float(np.average([score for score, _ in scores], weights=[n for _, n in scores]))


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
    entity_cardinality: int | None = None,
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
            entity_cardinality=entity_cardinality,
        )
        realised = schedule_burstiness(schedule)
        if best is None or abs(realised - target_b) < abs(best[1] - target_b):
            best = (alpha, realised, schedule)
        if abs(realised - target_b) <= tolerance:
            break
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
    entity_cardinality: int | None,
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
        entries.extend(
            _entries_from_times(
                fit,
                timestamps,
                seed=seed + segment_index,
                shape=shape,
                exponent=fit.zipf_exponent + 1.5
                if shape == "zipf_hot" and in_burst
                else None,
                anomalous=shape == "anomalous_burst" and in_burst,
                entity_cardinality=entity_cardinality,
            )
        )
    entries.sort(key=lambda entry: (entry.intended_send_ts, entry.entity, entry.event_id))
    return entries


def default_situations(fit: WorkloadFit, *, count: int = 4) -> tuple[SituationSpec, ...]:
    """Return the deterministic catalogue used when corpus metadata has none."""

    archetypes = ("vip_dispute", "production_incident", "churn_signal", "security_report")
    fractions = (0.34, 0.46, 0.58, 0.70)
    return tuple(
        SituationSpec(
            situation_id=f"constructed-{index}-{archetypes[index % len(archetypes)]}",
            entity=fit.entities[index % len(fit.entities)],
            onset_fraction=fractions[index % len(fractions)],
            archetype=archetypes[index % len(archetypes)],
            cost_weight=float(index + 1),
        )
        for index in range(count)
    )


def situations_from_meta(
    fit: WorkloadFit, meta: Mapping[str, Any] | None
) -> tuple[SituationSpec, ...]:
    """Read corpus-side injected-situation gold when present, else construct it."""

    if not meta or "injected_situations" not in meta:
        return default_situations(fit)
    raw = meta["injected_situations"]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("corpus meta injected_situations must be a sequence")
    parsed: list[SituationSpec] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError("each injected situation must be a mapping")
        parsed.append(
            SituationSpec(
                situation_id=str(item.get("situation_id", f"corpus-{index}")),
                entity=str(item.get("entity", fit.entities[index % len(fit.entities)])),
                onset_fraction=float(item.get("onset_fraction", 0.3 + 0.1 * index)),
                archetype=str(item.get("archetype", "corpus_injected")),
                cost_weight=float(item.get("cost_weight", 1.0)),
                pulses=int(item.get("pulses", 3)),
            )
        )
    if not parsed:
        raise ValueError("corpus injected_situations cannot be empty")
    return tuple(parsed)


def _situation_data(archetype: str, pulse: int) -> dict[str, Any]:
    # Causal signal fields only: no urgency answer and no situation identifier.
    signals = {
        "vip_dispute": {"signal": "billing_velocity_change", "tier": "enterprise"},
        "production_incident": {"signal": "error_rate_change", "scope": "service"},
        "churn_signal": {"signal": "usage_contraction", "scope": "account"},
        "security_report": {"signal": "authentication_pattern_change", "scope": "tenant"},
    }
    return {**signals.get(archetype, {"signal": "state_change"}), "pulse": pulse}


def inject_situations(
    entries: Sequence[ScheduleEntry],
    fit: WorkloadFit,
    situations: Sequence[SituationSpec],
    *,
    start: datetime,
    duration_s: float,
    entity_cardinality: int | None = None,
) -> list[ScheduleEntry]:
    """Inject the same independently scripted urgent population into a schedule."""

    population = _entity_population(fit, entity_cardinality)
    output = list(entries)
    template = fit.replay_events[0]
    pulse_gap = max(min(duration_s / 100.0, 1.0), 1e-6)
    for situation_index, situation in enumerate(situations):
        entity = (
            situation.entity
            if situation.entity in population
            else population[situation_index % len(population)]
        )
        for pulse in range(situation.pulses):
            offset = min(
                duration_s - 1e-9,
                duration_s * situation.onset_fraction + pulse * pulse_gap,
            )
            intended = start + timedelta(seconds=max(0.0, offset))
            event_id = hashlib.sha256(
                f"urgent:{situation.situation_id}:{pulse}".encode()
            ).hexdigest()[:24]
            event = template.model_copy(
                update={
                    "id": event_id,
                    "subject": entity,
                    "type": f"org.harnext.signal.{situation.archetype}",
                    "time": intended,
                    "intended_send_ts": intended,
                    "baseline_keys": [entity],
                    "data": _situation_data(situation.archetype, pulse),
                }
            )
            output.append(
                ScheduleEntry(
                    intended_send_ts=intended,
                    entity=entity,
                    event=event,
                    urgent=True,
                    situation_id=situation.situation_id,
                    cost_weight=situation.cost_weight,
                    shape="injected_situation",
                )
            )
    output.sort(key=lambda entry: (entry.intended_send_ts, entry.entity, entry.event_id))
    return output


def schedule_fingerprint(schedule: Sequence[ScheduleEntry]) -> str:
    """Hash IDs and intended timestamps to prove paired schedules are identical."""

    digest = hashlib.sha256()
    for entry in schedule:
        digest.update(entry.event_id.encode())
        digest.update(entry.intended_send_ts.isoformat().encode())
    return digest.hexdigest()


def build_schedule(
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
    situations: Sequence[SituationSpec] | None = None,
    entity_cardinality: int | None = None,
    require_convergence: bool = True,
) -> SchedulePlan:
    """Generate and freeze one paired workload cell with calibration metadata."""

    begin = _start_time(start)
    base_rate = (fit.mean_rate_hz if rate_hz is None else rate_hz) * rate_multiplier
    if burst_start_s is None:
        burst_start_s = max(0.0, (duration_s - min(burst_duration_s, duration_s)) / 2.0)
    burst_end_s = min(duration_s, burst_start_s + burst_duration_s)
    calibration: CalibrationResult | None = None
    if shape == "steady":
        entries = generate_steady_schedule(
            fit,
            duration_s=duration_s,
            start=begin,
            seed=seed,
            rate_hz=base_rate,
            entity_cardinality=entity_cardinality,
        )
    elif shape == "poisson":
        entries = generate_poisson_schedule(
            fit,
            duration_s=duration_s,
            start=begin,
            seed=seed,
            rate_hz=base_rate,
            entity_cardinality=entity_cardinality,
        )
    elif shape == "pareto_on_off":
        calibration = calibrate_pareto_on_off(
            fit,
            target_b=fit.burstiness_b if target_b is None else target_b,
            duration_s=duration_s,
            start=begin,
            seed=seed,
            rate_hz=base_rate,
            entity_cardinality=entity_cardinality,
        )
        if require_convergence and not calibration.converged:
            raise ValueError(
                f"ON/OFF calibration did not converge: target={calibration.target_b}, "
                f"realised={calibration.realised_b}"
            )
        entries = list(calibration.schedule)
    else:
        entries = _piecewise_schedule(
            fit,
            shape=shape,
            rate_hz=base_rate,
            duration_s=duration_s,
            burst_start_s=burst_start_s,
            burst_duration_s=burst_duration_s,
            start=begin,
            seed=seed,
            entity_cardinality=entity_cardinality,
        )
    entries = inject_situations(
        entries,
        fit,
        situations or default_situations(fit),
        start=begin,
        duration_s=duration_s,
        entity_cardinality=entity_cardinality,
    )
    if worker_kill:
        peak = burst_start_s + (burst_end_s - burst_start_s) / 2.0
        entries.append(
            ScheduleEntry(
                intended_send_ts=begin + timedelta(seconds=peak),
                entity="__control__",
                event=None,
                shape=shape,
                marker="worker_kill",
            )
        )
        entries.sort(key=lambda entry: (entry.intended_send_ts, entry.marker is None))
    realised = calibration.realised_b if calibration else schedule_burstiness(entries)
    return SchedulePlan(
        entries=tuple(entries),
        schedule_id=schedule_fingerprint(entries),
        shape=shape,
        seed=seed,
        rate_hz=base_rate,
        duration_s=duration_s,
        burst_start_s=burst_start_s,
        burst_end_s=burst_end_s,
        entity_cardinality=len(_entity_population(fit, entity_cardinality)),
        target_b=target_b,
        realised_b=realised,
        tail_index=calibration.tail_index if calibration else None,
        calibration_iterations=calibration.iterations if calibration else 0,
        calibration_converged=calibration.converged if calibration else True,
    )


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
    situations: Sequence[SituationSpec] | None = None,
    entity_cardinality: int | None = None,
    require_convergence: bool = True,
) -> list[ScheduleEntry]:
    """Compatibility wrapper returning only the entries of :func:`build_schedule`."""

    return list(
        build_schedule(
            fit,
            shape=shape,
            duration_s=duration_s,
            rate_multiplier=rate_multiplier,
            start=start,
            seed=seed,
            rate_hz=rate_hz,
            target_b=target_b,
            burst_start_s=burst_start_s,
            burst_duration_s=burst_duration_s,
            worker_kill=worker_kill,
            situations=situations,
            entity_cardinality=entity_cardinality,
            require_convergence=require_convergence,
        ).entries
    )


estimate_burstiness = kim_jo_burstiness
fit_from_replay = fit_workload


__all__ = [
    "CalibrationResult",
    "EntityInterArrivalFit",
    "ScheduleEntry",
    "SchedulePlan",
    "SituationSpec",
    "WorkloadFit",
    "WorkloadShape",
    "build_schedule",
    "calibrate_pareto_on_off",
    "default_situations",
    "estimate_burstiness",
    "fit_from_replay",
    "fit_workload",
    "generate_pareto_on_off",
    "generate_poisson_schedule",
    "generate_schedule",
    "generate_steady_schedule",
    "inject_situations",
    "kim_jo_burstiness",
    "memory_coefficient",
    "schedule_burstiness",
    "schedule_fingerprint",
    "situations_from_meta",
]
