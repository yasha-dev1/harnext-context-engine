"""Paired open-loop E6 runner for evaluation spec §7 E6 and §12 D10."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib
import json
import math
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from hdrh.histogram import HdrHistogram

from harnext_eval.config import EngineConfig
from harnext_eval.corpus import CorpusHandle
from harnext_eval.e1.policies import GuardedHBOSPolicy
from harnext_eval.e6.loadgen import (
    ScheduleEntry,
    SchedulePlan,
    SituationSpec,
    WorkloadFit,
    build_schedule,
    fit_workload,
    generate_steady_schedule,
    schedule_fingerprint,
    situations_from_meta,
)
from harnext_eval.e6.metrics import (
    URGENT_SLO_S,
    cross_entity_fairness,
    demand_curve,
    drain_time,
    duplicates_and_missed,
    histogram_percentile,
    linear_trend,
    make_histogram,
    partition_lag_gini,
    recovery_time,
    slo_attainment,
)
from harnext_eval.registry import ExperimentResult, register_experiment
from harnext_eval.replay.driver import RouterPolicy, RulesOnlyPolicy
from harnext_eval.stats.stats import paired_difference_bca
from harnext_eval.types import EvalEvent


@dataclass(frozen=True)
class RunnerConfig:
    """Transport, topology and deployment controls for one E6 cell."""

    transport: Literal["in-process", "kafka"] = "in-process"
    partitions: int = 8
    workers: int = 1
    service_time_s: float = 0.002
    admission_delay_s: float = 0.0
    warmup_fraction: float = 0.25
    kill_downtime_s: float = 0.05
    batch_windows: bool = True
    window_time_scale: float = 0.001
    burst_start_s: float | None = None
    burst_end_s: float | None = None
    service_time_tolerance_s: float = 0.002
    load_generator_host: str = "local"
    kafka_bootstrap_servers: str | None = None
    kafka_input_topic: str = "cms.events.raw.v1"
    kafka_output_topic: str | None = None
    kafka_timeout_s: float = 30.0
    kafka_telemetry_path: Path | None = None

    def __post_init__(self) -> None:
        if self.partitions <= 0 or self.workers <= 0:
            raise ValueError("partitions and workers must be positive")
        if self.service_time_s < 0 or self.admission_delay_s < 0:
            raise ValueError("service times cannot be negative")
        if self.kill_downtime_s < 0 or self.window_time_scale <= 0:
            raise ValueError("outage and window scaling must be non-negative/positive")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if (self.burst_start_s is None) != (self.burst_end_s is None):
            raise ValueError("burst_start_s and burst_end_s must be set together")
        if (
            self.burst_start_s is not None
            and self.burst_end_s is not None
            and self.burst_end_s <= self.burst_start_s
        ):
            raise ValueError("burst end must be after burst start")


@dataclass(frozen=True)
class GuardOverrides:
    absolute_floor: float | None = None
    multi_window: bool | None = None
    situation_dedup: bool | None = None


@dataclass(frozen=True)
class EventObservation:
    event_id: str
    entity: str
    lane: str
    urgent: bool
    situation_id: str | None
    cost_weight: float
    intended_offset_s: float
    actual_send_offset_s: float
    send_skew_s: float
    agent_start_offset_s: float
    latency_s: float
    service_start_offset_s: float
    service_end_offset_s: float
    service_elapsed_s: float
    window_close_offset_s: float | None
    commit_offset_s: float | None
    batch_fold_latency_s: float | None
    batch_staleness_s: float | None
    partition: int


@dataclass
class PipelineRun:
    observations: list[EventObservation]
    histograms: dict[str, HdrHistogram]
    lag_samples: pd.DataFrame
    metrics: dict[str, float]
    expected_by_lane: dict[str, list[str]] = field(default_factory=dict)
    observed_by_lane: dict[str, list[str]] = field(default_factory=dict)
    route_decisions: pd.DataFrame = field(default_factory=pd.DataFrame)
    kill_offset_s: float | None = None
    gold_by_event: dict[str, dict[str, Any]] = field(default_factory=dict)

    def write_histograms(
        self,
        directory: Path,
        prefix: str,
        *,
        names: Sequence[str] | None = None,
        full_distribution: bool = False,
    ) -> list[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for name, histogram in sorted(self.histograms.items()):
            if names is not None and name not in names:
                continue
            path = directory / f"{prefix}-{name}.hgrm"
            if full_distribution:
                with path.open("wb") as output:
                    histogram.output_percentile_distribution(output, 1_000_000.0)
            else:
                lines = ["#[HdrHistogram log format version 1.3]", "Value,Percentile,TotalCount"]
                for percentile_value in (50.0, 90.0, 99.0, 99.9, 100.0):
                    lines.append(
                        f"{histogram_percentile(histogram, percentile_value):.9f},"
                        f"{percentile_value:.3f},{histogram.get_total_count()}"
                    )
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            paths.append(path)
        return paths


@dataclass(frozen=True)
class _QueuedEvent:
    entry: ScheduleEntry
    intended_loop_time: float
    actual_send_loop_time: float
    intended_offset_s: float
    partition: int
    lane: str
    sequence: int


@dataclass(frozen=True)
class _QueuedFold:
    items: tuple[_QueuedEvent, ...]
    lane: str
    window_close_loop_time: float
    partition: int
    sequence: int


def _partition(entity: str, partitions: int) -> int:
    digest = hashlib.sha256(entity.encode()).digest()
    return int.from_bytes(digest[:8], "big") % partitions


def _constructed_situation_id(event: EvalEvent, offset_s: float, window_s: float) -> str:
    """Construct a causal dedup key without reading construction-gold metadata."""

    family = ".".join(event.type.split(".")[:4])
    bucket = math.floor(offset_s / max(window_s, 1e-9))
    raw = f"{event.mgtenant}:{event.subject}:{event.source}:{family}:{bucket}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


class E6RouterPolicy:
    """Causal budget wrapper around E1's real R5 ``RouterPolicy`` seam."""

    name = "R5-e6-causal-budget"

    def __init__(
        self,
        cfg: EngineConfig,
        *,
        training_events: Sequence[EvalEvent] = (),
        lane_design: Literal["two-lane", "single"] | None = None,
        guard_overrides: GuardOverrides | None = None,
        policy: RouterPolicy | None = None,
    ) -> None:
        self.cfg = cfg
        self.lane_design = lane_design or cfg.lane_design
        overrides = guard_overrides or GuardOverrides()
        self.absolute_floor = (
            max(2.0, cfg.router.guards.absolute_floor)
            if overrides.absolute_floor is None
            else overrides.absolute_floor
        )
        self.multi_window = (
            True if overrides.multi_window is None else overrides.multi_window
        )
        self.situation_dedup = (
            True if overrides.situation_dedup is None else overrides.situation_dedup
        )
        actual: RouterPolicy
        if self.lane_design == "single":
            actual = RulesOnlyPolicy()
        elif policy is None:
            # R5 supplies the causal HBOS scorer/rules. E6 applies the guards
            # explicitly so each can be ablated against an identical schedule.
            r5 = GuardedHBOSPolicy(absolute_floor=0.0, multi_window=False)
            r5.fit(list(training_events))
            actual = r5
        else:
            actual = copy.deepcopy(policy)
        if not isinstance(actual, RouterPolicy):
            raise TypeError("E6 router must implement the shared RouterPolicy protocol")
        self.policy = actual
        self._volume: dict[str, deque[float]] = defaultdict(deque)
        self._anomaly_windows: dict[str, set[int]] = defaultdict(set)
        self._admitted_situations: set[str] = set()
        self.rule_negative_seen = 0
        self.deviation_admitted = 0
        self.rule_admitted = 0

    def rules(self, event: EvalEvent) -> str | None:
        return self.policy.rules(event)

    def score(self, event: EvalEvent) -> float:
        return self.policy.score(event)

    def route(self, event: EvalEvent, offset_s: float) -> tuple[str, dict[str, Any]]:
        """Route using causal features only; gold is intentionally not accepted."""

        if self.lane_design == "single":
            return "single", {"single_lane": True, "policy": self.name}
        rule = self.rules(event) if self.cfg.router.rules.enabled else None
        score = float(self.score(event))
        if not math.isfinite(score):
            raise ValueError(f"R5 returned a non-finite score for {event.id}")
        key = event.baseline_keys[0] if event.baseline_keys else event.subject
        recent = self._volume[key]
        while recent and recent[0] < offset_s - 300.0:
            recent.popleft()
        recent.append(offset_s)
        volume_pass = len(recent) >= self.absolute_floor
        candidate = score > -1e11
        window_span = max(self.cfg.window.gap_s, 1e-9)
        window_index = math.floor(offset_s / window_span)
        prior_window = any(index < window_index for index in self._anomaly_windows[key])
        confirmed = not self.multi_window or (candidate and prior_window)
        if candidate:
            self._anomaly_windows[key].add(window_index)
        situation = _constructed_situation_id(event, offset_s, window_span)
        unique = not self.situation_dedup or situation not in self._admitted_situations

        if rule is not None:
            self.rule_admitted += 1
            admitted = True
            budget_pass = True
        else:
            self.rule_negative_seen += 1
            allowed = math.floor(
                self.rule_negative_seen * self.cfg.router.budget_pct / 100.0
            )
            budget_pass = self.deviation_admitted < allowed
            admitted = candidate and volume_pass and confirmed and unique and budget_pass
            if admitted:
                self.deviation_admitted += 1
        if admitted and rule is None:
            self._admitted_situations.add(situation)
        return (
            "fast" if admitted else "batch",
            {
                "policy": self.name,
                "score": score,
                "rule": rule,
                "deviation_candidate": candidate,
                "absolute_volume_count_5m": len(recent),
                "absolute_floor_pass": volume_pass,
                "multi_window_confirmed": confirmed,
                "constructed_situation_id": situation,
                "situation_unique": unique,
                "budget_pass": budget_pass,
                "budget_allowed": math.floor(
                    self.rule_negative_seen * self.cfg.router.budget_pct / 100.0
                ),
                "budget_used": self.deviation_admitted,
            },
        )


def _observations_frame(run: PipelineRun) -> pd.DataFrame:
    return pd.DataFrame([asdict(observation) for observation in run.observations])


def self_amplification_series(run: PipelineRun, *, bucket_s: float = 10.0) -> pd.DataFrame:
    """Fast admission and urgent SLO compliance in shared intended-time buckets."""

    rows = _observations_frame(run)
    if rows.empty:
        return pd.DataFrame(
            columns=["bucket_s", "fast_admission_rate_hz", "urgent_slo_attainment"]
        )
    bucket_s = max(bucket_s, 1e-6)
    rows["bucket_s"] = np.floor(rows["intended_offset_s"] / bucket_s) * bucket_s
    output: list[dict[str, float]] = []
    for bucket, group in rows.groupby("bucket_s", sort=True):
        urgent = group[group["urgent"].astype(bool)]
        output.append(
            {
                "bucket_s": float(bucket),
                "fast_admission_rate_hz": float((group["lane"] == "fast").sum() / bucket_s),
                "urgent_slo_attainment": slo_attainment(
                    urgent["latency_s"].astype(float).tolist()
                ),
            }
        )
    return pd.DataFrame(output)


async def _run_in_process(
    schedule: Sequence[ScheduleEntry],
    cfg: EngineConfig,
    runner_cfg: RunnerConfig,
    *,
    lane_design: Literal["two-lane", "single"] | None,
    guard_overrides: GuardOverrides | None,
    training_events: Sequence[EvalEvent],
    router_policy: RouterPolicy | None,
) -> PipelineRun:
    if not schedule:
        raise ValueError("schedule cannot be empty")
    ordered = sorted(schedule, key=lambda entry: (entry.intended_send_ts, entry.event_id))
    event_entries = [entry for entry in ordered if entry.event is not None]
    if not event_entries:
        raise ValueError("schedule must contain at least one event")
    first_timestamp = ordered[0].intended_send_ts
    last_timestamp = event_entries[-1].intended_send_ts
    duration_s = max((last_timestamp - first_timestamp).total_seconds(), 0.0)
    policy = E6RouterPolicy(
        cfg,
        training_events=training_events,
        lane_design=lane_design,
        guard_overrides=guard_overrides,
        policy=router_policy,
    )
    design = lane_design or cfg.lane_design
    lane_names = ("single",) if design == "single" else ("fast", "batch")
    queues: dict[str, list[asyncio.Queue[_QueuedFold]]] = {
        lane: [asyncio.Queue() for _ in range(runner_cfg.partitions)]
        for lane in lane_names
    }
    notifications: dict[str, list[asyncio.Event]] = {
        lane: [asyncio.Event() for _ in range(runner_cfg.workers)] for lane in lane_names
    }
    stopped = False
    loop = asyncio.get_running_loop()
    loop_anchor = loop.time() + 0.005
    observations: list[EventObservation] = []
    lag = [0 for _ in range(runner_cfg.partitions)]
    peak_lag = [0 for _ in range(runner_cfg.partitions)]
    lag_rows: list[dict[str, float]] = []
    expected: dict[str, list[str]] = defaultdict(list)
    observed: dict[str, list[str]] = defaultdict(list)
    decisions: list[dict[str, Any]] = []
    disabled_until: dict[tuple[str, int], float] = defaultdict(float)
    kill_offset: float | None = None
    batch_windows: dict[str, list[_QueuedEvent]] = defaultdict(list)
    fold_sequence = 0

    def record_lag(offset_s: float) -> None:
        lag_rows.append(
            {
                "time_s": offset_s,
                "total_lag": float(sum(lag)),
                **{f"partition_{index}": float(value) for index, value in enumerate(lag)},
            }
        )

    async def consumer(lane: str, worker_index: int) -> None:
        owned = [
            partition
            for partition in range(runner_cfg.partitions)
            if partition % runner_cfg.workers == worker_index
        ]
        event = notifications[lane][worker_index]
        while True:
            item: _QueuedFold | None = None
            for partition in owned:
                try:
                    item = queues[lane][partition].get_nowait()
                    break
                except asyncio.QueueEmpty:
                    continue
            if item is None:
                if stopped and all(queues[lane][partition].empty() for partition in owned):
                    return
                event.clear()
                if any(not queues[lane][partition].empty() for partition in owned):
                    continue
                await event.wait()
                continue
            now = loop.time()
            if disabled_until[(lane, worker_index)] > now:
                await asyncio.sleep(disabled_until[(lane, worker_index)] - now)
            if item.window_close_loop_time > loop.time():
                await asyncio.sleep(item.window_close_loop_time - loop.time())
            if runner_cfg.admission_delay_s:
                await asyncio.sleep(runner_cfg.admission_delay_s)
            service_start = loop.time()
            agent_start = service_start
            if runner_cfg.service_time_s:
                await asyncio.sleep(runner_cfg.service_time_s)
            service_end = loop.time()
            service_elapsed = service_end - service_start
            batch = item.lane in {"batch", "single"}
            fold_latency = max(0.0, service_end - item.window_close_loop_time) if batch else None
            for queued in item.items:
                latency = max(0.0, agent_start - queued.intended_loop_time)
                staleness = max(0.0, service_end - queued.intended_loop_time) if batch else None
                observations.append(
                    EventObservation(
                        event_id=queued.entry.event_id,
                        entity=queued.entry.entity,
                        lane=item.lane,
                        urgent=queued.entry.urgent,
                        situation_id=queued.entry.situation_id,
                        cost_weight=queued.entry.cost_weight,
                        intended_offset_s=queued.intended_offset_s,
                        actual_send_offset_s=queued.actual_send_loop_time - loop_anchor,
                        send_skew_s=max(0.0, queued.actual_send_loop_time - queued.intended_loop_time),
                        agent_start_offset_s=agent_start - loop_anchor,
                        latency_s=latency,
                        service_start_offset_s=service_start - loop_anchor,
                        service_end_offset_s=service_end - loop_anchor,
                        service_elapsed_s=service_elapsed,
                        window_close_offset_s=item.window_close_loop_time - loop_anchor if batch else None,
                        commit_offset_s=service_end - loop_anchor if batch else None,
                        batch_fold_latency_s=fold_latency,
                        batch_staleness_s=staleness,
                        partition=queued.partition,
                    )
                )
                observed[item.lane].append(queued.entry.event_id)
                lag[queued.partition] -= 1
            record_lag(service_end - loop_anchor)
            queues[lane][item.partition].task_done()

    async def enqueue_fold(
        items: Sequence[_QueuedEvent], lane: str, window_close_loop_time: float
    ) -> None:
        nonlocal fold_sequence
        if not items:
            return
        partition = items[0].partition
        if any(item.partition != partition for item in items):
            raise AssertionError("a fold cannot cross subject-keyed partitions")
        fold = _QueuedFold(
            items=tuple(items),
            lane=lane,
            window_close_loop_time=window_close_loop_time,
            partition=partition,
            sequence=fold_sequence,
        )
        fold_sequence += 1
        await queues[lane][partition].put(fold)
        notifications[lane][partition % runner_cfg.workers].set()

    def close_deadline(items: Sequence[_QueuedEvent]) -> float:
        gap = cfg.window.gap_s * runner_cfg.window_time_scale
        age = cfg.window.max_age_s * runner_cfg.window_time_scale
        return min(items[-1].intended_offset_s + gap, items[0].intended_offset_s + age)

    async def close_due_windows(offset_s: float) -> None:
        due = [
            (close_deadline(items), entity)
            for entity, items in batch_windows.items()
            if close_deadline(items) <= offset_s
        ]
        for close_offset, entity in sorted(due):
            items = batch_windows.pop(entity)
            await enqueue_fold(items, items[0].lane, loop_anchor + close_offset)

    workers = [
        asyncio.create_task(consumer(lane, index))
        for lane in lane_names
        for index in range(runner_cfg.workers)
    ]
    for sequence, entry in enumerate(ordered):
        offset = (entry.intended_send_ts - first_timestamp).total_seconds()
        intended_loop = loop_anchor + offset
        if intended_loop > loop.time():
            await asyncio.sleep(intended_loop - loop.time())
        if entry.marker == "worker_kill":
            kill_offset = offset
            outage_lane = "single" if design == "single" else "batch"
            disabled_until[(outage_lane, runner_cfg.workers - 1)] = (
                loop.time() + runner_cfg.kill_downtime_s
            )
            record_lag(loop.time() - loop_anchor)
            continue
        assert entry.event is not None
        await close_due_windows(offset)
        actual_send = loop.time()
        lane, features = policy.route(entry.event, offset)
        partition = _partition(entry.entity, runner_cfg.partitions)
        item = _QueuedEvent(
            entry=entry,
            intended_loop_time=intended_loop,
            actual_send_loop_time=actual_send,
            intended_offset_s=offset,
            partition=partition,
            lane=lane,
            sequence=sequence,
        )
        expected[lane].append(entry.event_id)
        lag[partition] += 1
        peak_lag[partition] = max(peak_lag[partition], lag[partition])
        record_lag(actual_send - loop_anchor)
        decisions.append({"event_id": entry.event_id, "lane": lane, **features})
        if lane == "fast" or not runner_cfg.batch_windows:
            await enqueue_fold([item], lane, actual_send)
        else:
            window = batch_windows[entry.entity]
            window.append(item)
            if len(window) >= cfg.window.max_events:
                await enqueue_fold(window, lane, actual_send)
                del batch_windows[entry.entity]

    final_due = sorted(
        (close_deadline(items), entity) for entity, items in batch_windows.items()
    )
    for close_offset, entity in final_due:
        target = loop_anchor + close_offset
        if target > loop.time():
            await asyncio.sleep(target - loop.time())
        items = batch_windows.pop(entity)
        await enqueue_fold(items, items[0].lane, target)
    for lane in lane_names:
        for queue in queues[lane]:
            await queue.join()
    stopped = True
    for lane in lane_names:
        for notification in notifications[lane]:
            notification.set()
    await asyncio.gather(*workers)

    observations.sort(key=lambda row: (row.intended_offset_s, row.event_id))
    burst_start = (
        duration_s * runner_cfg.warmup_fraction
        if runner_cfg.burst_start_s is None
        else runner_cfg.burst_start_s
    )
    burst_end = duration_s if runner_cfg.burst_end_s is None else runner_cfg.burst_end_s
    measured = [
        row for row in observations if burst_start <= row.intended_offset_s <= burst_end
    ]
    fast_values = [row.latency_s for row in measured if row.lane in {"fast", "single"}]
    urgent_values = [row.latency_s for row in measured if row.urgent]
    fold_latencies = {
        (row.window_close_offset_s, row.commit_offset_s): row.batch_fold_latency_s
        for row in measured
        if row.batch_fold_latency_s is not None
    }
    batch_fold_values = list(fold_latencies.values())
    batch_staleness_values = [
        row.batch_staleness_s for row in measured if row.batch_staleness_s is not None
    ]
    send_skews = [row.send_skew_s for row in measured]
    service_invocations = {
        (row.service_start_offset_s, row.service_end_offset_s): row.service_elapsed_s
        for row in observations
    }
    service_values = list(service_invocations.values())
    histograms = {
        "fast": make_histogram(fast_values),
        "urgent": make_histogram(urgent_values),
        "batch-fold": make_histogram(cast(Iterable[float], batch_fold_values)),
        "batch-staleness": make_histogram(cast(Iterable[float], batch_staleness_values)),
        "send-skew": make_histogram(send_skews),
        "service": make_histogram(service_values),
    }
    lag_frame = pd.DataFrame(lag_rows).sort_values("time_s").reset_index(drop=True)
    during_burst = lag_frame[
        (lag_frame["time_s"] >= burst_start) & (lag_frame["time_s"] <= burst_end)
    ]
    lag_slope = linear_trend(
        during_burst["total_lag"].astype(float).tolist(),
        during_burst["time_s"].astype(float).tolist(),
    )
    latency_buckets: list[float] = []
    if measured:
        frame = pd.DataFrame([asdict(row) for row in measured])
        frame["quartile"] = pd.cut(
            frame["intended_offset_s"], bins=4, labels=False, duplicates="drop"
        )
        for _, group in frame.groupby("quartile", dropna=True):
            latency_buckets.append(
                histogram_percentile(make_histogram(group["latency_s"].tolist()), 99)
            )
    p99_trend = linear_trend(latency_buckets)
    delivery = duplicates_and_missed(expected, observed)
    service_spread = max(service_values, default=0.0) - min(service_values, default=0.0)
    producer_span = max(
        (max(row.actual_send_offset_s for row in observations)
         - min(row.actual_send_offset_s for row in observations)),
        1e-9,
    )
    metrics: dict[str, float] = {
        "fast_p50_s": histogram_percentile(histograms["fast"], 50),
        "fast_p99_s": histogram_percentile(histograms["fast"], 99),
        "fast_p999_s": histogram_percentile(histograms["fast"], 99.9),
        "urgent_slo_attainment": slo_attainment(urgent_values),
        "urgent_denominator": float(len(urgent_values)),
        "batch_p99_s": histogram_percentile(histograms["batch-fold"], 99),
        "batch_staleness_p99_s": histogram_percentile(histograms["batch-staleness"], 99),
        "send_skew_p99_ms": histogram_percentile(histograms["send-skew"], 99) * 1000,
        "generator_skew_check": float(
            histogram_percentile(histograms["send-skew"], 99) <= 0.001
        ),
        "lag_peak": float(max((row["total_lag"] for row in lag_rows), default=0.0)),
        "lag_slope": lag_slope,
        "p99_trend_s_per_bucket": p99_trend,
        "partition_lag_gini": partition_lag_gini(peak_lag),
        "duplicates": float(sum(delivery.duplicates_by_lane.values())),
        "missed": float(sum(delivery.missed_by_lane.values())),
        "service_time_configured_s": runner_cfg.service_time_s,
        "service_time_p50_s": histogram_percentile(histograms["service"], 50),
        "service_time_p99_s": histogram_percentile(histograms["service"], 99),
        "service_time_spread_s": service_spread,
        "service_time_constant": float(service_spread <= runner_cfg.service_time_tolerance_s),
        "observed_events": float(len(observations)),
        "producer_throughput_hz": float(len(observations) / producer_span),
        "throughput_hz": float(len(observations) / producer_span),
        "dlq": 0.0,
        "rule_admissions": float(policy.rule_admitted),
        "deviation_admissions": float(policy.deviation_admitted),
        "rule_negative_events": float(policy.rule_negative_seen),
        "deviation_admission_share": (
            policy.deviation_admitted / policy.rule_negative_seen
            if policy.rule_negative_seen
            else 0.0
        ),
        "budget_compliant": float(
            policy.deviation_admitted
            <= math.floor(policy.rule_negative_seen * cfg.router.budget_pct / 100.0)
        ),
        "broker_cpu_peak_pct": float("nan"),
        "broker_disk_util_peak_pct": float("nan"),
        "broker_disk_io_bytes": float("nan"),
        "broker_telemetry_present": 0.0,
        "worker_outage_logged": float(kill_offset is not None),
    }
    for lane in sorted(set(expected) | set(observed)):
        metrics[f"duplicates_{lane}"] = float(delivery.duplicates_by_lane.get(lane, 0))
        metrics[f"missed_{lane}"] = float(delivery.missed_by_lane.get(lane, 0))
    pre_burst = lag_frame[lag_frame["time_s"] < burst_start]
    baseline_lag = (
        float(pre_burst["total_lag"].median()) if not pre_burst.empty else 0.0
    )
    metrics["baseline_lag"] = baseline_lag
    metrics["drain_time_s"] = drain_time(
        lag_frame["time_s"].tolist(),
        lag_frame["total_lag"].tolist(),
        burst_end_s=burst_end,
        baseline_lag=baseline_lag,
    )
    if kill_offset is not None:
        before = lag_frame[lag_frame["time_s"] < kill_offset]
        pre_kill = float(before.iloc[-1]["total_lag"]) if not before.empty else 0.0
        metrics["recovery_time_s"] = recovery_time(
            lag_frame["time_s"].tolist(),
            lag_frame["total_lag"].tolist(),
            kill_s=kill_offset,
            pre_kill_lag=pre_kill,
        )
    gold = {
        entry.event_id: {
            "urgent": entry.urgent,
            "situation_id": entry.situation_id,
            "cost_weight": entry.cost_weight,
            "entity": entry.entity,
        }
        for entry in event_entries
    }
    return PipelineRun(
        observations=observations,
        histograms=histograms,
        lag_samples=lag_frame,
        metrics=metrics,
        expected_by_lane=dict(expected),
        observed_by_lane=dict(observed),
        route_decisions=pd.DataFrame(decisions),
        kill_offset_s=kill_offset,
        gold_by_event=gold,
    )


def external_records_to_run(
    schedule: Sequence[ScheduleEntry],
    records: Sequence[Mapping[str, Any]],
    *,
    send_skews_s: Sequence[float] = (),
    telemetry: Sequence[Mapping[str, Any]] = (),
) -> PipelineRun:
    """Summarise the documented Kafka output schema; useful for fixture tests."""

    event_entries = {entry.event_id: entry for entry in schedule if entry.event is not None}
    if not event_entries:
        raise ValueError("schedule must contain events")
    first = min(entry.intended_send_ts for entry in event_entries.values())
    observations: list[EventObservation] = []
    expected: dict[str, list[str]] = defaultdict(list)
    observed: dict[str, list[str]] = defaultdict(list)
    lag_rows: list[dict[str, float]] = []
    for raw in records:
        event_id = str(raw["event_id"])
        if event_id not in event_entries:
            continue
        entry = event_entries[event_id]
        lane = str(raw["lane"])
        expected[lane].append(event_id)
        observed[lane].append(event_id)
        intended_offset = (entry.intended_send_ts - first).total_seconds()
        agent_start = datetime.fromisoformat(str(raw["agent_start_ts"]))
        service_start = datetime.fromisoformat(str(raw.get("service_start_ts", raw["agent_start_ts"])))
        service_end = datetime.fromisoformat(str(raw.get("service_end_ts", raw["agent_start_ts"])))
        close_raw = raw.get("window_close_ts")
        commit_raw = raw.get("commit_ts")
        close = datetime.fromisoformat(str(close_raw)) if close_raw else None
        commit = datetime.fromisoformat(str(commit_raw)) if commit_raw else None
        observations.append(
            EventObservation(
                event_id=event_id,
                entity=entry.entity,
                lane=lane,
                urgent=entry.urgent,
                situation_id=entry.situation_id,
                cost_weight=entry.cost_weight,
                intended_offset_s=intended_offset,
                actual_send_offset_s=float(raw.get("actual_send_offset_s", intended_offset)),
                send_skew_s=float(raw.get("send_skew_s", 0.0)),
                agent_start_offset_s=(agent_start - first).total_seconds(),
                latency_s=max(0.0, (agent_start - entry.intended_send_ts).total_seconds()),
                service_start_offset_s=(service_start - first).total_seconds(),
                service_end_offset_s=(service_end - first).total_seconds(),
                service_elapsed_s=max(0.0, (service_end - service_start).total_seconds()),
                window_close_offset_s=(close - first).total_seconds() if close else None,
                commit_offset_s=(commit - first).total_seconds() if commit else None,
                batch_fold_latency_s=(commit - close).total_seconds() if close and commit else None,
                batch_staleness_s=(commit - entry.intended_send_ts).total_seconds() if commit else None,
                partition=int(raw.get("partition", 0)),
            )
        )
        if "partition_lag" in raw:
            lag_rows.append(
                {
                    "time_s": (agent_start - first).total_seconds(),
                    "total_lag": float(raw["partition_lag"]),
                }
            )
    known = {event_id for values in expected.values() for event_id in values}
    missing = set(event_entries) - known
    for event_id in missing:
        expected["unknown"].append(event_id)
    delivery = duplicates_and_missed(expected, observed)
    urgent_ids = {entry.event_id for entry in schedule if entry.event is not None and entry.urgent}
    by_id: dict[str, EventObservation] = {}
    for row in observations:
        by_id.setdefault(row.event_id, row)
    urgent_values = [by_id[event_id].latency_s if event_id in by_id else float("inf") for event_id in urgent_ids]
    fast_values = [row.latency_s for row in observations if row.lane in {"fast", "single"}]
    batch_values = [row.batch_fold_latency_s for row in observations if row.batch_fold_latency_s is not None]
    staleness = [row.batch_staleness_s for row in observations if row.batch_staleness_s is not None]
    services = [row.service_elapsed_s for row in observations]
    histograms = {
        "fast": make_histogram(fast_values),
        "urgent": make_histogram(value for value in urgent_values if math.isfinite(value)),
        "batch-fold": make_histogram(cast(Iterable[float], batch_values)),
        "batch-staleness": make_histogram(cast(Iterable[float], staleness)),
        "send-skew": make_histogram(send_skews_s),
        "service": make_histogram(services),
    }
    cpu = [float(row["cpu_pct"]) for row in telemetry if "cpu_pct" in row]
    disk = [float(row["disk_util_pct"]) for row in telemetry if "disk_util_pct" in row]
    disk_io = [float(row.get("disk_io_bytes", 0.0)) for row in telemetry]
    service_spread = max(services, default=0.0) - min(services, default=0.0)
    metrics = {
        "fast_p50_s": histogram_percentile(histograms["fast"], 50),
        "fast_p99_s": histogram_percentile(histograms["fast"], 99),
        "fast_p999_s": histogram_percentile(histograms["fast"], 99.9),
        "urgent_slo_attainment": slo_attainment(urgent_values),
        "urgent_denominator": float(len(urgent_values)),
        "batch_p99_s": histogram_percentile(histograms["batch-fold"], 99),
        "batch_staleness_p99_s": histogram_percentile(histograms["batch-staleness"], 99),
        "send_skew_p99_ms": histogram_percentile(histograms["send-skew"], 99) * 1000,
        "generator_skew_check": float(histogram_percentile(histograms["send-skew"], 99) <= 0.001),
        "lag_peak": max((row["total_lag"] for row in lag_rows), default=0.0),
        "lag_slope": linear_trend(
            [row["total_lag"] for row in lag_rows], [row["time_s"] for row in lag_rows]
        ),
        "p99_trend_s_per_bucket": 0.0,
        "partition_lag_gini": 0.0,
        "duplicates": float(sum(delivery.duplicates_by_lane.values())),
        "missed": float(sum(delivery.missed_by_lane.values())),
        "service_time_configured_s": float("nan"),
        "service_time_p50_s": histogram_percentile(histograms["service"], 50),
        "service_time_p99_s": histogram_percentile(histograms["service"], 99),
        "service_time_spread_s": service_spread,
        "service_time_constant": float(service_spread <= 0.002),
        "observed_events": float(len(observations)),
        "throughput_hz": float("nan"),
        "producer_throughput_hz": float("nan"),
        "dlq": float(sum(float(row.get("dlq", 0.0)) for row in records)),
        "budget_compliant": float("nan"),
        "broker_cpu_peak_pct": max(cpu, default=float("nan")),
        "broker_disk_util_peak_pct": max(disk, default=float("nan")),
        "broker_disk_io_bytes": sum(disk_io),
        "broker_telemetry_present": float(bool(cpu and disk and telemetry)),
        "worker_outage_logged": float(any(bool(row.get("worker_restart")) for row in records)),
    }
    for lane in sorted(set(expected) | set(observed)):
        metrics[f"duplicates_{lane}"] = float(delivery.duplicates_by_lane.get(lane, 0))
        metrics[f"missed_{lane}"] = float(delivery.missed_by_lane.get(lane, 0))
    return PipelineRun(
        observations=observations,
        histograms=histograms,
        lag_samples=pd.DataFrame(lag_rows),
        metrics=metrics,
        expected_by_lane=dict(expected),
        observed_by_lane=dict(observed),
        gold_by_event={
            event_id: {
                "urgent": entry.urgent,
                "situation_id": entry.situation_id,
                "cost_weight": entry.cost_weight,
                "entity": entry.entity,
            }
            for event_id, entry in event_entries.items()
        },
    )


async def _run_kafka(
    schedule: Sequence[ScheduleEntry],
    cfg: EngineConfig,
    runner_cfg: RunnerConfig,
    *,
    lane_design: Literal["two-lane", "single"] | None,
    guard_overrides: GuardOverrides | None,
) -> PipelineRun:
    """Run the documented real transport; dependencies load only on selection."""

    if runner_cfg.kafka_bootstrap_servers is None or runner_cfg.kafka_output_topic is None:
        raise ValueError("Kafka E6 requires bootstrap servers and an output topic")
    if runner_cfg.load_generator_host in {"", "local", "localhost", "127.0.0.1"}:
        raise ValueError("research Kafka mode requires a separate load-generator host")
    try:
        from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    except ImportError as exc:  # pragma: no cover - optional real deployment
        raise RuntimeError("install aiokafka to use the E6 Kafka transport") from exc
    producer = AIOKafkaProducer(bootstrap_servers=runner_cfg.kafka_bootstrap_servers)
    consumer = AIOKafkaConsumer(
        runner_cfg.kafka_output_topic,
        bootstrap_servers=runner_cfg.kafka_bootstrap_servers,
        auto_offset_reset="latest",
        enable_auto_commit=False,
    )
    await producer.start()
    await consumer.start()
    records: list[Mapping[str, Any]] = []
    send_skews: list[float] = []
    try:
        first = min(entry.intended_send_ts for entry in schedule)
        loop = asyncio.get_running_loop()
        anchor = loop.time() + 0.02
        expected_ids = {entry.event_id for entry in schedule if entry.event is not None}
        send_start = loop.time()
        overrides = guard_overrides or GuardOverrides()
        headers = [
            ("e6-lane-design", (lane_design or cfg.lane_design).encode()),
            ("e6-partitions", str(runner_cfg.partitions).encode()),
            ("e6-workers", str(runner_cfg.workers).encode()),
            ("e6-budget-pct", str(cfg.router.budget_pct).encode()),
            (
                "e6-absolute-floor",
                str(
                    max(2.0, cfg.router.guards.absolute_floor)
                    if overrides.absolute_floor is None
                    else overrides.absolute_floor
                ).encode(),
            ),
            (
                "e6-multi-window",
                str(True if overrides.multi_window is None else overrides.multi_window).encode(),
            ),
            (
                "e6-situation-dedup",
                str(
                    True
                    if overrides.situation_dedup is None
                    else overrides.situation_dedup
                ).encode(),
            ),
        ]
        for entry in sorted(schedule, key=lambda item: (item.intended_send_ts, item.event_id)):
            if entry.event is None:
                continue
            intended_loop = anchor + (entry.intended_send_ts - first).total_seconds()
            if intended_loop > loop.time():
                await asyncio.sleep(intended_loop - loop.time())
            actual = loop.time()
            await producer.send_and_wait(
                runner_cfg.kafka_input_topic,
                entry.event.model_dump_json().encode(),
                key=entry.event.partition_key(),
                headers=headers,
            )
            send_skews.append(max(0.0, actual - intended_loop))
        send_end = loop.time()
        unique_seen: set[str] = set()
        deadline = loop.time() + runner_cfg.kafka_timeout_s
        while unique_seen != expected_ids and loop.time() < deadline:
            batch = await consumer.getmany(timeout_ms=100, max_records=len(expected_ids))
            for messages in batch.values():
                for message in messages:
                    if message.value is None:
                        continue
                    raw = json.loads(message.value)
                    event_id = str(raw.get("event_id", ""))
                    if event_id in expected_ids:
                        records.append(raw)
                        unique_seen.add(event_id)
        telemetry: list[Mapping[str, Any]] = []
        if runner_cfg.kafka_telemetry_path is not None:
            with runner_cfg.kafka_telemetry_path.open(encoding="utf-8") as source:
                telemetry = [json.loads(line) for line in source if line.strip()]
        result = external_records_to_run(
            schedule, records, send_skews_s=send_skews, telemetry=telemetry
        )
        span = max(send_end - send_start, 1e-9)
        result.metrics["producer_throughput_hz"] = len(expected_ids) / span
        result.metrics["throughput_hz"] = result.metrics["producer_throughput_hz"]
        return result
    finally:
        await consumer.stop()
        await producer.stop()


async def run_schedule(
    schedule: Sequence[ScheduleEntry],
    cfg: EngineConfig,
    runner_cfg: RunnerConfig | None = None,
    *,
    lane_design: Literal["two-lane", "single"] | None = None,
    guard_overrides: GuardOverrides | None = None,
    training_events: Sequence[EvalEvent] = (),
    router_policy: RouterPolicy | None = None,
) -> PipelineRun:
    """Send one frozen schedule open-loop through the selected transport."""

    runtime = runner_cfg or RunnerConfig()
    if runtime.transport == "kafka":
        return await _run_kafka(
            schedule,
            cfg,
            runtime,
            lane_design=lane_design,
            guard_overrides=guard_overrides,
        )
    return await _run_in_process(
        schedule,
        cfg,
        runtime,
        lane_design=lane_design,
        guard_overrides=guard_overrides,
        training_events=training_events,
        router_policy=router_policy,
    )


async def calibration_run(
    cfg: EngineConfig, *, samples: int = 24, admission_delay_s: float = 0.010
) -> PipelineRun:
    """Run the fixed 10 ms pass-through clock-chain calibration."""

    from harnext_eval.corpus.synthetic import generate_synthetic_events

    replay_event = generate_synthetic_events(event_count=2, days=1, entity_count=1)[0]
    start = datetime(2034, 1, 1, tzinfo=UTC)
    schedule = [
        ScheduleEntry(
            intended_send_ts=start + timedelta(milliseconds=index * 20),
            entity=replay_event.subject,
            event=replay_event.model_copy(
                update={
                    "id": f"calibration-{index}",
                    "time": start + timedelta(milliseconds=index * 20),
                    "intended_send_ts": start + timedelta(milliseconds=index * 20),
                }
            ),
            shape="calibration",
        )
        for index in range(samples)
    ]
    return await run_schedule(
        schedule,
        cfg,
        RunnerConfig(
            partitions=1,
            workers=1,
            service_time_s=0.0,
            admission_delay_s=admission_delay_s,
            warmup_fraction=0.25,
            batch_windows=False,
        ),
        lane_design="single",
    )


def calibration_valid(
    run: PipelineRun,
    *,
    median_target_s: float = 0.010,
    median_tolerance_s: float = 0.005,
    p99_trend_tolerance: float = 0.003,
) -> bool:
    """Require both the 10 ms median and a flat repeated-window p99."""

    return (
        abs(run.metrics["fast_p50_s"] - median_target_s) <= median_tolerance_s
        and abs(run.metrics["p99_trend_s_per_bucket"]) <= p99_trend_tolerance
    )


def find_knee(
    results: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    p99_trend_tolerance: float = 0.02,
    lag_slope_threshold: float = 0.05,
) -> float | None:
    """Highest steady rate with near-zero p99 trend and bounded lag slope."""

    frame = results.copy() if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    required = {"rate_hz", "fast_p99_s", "p99_trend_s_per_bucket", "lag_slope"}
    if not required.issubset(frame.columns):
        raise ValueError(f"results must contain {sorted(required)}")
    aggregate = frame.groupby("rate_hz", as_index=False).median(numeric_only=True)
    stable = aggregate[
        (aggregate["p99_trend_s_per_bucket"].abs() <= p99_trend_tolerance)
        & (aggregate["lag_slope"] <= lag_slope_threshold)
    ]
    if stable.empty:
        return None
    return float(stable["rate_hz"].max())


async def sweep_steady(
    fit: WorkloadFit,
    cfg: EngineConfig,
    *,
    rates_hz: Sequence[float],
    duration_s: float,
    runner_cfg: RunnerConfig | None = None,
    repetitions: int = 3,
    lane_design: Literal["two-lane", "single"] | None = None,
    out_dir: Path | None = None,
    seed: int = 1,
    p99_trend_tolerance: float = 0.02,
    lag_slope_threshold: float = 0.05,
) -> tuple[pd.DataFrame, float | None, list[Path]]:
    """Run repeated steady windows and return the preregistered knee."""

    if repetitions < 3:
        raise ValueError("E6 requires at least three steady repetitions")
    runtime = runner_cfg or RunnerConfig()
    rows: list[dict[str, Any]] = []
    artifacts: list[Path] = []
    fixed_start = datetime(2033, 1, 1, tzinfo=UTC)
    for rate_index, rate in enumerate(rates_hz):
        if rate <= 0:
            raise ValueError("sweep rates must be positive")
        for repetition in range(repetitions):
            schedule = generate_steady_schedule(
                fit,
                duration_s=duration_s,
                rate_hz=rate,
                seed=seed + repetition,
                start=fixed_start + timedelta(days=rate_index, hours=repetition),
            )
            result = await run_schedule(
                schedule,
                cfg,
                runtime,
                lane_design=lane_design,
                training_events=fit.replay_events,
            )
            rows.append(
                {
                    "rate_hz": rate,
                    "repetition": repetition,
                    "lane_design": lane_design or cfg.lane_design,
                    "partitions": runtime.partitions,
                    "workers": runtime.workers,
                    "schedule_id": schedule_fingerprint(schedule),
                    **result.metrics,
                }
            )
            if out_dir is not None:
                artifacts.extend(
                    result.write_histograms(
                        out_dir,
                        f"steady-{lane_design or cfg.lane_design}-{rate:g}-r{repetition}",
                        names=("fast",),
                    )
                )
    frame = pd.DataFrame(rows)
    knee = find_knee(
        frame,
        p99_trend_tolerance=p99_trend_tolerance,
        lag_slope_threshold=lag_slope_threshold,
    )
    return frame, knee, artifacts


@dataclass(frozen=True)
class BenchmarkConfig:
    """Explicit smoke/research matrix; smoke is structurally complete but tiny."""

    profile: Literal["smoke", "research"] = "smoke"
    reference_rate_hz: float = 80.0
    steady_duration_s: float = 0.04
    burst_duration_s: float = 0.05
    burst_window_s: float = 0.03
    repetitions: int = 3
    service_time_s: float = 0.001
    topologies: tuple[tuple[int, int], ...] = ((8, 1), (8, 4), (32, 1), (32, 4))
    shapes: tuple[str, ...] = ("poisson", "anomalous_burst")
    burst_loads: tuple[float, ...] = (1.5,)
    entity_cardinalities: tuple[int, ...] = (8,)
    burstiness_targets: tuple[float, ...] = (0.25, 0.5, 0.75)
    bootstrap_resamples: int = 100
    p99_trend_tolerance: float = 0.05
    lag_slope_threshold: float = 5.0
    runner: RunnerConfig = field(default_factory=RunnerConfig)

    @classmethod
    def smoke(cls) -> BenchmarkConfig:
        return cls()

    @classmethod
    def research(
        cls,
        fit: WorkloadFit,
        *,
        kafka_bootstrap_servers: str,
        kafka_output_topic: str,
        load_generator_host: str,
        kafka_telemetry_path: Path,
    ) -> BenchmarkConfig:
        """Full registered matrix on Kafka from a separately named load host."""

        return cls(
            profile="research",
            reference_rate_hz=fit.mean_rate_hz,
            steady_duration_s=1_200.0,
            burst_duration_s=1_200.0,
            burst_window_s=600.0,
            shapes=(
                "steady",
                "poisson",
                "pareto_on_off",
                "benign_flash",
                "anomalous_burst",
                "zipf_hot",
                "worker_kill",
            ),
            burst_loads=(1.0, 1.5),
            entity_cardinalities=(8, 32),
            bootstrap_resamples=10_000,
            p99_trend_tolerance=0.02,
            lag_slope_threshold=0.05,
            runner=RunnerConfig(
                transport="kafka",
                service_time_s=0.002,
                window_time_scale=1.0,
                load_generator_host=load_generator_host,
                kafka_bootstrap_servers=kafka_bootstrap_servers,
                kafka_output_topic=kafka_output_topic,
                kafka_telemetry_path=kafka_telemetry_path,
            ),
        )


def _shape_variants(settings: BenchmarkConfig, shape_name: str) -> tuple[float | None, ...]:
    if shape_name == "poisson":
        return (0.0,)
    if shape_name == "pareto_on_off":
        return tuple(settings.burstiness_targets)
    return (None,)


def _guard_conditions(cfg: EngineConfig) -> dict[str, GuardOverrides]:
    full = GuardOverrides(
        absolute_floor=max(2.0, cfg.router.guards.absolute_floor),
        multi_window=True,
        situation_dedup=True,
    )
    conditions = {
        "full": full,
        "minus_absolute_floor": replace(full, absolute_floor=0.0),
        "minus_multi_window": replace(full, multi_window=False),
        "minus_situation_dedup": replace(full, situation_dedup=False),
    }
    full_values = asdict(full)
    for name, value in conditions.items():
        if name == "full":
            continue
        changed = [key for key in full_values if getattr(value, key) != getattr(full, key)]
        if changed != [name.removeprefix("minus_")]:
            raise AssertionError(f"{name} must change exactly its named guard")
    return conditions


def _paired_outcomes(
    plan: SchedulePlan,
    two: PipelineRun,
    single: PipelineRun,
    *,
    metadata: Mapping[str, Any],
) -> pd.DataFrame:
    two_by_id = {row.event_id: row for row in two.observations}
    single_by_id = {row.event_id: row for row in single.observations}
    rows: list[dict[str, Any]] = []
    for entry in plan.entries:
        if (
            entry.event is None
            or not entry.urgent
            or not plan.burst_start_s <= (entry.intended_send_ts - plan.entries[0].intended_send_ts).total_seconds() <= plan.burst_end_s
        ):
            continue
        two_latency = two_by_id.get(entry.event_id)
        single_latency = single_by_id.get(entry.event_id)
        rows.append(
            {
                **metadata,
                "event_id": entry.event_id,
                "entity": entry.entity,
                "situation_id": entry.situation_id,
                "two_latency_s": two_latency.latency_s if two_latency else float("inf"),
                "single_latency_s": single_latency.latency_s if single_latency else float("inf"),
                "two_slo": float(two_latency is not None and two_latency.latency_s <= URGENT_SLO_S),
                "single_slo": float(single_latency is not None and single_latency.latency_s <= URGENT_SLO_S),
            }
        )
    if not rows:
        raise ValueError("paired E6 cell has no urgent gold in its burst window")
    return pd.DataFrame(rows)


def _gap_by_b(paired: pd.DataFrame, *, resamples: int, seed: int) -> pd.DataFrame:
    paired = paired.copy()
    paired["b_level"] = paired["target_b"].where(
        paired["target_b"].notna(), paired["realised_b"].round(2)
    )
    rows: list[dict[str, Any]] = []
    group_columns = ["shape", "b_level", "load", "partitions", "workers"]
    for keys, group in paired.groupby(group_columns, dropna=False, sort=True):
        key_values = dict(zip(group_columns, keys, strict=True))
        if group["entity"].nunique() >= 2:
            interval = paired_difference_bca(
                group["two_slo"].to_numpy(),
                group["single_slo"].to_numpy(),
                group["entity"].to_numpy(),
                n_resamples=resamples,
                random_state=seed,
            )
            ci_low, ci_high = interval.ci_low, interval.ci_high
        else:
            ci_low = ci_high = float("nan")
        rows.append(
            {
                **key_values,
                "target_b": float(group["target_b"].median())
                if group["target_b"].notna().any()
                else float("nan"),
                "realised_b": float(group["realised_b"].mean()),
                "two_slo_attainment": float(group["two_slo"].mean()),
                "single_slo_attainment": float(group["single_slo"].mean()),
                "gap": float((group["two_slo"] - group["single_slo"]).mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_events": len(group),
                "n_entities": group["entity"].nunique(),
                "bootstrap_resamples": resamples,
            }
        )
    return pd.DataFrame(rows)


async def run_benchmark(
    fit: WorkloadFit,
    cfg: EngineConfig,
    out_dir: Path,
    *,
    seed: int,
    benchmark: BenchmarkConfig | None = None,
    situations: Sequence[SituationSpec] | None = None,
) -> ExperimentResult:
    """Execute a paired E6 matrix and write exact named outputs."""

    settings = benchmark or BenchmarkConfig.smoke()
    await asyncio.to_thread(out_dir.mkdir, parents=True, exist_ok=True)
    hdr_dir = out_dir / "latency_hdr"
    await asyncio.to_thread(hdr_dir.mkdir, parents=True, exist_ok=True)
    artifacts: list[Path] = []
    calibration = await calibration_run(cfg)
    if not calibration_valid(calibration):
        raise RuntimeError(
            "E6 clock-chain calibration failed: both 10 ms p50 and flat p99 are required"
        )

    sweep_rates = [
        settings.reference_rate_hz * multiplier for multiplier in (0.25, 0.5, 1, 2, 4)
    ]
    steady_runtime = replace(
        settings.runner,
        partitions=8,
        workers=1,
        service_time_s=settings.service_time_s,
        burst_start_s=None,
        burst_end_s=None,
    )
    steady, knee, sweep_artifacts = await sweep_steady(
        fit,
        cfg,
        rates_hz=sweep_rates,
        duration_s=settings.steady_duration_s,
        runner_cfg=steady_runtime,
        repetitions=settings.repetitions,
        lane_design="single",
        out_dir=hdr_dir,
        seed=seed,
        p99_trend_tolerance=settings.p99_trend_tolerance,
        lag_slope_threshold=settings.lag_slope_threshold,
    )
    if knee is None:
        raise RuntimeError("E6 knee is undefined: no steady rate met both trend criteria")
    artifacts.extend(sweep_artifacts)
    knee_table = (
        steady.groupby(["lane_design", "rate_hz"], as_index=False)
        .median(numeric_only=True)
        .assign(
            knee_hz=knee,
            p99_trend_tolerance=settings.p99_trend_tolerance,
            lag_slope_threshold=settings.lag_slope_threshold,
        )
    )
    knee_path = out_dir / "knee.csv"
    knee_table.to_csv(knee_path, index=False)
    artifacts.append(knee_path)

    catalogue = tuple(situations or situations_from_meta(fit, None))
    r5_template = GuardedHBOSPolicy(absolute_floor=0.0, multi_window=False)
    r5_template.fit(list(fit.replay_events))
    plans: list[tuple[SchedulePlan, dict[str, Any]]] = []
    cell_index = 0
    fixed_start = datetime(2036, 1, 1, tzinfo=UTC)
    for load in settings.burst_loads:
        for shape_name in settings.shapes:
            shape = "benign_flash" if shape_name == "worker_kill" else shape_name
            for target_b in _shape_variants(settings, shape_name):
                for cardinality in settings.entity_cardinalities:
                    for repetition in range(settings.repetitions):
                        plan = build_schedule(
                            fit,
                            shape=cast(Any, shape),
                            duration_s=settings.burst_duration_s,
                            burst_duration_s=settings.burst_window_s,
                            burst_start_s=(
                                settings.burst_duration_s - settings.burst_window_s
                            )
                            / 2,
                            rate_hz=knee * load,
                            seed=seed + repetition,
                            start=fixed_start + timedelta(days=cell_index),
                            worker_kill=shape_name == "worker_kill",
                            target_b=target_b,
                            situations=catalogue,
                            entity_cardinality=cardinality,
                            require_convergence=settings.profile == "research",
                        )
                        plans.append(
                            (
                                plan,
                                {
                                    "shape": shape_name,
                                    "load": load,
                                    "rate_hz": knee * load,
                                    "entity_cardinality": cardinality,
                                    "repetition": repetition,
                                    "target_b": target_b,
                                    "realised_b": plan.realised_b,
                                    "schedule_id": plan.schedule_id,
                                    "calibration_tail_index": plan.tail_index,
                                    "calibration_iterations": plan.calibration_iterations,
                                    "calibration_converged": plan.calibration_converged,
                                },
                            )
                        )
                        cell_index += 1

    burst_rows: list[dict[str, Any]] = []
    paired_frames: list[pd.DataFrame] = []
    amplification_frames: list[pd.DataFrame] = []
    generator_checks: list[float] = []
    plan_for_guards: SchedulePlan | None = None
    for plan, metadata in plans:
        if metadata["shape"] == "anomalous_burst" and metadata["load"] == 1.5:
            plan_for_guards = plan_for_guards or plan
        for partitions, workers in settings.topologies:
            results_by_lane: dict[str, PipelineRun] = {}
            for lane_design in ("two-lane", "single"):
                runtime = replace(
                    settings.runner,
                    partitions=partitions,
                    workers=workers,
                    service_time_s=settings.service_time_s,
                    burst_start_s=plan.burst_start_s,
                    burst_end_s=plan.burst_end_s,
                )
                result = await run_schedule(
                    plan.entries,
                    cfg,
                    runtime,
                    lane_design=cast(Any, lane_design),
                    training_events=fit.replay_events,
                    router_policy=r5_template if lane_design == "two-lane" else None,
                )
                if schedule_fingerprint(plan.entries) != plan.schedule_id:
                    raise AssertionError("a paired schedule was mutated between lane arms")
                results_by_lane[lane_design] = result
                generator_checks.append(result.metrics["generator_skew_check"])
                observation_rows = _observations_frame(result)
                fairness = cross_entity_fairness(
                    observation_rows, fit.entities[:3]
                ) if not observation_rows.empty else float("nan")
                burst_rows.append(
                    {
                        **metadata,
                        "lane_design": lane_design,
                        "partitions": partitions,
                        "workers": workers,
                        "cross_entity_fairness": fairness,
                        **result.metrics,
                    }
                )
                prefix = (
                    f"burst-{metadata['shape']}-{metadata['load']}x-"
                    f"p{partitions}-w{workers}-r{metadata['repetition']}-{lane_design}"
                )
                artifacts.extend(
                    result.write_histograms(
                        hdr_dir,
                        prefix,
                        names=("fast", "urgent", "batch-fold", "batch-staleness"),
                        full_distribution=settings.profile == "research",
                    )
                )
                if metadata["shape"] == "anomalous_burst" and metadata["load"] == 1.5:
                    series = self_amplification_series(
                        result, bucket_s=max(settings.burst_window_s / 6, 1e-6)
                    )
                    for key, value in {
                        "lane_design": lane_design,
                        "partitions": partitions,
                        "workers": workers,
                        "repetition": metadata["repetition"],
                    }.items():
                        series.insert(0, key, value)
                    amplification_frames.append(series)
            paired_frames.append(
                _paired_outcomes(
                    plan,
                    results_by_lane["two-lane"],
                    results_by_lane["single"],
                    metadata={
                        **metadata,
                        "partitions": partitions,
                        "workers": workers,
                    },
                )
            )

    burst = pd.DataFrame(burst_rows)
    burst_path = out_dir / "burst_slo.csv"
    burst.to_csv(burst_path, index=False)
    artifacts.append(burst_path)
    paired = pd.concat(paired_frames, ignore_index=True)
    paired_path = out_dir / "paired_urgent_events.csv"
    paired.to_csv(paired_path, index=False)
    artifacts.append(paired_path)
    gap_by_b = _gap_by_b(
        paired, resamples=settings.bootstrap_resamples, seed=seed
    )
    gap_path = out_dir / "gap_by_b.csv"
    gap_by_b.to_csv(gap_path, index=False)
    artifacts.append(gap_path)

    amplification = (
        pd.concat(amplification_frames, ignore_index=True)
        if amplification_frames
        else pd.DataFrame(
            columns=["bucket_s", "fast_admission_rate_hz", "urgent_slo_attainment"]
        )
    )
    amplification_path = out_dir / "self_amplification.csv"
    amplification.to_csv(amplification_path, index=False)
    artifacts.append(amplification_path)

    topology_aggregate = (
        burst.groupby(
            [
                "lane_design",
                "shape",
                "load",
                "entity_cardinality",
                "target_b",
                "partitions",
                "workers",
            ],
            dropna=False,
            as_index=False,
        )
        .median(numeric_only=True)
    )
    demand = demand_curve(
        topology_aggregate,
        lag_slope_threshold=settings.lag_slope_threshold,
    )
    demand_path = out_dir / "demand.csv"
    demand.to_csv(demand_path, index=False)
    artifacts.append(demand_path)

    if plan_for_guards is None:
        plan_for_guards = plans[0][0]
    guard_rows: list[dict[str, Any]] = []
    full_decisions: pd.DataFrame | None = None
    for name, overrides in _guard_conditions(cfg).items():
        result = await run_schedule(
            plan_for_guards.entries,
            cfg,
            replace(
                settings.runner,
                partitions=settings.topologies[0][0],
                workers=settings.topologies[0][1],
                service_time_s=settings.service_time_s,
                burst_start_s=plan_for_guards.burst_start_s,
                burst_end_s=plan_for_guards.burst_end_s,
            ),
            lane_design="two-lane",
            guard_overrides=overrides,
            training_events=fit.replay_events,
            router_policy=r5_template,
        )
        if name == "full":
            full_decisions = result.route_decisions[["event_id", "lane"]].rename(
                columns={"lane": "full_lane"}
            )
            changed = 0
        else:
            assert full_decisions is not None
            joined = full_decisions.merge(
                result.route_decisions[["event_id", "lane"]], on="event_id", how="outer"
            )
            changed = int((joined["full_lane"] != joined["lane"]).sum())
        guard_rows.append(
            {
                "guard_condition": name,
                "schedule_id": plan_for_guards.schedule_id,
                "absolute_floor": overrides.absolute_floor,
                "multi_window": overrides.multi_window,
                "situation_dedup": overrides.situation_dedup,
                "changed_events_vs_full": changed,
                **result.metrics,
            }
        )
    guards = pd.DataFrame(guard_rows)
    guards_path = out_dir / "guards.csv"
    guards.to_csv(guards_path, index=False)
    artifacts.append(guards_path)

    popularity = dict(zip(fit.entities, fit.popularity, strict=True))
    workload = pd.DataFrame(
        [
            {
                "entity": entity,
                "count": item.count,
                "rate_hz": item.rate_hz,
                "mean_inter_arrival_s": item.mean_s,
                "std_inter_arrival_s": item.std_s,
                "p50_inter_arrival_s": item.quantiles_s[0],
                "p90_inter_arrival_s": item.quantiles_s[1],
                "p99_inter_arrival_s": item.quantiles_s[2],
                "burstiness_b": item.burstiness_b,
                "memory_m": item.memory_m,
                "popularity": popularity[entity],
            }
            for entity, item in fit.entity_fits.items()
        ]
    )
    workload_path = out_dir / "workload_fit.csv"
    workload.to_csv(workload_path, index=False)
    artifacts.append(workload_path)
    parameters = {
        "schema_version": 1,
        "profile": settings.profile,
        "mean_rate_hz": fit.mean_rate_hz,
        "type_mix": fit.type_mix,
        "zipf_exponent": fit.zipf_exponent,
        "fitted_burstiness_b": fit.burstiness_b,
        "fitted_memory_m": fit.memory_m,
        "steady_duration_s": settings.steady_duration_s,
        "burst_duration_s": settings.burst_duration_s,
        "burst_window_s": settings.burst_window_s,
        "loads": settings.burst_loads,
        "entity_cardinalities": settings.entity_cardinalities,
        "seeds": sorted({int(metadata["repetition"]) + seed for _, metadata in plans}),
        "schedule_calibrations": [
            {
                "schedule_id": plan.schedule_id,
                "shape": plan.shape,
                "target_b": plan.target_b,
                "realised_b": plan.realised_b,
                "tail_index": plan.tail_index,
                "iterations": plan.calibration_iterations,
                "converged": plan.calibration_converged,
            }
            for plan, _ in plans
        ],
    }
    parameters_path = out_dir / "workload_parameters.json"
    parameters_path.write_text(json.dumps(parameters, indent=2, sort_keys=True), encoding="utf-8")
    artifacts.append(parameters_path)
    support = {
        "profile_run": settings.profile,
        "research_corpus": "supported-not-run" if settings.profile == "smoke" else "run",
        "full_duration_matrix": "supported-not-run" if settings.profile == "smoke" else "run",
        "separate_load_host": "supported-not-run" if settings.profile == "smoke" else settings.runner.load_generator_host,
        "kafka_transport": "supported-not-run" if settings.runner.transport == "in-process" else "run",
        "research_factory": "BenchmarkConfig.research(...) requires broker, output topic, telemetry file, and non-local load host",
    }
    support_path = out_dir / "support_status.json"
    support_path.write_text(json.dumps(support, indent=2, sort_keys=True), encoding="utf-8")
    artifacts.append(support_path)

    anomaly = gap_by_b[
        (gap_by_b["shape"] == "anomalous_burst") & (gap_by_b["load"] == 1.5)
    ]
    primary_gap = float(anomaly["gap"].mean()) if not anomaly.empty else float(gap_by_b["gap"].mean())
    poisson = gap_by_b[gap_by_b["shape"] == "poisson"]
    poisson_gap = float(poisson["gap"].mean()) if not poisson.empty else float("nan")
    service_constant = float((burst["service_time_constant"] == 1.0).all())
    broker_check = (
        float((burst["broker_telemetry_present"] == 1.0).all())
        if settings.runner.transport == "kafka"
        else float("nan")
    )
    metrics = {
        "knee_hz": knee,
        "primary_gap_b": primary_gap,
        "poisson_gap": poisson_gap,
        "calibration_p50_s": calibration.metrics["fast_p50_s"],
        "calibration_p99_trend_s_per_bucket": calibration.metrics[
            "p99_trend_s_per_bucket"
        ],
        "knee.p99_trend_tolerance": settings.p99_trend_tolerance,
        "knee.lag_slope_threshold": settings.lag_slope_threshold,
        "fit.mean_rate_hz": fit.mean_rate_hz,
        "fit.burstiness_b": fit.burstiness_b,
        "fit.memory_m": fit.memory_m,
        "fit.zipf_exponent": fit.zipf_exponent,
        "checks.calibration_10ms_and_flat_p99": 1.0,
        "checks.generator_p99_skew_le_1ms": float(all(generator_checks)),
        "checks.service_time_constant": service_constant,
        "checks.repetitions_p99_within_20pct": _repetition_check(burst),
        "checks.urgent_gold_nonempty": float((burst["urgent_denominator"] > 0).all()),
        "checks.paired_schedule_ids": float(
            paired.groupby(
                ["shape", "load", "entity_cardinality", "repetition", "partitions", "workers"]
            )["schedule_id"].nunique().max()
            == 1
        ),
        "checks.budget_causal_and_enforced": float(
            (burst[burst["lane_design"] == "two-lane"]["budget_compliant"] == 1.0).all()
        ),
        "checks.poisson_advantage_shrinks": float(
            math.isfinite(poisson_gap) and abs(poisson_gap) <= abs(primary_gap) + 0.05
        ),
        "checks.broker_telemetry_present": broker_check,
    }
    result = ExperimentResult(
        name="e6",
        metrics=metrics,
        tables={
            "knee": knee_table,
            "burst_slo": burst,
            "gap_by_b": gap_by_b,
            "paired_urgent_events": paired,
            "self_amplification": amplification,
            "demand": demand,
            "guards": guards,
            "workload_fit": workload,
            "steady_repetitions": steady,
        },
        artifacts=artifacts,
        primary={
            "urgent_slo_attainment_gap_at_1.5x_knee": primary_gap,
            "single_lane_knee_hz": knee,
            "slo_s": URGENT_SLO_S,
            "gap_by_b_rows": len(gap_by_b),
        },
    )
    result.artifacts.extend(_write_charts(result, out_dir))
    return result


def _repetition_check(rows: pd.DataFrame) -> float:
    group_columns = [
        column
        for column in (
            "rate_hz",
            "shape",
            "lane_design",
            "partitions",
            "workers",
            "load",
            "entity_cardinality",
            "target_b",
        )
        if column in rows.columns
    ]
    for _, group in rows.groupby(group_columns, dropna=False):
        values = group["fast_p99_s"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) < 3:
            return 0.0
        array = values.to_numpy(dtype=float)
        median = float(np.median(array))
        if median > 0 and float((array.max() - array.min()) / median) > 0.2:
            return 0.0
    return 1.0


def _write_charts(result: ExperimentResult, out_dir: Path) -> list[Path]:
    try:
        charts = importlib.import_module("harnext_eval.report.charts")
    except ImportError:
        return []
    hooks = (
        (
            "e6_burst_slo",
            result.tables["burst_slo"].rename(
                columns={"urgent_slo_attainment": "slo_attainment"}
            ),
            out_dir / "burst_slo.png",
        ),
        (
            "self_amplification",
            result.tables["self_amplification"].rename(
                columns={
                    "bucket_s": "time",
                    "fast_admission_rate_hz": "admission_rate",
                    "urgent_slo_attainment": "slo_attainment",
                }
            ),
            out_dir / "self_amplification.png",
        ),
        ("demand_curve", result.tables["demand"], out_dir / "demand_curve.png"),
    )
    generated: list[Path] = []
    for hook_name, table, output in hooks:
        hook = getattr(charts, hook_name, None)
        if hook is not None and not table.empty:
            returned = hook(table, output)
            generated.append(Path(returned) if returned is not None else output)
    return generated


class E6Experiment:
    name = "e6"

    def run(
        self, cfg: EngineConfig, corpus: CorpusHandle, out_dir: Path, seed: int
    ) -> ExperimentResult:
        """Run smoke by default; corpus meta explicitly selects a deployment profile."""

        fit = fit_workload(corpus.events())
        raw = corpus.meta.get("e6", {})
        if raw and not isinstance(raw, Mapping):
            raise ValueError("corpus meta e6 must be a mapping")
        profile = str(raw.get("profile", "smoke")) if raw else "smoke"
        if profile == "smoke":
            benchmark = BenchmarkConfig.smoke()
        elif profile == "research":
            benchmark = BenchmarkConfig.research(
                fit,
                kafka_bootstrap_servers=str(raw["kafka_bootstrap_servers"]),
                kafka_output_topic=str(raw["kafka_output_topic"]),
                load_generator_host=str(raw["load_generator_host"]),
                kafka_telemetry_path=Path(str(raw["kafka_telemetry_path"])),
            )
        else:
            raise ValueError("e6 profile must be 'smoke' or 'research'")
        catalogue = situations_from_meta(fit, corpus.meta)
        return asyncio.run(
            run_benchmark(
                fit, cfg, out_dir, seed=seed, benchmark=benchmark, situations=catalogue
            )
        )

    def chart(self, result: ExperimentResult, out_dir: Path) -> list[Path]:
        """Write the exact three E6 PNG names."""

        out_dir.mkdir(parents=True, exist_ok=True)
        return _write_charts(result, out_dir)


experiment = register_experiment(E6Experiment())


__all__ = [
    "BenchmarkConfig",
    "E6Experiment",
    "E6RouterPolicy",
    "EventObservation",
    "GuardOverrides",
    "PipelineRun",
    "RunnerConfig",
    "calibration_run",
    "calibration_valid",
    "experiment",
    "external_records_to_run",
    "find_knee",
    "run_benchmark",
    "run_schedule",
    "self_amplification_series",
    "sweep_steady",
]
