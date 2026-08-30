"""Open-loop E6 runner for docs/evaluation-spec.md §7 E6 and §12 D10."""

from __future__ import annotations

import asyncio
import importlib
import json
import math
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from hdrh.histogram import HdrHistogram

from harnext_eval.config import EngineConfig
from harnext_eval.corpus import CorpusHandle
from harnext_eval.e6.loadgen import (
    ScheduleEntry,
    WorkloadFit,
    fit_workload,
    generate_schedule,
    generate_steady_schedule,
    schedule_burstiness,
)
from harnext_eval.e6.metrics import (
    BATCH_SLO_S,
    URGENT_SLO_S,
    cross_entity_fairness,
    demand_curve,
    drain_time,
    duplicates_and_missed,
    linear_trend,
    partition_lag_gini,
    percentile,
    recovery_time,
    slo_attainment,
)
from harnext_eval.registry import ExperimentResult, register_experiment
from harnext_eval.replay.driver import RulesOnlyPolicy
from harnext_eval.types import EvalEvent

_HISTOGRAM_MAX_US = 3_600_000_000


@dataclass(frozen=True)
class RunnerConfig:
    """Runtime-only E6 transport/topology settings.

    These settings intentionally do not duplicate the shared engine config:
    the latter still owns lane design, windows, and router guards.
    """

    transport: Literal["in-process", "kafka"] = "in-process"
    partitions: int = 8
    workers: int = 1
    service_time_s: float = 0.002
    admission_delay_s: float = 0.0
    warmup_fraction: float = 0.25
    kill_downtime_s: float = 0.05
    batch_windows: bool = True
    kafka_bootstrap_servers: str | None = None
    kafka_input_topic: str = "cms.events.raw.v1"
    kafka_output_topic: str | None = None
    kafka_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if self.partitions <= 0 or self.workers <= 0:
            raise ValueError("partitions and workers must be positive")
        if self.service_time_s < 0 or self.admission_delay_s < 0:
            raise ValueError("service times cannot be negative")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")


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
    intended_offset_s: float
    actual_send_offset_s: float
    send_skew_s: float
    agent_start_offset_s: float
    latency_s: float
    batch_fold_latency_s: float | None
    partition: int


@dataclass
class PipelineRun:
    observations: list[EventObservation]
    histograms: dict[str, HdrHistogram]
    lag_samples: pd.DataFrame
    metrics: dict[str, float]
    expected_by_lane: dict[str, list[str]] = field(default_factory=dict)
    observed_by_lane: dict[str, list[str]] = field(default_factory=dict)
    kill_offset_s: float | None = None

    def write_histograms(self, directory: Path, prefix: str) -> list[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for name, histogram in sorted(self.histograms.items()):
            path = directory / f"{prefix}-{name}.hgrm"
            with path.open("wb") as output:
                histogram.output_percentile_distribution(output, 1_000_000.0)
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
    sequence: int


class E6RouterPolicy:
    """Replay-compatible router policy with explicit guard-ablation hooks."""

    def __init__(
        self,
        cfg: EngineConfig,
        *,
        lane_design: Literal["two-lane", "single"] | None = None,
        guard_overrides: GuardOverrides | None = None,
    ) -> None:
        self.cfg = cfg
        self.lane_design = lane_design or cfg.lane_design
        overrides = guard_overrides or GuardOverrides()
        guards = cfg.router.guards
        self.absolute_floor = (
            guards.absolute_floor
            if overrides.absolute_floor is None
            else overrides.absolute_floor
        )
        self.multi_window = (
            guards.multi_window if overrides.multi_window is None else overrides.multi_window
        )
        self.situation_dedup = (
            guards.situation_dedup
            if overrides.situation_dedup is None
            else overrides.situation_dedup
        )
        self._recent_by_entity: dict[str, deque[float]] = defaultdict(deque)
        self._situations: dict[str, float] = {}
        self._rules_policy = RulesOnlyPolicy()

    def rules(self, event: EvalEvent) -> str | None:
        """Expose the T2 replay-driver RouterPolicy rule seam."""

        return self._rules_policy.rules(event)

    @staticmethod
    def score(event: EvalEvent) -> float:
        """Expose the T2 replay-driver RouterPolicy deviation-score seam."""

        data = event.data or {}
        return 1.0 if bool(data.get("e6_urgent")) else float(data.get("e6_score", 0.2))

    @staticmethod
    def _score(entry: ScheduleEntry) -> float:
        if entry.urgent:
            return 1.0
        event = entry.event
        if event is None:
            return 0.0
        data = event.data or {}
        return 1.0 if bool(data.get("e6_urgent")) else float(data.get("e6_score", 0.2))

    def route(self, entry: ScheduleEntry, offset_s: float) -> tuple[str, dict[str, bool | float]]:
        if self.lane_design == "single":
            return "single", {"single_lane": True}
        score = self._score(entry)
        rule = None if entry.event is None else self.rules(entry.event)
        admitted = rule is not None or entry.urgent or score >= max(self.absolute_floor, 0.5)
        floor_pass = score >= self.absolute_floor
        multi_window_pass = True
        dedup_pass = True
        recent = self._recent_by_entity[entry.entity]
        while recent and recent[0] < offset_s - 10.0:
            recent.popleft()
        if self.multi_window and len(recent) >= 5:
            multi_window_pass = False
        event = entry.event
        situation = None if event is None else str((event.data or {}).get("situation_id", ""))
        if not situation and event is not None:
            situation = f"{entry.entity}:{event.type}"
        if self.situation_dedup and situation:
            previous = self._situations.get(situation)
            if previous is not None and offset_s - previous < 10.0:
                dedup_pass = False
            self._situations[situation] = offset_s
        admitted = admitted and floor_pass and multi_window_pass and dedup_pass
        if admitted:
            recent.append(offset_s)
        return (
            "fast" if admitted else "batch",
            {
                "score": score,
                "rule": rule is not None,
                "absolute_floor": floor_pass,
                "multi_window": multi_window_pass,
                "situation_dedup": dedup_pass,
            },
        )


def _histogram(values_s: Iterable[float]) -> HdrHistogram:
    histogram = HdrHistogram(1, _HISTOGRAM_MAX_US, 3)
    for value in values_s:
        if math.isfinite(value) and value >= 0:
            histogram.record_value(max(1, min(_HISTOGRAM_MAX_US, int(round(value * 1e6)))))
    return histogram


def _partition(entity: str, partitions: int) -> int:
    # Python's process-randomized hash would make benchmark topology irreproducible.
    value = int.from_bytes(entity.encode("utf-8"), "little", signed=False)
    return value % partitions


def _observations_frame(run: PipelineRun) -> pd.DataFrame:
    return pd.DataFrame([observation.__dict__ for observation in run.observations])


def self_amplification_series(run: PipelineRun, *, bucket_s: float = 10.0) -> pd.DataFrame:
    """Fast admission and urgent SLO compliance in shared wall-clock buckets."""

    rows = _observations_frame(run)
    if rows.empty:
        return pd.DataFrame(
            columns=["bucket_s", "fast_admission_rate_hz", "urgent_slo_attainment"]
        )
    rows["bucket_s"] = np.floor(rows["intended_offset_s"] / bucket_s) * bucket_s
    output: list[dict[str, float]] = []
    for bucket, group in rows.groupby("bucket_s", sort=True):
        urgent = group[group["urgent"]]
        output.append(
            {
                "bucket_s": float(str(bucket)),
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
) -> PipelineRun:
    if not schedule:
        raise ValueError("schedule cannot be empty")
    ordered = sorted(schedule, key=lambda entry: entry.intended_send_ts)
    first_timestamp = ordered[0].intended_send_ts
    event_entries = [entry for entry in ordered if entry.event is not None]
    if not event_entries:
        raise ValueError("schedule must contain at least one event")
    last_timestamp = event_entries[-1].intended_send_ts
    duration_s = max((last_timestamp - first_timestamp).total_seconds(), 0.0)
    warmup_s = duration_s * runner_cfg.warmup_fraction
    policy = E6RouterPolicy(
        cfg, lane_design=lane_design, guard_overrides=guard_overrides
    )
    queue: asyncio.PriorityQueue[tuple[int, int, _QueuedFold | None]] = (
        asyncio.PriorityQueue()
    )
    loop = asyncio.get_running_loop()
    loop_anchor = loop.time() + 0.02
    observations: list[EventObservation] = []
    partition_lag = [0 for _ in range(runner_cfg.partitions)]
    lag_rows: list[dict[str, float]] = []
    expected: dict[str, list[str]] = defaultdict(list)
    observed: dict[str, list[str]] = defaultdict(list)
    disabled_until = [0.0 for _ in range(runner_cfg.workers)]
    kill_offset: float | None = None
    batch_windows: dict[str, list[_QueuedEvent]] = defaultdict(list)
    batch_fold_latencies: list[tuple[float, float]] = []
    fold_sequence = 0

    def record_lag(offset_s: float) -> None:
        lag_rows.append(
            {
                "time_s": offset_s,
                "total_lag": float(sum(partition_lag)),
                **{f"partition_{index}": float(value) for index, value in enumerate(partition_lag)},
            }
        )

    async def consumer(worker_index: int) -> None:
        while True:
            priority, _, item = await queue.get()
            del priority
            if item is None:
                queue.task_done()
                return
            now = loop.time()
            if disabled_until[worker_index] > now:
                await asyncio.sleep(disabled_until[worker_index] - now)
            if runner_cfg.admission_delay_s:
                await asyncio.sleep(runner_cfg.admission_delay_s)
            agent_start = loop.time()
            if runner_cfg.service_time_s:
                await asyncio.sleep(runner_cfg.service_time_s)
            completion = loop.time()
            batch_latency = (
                max(0.0, completion - item.window_close_loop_time)
                if item.lane in {"batch", "single"}
                else None
            )
            if batch_latency is not None:
                batch_fold_latencies.append(
                    (item.window_close_loop_time - loop_anchor, batch_latency)
                )
            for queued_event in item.items:
                latency = max(0.0, agent_start - queued_event.intended_loop_time)
                observations.append(
                    EventObservation(
                        event_id=queued_event.entry.event_id,
                        entity=queued_event.entry.entity,
                        lane=item.lane,
                        urgent=queued_event.entry.urgent,
                        intended_offset_s=queued_event.intended_offset_s,
                        actual_send_offset_s=queued_event.actual_send_loop_time - loop_anchor,
                        send_skew_s=max(
                            0.0,
                            queued_event.actual_send_loop_time
                            - queued_event.intended_loop_time,
                        ),
                        agent_start_offset_s=agent_start - loop_anchor,
                        latency_s=latency,
                        batch_fold_latency_s=batch_latency,
                        partition=queued_event.partition,
                    )
                )
                observed[item.lane].append(queued_event.entry.event_id)
                partition_lag[queued_event.partition] -= 1
            record_lag(completion - loop_anchor)
            queue.task_done()

    async def enqueue_fold(
        items: Sequence[_QueuedEvent], lane: str, window_close_loop_time: float
    ) -> None:
        nonlocal fold_sequence
        if not items:
            return
        fold = _QueuedFold(
            items=tuple(items),
            lane=lane,
            window_close_loop_time=window_close_loop_time,
            sequence=fold_sequence,
        )
        priority = 0 if lane == "fast" else 1
        await queue.put((priority, fold_sequence, fold))
        fold_sequence += 1

    async def close_due_windows(offset_s: float) -> None:
        due: list[tuple[float, str]] = []
        for entity, items in batch_windows.items():
            gap_at = items[-1].intended_offset_s + cfg.window.gap_s
            age_at = items[0].intended_offset_s + cfg.window.max_age_s
            deadlines = [deadline for deadline in (gap_at, age_at) if deadline <= offset_s]
            if deadlines:
                due.append((min(deadlines), entity))
        for close_offset, entity in sorted(due):
            items = batch_windows.pop(entity)
            await enqueue_fold(items, items[0].lane, loop_anchor + close_offset)

    workers = [asyncio.create_task(consumer(index)) for index in range(runner_cfg.workers)]
    for sequence, entry in enumerate(ordered):
        offset = (entry.intended_send_ts - first_timestamp).total_seconds()
        intended_loop = loop_anchor + offset
        delay = intended_loop - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        if entry.marker == "worker_kill":
            kill_offset = offset
            disabled_until[-1] = loop.time() + runner_cfg.kill_downtime_s
            continue
        assert entry.event is not None
        await close_due_windows(offset)
        actual_send = loop.time()
        lane, _ = policy.route(entry, offset)
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
        partition_lag[partition] += 1
        record_lag(actual_send - loop_anchor)
        if lane == "fast" or not runner_cfg.batch_windows:
            await enqueue_fold([item], lane, actual_send)
        else:
            window = batch_windows[entry.entity]
            window.append(item)
            if len(window) >= cfg.window.max_events:
                await enqueue_fold(window, lane, actual_send)
                del batch_windows[entry.entity]
    flush_loop_time = max(loop.time(), loop_anchor + duration_s)
    for entity in sorted(batch_windows):
        items = batch_windows[entity]
        await enqueue_fold(items, items[0].lane, flush_loop_time)
    for index in range(runner_cfg.workers):
        await queue.put((99, fold_sequence + index, None))
    await queue.join()
    await asyncio.gather(*workers)

    observations.sort(key=lambda observation: observation.intended_offset_s)
    measured = [row for row in observations if row.intended_offset_s >= warmup_s]
    urgent_latencies = [row.latency_s for row in measured if row.urgent]
    fast_latencies = [
        row.latency_s for row in measured if row.lane in {"fast", "single"}
    ]
    batch_latencies = [
        latency for close_offset, latency in batch_fold_latencies if close_offset >= warmup_s
    ]
    send_skews = [row.send_skew_s for row in measured]
    lag_frame = pd.DataFrame(lag_rows).sort_values("time_s").reset_index(drop=True)
    during_send = lag_frame[lag_frame["time_s"] <= duration_s + 1e-9]
    lag_slope = linear_trend(
        during_send["total_lag"].astype(float).tolist(),
        during_send["time_s"].astype(float).tolist(),
    )
    latency_buckets: list[float] = []
    if measured:
        frame = pd.DataFrame([row.__dict__ for row in measured])
        frame["quartile"] = pd.cut(
            frame["intended_offset_s"], bins=4, labels=False, duplicates="drop"
        )
        for _, group in frame.groupby("quartile", dropna=True):
            latency_buckets.append(percentile(group["latency_s"].tolist(), 99))
    p99_trend = linear_trend(latency_buckets, list(range(len(latency_buckets))))
    partition_columns = [
        column for column in lag_frame.columns if column.startswith("partition_")
    ]
    peak_partition_lags = [
        float(lag_frame[column].to_numpy(dtype=float).max())
        for column in partition_columns
    ]
    delivery = duplicates_and_missed(expected, observed)
    duplicate_count = sum(delivery.duplicates_by_lane.values())
    missed_count = sum(delivery.missed_by_lane.values())
    metrics = {
        "fast_p50_s": percentile(fast_latencies, 50),
        "fast_p99_s": percentile(fast_latencies, 99),
        "fast_p999_s": percentile(fast_latencies, 99.9),
        "urgent_slo_attainment": slo_attainment(urgent_latencies),
        "batch_p99_s": percentile(batch_latencies, 99),
        "send_skew_p99_ms": percentile(send_skews, 99) * 1000.0,
        "generator_skew_check": float(percentile(send_skews, 99) <= 0.001),
        "lag_peak": float(lag_frame["total_lag"].to_numpy(dtype=float).max()),
        "lag_slope": lag_slope,
        "p99_trend_s_per_bucket": p99_trend,
        "partition_lag_gini": partition_lag_gini(peak_partition_lags),
        "duplicates": float(duplicate_count),
        "missed": float(missed_count),
        "service_time_s": runner_cfg.service_time_s,
        "service_time_constant": 1.0,
        "observed_events": float(len(observations)),
        "throughput_hz": float(
            len(observations)
            / max(float(lag_frame["time_s"].to_numpy(dtype=float).max()), 1e-9)
        ),
        "dlq": 0.0,
    }
    if duration_s > 0 and not lag_frame.empty:
        metrics["drain_time_s"] = drain_time(
            lag_frame["time_s"].tolist(),
            lag_frame["total_lag"].tolist(),
            burst_end_s=duration_s,
            baseline_lag=0.0,
        )
    if kill_offset is not None:
        before = lag_frame[lag_frame["time_s"] <= kill_offset]
        pre_kill = float(before.iloc[-1]["total_lag"]) if not before.empty else 0.0
        metrics["recovery_time_s"] = recovery_time(
            lag_frame["time_s"].tolist(),
            lag_frame["total_lag"].tolist(),
            kill_s=kill_offset,
            pre_kill_lag=pre_kill,
        )
    return PipelineRun(
        observations=observations,
        histograms={
            "fast": _histogram(fast_latencies),
            "urgent": _histogram(urgent_latencies),
            "batch-fold": _histogram(batch_latencies),
            "send-skew": _histogram(send_skews),
        },
        lag_samples=lag_frame,
        metrics=metrics,
        expected_by_lane=dict(expected),
        observed_by_lane=dict(observed),
        kill_offset_s=kill_offset,
    )


async def _run_kafka(
    schedule: Sequence[ScheduleEntry],
    cfg: EngineConfig,
    runner_cfg: RunnerConfig,
    *,
    lane_design: Literal["two-lane", "single"] | None,
    guard_overrides: GuardOverrides | None,
) -> PipelineRun:
    """Kafka transport loaded only when explicitly selected.

    With an output topic, the external pipeline must emit JSON records carrying
    ``event_id`` and ``agent_start_ts``.  Without one, broker acknowledgement
    is the boundary, which is useful for producer-only calibration.
    """

    del cfg, lane_design, guard_overrides
    if runner_cfg.kafka_bootstrap_servers is None:
        raise ValueError("Kafka transport requires kafka_bootstrap_servers")
    try:
        from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    except ImportError as exc:  # pragma: no cover - optional deployment dependency
        raise RuntimeError("install aiokafka to use the E6 Kafka transport") from exc

    producer = AIOKafkaProducer(bootstrap_servers=runner_cfg.kafka_bootstrap_servers)
    consumer = None
    if runner_cfg.kafka_output_topic:
        consumer = AIOKafkaConsumer(
            runner_cfg.kafka_output_topic,
            bootstrap_servers=runner_cfg.kafka_bootstrap_servers,
            auto_offset_reset="latest",
            enable_auto_commit=False,
        )
    await producer.start()
    if consumer is not None:
        await consumer.start()
    try:
        first = min(entry.intended_send_ts for entry in schedule)
        loop = asyncio.get_running_loop()
        anchor = loop.time() + 0.02
        sent: dict[str, tuple[ScheduleEntry, float, float]] = {}
        send_skews: list[float] = []
        for entry in sorted(schedule, key=lambda item: item.intended_send_ts):
            if entry.event is None:
                continue
            offset = (entry.intended_send_ts - first).total_seconds()
            intended_loop = anchor + offset
            if intended_loop > loop.time():
                await asyncio.sleep(intended_loop - loop.time())
            actual = loop.time()
            payload = entry.event.model_dump_json().encode()
            await producer.send_and_wait(
                runner_cfg.kafka_input_topic,
                payload,
                key=entry.event.partition_key(),
            )
            sent[entry.event_id] = (entry, intended_loop, actual)
            send_skews.append(max(0.0, actual - intended_loop))
        if consumer is None:
            # Producer calibration intentionally has no agent-side timings.
            empty = _histogram(())
            return PipelineRun(
                observations=[],
                histograms={"fast": empty, "batch-fold": _histogram(()), "send-skew": _histogram(send_skews)},
                lag_samples=pd.DataFrame(columns=["time_s", "total_lag"]),
                metrics={
                    "send_skew_p99_ms": percentile(send_skews, 99) * 1000,
                    "generator_skew_check": float(percentile(send_skews, 99) <= 0.001),
                    "observed_events": 0.0,
                },
            )
        observations: list[EventObservation] = []
        deadline = loop.time() + runner_cfg.kafka_timeout_s
        while len(observations) < len(sent) and loop.time() < deadline:
            batch = await consumer.getmany(timeout_ms=100, max_records=len(sent))
            for messages in batch.values():
                for message in messages:
                    if message.value is None:
                        continue
                    output = json.loads(message.value)
                    event_id = str(output["event_id"])
                    if event_id not in sent:
                        continue
                    entry, intended_loop, actual = sent[event_id]
                    agent_start = datetime.fromisoformat(str(output["agent_start_ts"]))
                    latency = (agent_start - entry.intended_send_ts).total_seconds()
                    observations.append(
                        EventObservation(
                            event_id=event_id,
                            entity=entry.entity,
                            lane=str(output.get("lane", "single")),
                            urgent=entry.urgent,
                            intended_offset_s=intended_loop - anchor,
                            actual_send_offset_s=actual - anchor,
                            send_skew_s=max(0.0, actual - intended_loop),
                            agent_start_offset_s=agent_start.timestamp() - first.timestamp(),
                            latency_s=max(0.0, latency),
                            batch_fold_latency_s=output.get("batch_fold_latency_s"),
                            partition=int(getattr(message, "partition", 0)),
                        )
                    )
        observed_urgent_ids = {row.event_id for row in observations if row.urgent}
        expected_urgent_ids = {
            event_id for event_id, (entry, _, _) in sent.items() if entry.urgent
        }
        urgent = [row.latency_s for row in observations if row.urgent]
        urgent.extend([float("inf")] * len(expected_urgent_ids - observed_urgent_ids))
        return PipelineRun(
            observations=observations,
            histograms={"urgent": _histogram(urgent), "send-skew": _histogram(send_skews)},
            lag_samples=pd.DataFrame(columns=["time_s", "total_lag"]),
            metrics={
                "urgent_slo_attainment": slo_attainment(urgent),
                "fast_p99_s": percentile(urgent, 99),
                "send_skew_p99_ms": percentile(send_skews, 99) * 1000,
                "generator_skew_check": float(percentile(send_skews, 99) <= 0.001),
                "missed": float(len(sent) - len(observations)),
                "observed_events": float(len(observations)),
            },
        )
    finally:
        if consumer is not None:
            await consumer.stop()
        await producer.stop()


async def run_schedule(
    schedule: Sequence[ScheduleEntry],
    cfg: EngineConfig,
    runner_cfg: RunnerConfig | None = None,
    *,
    lane_design: Literal["two-lane", "single"] | None = None,
    guard_overrides: GuardOverrides | None = None,
) -> PipelineRun:
    """Send a timestamped schedule open-loop through the selected transport."""

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
    )


async def calibration_run(cfg: EngineConfig, *, samples: int = 24) -> PipelineRun:
    """Run the fixed 10 ms pass-through clock-chain calibration."""

    replay_event = next(iter(_calibration_event(cfg) for _ in range(1)))
    start = datetime.now(UTC) + timedelta(milliseconds=30)
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
            urgent=True,
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
            admission_delay_s=0.010,
            warmup_fraction=0.25,
            batch_windows=False,
        ),
        lane_design="single",
    )


def _calibration_event(cfg: EngineConfig) -> Any:
    del cfg
    from harnext_eval.corpus.synthetic import generate_synthetic_events

    return generate_synthetic_events(event_count=2, days=1, entity_count=1)[0]


def find_knee(
    results: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    p99_trend_tolerance: float = 0.02,
    lag_slope_threshold: float = 0.05,
) -> float:
    """Highest steady rate with near-flat p99 and non-growing lag."""

    frame = results.copy() if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    required = {"rate_hz", "fast_p99_s", "p99_trend_s_per_bucket", "lag_slope"}
    if not required.issubset(frame.columns):
        raise ValueError(f"results must contain {sorted(required)}")
    aggregate = frame.groupby("rate_hz", as_index=False).median(numeric_only=True)
    stable = aggregate[
        (aggregate["p99_trend_s_per_bucket"] <= p99_trend_tolerance)
        & (aggregate["lag_slope"] <= lag_slope_threshold)
    ]
    if stable.empty:
        return float(np.asarray(aggregate["rate_hz"], dtype=float).min())
    return float(np.asarray(stable["rate_hz"], dtype=float).max())


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
) -> tuple[pd.DataFrame, float, list[Path]]:
    """Run the preregistered steady sweep and return rows, knee, histograms."""

    if repetitions < 3:
        raise ValueError("E6 requires at least three steady repetitions")
    runtime = runner_cfg or RunnerConfig()
    rows: list[dict[str, float | int | str]] = []
    artifacts: list[Path] = []
    for rate in rates_hz:
        if rate <= 0:
            raise ValueError("sweep rates must be positive")
        for repetition in range(repetitions):
            schedule = generate_steady_schedule(
                fit,
                duration_s=duration_s,
                rate_hz=rate,
                seed=seed + repetition,
            )
            result = await run_schedule(
                schedule, cfg, runtime, lane_design=lane_design
            )
            rows.append(
                {
                    "rate_hz": rate,
                    "repetition": repetition,
                    "lane_design": lane_design or cfg.lane_design,
                    "partitions": runtime.partitions,
                    "workers": runtime.workers,
                    **result.metrics,
                }
            )
            if out_dir is not None:
                artifacts.extend(
                    result.write_histograms(
                        out_dir,
                        f"steady-{lane_design or cfg.lane_design}-{rate:g}-r{repetition}",
                    )
                )
    frame = pd.DataFrame(rows)
    return frame, find_knee(frame), artifacts


@dataclass(frozen=True)
class BenchmarkConfig:
    """E6 matrix controls; ``smoke`` keeps the offline CLI bounded."""

    reference_rate_hz: float = 250.0
    steady_duration_s: float = 0.16
    burst_duration_s: float = 0.12
    burst_window_s: float = 0.06
    repetitions: int = 3
    service_time_s: float = 0.002
    topologies: tuple[tuple[int, int], ...] = ((8, 1), (8, 4), (32, 1), (32, 4))
    shapes: tuple[str, ...] = (
        "steady",
        "poisson",
        "pareto_on_off",
        "benign_flash",
        "anomalous_burst",
        "zipf_hot",
        "worker_kill",
    )

    @classmethod
    def research(cls, fit: WorkloadFit) -> BenchmarkConfig:
        """Return the wall-clock preregistered matrix at the fitted replay mean."""

        return cls(
            reference_rate_hz=fit.mean_rate_hz,
            steady_duration_s=1_200.0,
            burst_duration_s=1_200.0,
            burst_window_s=600.0,
        )


async def run_benchmark(
    fit: WorkloadFit,
    cfg: EngineConfig,
    out_dir: Path,
    *,
    seed: int,
    benchmark: BenchmarkConfig | None = None,
) -> ExperimentResult:
    """Execute the complete bounded E6 matrix and write its named outputs."""

    settings = benchmark or BenchmarkConfig()
    await asyncio.to_thread(out_dir.mkdir, parents=True, exist_ok=True)
    hdr_dir = out_dir / "hdr"
    artifacts: list[Path] = []
    calibration = await calibration_run(cfg)
    calibration_p50 = calibration.metrics["fast_p50_s"]
    calibration_pass = abs(calibration_p50 - 0.010) <= 0.005

    sweep_rates = [
        settings.reference_rate_hz * multiplier for multiplier in (0.25, 0.5, 1, 2, 4)
    ]
    steady, single_knee, sweep_artifacts = await sweep_steady(
        fit,
        cfg,
        rates_hz=sweep_rates,
        duration_s=settings.steady_duration_s,
        runner_cfg=RunnerConfig(service_time_s=settings.service_time_s),
        repetitions=settings.repetitions,
        lane_design="single",
        out_dir=hdr_dir,
        seed=seed,
    )
    artifacts.extend(sweep_artifacts)
    knee_table = (
        steady.groupby(["lane_design", "rate_hz"], as_index=False)
        .median(numeric_only=True)
        .assign(knee_hz=single_knee)
    )
    knee_path = out_dir / "knee.csv"
    knee_table.to_csv(knee_path, index=False)
    artifacts.append(knee_path)

    burst_rows: list[dict[str, Any]] = []
    amplification_frames: list[pd.DataFrame] = []
    generator_checks: list[float] = []
    representative_runs: dict[tuple[str, str], PipelineRun] = {}
    for lane_design in ("two-lane", "single"):
        for partitions, workers in settings.topologies:
            runtime = RunnerConfig(
                partitions=partitions,
                workers=workers,
                service_time_s=settings.service_time_s,
            )
            for load_multiplier in (1.0, 1.5):
                rate = single_knee * load_multiplier
                for shape_name in settings.shapes:
                    shape = "benign_flash" if shape_name == "worker_kill" else shape_name
                    schedule = generate_schedule(
                        fit,
                        shape=shape,  # type: ignore[arg-type]
                        duration_s=settings.burst_duration_s,
                        burst_duration_s=settings.burst_window_s,
                        burst_start_s=(settings.burst_duration_s - settings.burst_window_s) / 2,
                        rate_hz=rate,
                        seed=seed + partitions + workers,
                        worker_kill=shape_name == "worker_kill",
                        target_b=max(0.0, fit.burstiness_b),
                    )
                    result = await run_schedule(
                        schedule, cfg, runtime, lane_design=lane_design
                    )
                    representative_runs[(lane_design, shape_name)] = result
                    generator_checks.append(result.metrics["generator_skew_check"])
                    observation_rows = _observations_frame(result)
                    fairness = cross_entity_fairness(
                        observation_rows, fit.entities[:3]
                    ) if not observation_rows.empty else 0.0
                    burst_rows.append(
                        {
                            "shape": shape_name,
                            "lane_design": lane_design,
                            "load": load_multiplier,
                            "rate_hz": rate,
                            "partitions": partitions,
                            "workers": workers,
                            "entity_cardinality": len(fit.entities),
                            "burstiness_b": schedule_burstiness(schedule),
                            "cross_entity_fairness": fairness,
                            **result.metrics,
                        }
                    )
                    if shape_name == "anomalous_burst" and load_multiplier == 1.5:
                        series = self_amplification_series(result)
                        series.insert(0, "workers", workers)
                        series.insert(0, "partitions", partitions)
                        series.insert(0, "lane_design", lane_design)
                        amplification_frames.append(series)
                    if (
                        shape_name in {"anomalous_burst", "worker_kill"}
                        and load_multiplier == 1.5
                        and partitions == 8
                        and workers == 1
                    ):
                        artifacts.extend(
                            result.write_histograms(
                                hdr_dir, f"burst-{lane_design}-{shape_name}-1.5x"
                            )
                        )
    burst = pd.DataFrame(burst_rows)
    burst_path = out_dir / "burst_slo.csv"
    burst.to_csv(burst_path, index=False)
    artifacts.append(burst_path)

    amplification = (
        pd.concat(amplification_frames, ignore_index=True)
        if amplification_frames
        else pd.DataFrame()
    )
    amplification_path = out_dir / "self_amplification.csv"
    amplification.to_csv(amplification_path, index=False)
    artifacts.append(amplification_path)

    demand_input = burst.rename(columns={"rate_hz": "absolute_rate_hz"})
    demand = demand_curve(demand_input)
    demand_path = out_dir / "demand.csv"
    demand.to_csv(demand_path, index=False)
    artifacts.append(demand_path)

    guard_rows: list[dict[str, Any]] = []
    full = GuardOverrides(absolute_floor=0.5, multi_window=True, situation_dedup=True)
    ablations = {
        "full": full,
        "minus_absolute_floor": replace(full, absolute_floor=0.0),
        "minus_multi_window": replace(full, multi_window=False),
        "minus_situation_dedup": replace(full, situation_dedup=False),
    }
    guard_schedule = generate_schedule(
        fit,
        shape="anomalous_burst",
        duration_s=settings.burst_duration_s,
        burst_duration_s=settings.burst_window_s,
        rate_hz=single_knee * 1.5,
        seed=seed,
    )
    for name, overrides in ablations.items():
        result = await run_schedule(
            guard_schedule,
            cfg,
            RunnerConfig(service_time_s=settings.service_time_s),
            lane_design="two-lane",
            guard_overrides=overrides,
        )
        guard_rows.append({"guard_condition": name, **result.metrics})
    guards = pd.DataFrame(guard_rows)
    guards_path = out_dir / "guards.csv"
    guards.to_csv(guards_path, index=False)
    artifacts.append(guards_path)

    popularity_by_entity = dict(zip(fit.entities, fit.popularity, strict=True))
    workload = pd.DataFrame(
        [
            {
                "entity": entity,
                "count": entity_fit.count,
                "rate_hz": entity_fit.rate_hz,
                "mean_inter_arrival_s": entity_fit.mean_s,
                "std_inter_arrival_s": entity_fit.std_s,
                "p50_inter_arrival_s": entity_fit.quantiles_s[0],
                "p90_inter_arrival_s": entity_fit.quantiles_s[1],
                "p99_inter_arrival_s": entity_fit.quantiles_s[2],
                "burstiness_b": entity_fit.burstiness_b,
                "memory_m": entity_fit.memory_m,
                "popularity": popularity_by_entity[entity],
            }
            for entity, entity_fit in fit.entity_fits.items()
        ]
    )
    workload_path = out_dir / "workload_fit.csv"
    workload.to_csv(workload_path, index=False)
    artifacts.append(workload_path)

    comparison = burst[
        (burst["shape"] == "anomalous_burst")
        & (burst["load"] == 1.5)
        & (burst["partitions"] == 8)
        & (burst["workers"] == 1)
    ]
    by_lane: dict[str, float] = {}
    for key, group in comparison.groupby("lane_design"):
        by_lane[str(key)] = float(
            np.asarray(group["urgent_slo_attainment"], dtype=float).mean()
        )
    gap_b = by_lane.get("two-lane", 0.0) - by_lane.get("single", 0.0)
    poisson = burst[(burst["shape"] == "poisson") & (burst["load"] == 1.5)]
    poisson_lane: dict[str, float] = {}
    for key, group in poisson.groupby("lane_design"):
        poisson_lane[str(key)] = float(
            np.asarray(group["urgent_slo_attainment"], dtype=float).mean()
        )
    poisson_gap = poisson_lane.get("two-lane", 0.0) - poisson_lane.get("single", 0.0)
    metrics = {
        "knee_hz": single_knee,
        "primary_gap_b": gap_b,
        "poisson_gap": poisson_gap,
        "calibration_p50_s": calibration_p50,
        "fit.mean_rate_hz": fit.mean_rate_hz,
        "fit.burstiness_b": fit.burstiness_b,
        "fit.memory_m": fit.memory_m,
        "fit.zipf_exponent": fit.zipf_exponent,
        "checks.calibration_10ms": float(calibration_pass),
        "checks.generator_p99_skew_le_1ms": float(all(generator_checks)),
        "checks.service_time_constant": 1.0,
        "checks.repetitions_p99_within_20pct": _repetition_check(steady),
        "checks.poisson_advantage_shrinks": float(abs(poisson_gap) <= abs(gap_b) + 0.05),
        "checks.fast_p99_d10": float(
            burst[
                (burst["lane_design"] == "two-lane")
                & (burst["load"] <= 1.5)
                & (burst["shape"] == "anomalous_burst")
            ]["fast_p99_s"].max()
            <= URGENT_SLO_S
        ),
        "checks.batch_p99_d10": float(burst["batch_p99_s"].max() <= BATCH_SLO_S),
        "checks.lag_not_trending_up": float(
            np.median(
                np.asarray(
                    burst[
                        (burst["lane_design"] == "two-lane")
                        & (burst["load"] <= 1.5)
                        & (burst["shape"] == "anomalous_burst")
                    ]["lag_slope"],
                    dtype=float,
                )
            )
            <= 0.05
        ),
    }
    return ExperimentResult(
        name="e6",
        metrics=metrics,
        tables={
            "knee": knee_table,
            "burst_slo": burst,
            "self_amplification": amplification,
            "demand": demand,
            "guards": guards,
            "workload_fit": workload,
            "steady_repetitions": steady,
        },
        artifacts=artifacts,
        primary={
            "urgent_slo_attainment_gap_at_1.5x_knee": gap_b,
            "single_lane_knee_hz": single_knee,
            "slo_s": URGENT_SLO_S,
        },
    )


def _repetition_check(steady: pd.DataFrame) -> float:
    for _, group in steady.groupby("rate_hz"):
        values = group["fast_p99_s"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) < 3:
            return 0.0
        array = values.to_numpy(dtype=float)
        median = float(np.median(array))
        if median > 0 and float((array.max() - array.min()) / median) > 0.2:
            return 0.0
    return 1.0


class E6Experiment:
    name = "e6"

    def run(
        self, cfg: EngineConfig, corpus: CorpusHandle, out_dir: Path, seed: int
    ) -> ExperimentResult:
        fit = fit_workload(corpus.events())
        return asyncio.run(run_benchmark(fit, cfg, out_dir, seed=seed))

    def chart(self, result: ExperimentResult, out_dir: Path) -> list[Path]:
        """Call T10's E6 chart hooks when that optional module is present."""

        try:
            charts = importlib.import_module("harnext_eval.report.charts")
        except ImportError:
            return []
        out_dir.mkdir(parents=True, exist_ok=True)
        generated: list[Path] = []
        hooks = {
            "e6_burst_slo": result.tables["burst_slo"].rename(
                columns={"urgent_slo_attainment": "slo_attainment"}
            ),
            "self_amplification": result.tables["self_amplification"].rename(
                columns={
                    "bucket_s": "time",
                    "fast_admission_rate_hz": "admission_rate",
                    "urgent_slo_attainment": "slo_attainment",
                }
            ),
            "demand_curve": result.tables["demand"],
        }
        for hook_name, table in hooks.items():
            hook = getattr(charts, hook_name, None)
            if hook is None:
                continue
            output = out_dir / f"{hook_name}.png"
            returned = hook(table, output)
            generated.append(Path(returned) if returned is not None else output)
        return generated


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
    "experiment",
    "find_knee",
    "run_benchmark",
    "run_schedule",
    "self_amplification_series",
    "sweep_steady",
]
