"""End-to-end and seam tests for evaluation spec §7 E6."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from harnext_eval.config import EngineConfig, load_config
from harnext_eval.corpus.synthetic import generate_synthetic_events
from harnext_eval.e1.policies import GuardedHBOSPolicy
from harnext_eval.e6.loadgen import (
    ScheduleEntry,
    fit_workload,
    generate_steady_schedule,
)
from harnext_eval.e6.run import (
    BenchmarkConfig,
    E6Experiment,
    E6RouterPolicy,
    GuardOverrides,
    PipelineRun,
    RunnerConfig,
    calibration_run,
    calibration_valid,
    external_records_to_run,
    find_knee,
    run_benchmark,
    run_schedule,
    sweep_steady,
)
from harnext_eval.registry import ExperimentResult
from harnext_eval.types import EvalEvent
from PIL import Image


class AlwaysCandidate:
    name = "test-always-candidate"

    def rules(self, event: EvalEvent) -> str | None:
        del event
        return None

    def score(self, event: EvalEvent) -> float:
        del event
        return 1.0


class NeverCandidate(AlwaysCandidate):
    name = "test-never-candidate"

    def score(self, event: EvalEvent) -> float:
        del event
        return -1e12


def _cfg(*, budget_pct: float = 2.0) -> EngineConfig:
    base = load_config("apps/eval/configs/e6-twolane.yaml").engine
    return base.model_copy(
        update={
            "router": base.router.model_copy(update={"budget_pct": budget_pct}),
        }
    )


def _events(count: int = 120, entities: int = 6) -> list[EvalEvent]:
    return generate_synthetic_events(seed=2, event_count=count, days=1, entity_count=entities)


def test_find_knee_has_one_exact_answer_and_no_fictitious_fallback() -> None:
    rows = pd.DataFrame(
        [
            {"rate_hz": 30, "fast_p99_s": 0.2, "p99_trend_s_per_bucket": 0.01, "lag_slope": 0.0},
            {"rate_hz": 60, "fast_p99_s": 0.3, "p99_trend_s_per_bucket": -0.01, "lag_slope": 0.01},
            {"rate_hz": 120, "fast_p99_s": 2.0, "p99_trend_s_per_bucket": 0.2, "lag_slope": 1.0},
        ]
    )
    assert find_knee(rows, p99_trend_tolerance=0.02, lag_slope_threshold=0.05) == 60
    rows["p99_trend_s_per_bucket"] = -0.2
    assert find_knee(rows, p99_trend_tolerance=0.02, lag_slope_threshold=0.05) is None


def test_steady_sweep_repeats_and_writes_hdr(tmp_path: Path) -> None:
    fit = fit_workload(_events(80, 5))

    async def exercise():
        return await sweep_steady(
            fit,
            _cfg(),
            rates_hz=[30.0, 120.0],
            duration_s=0.03,
            runner_cfg=RunnerConfig(partitions=2, workers=1, service_time_s=0.0),
            repetitions=3,
            lane_design="single",
            out_dir=tmp_path / "latency_hdr",
            seed=8,
            lag_slope_threshold=10,
        )

    rows, knee, artifacts = asyncio.run(exercise())
    assert len(rows) == 6
    assert knee in {30.0, 120.0}
    assert set(rows["repetition"]) == {0, 1, 2}
    assert artifacts
    assert all(path.suffix == ".hgrm" and path.stat().st_size > 0 for path in artifacts)


def test_router_never_receives_gold_and_enforces_every_causal_prefix() -> None:
    cfg = _cfg(budget_pct=10)
    template = _events(1, 1)[0]
    router = E6RouterPolicy(
        cfg,
        lane_design="two-lane",
        policy=AlwaysCandidate(),
        guard_overrides=GuardOverrides(
            absolute_floor=0, multi_window=False, situation_dedup=False
        ),
    )
    fast = 0
    for index in range(100):
        event = template.model_copy(update={"id": f"budget-{index}"})
        lane, details = router.route(event, float(index))
        fast += lane == "fast"
        assert fast <= math.floor((index + 1) * 0.10)
        assert "urgent" not in details
    assert fast == 10


def test_two_lane_uses_real_r5_threshold_and_its_guard_object() -> None:
    cfg = _cfg(budget_pct=100)
    event = _events(1, 1)[0].model_copy(
        update={"baseline_keys": ["entity:test"], "data": {"body": "ordinary"}}
    )
    r5 = GuardedHBOSPolicy(absolute_floor=0, multi_window=False, budget_pct=100)
    r5.threshold = 0.5  # An unfitted R5 has raw score 0.0.
    router = E6RouterPolicy(
        cfg,
        policy=r5,
        guard_overrides=GuardOverrides(
            absolute_floor=0, multi_window=False, situation_dedup=False
        ),
    )
    assert isinstance(router.policy, GuardedHBOSPolicy)
    below_lane, below_details = router.route(event, 0)
    assert below_details["deviation_candidate"] is False
    assert below_lane == "batch"

    router.policy.threshold = -0.5
    above = event.model_copy(update={"id": "above-r5-threshold"})
    lane, details = router.route(above, 1)
    assert details["deviation_candidate"] is True
    assert details["r5_eligible"] is True
    assert lane == "fast"

    registered = E6RouterPolicy(cfg, training_events=_events(20, 2))
    assert isinstance(registered.policy, GuardedHBOSPolicy)
    assert registered.policy.absolute_floor >= 2
    assert registered.policy.multi_window is True


def test_rule_admissions_are_gated_by_total_prefix_budget_and_overflow_recorded() -> None:
    cfg = _cfg(budget_pct=10)
    template = _events(1, 1)[0].model_copy(update={"data": {"priority": "Critical"}})
    router = E6RouterPolicy(cfg)
    fast = 0
    for index in range(20):
        event = template.model_copy(update={"id": f"rule-{index}"})
        lane, details = router.route(event, float(index))
        fast += lane == "fast"
        assert details["rule"] == "declared_priority"
        assert fast <= math.floor((index + 1) * 0.10)
    assert fast == 2
    assert router.rule_admitted == 2
    assert router.rule_overflow == 18
    assert router.admitted == 2


def test_each_guard_ablation_is_individually_non_vacuous() -> None:
    cfg = _cfg(budget_pct=100)
    template = _events(1, 1)[0].model_copy(update={"baseline_keys": ["entity:test"]})

    full_floor = E6RouterPolicy(
        cfg,
        policy=AlwaysCandidate(),
        guard_overrides=GuardOverrides(
            absolute_floor=2, multi_window=False, situation_dedup=False
        ),
    )
    no_floor = E6RouterPolicy(
        cfg,
        policy=AlwaysCandidate(),
        guard_overrides=GuardOverrides(
            absolute_floor=0, multi_window=False, situation_dedup=False
        ),
    )
    assert full_floor.route(template, 0)[0] == "batch"
    assert no_floor.route(template, 0)[0] == "fast"

    full_confirmation = E6RouterPolicy(
        cfg,
        policy=AlwaysCandidate(),
        guard_overrides=GuardOverrides(
            absolute_floor=0, multi_window=True, situation_dedup=False
        ),
    )
    no_confirmation = E6RouterPolicy(
        cfg,
        policy=AlwaysCandidate(),
        guard_overrides=GuardOverrides(
            absolute_floor=0, multi_window=False, situation_dedup=False
        ),
    )
    assert full_confirmation.route(template, 0)[0] == "batch"
    assert no_confirmation.route(template, 0)[0] == "fast"
    assert full_confirmation.route(template.model_copy(update={"id": "next"}), 31)[0] == "fast"

    full_dedup = E6RouterPolicy(
        cfg,
        policy=AlwaysCandidate(),
        guard_overrides=GuardOverrides(
            absolute_floor=0, multi_window=False, situation_dedup=True
        ),
    )
    no_dedup = E6RouterPolicy(
        cfg,
        policy=AlwaysCandidate(),
        guard_overrides=GuardOverrides(
            absolute_floor=0, multi_window=False, situation_dedup=False
        ),
    )
    second = template.model_copy(update={"id": "same-situation"})
    assert [full_dedup.route(event, offset)[0] for event, offset in ((template, 1), (second, 2))] == [
        "fast",
        "batch",
    ]
    assert [no_dedup.route(event, offset)[0] for event, offset in ((template, 1), (second, 2))] == [
        "fast",
        "fast",
    ]


def test_partition_queues_are_subject_keyed_and_workers_change_capacity() -> None:
    fit = fit_workload(_events(120, 12))
    schedule = generate_steady_schedule(fit, duration_s=0.05, rate_hz=800, seed=3)

    async def exercise(workers: int):
        return await run_schedule(
            schedule,
            _cfg(),
            RunnerConfig(
                partitions=8,
                workers=workers,
                service_time_s=0.002,
                batch_windows=False,
                warmup_fraction=0,
            ),
            lane_design="single",
        )

    one = asyncio.run(exercise(1))
    four = asyncio.run(exercise(4))
    frame = pd.DataFrame([row.__dict__ for row in four.observations])
    assert frame.groupby("entity")["partition"].nunique().max() == 1
    assert frame["partition"].nunique() > 1
    assert four.metrics["fast_p99_s"] < one.metrics["fast_p99_s"]
    assert 0 <= four.metrics["partition_lag_gini"] <= 1


def test_hdr_percentiles_use_declared_burst_window_only() -> None:
    fit = fit_workload(_events(40, 3))
    start = datetime(2035, 1, 1, tzinfo=UTC)
    schedule = generate_steady_schedule(
        fit, duration_s=0.05, rate_hz=100, start=start, seed=4
    )
    result = asyncio.run(
        run_schedule(
            schedule,
            _cfg(),
            RunnerConfig(
                partitions=2,
                workers=1,
                service_time_s=0,
                batch_windows=False,
                burst_start_s=0.019,
                burst_end_s=0.021,
            ),
            lane_design="single",
        )
    )
    assert result.histograms["fast"].get_total_count() == 1


def test_batch_close_commit_and_staleness_use_window_deadline() -> None:
    template = _events(1, 1)[0]
    start = datetime(2035, 1, 1, tzinfo=UTC)
    schedule = [
        ScheduleEntry(
            start + timedelta(seconds=index * 0.01),
            template.subject,
            template.model_copy(
                update={
                    "id": f"batch-{index}",
                    "time": start + timedelta(seconds=index * 0.01),
                    "intended_send_ts": start + timedelta(seconds=index * 0.01),
                }
            ),
        )
        for index in range(2)
    ]
    result = asyncio.run(
        run_schedule(
            schedule,
            _cfg(budget_pct=100),
            RunnerConfig(service_time_s=0.001, window_time_scale=0.001, warmup_fraction=0),
            lane_design="two-lane",
            router_policy=NeverCandidate(),
        )
    )
    assert all(row.lane == "batch" for row in result.observations)
    assert result.observations[0].window_close_offset_s == pytest.approx(0.04, abs=0.004)
    assert result.observations[0].commit_offset_s >= result.observations[0].window_close_offset_s
    assert result.metrics["batch_staleness_p99_s"] >= 0.03


def test_urgent_batch_event_is_a_fast_slo_miss() -> None:
    template = _events(1, 1)[0]
    start = datetime(2035, 1, 1, tzinfo=UTC)
    event = template.model_copy(
        update={"id": "urgent-batch", "time": start, "intended_send_ts": start}
    )
    result = asyncio.run(
        run_schedule(
            [ScheduleEntry(start, event.subject, event, urgent=True, situation_id="urgent-1")],
            _cfg(budget_pct=100),
            RunnerConfig(
                service_time_s=0.001,
                window_time_scale=0.0001,
                warmup_fraction=0,
                deterministic_clock=True,
            ),
            lane_design="two-lane",
            router_policy=NeverCandidate(),
        )
    )
    observation = result.observations[0]
    assert observation.lane == "batch"
    assert observation.agent_start_offset_s == observation.service_start_offset_s
    assert math.isinf(observation.latency_s)
    assert result.metrics["urgent_denominator"] == 1
    assert result.metrics["urgent_slo_attainment"] == 0


def test_single_lane_is_fifo_per_event_with_same_fast_service_model() -> None:
    template = _events(1, 1)[0]
    start = datetime(2035, 1, 1, tzinfo=UTC)
    schedule = [
        ScheduleEntry(
            start + timedelta(milliseconds=index),
            template.subject,
            template.model_copy(
                update={
                    "id": f"fifo-{index}",
                    "time": start + timedelta(milliseconds=index),
                    "intended_send_ts": start + timedelta(milliseconds=index),
                }
            ),
        )
        for index in range(3)
    ]
    runtime = RunnerConfig(
        workers=2,
        service_time_s=0.002,
        deterministic_clock=True,
        warmup_fraction=0,
    )
    single = asyncio.run(run_schedule(schedule, _cfg(budget_pct=100), runtime, lane_design="single"))
    two = asyncio.run(
        run_schedule(
            schedule,
            _cfg(budget_pct=100),
            runtime,
            lane_design="two-lane",
            router_policy=AlwaysCandidate(),
            guard_overrides=GuardOverrides(
                absolute_floor=0, multi_window=False, situation_dedup=False
            ),
        )
    )
    assert single.metrics["service_invocations"] == len(schedule)
    assert two.metrics["service_invocations"] == len(schedule)
    assert [row.service_order for row in single.observations] == [0, 1, 2]
    assert all(row.window_close_offset_s is None for row in single.observations)
    assert single.metrics["service_time_p50_s"] == two.metrics["service_time_p50_s"]


def test_calibration_checks_median_and_p99_flatness() -> None:
    good = asyncio.run(calibration_run(_cfg(), samples=12))
    assert calibration_valid(good)
    broken = PipelineRun(
        observations=[],
        histograms={},
        lag_samples=pd.DataFrame(),
        metrics={"fast_p50_s": 0.010, "p99_trend_s_per_bucket": 0.02},
    )
    assert not calibration_valid(broken)


def test_fixture_backed_kafka_schema_counts_unique_ids_and_telemetry() -> None:
    fit = fit_workload(_events(40, 3))
    schedule = generate_steady_schedule(
        fit,
        duration_s=0.03,
        rate_hz=100,
        start=datetime(2035, 1, 1, tzinfo=UTC),
        seed=1,
    )
    schedule = [replace(row, urgent=True, situation_id=f"s-{index}") for index, row in enumerate(schedule)]
    records: list[dict[str, object]] = []
    for index, entry in enumerate(schedule[:2]):
        start = entry.intended_send_ts + timedelta(milliseconds=10)
        record = {
            "event_id": entry.event_id,
            "lane": "fast" if index == 0 else "batch",
            "agent_start_ts": start.isoformat(),
            "service_start_ts": start.isoformat(),
            "service_end_ts": (start + timedelta(milliseconds=2)).isoformat(),
            "partition": index,
            "partition_lag": 2 - index,
        }
        if index == 1:
            record["window_close_ts"] = entry.intended_send_ts.isoformat()
            record["commit_ts"] = (entry.intended_send_ts + timedelta(milliseconds=20)).isoformat()
        records.append(record)
    records.append(dict(records[0]))
    result = external_records_to_run(
        schedule,
        records,
        send_skews_s=[0.0001, 0.0002, 0.0001],
        telemetry=[
            {"cpu_pct": 40, "disk_util_pct": 50, "disk_io_bytes": 100},
            {"cpu_pct": 60, "disk_util_pct": 70, "disk_io_bytes": 200},
        ],
    )
    assert result.metrics["duplicates_fast"] == 1
    assert result.metrics["missed_unknown"] == 1
    assert result.metrics["broker_cpu_peak_pct"] == 60
    assert result.metrics["broker_disk_util_peak_pct"] == 70
    assert result.metrics["broker_disk_io_bytes"] == 300
    assert result.metrics["broker_telemetry_present"] == 1
    assert result.metrics["urgent_slo_attainment"] == pytest.approx(1 / 3)


def test_bounded_benchmark_has_paired_matrix_and_exact_outputs(tmp_path: Path) -> None:
    fit = fit_workload(_events(60, 5))
    settings = BenchmarkConfig(
        reference_rate_hz=40,
        steady_duration_s=0.02,
        burst_duration_s=0.03,
        burst_window_s=0.02,
        repetitions=3,
        service_time_s=0,
        topologies=((8, 1), (32, 4)),
        shapes=("poisson", "anomalous_burst"),
        burst_loads=(1.5,),
        entity_cardinalities=(8,),
        bootstrap_resamples=100,
        runner=RunnerConfig(
            service_time_s=0, window_time_scale=0.0001, deterministic_clock=True
        ),
    )
    result = asyncio.run(
        run_benchmark(fit, _cfg(), tmp_path, seed=4, benchmark=settings)
    )
    burst = result.tables["burst_slo"]
    assert len(burst) == 2 * 3 * 2 * 2
    assert set(burst["lane_design"]) == {"two-lane", "single"}
    assert set(zip(burst["partitions"], burst["workers"], strict=True)) == {
        (8, 1),
        (32, 4),
    }
    assert set(burst["repetition"]) == {0, 1, 2}
    assert (burst["urgent_denominator"] > 0).all()
    assert (burst[burst["lane_design"] == "two-lane"]["budget_compliant"] == 1).all()
    assert set(result.tables["gap_by_b"]["shape"]) == {"poisson", "anomalous_burst"}
    assert (result.tables["gap_by_b"]["bootstrap_resamples"] == 100).all()
    assert result.tables["paired_urgent_events"]["schedule_id"].notna().all()
    for name in (
        "knee.csv",
        "guards.csv",
        "burst_slo.png",
        "self_amplification.png",
        "demand_curve.png",
        "workload_parameters.json",
        "support_status.json",
    ):
        assert (tmp_path / name).is_file()
    assert list((tmp_path / "latency_hdr").glob("*.hgrm"))
    support = json.loads((tmp_path / "support_status.json").read_text())
    assert support["kafka_transport"] == "supported-not-run"
    applicable_checks = {
        name: value
        for name, value in result.metrics.items()
        if name.startswith("checks.") and math.isfinite(value)
    }
    assert applicable_checks
    assert all(bool(value) for value in applicable_checks.values())
    assert result.primary["valid"] is True
    assert result.primary["invalid_reasons"] == []


def test_invalid_validity_check_gates_primary_and_annotates_headline_chart(
    tmp_path: Path,
) -> None:
    fit = fit_workload(_events(40, 4))
    settings = BenchmarkConfig(
        reference_rate_hz=40,
        steady_duration_s=0.01,
        burst_duration_s=0.02,
        burst_window_s=0.01,
        repetitions=3,
        service_time_s=0,
        topologies=((8, 1),),
        shapes=("anomalous_burst",),
        burst_loads=(1.5,),
        entity_cardinalities=(4,),
        bootstrap_resamples=100,
        runner=RunnerConfig(
            service_time_s=0,
            service_time_tolerance_s=-1,
            window_time_scale=0.0001,
            deterministic_clock=True,
        ),
    )
    result = asyncio.run(run_benchmark(fit, _cfg(), tmp_path, seed=3, benchmark=settings))
    assert result.primary["valid"] is False
    assert "service_time_constant" in result.primary["invalid_reasons"]
    assert result.check_details["primary_validity_gate"]["passed"] is False
    with Image.open(tmp_path / "burst_slo.png") as chart:
        assert chart.getpixel((1, 1)) == (254, 226, 226)


def test_chart_hooks_use_exact_output_names(tmp_path: Path) -> None:
    result = ExperimentResult(
        name="e6",
        tables={
            "burst_slo": pd.DataFrame(
                [{"lane_design": "two-lane", "load": 1.5, "urgent_slo_attainment": 1.0}]
            ),
            "self_amplification": pd.DataFrame(
                [{"bucket_s": 0.0, "fast_admission_rate_hz": 10.0, "urgent_slo_attainment": 1.0}]
            ),
            "demand": pd.DataFrame([{"load": 1.5, "partitions": 8, "workers": 4}]),
        },
    )
    charts = E6Experiment().chart(result, tmp_path)
    assert {path.name for path in charts} == {
        "burst_slo.png",
        "self_amplification.png",
        "demand_curve.png",
    }
