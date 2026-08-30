"""Open-loop runner tests for docs/evaluation-spec.md §7 E6."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pandas as pd
import pytest
from harnext_eval.config import load_config
from harnext_eval.corpus.synthetic import generate_synthetic_events
from harnext_eval.e6.loadgen import fit_workload
from harnext_eval.e6.run import E6Experiment, RunnerConfig, calibration_run, sweep_steady
from harnext_eval.registry import ExperimentResult


def test_in_process_runner_writes_histograms_and_finds_tiny_knee(tmp_path: Path) -> None:
    cfg = load_config("apps/eval/configs/e6-twolane.yaml").engine
    fit = fit_workload(
        generate_synthetic_events(seed=2, event_count=160, days=1, entity_count=5)
    )

    async def exercise():
        return await sweep_steady(
            fit,
            cfg,
            rates_hz=[30.0, 120.0],
            duration_s=0.06,
            runner_cfg=RunnerConfig(
                partitions=2,
                workers=1,
                service_time_s=0.001,
            ),
            repetitions=3,
            lane_design="single",
            out_dir=tmp_path / "hdr",
            seed=8,
        )

    started = time.monotonic()
    rows, knee, artifacts = asyncio.run(exercise())

    assert time.monotonic() - started < 5.0
    assert len(rows) == 6
    assert knee in {30.0, 120.0}
    assert artifacts
    assert all(path.suffix == ".hgrm" and path.stat().st_size > 0 for path in artifacts)


def test_ten_millisecond_calibration() -> None:
    cfg = load_config("apps/eval/configs/e6-twolane.yaml").engine
    result = asyncio.run(calibration_run(cfg, samples=12))

    assert result.metrics["fast_p50_s"] == pytest.approx(0.010, abs=0.005)
    assert result.metrics["service_time_constant"] == 1.0


def test_chart_hooks_accept_e6_native_tables(tmp_path: Path) -> None:
    result = ExperimentResult(
        name="e6",
        tables={
            "burst_slo": pd.DataFrame(
                [{"lane_design": "two-lane", "load": 1.5, "urgent_slo_attainment": 1.0}]
            ),
            "self_amplification": pd.DataFrame(
                [
                    {
                        "bucket_s": 0.0,
                        "fast_admission_rate_hz": 10.0,
                        "urgent_slo_attainment": 1.0,
                    }
                ]
            ),
            "demand": pd.DataFrame(
                [{"load": 1.5, "partitions": 8, "workers": 4}]
            ),
        },
    )

    charts = E6Experiment().chart(result, tmp_path)

    assert len(charts) == 3
    assert all(path.is_file() and path.stat().st_size > 0 for path in charts)
