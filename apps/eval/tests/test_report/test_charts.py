"""Smoke tests for every required evaluation chart."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from harnext_eval.report.charts import (
    calibration,
    checks_table,
    demand_curve,
    e2_family_bars,
    e3_curve,
    e4_envelopes,
    e5_pareto,
    e6_burst_slo,
    erosion,
    health_table,
    operating_curves,
    self_amplification,
)


def _assert_png(path: Path) -> None:
    assert path.exists()
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert path.stat().st_size > 2_000


@pytest.mark.parametrize(
    ("name", "renderer"),
    [
        (
            "calibration",
            lambda out: calibration(
                pd.DataFrame(
                    {
                        "decile": [1, 2, 3],
                        "observed_rate": [0.1, 0.25, 0.55],
                        "predicted_rate": [0.12, 0.3, 0.5],
                    }
                ),
                out,
            ),
        ),
        (
            "operating",
            lambda out: operating_curves(
                {
                    "R1": ([1, 2, 5], [0.2, 0.35, 0.6], [0.7, 0.55, 0.4]),
                    "R5": ([1, 2, 5], [0.3, 0.5, 0.75], [0.8, 0.65, 0.5]),
                },
                out,
            ),
        ),
        (
            "e2",
            lambda out: e2_family_bars(
                pd.DataFrame(
                    {
                        "arm": ["A3", "A4", "A3", "A4"],
                        "family": ["extraction", "extraction", "update", "update"],
                        "accuracy": [0.6, 0.8, 0.5, 0.75],
                    }
                ),
                out,
            ),
        ),
        (
            "e3",
            lambda out: e3_curve(
                pd.DataFrame(
                    {
                        "store": ["S1", "S1", "S3", "S3"],
                        "budget": [2_000, 8_000, 2_000, 8_000],
                        "acc": [0.5, 0.62, 0.66, 0.8],
                        "ci_low": [0.45, 0.57, 0.61, 0.75],
                        "ci_high": [0.55, 0.67, 0.71, 0.85],
                    }
                ),
                out,
            ),
        ),
        (
            "erosion",
            lambda out: erosion(
                pd.DataFrame(
                    {
                        "store": ["S1", "S1", "S3", "S3"],
                        "checkpoint": [1, 8, 1, 8],
                        "acc": [0.62, 0.55, 0.78, 0.76],
                    }
                ),
                out,
            ),
        ),
        (
            "health",
            lambda out: health_table(
                pd.DataFrame(
                    {
                        "store": ["S1", "S1", "S3", "S3"],
                        "metric": ["index", "duplicates", "index", "duplicates"],
                        "value": [0.8, 0.12, 0.98, 0.03],
                    }
                ),
                out,
            ),
        ),
        (
            "e4",
            lambda out: e4_envelopes(
                pd.DataFrame(
                    {
                        "envelope": ["V1", "V3", "V6"],
                        "Q": [0.55, 0.79, 0.65],
                        "ci_low": [0.5, 0.74, 0.6],
                        "ci_high": [0.6, 0.84, 0.7],
                        "tokens": [4_000, 9_000, 30_000],
                    }
                ),
                out,
            ),
        ),
        (
            "e5",
            lambda out: e5_pareto(
                pd.DataFrame(
                    {
                        "cadence": ["W1", "W20", "W20+rules"],
                        "cost": [20.0, 5.0, 7.0],
                        "acc": [0.8, 0.76, 0.79],
                        "freshness": [0.1, 8.0, 2.0],
                    }
                ),
                out,
            ),
        ),
        (
            "e6",
            lambda out: e6_burst_slo(
                pd.DataFrame(
                    {
                        "lane_design": ["single", "single", "two-lane", "two-lane"],
                        "load": [1.0, 1.5, 1.0, 1.5],
                        "slo_attainment": [0.96, 0.7, 0.995, 0.97],
                    }
                ),
                out,
            ),
        ),
        (
            "amplification",
            lambda out: self_amplification(
                pd.DataFrame(
                    {
                        "time": [0, 1, 2, 3],
                        "admission_rate": [5, 8, 20, 7],
                        "slo_attainment": [1.0, 0.98, 0.75, 0.96],
                    }
                ),
                out,
            ),
        ),
        (
            "demand",
            lambda out: demand_curve(
                pd.DataFrame(
                    {
                        "load": [0.5, 1.0, 1.5],
                        "partitions": [8, 8, 32],
                        "workers": [1, 2, 4],
                    }
                ),
                out,
            ),
        ),
        (
            "checks",
            lambda out: checks_table(
                {
                    "leakage_gate": True,
                    "prior_floor": {"value": 0.2, "passed": True},
                    "judge_kappa": False,
                },
                out,
            ),
        ),
    ],
)
def test_chart_renders_png(tmp_path: Path, name: str, renderer: object) -> None:
    del name
    path = renderer(tmp_path)  # type: ignore[operator]
    _assert_png(path)
