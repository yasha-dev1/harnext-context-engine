"""Metric tests for docs/evaluation-spec.md §7 E6 and §12 D10."""

from __future__ import annotations

import pandas as pd
import pytest
from harnext_eval.e6.metrics import (
    demand_curve,
    partition_lag_gini,
    slo_attainment,
    slo_gap,
)


def test_slo_attainment_on_hand_built_latencies() -> None:
    latencies = [0.1, 2.0, 2.001, float("inf"), 9.0]
    urgent = [True, True, True, True, False]

    assert slo_attainment(latencies, urgent) == pytest.approx(0.5)
    assert slo_gap([0.1, 0.2], [0.1, 3.0]) == pytest.approx(0.5)


def test_demand_and_partition_gini() -> None:
    results = pd.DataFrame(
        [
            {"load": 1.0, "partitions": 8, "workers": 1, "fast_p99_s": 2.1, "lag_slope": 0.0},
            {"load": 1.0, "partitions": 8, "workers": 4, "fast_p99_s": 1.0, "lag_slope": -0.1},
            {"load": 1.0, "partitions": 32, "workers": 1, "fast_p99_s": 1.5, "lag_slope": -0.1},
        ]
    )

    demand = demand_curve(results)
    assert demand.iloc[0]["partitions"] == 8
    assert demand.iloc[0]["workers"] == 4
    assert bool(demand.iloc[0]["meets_slo"])
    assert partition_lag_gini([0, 0, 10, 10]) == pytest.approx(0.5)
