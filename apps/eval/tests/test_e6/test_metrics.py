"""Literal formula tests for evaluation spec §7 E6 and §12 D10."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from harnext_eval.e6.metrics import (
    demand_curve,
    drain_time,
    duplicates_and_missed,
    histogram_percentile,
    make_histogram,
    partition_lag_gini,
    recovery_time,
    slo_attainment,
    slo_gap,
)


def test_slo_attainment_and_empty_gold_is_invalid() -> None:
    latencies = [0.1, 2.0, 2.001, float("inf"), 9.0]
    urgent = [True, True, True, True, False]
    assert slo_attainment(latencies, urgent) == pytest.approx(0.5)
    assert slo_gap([0.1, 0.2], [0.1, 3.0]) == pytest.approx(0.5)
    assert math.isnan(slo_attainment([]))
    assert math.isnan(slo_gap([], []))


def test_reported_percentiles_are_read_from_hdrhistogram() -> None:
    histogram = make_histogram([0.001, 0.002, 0.003, 1.0])
    assert histogram.get_total_count() == 4
    assert histogram_percentile(histogram, 50) == pytest.approx(0.002, abs=2e-6)
    assert histogram_percentile(histogram, 99) == pytest.approx(1.0, abs=0.001)
    assert histogram_percentile(histogram, 99.9) == pytest.approx(1.0, abs=0.001)


def test_exact_drain_recovery_and_per_lane_delivery_counts() -> None:
    times = [0, 1, 2, 3, 4, 5]
    lags = [2, 2, 8, 5, 2, 1]
    assert drain_time(times, lags, burst_end_s=2, baseline_lag=2) == 2
    assert recovery_time(times, lags, kill_s=1, pre_kill_lag=2) == 3

    delivery = duplicates_and_missed(
        {"fast": ["a", "b"], "batch": ["c", "d"]},
        {"fast": ["a", "a"], "batch": ["c", "d", "d"]},
    )
    assert delivery.duplicates_by_lane == {"batch": 1, "fast": 1}
    assert delivery.missed_by_lane == {"batch": 0, "fast": 1}


def test_demand_groups_conditions_and_requires_complete_d10() -> None:
    common = {
        "lane_design": "two-lane",
        "shape": "anomalous_burst",
        "load": 1.5,
        "entity_cardinality": 32,
        "urgent_slo_attainment": 1.0,
        "batch_p99_s": 100.0,
        "lag_slope": 0.0,
    }
    results = pd.DataFrame(
        [
            {**common, "partitions": 8, "workers": 1, "fast_p99_s": 2.1},
            {**common, "partitions": 8, "workers": 4, "fast_p99_s": 1.0},
            {**common, "partitions": 32, "workers": 1, "fast_p99_s": 1.5},
            {
                **common,
                "lane_design": "single",
                "partitions": 8,
                "workers": 4,
                "fast_p99_s": 1.0,
                "batch_p99_s": 301.0,
            },
        ]
    )
    demand = demand_curve(results)
    two = demand[demand["lane_design"] == "two-lane"].iloc[0]
    single = demand[demand["lane_design"] == "single"].iloc[0]
    assert two["partitions"] == 8
    assert two["workers"] == 4
    assert bool(two["meets_slo"])
    assert not bool(single["meets_slo"])
    assert math.isnan(single["partitions"])


def test_partition_gini_literal_cases() -> None:
    assert partition_lag_gini([0, 0, 10, 10]) == pytest.approx(0.5)
    assert partition_lag_gini([1, 1, 1, 1]) == pytest.approx(0.0)
