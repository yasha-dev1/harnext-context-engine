"""Pure E6 system metrics from docs/evaluation-spec.md §7 E6 and §12 D10."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

URGENT_SLO_S = 2.0
BATCH_SLO_S = 300.0


def slo_attainment(
    latencies_s: Sequence[float],
    urgent: Sequence[bool] | None = None,
    *,
    slo_s: float = URGENT_SLO_S,
) -> float:
    """Fraction of urgent events observed at or below the latency SLO."""

    values = np.asarray(latencies_s, dtype=float)
    if urgent is not None:
        mask = np.asarray(urgent, dtype=bool)
        if mask.size != values.size:
            raise ValueError("urgent mask and latencies must have equal length")
        values = values[mask]
    if values.size == 0:
        return 0.0
    return float(np.mean(np.isfinite(values) & (values <= slo_s)))


def slo_gap(
    two_lane_latencies_s: Sequence[float],
    single_lane_latencies_s: Sequence[float],
    *,
    slo_s: float = URGENT_SLO_S,
) -> float:
    """Compute ``gap(B) = SLO_att_two - SLO_att_single``."""

    return slo_attainment(two_lane_latencies_s, slo_s=slo_s) - slo_attainment(
        single_lane_latencies_s, slo_s=slo_s
    )


def percentile(latencies_s: Sequence[float], value: float) -> float:
    """Finite-only percentile, returning infinity when no observation exists."""

    values = np.asarray(latencies_s, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, value)) if values.size else float("inf")


def linear_trend(values: Sequence[float], times_s: Sequence[float] | None = None) -> float:
    """Least-squares slope, with zero for fewer than two points."""

    y = np.asarray(values, dtype=float)
    if y.size < 2:
        return 0.0
    x = np.arange(y.size, dtype=float) if times_s is None else np.asarray(times_s, dtype=float)
    if x.size != y.size:
        raise ValueError("times and values must have equal length")
    if np.all(x == x[0]):
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def partition_lag_gini(lags: Sequence[float]) -> float:
    """Gini coefficient of non-negative per-partition lag."""

    values = np.asarray(lags, dtype=float)
    if values.size == 0:
        return 0.0
    if np.any(values < 0):
        raise ValueError("partition lag cannot be negative")
    total = float(values.sum())
    if total == 0:
        return 0.0
    sorted_values = np.sort(values)
    indexes = np.arange(1, values.size + 1, dtype=float)
    return float(np.sum((2 * indexes - values.size - 1) * sorted_values) / (
        values.size * total
    ))


def _first_recovered_time(
    times_s: Sequence[float],
    lags: Sequence[float],
    *,
    after_s: float,
    target_lag: float,
) -> float:
    if len(times_s) != len(lags):
        raise ValueError("times and lags must have equal length")
    for timestamp, lag in zip(times_s, lags, strict=True):
        if timestamp >= after_s and lag <= target_lag:
            return max(0.0, float(timestamp - after_s))
    return float("inf")


def drain_time(
    times_s: Sequence[float],
    lags: Sequence[float],
    *,
    burst_end_s: float,
    baseline_lag: float,
) -> float:
    """Seconds from burst end until lag returns to baseline."""

    return _first_recovered_time(
        times_s, lags, after_s=burst_end_s, target_lag=baseline_lag
    )


def recovery_time(
    times_s: Sequence[float],
    lags: Sequence[float],
    *,
    kill_s: float,
    pre_kill_lag: float,
) -> float:
    """Seconds from worker kill until lag returns to its pre-kill level."""

    return _first_recovered_time(times_s, lags, after_s=kill_s, target_lag=pre_kill_lag)


@dataclass(frozen=True)
class DeliveryCounts:
    duplicates_by_lane: dict[str, int]
    missed_by_lane: dict[str, int]


def duplicates_and_missed(
    expected_by_lane: Mapping[str, Iterable[str]],
    observed_by_lane: Mapping[str, Iterable[str]],
) -> DeliveryCounts:
    """Count duplicate deliveries and expected IDs never observed, per lane."""

    duplicates: dict[str, int] = {}
    missed: dict[str, int] = {}
    for lane in sorted(set(expected_by_lane) | set(observed_by_lane)):
        expected = set(expected_by_lane.get(lane, ()))
        observed = Counter(observed_by_lane.get(lane, ()))
        duplicates[lane] = sum(max(0, count - 1) for count in observed.values())
        missed[lane] = len(expected - set(observed))
    return DeliveryCounts(duplicates, missed)


def cross_entity_fairness(
    rows: pd.DataFrame,
    hot_entities: Iterable[str],
    *,
    slo_s: float = URGENT_SLO_S,
) -> float:
    """Urgent-event SLO attainment on cold entities during a hot burst."""

    required = {"entity", "urgent", "latency_s"}
    if not required.issubset(rows.columns):
        raise ValueError(f"rows must contain {sorted(required)}")
    hot = set(hot_entities)
    cold = rows[(rows["urgent"].astype(bool)) & (~rows["entity"].isin(list(hot)))]
    return slo_attainment(cold["latency_s"].astype(float).tolist(), slo_s=slo_s)


def demand_curve(
    results: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    slo_s: float = URGENT_SLO_S,
) -> pd.DataFrame:
    """Return minimal ``(partitions, workers)`` meeting D10 at each load/cardinality."""

    frame = results.copy() if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    required = {"load", "partitions", "workers", "fast_p99_s", "lag_slope"}
    if not required.issubset(frame.columns):
        raise ValueError(f"results must contain {sorted(required)}")
    group_columns = ["load"] + (["entity_cardinality"] if "entity_cardinality" in frame else [])
    selected: list[pd.Series] = []
    for _, group in frame.groupby(group_columns, dropna=False, sort=True):
        candidates = cast(
            pd.DataFrame,
            group[(group["fast_p99_s"] <= slo_s) & (group["lag_slope"] <= 0)],
        )
        if candidates.empty:
            row = group.iloc[0].copy()
            row["partitions"] = np.nan
            row["workers"] = np.nan
            row["meets_slo"] = False
        else:
            # The preregistered topology grid treats demand as a lexicographic
            # resource tuple, minimizing partitions before worker replicas.
            candidates = candidates.sort_values("workers").sort_values(
                "partitions", kind="stable"
            )
            row = candidates.iloc[0].copy()
            row["meets_slo"] = True
        selected.append(row)
    return pd.DataFrame(selected).reset_index(drop=True)


# Specification notation aliases.
gap = slo_gap
demand = demand_curve
gini = partition_lag_gini


__all__ = [
    "BATCH_SLO_S",
    "URGENT_SLO_S",
    "DeliveryCounts",
    "cross_entity_fairness",
    "demand",
    "demand_curve",
    "drain_time",
    "duplicates_and_missed",
    "gap",
    "gini",
    "linear_trend",
    "partition_lag_gini",
    "percentile",
    "recovery_time",
    "slo_attainment",
    "slo_gap",
]
