"""E1 ranking and temporal scores from docs/evaluation-spec.md §7 E1.

`vus_pr` ports the buffer-integrated RangeAUC-volume PR algorithm published by
Paparrizos et al.  `affiliation_precision_recall` implements Huet et al.'s
affiliation-zone probability normalisation.  Both are dependency-free ports of
the authors' reference algorithms so offline runs and exact fixture tests use
the same implementation.
"""

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.integrate import quad


def _arrays(y_true: Sequence[float], admitted: Sequence[bool]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true, dtype=float)
    decisions = np.asarray(admitted, dtype=bool)
    if labels.ndim != 1 or decisions.ndim != 1 or len(labels) != len(decisions):
        raise ValueError("labels and decisions must be equal-length one-dimensional arrays")
    return labels, decisions


def recall_at_budget(y_true: Sequence[float], admitted: Sequence[bool]) -> float:
    labels, decisions = _arrays(y_true, admitted)
    positive = labels >= 0.5
    return float(np.sum(positive & decisions) / positive.sum()) if positive.any() else float("nan")


def precision_at_budget(y_true: Sequence[float], admitted: Sequence[bool]) -> float:
    labels, decisions = _arrays(y_true, admitted)
    return float(np.mean(labels[decisions] >= 0.5)) if decisions.any() else float("nan")


def scores_by_source(
    frame: pd.DataFrame,
    *,
    label_col: str = "label",
    admitted_col: str = "admitted",
    source_col: str = "source",
    rule_col: str = "rule_flag",
) -> pd.DataFrame:
    """Recall/precision per source on full and primary rule-negative populations."""

    rows: list[dict[str, float | str | int]] = []
    sources = ["all", *sorted(frame[source_col].dropna().astype(str).unique())]
    for source in sources:
        source_frame = frame if source == "all" else frame[frame[source_col] == source]
        for population, subset in (
            ("full", source_frame),
            ("rule_negative", source_frame[~source_frame[rule_col].astype(bool)]),
        ):
            rows.append(
                {
                    "source": source,
                    "population": population,
                    "n": len(subset),
                    "recall": recall_at_budget(subset[label_col], subset[admitted_col]),
                    "precision": precision_at_budget(subset[label_col], subset[admitted_col]),
                }
            )
    return pd.DataFrame(rows)


def _coordinate_axis(
    length: int, timestamps: Sequence[object] | None,
) -> np.ndarray:
    """Return elapsed-time coordinates in median-cadence units.

    Regular timestamped input is therefore numerically identical to the
    published discrete reference, while real gaps are not collapsed into
    adjacent rows merely because no event happened between them.
    """

    if timestamps is None:
        return np.arange(length, dtype=float)
    parsed = pd.to_datetime(list(timestamps), utc=True)
    if len(parsed) != length:
        raise ValueError("timestamps must have the same length as labels")
    raw = parsed.asi8.astype(float) / 1_000_000_000.0
    if len(raw) > 1 and np.any(np.diff(raw) < 0):
        raise ValueError("timestamps must be non-decreasing")
    positive_steps = np.diff(raw)
    positive_steps = positive_steps[positive_steps > 0]
    cadence = float(np.median(positive_steps)) if len(positive_steps) else 1.0
    return (raw - raw[0]) / cadence if len(raw) else raw


def _buffered_soft_labels(
    binary: np.ndarray,
    ranges: list[tuple[int, int]],
    coordinates: np.ndarray,
    window: float,
) -> np.ndarray:
    """Reference square-root transition buffer, half on either range side."""

    soft = binary.astype(float).copy()
    if window <= 0:
        return soft
    half = window / 2.0
    for start, end in ranges:
        left_distance = coordinates[start] - coordinates
        right_distance = coordinates - coordinates[end]
        distance = np.where(coordinates < coordinates[start], left_distance, right_distance)
        outside = (coordinates < coordinates[start]) | (coordinates > coordinates[end])
        buffered = outside & (distance > 0) & (distance <= half)
        soft[buffered] += np.sqrt(np.maximum(0.0, 1.0 - distance[buffered] / window))
    return np.minimum(soft, 1.0)


def _buffered_index_ranges(
    ranges: list[tuple[int, int]], coordinates: np.ndarray, window: float
) -> list[tuple[int, int]]:
    """Return merged reference range boundaries after a symmetric buffer."""

    if not ranges:
        return []
    half = window / 2.0
    expanded: list[tuple[int, int]] = []
    for start, end in ranges:
        left = int(np.searchsorted(coordinates, coordinates[start] - half, side="left"))
        right = int(np.searchsorted(coordinates, coordinates[end] + half, side="right") - 1)
        if expanded and left <= expanded[-1][1] + 1:
            expanded[-1] = (expanded[-1][0], max(expanded[-1][1], right))
        else:
            expanded.append((left, right))
    return expanded


def _range_pr_area(
    binary: np.ndarray,
    scores: np.ndarray,
    ranges: list[tuple[int, int]],
    coordinates: np.ndarray,
    window: float,
    *,
    thresholds: int,
) -> float:
    """One RangeAUC-PR slice, algebraically equivalent to the reference loop."""

    soft = _buffered_soft_labels(binary, ranges, coordinates, window)
    buffered_ranges = _buffered_index_ranges(ranges, coordinates, window)
    original_positives = float(binary.sum())
    if original_positives <= 0 or not buffered_ranges:
        return float("nan")

    sorted_scores = np.sort(scores)[::-1]
    sampled = np.linspace(0, len(scores) - 1, thresholds).astype(int)
    threshold_values = sorted_scores[sampled]
    range_maxima = np.asarray(
        [float(np.max(scores[start : end + 1])) for start, end in buffered_ranges]
    )
    recalls = np.zeros(len(threshold_values), dtype=float)
    precisions = np.ones(len(threshold_values), dtype=float)
    for index, threshold in enumerate(threshold_values):
        predicted = scores >= threshold
        predicted_count = int(predicted.sum())
        true_positive = float(np.dot(soft, predicted))
        buffered_credit = float(np.dot(soft - binary, predicted))
        # The reference recomputes N_labels after masking transition-buffer
        # labels by the prediction while restoring every original anomaly.
        effective_positives = original_positives + buffered_credit / 2.0
        existence_ratio = float(np.mean(range_maxima >= threshold))
        recalls[index] = min(true_positive / effective_positives, 1.0) * existence_ratio
        precisions[index] = true_positive / predicted_count
    return float(np.dot(np.diff(np.r_[0.0, recalls]), precisions))


def vus_pr(
    y_true: Sequence[float],
    scores: Sequence[float],
    *,
    max_buffer: int = 0,
    n_buffers: int | None = None,
    timestamps: Sequence[object] | None = None,
    thresholds: int = 250,
) -> float:
    """Paparrizos VUS-PR over range buffers from zero through ``max_buffer``.

    The default 250 threshold slices and square-root boundary weighting match
    the authors' ``RangeAUC_volume_opt`` reference implementation.
    """

    labels = np.asarray(y_true, dtype=float)
    values = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or values.ndim != 1 or len(labels) != len(values):
        raise ValueError("labels and scores must be equal-length vectors")
    if max_buffer < 0:
        raise ValueError("max_buffer must be non-negative")
    if thresholds <= 0:
        raise ValueError("thresholds must be positive")
    if not len(labels):
        return float("nan")
    binary = (labels >= 0.5).astype(int)
    ranges = _intervals(binary)
    if not ranges:
        return float("nan")
    coordinates = _coordinate_axis(len(labels), timestamps)
    if n_buffers is None:
        radii = np.arange(max_buffer + 1, dtype=int)
    else:
        if n_buffers <= 0:
            raise ValueError("n_buffers must be positive")
        radii = np.unique(np.linspace(0, max_buffer, n_buffers).round().astype(int))
    areas = [
        _range_pr_area(
            binary,
            values,
            ranges,
            coordinates,
            float(radius),
            thresholds=thresholds,
        )
        for radius in radii
    ]
    finite = [area for area in areas if np.isfinite(area)]
    return float(np.mean(finite)) if finite else float("nan")


def _intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            result.append((start, index - 1))
            start = None
    if start is not None:
        result.append((start, len(mask) - 1))
    return result


type Interval = tuple[float, float]


def _affiliation_zones(intervals: list[Interval], time_range: Interval) -> list[Interval]:
    return [
        (
            time_range[0] if index == 0 else (intervals[index - 1][1] + interval[0]) / 2.0,
            time_range[1]
            if index == len(intervals) - 1
            else (interval[1] + intervals[index + 1][0]) / 2.0,
        )
        for index, interval in enumerate(intervals)
    ]


def _clip_interval(interval: Interval, zone: Interval) -> Interval | None:
    clipped = (max(interval[0], zone[0]), min(interval[1], zone[1]))
    return clipped if clipped[0] < clipped[1] else None


def _integrate_piecewise(
    function: Callable[[float], float], start: float, end: float
) -> float:
    """Integrate the paper's piecewise-affine distance reward."""

    if end <= start:
        return 0.0
    value, _ = quad(function, start, end, epsabs=1e-9, epsrel=1e-9, limit=100)
    return float(value)


def affiliation_pr_from_events(
    predicted_events: Sequence[Interval],
    ground_truth_events: Sequence[Interval],
    time_range: Interval,
) -> tuple[float, float]:
    """Huet affiliation P/R for continuous half-open event intervals.

    The probability rewards are the closed-form distance CDFs from the paper.
    Equal weighting across ground-truth affiliation zones matches the authors'
    reference implementation.
    """

    origin = float(time_range[0])
    ground_truth = sorted(
        (float(a) - origin, float(b) - origin) for a, b in ground_truth_events
    )
    predicted = sorted(
        (float(a) - origin, float(b) - origin) for a, b in predicted_events
    )
    time_range = (0.0, float(time_range[1]) - origin)
    if not ground_truth:
        raise ValueError("affiliation requires at least one ground-truth event")
    if any(start >= end for start, end in [*ground_truth, *predicted]):
        raise ValueError("affiliation events must have positive duration")
    if time_range[0] > ground_truth[0][0] or time_range[1] < ground_truth[-1][1]:
        raise ValueError("time_range must contain every ground-truth event")
    zones = _affiliation_zones(ground_truth, time_range)
    precision_parts: list[float] = []
    recall_parts: list[float] = []
    for truth, zone in zip(ground_truth, zones, strict=True):
        affiliated = [
            clipped for interval in predicted if (clipped := _clip_interval(interval, zone))
        ]
        zone_length = zone[1] - zone[0]
        predicted_length = sum(end - start for start, end in affiliated)
        if predicted_length:
            def precision_reward(
                point: float,
                truth: Interval = truth,
                zone: Interval = zone,
                zone_length: float = zone_length,
            ) -> float:
                if truth[0] <= point <= truth[1]:
                    return 1.0
                distance = max(truth[0] - point, 0.0, point - truth[1])
                other_margin = min(truth[0] - zone[0], zone[1] - truth[1])
                return max(
                    0.0,
                    1.0
                    - ((truth[1] - truth[0]) + distance + min(distance, other_margin))
                    / zone_length,
                )

            precision_parts.append(
                sum(
                    _integrate_piecewise(precision_reward, start, end)
                    for start, end in affiliated
                )
                / predicted_length
            )
        if not affiliated:
            recall_parts.append(0.0)
            continue

        def recall_reward(
            point: float,
            affiliated: list[Interval] = affiliated,
            zone: Interval = zone,
            zone_length: float = zone_length,
        ) -> float:
            distance = min(
                max(start - point, 0.0, point - end) for start, end in affiliated
            )
            if distance == 0:
                return 1.0
            random_nearer = min(distance, point - zone[0]) + min(
                distance, zone[1] - point
            )
            return max(0.0, 1.0 - random_nearer / zone_length)

        recall_parts.append(
            _integrate_piecewise(recall_reward, truth[0], truth[1])
            / (truth[1] - truth[0])
        )
    precision = float(np.mean(precision_parts)) if precision_parts else float("nan")
    return precision, float(np.mean(recall_parts))


def affiliation_precision_recall(
    y_true: Sequence[float], predictions: Sequence[bool]
) -> tuple[float, float]:
    """Huet affiliation P/R using the reference vector-to-event convention."""

    labels, predicted = _arrays(y_true, predictions)
    ground_truth = [(float(start), float(end + 1)) for start, end in _intervals(labels >= 0.5)]
    if not ground_truth:
        return (float("nan"), float("nan"))
    predicted_events = [
        (float(start), float(end + 1)) for start, end in _intervals(predicted)
    ]
    return affiliation_pr_from_events(predicted_events, ground_truth, (0.0, float(len(labels))))


def timestamped_affiliation_precision_recall(
    situations: pd.DataFrame,
    admissions: pd.DataFrame,
    *,
    entity_col: str = "entity",
    onset_col: str = "onset",
    end_col: str = "end",
    time_col: str = "time",
    admitted_col: str = "admitted",
) -> tuple[float, float]:
    """Affiliation P/R over timestamped, entity-separated situation intervals.

    Voronoi affiliation zones and rewards are calculated in elapsed seconds.
    This preserves real distance and never merges distinct entities or
    separately injected situations into one row-index interval.
    """

    if situations.empty:
        return float("nan"), float("nan")
    precision_parts: list[float] = []
    recall_parts: list[float] = []
    for entity, entity_situations in situations.groupby(entity_col, sort=True):
        entity_rows = pd.DataFrame(
            admissions.loc[admissions[entity_col] == entity, :].copy()
        )
        entity_rows["_affiliation_time"] = pd.to_datetime(
            entity_rows[time_col], utc=True
        )
        entity_rows = entity_rows.sort_values("_affiliation_time")
        entity_fast = entity_rows[entity_rows[admitted_col].astype(bool)]
        starts = pd.to_datetime(entity_situations[onset_col], utc=True)
        ends = pd.to_datetime(entity_situations[end_col], utc=True).fillna(starts)
        prediction_times = pd.to_datetime(entity_fast[time_col], utc=True)
        all_times = pd.to_datetime(entity_rows[time_col], utc=True)
        raw_times = sorted(
            {timestamp.timestamp() for timestamp in [*starts, *ends, *prediction_times, *all_times]}
        )
        steps = np.diff(raw_times)
        tick = float(np.min(steps[steps > 0])) if np.any(steps > 0) else 1.0
        truth = sorted(
            (start.timestamp(), max(end.timestamp() + tick, start.timestamp() + tick))
            for start, end in zip(starts, ends, strict=True)
        )
        prediction_events = _merge_intervals(
            [(timestamp.timestamp(), timestamp.timestamp() + tick) for timestamp in prediction_times]
        )
        left = min(raw_times) if raw_times else truth[0][0]
        right = max(raw_times) + tick if raw_times else truth[-1][1]
        precision, recall = affiliation_pr_from_events(prediction_events, truth, (left, right))
        if math.isfinite(precision):
            precision_parts.append(precision)
        recall_parts.append(recall)
    return (
        float(np.mean(precision_parts))
        if precision_parts else float("nan"),
        float(np.mean(recall_parts))
        if recall_parts
        else float("nan"),
    )


def _merge_intervals(intervals: Sequence[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def detection_delays(
    situations: pd.DataFrame,
    admissions: pd.DataFrame,
    *,
    entity_col: str = "entity",
    onset_col: str = "onset",
    time_col: str = "time",
    admitted_col: str = "admitted",
) -> pd.DataFrame:
    """Return onset-to-first-admission delay for each situation."""

    rows: list[dict[str, object]] = []
    fast = admissions[admissions[admitted_col].astype(bool)].copy()
    for index, situation in situations.iterrows():
        candidates = fast[
            (fast[entity_col] == situation[entity_col]) & (fast[time_col] >= situation[onset_col])
        ]
        if "end" in situation and pd.notna(situation["end"]):
            candidates = candidates[candidates[time_col] <= situation["end"]]
        first = candidates[time_col].min() if not candidates.empty else pd.NaT
        delay = (first - situation[onset_col]).total_seconds() if pd.notna(first) else float("nan")
        rows.append(
            {
                "situation_id": situation.get("situation_id", index),
                entity_col: situation[entity_col],
                "onset": situation[onset_col],
                "first_admission": first,
                "delay_s": delay,
            }
        )
    return pd.DataFrame(rows)


def delay_summary(delays: Sequence[float]) -> dict[str, float]:
    values = np.asarray(delays, dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "p50_s": float(np.quantile(finite, 0.5)) if len(finite) else float("nan"),
        "p95_s": float(np.quantile(finite, 0.95)) if len(finite) else float("nan"),
        "detected_rate": float(len(finite) / len(values)) if len(values) else float("nan"),
    }


def nab_low_fn_score(
    y_true: Sequence[float], predictions: Sequence[bool], *, window: int = 5
) -> float:
    """NAB-like early reward using the low-FN profile (FP=-0.11, FN=-2)."""

    labels, predicted = _arrays(y_true, predictions)
    intervals = _intervals(labels >= 0.5)
    used: set[int] = set()
    score = 0.0
    for start, end in intervals:
        early_start = max(0, start - window)
        candidates = [index for index in np.flatnonzero(predicted) if early_start <= index <= end]
        if candidates:
            first = int(candidates[0])
            used.add(first)
            if first <= start:
                score += 1.0
            else:
                score += max(0.0, 1.0 - (first - start) / max(end - start + 1, 1))
        else:
            score -= 2.0
    false_positives = sum(
        int(index) not in used and labels[int(index)] < 0.5 for index in np.flatnonzero(predicted)
    )
    score -= 0.11 * false_positives
    perfect = max(len(intervals), 1)
    return float(score / perfect)


@dataclass(frozen=True)
class SanityScores:
    prevalence: float
    precision: float
    recall: float
    vus_pr: float


def random_sanity_scorer(
    y_true: Sequence[float],
    *,
    budget_pct: float,
    seed: int = 0,
    repeats: int = 200,
    max_buffer: int = 5,
) -> SanityScores:
    labels = np.asarray(y_true, dtype=float)
    count = max(1, int(round(len(labels) * budget_pct / 100.0))) if len(labels) else 0
    rng = np.random.default_rng(seed)
    precision: list[float] = []
    recall: list[float] = []
    vus: list[float] = []
    for _ in range(repeats):
        values = rng.random(len(labels))
        admitted = np.zeros(len(labels), dtype=bool)
        if count:
            admitted[np.argsort(-values)[:count]] = True
        precision.append(precision_at_budget(labels, admitted))
        recall.append(recall_at_budget(labels, admitted))
        vus.append(vus_pr(labels, values, max_buffer=max_buffer))
    return SanityScores(
        prevalence=float(np.mean(labels >= 0.5)) if len(labels) else float("nan"),
        precision=float(np.nanmean(precision)),
        recall=float(np.nanmean(recall)),
        vus_pr=float(np.nanmean(vus)),
    )


def always_flag_sanity_scorer(
    y_true: Sequence[float], *, max_buffer: int = 5
) -> SanityScores:
    labels = np.asarray(y_true, dtype=float)
    admitted = np.ones(len(labels), dtype=bool)
    prevalence = float(np.mean(labels >= 0.5)) if len(labels) else float("nan")
    return SanityScores(
        prevalence=prevalence,
        precision=precision_at_budget(labels, admitted),
        recall=recall_at_budget(labels, admitted),
        vus_pr=vus_pr(labels, np.ones(len(labels)), max_buffer=max_buffer),
    )


def flip_labels(y_true: Sequence[float], *, fraction: float = 0.1, seed: int = 0) -> np.ndarray:
    """Flip an exact seeded fraction for E1 label-noise robustness."""

    labels = (np.asarray(y_true, dtype=float) >= 0.5).astype(int)
    rng = np.random.default_rng(seed)
    count = int(round(len(labels) * fraction))
    if count:
        indices = rng.choice(len(labels), count, replace=False)
        labels[indices] = 1 - labels[indices]
    return labels


def jitter_onsets(
    onsets: Sequence[pd.Timestamp], *, minutes: float = 5.0, seed: int = 0
) -> pd.DatetimeIndex:
    """Uniformly jitter constructed situation onsets within ±minutes."""

    rng = np.random.default_rng(seed)
    offsets = rng.uniform(-minutes * 60, minutes * 60, len(onsets))
    return pd.DatetimeIndex(
        [
            pd.Timestamp(onset) + timedelta(seconds=float(offset))
            for onset, offset in zip(onsets, offsets, strict=True)
        ]
    )
