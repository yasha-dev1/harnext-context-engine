"""E1 ranking and temporal scores from docs/evaluation-spec.md §7 E1.

`vus_pr` is a deterministic discrete implementation of the VUS-PR construction
of Paparrizos et al.: PR area is computed for linearly buffered anomaly ranges
and averaged over buffer radii (the volume axis).  `affiliation_precision_recall`
implements the discrete-time affiliation-zone construction of Huet et al.: each
prediction belongs to the closest ground-truth interval and receives a
zone-normalised distance reward.  These implementations intentionally expose
small pure functions so reference toy series can pin their behavior.
"""

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd


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


def _soft_ranges(labels: np.ndarray, radius: int) -> np.ndarray:
    soft = (labels >= 0.5).astype(float)
    if radius <= 0:
        return soft
    positive_indices = np.flatnonzero(soft)
    for index in positive_indices:
        start = max(0, index - radius)
        stop = min(len(soft), index + radius + 1)
        positions = np.arange(start, stop)
        reward = 1.0 - np.abs(positions - index) / (radius + 1.0)
        soft[start:stop] = np.maximum(soft[start:stop], reward)
    return soft


def _weighted_average_precision(weights: np.ndarray, scores: np.ndarray) -> float:
    if not len(weights) or weights.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ordered_weights = weights[order]
    ordered_scores = scores[order]
    cumulative_tp = np.cumsum(ordered_weights)
    cumulative_n = np.arange(1, len(weights) + 1)
    precision = cumulative_tp / cumulative_n
    recall = cumulative_tp / weights.sum()
    boundaries = np.r_[ordered_scores[1:] != ordered_scores[:-1], True]
    precision = precision[boundaries]
    recall = recall[boundaries]
    delta = np.diff(np.r_[0.0, recall])
    return float(np.sum(delta * precision))


def vus_pr(
    y_true: Sequence[float],
    scores: Sequence[float],
    *,
    max_buffer: int = 0,
    n_buffers: int | None = None,
) -> float:
    """Volume under the PR surface over linearly tolerant range buffers."""

    labels = np.asarray(y_true, dtype=float)
    values = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or values.ndim != 1 or len(labels) != len(values):
        raise ValueError("labels and scores must be equal-length vectors")
    if max_buffer < 0:
        raise ValueError("max_buffer must be non-negative")
    if n_buffers is None:
        radii = np.arange(max_buffer + 1, dtype=int)
    else:
        if n_buffers <= 0:
            raise ValueError("n_buffers must be positive")
        radii = np.unique(np.linspace(0, max_buffer, n_buffers).round().astype(int))
    areas = [
        _weighted_average_precision(_soft_ranges(labels, int(radius)), values) for radius in radii
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


def _distance_to_interval(point: int, interval: tuple[int, int]) -> int:
    start, end = interval
    return max(start - point, 0, point - end)


def _zones(intervals: list[tuple[int, int]], length: int) -> list[tuple[int, int]]:
    zones: list[tuple[int, int]] = []
    for index, interval in enumerate(intervals):
        left = 0 if index == 0 else (intervals[index - 1][1] + interval[0]) // 2 + 1
        right = (
            length - 1
            if index == len(intervals) - 1
            else (interval[1] + intervals[index + 1][0]) // 2
        )
        zones.append((left, right))
    return zones


def affiliation_precision_recall(
    y_true: Sequence[float], predictions: Sequence[bool]
) -> tuple[float, float]:
    """Discrete Huet affiliation precision/recall with Voronoi affiliation zones."""

    labels, predicted = _arrays(y_true, predictions)
    ground_truth = _intervals(labels >= 0.5)
    if not ground_truth:
        return (0.0 if predicted.any() else 1.0), float("nan")
    if not predicted.any():
        return 0.0, 0.0
    zones = _zones(ground_truth, len(labels))
    predicted_points = np.flatnonzero(predicted)

    precision_rewards: list[float] = []
    for point in predicted_points:
        interval_index = min(
            range(len(ground_truth)),
            key=lambda index: (_distance_to_interval(int(point), ground_truth[index]), index),
        )
        interval = ground_truth[interval_index]
        zone = zones[interval_index]
        distance = _distance_to_interval(int(point), interval)
        normalizer = max(interval[0] - zone[0], zone[1] - interval[1], 1)
        precision_rewards.append(max(0.0, 1.0 - distance / normalizer))

    recall_rewards: list[float] = []
    for interval, zone in zip(ground_truth, zones, strict=True):
        affiliated = predicted_points[(predicted_points >= zone[0]) & (predicted_points <= zone[1])]
        for point in range(interval[0], interval[1] + 1):
            if len(affiliated):
                distance = int(np.min(np.abs(affiliated - point)))
                normalizer = max(point - zone[0], zone[1] - point, 1)
                recall_rewards.append(max(0.0, 1.0 - distance / normalizer))
            else:
                recall_rewards.append(0.0)
    return float(np.mean(precision_rewards)), float(np.mean(recall_rewards))


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
    fast = admissions[admissions[admitted_col].astype(bool)]
    for entity, entity_situations in situations.groupby(entity_col, sort=True):
        entity_fast = fast[fast[entity_col] == entity]
        starts = pd.to_datetime(entity_situations[onset_col], utc=True)
        ends = pd.to_datetime(entity_situations[end_col], utc=True).fillna(starts)
        prediction_times = pd.to_datetime(entity_fast[time_col], utc=True)
        intervals = sorted(
            [(start.timestamp(), end.timestamp()) for start, end in zip(starts, ends, strict=True)]
        )
        predictions = np.asarray([timestamp.timestamp() for timestamp in prediction_times])
        if not len(predictions):
            recall_parts.extend([0.0] * len(intervals))
            continue
        zones: list[tuple[float, float]] = []
        outer_left = min(intervals[0][0], float(predictions.min()))
        outer_right = max(intervals[-1][1], float(predictions.max()))
        for index, interval in enumerate(intervals):
            left = outer_left if index == 0 else (intervals[index - 1][1] + interval[0]) / 2
            right = outer_right if index == len(intervals) - 1 else (interval[1] + intervals[index + 1][0]) / 2
            zones.append((left, right))
        for prediction in predictions:
            interval_index = min(
                range(len(intervals)),
                key=lambda index: (
                    max(intervals[index][0] - prediction, 0, prediction - intervals[index][1]),
                    index,
                ),
            )
            start, end = intervals[interval_index]
            left, right = zones[interval_index]
            distance = max(start - prediction, 0, prediction - end)
            scale = max(start - left, right - end, 1.0)
            precision_parts.append(max(0.0, 1.0 - distance / scale))
        for (start, end), (left, right) in zip(intervals, zones, strict=True):
            affiliated = predictions[(predictions >= left) & (predictions <= right)]
            if not len(affiliated):
                recall_parts.append(0.0)
                continue
            anchors = np.asarray([start, (start + end) / 2, end])
            distance = np.min(np.abs(anchors[:, None] - affiliated[None, :]), axis=1)
            scale = np.maximum(np.maximum(anchors - left, right - anchors), 1.0)
            recall_parts.append(float(np.mean(np.maximum(0.0, 1.0 - distance / scale))))
    return (
        float(np.mean(precision_parts))
        if precision_parts
        else float("nan"),
        float(np.mean(recall_parts))
        if recall_parts
        else float("nan"),
    )


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
    y_true: Sequence[float], *, budget_pct: float, seed: int = 0, repeats: int = 200
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
        vus.append(vus_pr(labels, values))
    return SanityScores(
        prevalence=float(np.mean(labels >= 0.5)) if len(labels) else float("nan"),
        precision=float(np.nanmean(precision)),
        recall=float(np.nanmean(recall)),
        vus_pr=float(np.nanmean(vus)),
    )


def always_flag_sanity_scorer(y_true: Sequence[float]) -> SanityScores:
    labels = np.asarray(y_true, dtype=float)
    admitted = np.ones(len(labels), dtype=bool)
    prevalence = float(np.mean(labels >= 0.5)) if len(labels) else float("nan")
    return SanityScores(
        prevalence=prevalence,
        precision=precision_at_budget(labels, admitted),
        recall=recall_at_budget(labels, admitted),
        vus_pr=vus_pr(labels, np.ones(len(labels))),
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
