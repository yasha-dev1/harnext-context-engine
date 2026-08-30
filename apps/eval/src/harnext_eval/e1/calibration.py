"""Score calibration diagnostics for docs/evaluation-spec.md §7 E1 and D13."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def decile_rates(
    scores: Sequence[float], labels: Sequence[float], *, bins: int = 10
) -> pd.DataFrame:
    """Return equal-count score bins and revealed-urgency rate, low to high."""

    values = np.asarray(scores, dtype=float)
    outcomes = np.asarray(labels, dtype=float)
    if len(values) != len(outcomes):
        raise ValueError("scores and labels must have equal length")
    if bins <= 0:
        raise ValueError("bins must be positive")
    if not len(values):
        return pd.DataFrame(columns=["decile", "n", "score_min", "score_max", "urgency_rate"])
    order = np.argsort(values, kind="stable")
    assigned = np.empty(len(values), dtype=int)
    for bucket, indices in enumerate(np.array_split(order, min(bins, len(values))), start=1):
        assigned[indices] = bucket
    rows = []
    for bucket in sorted(np.unique(assigned)):
        selected = assigned == bucket
        rows.append(
            {
                "decile": int(bucket),
                "n": int(selected.sum()),
                "score_min": float(np.min(values[selected])),
                "score_max": float(np.max(values[selected])),
                "urgency_rate": float(np.mean(outcomes[selected] >= 0.5)),
            }
        )
    return pd.DataFrame(rows)


def calibration_spearman(calibration: pd.DataFrame) -> float:
    """Spearman rho between score-bin order and urgency rate."""

    if len(calibration) < 2 or calibration["urgency_rate"].nunique() < 2:
        return float("nan")
    statistic = spearmanr(calibration["decile"], calibration["urgency_rate"]).statistic
    return float(statistic)


def lift_over_rules(
    labels: Sequence[float], deviation_admitted: Sequence[bool], rule_flags: Sequence[bool]
) -> float:
    """Positive rate among deviation admissions divided by all rule negatives."""

    outcomes = np.asarray(labels, dtype=float) >= 0.5
    deviation = np.asarray(deviation_admitted, dtype=bool)
    rules = np.asarray(rule_flags, dtype=bool)
    if not (len(outcomes) == len(deviation) == len(rules)):
        raise ValueError("all inputs must have equal length")
    rule_negative = ~rules
    selected = deviation & rule_negative
    if not selected.any() or not rule_negative.any():
        return float("nan")
    baseline = float(outcomes[rule_negative].mean())
    return float(outcomes[selected].mean() / baseline) if baseline > 0 else float("inf")
