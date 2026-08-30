"""Statistics required by docs/evaluation-spec.md §8 and §9."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import binom, norm


@dataclass(frozen=True)
class PairedDifferenceResult:
    """Mean paired effect and its entity-clustered BCa confidence interval."""

    effect: float
    ci_low: float
    ci_high: float
    confidence_level: float
    n_observations: int
    n_clusters: int
    n_resamples: int


@dataclass(frozen=True)
class McNemarResult:
    """McNemar test result for the two discordant cells."""

    b: int
    c: int
    statistic: float
    p_value: float
    exact_p_value: float
    mid_p_value: float


def _one_dimensional(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def paired_difference_bca(
    condition_a: ArrayLike,
    condition_b: ArrayLike,
    clusters: ArrayLike,
    *,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    random_state: int | np.random.Generator | None = None,
) -> PairedDifferenceResult:
    """Estimate ``mean(condition_a - condition_b)`` with a clustered BCa CI.

    The observations must be paired and ``clusters`` must contain the entity for
    each pair. A bootstrap draw samples whole entities with replacement and keeps
    every observation belonging to each selected entity. The BCa acceleration is
    calculated by leave-one-entity-out jackknife, so neither bootstrap nor
    jackknife treats repeated probes from one entity as independent.
    """

    a = _one_dimensional(condition_a, "condition_a").astype(float)
    b = _one_dimensional(condition_b, "condition_b").astype(float)
    entity = _one_dimensional(clusters, "clusters")
    if not (len(a) == len(b) == len(entity)):
        raise ValueError("paired conditions and clusters must have equal lengths")
    if len(a) == 0:
        raise ValueError("at least one paired observation is required")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("paired observations must be finite")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")

    unique_entities = pd.unique(entity)
    if len(unique_entities) < 2:
        raise ValueError("at least two entity clusters are required")
    cluster_rows = [np.flatnonzero(entity == value) for value in unique_entities]
    differences = a - b
    effect = float(np.mean(differences))
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    bootstrap = np.empty(n_resamples, dtype=float)
    cluster_count = len(cluster_rows)
    for draw_index in range(n_resamples):
        selected = rng.integers(0, cluster_count, size=cluster_count)
        rows = np.concatenate([cluster_rows[index] for index in selected])
        bootstrap[draw_index] = float(np.mean(differences[rows]))

    # Half the ties receive mass on either side, which avoids a biased z0 for
    # discrete correctness data while preserving the usual BCa definition.
    proportion_less = (
        np.count_nonzero(bootstrap < effect)
        + 0.5 * np.count_nonzero(bootstrap == effect)
    ) / n_resamples
    epsilon = 0.5 / n_resamples
    bias_correction = float(norm.ppf(np.clip(proportion_less, epsilon, 1 - epsilon)))

    jackknife = np.empty(cluster_count, dtype=float)
    for index, rows in enumerate(cluster_rows):
        keep = np.ones(len(differences), dtype=bool)
        keep[rows] = False
        jackknife[index] = float(np.mean(differences[keep]))
    jackknife_mean = float(np.mean(jackknife))
    centred = jackknife_mean - jackknife
    denominator = 6.0 * float(np.sum(centred**2)) ** 1.5
    acceleration = 0.0 if denominator == 0 else float(np.sum(centred**3)) / denominator

    alpha = (1.0 - confidence_level) / 2.0
    normal_quantiles = norm.ppf([alpha, 1.0 - alpha])
    adjusted: list[float] = []
    for quantile in normal_quantiles:
        shifted = bias_correction + quantile
        divisor = 1.0 - acceleration * shifted
        if math.isclose(divisor, 0.0, abs_tol=1e-15):
            adjusted.append(0.0 if shifted < 0 else 1.0)
        else:
            adjusted.append(float(norm.cdf(bias_correction + shifted / divisor)))
    low_q, high_q = np.clip(np.sort(adjusted), 0.0, 1.0)
    ci_low, ci_high = np.quantile(bootstrap, [low_q, high_q])

    return PairedDifferenceResult(
        effect=effect,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence_level=confidence_level,
        n_observations=len(differences),
        n_clusters=cluster_count,
        n_resamples=n_resamples,
    )


# A discoverable alias matching the wording used in the evaluation plan.
paired_clustered_bca = paired_difference_bca


def mcnemar_test(
    condition_a_or_table: ArrayLike,
    condition_b: ArrayLike | None = None,
    *,
    mid_p: bool = False,
) -> McNemarResult:
    """Run two-sided McNemar inference on paired correctness or a 2×2 table.

    With paired correctness vectors, ``b`` counts A-correct/B-wrong pairs and
    ``c`` counts A-wrong/B-correct pairs. With a table, rows are condition A and
    columns condition B, both ordered ``correct, wrong``; therefore the
    discordant cells are ``table[0, 1]`` and ``table[1, 0]``. ``p_value`` is the
    exact binomial p-value unless ``mid_p=True``.
    """

    if condition_b is None:
        table = np.asarray(condition_a_or_table)
        if table.shape != (2, 2):
            raise ValueError("a McNemar contingency table must have shape (2, 2)")
        if np.any(table < 0) or not np.all(table == np.floor(table)):
            raise ValueError("contingency counts must be non-negative integers")
        b = int(table[0, 1])
        c = int(table[1, 0])
    else:
        a = _one_dimensional(condition_a_or_table, "condition_a").astype(bool)
        paired_b = _one_dimensional(condition_b, "condition_b").astype(bool)
        if len(a) != len(paired_b):
            raise ValueError("paired correctness vectors must have equal lengths")
        b = int(np.count_nonzero(a & ~paired_b))
        c = int(np.count_nonzero(~a & paired_b))

    discordant = b + c
    if discordant == 0:
        exact = 1.0
        mid = 1.0
        statistic = 0.0
    else:
        smaller = min(b, c)
        exact = min(1.0, 2.0 * float(binom.cdf(smaller, discordant, 0.5)))
        mid = min(
            1.0,
            2.0 * float(binom.cdf(smaller - 1, discordant, 0.5))
            + float(binom.pmf(smaller, discordant, 0.5)),
        )
        statistic = float((b - c) ** 2 / discordant)
    return McNemarResult(
        b=b,
        c=c,
        statistic=statistic,
        p_value=mid if mid_p else exact,
        exact_p_value=exact,
        mid_p_value=mid,
    )


def holm_bonferroni(
    p_values: Mapping[str, float] | Sequence[float], *, alpha: float = 0.05
) -> pd.DataFrame:
    """Return Holm-adjusted p-values in ascending raw-p order.

    A mapping preserves hypothesis labels; a sequence is labelled by its
    zero-based position. Rejection is step-down: after the first failure, all
    larger p-values are retained. Adjusted p-values are monotone by construction.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if isinstance(p_values, Mapping):
        items = [(str(label), float(value)) for label, value in p_values.items()]
    else:
        items = [(str(index), float(value)) for index, value in enumerate(p_values)]
    if not items:
        return pd.DataFrame(
            columns=["hypothesis", "p_value", "adjusted_p", "threshold", "reject", "rank"]
        )
    if any(not np.isfinite(value) or not 0 <= value <= 1 for _, value in items):
        raise ValueError("p-values must be finite and lie in [0, 1]")

    ordered = sorted(items, key=lambda item: item[1])
    count = len(ordered)
    raw_adjusted = [(count - index) * value for index, (_, value) in enumerate(ordered)]
    adjusted = np.minimum(1.0, np.maximum.accumulate(raw_adjusted))
    still_rejecting = True
    rows: list[dict[str, object]] = []
    for index, ((label, value), adjusted_value) in enumerate(zip(ordered, adjusted, strict=True)):
        threshold = alpha / (count - index)
        reject = still_rejecting and value <= threshold
        if not reject:
            still_rejecting = False
        rows.append(
            {
                "hypothesis": label,
                "p_value": value,
                "adjusted_p": float(adjusted_value),
                "threshold": threshold,
                "reject": reject,
                "rank": index + 1,
            }
        )
    return pd.DataFrame(rows)


def paired_proportion_sample_size(
    *,
    effect: float = 0.10,
    discordant_rate: float = 0.20,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> int:
    """Approximate paired-proportion sample size for a McNemar contrast.

    ``effect`` is the absolute paired proportion difference and
    ``discordant_rate`` is P(A correct, B wrong) + P(A wrong, B correct). The
    defaults encode the §8 planning case and yield approximately 150 pairs.
    """

    _validate_power_inputs(effect, discordant_rate, alpha, power)
    critical = norm.ppf(1 - alpha / (2 if two_sided else 1))
    target = norm.ppf(power)
    alternative_variance = discordant_rate - effect**2
    numerator = critical * math.sqrt(discordant_rate) + target * math.sqrt(
        alternative_variance
    )
    return math.ceil((numerator / effect) ** 2)


def paired_proportion_power(
    n_pairs: int,
    *,
    effect: float = 0.10,
    discordant_rate: float = 0.20,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Approximate power of a paired-proportion (McNemar) comparison."""

    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    _validate_power_inputs(effect, discordant_rate, alpha, 0.8)
    critical = norm.ppf(1 - alpha / (2 if two_sided else 1))
    alternative_sd = math.sqrt(discordant_rate - effect**2)
    z_power = (
        math.sqrt(n_pairs) * effect - critical * math.sqrt(discordant_rate)
    ) / alternative_sd
    return float(norm.cdf(z_power))


def _validate_power_inputs(
    effect: float, discordant_rate: float, alpha: float, power: float
) -> None:
    if not 0 < effect < 1:
        raise ValueError("effect must lie strictly between zero and one")
    if not effect <= discordant_rate <= 1:
        raise ValueError("discordant_rate must be at least effect and at most one")
    if discordant_rate <= effect**2:
        raise ValueError("discordant_rate must exceed effect squared")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must lie strictly between zero and one")


def pass_k(
    correctness: float | ArrayLike,
    *,
    k: int = 3,
) -> float:
    """Compute pass^k: the share of tasks correct in all ``k`` runs.

    A scalar is treated as an independent per-run pass probability and returns
    ``p**k``. A 2-D boolean array is interpreted as task × run. A flat vector is
    interpreted as consecutive blocks of ``k`` runs per task.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if isinstance(correctness, (int, float, np.integer, np.floating)):
        probability = float(correctness)
        if not 0 <= probability <= 1:
            raise ValueError("a pass probability must lie in [0, 1]")
        return probability**k

    values = np.asarray(correctness, dtype=bool)
    if values.ndim == 1:
        if values.size == 0 or values.size % k:
            raise ValueError("flat correctness must contain complete blocks of k runs")
        values = values.reshape((-1, k))
    elif values.ndim != 2:
        raise ValueError("correctness must be scalar, one-dimensional, or two-dimensional")
    if values.shape[1] != k:
        raise ValueError("the run dimension must equal k")
    return float(np.mean(np.all(values, axis=1)))


def between_seed_spread(values: ArrayLike, *, ddof: int = 1) -> float:
    """Return the between-seed standard deviation for one metric."""

    array = _one_dimensional(values, "values").astype(float)
    if array.size <= ddof:
        raise ValueError("more seed values than ddof are required")
    if not np.all(np.isfinite(array)):
        raise ValueError("seed values must be finite")
    return float(np.std(array, ddof=ddof))
