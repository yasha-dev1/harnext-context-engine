"""Tests for the statistics required by evaluation-spec §8."""

from __future__ import annotations

import numpy as np
import pytest
from harnext_eval.stats.stats import (
    between_seed_spread,
    holm_bonferroni,
    mcnemar_test,
    paired_difference_bca,
    paired_proportion_power,
    paired_proportion_sample_size,
    pass_k,
)


def test_clustered_bca_covers_known_effect_and_resamples_whole_clusters() -> None:
    rng = np.random.default_rng(42)
    cluster_effects = 0.20 + rng.normal(0, 0.08, size=30)
    clusters = np.repeat([f"entity-{index}" for index in range(30)], 5)
    baseline = rng.normal(0.55, 0.04, size=len(clusters))
    treatment = baseline + np.repeat(cluster_effects, 5)

    result = paired_difference_bca(
        treatment,
        baseline,
        clusters,
        n_resamples=2_000,
        random_state=7,
    )

    assert result.n_observations == 150
    assert result.n_clusters == 30
    assert result.effect == pytest.approx(np.mean(cluster_effects))
    assert result.ci_low < 0.20 < result.ci_high

    # Replicating rows inside every entity must not create new independent units.
    replicated = paired_difference_bca(
        np.repeat(treatment, 3),
        np.repeat(baseline, 3),
        np.repeat(clusters, 3),
        n_resamples=2_000,
        random_state=7,
    )
    assert replicated.n_clusters == 30
    assert replicated.effect == pytest.approx(result.effect)
    assert replicated.ci_low == pytest.approx(result.ci_low)
    assert replicated.ci_high == pytest.approx(result.ci_high)


def test_mcnemar_exact_and_mid_p_on_known_table() -> None:
    table = np.array([[50, 10], [2, 38]])
    exact = mcnemar_test(table)
    middle = mcnemar_test(table, mid_p=True)

    assert (exact.b, exact.c) == (10, 2)
    assert exact.exact_p_value == pytest.approx(0.03857421875)
    assert middle.p_value == pytest.approx(0.0224609375)
    assert middle.p_value < exact.p_value


def test_holm_orders_p_values_and_enforces_step_down_rejection() -> None:
    corrected = holm_bonferroni({"third": 0.04, "first": 0.001, "second": 0.03})

    assert corrected["hypothesis"].tolist() == ["first", "second", "third"]
    assert corrected["adjusted_p"].tolist() == pytest.approx([0.003, 0.06, 0.06])
    assert corrected["reject"].tolist() == [True, False, False]


def test_power_pass_k_and_seed_spread_helpers() -> None:
    required = paired_proportion_sample_size()
    assert 145 <= required <= 160
    assert paired_proportion_power(required) >= 0.8
    assert pass_k(0.8, k=3) == pytest.approx(0.512)
    assert pass_k([[True, True, True], [True, False, True]], k=3) == 0.5
    assert between_seed_spread([0.7, 0.8, 0.9]) == pytest.approx(0.1)
