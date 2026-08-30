"""Hand-built metric series for docs/evaluation-spec.md §7 E1 validity checks."""

import numpy as np
from harnext_eval.e1.calibration import calibration_spearman, decile_rates, lift_over_rules
from harnext_eval.e1.score import (
    affiliation_precision_recall,
    always_flag_sanity_scorer,
    nab_low_fn_score,
    precision_at_budget,
    random_sanity_scorer,
    recall_at_budget,
    vus_pr,
)


def test_point_metrics_have_known_answers() -> None:
    labels = [1, 0, 1, 0]
    admitted = [True, False, False, True]
    assert recall_at_budget(labels, admitted) == 0.5
    assert precision_at_budget(labels, admitted) == 0.5


def test_vus_and_affiliation_toy_series() -> None:
    labels = [0, 1, 1, 0, 0, 1, 0]
    perfect_scores = [0.1, 0.9, 0.8, 0.2, 0.0, 0.7, 0.3]
    predictions = [False, True, True, False, False, True, False]
    assert vus_pr(labels, perfect_scores, max_buffer=0) == 1.0
    assert vus_pr(labels, [1.0] * len(labels), max_buffer=0) == np.mean(labels)
    precision, recall = affiliation_precision_recall(labels, predictions)
    assert precision == 1.0
    assert recall == 1.0
    assert affiliation_precision_recall(labels, [False] * len(labels)) == (0.0, 0.0)
    assert nab_low_fn_score(labels, predictions) == 1.0


def test_random_and_always_flag_sanity_floors() -> None:
    labels = ([1] * 200) + ([0] * 800)
    random = random_sanity_scorer(labels, budget_pct=10, seed=7, repeats=500)
    always = always_flag_sanity_scorer(labels)
    assert abs(random.precision - 0.2) < 0.02
    assert abs(random.vus_pr - 0.2) < 0.02
    assert always.recall == 1.0
    assert always.precision == 0.2


def test_calibration_and_lift_known_answers() -> None:
    scores = list(range(100))
    labels = [0] * 50 + [1] * 50
    curve = decile_rates(scores, labels)
    assert calibration_spearman(curve) > 0.85
    rules = [False] * 100
    admitted = [False] * 50 + [True] * 50
    assert lift_over_rules(labels, admitted, rules) == 2.0
