"""Hand-built metric series for docs/evaluation-spec.md §7 E1 validity checks."""

import numpy as np
import pandas as pd
from harnext_eval.e1.calibration import calibration_spearman, decile_rates, lift_over_rules
from harnext_eval.e1.score import (
    affiliation_precision_recall,
    always_flag_sanity_scorer,
    nab_low_fn_score,
    precision_at_budget,
    random_sanity_scorer,
    recall_at_budget,
    timestamped_affiliation_precision_recall,
    vus_pr,
)


def test_point_metrics_have_known_answers() -> None:
    labels = [1, 0, 1, 0]
    admitted = [True, False, False, True]
    assert recall_at_budget(labels, admitted) == 0.5
    assert precision_at_budget(labels, admitted) == 0.5
    assert np.isnan(precision_at_budget(labels, [False] * len(labels)))


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


def test_timestamped_affiliation_uses_elapsed_time_and_situation_entities() -> None:
    situations = pd.DataFrame(
        [
            {"situation_id": "a", "entity": "one", "onset": "2026-01-01T00:00:00Z", "end": "2026-01-01T00:00:10Z"},
            {"situation_id": "b", "entity": "two", "onset": "2026-01-01T00:00:00Z", "end": "2026-01-01T00:00:10Z"},
        ]
    )
    exact = pd.DataFrame(
        [
            {"entity": entity, "time": f"2026-01-01T00:00:{second:02d}Z", "admitted": True}
            for entity in ("one", "two")
            for second in (0, 5, 10)
        ]
    )
    assert timestamped_affiliation_precision_recall(situations, exact) == (1.0, 1.0)

    late = exact.copy()
    late.loc[late["entity"] == "one", "time"] = "2026-01-01T00:01:00Z"
    late_precision, late_recall = timestamped_affiliation_precision_recall(situations, late)
    assert late_precision < 1.0
    assert late_recall < 1.0

    wrong_entity = exact[exact["entity"] == "one"].copy()
    _, missing_recall = timestamped_affiliation_precision_recall(situations, wrong_entity)
    assert missing_recall == 0.5
