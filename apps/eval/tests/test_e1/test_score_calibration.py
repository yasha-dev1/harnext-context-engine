"""Hand-built metric series for docs/evaluation-spec.md §7 E1 validity checks."""

import numpy as np
import pandas as pd
from harnext_eval.e1.calibration import calibration_spearman, decile_rates, lift_over_rules
from harnext_eval.e1.score import (
    affiliation_pr_from_events,
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


def test_vus_matches_three_paparrizos_reference_cases() -> None:
    # Values independently produced by the authors' RangeAUC_volume_opt
    # implementation with 250 thresholds and buffers 0, 1, 2.
    cases = (
        ([0, 1, 1, 0, 0, 1, 0], [0.1, 0.9, 0.8, 0.2, 0.0, 0.7, 0.3], 1.0),
        ([0, 1, 0, 0], [0.1, 0.8, 0.9, 0.2], 0.5923495156295323),
        ([0, 1, 1, 0], [1.0, 1.0, 1.0, 1.0], 0.617851130197758),
    )
    for labels, scores, expected in cases:
        assert vus_pr(labels, scores, max_buffer=2) == expected


def test_affiliation_matches_three_huet_reference_cases() -> None:
    # These are direct integrals over E=[0,6], J=[2,4].
    cases = (
        ([(2.0, 4.0)], (1.0, 1.0)),
        ([(3.0, 4.0)], (1.0, 11.0 / 12.0)),
        ([(4.0, 5.0)], (0.5, 2.0 / 3.0)),
    )
    for prediction, expected in cases:
        assert np.allclose(
            affiliation_pr_from_events(prediction, [(2.0, 4.0)], (0.0, 6.0)),
            expected,
        )

    labels = [0, 1, 1, 0, 0, 1, 0]
    predictions = [False, True, True, False, False, True, False]
    assert affiliation_precision_recall(labels, predictions) == (1.0, 1.0)
    empty_precision, empty_recall = affiliation_precision_recall(
        labels, [False] * len(labels)
    )
    assert np.isnan(empty_precision)
    assert empty_recall == 0.0
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

    probabilistic = decile_rates([0.0, 0.1, 0.9, 1.0], [0.1, 0.3, 0.6, 0.8], bins=2)
    assert probabilistic["urgency_rate"].tolist() == [0.2, 0.7]
    assert np.isclose(
        lift_over_rules(
            [0.1, 0.3, 0.6, 0.8], [False, False, True, True], rules[:4]
        ),
        0.7 / 0.45,
    )


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
