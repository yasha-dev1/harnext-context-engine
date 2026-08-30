"""Offline rolling E1 runner regressions for docs/evaluation-spec.md §7 E1."""

from dataclasses import replace

import numpy as np
import pandas as pd
from harnext_eval.config import load_config
from harnext_eval.corpus.synthetic import generate_synthetic_corpus
from harnext_eval.e1.run import E1Experiment
from harnext_eval.registry import get_experiment


def test_e1_uses_sidecar_gold_global_monthly_admission_and_required_outputs(tmp_path) -> None:
    corpus = generate_synthetic_corpus(
        tmp_path / "replay.jsonl", seed=5, event_count=300, days=70, entity_count=10
    )
    corpus = replace(
        corpus,
        meta={
            **corpus.meta,
            "smoke": True,
            "harm_results": [
                {
                    "event_id": "fixture",
                    "quality_now": 0.8,
                    "quality_window_close": 0.5,
                    "harm_delta": 0.3,
                    "tokens": 0,
                    "dollars": 0.0,
                    "latency_s": 0.0,
                }
            ],
        },
    )
    cfg = load_config("apps/eval/configs/baseline-minimal.yaml").engine
    experiment = get_experiment("e1")
    assert isinstance(experiment, E1Experiment)
    out = tmp_path / "e1"
    result = experiment.run(cfg, corpus, out, seed=5)

    assert result.primary["metric"] == "recall_at_2pct_rule_negative"
    assert {"r5_minus_r1_ci_low", "r5_minus_r1_ci_high", "r5_minus_r2_ci_low", "r5_minus_r2_ci_high"} <= set(result.primary)
    assert set(result.tables["scores"]["policy"]) == {f"R{index}" for index in range(8)}

    exact_positive_ids = {
        item["event_id"] for item in corpus.meta["injected_situations"]
    }
    evaluated = result.tables["scores"]
    labelled_positive = set(evaluated.loc[evaluated["label"] >= 0.5, "event_id"])
    assert labelled_positive <= exact_positive_ids
    assert evaluated["constructed_label"].all()

    r0 = evaluated[
        (evaluated["policy"] == "R0") & (evaluated["population"] == "full")
    ]
    for (_, budget), group in r0.groupby(["month", "budget_pct"]):
        expected = max(1, int(round(len(group) * float(budget) / 100.0)))
        assert group["admitted"].sum() == expected
        assert group["event_id"].nunique() == len(group)

    r5 = evaluated[(evaluated["policy"] == "R5") & (evaluated["population"] == "full")]
    assert not r5.loc[r5["rule_flag"], "admitted"].eq(False).any()
    assert not r5.loc[~r5["eligible"] & ~r5["mandatory"], "admitted"].any()
    assert result.metrics["check.r5_rules_floor_preserved"] == 1.0
    assert result.metrics["check.r5_ineligible_never_admitted"] == 1.0
    assert result.metrics["check.tuning_precedes_evaluation"] == 1.0
    assert result.metrics["check.valid"] == 1.0

    metrics = result.tables["metrics"]
    assert "all" in set(metrics["source"])
    assert {"tokens", "dollars", "decision_latency_ms", "unused_capacity"} <= set(metrics)
    assert (metrics["tokens"] == 0).all()
    assert np.allclose(metrics["dollars"], 0.0)
    assert not result.tables["delays"].empty
    assert {"delay_p50_s", "jitter_delay_p50_s", "affiliation_recall"} <= set(
        result.tables["robustness"].columns
    )

    required = {
        "metrics.csv",
        "calibration.png",
        "operating_curves.png",
        "attribution.md",
        "harm.csv",
        "label_diagnostics.csv",
        "robustness.csv",
        "delays.csv",
        "preflight.csv",
    }
    assert required <= {path.name for path in out.iterdir()}
    assert "future human-analysis step" not in (out / "attribution.md").read_text()
    assert not pd.read_csv(out / "harm.csv").empty
    assert set(pd.read_csv(out / "preflight.csv")["status"]) <= {
        "run",
        "supported-not-run",
    }
    assert result.metrics["check.harm_supported"] == 1.0

    chart_paths = experiment.chart(result, out / "charts")
    assert {path.name for path in chart_paths} == {"calibration.png", "operating_curves.png"}
