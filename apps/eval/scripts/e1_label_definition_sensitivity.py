"""Registered post-hoc label-definition sensitivity for E1 (review finding 2).

Recomputes the primary metric (recall@2 % on the rule-negative population, per
policy) under the label definitions registered in ``harnext_eval.e1.prereg.LABEL_DEFINITIONS``
from a finished run's ``label_votes.parquet`` and ``scores.parquet``. No labelling
functions are re-applied and no policy is re-scored, so the table is a pure
re-reading of the run's own outputs. Read-only.

Run:
  UV_NO_SYNC=1 uv run python apps/eval/scripts/e1_label_definition_sensitivity.py \
      --run apps/eval/out/<run-id>-e1-kafka/e1/seed-1 --out apps/eval/reports/e1-kafka-run3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from harnext_eval.e1.labels import ABSTAIN, DEFAULT_LABELING_FUNCTIONS, POSITIVE, fit_label_model
from harnext_eval.e1.prereg import LABEL_DEFINITIONS, PRIMARY_CONTRASTS
from harnext_eval.stats.stats import paired_difference_bca

BUDGET = 2.0


def _definitions(votes: pd.DataFrame) -> dict[str, np.ndarray]:
    declared = {f.name for f in DEFAULT_LABELING_FUNCTIONS if f.declared}
    outcome_cols = [c for c in votes.columns if c not in declared]
    matrix = votes.to_numpy()
    positives = (matrix == POSITIVE).sum(1)
    fired = (matrix != ABSTAIN).sum(1)
    outcome_positive = (votes[outcome_cols].to_numpy() == POSITIVE).sum(1)
    registered = fit_label_model(votes).probabilities.to_numpy()
    table = {
        "weighted_model_p50": registered >= 0.5,
        "any_outcome_lf": outcome_positive >= 1,
        "two_outcome_lfs": outcome_positive >= 2,
        "half_of_votes": (fired > 0) & (2 * positives >= fired),
    }
    missing = set(LABEL_DEFINITIONS) - set(table)
    if missing:
        raise ValueError(f"registered label definitions without an implementation: {sorted(missing)}")
    return {name: table[name] for name in LABEL_DEFINITIONS}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path, help="…/e1/seed-<n> directory")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    votes = pd.read_parquet(args.run / "label_votes.parquet")
    definitions = _definitions(votes)
    labels = pd.DataFrame(definitions, index=votes.index)

    columns = ["event_id", "policy", "admitted", "rule_negative", "subject", "month"]
    scores = pq.read_table(
        args.run / "scores.parquet", columns=columns, filters=[("budget_pct", "=", BUDGET)]
    ).to_pandas()
    scores = scores[scores["rule_negative"].astype(bool)]
    scores["event_id"] = scores["event_id"].astype(str)
    joined = scores.merge(labels, left_on="event_id", right_index=True, how="inner")

    rows: list[dict[str, object]] = []
    contrasts: list[dict[str, object]] = []
    for definition in LABEL_DEFINITIONS:
        positive = joined[joined[definition].astype(bool)]
        pivot = positive.pivot_table(
            index=["event_id", "subject"], columns="policy", values="admitted", aggfunc="first", observed=True
        ).reset_index()
        row: dict[str, object] = {
            "label_definition": definition,
            "positives_rule_negative": int(pivot.shape[0]),
            "subjects": int(pivot["subject"].nunique()) if not pivot.empty else 0,
            "prevalence_rule_negative": float(len(pivot) / max(joined["event_id"].nunique(), 1)),
        }
        for policy in sorted(pivot.columns.drop(["event_id", "subject"])):
            row[f"recall_{policy}"] = float(pivot[policy].astype(float).mean())
        rows.append(row)
        for contrast in PRIMARY_CONTRASTS:
            left, right = contrast.split("-")
            if {left, right} <= set(pivot.columns) and pivot["subject"].nunique() >= 2:
                estimate = paired_difference_bca(
                    pivot[left].astype(float), pivot[right].astype(float), pivot["subject"],
                    n_resamples=10_000, random_state=args.seed,
                )
                contrasts.append(
                    {
                        "label_definition": definition, "contrast": contrast,
                        "effect": estimate.effect, "ci_low": estimate.ci_low, "ci_high": estimate.ci_high,
                        "n_clusters": estimate.n_clusters,
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "label_definition_recall.csv", index=False)
    pd.DataFrame(contrasts).to_csv(args.out / "label_definition_contrasts.csv", index=False)
    (args.out / "label_definition_sensitivity.json").write_text(
        json.dumps({"definitions": list(LABEL_DEFINITIONS), "budget_pct": BUDGET,
                    "rows": rows, "contrasts": contrasts}, indent=2, default=float),
        encoding="utf-8",
    )
    print(table.to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
