"""Run-5 report: the Jev routing arms (R14 category router, R15 urgency) against the E1 baselines.

Reads the Jev-only run (5b: R0 R1 R2 R7 R14 R15) and run 4 (all other arms; same replay and
configuration, deterministic), pairs them on event id, and writes to ``--out``:

  recall_at_budget_rule_negative.csv / fig_recall_at_budget.png   pooled over rule-negative positives
  recall_by_source_2pct.csv, recall_by_year_2pct.csv               (year split = memorisation check)
  contrasts_2pct.csv          paired R14/R15 vs R0/R2/R10/R11 (subject and month clusters, BCa)
  jev_categories.csv          category mix of all events, of positives, of R14's admissions
  natural_operating_point.csv argmax-category routing without a budget
  results_summary.md

  UV_NO_SYNC=1 uv run --package harnext-eval python apps/eval/scripts/e1_jev_report.py \
      --run apps/eval/out/<ts>-e1-kafka-jev-only/e1/seed-1 \
      --run4 apps/eval/out/20260907T000315Z-e1-kafka/e1/seed-1 --out apps/eval/reports/e1-kafka-run5
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from harnext_eval.e1 import jev  # noqa: E402
from harnext_eval.stats.stats import paired_difference_bca  # noqa: E402

BUDGETS = (1.0, 2.0, 5.0, 10.0)
RUN4_ARMS = ("R4", "R5", "R10", "R11", "R12", "R13")
ORDER = ("R0", "R2", "R4", "R5", "R10", "R11", "R12", "R13", "R14", "R15")
LABEL = {
    "R0": "R0 random", "R2": "R2 global HBOS", "R4": "R4 per-entity HBOS", "R5": "R5 guarded HBOS",
    "R10": "R10 per-source HBOS", "R11": "R11 IsolationForest", "R12": "R12 ECOD", "R13": "R13 global LOF",
    "R14": "R14 Jev SE-category router", "R15": "R15 Jev urgency",
}
COLOR = {"R0": "#7a756d", "R2": "#2a78d6", "R4": "#1baf7a", "R5": "#eb6834", "R10": "#4a3aa7",
         "R11": "#e87ba4", "R12": "#8a5fbf", "R13": "#c98500", "R14": "#0b6e6e", "R15": "#b8336a"}
COLUMNS = ["event_id", "policy", "budget_pct", "admitted", "label", "rule_flag", "source", "subject", "month"]


def _positives(path: Path, policies: tuple[str, ...]) -> pd.DataFrame:
    """Rule-negative positive rows (label >= 0.5) of the given arms at every budget."""
    table = pq.read_table(
        path, columns=COLUMNS,
        filters=[("policy", "in", list(policies)), ("rule_flag", "=", False), ("label", ">=", 0.5)],
    )
    return table.to_pandas()


def _markdown(frame: pd.DataFrame) -> str:
    def cell(value) -> str:
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def _recall(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    grouped = frame.groupby([*by, "policy", "budget_pct"], observed=True)["admitted"]
    out = grouped.agg(recall="mean", admitted="sum", positives="size").reset_index()
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--run4", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    results = json.loads((args.run / "results.json").read_text())
    details = results.get("checks") or results.get("check_details") or {}
    jev_info = (details.get("jev_decisions") or {}).get("value") or {}
    run5 = _positives(args.run / "scores.parquet", ("R0", "R2", "R14", "R15"))
    run4 = _positives(args.run4 / "scores.parquet", RUN4_ARMS + ("R2",))
    # Same replay + config: run 4's R2 must reproduce run 5b's R2 exactly, or the cross-run join is void.
    r2_new = run5[run5["policy"] == "R2"].set_index(["event_id", "budget_pct"])["admitted"].sort_index()
    r2_old = run4[run4["policy"] == "R2"].set_index(["event_id", "budget_pct"])["admitted"].sort_index()
    r2_identical = bool(r2_new.index.equals(r2_old.index) and (r2_new.values == r2_old.values).all())
    frame = pd.concat([run5, run4[run4["policy"] != "R2"]], ignore_index=True)
    frame["year"] = frame["month"].astype(str).str[:4]
    for column in ("policy", "source", "subject", "month"):
        frame[column] = frame[column].astype(str)

    pooled = _recall(frame, [])
    pooled.to_csv(args.out / "recall_at_budget_rule_negative.csv", index=False)
    two = frame[frame["budget_pct"] == 2.0]
    by_source = _recall(two, ["source"])
    by_source.to_csv(args.out / "recall_by_source_2pct.csv", index=False)
    by_year = _recall(two, ["year"])
    by_year.to_csv(args.out / "recall_by_year_2pct.csv", index=False)

    pivot = two.pivot_table(index=["event_id", "subject", "month"], columns="policy", values="admitted",
                            aggfunc="first").reset_index()
    rows = []
    for left in ("R14", "R15"):
        for right in ("R0", "R2", "R10", "R11", "R15"):
            if left == right or left not in pivot or right not in pivot:
                continue
            a, b = pivot[left].astype(float), pivot[right].astype(float)
            subject = paired_difference_bca(a, b, pivot["subject"], n_resamples=10_000, random_state=1)
            month = paired_difference_bca(a, b, pivot["month"], n_resamples=10_000, random_state=1)
            rows.append({"contrast": f"{left}-{right}", "effect": subject.effect,
                         "ci_low_subject": subject.ci_low, "ci_high_subject": subject.ci_high,
                         "ci_low_month": month.ci_low, "ci_high_month": month.ci_high,
                         "n_positives": len(pivot), "n_subjects": subject.n_clusters,
                         "cross_run": right in RUN4_ARMS})
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(args.out / "contrasts_2pct.csv", index=False)

    # Category mix and the unbudgeted operating point, from the decision cache.
    decisions = jev.load_decisions(jev_info.get("path") or "apps/eval/out/e1-jev/se-triage-v2/scores.jsonl",
                                   jev_info.get("prompt_sha256") or jev.prompt_sha256())
    all_events = pq.read_table(args.run / "scores.parquet", columns=["event_id", "rule_flag", "label", "month", "admitted"],
                               filters=[("policy", "=", "R14"), ("budget_pct", "=", 2.0)]).to_pandas()
    all_events["category"] = all_events["event_id"].map(lambda i: decisions[i].category if i in decisions else None)
    all_events["urgency"] = all_events["event_id"].map(lambda i: decisions[i].urgency if i in decisions else np.nan)
    positive = (all_events["label"] >= 0.5) & (~all_events["rule_flag"].astype(bool))
    mix = pd.DataFrame({
        "all_events": all_events["category"].value_counts(normalize=True),
        "rule_negative_positives": all_events.loc[positive, "category"].value_counts(normalize=True),
        "r14_admitted_2pct": all_events.loc[all_events["admitted"].astype(bool), "category"].value_counts(normalize=True),
        "precision_rule_negative": all_events[~all_events["rule_flag"].astype(bool) & all_events["label"].notna()]
        .groupby("category")["label"].apply(lambda s: float((s >= 0.5).mean())),
    }).fillna(0.0)
    mix.index.name = "category"
    mix["fast_lane"] = mix.index.isin(jev.FAST_CATEGORIES)
    mix.sort_values("all_events", ascending=False).to_csv(args.out / "jev_categories.csv")
    known = all_events["label"].notna() & ~all_events["rule_flag"].astype(bool)
    natural = []
    for name, routed in (("argmax category in fast set", all_events["category"].isin(jev.FAST_CATEGORIES)),
                         ("urgency >= 0.5", all_events["urgency"] >= 0.5)):
        natural.append({"rule": name, "admission_rate": float(routed.mean()),
                        "recall_rule_negative": float(routed[positive].mean()),
                        "precision_rule_negative": float((all_events.loc[routed & known, "label"] >= 0.5).mean()),
                        "prevalence_rule_negative": float((all_events.loc[known, "label"] >= 0.5).mean())})
    natural = pd.DataFrame(natural)
    natural.to_csv(args.out / "natural_operating_point.csv", index=False)

    # Figure: recall vs budget.
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    wide = pooled.pivot_table(index="policy", columns="budget_pct", values="recall")
    for policy in ORDER:
        if policy in wide.index:
            axis.plot(list(wide.columns), wide.loc[policy].values, marker="o", markersize=5, linewidth=2.4 if policy in ("R14", "R15") else 1.6,
                      color=COLOR[policy], label=LABEL[policy], linestyle="--" if policy == "R0" else "-")
    axis.set_xscale("log")
    axis.set_xticks(list(BUDGETS), [f"{b:g} %" for b in BUDGETS])
    axis.set_xlabel("fast-lane budget (share of the month's events)")
    axis.set_ylabel("recall, rule-negative urgent events")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#ECE9E2")
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out / "fig_recall_at_budget.png", dpi=180)

    for name in ("results.json", "metrics.csv", "validity.csv", "attribution.md", "metric_remediation.csv",
                 "label_diagnostics.csv", "calibration.csv", "robustness.csv", "delays.csv", "human_sanity_sample.csv"):
        if (args.run / name).exists():
            shutil.copy2(args.run / name, args.out / name)

    table = wide.reindex([p for p in ORDER if p in wide.index])
    lines = ["# E1 run 5 — Jev routing arms", "",
             f"Run `{args.run}`; run-4 arms from `{args.run4}` (R2 identical across runs: {r2_identical}).",
             "Required gates not passed (all pre-existing, identical to run 4): "
             + ", ".join(sorted(k for k, v in details.items() if isinstance(v, dict) and v.get("required")
                                and v.get("passed") is not True and k != "valid")) + "; Jev gates: "
             + ", ".join(f"{k} {'pass' if details[k].get('passed') else 'FAIL'}" for k in sorted(details) if k.startswith("jev_") and details[k].get("required")),
             "",
             f"Jev: {jev_info.get('decisions')} decisions, failed {jev_info.get('failed')}, "
             f"${jev_info.get('cost_usd_all_attempts', jev_info.get('cost_usd'))} total, models {jev_info.get('models')}, "
             f"prompt {jev_info.get('prompt_version')} `{str(jev_info.get('prompt_sha256'))[:12]}`.", "",
             "## Recall on rule-negative urgent events (pooled)", "",
             "| arm | " + " | ".join(f"{b:g} %" for b in table.columns) + " |",
             "|---|" + "---|" * len(table.columns)]
    lines += [f"| {LABEL[p]} | " + " | ".join(f"{v:.3f}" for v in table.loc[p].values) + " |" for p in table.index]
    lines += ["", "## Paired contrasts at 2 % (BCa, subject clusters; month clusters as sensitivity)", "",
              "| contrast | effect | 95 % CI subjects | 95 % CI months | cross-run |", "|---|---|---|---|---|"]
    lines += [f"| {r.contrast} | {r.effect:+.3f} | [{r.ci_low_subject:+.3f}, {r.ci_high_subject:+.3f}] | "
              f"[{r.ci_low_month:+.3f}, {r.ci_high_month:+.3f}] | {'yes' if r.cross_run else ''} |" for r in contrasts.itertuples()]
    for title, data, key in (("By source", by_source, "source"), ("By year (memorisation check)", by_year, "year")):
        wide_part = data.pivot_table(index=key, columns="policy", values="recall")
        counts = data.groupby(key)["positives"].max()
        cols = [p for p in ORDER if p in wide_part.columns]
        lines += ["", f"## {title}, recall at 2 %", "", f"| {key} | positives | " + " | ".join(cols) + " |",
                  "|---|---|" + "---|" * len(cols)]
        lines += [f"| {k} | {int(counts[k])} | " + " | ".join(f"{wide_part.loc[k, c]:.3f}" for c in cols) + " |" for k in wide_part.index]
    lines += ["", "## Unbudgeted operating point", "", _markdown(natural),
              "", "## Category mix", "", _markdown(pd.read_csv(args.out / "jev_categories.csv"))]
    (args.out / "results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
