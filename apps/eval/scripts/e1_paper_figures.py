"""Paper-ready figures and tables for one finished E1 run (read-only over run outputs).

Reads ``…/e1/seed-<n>`` (scores.parquet, metrics.csv, results.json, label_diagnostics.csv,
robustness.csv, calibration.csv, validity.csv, metric_remediation.csv) and writes to ``--out``:

  recall_at_budget_rule_negative_pooled.csv / fig_recall_at_budget.png
  rule_hit_share_by_month.csv / fig_rule_floor_saturation.png
  guard_funnel.csv                    (R5/R8/R9 eligible → admitted → positives per budget)
  per_source_recall_2pct.csv
  primary_contrasts.csv
  lateness_rule_negative.csv          (timestamped affiliation / detection delay per policy)
  fig_label_posterior.png
  results_summary.md                  (Markdown + LaTeX tables for the paper)

Run:
  UV_NO_SYNC=1 uv run python apps/eval/scripts/e1_paper_figures.py \
      --run apps/eval/out/<run>/e1/seed-1 --out apps/eval/reports/e1-kafka-run3
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

BUDGETS = (1.0, 2.0, 5.0, 10.0)
POLICY_LABEL = {
    "R0": "R0 random", "R1": "R1 rules only", "R2": "R2 global HBOS", "R3": "R3 gap robust-z",
    "R4": "R4 per-entity HBOS", "R5": "R5 guarded HBOS (rules share budget)",
    "R6": "R6 per-entity LOF", "R7": "R7 always fast",
    "R8": "R8 guarded HBOS (rules outside budget)", "R9": "R9 = R8 + any-key guards",
}
# Fixed categorical assignment (identity follows the policy, never its rank).
COLOR = {
    "R2": "#2a78d6", "R5": "#eb6834", "R4": "#1baf7a", "R1": "#eda100",
    "R9": "#e87ba4", "R6": "#008300", "R8": "#4a3aa7", "R3": "#e34948", "R0": "#7a756d",
}
MAIN_POLICIES = ("R0", "R1", "R2", "R4", "R5", "R8", "R9")
SMALL_FILES = (
    "metrics.csv", "label_diagnostics.csv", "validity.csv", "calibration.csv", "robustness.csv",
    "delays.csv", "metric_remediation.csv", "label_situations.csv", "human_sanity_sample.csv",
    "human_sanity_key.csv", "results.json", "attribution.md", "preflight.csv",
)


def _style(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#DCD8CF")
    axis.grid(axis="y", color="#ECE9E2", linewidth=1)
    axis.set_axisbelow(True)
    axis.tick_params(colors="#4F4B45", labelsize=9)


def _fmt(value: float, digits: int = 3) -> str:
    return "—" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value:.{digits}f}"


def _md(frame: pd.DataFrame, digits: int = 3) -> str:
    """Markdown table without the optional tabulate dependency."""

    if frame is None or frame.empty:
        return "(none)"
    body = frame.copy()
    for column in body.columns:
        if body[column].dtype.kind == "f":
            body[column] = body[column].map(lambda v: _fmt(v, digits))
    header = "| " + " | ".join(str(c) for c in body.columns) + " |"
    rule = "|" + "|".join("---" for _ in body.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in body.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def _latex(frame: pd.DataFrame, caption: str, label: str, digits: int = 3) -> str:
    body = frame.copy()
    for column in body.columns:
        if body[column].dtype.kind == "f":
            body[column] = body[column].map(lambda v: _fmt(v, digits))
    header = " & ".join(str(c).replace("_", r"\_").replace("%", r"\%") for c in body.columns)
    rows = "\n".join(" & ".join(str(v).replace("_", r"\_").replace("%", r"\%") for v in row) + r" \\" for row in body.itertuples(index=False))
    align = "l" + "r" * (len(body.columns) - 1)
    return (
        "\\begin{table}[t]\n\\centering\n\\small\n"
        f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{header} \\\\\n\\midrule\n{rows}\n\\bottomrule\n"
        f"\\end{{tabular}}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    run, out = args.run, args.out
    out.mkdir(parents=True, exist_ok=True)
    for name in SMALL_FILES:
        if (run / name).exists():
            shutil.copy(run / name, out / name)
    for name in ("manifest.json", "config.yaml"):
        candidate = run.parent.parent / name
        if candidate.exists():
            shutil.copy(candidate, out / name)

    results = json.loads((run / "results.json").read_text())
    primary = results["primary"]
    metrics = pd.read_csv(run / "metrics.csv")
    columns = ["event_id", "policy", "budget_pct", "admitted", "rule_negative", "p_urgent",
               "source", "eligible", "mandatory", "month", "rule", "rule_flag", "subject"]
    scores = pq.read_table(run / "scores.parquet", columns=columns).to_pandas()
    scores["positive"] = scores["p_urgent"] >= 0.5
    negative = scores[scores["rule_negative"].astype(bool)]
    n_events = scores.loc[scores["policy"] == "R0"].drop_duplicates(["event_id", "budget_pct"])["event_id"].nunique()
    n_positive_neg = int(negative[(negative["policy"] == "R1") & (negative["budget_pct"] == 2.0)]["positive"].sum())

    # 1. Pooled recall@b on rule-negative positives, per policy × budget.
    pooled = (
        negative[negative["positive"]]
        .groupby(["policy", "budget_pct"], observed=True)["admitted"]
        .agg(["sum", "size"]).reset_index()
        .rename(columns={"sum": "admitted_positives", "size": "positives"})
    )
    pooled["recall"] = pooled["admitted_positives"] / pooled["positives"].clip(lower=1)
    pooled.to_csv(out / "recall_at_budget_rule_negative_pooled.csv", index=False)
    wide = pooled.pivot(index="policy", columns="budget_pct", values="recall").reindex([f"R{i}" for i in range(10)])
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.plot(BUDGETS, [b / 100 for b in BUDGETS], color="#9A958C", linestyle=":", linewidth=1.5, label="recall = budget (chance)")
    for policy in MAIN_POLICIES:
        if policy in wide.index:
            axis.plot(list(wide.columns), wide.loc[policy].values, marker="o", markersize=5, linewidth=2,
                      color=COLOR[policy], label=POLICY_LABEL[policy], linestyle="--" if policy == "R0" else "-")
    axis.set_xscale("log"); axis.set_xticks(BUDGETS); axis.set_xticklabels([f"{b:g} %" for b in BUDGETS])
    axis.set_xlabel("admission budget b"); axis.set_ylabel("recall@b, rule-negative revealed-urgent events")
    axis.set_title(f"E1 recall at budget (Kafka, {n_positive_neg} rule-negative positives)", fontsize=10)
    _style(axis); axis.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "fig_recall_at_budget.png", dpi=200); plt.close(fig)

    # 2. Rule-floor share by month against budgets.
    rules_ref = scores[(scores["policy"] == "R1") & (scores["budget_pct"] == 2.0)]
    by_month = rules_ref.groupby("month", observed=True).agg(events=("event_id", "size"), rule_hits=("rule_flag", "sum")).reset_index()
    by_month["rule_share"] = by_month["rule_hits"] / by_month["events"]
    for budget in BUDGETS:
        by_month[f"infeasible_{budget:g}pct"] = by_month["rule_share"] > budget / 100
    by_month.to_csv(out / "rule_hit_share_by_month.csv", index=False)
    fig, axis = plt.subplots(figsize=(7.2, 3.6))
    axis.bar(range(len(by_month)), by_month["rule_share"] * 100, color="#C2BDB3", width=0.8)
    for budget, color in zip((1.0, 2.0, 5.0), ("#eb6834", "#2a78d6", "#1baf7a"), strict=True):
        axis.axhline(budget, color=color, linewidth=1.5, label=f"budget {budget:g} %")
    step = max(1, len(by_month) // 8)
    axis.set_xticks(range(0, len(by_month), step)); axis.set_xticklabels(by_month["month"].iloc[::step], rotation=0, fontsize=8)
    axis.set_ylabel("rule hits, % of month's events"); axis.set_title("Rule floor per month vs. admission budgets", fontsize=10)
    _style(axis); axis.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(out / "fig_rule_floor_saturation.png", dpi=200); plt.close(fig)
    rule_breakdown = rules_ref[rules_ref["rule_flag"].astype(bool)].groupby(["rule", "source"], observed=True).size().reset_index(name="events")
    rule_breakdown.to_csv(out / "rule_floor_breakdown.csv", index=False)

    # 3. Guard funnel for guarded policies.
    funnel_rows = []
    for policy in ("R5", "R8", "R9"):
        for budget in BUDGETS:
            sub = negative[(negative["policy"] == policy) & (negative["budget_pct"] == budget)]
            if sub.empty:
                continue
            funnel_rows.append({
                "policy": policy, "budget_pct": budget, "rule_negative_events": len(sub),
                "eligible": int(sub["eligible"].sum()), "admitted": int(sub["admitted"].sum()),
                "positives_among_admitted": int((sub["admitted"] & sub["positive"]).sum()),
                "positives_among_eligible": int((sub["eligible"] & sub["positive"]).sum()),
            })
    funnel = pd.DataFrame(funnel_rows)
    funnel.to_csv(out / "guard_funnel.csv", index=False)

    # 4. Per-source recall@2 % (rule-negative), pooled.
    per_source = (
        negative[negative["positive"] & (negative["budget_pct"] == 2.0)]
        .groupby(["policy", "source"], observed=True)["admitted"].agg(["mean", "size"]).reset_index()
        .rename(columns={"mean": "recall_2pct", "size": "positives"})
    )
    per_source.to_csv(out / "per_source_recall_2pct.csv", index=False)
    source_wide = per_source.pivot(index="policy", columns="source", values="recall_2pct").reindex([f"R{i}" for i in range(10)])

    # 5. Primary contrasts.
    contrast_rows = []
    for name in primary.get("contrasts", []):
        key = name.replace("-", "_minus_").lower()
        contrast_rows.append({
            "contrast": name, "effect": primary.get(key), "ci_low": primary.get(f"{key}_ci_low"),
            "ci_high": primary.get(f"{key}_ci_high"), "subject_clusters": primary.get(f"{key}_n_entities"),
            "month_ci_low": primary.get(f"{key}_month_ci_low"), "month_ci_high": primary.get(f"{key}_month_ci_high"),
        })
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(out / "primary_contrasts.csv", index=False)

    # 6. Lateness (timestamped, rule-negative situations).
    robustness = pd.read_csv(run / "robustness.csv")
    lateness = pd.DataFrame()
    if "population" in robustness.columns:
        lateness = robustness[(robustness["population"] == "rule_negative") & (robustness["budget_pct"] == 2.0)][
            ["policy", "n_situations", "affiliation_precision", "affiliation_recall", "delay_p50_s", "delay_p95_s", "detected_rate"]
        ].dropna(subset=["affiliation_recall"]).reset_index(drop=True)
        lateness.to_csv(out / "lateness_rule_negative.csv", index=False)

    # 7. Label posterior histogram.
    posterior = scores[(scores["policy"] == "R0") & (scores["budget_pct"] == 2.0)]["p_urgent"].dropna()
    fig, axis = plt.subplots(figsize=(6.0, 3.4))
    edges = np.logspace(-10, 0, 41)
    axis.hist(np.clip(posterior, 1e-10, 1), bins=edges, color="#2a78d6")
    axis.axvline(0.5, color="#eb6834", linewidth=1.5, label="registered threshold 0.5")
    axis.set_xscale("log"); axis.set_yscale("log"); axis.set_xlabel("label-model posterior p(urgent)"); axis.set_ylabel("events")
    axis.set_title(f"Posterior over {len(posterior):,} events; {(posterior >= 0.5).sum():,} above 0.5", fontsize=10)
    _style(axis); axis.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(out / "fig_label_posterior.png", dpi=200); plt.close(fig)

    # 8. Summary document.
    diagnostics = pd.read_csv(run / "label_diagnostics.csv")
    validity = pd.read_csv(run / "validity.csv")
    remediation = pd.read_csv(run / "metric_remediation.csv") if (run / "metric_remediation.csv").exists() else pd.DataFrame()
    recall_table = wide.reset_index().rename(columns={b: f"{b:g} %" for b in wide.columns})
    recall_table.insert(1, "condition", recall_table["policy"].map(POLICY_LABEL))
    lines = [
        f"# E1 results summary — run `{run.parent.parent.name}`", "",
        f"Window {primary.get('window')} · events {n_events:,} · rule-negative positives {n_positive_neg} "
        f"({primary.get('n_subjects')} subjects, {primary.get('n_months')} months) · rule hit rate "
        f"{_fmt(primary.get('rule_hit_rate', float('nan')), 4)} · evidence status **{primary.get('evidence_status')}**", "",
        "## Primary metric: recall@2 % on rule-negative events", "",
        _md(pd.DataFrame({"policy": list(primary["by_policy"]), "recall_2pct": list(primary["by_policy"].values())})), "",
        "## Registered contrasts (paired BCa, subject clusters; month-cluster CI as sensitivity)", "",
        _md(contrasts), "",
        "## Recall at every budget (pooled, rule-negative)", "", _md(recall_table), "",
        "## Guard funnel (rule-negative)", "", _md(funnel, 0), "",
        "## Per-source recall@2 % (rule-negative)", "", _md(source_wide.reset_index()), "",
        "## Lateness, timestamped subject-separated situations (rule-negative, 2 %)", "", _md(lateness), "",
        "## Label diagnostics", "", _md(diagnostics, 4), "",
        "## Metric remediation record", "", _md(remediation, 5), "",
        "## Validity gates not passed", "",
        _md(validity[validity["status"] != "pass"][["gate", "status", "required", "reason"]]), "",
        "## LaTeX", "", "```latex",
        _latex(recall_table.drop(columns=["condition"]), "Recall at budget on rule-negative revealed-urgent events (Apache Kafka).", "tab:e1-recall"),
        _latex(contrasts[["contrast", "effect", "ci_low", "ci_high", "subject_clusters"]] if not contrasts.empty else contrasts,
               "Registered paired contrasts of recall@2\\% (BCa bootstrap, subject clusters).", "tab:e1-contrasts"),
        _latex(funnel, "Guard funnel for the guarded conditions on rule-negative events.", "tab:e1-funnel", 0) if not funnel.empty else "",
        "```", "",
    ]
    (out / "results_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((out / "results_summary.md").read_text()[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
