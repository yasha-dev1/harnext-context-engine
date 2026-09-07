# ruff: noqa: E702, E401
"""Copy the small E1 artifacts of a run into apps/eval/reports/<name>/ and draw the report figures.

  UV_NO_SYNC=1 uv run python apps/eval/scripts/e1_report_figures.py \
      --run apps/eval/out/20260906T072507Z-e1-kafka --name e1-kafka-run2 --title "run 2"
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SMALL = ["metrics.csv", "label_diagnostics.csv", "validity.csv", "calibration.csv", "robustness.csv",
         "results.json", "attribution.md", "delays.csv", "preflight.csv"]
NAMES = {"R0": "R0 random", "R1": "R1 rules only", "R2": "R2 global HBOS", "R3": "R3 per-entity gap z",
         "R4": "R4 per-entity HBOS", "R5": "R5 ours (rules+HBOS+guards)", "R6": "R6 LOF"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    seed = args.run / "e1" / "seed-1"
    out = Path("apps/eval/reports") / args.name
    out.mkdir(parents=True, exist_ok=True)
    for f in SMALL:
        if (seed / f).exists():
            shutil.copy(seed / f, out / f)
    for f in ["manifest.json", "config.yaml"]:
        if (args.run / f).exists():
            shutil.copy(args.run / f, out / f)
    for png in (seed / "charts").glob("*.png"):
        shutil.copy(png, out / png.name)

    cols = ["policy", "budget_pct", "source", "rule_flag", "rule", "p_urgent", "admitted", "rule_negative",
            "eligible", "month"]
    df = pq.read_table(seed / "scores.parquet", columns=cols).to_pandas()
    one = df[(df.policy == "R0") & (df.budget_pct == 2.0)]
    pos = df.p_urgent > 0.5
    sub = df[df.rule_negative & pos]
    n_pos = int(sub[(sub.policy == "R0") & (sub.budget_pct == 2.0)].shape[0])
    rec = (sub[sub.admitted].groupby(["policy", "budget_pct"]).size()
           / sub.groupby(["policy", "budget_pct"]).size()).fillna(0).unstack("budget_pct")
    rec.to_csv(out / "recall_at_budget_rule_negative_pooled.csv")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for p in ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]:
        ax.plot(rec.columns, rec.loc[p], marker="o", label=NAMES[p], lw=2.5 if p == "R5" else 1.4)
    ax.plot([1, 2, 5, 10], [0.01, 0.02, 0.05, 0.10], ls=":", c="grey", label="recall = budget (chance)")
    ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10]); ax.set_xticklabels(["1%", "2%", "5%", "10%"])
    ax.set_xlabel("admission budget b"); ax.set_ylabel(f"recall@b, rule-negative positives (n={n_pos})")
    ax.set_title(f"E1 Kafka {args.title}: recall of revealed-urgent rule-negative events")
    ax.legend(fontsize=7.5, frameon=False); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(out / "fig_recall_at_budget.png", dpi=160)

    r1 = df[(df.policy == "R1") & (df.budget_pct == 2.0)]
    share = r1.groupby("month", observed=True).rule_flag.mean()
    share.to_csv(out / "rule_hit_share_by_month.csv")
    fig, ax = plt.subplots(figsize=(8, 3.6)); ax.bar(range(len(share)), share.values * 100, color="#4c72b0")
    for b, c in [(1, "#c44e52"), (2, "#dd8452"), (5, "#55a868")]:
        ax.axhline(b, color=c, ls="--", lw=1, label=f"budget {b}%")
    ax.set_xticks(range(0, len(share), 6)); ax.set_xticklabels(list(share.index[::6]), rotation=45, fontsize=7)
    ax.set_ylabel("% of month's events flagged by rules floor")
    ax.set_title(f"Rules floor vs admission budgets ({args.title})"); ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=.3); fig.tight_layout(); fig.savefig(out / "fig_rule_floor_saturation.png", dpi=160)
    r1[r1.rule_flag].groupby(["source", "rule"], observed=True).size().rename("events").to_csv(out / "rule_floor_breakdown.csv")

    fig, ax = plt.subplots(figsize=(6, 3.4)); v = one.p_urgent.values
    ax.hist(np.clip(v, 1e-9, 1), bins=np.logspace(-9, 0, 60), color="#4c72b0"); ax.set_xscale("log"); ax.set_yscale("log")
    ax.axvline(0.5, color="#c44e52", ls="--", label="positive threshold 0.5")
    ax.set_xlabel("label-model posterior p(urgent)"); ax.set_ylabel("events")
    ax.set_title(f"Posterior over {len(one):,} events ({int((v > 0.5).sum())} above 0.5)"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "fig_label_posterior.png", dpi=160)

    r5 = df[(df.policy == "R5") & df.rule_negative]
    g = r5.groupby("budget_pct").agg(rule_neg=("p_urgent", "size"), eligible=("eligible", "sum"), admitted=("admitted", "sum"))
    g["positives_admitted"] = r5[r5.admitted].groupby("budget_pct").p_urgent.apply(lambda s: int((s > 0.5).sum()))
    g.to_csv(out / "r5_guard_funnel.csv")
    summary = {
        "run": str(args.run), "events": int(len(one)), "positives_full": int((one.p_urgent > 0.5).sum()),
        "positives_rule_negative": n_pos, "rule_share": float(r1.rule_flag.mean()),
        "rules": r1[r1.rule_flag].rule.value_counts().to_dict(),
        "recall_at_budget_rule_negative": {p: [float(x) for x in rec.loc[p]] for p in rec.index},
        "r5_funnel": g.reset_index().to_dict("records"),
        "positives_by_source": one[one.p_urgent > 0.5].source.value_counts().to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
