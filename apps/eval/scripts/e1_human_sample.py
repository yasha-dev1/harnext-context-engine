"""Build the human sanity-sample sheet for E1 (spec §7 E1 validity checks; gate
``human_sanity_100_two_annotators``) plus a label-definition audit sample.

Sheet A (spec): the 100 top-scored rule-negative events under R5 at the 2 % budget, pooled over
months — annotators answer "would you want to be interrupted for this?" (yes / no / unsure).
Sheet B (label audit): 100 rule-negative events stratified over the fusion rules' disagreement
cells (registered model vs ">= 2 outcome LFs" vs ">= half of votes"), so κ against each rule can
be computed from the same annotations. Both sheets hide all label and score columns; a separate
key file keeps them for scoring.

  UV_NO_SYNC=1 uv run python apps/eval/scripts/e1_human_sample.py \
      --run apps/eval/out/20260906T072507Z-e1-kafka \
      --votes apps/eval/out/label-sensitivity/votes.parquet \
      --replay apps/eval/out/corpus/kafka/replay/kafka-rlong.jsonl \
      --out apps/eval/out/human-sample --seed 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OUTCOME = ["jira_committer_comment_1h", "jira_priority_raised_later", "jira_resolved_24h", "jira_linked_pr_24h",
           "dev_committer_reply_1h", "dev_three_responders_2h", "dev_vote_cancelled_recast_later",
           "dev_cve_blocker_later", "github_reverted_48h", "github_hotfix_reference_24h"]


def _excerpt(data: dict) -> tuple[str, str]:
    title = data.get("summary") or data.get("subject") or data.get("title") or ""
    body = data.get("body") or data.get("description") or data.get("comment") or data.get("message") or data.get("message_headline") or ""
    if data.get("field"):
        body = f"{data.get('field')}: {data.get('from')} → {data.get('to')}\n" + str(body)
    return str(title)[:200], str(body)[:1200]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--votes", type=Path, required=True)
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    s = pq.read_table(args.run / "e1/seed-1/scores.parquet",
                      columns=["event_id", "subject", "source", "t", "score", "p_urgent", "rule_negative", "admitted", "eligible"],
                      filters=[("policy", "=", "R5"), ("budget_pct", "=", 2.0), ("rule_negative", "=", True)]).to_pandas()
    sheet_a = s.sort_values("score", ascending=False).head(args.n).copy()
    sheet_a["sheet"] = "A_top_r5"

    votes = pd.read_parquet(args.votes)
    votes = votes.loc[votes.index.intersection(s.event_id)]
    m = votes.to_numpy()
    pos_out = (votes[[c for c in OUTCOME if c in votes]].to_numpy() == 1).sum(1)
    fired = (m != -1).sum(1)
    pos = (m == 1).sum(1)
    lab = pd.DataFrame({
        "event_id": votes.index,
        "reg_model": s.set_index("event_id").loc[votes.index, "p_urgent"].to_numpy() > 0.5,
        "ge2_outcome": pos_out >= 2,
        "ge_half": (fired > 0) & (2 * pos >= fired),
    })
    lab["cell"] = lab.reg_model.astype(int).astype(str) + lab.ge2_outcome.astype(int).astype(str) + lab.ge_half.astype(int).astype(str)
    cells = lab.cell.value_counts()
    per_cell = max(1, args.n // len(cells))
    picks = []
    for cell, group in lab.groupby("cell"):
        k = min(per_cell, len(group))
        picks.append(group.sample(k, random_state=int(rng.integers(1 << 31))))
    sheet_b = pd.concat(picks)
    remaining = args.n - len(sheet_b)
    if remaining > 0:
        pool = lab[~lab.event_id.isin(sheet_b.event_id)]
        sheet_b = pd.concat([sheet_b, pool.sample(min(remaining, len(pool)), random_state=args.seed)])
    sheet_b = s.set_index("event_id").loc[sheet_b.event_id].reset_index().merge(lab, on="event_id")
    sheet_b["sheet"] = "B_label_audit"

    ids = set(sheet_a.event_id) | set(sheet_b.event_id)
    ctx: dict[str, dict] = {}
    with args.replay.open() as fh:
        for line in fh:
            i = line.find('"id":"')
            eid = line[i + 6 : line.find('"', i + 6)]
            if eid in ids:
                ev = json.loads(line)
                title, body = _excerpt(ev.get("data") or {})
                ctx[eid] = {"type": ev["type"], "title": title, "body": body}
    both = pd.concat([sheet_a, sheet_b], ignore_index=True)
    both["type"] = both.event_id.map(lambda e: ctx.get(e, {}).get("type"))
    both["title"] = both.event_id.map(lambda e: ctx.get(e, {}).get("title"))
    both["body"] = both.event_id.map(lambda e: ctx.get(e, {}).get("body"))
    both = both.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    both["item"] = range(1, len(both) + 1)

    public = both[["item", "sheet", "event_id", "subject", "source", "t", "type", "title", "body"]].copy()
    for col in ["annotator_1", "annotator_2"]:
        public[col] = ""
    public.to_csv(args.out / "annotation_sheet.csv", index=False)
    key_cols = ["item", "event_id", "sheet", "score", "p_urgent", "admitted", "eligible", "reg_model", "ge2_outcome", "ge_half", "cell"]
    both[[c for c in key_cols if c in both]].to_csv(args.out / "annotation_key.csv", index=False)
    (args.out / "README.md").write_text(
        "# E1 human sanity sample\n\n"
        f"{len(sheet_a)} items = top-scored rule-negative events under R5 at 2 % (spec sheet A); "
        f"{len(sheet_b)} items = label-definition audit stratified over {len(cells)} disagreement cells (sheet B). "
        "Rows are shuffled; `annotation_sheet.csv` hides scores and labels. Each annotator fills their column with "
        "`yes` / `no` / `unsure` for: *Would you, as a Kafka committer, want to be interrupted for this event right now?* "
        "Score with `annotation_key.csv`: Cohen κ between annotators, then κ of the majority human label against "
        "`reg_model`, `ge2_outcome`, `ge_half` (sheet B) and the yes-rate by R5 score bucket (sheet A).\n"
        f"\nDisagreement cells (reg_model, ge2_outcome, ge_half) and their population sizes: {cells.to_dict()}\n"
    )
    print(f"wrote {args.out}/annotation_sheet.csv ({len(public)} items), key, README; cells={cells.to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
