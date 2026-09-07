"""Label-fusion sensitivity on the real replay (E1 review finding 2).

Applies the registered labelling functions once over a window and reports the prevalence that
alternative fusion rules would imply, overall and per source. Read-only; no policy scoring.

Run (≈ 30–40 min, ≈ 4 GB):
  UV_NO_SYNC=1 uv run python apps/eval/scripts/e1_label_sensitivity.py \
      --replay apps/eval/out/corpus/kafka/replay/kafka-rlong.jsonl \
      --window 2022-01-01 2026-07-01 --out apps/eval/out/label-sensitivity
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from harnext_eval.corpus.committers import stamp_events
from harnext_eval.e1.labels import (
    ABSTAIN,
    DEFAULT_LABELING_FUNCTIONS,
    POSITIVE,
    apply_labeling_functions,
    fit_label_model,
)
from harnext_eval.types import EvalEvent


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True, type=Path)
    ap.add_argument("--window", nargs=2, required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--exclude", nargs="*", default=["github_trunk_ci_failure_fix_6h"])
    args = ap.parse_args()
    end = _parse(args.window[1])
    args.out.mkdir(parents=True, exist_ok=True)

    events: list[EvalEvent] = []
    with args.replay.open() as fh:
        for line in fh:
            # cheap pre-filter on the ISO timestamp before validating
            i = line.find('"time":"')
            if i < 0:
                continue
            ts = line[i + 8 : i + 8 + 19]
            if not (args.window[0] <= ts and ts < args.window[1]):
                continue
            events.append(EvalEvent.model_validate_json(line))
    events.sort(key=lambda e: (e.time, e.id))
    events, _ = stamp_events(events)
    print(f"events in window: {len(events)}", flush=True)

    functions = [f for f in DEFAULT_LABELING_FUNCTIONS if f.name not in set(args.exclude)]
    votes = apply_labeling_functions(events, functions, observation_end=end)
    observability = votes.attrs.pop("observability", None)
    votes.attrs = {}
    source = pd.Series(
        ["jira" if ".jira." in e.type else "mail" if ".mail." in e.type else "github" for e in events],
        index=votes.index,
    )
    outcome_cols = [f.name for f in functions if not f.declared]
    m = votes.to_numpy()
    pos = (m == POSITIVE).sum(1)
    fired = (m != ABSTAIN).sum(1)
    pos_outcome = (votes[outcome_cols].to_numpy() == POSITIVE).sum(1)

    rules: dict[str, np.ndarray] = {}
    full = fit_label_model(votes)
    rules["A  weighted model as registered (p>=0.5)"] = (full.probabilities.to_numpy() >= 0.5)
    if "jira_fix_version_in_flight_later" in votes:
        nofix = fit_label_model(votes.drop(columns=["jira_fix_version_in_flight_later"]))
        rules["A' weighted model minus fix-version LF"] = (nofix.probabilities.to_numpy() >= 0.5)
    rules["B1 any outcome LF positive"] = pos_outcome >= 1
    rules["B2 >=2 distinct outcome LFs positive"] = pos_outcome >= 2
    rules["B3 >=half of non-abstaining votes positive"] = (fired > 0) & (2 * pos >= fired)
    rules["B4 strict majority positive"] = 2 * pos > fired

    rows = []
    for name, lab in rules.items():
        r = {"rule": name, "positives": int(lab.sum()), "prevalence": float(lab.mean())}
        for src in ["jira", "github", "mail"]:
            sel = (source == src).to_numpy()
            r[f"{src}_positives"] = int(lab[sel].sum())
            r[f"{src}_prevalence"] = float(lab[sel].mean()) if sel.any() else float("nan")
        rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "fusion_prevalence.csv", index=False)
    print(table.to_string(), flush=True)

    fired_col = (m != ABSTAIN).sum(0)
    firing = pd.DataFrame({
        "function": votes.columns,
        "positive_votes": (m == POSITIVE).sum(0),
        "non_abstain": fired_col,
        "positive_rate_when_observable": np.where(fired_col > 0, (m == POSITIVE).sum(0) / np.maximum(fired_col, 1), np.nan),
    })
    firing.to_csv(args.out / "lf_firing.csv", index=False)
    print(firing.to_string(), flush=True)
    pattern = pd.Series(pos_outcome).value_counts().sort_index()
    pattern.to_csv(args.out / "outcome_vote_count_histogram.csv")
    print("events by number of positive outcome LFs:", pattern.to_dict(), flush=True)
    full.diagnostics.to_csv(args.out / "diagnostics_registered_model.csv")
    try:  # bulky artefacts last, so a serialisation problem cannot lose the tables above
        pd.DataFrame(votes.to_numpy(), index=votes.index, columns=votes.columns).to_parquet(args.out / "votes.parquet")
        if isinstance(observability, pd.DataFrame):
            pd.DataFrame(observability.to_numpy(), index=observability.index, columns=observability.columns).to_parquet(args.out / "observability.parquet")
    except Exception as exc:  # noqa: BLE001
        print(f"parquet write skipped: {exc}", flush=True)
    json.dump({"events": len(events), "window": args.window, "excluded": args.exclude},
              open(args.out / "manifest.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
