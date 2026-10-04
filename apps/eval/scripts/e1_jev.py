"""Jev routing decisions for E1 (R14/R15): score events into the cache, and the dev report.

    # what Jev would see, token estimate, no API call
    uv run --package harnext-eval python apps/eval/scripts/e1_jev.py show --window 2021-10-01 2022-01-01
    # score every event of a window (resumable; append-only JSONL keyed by event id + prompt sha)
    uv run --package harnext-eval python apps/eval/scripts/e1_jev.py score \
        --window 2022-01-01 2026-07-01 --out apps/eval/out/e1-jev/<prompt>/scores.jsonl --max-usd 40
    # development check OUTSIDE the registered window: labels on 2021-07..2021-12, metrics on --months
    uv run --package harnext-eval python apps/eval/scripts/e1_jev.py dev --out <cache> --report <dir>

The dev set is 2021 (before the registered window 2022-01 -> 2026-07), so prompt changes are
never tuned on evaluation months.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from harnext_eval.config import load_config
from harnext_eval.corpus.build_replay import read_replay
from harnext_eval.corpus.committers import stamp_events
from harnext_eval.e1 import jev
from harnext_eval.e1.labels import DEFAULT_LABELING_FUNCTIONS, build_labels
from harnext_eval.e1.policies import RuleSettings, match_rule
from harnext_eval.e1.prereg import parse_window
from harnext_eval.e1.run import _router_events

ROOT = Path(__file__).resolve().parents[3]
REPLAY = ROOT / "apps/eval/out/corpus/kafka/replay/kafka-rlong.jsonl"
CONFIG = ROOT / "apps/eval/configs/e1-kafka.yaml"


def _events(replay: Path, window: tuple[datetime, datetime]):
    """The policy-visible events of a window, prepared exactly as E1's run does."""
    events = sorted(
        (event for event in read_replay(replay) if window[0] <= event.time < window[1]),
        key=lambda event: (event.time, event.id),
    )
    stamped, _ = stamp_events(events)
    return stamped, _router_events(stamped)


def _as_of(replay: Path) -> jev.AsOfText:
    """JIRA text timelines from the WHOLE replay: a value at t may come from an edit after t."""
    return jev.AsOfText.from_events(sorted(read_replay(replay), key=lambda event: (event.time, event.id)))


def _month(event) -> str:
    return f"{event.time.year:04d}-{event.time.month:02d}"


def cmd_show(args) -> None:
    window = parse_window(tuple(args.window))
    _, events = _events(args.replay, window)
    as_of = _as_of(args.replay)
    rng = random.Random(0)
    by_type: dict[str, list] = {}
    for event in events:
        by_type.setdefault(event.type, []).append(event)
    chars = [len(json.dumps(jev.event_state(event, as_of), ensure_ascii=False)) for event in events]
    fixed = len(json.dumps(jev.questions(), ensure_ascii=False))
    for kind, group in sorted(by_type.items()):
        print(f"--- {kind} ({len(group)})")
        print(json.dumps(jev.event_state(rng.choice(group), as_of), ensure_ascii=False, indent=1)[:2500])
    est = (np.mean(chars) + fixed) / 3.6
    print(f"\n{len(events)} events; state chars mean {np.mean(chars):.0f} p90 {np.percentile(chars, 90):.0f}; "
          f"questions {fixed} chars; ~{est:.0f} input tokens/call -> ~${len(events) * est * 0.042e-6:.2f} at $0.042/M")
    print("prompt_sha256", jev.prompt_sha256())


def cmd_score(args) -> None:
    from harnext_eval.e2.v2.engine.clients import JevClient
    from harnext_eval.e2.v2.engine.config import EngineConfig

    window = parse_window(tuple(args.window))
    _, events = _events(args.replay, window)
    if args.months:
        events = [event for event in events if _month(event) in set(args.months)]
    if args.limit:
        events = events[: args.limit]
    print(f"prompt {jev.PROMPT_VERSION} sha {jev.prompt_sha256()}; {len(events)} events -> {args.out}", flush=True)
    as_of = _as_of(args.replay)
    config = EngineConfig(jev_model=jev.JEV_MODEL, jev_timeout_s=args.timeout, retries=args.retries)

    async def main():
        import httpx

        limits = httpx.Limits(max_connections=args.concurrency * 2, max_keepalive_connections=args.concurrency * 2)
        client = JevClient(config, http=httpx.AsyncClient(limits=limits))
        try:
            return await jev.score_events(events, args.out, client=client, as_of=as_of, concurrency=args.concurrency,
                                          max_usd=args.max_usd, progress_every=args.progress)
        finally:
            await client.close()

    stats = asyncio.run(main())
    print(json.dumps(stats, indent=1))
    if stats["stopped_early"] or stats["failed"]:
        sys.exit(f"incomplete: stopped_early={stats['stopped_early']} failed={stats['failed']} -- run score again")


def _recall_at(frame: pd.DataFrame, column: str, budget: float) -> tuple[float, int, int]:
    """Recall over rule-negative positives of the month-wise top-b% (all events ranked, R2-style)."""
    admitted_pos = positives = 0
    for _, month in frame.groupby("month"):
        capacity = max(1, int(round(len(month) * budget / 100)))
        order = month.sort_values([column, "event_id"], ascending=[False, True])
        top = set(order["event_id"].head(capacity))
        target = month[(~month["rule"]) & (month["label"] >= 0.5)]
        positives += len(target)
        admitted_pos += int(target["event_id"].isin(top).sum())
    return (admitted_pos / positives if positives else float("nan")), admitted_pos, positives


def cmd_dev(args) -> None:
    cfg = load_config(args.config)
    label_window = parse_window(tuple(args.label_window))
    stamped, events = _events(args.replay, label_window)
    excluded = set(cfg.e1.exclude_label_functions)
    functions = [function for function in DEFAULT_LABELING_FUNCTIONS if function.name not in excluded]
    labels = build_labels(stamped, functions, observation_end=label_window[1]).probabilities.to_dict()
    months = set(args.months)
    settings = RuleSettings(enabled=True, vote_thread_start_only=True, dedup_per_subject=True)
    seen: set[tuple[str, str, str]] = set()
    rows = []
    decisions = jev.load_decisions(args.out, jev.prompt_sha256())
    for event in events:
        month = _month(event)
        if month not in months:
            continue
        rule = match_rule(event, settings)
        if rule and (month, event.subject, rule) in seen:
            rule = None
        elif rule:
            seen.add((month, event.subject, rule))
        decision = decisions.get(event.id)
        rows.append({
            "event_id": event.id, "month": month, "source": event.source.split(":", 1)[0], "type": event.type,
            "rule": rule is not None, "label": labels.get(event.id, np.nan),
            "R0": random.Random(f"0:{event.id}").random(),
            "R14": decision.score("R14") if decision else np.nan,
            "R15": decision.score("R15") if decision else np.nan,
            "category": decision.category if decision else None,
            "fast_category_p": decision.fast_category_p if decision else np.nan,
            "urgency": decision.urgency if decision else np.nan,
            "ok": bool(decision and decision.ok),
            "cost_usd": decision.cost_usd if decision else 0.0,
            "input_tokens": decision.input_tokens if decision else 0,
            "seconds": decision.seconds if decision else np.nan,
        })
    frame = pd.DataFrame(rows)
    missing = int((~frame["ok"]).sum())
    known = frame[frame["label"].notna()].copy()
    lines = [f"# E1 Jev dev check — prompt {jev.PROMPT_VERSION} `{jev.prompt_sha256()[:12]}`", "",
             f"Months {sorted(months)} (outside the registered window); events {len(frame)}, "
             f"without a decision {missing}; labelled {len(known)}; "
             f"rule-negative positives {int(((~known['rule']) & (known['label'] >= 0.5)).sum())}.", "",
             "| budget | R0 random | R14 Jev category | R15 Jev urgency |", "|---|---|---|---|"]
    for budget in (1.0, 2.0, 5.0, 10.0):
        cells = []
        for column in ("R0", "R14", "R15"):
            value, hit, total = _recall_at(known.fillna({column: -1.0}), column, budget)
            cells.append(f"{value:.3f} ({hit}/{total})")
        lines.append(f"| {budget:g} % | " + " | ".join(cells) + " |")
    ok = frame[frame["ok"]]
    fast = ok["category"].isin(jev.FAST_CATEGORIES)
    lines += ["", f"Natural fast-lane rate (argmax category in {list(jev.FAST_CATEGORIES)}): {fast.mean():.3%}; "
              f"urgency >= 0.5: {(ok['urgency'] >= 0.5).mean():.3%}.",
              f"Cost ${ok['cost_usd'].sum():.4f} for {len(ok)} decisions (${ok['cost_usd'].mean() * 1e6:.1f} per million events "
              f"-> ${ok['cost_usd'].mean() * 395_000:.2f} for the registered window); mean input tokens "
              f"{ok['input_tokens'].mean():.0f}; latency p50 {ok['seconds'].median():.2f} s p95 {ok['seconds'].quantile(.95):.2f} s.", "",
              "| category | events | share | positives (rule-neg) | precision |", "|---|---|---|---|---|"]
    known_ok = known[known["ok"]]
    for category, count in Counter(ok["category"]).most_common():
        part = known_ok[(known_ok["category"] == category) & (~known_ok["rule"])]
        pos = int((part["label"] >= 0.5).sum())
        lines.append(f"| {category}{' (fast)' if category in jev.FAST_CATEGORIES else ''} | {count} | {count / len(ok):.2%} | "
                     f"{pos} | {pos / len(part) if len(part) else float('nan'):.3%} |")
    lines += ["", "| source | events | R14 recall@2 % | R15 recall@2 % | R0 recall@2 % |", "|---|---|---|---|---|"]
    for source, part in known.groupby("source"):
        cells = [f"{_recall_at(known.fillna({c: -1.0}).assign(label=np.where(known['source'] == source, known['label'], 0.0)), c, 2.0)[0]:.3f}"
                 for c in ("R14", "R15", "R0")]
        lines.append(f"| {source} | {len(part)} | " + " | ".join(cells) + " |")
    args.report.mkdir(parents=True, exist_ok=True)
    (args.report / "dev.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    frame.to_csv(args.report / "dev_events.csv", index=False)
    print("\n".join(lines))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("show", "score", "dev"):
        command = sub.add_parser(name)
        command.add_argument("--replay", type=Path, default=REPLAY)
        if name != "dev":
            command.add_argument("--window", nargs=2, required=True, metavar=("START", "END"))
        if name in ("score", "dev"):
            command.add_argument("--out", type=Path, required=True, help="decision cache (JSONL)")
    score = sub.choices["score"]
    score.add_argument("--months", nargs="*")
    score.add_argument("--limit", type=int)
    score.add_argument("--concurrency", type=int, default=16)
    score.add_argument("--max-usd", type=float, default=None, help="cap on the total cost recorded in the cache for this prompt")
    score.add_argument("--timeout", type=float, default=20.0)
    score.add_argument("--retries", type=int, default=4)
    score.add_argument("--progress", type=int, default=1000)
    dev = sub.choices["dev"]
    dev.add_argument("--config", type=Path, default=CONFIG)
    dev.add_argument("--label-window", nargs=2, default=["2021-07-01", "2022-01-01"])
    dev.add_argument("--months", nargs="+", default=["2021-10", "2021-11", "2021-12"])
    dev.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    {"show": cmd_show, "score": cmd_score, "dev": cmd_dev}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
