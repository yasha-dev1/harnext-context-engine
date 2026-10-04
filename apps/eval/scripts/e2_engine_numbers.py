"""Every number of the engine's section in E2-ENGINE-DESIGN.md, recomputed from dev-run traces.

    .venv/bin/python apps/eval/scripts/e2_engine_numbers.py RUN_DIR [RUN_DIR ...] [--json out.json]

RUN_DIR = apps/eval/out/kafka-500-v2/engine-dev/<run> (results.jsonl + traces/). Per run: tasks,
correct, cost per question split Jev / chat (local chat = list-price equivalent), unmetered upper
bound, p50 / p95 seconds, turns, find plans vs model SQL, verification actions, "not found"
answers, guard refusals, code / config / store hashes. Over all runs: Jev latency p50 / p95, Jev
INPUT tokens per call (median), a grounding audit of the final answers (checkable values and
citations not found in the rows shown to a model or in the findings) and accuracy split by the
number of sources routed. Timing of the branches (engine v2): end times, branches that reached
the cut-off (answered by the conclusion call) or ended unanswered, answer-now calls (seconds,
timeouts), how many branches ran at once, and the same split for questions routed to two or
more sources. Runs of invented questions (e2_engine_dev.py --questions) have no grades.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from harnext_eval.e2.v2.engine import prompts, verify
from harnext_eval.e2.v2.engine.config import ENDPOINTS, EngineConfig


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pct(values: list[float], q: float) -> float | None:
    values = sorted(values)
    return round(values[int(q * (len(values) - 1))], 3) if values else None


def grounding(trace: dict) -> dict:
    """Values and citations of the final answer not found in what was shown or found."""
    shown = "\n".join(r for b in trace["branches"] for r in b.get("results", []))
    findings = "\n".join(b["finding"] for b in trace["branches"])
    known = trace["question"] + "\n" + "\n".join(map(str, prompts.texts().values()))
    values = verify.unverified(trace["answer"], shown + "\n" + findings, known)
    ids = {i for b in trace["branches"] for c in b.get("calls", []) for i in c.get("shown") or []}
    ids |= {i for b in trace["branches"] for c in b.get("calls", []) for i in c.get("events") or []}
    ids |= {
        i for b in trace["branches"] for i in b.get("evidence", []) if i in shown
    }  # read / thread / sql rows
    _, citations = verify.citations(trace["answer"], ids)
    return {"values": values, "citations": citations}


def list_price(model: str) -> tuple[float, float]:
    for preset in ENDPOINTS:
        config = EngineConfig(**ENDPOINTS[preset])
        if config.chat_model == model:
            return config.chat_price
    return EngineConfig().chat_price


def parallelism(trace: dict) -> float | None:
    """Seconds of branch chat calls per second of branch wall time: above 1, branches' model calls
    ran side by side."""
    spans = [(b["started_s"], b["ended_s"]) for b in trace["branches"] if b.get("started_s") is not None and b.get("ended_s") is not None]  # fmt: skip
    busy = sum(c["seconds"] for c in trace["calls"] if c["kind"] == "chat" and c["purpose"].startswith("branch"))  # fmt: skip
    wall = max(e for _, e in spans) - min(s for s, _ in spans) if spans else 0
    return busy / wall if wall > 0 else None


def timing(traces: list[dict]) -> dict:
    """Branch timing and endings of a set of research calls (engine v2)."""
    branches = [b for t in traces for b in t["branches"] if b["source"] != "combined"]
    calls = [c for t in traces for c in t["calls"] if c["kind"] == "chat"]
    answer = [
        c for c in calls if c["purpose"].rsplit(":", 1)[-1] in ("answer", "final", "conclude")
    ]
    conclude = [c for c in answer if c["purpose"].endswith(":conclude")]
    ended = [b["ended_s"] for b in branches if b.get("ended_s") is not None]
    return {
        "questions": len(traces), "branches": len(branches),
        "branch_end_s_p50_p95_max": [pct(ended, q) for q in (0.5, 0.95, 1.0)],
        "seconds_p50_p95_max": [pct([t["seconds"] for t in traces], q) for q in (0.5, 0.95, 1.0)],
        "reached_cut_off": len(conclude),  # still running at the cut-off: the conclusion call
        "unanswered": sum(b.get("ending") == "cutoff" for b in branches),  # no own answer at all
        "ended_by_budget_message": sum(b.get("ending") == "budget" for b in branches),
        "ended_by_time_message": sum(b.get("ending") == "time" for b in branches) - sum(not c.get("cancelled") for c in conclude),  # fmt: skip
        "answer_now_calls": len(answer),
        "answer_now_s_p50_p95_max": [pct([c["seconds"] for c in answer], q) for q in (0.5, 0.95, 1.0)],
        "answer_now_timeouts": sum(bool(c.get("cancelled")) for c in answer),
        "conclusion_timeouts": sum(bool(c.get("cancelled")) for c in conclude),
        "branch_chat_s_per_wall_s": pct([x for t in traces if len(t["branches"]) > 1 and (x := parallelism(t)) is not None], 0.5),  # fmt: skip
        "reports_not_written": sum(t["answer"].startswith("(report not written") for t in traces),
    }  # fmt: skip


def chat_numbers(traces: dict, correct: int, total_cost: float) -> dict:
    """Per question: chat tokens (reasoning and cached shares), turns, reported vs list cost;
    per branch: tool calls, how it ended; retries and cancellations."""
    n = max(1, len(traces))
    calls = [c for t in traces.values() for c in t["calls"] if c["kind"] == "chat"]
    jev = [c for t in traces.values() for c in t["calls"] if c["kind"] == "jev"]
    tokens_in, tokens_out = (
        sum(c["input_tokens"] for c in calls),
        sum(c["output_tokens"] for c in calls),
    )
    reasoning, cached = sum(c.get("reasoning_tokens", 0) for c in calls), sum(c.get("cached_tokens", 0) for c in calls)  # fmt: skip
    model = next((t["identity"]["chat_model"] for t in traces.values()), "")
    price = list_price(model)
    listed = (tokens_in * price[0] + tokens_out * price[1]) / 1e6
    reported = sum(c["cost_usd"] for c in calls if c.get("cost_reported"))
    branches = [b for t in traces.values() for b in t["branches"] if b["source"] != "combined"]
    budget = next((t["identity"].get("tool_budget") for t in traces.values()), None) or 8
    endings = {}
    for b in branches:
        endings[b.get("ending", "?")] = endings.get(b.get("ending", "?"), 0) + 1
    return {
        "chat_model": model, "list_price_per_m": price,
        "chat_input_tokens_per_q": round(tokens_in / n), "chat_output_tokens_per_q": round(tokens_out / n),
        "reasoning_share_of_output": round(reasoning / tokens_out, 3) if tokens_out else None,
        "cached_input_tokens_per_q": round(cached / n), "cached_share": round(cached / tokens_in, 3) if tokens_in else None,
        "chat_turns_per_q": round(sum(c["purpose"].startswith("branch") for c in calls) / n, 2),
        "chat_cost_reported_per_q": round(reported / n, 6), "chat_cost_at_list_price_per_q": round(listed / n, 6),
        "reported_over_list": round(reported / listed, 3) if listed else None,
        "tool_calls_per_branch": round(statistics.mean(b.get("tool_calls", 0) for b in branches), 2) if branches else None,
        "branches_using_all_calls": f"{sum(b.get('tool_calls', 0) >= budget for b in branches)}/{len(branches)}",
        "branch_endings": dict(sorted(endings.items())),
        "branch_end_s_p50_p95_max": [pct([b["ended_s"] for b in branches if b.get("ended_s") is not None], q) for q in (0.5, 0.95, 1.0)],  # fmt: skip
        "aggregation_s_p95": pct([t["seconds"] - max([b.get("ended_s") or 0 for b in t["branches"]] or [0]) for t in traces.values()], 0.95),  # fmt: skip
        "answers_partial": sum(bool(t.get("partial")) for t in traces.values()),
        "malformed_calls": sum(b.get("malformed_calls", 0) for b in branches),
        "retries": {"jev": sum(c.get("attempts", 1) - 1 for c in jev), "chat": sum(c.get("attempts", 1) - 1 for c in calls)},  # fmt: skip
        "cancelled_calls": sum(bool(c.get("cancelled")) for c in [*calls, *jev]),
        "cost_per_correct": round(total_cost / correct, 6) if correct else None,
    }  # fmt: skip


def run_numbers(run: Path) -> dict:
    rows = jsonl(run / "results.jsonl")
    traces = {r["id"]: json.loads((run / "traces" / f"{r['id']}.json").read_text()) for r in rows}
    mean = lambda key: round(statistics.mean(r.get(key, 0) or 0 for r in rows), 6)  # noqa: E731
    finds = sum(r["stats"]["find"] for r in rows)
    sql = sum(r["stats"]["sql"] for r in rows)
    checks = [t.get("verification") or {} for t in traces.values()]
    audit = {i: grounding(t) for i, t in traces.items()}
    correct = sum(bool(r["correct"]) for r in rows)
    graded = all(r["correct"] is not None for r in rows)
    multi = [r for r in rows if len(r["routes"]["chosen"]) >= 2]
    engine_cost = sum(t["usage"]["cost_usd"] for t in traces.values())
    return {
        **chat_numbers(traces, correct, engine_cost),
        "run": run.name, "tasks": len(rows), "correct": correct if graded else None,
        "timing": timing(list(traces.values())),
        "two_or_more_sources": {**timing([traces[r["id"]] for r in multi]), "correct": f"{sum(bool(r['correct']) for r in multi)}/{len(multi)}" if graded else None},  # fmt: skip
        "mean_cost_usd": mean("cost_usd"), "mean_cost_jev_usd": mean("cost_jev_usd"),
        "mean_cost_chat_usd": mean("cost_chat_usd"), "unmetered_upper_usd": round(sum(r.get("unmetered_upper_usd", 0) for r in rows), 6),  # fmt: skip
        "p50_seconds": pct([r["seconds"] for r in rows], 0.5), "p95_seconds": pct([r["seconds"] for r in rows], 0.95),  # fmt: skip
        "mean_turns": mean("turns"), "find_plans": finds, "model_sql": sql,
        "plan_share": round(finds / (finds + sql), 4) if finds + sql else None,
        "verification_changed": sum(bool(c.get("removed_citations") or c.get("regenerated") or c.get("stripped")) for c in checks),  # fmt: skip
        "verification_regenerated": sum(bool(c.get("regenerated")) for c in checks),
        "not_found": sum(c.get("verdict") == "declined" for c in checks),
        "guard_refusals": [x for r in rows for x in r["stats"].get("rejected", [])],
        "ungrounded_answers": {i: a for i, a in audit.items() if a["values"] or a["citations"]},
        "code_sha256": sorted({t["identity"].get("code_sha256") for t in traces.values()} - {None}),
        "config_sha256": sorted({t["identity"].get("config_sha256") for t in traces.values()}),
        "stores": sorted({json.dumps(r.get("store"), sort_keys=True) for r in rows}),
        "by_sources": {n: f"{sum(bool(r['correct']) for r in rows if len(r['routes']['chosen']) == n)}/{sum(len(r['routes']['chosen']) == n for r in rows)}"  # fmt: skip
                       for n in sorted({len(r["routes"]["chosen"]) for r in rows})},
        "_jev": [(c["seconds"], c["input_tokens"]) for t in traces.values() for c in t["calls"] if c["kind"] == "jev" and not c.get("cancelled")],  # fmt: skip
        "_jev_per_call": [sum(c["kind"] == "jev" for c in t["calls"]) for t in traces.values()],
    }  # fmt: skip


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    runs = [run_numbers(r) for r in args.runs]
    jev = [x for r in runs for x in r.pop("_jev")]
    per_call = [x for r in runs for x in r.pop("_jev_per_call")]
    overall = {"jev_calls": len(jev), "jev_p50_s": pct([s for s, _ in jev], 0.5), "jev_p95_s": pct([s for s, _ in jev], 0.95),  # fmt: skip
               "jev_input_tokens_median": statistics.median(t for _, t in jev) if jev else None,
               "jev_calls_per_research_median": statistics.median(per_call) if per_call else None,
               "correct_per_run": [r["correct"] for r in runs]}  # fmt: skip
    print(json.dumps({"runs": runs, "overall": overall}, indent=1))
    if args.json:
        args.json.write_text(json.dumps({"runs": runs, "overall": overall}, indent=1) + "\n")


if __name__ == "__main__":
    main()
