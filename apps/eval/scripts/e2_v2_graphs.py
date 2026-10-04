"""Graphs for an E2 v2 results directory: one self-contained HTML page (inline SVG, no assets).

    python apps/eval/scripts/e2_v2_graphs.py <results dir> [--out graphs.html] [--baseline S-tools]

Reads the per-trial archives (`trials/<task>/<arm>/result.json`, `mcp-receipts.jsonl`), so it
works on a partial run and says how many trials each number rests on.
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ARM_ORDER = ["N", "S-tools", "R-hyb", "G-graphiti", "F0", "F1", "F2", "F3",
             "F4-fact", "F4-entity", "F4-topic", "F5", "F6",
             "F0-search", "F1-search", "F2-search", "F3-search", "F4-fact-search",
             "F4-entity-search", "F4-topic-search", "F5-search", "F6-search"]  # fmt: skip
CSS = """
:root{--ground:#f6f7f9;--panel:#fff;--ink:#141922;--ink2:#4d5766;--ink3:#7a8494;--rule:#dce0e7;
--series:#2a78d6;--good:#0a7d2c;--bad:#c23030;--warn:#b7791f;--mute:#c9ced6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--ground:#12151a;--panel:#1a1e25;
--ink:#f1f3f6;--ink2:#b6bdc9;--ink3:#8a93a1;--rule:#2c323c;--series:#3987e5;--good:#4cc26a;--bad:#ef7070;--warn:#e0b55a;--mute:#3a414c}}
:root[data-theme="dark"]{--ground:#12151a;--panel:#1a1e25;--ink:#f1f3f6;--ink2:#b6bdc9;--ink3:#8a93a1;
--rule:#2c323c;--series:#3987e5;--good:#4cc26a;--bad:#ef7070;--warn:#e0b55a;--mute:#3a414c}
body{background:var(--ground);color:var(--ink);font:15px/1.55 "IBM Plex Sans",system-ui,sans-serif;
padding-inline:20px;padding-block:32px 64px}
.wrap{max-width:960px;margin:0 auto;display:flex;flex-direction:column;gap:36px}
h1{font-size:30px;line-height:1.15;margin:0}h2{font-size:20px;margin:0}p{margin:0;max-width:72ch;color:var(--ink2)}
section{display:flex;flex-direction:column;gap:12px;min-width:0}
.panel{background:var(--panel);border:1px solid var(--rule);padding:16px;overflow-x:auto}
svg text{fill:var(--ink2);font:12px "IBM Plex Mono",ui-monospace,monospace}
svg .lab{fill:var(--ink);font:13px "IBM Plex Sans",system-ui,sans-serif}
.grid{stroke:var(--rule);stroke-width:1}.mark{fill:var(--series)}.ci{stroke:var(--series);stroke-width:2}
.good{fill:var(--good)}.bad{fill:var(--bad)}.warn{fill:var(--warn)}.mute{fill:var(--mute)}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--ink2)}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:5px 10px;border-bottom:1px solid var(--rule);text-align:right}th:first-child,td:first-child{text-align:left}
th{color:var(--ink3);font-weight:500;font-size:11.5px;text-transform:uppercase;letter-spacing:.05em}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
"""


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    centre, half = (
        (p + z * z / (2 * n)) / d,
        z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d,
    )
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar(helped, harmed):
    n = helped + harmed
    if not n:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(helped, harmed) + 1)) / 2**n
    return min(1.0, 2 * tail)


def golden(benchmark: Path):
    path = benchmark / "tasks.private.jsonl"
    return {json.loads(x)["id"]: json.loads(x) for x in path.open()} if path.exists() else {}


def similarity_of(folder: Path, row, private):
    """The grader's similarity block, or the same computation on the saved patch for trials
    recorded before the grader had it."""
    grade = row.get("grade") or {}
    if grade.get("similarity") or row["kind"] != "coding" or row["task_id"] not in private:
        return grade.get("similarity")
    patch = next((p for p in (folder / "patch.diff", folder / "patch.diff.gz") if p.exists()), None)
    if patch is None:
        return None
    from harnext_eval.e2.v2.grade import similarity, split_patch, without_paths

    text = (gzip.open(patch, "rt") if patch.suffix == ".gz" else patch.open()).read()
    task = private[row["task_id"]]
    main, _ = without_paths(text, {p for p, _ in split_patch(task["test_patch"])})
    return similarity(main, task["reference_patch"])


def load(results: Path, benchmark: Path | None = None):
    private = golden(benchmark) if benchmark else {}
    rows = []
    for path in sorted(results.glob("trials/*/*/result.json")):
        row = json.loads(path.read_text())
        calls = []
        receipts = path.with_name("mcp-receipts.jsonl")
        if receipts.exists():
            for line in receipts.read_text().splitlines():
                item = json.loads(line)
                if item.get("tool"):
                    calls.append(item)
        row["mcp"] = calls
        row["turns"] = turns_of(path.parent, row)
        row["similarity"] = similarity_of(path.parent, row, private)
        rows.append(row)
    return rows


CODEX_ACTIONS = {"agent_message", "command_execution", "mcp_tool_call", "file_change", "web_search"}


def turns_of(folder: Path, row) -> int:
    """Model responses in the saved trace, counted the way the turn cap counts them: Claude Code
    = distinct assistant message ids (it streams one event per content block); Codex = completed
    messages and tool actions, reasoning excluded. Falls back to the recorded field."""
    trace = next(
        (p for p in (folder / "trace.jsonl.gz", folder / "trace.jsonl") if p.exists()), None
    )
    if trace is None:
        return row.get("model_turns") or 0
    opener = gzip.open if trace.suffix == ".gz" else open
    ids, actions = set(), 0
    with opener(trace, "rt", errors="replace") as lines:
        for line in lines:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "assistant":
                ids.add((event.get("message") or {}).get("id") or len(ids))
            elif event.get("type") == "item.completed":
                actions += (event.get("item") or {}).get("type") in CODEX_ACTIONS
    return len(ids) or actions


def strip_chart(rows, arms, cap):
    """Every trial's turn count as a dot, per arm; tick = median; dashed line = the proposed cap."""
    top = max([r["turns"] for r in rows] + [cap]) * 1.05
    left, width, row_h = 110, 560, 30
    height = row_h * len(arms) + 40
    out = [f'<svg viewBox="0 0 {left + width + 210} {height}" width="100%" role="img">']
    step = 50 if top <= 400 else 100
    for tick in range(0, int(top) + 1, step):
        x = left + width * tick / top
        out.append(f'<line class="grid" x1="{x}" y1="8" x2="{x}" y2="{height - 28}"/>')
        out.append(f'<text x="{x}" y="{height - 10}" text-anchor="middle">{tick}</text>')
    x_cap = left + width * cap / top
    out.append(
        f'<line x1="{x_cap}" y1="4" x2="{x_cap}" y2="{height - 28}" stroke="var(--bad)" stroke-width="1.5" stroke-dasharray="5 4"/>'
    )
    out.append(f'<text x="{x_cap + 5}" y="12" style="fill:var(--bad)">cap {cap}</text>')
    for i, arm in enumerate(arms):
        mine = sorted(r["turns"] for r in rows if r["arm"] == arm)
        y = 24 + i * row_h
        out.append(
            f'<text class="lab" x="{left - 12}" y="{y + 4}" text-anchor="end">{esc(arm)}</text>'
        )
        for r in (r for r in rows if r["arm"] == arm):
            tip = f"{r['task_id']} · {arm}: {r['turns']} turns, {'resolved/correct' if r['success'] else r['status'] if r['status'] != 'completed' else 'not resolved/wrong'}"
            out.append(
                f'<circle class="{"good" if r["success"] else "bad"}" cx="{left + width * r["turns"] / top}" cy="{y}" r="4.5" opacity=".7"><title>{esc(tip)}</title></circle>'
            )
        if mine:
            mid = left + width * median(mine) / top
            over = sum(v > cap for v in mine)
            out.append(
                f'<line x1="{mid}" y1="{y - 10}" x2="{mid}" y2="{y + 10}" stroke="var(--ink)" stroke-width="2"/>'
            )
            out.append(
                f'<text x="{left + width + 10}" y="{y + 4}">median {median(mine):.0f} · {over}/{len(mine)} over</text>'
            )
    return "".join(out) + "</svg>"


def esc(value):
    return html.escape(str(value), quote=True)


def arms_of(rows):
    seen = {r["arm"] for r in rows}
    return [a for a in ARM_ORDER if a in seen] + sorted(seen - set(ARM_ORDER))


def rate_chart(rows, arms, title_unit):
    """Dot + 95 % Wilson interval per arm, one shared 0–100 % axis."""
    left, width, row_h = 110, 560, 30
    height = row_h * len(arms) + 40
    out = [f'<svg viewBox="0 0 {left + width + 130} {height}" width="100%" role="img">']
    for tick in range(0, 101, 25):
        x = left + width * tick / 100
        out.append(f'<line class="grid" x1="{x}" y1="8" x2="{x}" y2="{height - 28}"/>')
        out.append(f'<text x="{x}" y="{height - 10}" text-anchor="middle">{tick} %</text>')
    for i, arm in enumerate(arms):
        mine = [r for r in rows if r["arm"] == arm]
        k, n = sum(bool(r["success"]) for r in mine), len(mine)
        low, high = wilson(k, n)
        y = 22 + i * row_h
        tip = (
            f"{arm}: {k} of {n} {title_unit} ({k / n:.0%}), 95 % interval {low:.0%}–{high:.0%}"
            if n
            else arm
        )
        out.append(
            f'<g><title>{esc(tip)}</title><text class="lab" x="{left - 12}" y="{y + 4}" text-anchor="end">{esc(arm)}</text>'
        )
        if n:
            out.append(
                f'<line class="ci" x1="{left + width * low}" y1="{y}" x2="{left + width * high}" y2="{y}"/>'
            )
            out.append(
                f'<circle class="mark" cx="{left + width * k / n}" cy="{y}" r="5" stroke="var(--panel)" stroke-width="2"/>'
            )
            out.append(f'<text x="{left + width + 10}" y="{y + 4}">{k}/{n} · {k / n:.0%}</text>')
        out.append("</g>")
    return "".join(out) + "</svg>"


def paired_chart(rows, arms, baseline):
    """Per arm vs the baseline on the same tasks: helped to the right, harmed to the left."""
    by = {(r["task_id"], r["arm"]): bool(r["success"]) for r in rows}
    data = []
    for arm in arms:
        if arm == baseline:
            continue
        tasks = [t for (t, a) in by if a == arm and (t, baseline) in by]
        helped = sum(by[(t, arm)] and not by[(t, baseline)] for t in tasks)
        harmed = sum(by[(t, baseline)] and not by[(t, arm)] for t in tasks)
        data.append((arm, len(tasks), helped, harmed))
    span = max([max(h, m) for _, _, h, m in data] + [1])
    left, half, row_h = 110, 260, 30
    centre, height = left + half, row_h * len(data) + 40
    out = [f'<svg viewBox="0 0 {left + 2 * half + 170} {height}" width="100%" role="img">',
           f'<line class="grid" x1="{centre}" y1="6" x2="{centre}" y2="{height - 28}"/>',
           f'<text x="{centre - 8}" y="{height - 10}" text-anchor="end">← worse than {esc(baseline)}</text>',
           f'<text x="{centre + 8}" y="{height - 10}">better than {esc(baseline)} →</text>']  # fmt: skip
    for i, (arm, n, helped, harmed) in enumerate(data):
        y = 12 + i * row_h
        tip = f"{arm} vs {baseline} on {n} shared tasks: better on {helped}, worse on {harmed}, exact McNemar p = {mcnemar(helped, harmed):.3f}"
        out.append(
            f'<g><title>{esc(tip)}</title><text class="lab" x="{left - 12}" y="{y + 14}" text-anchor="end">{esc(arm)}</text>'
        )
        out.append(
            f'<rect class="good" x="{centre + 1}" y="{y}" width="{half * helped / span}" height="18" rx="3"/>'
        )
        out.append(
            f'<rect class="bad" x="{centre - 1 - half * harmed / span}" y="{y}" width="{half * harmed / span}" height="18" rx="3"/>'
        )
        out.append(
            f'<text x="{left + 2 * half + 10}" y="{y + 14}">+{helped} / −{harmed} · n={n} · p={mcnemar(helped, harmed):.2f}</text></g>'
        )
    return "".join(out) + "</svg>"


def bar_chart(values, unit, fmt):
    """Horizontal bars, one series, value labels; `values` = [(arm, number, n)]."""
    top = max([v for _, v, _ in values] + [1e-9])
    left, width, row_h = 96, 190, 24
    out = [
        f'<svg viewBox="0 0 {left + width + 84} {row_h * len(values) + 8}" width="100%" role="img">'
    ]
    for i, (arm, value, n) in enumerate(values):
        y = 4 + i * row_h
        out.append(f'<g><title>{esc(f"{arm}: {fmt(value)} {unit} (median over {n} trials)")}</title>'
                   f'<text class="lab" x="{left - 12}" y="{y + 13}" text-anchor="end">{esc(arm)}</text>'
                   f'<rect class="mark" x="{left}" y="{y}" width="{max(2, width * value / top)}" height="16" rx="3"/>'
                   f'<text x="{left + width * value / top + 8}" y="{y + 13}">{fmt(value)}</text></g>')  # fmt: skip
    return "".join(out) + "</svg>"


def matrix(rows, arms):
    """Tasks × arms: what happened in every single trial."""
    tasks = sorted(
        {r["task_id"] for r in rows},
        key=lambda t: (min(r["run_index"] for r in rows if r["task_id"] == t), t),
    )  # task id breaks run_index ties (a run's appended tasks reuse indices): stable across runs
    by = {(r["task_id"], r["arm"]): r for r in rows}
    cell, left, top = 26, 150, 70
    out = [f'<svg viewBox="0 0 {left + cell * len(arms) + 20} {top + cell * len(tasks) + 10}" '
           f'width="{left + cell * len(arms) + 20}" role="img">']  # fmt: skip
    for j, arm in enumerate(arms):
        x = left + j * cell + cell / 2
        out.append(f'<text transform="translate({x + 4},{top - 8}) rotate(-50)">{esc(arm)}</text>')
    for i, task in enumerate(tasks):
        y = top + i * cell
        out.append(
            f'<text x="{left - 8}" y="{y + 17}" text-anchor="end">{esc(task.replace("C-KAFKA-", "").replace("H-", "")[:20])}</text>'
        )
        for j, arm in enumerate(arms):
            r = by.get((task, arm))
            x = left + j * cell
            if r is None:
                out.append(
                    f'<rect class="mute" x="{x + 2}" y="{y + 2}" width="{cell - 4}" height="{cell - 4}" rx="4" opacity=".35"><title>{esc(task)} · {esc(arm)}: not run yet</title></rect>'
                )
                continue
            grade = r.get("grade") or {}
            if r["success"]:
                kind, glyph, note = "good", "✓", "resolved" if r["kind"] == "coding" else "correct"
            elif r["status"] == "timeout":
                kind, glyph, note = "warn", "…", "timed out"
            elif r["kind"] == "coding" and grade.get("patch_empty"):
                kind, glyph, note = "mute", "∅", "no patch"
            else:
                kind, glyph, note = "bad", "✗", "not resolved" if r["kind"] == "coding" else "wrong"
            counts = grade.get("counts") or {}
            extra = (
                f"; tests {counts.get('passed', 0)} passed / {counts.get('failed', 0)} failed"
                if counts
                else ""
            )
            tip = f"{task} · {arm}: {note}{extra}; {r.get('elapsed_s', 0):.0f} s, {(r.get('usage') or {}).get('input_tokens', 0):,} input tokens"
            out.append(f'<g><title>{esc(tip)}</title><rect class="{kind}" x="{x + 2}" y="{y + 2}" width="{cell - 4}" height="{cell - 4}" rx="4"/>'
                       f'<text x="{x + cell / 2}" y="{y + 18}" text-anchor="middle" style="fill:#fff;font-size:13px">{glyph}</text></g>')  # fmt: skip
    return "".join(out) + "</svg>"


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else 0


def research_table(rows, arms):
    lines = ["<table><tr><th>Arm</th><th>Trials</th><th>Calls / trial</th><th>Seconds / call</th>"
             "<th>Files read / call</th><th>Reader tokens / call</th><th>Trials with no call</th></tr>"]  # fmt: skip
    for arm in arms:
        mine = [r for r in rows if r["arm"] == arm]
        calls = [c for r in mine for c in r["mcp"]]
        if not any(c.get("research") or c.get("tool") for c in calls) and not calls:
            continue
        info = [c.get("research") or {} for c in calls]
        tokens = [
            sum(v for v in (i.get("usage") or {}).values() if isinstance(v, (int, float)))
            for i in info
        ]
        lines.append(f"<tr><td>{esc(arm)}</td><td>{len(mine)}</td><td>{len(calls) / max(1, len(mine)):.1f}</td>"
                     f"<td>{median([i.get('seconds') or c.get('ms', 0) / 1000 for i, c in zip(info, calls, strict=True)]):.1f}</td>"
                     f"<td>{median([len(i.get('files') or []) for i in info]):.0f}</td>"
                     f"<td>{median(tokens):,.0f}</td><td>{sum(not r['mcp'] for r in mine)}</td></tr>")  # fmt: skip
    return "".join(lines) + "</table>" if len(lines) > 1 else ""


def page(rows, baseline, name, cap=100):
    arms = arms_of(rows)
    parts = [f"<title>E2 Run Graphs</title><style>{CSS}</style><div class='wrap'>",
             f"<section><h1>{esc(name)}</h1><p>{len(rows)} recorded trials over "
             f"{len({r['task_id'] for r in rows})} tasks and {len(arms)} arms · agent "
             f"{esc(rows[0]['agent'])} · model {esc(rows[0]['model'])}. Every figure is computed from the "
             "trial archives and counts only trials whose agent and grader both finished on working "
             "infrastructure.</p></section>"]  # fmt: skip
    for kind, unit, heading in (("coding", "tasks resolved", "Coding: tasks resolved (every test of the golden PR passes)"),
                                ("historical", "answers correct", "Historical: exact-match accuracy")):  # fmt: skip
        mine = [r for r in rows if r["kind"] == kind]
        if not mine:
            continue
        have = [a for a in arms if any(r["arm"] == a for r in mine)]
        parts.append(f"<section><h2>{heading}</h2><p>Dot = share, line = 95 % Wilson interval. With few "
                     "tasks the intervals are wide; overlapping intervals are not a difference.</p>"
                     f"<div class='panel'>{rate_chart(mine, have, unit)}</div></section>")  # fmt: skip
        if baseline in have and len(have) > 1:
            parts.append(f"<section><h2>{kind.capitalize()}: each arm against {esc(baseline)} on the same tasks</h2>"
                         "<p>Green = tasks the arm got and the baseline missed; red = the reverse. Ties are not drawn. "
                         "p is the exact McNemar test.</p>"
                         f"<div class='panel'>{paired_chart(mine, have, baseline)}</div></section>")  # fmt: skip
        cost = [("Input tokens per trial (median, incl. cached)", "tokens", lambda r: (r.get("usage") or {}).get("input_tokens"), lambda v: f"{v:,.0f}"),
                ("Agent seconds per trial (median)", "s", lambda r: r.get("elapsed_s"), lambda v: f"{v:,.0f} s"),
                ("Tool calls per trial (median)", "calls", lambda r: sum((r.get("tool_calls") or {}).values()), lambda v: f"{v:,.0f}")]  # fmt: skip
        parts.append(
            f"<section><h2>{kind.capitalize()}: what each arm costs the agent</h2><div class='cols'>"
        )
        for label, unit_, get, fmt in cost:
            values = [
                (
                    a,
                    median([get(r) for r in mine if r["arm"] == a]),
                    sum(r["arm"] == a for r in mine),
                )
                for a in have
            ]
            parts.append(
                f"<div class='panel'><p style='margin-bottom:8px'>{label}</p>{bar_chart(values, unit_, fmt)}</div>"
            )
        parts.append("</div></section>")
        scored = [r for r in mine if r.get("similarity")]
        if scored:
            measures = [("Files changed: F1 against the golden patch's files (median)", lambda s: s["files"]["f1"]),
                        ("Location: share of the golden change's place in the code the agent also edited (median)", lambda s: s["location_recall"]),
                        ("Added lines identical to the golden patch: F1 (median)", lambda s: s["added_lines"]["f1"])]  # fmt: skip
            parts.append("<section><h2>Coding: how close the change is to the golden patch</h2><p>Descriptive, never "
                         "part of <em>resolved</em>: a different correct fix scores low here. It shows whether the "
                         "agent found the right files and the right place, also on tasks it did not resolve.</p>"
                         "<div class='cols'>")  # fmt: skip
            for label, get in measures:
                values = [(a, median([get(r["similarity"]) for r in scored if r["arm"] == a]),
                           sum(r["arm"] == a for r in scored)) for a in have]  # fmt: skip
                parts.append(f"<div class='panel'><p style='margin-bottom:8px'>{label}</p>"
                             f"{bar_chart(values, '', lambda v: f'{v:.2f}')}</div>")  # fmt: skip
            parts.append("</div></section>")
        parts.append(f"<section><h2>{kind.capitalize()}: agent turns per trial</h2><p>One dot per trial "
                     "(green = resolved / correct). A turn is one model response: a message or a tool action. "
                     f"Black tick = median; dashed line = a cap of {cap}. Trials to the right of the line "
                     "would have been stopped early.</p>"
                     f"<div class='panel'>{strip_chart(mine, have, cap)}</div></section>")  # fmt: skip
        parts.append(f"<section><h2>{kind.capitalize()}: every trial</h2><div class='legend'>"
                     "<span><i class='sw' style='background:var(--good)'></i>✓ resolved / correct</span>"
                     "<span><i class='sw' style='background:var(--bad)'></i>✗ not resolved / wrong</span>"
                     "<span><i class='sw' style='background:var(--warn)'></i>… timed out</span>"
                     "<span><i class='sw' style='background:var(--mute)'></i>∅ no patch · faint = not run yet</span></div>"
                     f"<div class='panel'>{matrix(mine, have)}</div></section>")  # fmt: skip
    table = research_table(rows, arms)
    if table:
        parts.append("<section><h2>How the context engine was used</h2><p>From the server-side receipts of every "
                     "MCP call: for the file arms one call is one <code>context_research</code> question answered "
                     f"by Harnext's read agent.</p><div class='panel'>{table}</div></section>")  # fmt: skip
    by_status = defaultdict(int)
    for r in rows:
        by_status[r["status"]] += 1
    parts.append(f"<section><p>Trial statuses: {esc(dict(by_status))}.</p></section></div>")
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--baseline", default="S-tools")
    parser.add_argument("--name")
    parser.add_argument("--benchmark", type=Path,
                        default=Path(__file__).resolve().parents[1] / "benchmarks/kafka-500-v2")  # fmt: skip
    parser.add_argument("--cap", type=int, default=100, help="turn cap drawn on the turns chart")
    args = parser.parse_args()
    rows = load(args.results, args.benchmark)
    if not rows:
        raise SystemExit("no recorded trials under " + str(args.results))
    out = args.out or args.results / "graphs.html"
    out.write_text(page(rows, args.baseline, args.name or args.results.name, args.cap))
    print(out, len(rows), "trials")


if __name__ == "__main__":
    main()
