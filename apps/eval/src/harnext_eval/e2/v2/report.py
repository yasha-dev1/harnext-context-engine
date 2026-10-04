"""report.json + one self-contained report.html from a results directory. Works mid-run:
every number is computed from the trials recorded so far and says how many that is.

    python -m harnext_eval.e2.v2.report <results dir>
"""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from pathlib import Path

from harnext_eval.e2.benchmark import read

BASELINES = ("N", "R-hyb")
Z = 1.959963984540054


def wilson(successes, n):
    if not n:
        return None
    p = successes / n
    centre = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def mcnemar_exact(helped, harmed):
    """Two-sided exact binomial test on the discordant pairs."""
    n = helped + harmed
    if not n:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(helped, harmed) + 1)) / 2**n
    return min(1.0, 2 * tail)


CLI_TOOLS = ("gh", "jira", "mail", "git")


def rate(rows):
    n, k = len(rows), sum(r["success"] for r in rows)
    return {"n": n, "success": k, "rate": k / n if n else None, "wilson95": wilson(k, n)}


def cost(rows):
    n = len(rows) or 1
    total = lambda f: sum(f(r) for r in rows)  # noqa: E731
    return {
        "n": len(rows),
        "input_tokens": total(lambda r: r["usage"].get("input_tokens", 0)),
        "cached_input_tokens": total(lambda r: r["usage"].get("cached_input_tokens", 0)),
        "output_tokens": total(lambda r: r["usage"].get("output_tokens", 0)),
        "agent_seconds": round(total(lambda r: r["elapsed_s"]), 1),
        "grading_seconds": round(total(lambda r: r["grade"].get("seconds", 0)), 1),
        "tool_calls": total(lambda r: sum(r["tool_calls"].values())),
        "context_bytes": total(lambda r: r["context_output_bytes"]),
        **{  # shell invocations of the source CLIs and git (results before S-tools lack the key)
            f"{tool}_calls": total(lambda r, tool=tool: r.get("cli_calls", {}).get(tool, 0))
            for tool in CLI_TOOLS
        },
        "mean_input_tokens": round(total(lambda r: r["usage"].get("input_tokens", 0)) / n),
        "mean_agent_seconds": round(total(lambda r: r["elapsed_s"]) / n, 1),
        "mean_tool_calls": round(total(lambda r: sum(r["tool_calls"].values())) / n, 2),
    }


def paired(results, arm, baseline, kind=None):
    pairs = [
        (results[(t, baseline)]["success"], r["success"])
        for (t, a), r in results.items()
        if a == arm and (t, baseline) in results and (kind is None or r["kind"] == kind)
    ]
    helped = sum(not b and a for b, a in pairs)
    harmed = sum(b and not a for b, a in pairs)
    n = len(pairs)
    return {
        "pairs": n,
        "helped": helped,
        "harmed": harmed,
        "difference": (helped - harmed) / n if n else None,
        "mcnemar_exact_p": mcnemar_exact(helped, harmed),
    }


def load(results_dir):
    rows = [read(p) for p in sorted(results_dir.glob("trials/*/*/result.json"))]
    return {(r["task_id"], r["arm"]): r for r in rows}


def build(results_dir):
    results = load(results_dir)
    summary_path = results_dir / "summary.json"
    summary = read(summary_path) if summary_path.exists() else {}
    arms = list(summary.get("arms") or sorted({a for _, a in results}))
    kinds = sorted({r["kind"] for r in results.values()})
    accuracy, costs = {}, {}
    for arm in arms:
        rows = [r for r in results.values() if r["arm"] == arm]
        families = defaultdict(list)
        for row in rows:
            families[(row["kind"], row["family"])].append(row)
        accuracy[arm] = rate(rows) | {
            "by_kind": {
                kind: rate([r for r in rows if r["kind"] == kind])
                | {"by_family": {f: rate(v) for (k, f), v in sorted(families.items()) if k == kind}}
                for kind in kinds
            }
        }
        costs[arm] = cost(rows) | {
            "by_kind": {kind: cost([r for r in rows if r["kind"] == kind]) for kind in kinds}
        }
    comparisons = {
        base: {
            arm: paired(results, arm, base)
            | {"by_kind": {kind: paired(results, arm, base, kind) for kind in kinds}}
            for arm in arms
            if arm != base
        }
        for base in BASELINES
        if base in arms
    }
    trials = [
        {
            "task_id": r["task_id"],
            "arm": r["arm"],
            "kind": r["kind"],
            "family": r["family"],
            "status": r["status"],
            "success": r["success"],
            "input_tokens": r["usage"].get("input_tokens", 0),
            "output_tokens": r["usage"].get("output_tokens", 0),
            "seconds": round(r["elapsed_s"], 1),
            "tool_calls": sum(r["tool_calls"].values()),
            "context_paths": len(r["context_paths"]),
            "cli_calls": r.get("cli_calls", {}),
            "patch_empty": r["grade"].get("patch_empty"),
            "violations": r.get("violations", []),
        }
        for r in sorted(results.values(), key=lambda r: (r["run_index"], r["arm"]))
    ]
    return {
        "results_dir": results_dir.name,
        "run_status": summary.get("status", "unknown"),
        "recorded": len(results),
        "target": summary.get("target"),
        "partial": summary.get("status") != "complete",
        "arms": arms,
        "kinds": kinds,
        "accuracy": accuracy,
        "paired": comparisons,
        "cost": costs,
        "trials": trials,
    }


CSS = """
:root{--bg:#fff;--fg:#1a1d21;--muted:#667;--line:#dde1e6;--good:#1b7f4b;--bad:#b3261e;--bar:#4a78c2}
@media(prefers-color-scheme:dark){:root{--bg:#14171a;--fg:#e6e8ea;--muted:#9aa3ad;--line:#2c3238;
--good:#5fd39a;--bad:#ff8a80;--bar:#7aa5ea}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px}
main{max-width:1200px;margin:auto}h1{font-size:22px}h2{font-size:17px;margin-top:32px}
.note{color:var(--muted)}.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%}
th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap;
font-variant-numeric:tabular-nums}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600}.good{color:var(--good)}.bad{color:var(--bad)}
.bar{display:inline-block;height:8px;background:var(--bar);border-radius:2px;vertical-align:middle}
.banner{border:1px solid var(--line);border-radius:6px;padding:8px 12px;margin:12px 0}
"""


def pct(value):
    return "–" if value is None else f"{100 * value:.1f}%"


def rate_cell(cell):
    if not cell or not cell["n"]:
        return "<td>–</td>"
    low, high = cell["wilson95"]
    return (
        f"<td title='{cell['success']}/{cell['n']}'><span class='bar' style='width:"
        f"{round(60 * cell['rate'])}px'></span> {pct(cell['rate'])} "
        f"<span class='note'>[{pct(low)}, {pct(high)}] n={cell['n']}</span></td>"
    )


def table(head, rows):
    cells = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    return f"<div class='wrap'><table><tr>{cells}</tr>{''.join(rows)}</table></div>"


def render(report):
    e = html.escape
    arms, kinds = report["arms"], report["kinds"]
    parts = [
        f"<h1>E2 v2 — {e(report['results_dir'])}</h1>",
        f"<div class='banner'>Run status <b>{e(str(report['run_status']))}</b>: "
        f"{report['recorded']} of {report['target'] or '?'} trials recorded."
        + (" <b>Partial run</b> — arms may cover different task subsets; paired rows use only "
           "tasks recorded under both arms." if report["partial"] else "")
        + "</div>",
        "<h2>Success rate by arm and kind (Wilson 95 %)</h2>",
        table(
            ["arm", "all", *kinds],
            [
                f"<tr><td>{e(a)}</td>{rate_cell(report['accuracy'][a])}"
                + "".join(rate_cell(report["accuracy"][a]["by_kind"].get(k)) for k in kinds)
                + "</tr>"
                for a in arms
            ],
        ),
    ]  # fmt: skip
    for kind in kinds:
        families = sorted(
            {f for a in arms for f in report["accuracy"][a]["by_kind"][kind]["by_family"]}
        )
        if len(families) > 1:
            parts += [
                f"<h2>{e(kind)}: success rate by family</h2>",
                table(
                    ["arm", *families],
                    [
                        f"<tr><td>{e(a)}</td>"
                        + "".join(
                            rate_cell(report["accuracy"][a]["by_kind"][kind]["by_family"].get(f))
                            for f in families
                        )
                        + "</tr>"
                        for a in arms
                    ],
                ),
            ]
    for base, rows in report["paired"].items():
        scopes = [("all", lambda c: c)] + [(k, lambda c, k=k: c["by_kind"][k]) for k in kinds]
        body = []
        for arm, cell in rows.items():
            tds = ""
            for _, pick in scopes:
                c = pick(cell)
                if not c["pairs"]:
                    tds += "<td>–</td>"
                    continue
                tone = "good" if c["difference"] > 0 else "bad" if c["difference"] < 0 else ""
                tds += (
                    f"<td><span class='{tone}'>{100 * c['difference']:+.1f} pp</span> "
                    f"<span class='note'>+{c['helped']}/−{c['harmed']} of {c['pairs']}, "
                    f"p={c['mcnemar_exact_p']:.3g}</span></td>"
                )
            body.append(f"<tr><td>{e(arm)}</td>{tds}</tr>")
        parts += [
            f"<h2>Paired difference vs {e(base)} (exact McNemar)</h2>",
            table(["arm", *(name for name, _ in scopes)], body),
        ]
    parts += [
        "<h2>Cost</h2>",
        table(
            ["arm", "n", "input tok", "cached", "output tok", "mean input", "agent s", "mean s",
             "grading s", "tool calls", "mean calls", "context bytes read", "gh", "jira", "mail",
             "git"],
            [
                f"<tr><td>{e(a)}</td>"
                + "".join(
                    f"<td>{report['cost'][a][k]:,}</td>"
                    for k in ("n", "input_tokens", "cached_input_tokens", "output_tokens",
                              "mean_input_tokens", "agent_seconds", "mean_agent_seconds",
                              "grading_seconds", "tool_calls", "mean_tool_calls", "context_bytes",
                              *(f"{tool}_calls" for tool in CLI_TOOLS))
                )
                + "</tr>"
                for a in arms
            ],
        ),
        "<h2>Trials</h2>",
        table(
            ["task", "arm", "kind", "family", "status", "ok", "input tok", "output tok", "s",
             "tools", "gh/jira/mail/git", "ctx paths", "empty patch", "violations"],
            [
                f"<tr><td>{e(t['task_id'])}</td><td>{e(t['arm'])}</td><td>{e(t['kind'])}</td>"
                f"<td>{e(t['family'])}</td><td>{e(t['status'])}</td>"
                f"<td class='{'good' if t['success'] else 'bad'}'>{'✓' if t['success'] else '✗'}</td>"
                f"<td>{t['input_tokens']:,}</td><td>{t['output_tokens']:,}</td><td>{t['seconds']}</td>"
                f"<td>{t['tool_calls']}</td>"
                f"<td>{'/'.join(str(t['cli_calls'].get(c, 0)) for c in CLI_TOOLS)}</td>"
                f"<td>{t['context_paths']}</td>"
                f"<td>{'' if t['patch_empty'] is None else t['patch_empty']}</td>"
                f"<td>{e('; '.join(t['violations']))}</td></tr>"
                for t in report["trials"]
            ],
        ),
    ]  # fmt: skip
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' "
        f"content='width=device-width,initial-scale=1'><title>E2 v2 report</title><style>{CSS}"
        f"</style><main>{''.join(parts)}</main></html>"
    )


def write(results_dir):
    report = build(results_dir)
    (results_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (results_dir / "report.html").write_text(render(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    report = write(parser.parse_args().results)
    print(json.dumps({"recorded": report["recorded"], "partial": report["partial"]}))


if __name__ == "__main__":
    main()
