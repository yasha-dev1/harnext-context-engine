"""One command for the E2 results page, driven by the run config, so every publish is built the
same way from the same inputs:

    .venv/bin/python apps/eval/scripts/e2_charts.py --config apps/eval/configs/e2-w2026-claude-haiku.yaml

Writes `apps/eval/reports/e2/<run_id>/charts.html` (a stable path: republishing that file keeps
the artifact's URL). The trial charts come from `e2_v2_graphs.py`; this script adds what the run
config knows and the trial archives do not: progress against the planned trials, the historical
part in the four family groups of E2-V2-DESIGN.md (never per single family), and the store builds
(progress through the scope's event stream, cost, failures). Read-only: no model calls, no writes
outside the output file.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e2_v2_graphs as graphs  # noqa: E402
from harnext_eval.e2.v2 import runner as run_mod  # noqa: E402
from harnext_eval.e2.v2 import stream  # noqa: E402

EVAL = HERE.parent
# E2-V2-DESIGN.md: historical results are reported as a whole and in these four groups.
GROUPS = {
    "State lookups": ["assignee_at_snapshot", "status_at_snapshot", "resolution_at_snapshot",
                      "closed_at_snapshot", "priority_at_snapshot", "fix_version_at_snapshot"],
    "Change histories": ["assignment_history", "status_transition_history",
                         "resolution_transition_history", "reopen_count"],
    "Cross-source / multi-hop": ["issue_prs", "issue_changed_files", "pr_approvers",
                                 "review_hotspot", "issue_links", "resolution_vs_merge_order",
                                 "kip_thread"],
    "Aggregation": ["author_activity", "top_reviewer_for_author", "open_blockers_unassigned"],
}  # fmt: skip
GROUP_OF = {family: group for group, families in GROUPS.items() for family in families}
STORE_KIND = {"N": "none", "S-tools": "offline gh / jira / mail replicas", "R-hyb": "hybrid RAG index",
              "F0": "raw event files", "F1": "templated files",
              "H-engine": "raw store (ClickHouse) + query-time engine",
              "H-engine-mem": "raw store + query-time engine + F5 notes",
              "H-engine-b8": "raw store + engine v2 (gpt-oss-20b, tool-call budget feedback 8)",
              "H-engine-b8-ds": "raw store + engine v2 (DeepSeek V4 Flash, tool-call budget feedback 8)"}  # fmt: skip
STORE_DIR = {"H-engine": "RAW", "H-engine-mem": "RAW", "H-engine-b8": "RAW", "H-engine-b8-ds": "RAW"}  # kind engine reads the RAW export
EXTRA_CSS = """
table.t{border-collapse:collapse;font-size:13px;width:100%;font-variant-numeric:tabular-nums}
table.t th,table.t td{padding:6px 10px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}
table.t th:first-child,table.t td:first-child{text-align:left}
table.t th{color:var(--ink3);font-weight:500;font-size:12px;letter-spacing:.02em}
table.st td:nth-child(2),table.st th:nth-child(2){text-align:left;white-space:normal;color:var(--ink2)}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;border:1px solid var(--rule);color:var(--ink2)}
.pill.ok{border-color:var(--good);color:var(--good)}.pill.run{border-color:var(--warn);color:var(--warn)}
.bar{display:inline-block;height:8px;border-radius:4px;background:var(--series);vertical-align:1px}
.track{display:inline-block;width:120px;height:8px;border-radius:4px;background:var(--mute);margin-right:8px;vertical-align:1px;overflow:hidden}
.kpis{display:flex;flex-wrap:wrap;gap:12px}.kpi{background:var(--panel);border:1px solid var(--rule);padding:12px 16px;min-width:150px}
.kpi b{display:block;font-size:24px;font-variant-numeric:tabular-nums;color:var(--ink)}.kpi span{font-size:12px;color:var(--ink3)}
"""


def pct(k, n):
    return f"{k}/{n} · {k / n:.0%}" if n else "–"


def planned(config):
    """(recorded per arm, planned per arm) from the runner's own plan: no docker, no model."""
    runner = run_mod.Runner(config)
    left = Counter(arm for _, _, arm in runner.remaining())
    done = Counter(arm for _, arm in runner.results)
    arms = [a["id"] for a in runner.arms]
    return arms, {a: done[a] for a in arms}, {a: done[a] + left[a] for a in arms}


def event_times(scope_dir: Path):
    times = []
    with (scope_dir / "events.jsonl").open() as lines:
        for line in lines:
            times.append(json.loads(line)["time"])
    return sorted(times)


def store_rows(config, arms, times):
    """One row per arm: build progress through the scope's stream, cost, failures."""
    import bisect

    months = sorted({t[:7] for t in times})
    out = []
    for arm in arms:
        row = {"arm": arm, "kind": STORE_KIND.get(arm, "LLM-built files"), "progress": 1.0,
               "cost": 0.0, "failed": 0, "state": "ready", "seconds": 0.0}  # fmt: skip
        state_file = config.stores / STORE_DIR.get(arm, arm) / "build-state.json"
        if arm == "N":
            row["kind"] = "no store (closed book)"
        elif arm == "S-tools":
            pass  # the systems of record: time-frozen replicas, built once for every scope
        elif not state_file.exists():
            row.update(progress=0.0, state="not started")
        else:
            state = json.loads(state_file.read_text())
            usage = state.get("usage") or {}
            row["cost"] = usage.get("cost_usd", 0.0)
            row["seconds"] = usage.get("seconds", state.get("seconds", 0.0))
            row["failed"] = len(state.get("failed_windows") or [])
            if "done" in state:  # file stores: a cursor through the stream
                done = bool(state["done"])
                last = state.get("last_event_time") or ""
                row["progress"] = 1.0 if done else bisect.bisect_right(times, last) / max(len(times), 1)
            else:  # R-hyb: one index shard per month
                row["progress"] = len(state.get("checkpoints") or {}) / max(len(months), 1)
                done = row["progress"] >= 1.0
            row["state"] = "ready" if done else "building"
        out.append(row)
    return out


def store_table(rows):
    head = ("<table class='t st'><tr><th>Arm</th><th>Store</th><th>Ingested</th>"
            "<th>Cost</th><th>Time</th><th>Failed</th><th>State</th></tr>")  # fmt: skip
    body = []
    for r in rows:
        share = r["progress"] or 0.0
        pill = {"ready": "ok", "building": "run"}.get(r["state"], "")
        body.append(
            f"<tr><td>{graphs.esc(r['arm'])}</td><td>{graphs.esc(r['kind'])}</td>"
            f"<td><span class='track'><span class='bar' style='width:{share * 100:.1f}%'></span></span>{share:.0%}</td>"
            f"<td>${r['cost']:,.2f}</td><td>{r['seconds'] / 3600:,.1f} h</td><td>{r['failed']}</td>"
            f"<td><span class='pill {pill}'>{graphs.esc(r['state'])}</span></td></tr>"
        )
    return head + "".join(body) + "</table>"


def group_table(rows, arms):
    """Historical accuracy per arm in the four family groups (never per single family)."""
    hist = [r for r in rows if r["kind"] == "historical"]
    if not hist:
        return ""
    have = [a for a in arms if any(r["arm"] == a for r in hist)]
    cells = defaultdict(lambda: [0, 0])
    for r in hist:
        group = GROUP_OF.get(r.get("family"), "Other")
        cells[(r["arm"], group)][0] += bool(r["success"])
        cells[(r["arm"], group)][1] += 1
        cells[(r["arm"], "All")][0] += bool(r["success"])
        cells[(r["arm"], "All")][1] += 1
    groups = [g for g in GROUPS if any(cells[(a, g)][1] for a in have)] + ["All"]
    out = ["<table class='t'><tr><th>Arm</th>" + "".join(f"<th>{graphs.esc(g)}</th>" for g in groups) + "</tr>"]
    for a in have:
        out.append(f"<tr><td>{graphs.esc(a)}</td>" + "".join(f"<td>{pct(*cells[(a, g)])}</td>" for g in groups) + "</tr>")
    return "".join(out) + "</table>"


def money(value, spec):
    """As e2_report.money: a non-zero cost never prints as $0.000 (engine costs are fractions of a cent)."""
    import e2_report  # noqa: PLC0415

    return e2_report.money(value, spec)


def cost_table(rows, arms):
    """Per trial: the agent's own model cost and the store's read agent (context_research; for the
    engine arms the engine's Jev + chat cost, recorded the same way in the receipts)."""
    have = [a for a in arms if any(r["arm"] == a for r in rows)]
    out = ["<table class='t'><tr><th>Arm</th><th>Trials</th><th>Agent $ / trial (median)</th>"
           "<th>Read agent $ / trial (median)</th><th>Total $ so far</th></tr>"]  # fmt: skip
    for a in have:
        mine = [r for r in rows if r["arm"] == a]
        agent = [(r.get("usage") or {}).get("cost_usd") or 0.0 for r in mine]
        reader = [sum((((c.get("research") or {}).get("usage") or {}).get("cost_usd") or 0.0)
                      + ((c.get("research") or {}).get("provider_retry_usd") or 0.0) for c in r["mcp"]) for r in mine]  # incl. discarded engine attempts
        out.append(f"<tr><td>{graphs.esc(a)}</td><td>{len(mine)}</td><td>{money(statistics.median(agent), '.3f')}</td>"
                   f"<td>{money(statistics.median(reader), '.3f')}</td><td>{money(sum(agent) + sum(reader), ',.2f')}</td></tr>")  # fmt: skip
    return "".join(out) + "</table>"


def merge_runs(extras, arms, done, target):
    """Trials of further runs beside the main one; an arm id already shown gets its run id."""
    import e2_report  # noqa: PLC0415 - the provenance rule is the paper report's

    rows, runs = [], []
    for path in extras:
        cfg = run_mod.load(Path(path))
        res = cfg.results or run_mod.ROOT.parents[1] / ".harnext/evaluations/kafka-500-v2" / cfg.run_id
        x_arms, x_done, x_target = planned(cfg)
        names = {a: a if a not in arms else f"{a} ({cfg.run_id})" for a in x_arms}
        for a in x_arms:
            arms.append(names[a])
            done[names[a]], target[names[a]] = x_done[a], x_target[a]
        rows += [r | {"arm": names[r["arm"]]} for r in graphs.load(res, cfg.benchmark) if r["arm"] in names]
        runs.append(e2_report.provenance(cfg, res, [names[a] for a in x_arms]))
    return rows, runs


def provenance_section(runs):
    body = "".join(f"<tr><td><code>{graphs.esc(r['run'])}</code></td><td style='text-align:left'>{graphs.esc(', '.join(r['arms']))}</td>"
                   f"<td>{graphs.esc(r['dates'])}</td><td><code>{graphs.esc(r['image'])}</code></td>"
                   f"<td style='text-align:left'>{graphs.esc(', '.join(r['allowlist']))}</td></tr>" for r in runs)  # fmt: skip
    return ("<section><h2>Runs combined on this page</h2><p>Arms from different runs share the benchmark and the agent, "
            "not the date, the image or the proxy; an arm with a run id in parentheses is a re-run in the later run.</p>"
            "<div class='panel'><table class='t'><tr><th>Run</th><th>Arms</th><th>Dates (UTC)</th><th>Image</th><th>Proxy allowlist</th></tr>"
            f"{body}</table></div></section>")  # fmt: skip


def build(config_path: Path, out: Path | None, title: str, extras=()):
    config = run_mod.load(config_path)
    results = config.results or run_mod.ROOT.parents[1] / ".harnext/evaluations/kafka-500-v2" / config.run_id
    rows = graphs.load(results, config.benchmark)
    arms, done, target = planned(config)
    runs = []
    if extras:
        import e2_report  # noqa: PLC0415

        main_arms = list(arms)
        extra_rows, runs = merge_runs(extras, arms, done, target)
        rows += extra_rows
        runs = [e2_report.provenance(config, results, main_arms)] + runs
    manifest = json.loads((config.benchmark / "manifest.json").read_text())
    scope = manifest.get("scope", "?")
    stream_dir = stream.OUT / stream.SCOPES[scope]["dir"] if scope in stream.SCOPES else None
    times = event_times(stream_dir) if stream_dir else []
    every_arm = [a["id"] for a in run_mod.load_arms(config.arms_file)[0]]
    stores = store_rows(config, every_arm, times)
    newest = max((p.stat().st_mtime for p in results.glob("trials/*/*/result.json")), default=0)
    as_of = datetime.fromtimestamp(newest, timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if newest else "no trials yet"
    window = stream.SCOPES.get(scope, {}).get("window", (None, None))
    recorded, total = sum(done.values()), sum(target.values())
    intro = (
        f"<section><h1>{graphs.esc(title)}</h1><p>Run <code>{graphs.esc(config.run_id)}</code> · agent "
        f"{graphs.esc(config.agent)} · model {graphs.esc(config.model)} · benchmark "
        f"<code>{graphs.esc(config.benchmark.name)}</code> ({manifest.get('historical', '?')} historical + "
        f"{manifest.get('coding', '?')} coding tasks). Stores ingest {graphs.esc((window[0] or 'the first event')[:10])} "
        f"to {graphs.esc((window[1] or '')[:10])} (exclusive), {len(times):,} events. Latest trial: {as_of}.</p>"
        "<div class='kpis'>"
        f"<div class='kpi'><b>{recorded}/{total}</b><span>trials recorded in this chunk</span></div>"
        f"<div class='kpi'><b>{len({r['task_id'] for r in rows})}</b><span>tasks with a result</span></div>"
        f"<div class='kpi'><b>{sum(1 for s in stores if s['state'] == 'ready')}/{len(stores)}</b><span>arms with a finished store</span></div>"
        f"<div class='kpi'><b>${sum(s['cost'] for s in stores):,.0f}</b><span>store builds so far (API-equivalent)</span></div>"
        "</div></section>"
    )  # fmt: skip
    progress = ("<section><h2>Progress per arm</h2><p>Trials recorded against the trials planned for "
                "the current chunk. Arms without a finished store are not scheduled yet.</p><div class='panel'>"
                "<table class='t'><tr><th>Arm</th><th>Recorded</th><th>Planned</th></tr>"
                + "".join(f"<tr><td>{graphs.esc(a)}</td><td>{done[a]}</td><td>{target[a]}</td></tr>" for a in arms)
                + "</table></div></section>")  # fmt: skip
    extra = [intro, progress] + ([provenance_section(runs)] if runs else [])
    if rows:
        groups = group_table(rows, arms)
        if groups:
            extra.append("<section><h2>Historical: accuracy by family group</h2><p>Correct answers over trials. "
                         "Each group holds only a few tasks per arm, so read these as a rough profile, "
                         f"not a ranking.</p><div class='panel'>{groups}</div></section>")  # fmt: skip
        extra.append("<section><h2>Model cost per trial</h2><p>What Claude Code reports at API prices. "
                     "The read agent is the store's own <code>context_research</code> agent (file arms only).</p>"
                     f"<div class='panel'>{cost_table(rows, arms)}</div></section>")  # fmt: skip
    extra.append("<section><h2>Store builds</h2><p>How far each store has ingested the window's event "
                 "stream, what building it cost at API prices (builds run on the subscription), and how many builder windows failed (their events are kept "
                 "unorganised in the store). A store's arm joins the run once its build is finished.</p>"
                 f"<div class='panel'>{store_table(stores)}</div></section>")  # fmt: skip
    head = f"<title>{graphs.esc(title)}</title><style>{graphs.CSS}{EXTRA_CSS}</style><div class='wrap'>"
    if rows:
        body = graphs.page(rows, "S-tools", title)
        body = body[body.index("</section>", body.index("<div class='wrap'>")) + len("</section>"):]
    else:
        body = "<section><p>No trials recorded yet.</p></section></div>"
    out = out or EVAL / "reports/e2" / config.run_id / "charts.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(head + "".join(extra) + body)
    return out, recorded, total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="the run's yaml")
    parser.add_argument("--out", type=Path, help="default: apps/eval/reports/e2/<run_id>/charts.html")
    parser.add_argument("--title", default="E2 Kafka Window Results")
    parser.add_argument("--extra-config", type=Path, action="append", default=[],
                        help="another run's config whose arms are shown beside this run's (repeatable)")  # fmt: skip
    args = parser.parse_args()
    out, recorded, total = build(args.config, args.out, args.title, args.extra_config)
    print(out, f"{recorded}/{total} trials")


if __name__ == "__main__":
    main()
