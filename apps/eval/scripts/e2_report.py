"""The E2 paper report: every chart and table of the thesis's E2 chapter, from the trial archives.

    .venv/bin/python apps/eval/scripts/e2_report.py --config apps/eval/configs/e2-w2026-claude-haiku.yaml
    # arms of a later run beside it (provenance table: run id, dates, image, proxy per run):
    .venv/bin/python apps/eval/scripts/e2_report.py --config <main run yaml> --extra-config <later run yaml>

Writes apps/eval/reports/e2/<run_id>/report.html (one self-contained page, inline SVG, both
themes). Read-only: it never calls a model and never writes outside the output file. Numbers come
from trials/<task>/<arm>/{result.json, answer.json, mcp-receipts.jsonl}, the benchmark's private
tasks, the stores' build-state.json and the side runs named below (leak smoke, retired arms).
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e2_v2_graphs as graphs  # noqa: E402
from harnext_eval.e2.v2 import runner as run_mod  # noqa: E402

EVAL = HERE.parent
ROOT = EVAL.parents[1]
EVALS = ROOT / ".harnext/evaluations/kafka-500-v2"
esc = html.escape

# ---- taxonomy ------------------------------------------------------------------------------------
GROUPS = {  # E2-V2-DESIGN.md: historical results by family group, never per single family
    "State lookups": ["assignee_at_snapshot", "status_at_snapshot", "resolution_at_snapshot",
                      "closed_at_snapshot", "priority_at_snapshot", "fix_version_at_snapshot"],
    "Change histories": ["assignment_history", "status_transition_history",
                         "resolution_transition_history", "reopen_count"],
    "Cross-source / multi-hop": ["issue_prs", "issue_changed_files", "pr_approvers", "review_hotspot",
                                 "issue_links", "resolution_vs_merge_order", "kip_thread"],
    "Aggregation": ["author_activity", "top_reviewer_for_author", "open_blockers_unassigned"],
    "Design decisions": ["design_decision"],
    "Implicit links": ["implicit_link"],
}  # fmt: skip
GROUP_OF = {f: g for g, fs in GROUPS.items() for f in fs}
CATEGORIES = ["Factual history", "Design decisions", "Implicit links", "Coding"]
ARM_TYPE = {"N": "Closed book", "S-tools": "Systems of record", "R-hyb": "Retrieval / raw files",
            "F0-search": "Retrieval / raw files", "F1-search": "Retrieval / raw files",
            "F2-search": "Retrieval / raw files", "F3-search": "Retrieval / raw files",
            "F4-fact-search": "LLM-curated", "F4-entity-search": "LLM-curated",
            "F4-topic-search": "LLM-curated", "F5-search": "LLM-curated", "F6-search": "LLM-curated",
            "H-engine": "Raw store + decision model", "H-engine-mem": "Raw store + decision model",
            "H-engine-b8": "Raw store + decision model", "H-engine-b8-ds": "Raw store + decision model"}  # fmt: skip
TYPE_CLASS = {"Closed book": "t0", "Systems of record": "t1", "Retrieval / raw files": "t2", "LLM-curated": "t3",
              "Raw store + decision model": "t4"}  # fmt: skip
ENGINE_STORE = {"H-engine": "RAW", "H-engine-mem": "RAW", "H-engine-b8": "RAW", "H-engine-b8-ds": "RAW"}  # kind engine: the RAW export (+ F5 notes)
STORE_OF = lambda arm: ENGINE_STORE.get(arm, arm.replace("-search", ""))  # noqa: E731
# Only pages that show an engine arm get the fifth colour, so every other page stays byte-identical.
ENGINE_CSS = ":root{--t4:#8a5cd0}@media (prefers-color-scheme: dark){:root:not([data-theme=\"light\"]){--t4:#a57fe0}}:root[data-theme=\"dark\"]{--t4:#a57fe0}.t4{fill:var(--t4);stroke:var(--t4)}"
ARM_LABEL = {"F5-search": "F5-search (Harnext)"}
DESCRIPTION = {
    "N": "no project history (closed book)",
    "S-tools": "time-frozen offline gh / jira / mail CLIs over the systems of record",
    "R-hyb": "hybrid RAG: BM25 + dense embeddings, RRF, cross-encoder rerank",
    "F0-search": "raw event files, one per event",
    "F1-search": "templated files per entity (code-rendered)",
    "F2-search": "LLM-designed folders, event files moved intact",
    "F3-search": "LLM reorganisation under a keep-every-fact rule",
    "F4-fact-search": "LLM-curated, one file per fact",
    "F4-entity-search": "LLM-curated, one file per entity",
    "F4-topic-search": "LLM-curated, one large file per topic",
    "F5-search": "Harnext as shipped: the product's builder, seed layout and prompts",
    "F6-search": "LLM agent curates freely from an empty store",
    "H-engine": "every source kept raw in ClickHouse + a file catalog; at question time Jev 1.13 routes and "
    "fills query slots, gpt-oss-20b researches each source and writes the cited answer (same context_research tool)",
    "H-engine-mem": "H-engine plus the F5 store (Harnext's curated files) read as notes beside the raw sources",
    "H-engine-b8": "engine v2 over the raw store: chat model gpt-oss-20b (OpenRouter), tool-call budget feedback 8, "
    "research deadline 240 s (same context_research tool)",
    "H-engine-b8-ds": "engine v2 over the raw store: chat model DeepSeek V4 Flash (deepseek-v4-flash-0731, OpenRouter), "
    "tool-call budget feedback 8, research deadline 240 s (same context_research tool)",
}

CSS = """
:root{--bg:#f6f6f3;--panel:#fcfcfb;--ink:#15171a;--ink2:#52514e;--ink3:#86857f;--rule:#e1e0da;
--t0:#9a9993;--t1:#2a78d6;--t2:#1baf7a;--t3:#eb6834;--good:#1f8a4c;--bad:#c73a3a;--warn:#b7791f;--heat:42,120,214}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121316;--panel:#1a1a19;--ink:#f3f3f1;
--ink2:#c3c2b7;--ink3:#8f8e87;--rule:#2e2e2b;--t0:#7c7b75;--t1:#3987e5;--t2:#199e70;--t3:#d95926;
--good:#4cc26a;--bad:#ef7070;--warn:#e0b55a;--heat:57,135,229;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#121316;--panel:#1a1a19;--ink:#f3f3f1;--ink2:#c3c2b7;--ink3:#8f8e87;--rule:#2e2e2b;
--t0:#7c7b75;--t1:#3987e5;--t2:#199e70;--t3:#d95926;--good:#4cc26a;--bad:#ef7070;--warn:#e0b55a;--heat:57,135,229;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font:15px/1.6 "Source Serif 4",Georgia,serif;padding-inline:18px;padding-block:36px 80px;margin:0}
main{max-width:1000px;margin:0 auto;display:flex;flex-direction:column;gap:40px}
h1{font:600 32px/1.15 "Source Sans 3",system-ui,sans-serif;margin:0;text-wrap:balance}
h2{font:600 22px/1.25 "Source Sans 3",system-ui,sans-serif;margin:0;text-wrap:balance}
h3{font:600 16px/1.3 "Source Sans 3",system-ui,sans-serif;margin:0}
p{margin:0;max-width:74ch;color:var(--ink2)} section{display:flex;flex-direction:column;gap:14px;min-width:0}
.fig{background:var(--panel);border:1px solid var(--rule);padding:16px 18px;display:flex;flex-direction:column;gap:10px;overflow-x:auto}
.cap{font:13px/1.5 "Source Sans 3",system-ui,sans-serif;color:var(--ink2);max-width:none}
.cap b{color:var(--ink)} .num{font-variant-numeric:tabular-nums}
svg text{font:12px "Source Sans 3",system-ui,sans-serif;fill:var(--ink2)} svg .lab{fill:var(--ink);font-size:12.5px}
svg .grid{stroke:var(--rule);stroke-width:1} svg .axis{stroke:var(--ink3);stroke-width:1}
.t0{fill:var(--t0);stroke:var(--t0)}.t1{fill:var(--t1);stroke:var(--t1)}.t2{fill:var(--t2);stroke:var(--t2)}.t3{fill:var(--t3);stroke:var(--t3)}
.ci{stroke-width:2;fill:none}
table{border-collapse:collapse;font:13px/1.45 "Source Sans 3",system-ui,sans-serif;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 9px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left} th{color:var(--ink3);font-weight:600;font-size:12px;letter-spacing:.02em}
td.heat{color:var(--ink)} .legend{display:flex;gap:16px;flex-wrap:wrap;font:13px "Source Sans 3",system-ui,sans-serif;color:var(--ink2)}
.sw{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.kpi{background:var(--panel);border:1px solid var(--rule);padding:12px 14px;font-family:"Source Sans 3",system-ui,sans-serif}
.kpi b{display:block;font-size:24px;font-variant-numeric:tabular-nums;color:var(--ink)} .kpi span{font-size:12.5px;color:var(--ink3)}
.note{border-left:3px solid var(--warn);padding:8px 12px;background:var(--panel);font:13.5px/1.55 "Source Sans 3",system-ui,sans-serif;color:var(--ink2)}
.good{color:var(--good)}.bad{color:var(--bad)} code{font-size:12.5px}
ol,ul{margin:0;padding-left:20px;color:var(--ink2);max-width:74ch} li{margin:3px 0}
"""


# ---- data ----------------------------------------------------------------------------------------
def reader_usage(folder: Path):
    cost = tin = tout = 0.0
    turns = calls = 0
    path = folder / "mcp-receipts.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("tool") == "context_research":
                r = row.get("research") or {}
                u = r.get("usage") or {}
                cost += u.get("cost_usd") or 0
                tin += (u.get("input_tokens") or 0) + (u.get("cache_read_tokens") or 0) + (u.get("cache_write_tokens") or 0)
                tout += u.get("output_tokens") or 0
                turns += r.get("turns") or 0
            if row.get("tool"):
                calls += 1
    return cost, tin, tout, turns, calls


def provider_retry(folder: Path):
    """Engine arms: cost of research attempts discarded because of provider trouble (redone, never
    shown to the agent), from the receipts; 0 for every other arm."""
    path = folder / "mcp-receipts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return sum((r.get("research") or {}).get("provider_retry_usd") or 0 for r in rows if r.get("tool") == "context_research")  # fmt: skip


def unmetered(folder: Path):
    """Engine arms: upper bound of model requests that returned no usage (failed or cancelled)."""
    path = folder / "mcp-receipts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return sum(((r.get("research") or {}).get("usage") or {}).get("unmetered_upper_usd") or 0 for r in rows if r.get("tool") == "context_research")  # fmt: skip


def load_trials(results: Path, private: dict, arms: list[str]):
    rows = []
    for path in results.glob("trials/*/*/result.json"):
        r = json.loads(path.read_text())
        if r["arm"] not in arms or r["task_id"] not in private:
            continue
        t = private[r["task_id"]]
        folder = path.parent
        answer_file = folder / "answer.json"
        answer = json.loads(answer_file.read_text()).get("value") if answer_file.exists() else None
        u = r.get("usage") or {}
        rc, rin, rout, rturns, calls = reader_usage(folder)
        retry = provider_retry(folder)
        family = t.get("family")
        category = "Coding" if r["kind"] == "coding" else (
            "Design decisions" if family == "design_decision" else "Implicit links" if family == "implicit_link" else "Factual history")
        rows.append({
            "task": r["task_id"], "arm": r["arm"], "kind": r["kind"], "family": family, "category": category,
            "group": GROUP_OF.get(family, "Coding"), "ok": bool(r["success"]), "answer": answer,
            "expected": (t.get("expected") or {}).get("value"), "agent_cost": u.get("cost_usd") or 0.0,
            "reader_cost": rc, "cost": (u.get("cost_usd") or 0.0) + rc + retry, "retry_cost": retry, "agent_in": u.get("input_tokens") or 0,
            "agent_cached": u.get("cached_input_tokens") or 0, "agent_out": u.get("output_tokens") or 0,
            "reader_in": rin, "reader_out": rout, "reader_turns": rturns, "mcp_calls": calls,
            "cli_calls": sum((r.get("cli_calls") or {}).values()), "turns": r.get("model_turns") or 0,
            "secs": r["elapsed_s"], "similarity": (r.get("grade") or {}).get("similarity"), "status": r["status"],
            "unmetered": unmetered(folder),
        })  # fmt: skip
    return rows


def lenient_equal(expected, answer):
    """Normalised match for short free answers (documented in the report): case, spacing and
    punctuation ignored; one typo allowed in identifiers of 10+ characters."""
    if answer is None or expected is None:
        return False
    norm = lambda s: re.sub(r"[^a-z0-9]+", "", str(s).lower())  # noqa: E731
    a, b = norm(expected), norm(answer)
    if a == b:
        return True
    if len(a) >= 10 and abs(len(a) - len(b)) <= 1:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1] <= 1
    return False


def money(value, spec):
    """`$` + value in the given format, except that a non-zero value the format would print as
    zero ("$0", "$0.00") gets two significant digits instead ("$0.0010"); every other figure is
    printed exactly as before."""
    text = f"${value:{spec}}"
    if not value or any(ch not in "$0.,-" for ch in text):
        return text
    return f"${value:.{max(0, 1 - math.floor(math.log10(abs(value)))) + 0}f}"


ENGINE_TYPE = "Raw store + decision model"
ALLOWLIST_BEFORE = ["api.openai.com", "chatgpt.com", "auth.openai.com", "api.anthropic.com"]  # proxy until 2026-09-28


def provenance(config, results: Path, arms: list[str]) -> dict:
    """Run id, trial dates, image ID and proxy allowlist of one run, from its archives."""
    manifest = json.loads((results / "manifest.json").read_text()) if (results / "manifest.json").exists() else {}
    records = [json.loads(p.read_text()) for p in results.glob("trials/*/*/result.json")]
    times = sorted(p.stat().st_mtime for p in results.glob("trials/*/*/result.json"))
    engine = manifest.get("engine") or {}
    images = sorted({r["image_id"] for r in records if r.get("image_id")}) or ([engine["image_id"]] if engine.get("image_id") else [])
    allow = engine.get("proxy_allowlist")
    allow = [re.sub(r"[\\^$]", "", a) for a in allow] if allow else ALLOWLIST_BEFORE
    day = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")  # noqa: E731
    return {"run": config.run_id, "arms": arms, "dates": f"{day(times[0])} – {day(times[-1])}" if times else "–",
            "image": ", ".join(i.removeprefix("sha256:")[:12] for i in images) or "not recorded (tag harnext-e2-agent)",
            "allowlist": allow}  # fmt: skip


def merge_runs(extras, arms: list[str]):
    """Rows of further runs next to the main one: an arm id already shown gets its run id in the
    label (e.g. an S-tools drift control); the type/description tables follow the base arm."""
    rows, runs = [], []
    for path in extras:
        cfg = run_mod.load(Path(path))
        res = cfg.results or EVALS / cfg.run_id
        private = {json.loads(l)["id"]: json.loads(l) for l in (cfg.benchmark / "tasks.private.jsonl").open()}  # noqa: E741
        pub = {json.loads(l)["id"]: json.loads(l) for l in (cfg.benchmark / "tasks.public.jsonl").open()}  # noqa: E741
        order = json.loads((cfg.benchmark / "manifest.json").read_text())["run_order"]
        scope = set([t for t in order if pub[t]["kind"] == "coding"][: cfg.max_coding or 10**9])
        names = {}
        for arm in cfg.arms:
            key = arm if arm not in arms else f"{arm} ({cfg.run_id})"
            names[arm] = key
            ARM_TYPE[key], DESCRIPTION[key] = ARM_TYPE[arm], DESCRIPTION.get(arm, "")
            ARM_LABEL.setdefault(key, ARM_LABEL.get(arm, key) if key == arm else key)
            ENGINE_STORE.setdefault(key, ENGINE_STORE.get(arm, arm.replace("-search", "")))
            arms.append(key)
        for r in load_trials(res, private, list(cfg.arms)):
            if r["kind"] != "coding" or r["task"] in scope:
                rows.append(r | {"arm": names[r["arm"]]})
        runs.append(provenance(cfg, res, [names[a] for a in cfg.arms]))
    return rows, runs


# ---- statistics ----------------------------------------------------------------------------------
wilson = graphs.wilson
mcnemar = graphs.mcnemar


def med(values):
    values = [v for v in values if v is not None]
    return st.median(values) if values else None


# ---- charts --------------------------------------------------------------------------------------
def dot_ci_chart(rows, arms, title_unit):
    left, width, rh = 170, 520, 26
    h = rh * len(arms) + 46
    out = [f'<svg viewBox="0 0 {left + width + 120} {h}" width="100%" role="img" aria-label="{esc(title_unit)} by arm">']
    for tick in range(0, 101, 20):
        x = left + width * tick / 100
        out.append(f'<line class="grid" x1="{x}" y1="10" x2="{x}" y2="{h - 30}"/><text x="{x}" y="{h - 12}" text-anchor="middle">{tick} %</text>')
    for i, arm in enumerate(arms):
        mine = [r for r in rows if r["arm"] == arm]
        k, n = sum(r["ok"] for r in mine), len(mine)
        y = 22 + i * rh
        cls = TYPE_CLASS[ARM_TYPE[arm]]
        out.append(f'<g><title>{esc(arm)}: {k}/{n} {esc(title_unit)}</title><text class="lab" x="{left - 12}" y="{y + 4}" text-anchor="end">{esc(ARM_LABEL.get(arm, arm))}</text>')
        if n:
            lo, hi = wilson(k, n)
            out.append(f'<line class="ci {cls}" x1="{left + width * lo}" y1="{y}" x2="{left + width * hi}" y2="{y}"/>'
                       f'<circle class="{cls}" cx="{left + width * k / n}" cy="{y}" r="5.5" stroke="var(--panel)" stroke-width="2"/>'
                       f'<text x="{left + width + 10}" y="{y + 4}" class="num">{k}/{n} · {k / n:.0%}</text>')  # fmt: skip
        out.append("</g>")
    return "".join(out) + "</svg>"


def bar_chart(values, fmt, unit, ref=None):
    """values = [(arm, value, note)]: horizontal bars coloured by arm type."""
    top = max([v for _, v, _ in values if v is not None] + [1e-9])
    left, width, rh = 170, 460, 24
    h = rh * len(values) + 12
    out = [f'<svg viewBox="0 0 {left + width + 150} {h}" width="100%" role="img" aria-label="{esc(unit)} by arm">']
    if ref is not None:
        x = left + width * ref / top
        out.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{h}" stroke="var(--ink3)" stroke-dasharray="3 3"/>')
    for i, (arm, v, note) in enumerate(values):
        y = 5 + i * rh
        cls = TYPE_CLASS[ARM_TYPE[arm]]
        w = 0 if v is None else max(2, width * v / top)
        out.append(f'<g><title>{esc(arm)}: {fmt(v) if v is not None else "–"} {esc(unit)} {esc(note or "")}</title>'
                   f'<text class="lab" x="{left - 12}" y="{y + 13}" text-anchor="end">{esc(ARM_LABEL.get(arm, arm))}</text>'
                   f'<rect class="{cls}" x="{left}" y="{y + 2}" width="{w}" height="15" rx="3" stroke="none"/>'
                   f'<text x="{left + w + 8}" y="{y + 13}" class="num">{fmt(v) if v is not None else "–"}{esc(note or "")}</text></g>')  # fmt: skip
    return "".join(out) + "</svg>"


def scatter(points, xlabel, ylabel, xfmt):
    """points = [(arm, x, y)]: cost (x) vs accuracy (y); every point labelled."""
    W, H, L, B, T, R = 760, 380, 60, 44, 16, 150
    xs = [x for _, x, _ in points if x is not None]
    xmax = max(xs + [1e-9]) * 1.12
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{esc(ylabel)} against {esc(xlabel)}">']
    for t in range(0, 101, 20):
        y = T + (H - T - B) * (1 - t / 100)
        out.append(f'<line class="grid" x1="{L}" y1="{y}" x2="{W - R}" y2="{y}"/><text x="{L - 8}" y="{y + 4}" text-anchor="end">{t} %</text>')
    for i in range(0, 6):
        v = xmax * i / 5
        x = L + (W - L - R) * i / 5
        out.append(f'<line class="grid" x1="{x}" y1="{T}" x2="{x}" y2="{H - B}"/><text x="{x}" y="{H - B + 16}" text-anchor="middle">{xfmt(v)}</text>')
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 6}" text-anchor="middle">{esc(xlabel)}</text>')
    out.append(f'<text x="14" y="{(T + H - B) / 2}" transform="rotate(-90 14 {(T + H - B) / 2})" text-anchor="middle">{esc(ylabel)}</text>')
    placed = []
    for arm, x, y in sorted(points, key=lambda p: -(p[2] or 0)):
        if x is None or y is None:
            continue
        px = L + (W - L - R) * x / xmax
        py = T + (H - T - B) * (1 - y)
        ly = py + 4
        while any(abs(ly - q) < 13 and abs(px - qx) < 115 for qx, q in placed):
            ly += 13
        placed.append((px, ly))
        cls = TYPE_CLASS[ARM_TYPE[arm]]
        out.append(f'<g><title>{esc(arm)}: {y:.0%} at {xfmt(x)}</title><circle class="{cls}" cx="{px}" cy="{py}" r="6" stroke="var(--panel)" stroke-width="2"/>'
                   f'<text x="{px + 10}" y="{ly}" class="lab">{esc(arm)}</text></g>')  # fmt: skip
    return "".join(out) + "</svg>"


def heat_table(rows, arms, cols, value, fmt, head="Arm"):
    cells = []
    for a in arms:
        vals = [value(a, c) for c in cols]
        tds = []
        for (v, n) in vals:
            if n == 0 or v is None:
                tds.append("<td>–</td>")
                continue
            alpha = 0.08 + 0.55 * max(0.0, min(1.0, v))
            tds.append(f'<td class="heat" style="background:rgba(var(--heat),{alpha:.2f})">{fmt(v, n)}</td>')
        cells.append(f"<tr><td>{esc(ARM_LABEL.get(a, a))}</td>{''.join(tds)}</tr>")
    return (f"<table><tr><th>{esc(head)}</th>" + "".join(f"<th>{esc(c)}</th>" for c in cols) + "</tr>" + "".join(cells) + "</table>")


def legend(arms=()):
    items = [("t0", "Closed book"), ("t1", "Systems of record"), ("t2", "Retrieval / raw & light files"), ("t3", "LLM-curated stores")]
    if any(ARM_TYPE.get(a) == "Raw store + decision model" for a in arms):
        items.append(("t4", "Raw store + decision model"))
    return "<div class='legend'>" + "".join(f"<span><i class='sw' style='background:var(--{c})'></i>{esc(n)}</span>" for c, n in items) + "</div>"


def fig(inner, caption):
    return f"<div class='fig'>{inner}<p class='cap'>{caption}</p></div>"


# ---- report --------------------------------------------------------------------------------------
def build(config_path: Path, out: Path | None, extras=()):
    config = run_mod.load(config_path)
    arms = list(config.arms)
    main_arms = list(arms)
    results = config.results or EVALS / config.run_id
    private = {json.loads(l)["id"]: json.loads(l) for l in (config.benchmark / "tasks.private.jsonl").open()}
    manifest = json.loads((config.benchmark / "manifest.json").read_text())
    pub = {json.loads(l)["id"]: json.loads(l) for l in (config.benchmark / "tasks.public.jsonl").open()}
    coding_scope = set([t for t in manifest["run_order"] if pub[t]["kind"] == "coding"][: config.max_coding or 10**9])
    rows = [r for r in load_trials(results, private, arms) if r["kind"] != "coding" or r["task"] in coding_scope]
    extra_rows, runs = merge_runs(extras, arms) if extras else ([], [])
    rows += extra_rows
    engine_view = any(ARM_TYPE.get(a) == ENGINE_TYPE for a in arms)
    by = defaultdict(list)
    for r in rows:
        by[(r["arm"], r["category"])].append(r)
    deps_file = config.benchmark / "dependencies.json"
    deps = json.loads(deps_file.read_text()) if deps_file.exists() else {}
    newest = max((p.stat().st_mtime for p in results.glob("trials/*/*/result.json")), default=0)
    as_of = datetime.fromtimestamp(newest, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_tasks = Counter(r["category"] for r in rows if r["arm"] == "S-tools")
    total_trials = len(rows)
    total_cost = sum(r["cost"] for r in rows)

    stores = {}
    for arm in arms:
        s = STORE_OF(arm)
        state = EVAL / "out/kafka-500-v2/stores-w2026" / s / "build-state.json"
        if state.exists():
            d = json.loads(state.read_text())
            u = d.get("usage") or {}
            stores[s] = {"cost": u.get("cost_usd", 0.0), "hours": (u.get("seconds") or d.get("seconds") or 0) / 3600,
                         "failed": len(d.get("failed_windows") or []), "windows": d.get("window")}  # fmt: skip
    build_cost = {a: stores.get(STORE_OF(a), {}).get("cost", 0.0) for a in arms}
    per_task_build = {a: build_cost[a] / max(1, len({r["task"] for r in rows if r["arm"] == a})) for a in arms}

    P = []
    engine_css = ENGINE_CSS if any(ARM_TYPE.get(a) == "Raw store + decision model" for a in arms) else ""
    P.append(f"<title>Harnext E2 Evaluation Report</title><style>{CSS}{engine_css}</style><main>")
    # 1. header
    P.append(f"""<section><h1>Harnext E2: project memory for coding agents</h1>
<p>Apache Kafka, January–May 2026 (31,566 events from JIRA, GitHub and the dev mailing list). Agent: Claude Code with
<code>{esc(config.model)}</code>, one trial per task and arm, turn cap {config.max_turns}. Every figure below is computed
from the trial archives of run <code>{esc(config.run_id)}</code> by <code>apps/eval/scripts/e2_report.py</code>; latest trial {as_of}.
Costs are API-equivalent (what the tokens would cost at list price).</p>
<div class="kpis"><div class="kpi"><b class="num">{total_trials:,}</b><span>trials in this report</span></div>
<div class="kpi"><b class="num">{len(arms)}</b><span>arms</span></div>
<div class="kpi"><b class="num">{sum(n_tasks.values())}</b><span>tasks ({', '.join(f'{v} {k.lower()}' for k, v in n_tasks.items())})</span></div>
<div class="kpi"><b class="num">{money(total_cost, ',.0f')}</b><span>trial cost · {money(sum(s['cost'] for s in stores.values()), ',.0f')} store builds</span></div></div></section>""")  # fmt: skip

    # 2. setup
    arm_rows = "".join(f"<tr><td>{esc(ARM_LABEL.get(a, a))}</td><td style='text-align:left'>{esc(ARM_TYPE[a])}</td><td style='text-align:left;white-space:normal'>{esc(DESCRIPTION.get(a, ''))}</td></tr>" for a in arms)
    cats = {
        "Factual history": "Exact values from the systems of record at a snapshot: states, change histories with timestamps, cross-source joins (issue → PRs → files → approvers), aggregations. Generated from the stream, every question passes a scripted answerability audit with the offline CLIs.",
        "Design decisions": "The final value of an architecture decision settled in a long KIP mailing-list discussion, described without naming the KIP, ticket or PR; most changed during the discussion. Proposed by Claude Sonnet 5 from the thread text, verified automatically (verbatim quotes in the cited messages).",
        "Implicit links": "Which JIRA issue tracks a problem raised in a mailing-list thread when neither names the other (same author, similar title, within 7 days). Mined deterministically.",
        "Coding": "Real merged Kafka PRs: the agent edits the repository; the PR's own tests decide (fail before, pass after). 12 tasks, most of them part of multi-PR dependency chains.",
    }  # fmt: skip
    P.append(f"""<section><h2>1 Setup</h2>
<div class="fig"><table><tr><th>Arm</th><th style="text-align:left">Type</th><th style="text-align:left">How the agent reaches project history</th></tr>{arm_rows}</table>
<p class="cap">Every file arm (F*) is served by the Harnext MCP server with its two read tools: <code>context_research</code> (a read agent writes a cited report) and <code>search_context</code> (BM25 over the store, passages with file paths). The standing instruction: use <code>context_research</code> for full reports, <code>search_context</code> for lightweight retrievals. All stores ingest the same window; LLM-built stores use Claude Haiku 4.5 at 250 events per builder run.</p></div>
<div class="fig"><table><tr><th>Category</th><th>Tasks</th><th style="text-align:left">What it tests</th></tr>{''.join(f"<tr><td>{esc(c)}</td><td>{n_tasks.get(c, 0)}</td><td style='text-align:left;white-space:normal'>{esc(cats[c])}</td></tr>" for c in CATEGORIES)}</table></div>
{legend(arms)}</section>""")  # fmt: skip
    if runs:  # arms from further run ids: say where each one comes from
        runs = [provenance(config, results, main_arms)] + runs
        later = [r for r in runs[1:] if "openrouter.ai" in " ".join(r["allowlist"])]
        prow = "".join(f"<tr><td><code>{esc(r['run'])}</code></td><td style='text-align:left;white-space:normal'>{esc(', '.join(r['arms']))}</td><td>{esc(r['dates'])}</td><td><code>{esc(r['image'])}</code></td><td style='text-align:left;white-space:normal'>{esc(', '.join(r['allowlist']))}</td></tr>" for r in runs)  # fmt: skip
        P.append(f"""<section><h2>Runs combined in this report</h2><div class="fig"><table><tr><th>Run</th><th style="text-align:left">Arms</th><th>Trial dates (UTC)</th><th>Agent image</th><th style="text-align:left">Proxy allowlist</th></tr>{prow}</table>
<p class="cap">Arms from different runs share the benchmark, the agent and its model, but not the date, the image or the proxy.{" The engine arms (" + esc(", ".join(a for r in later for a in r["arms"] if ARM_TYPE.get(a) == ENGINE_TYPE)) + ") ran on a later date, on a different image, behind a proxy that additionally allows openrouter.ai." if later else ""} An arm shown with a run id in parentheses is a re-run of that arm in the later run (a drift control), not the original.</p></div></section>""")  # fmt: skip

    # 3. accuracy overview
    def acc(a, c):
        v = by[(a, c)]
        return (sum(r["ok"] for r in v) / len(v), len(v)) if v else (None, 0)

    P.append("<section><h2>2 Accuracy by category</h2><p>Share of tasks answered correctly (history: exact typed match; coding: every test of the golden PR passes). Colour intensity follows the share.</p>"
             + fig(heat_table(rows, arms, CATEGORIES, acc, lambda v, n: f"{v:.0%} <span style='color:var(--ink3)'>/{n}</span>"),
                   "<b>Table 1.</b> Accuracy per arm and category. Denominators differ slightly where a trial is missing.")  # fmt: skip
             + "</section>")
    for c in CATEGORIES:
        mine = [r for r in rows if r["category"] == c]
        if not mine:
            continue
        P.append(fig(dot_ci_chart(mine, arms, "correct" if c != "Coding" else "resolved"),
                     f"<b>Figure: {esc(c)}.</b> Dot = share correct, line = 95 % Wilson interval. Overlapping intervals are not a difference."))
    # history groups
    hist = [r for r in rows if r["category"] == "Factual history"]
    hgroups = [g for g in ["State lookups", "Change histories", "Cross-source / multi-hop", "Aggregation"] if any(r["group"] == g for r in hist)]

    def gacc(a, g):
        v = [r for r in hist if r["arm"] == a and r["group"] == g]
        return (sum(r["ok"] for r in v) / len(v), len(v)) if v else (None, 0)

    P.append(fig(heat_table(hist, arms, hgroups, gacc, lambda v, n: f"{v:.0%} <span style='color:var(--ink3)'>/{n}</span>"),
                 "<b>Table 2.</b> Factual history split into the four family groups of the design (never per single family: 1–7 questions each)."))

    # design decisions strict vs lenient
    dd = [r for r in rows if r["category"] == "Design decisions"]
    if dd:
        drows = "".join(
            f"<tr><td>{esc(ARM_LABEL.get(a, a))}</td><td class='num'>{sum(r['ok'] for r in dd if r['arm'] == a)}/{sum(r['arm'] == a for r in dd)}</td>"
            f"<td class='num'>{sum(r['ok'] or lenient_equal(r['expected'], r['answer']) for r in dd if r['arm'] == a)}/{sum(r['arm'] == a for r in dd)}</td>"
            f"<td class='num'>{sum((not r['ok']) and str(r['answer']).lower() == str(private[r['task']].get('superseded') or '').lower() for r in dd if r['arm'] == a)}</td></tr>"
            for a in arms)  # fmt: skip
        P.append(fig(f"<table><tr><th>Arm</th><th>Strict</th><th>Lenient</th><th>Answered the superseded value</th></tr>{drows}</table>",
                     "<b>Table 3.</b> Design decisions. <b>Strict</b> = exact typed match (casefolded). <b>Lenient</b> additionally ignores case, spacing and punctuation and allows one typo in identifiers of 10+ characters (the final message of one thread itself misspells its identifier). <b>Superseded</b> = the answer is exactly the earlier proposal the discussion later replaced."))  # fmt: skip

    # 4. cost
    P.append("<section><h2>3 Cost</h2><p>Model cost per trial at API list prices: the agent's own model plus, for file arms, Harnext's read agent inside <code>context_research</code>. Dashed line = S-tools.</p>")
    for c in CATEGORIES:
        mine = [r for r in rows if r["category"] == c]
        if not mine:
            continue
        vals = [(a, st.mean([r["cost"] for r in by[(a, c)]]) if by[(a, c)] else None, "") for a in arms]
        ref = next((v for a, v, _ in vals if a == "S-tools"), None)
        P.append(fig(bar_chart(vals, lambda v: money(v, ".3f"), "per trial", ref), f"<b>Figure: cost per trial, {esc(c.lower())}</b> (mean)."))
    def spread(a):  # engine view: median and p10-p90 of the per-trial cost, unmetered upper bound
        if not engine_view:
            return ""
        costs = sorted(r["cost"] for r in rows if r["arm"] == a)
        if not costs:
            return "<td>–</td><td>–</td><td>–</td><td>–</td>"
        q = lambda f: costs[min(len(costs) - 1, int(f * len(costs)))]  # noqa: E731
        return (f"<td class='num'>{money(st.median(costs), '.3f')}</td><td class='num'>{money(q(0.1), '.3f')}–{money(q(0.9), '.3f')}</td>"
                f"<td class='num'>{money(sum(r['unmetered'] for r in rows if r['arm'] == a), ',.2f')}</td>"
                f"<td class='num'>{money(sum(r['retry_cost'] for r in rows if r['arm'] == a), ',.2f')}</td>")  # fmt: skip

    split = "".join(
        f"<tr><td>{esc(ARM_LABEL.get(a, a))}</td><td class='num'>{money(sum(r['agent_cost'] for r in rows if r['arm'] == a), ',.2f')}</td>"
        f"<td class='num'>{money(sum(r['reader_cost'] for r in rows if r['arm'] == a), ',.2f')}</td><td class='num'>{money(sum(r['cost'] for r in rows if r['arm'] == a), ',.2f')}</td>"
        f"<td class='num'>{money(build_cost[a], ',.2f')}</td><td class='num'>{money(per_task_build[a], ',.2f')}</td>{spread(a)}</tr>" for a in arms)  # fmt: skip
    extra_head = "<th>Median $ / trial</th><th>p10–p90 $ / trial</th><th>Unmetered upper bound $</th><th>Provider-retry $</th>" if engine_view else ""
    engine_note = (" For the engine arms the read-agent cost is the engine's: Jev as reported by the provider (OpenRouter); the chat model as reported by OpenRouter or, for a local model, API-equivalent from token counts at list price ($0.018 / $0.09 per million input / output tokens). "
                   "The unmetered upper bound is what model requests that returned no usage (failed, retried or cancelled at the deadline) could have cost at most; it is not included in the other columns. "
                   "Provider-retry cost is what research attempts cost that the engine's MCP server discarded and redid because the provider degraded them (never shown to the agent); it is included in the trial totals. "
                   "The outer agent's cost varies up to 4× with its prompt-cache state, so compare medians and spreads over the full task set, not a few trials.") if engine_view else ""  # fmt: skip
    P.append(fig(f"<table><tr><th>Arm</th><th>Agent $</th><th>Read agent $</th><th>Trials total $</th><th>Store build $</th><th>Build $ per task served</th>{extra_head}</tr>{split}</table>",
                 "<b>Table 4.</b> Where the money goes. The store build is a one-off per store (shared by an arm and its research-only variant); spread over the tasks of this benchmark it often exceeds the per-trial cost — a store pays off only over many more tasks." + engine_note) + "</section>")  # fmt: skip

    # 5. cost vs accuracy
    P.append("<section><h2>4 Cost against accuracy</h2><p>Top-left is best: accurate and cheap. S-tools is the reference; an arm to its left at the same height is cheaper for the same accuracy.</p>")
    for c in CATEGORIES:
        pts = [(a, st.mean([r["cost"] for r in by[(a, c)]]) if by[(a, c)] else None,
                sum(r["ok"] for r in by[(a, c)]) / len(by[(a, c)]) if by[(a, c)] else None) for a in arms]  # fmt: skip
        if any(p[1] is not None for p in pts):
            P.append(fig(scatter(pts, "mean cost per trial (USD)", "share correct", lambda v: money(v, ".2f")), f"<b>Figure: cost vs accuracy, {esc(c.lower())}.</b>"))
    P.append("</section>")

    # 6. effort: turns, time, tokens
    P.append("<section><h2>5 Effort: turns, time and tokens</h2>")
    for c in CATEGORIES:
        if not any(by[(a, c)] for a in arms):
            continue
        t_rows = "".join(
            f"<tr><td>{esc(ARM_LABEL.get(a, a))}</td><td class='num'>{med([r['turns'] for r in by[(a, c)]]) or 0:.0f}</td>"
            f"<td class='num'>{(med([r['secs'] for r in by[(a, c)]]) or 0) / 60:.1f}</td>"
            f"<td class='num'>{(med([r['agent_in'] for r in by[(a, c)]]) or 0) / 1000:,.0f}k</td>"
            f"<td class='num'>{(med([r['agent_out'] for r in by[(a, c)]]) or 0) / 1000:,.1f}k</td>"
            f"<td class='num'>{(med([r['reader_in'] for r in by[(a, c)]]) or 0) / 1000:,.0f}k</td>"
            f"<td class='num'>{med([r['mcp_calls'] + r['cli_calls'] for r in by[(a, c)]]) or 0:.0f}</td></tr>"
            for a in arms if by[(a, c)])  # fmt: skip
        P.append(fig(f"<table><tr><th>Arm</th><th>Agent turns</th><th>Minutes</th><th>Agent input tokens</th><th>Agent output tokens</th><th>Read-agent input tokens</th><th>Tool calls to history</th></tr>{t_rows}</table>",
                     f"<b>Table: effort per trial, {esc(c.lower())}</b> (medians). Input tokens include cached tokens. Tool calls to history = MCP calls (file arms, R-hyb) or gh/jira/mail CLI calls (S-tools)."))  # fmt: skip
        vals = [(a, (med([r["secs"] for r in by[(a, c)]]) or 0) / 60 if by[(a, c)] else None, "") for a in arms]
        ref = next((v for a, v, _ in vals if a == "S-tools"), None)
        P.append(fig(bar_chart(vals, lambda v: f"{v:.1f} min", "median wall time", ref), f"<b>Figure: median wall time per trial, {esc(c.lower())}.</b>"))
    P.append("</section>")

    # 7. paired vs S-tools
    tasks_by = defaultdict(dict)
    for r in rows:
        tasks_by[(r["category"], r["task"])][r["arm"]] = r
    prow = []
    for a in arms:
        if a == "S-tools":
            continue
        cells = []
        for c in CATEGORIES:
            pairs = [(v[a], v["S-tools"]) for (cc, _), v in tasks_by.items() if cc == c and a in v and "S-tools" in v]
            if not pairs:
                cells.append("<td>–</td>")
                continue
            win = sum(x["ok"] and not y["ok"] for x, y in pairs)
            loss = sum(y["ok"] and not x["ok"] for x, y in pairs)
            dc = sum(x["cost"] for x, _ in pairs) / max(1e-9, sum(y["cost"] for _, y in pairs)) - 1
            dt = (med([x["secs"] for x, _ in pairs]) or 0) / max(1e-9, med([y["secs"] for _, y in pairs]) or 1)
            p = mcnemar(win, loss)
            cells.append(f"<td class='num'><span class='good'>+{win}</span> / <span class='bad'>−{loss}</span> · p={p:.2f}<br><span style='color:var(--ink3)'>cost {dc:+.0%} · time {dt:.1f}×</span></td>")
        prow.append(f"<tr><td>{esc(ARM_LABEL.get(a, a))}</td>{''.join(cells)}</tr>")
    P.append("<section><h2>6 Each arm against S-tools on the same tasks</h2>"
             + fig(f"<table><tr><th>Arm</th>{''.join(f'<th>{esc(c)}</th>' for c in CATEGORIES)}</tr>{''.join(prow)}</table>",
                   "<b>Table 5.</b> Paired comparison with the systems-of-record baseline. <span class='good'>+wins</span> = tasks the arm solved and S-tools did not; <span class='bad'>−losses</span> the reverse; p = exact McNemar test on the discordant pairs. Cost = total cost relative to S-tools on the same tasks; time = ratio of median wall times.")  # fmt: skip
             + "</section>")

    # 8. coding details
    cod = [r for r in rows if r["category"] == "Coding"]
    if cod:
        def split_ok(a, dep):
            v = [r for r in cod if r["arm"] == a and bool((deps.get(r["task"]) or {}).get("dependent")) == dep]
            return f"{sum(r['ok'] for r in v)}/{len(v)}" if v else "–"
        sim = lambda a, k: med([(r["similarity"] or {}).get(k) if k == "location_recall" else ((r["similarity"] or {}).get("files") or {}).get("f1") for r in cod if r["arm"] == a and r["similarity"]])  # noqa: E731
        crow = "".join(f"<tr><td>{esc(ARM_LABEL.get(a, a))}</td><td class='num'>{sum(r['ok'] for r in cod if r['arm'] == a)}/{sum(r['arm'] == a for r in cod)}</td><td class='num'>{split_ok(a, True)}</td><td class='num'>{split_ok(a, False)}</td>"
                       f"<td class='num'>{(sim(a, 'files') or 0):.2f}</td><td class='num'>{(sim(a, 'location_recall') or 0):.2f}</td></tr>" for a in arms)  # fmt: skip
        P.append("<section><h2>7 Coding</h2>" + fig(f"<table><tr><th>Arm</th><th>Resolved</th><th>Dependent tasks</th><th>Standalone tasks</th><th>Files F1 vs golden</th><th>Location recall</th></tr>{crow}</table>",
                 "<b>Table 6.</b> Coding. Dependent = the PR builds on earlier work in the window (JIRA links, references, shared files, multi-PR series; <code>dependencies.json</code>). Files F1 and location recall compare the agent's patch with the golden one (descriptive, never part of <i>resolved</i>).") + "</section>")  # fmt: skip

    # 9. ablation: research-only vs research+search
    abl = []
    for base in ("F0", "F1", "F3"):
        rr = [r for r in load_trials(results, private, [base, base + "-search"]) if r["kind"] == "historical" and r["category"] == "Factual history"]
        a_rows = [r for r in rr if r["arm"] == base]
        b_rows = [r for r in rr if r["arm"] == base + "-search"]
        if a_rows and b_rows:
            abl.append((base, a_rows, b_rows))
    if abl:
        arow = "".join(f"<tr><td>{b}</td><td class='num'>{sum(r['ok'] for r in a)}/{len(a)}</td><td class='num'>{sum(r['ok'] for r in s)}/{len(s)}</td>"
                       f"<td class='num'>${st.mean(r['cost'] for r in a):.3f}</td><td class='num'>${st.mean(r['cost'] for r in s):.3f}</td>"
                       f"<td class='num'>{(med([r['secs'] for r in a]) or 0):.0f} s</td><td class='num'>{(med([r['secs'] for r in s]) or 0):.0f} s</td></tr>" for b, a, s in abl)  # fmt: skip
        P.append("<section><h2>8 Ablation: research agent only vs research + search</h2>" + fig(f"<table><tr><th>Store</th><th>Research only</th><th>+ search</th><th>$ research only</th><th>$ + search</th><th>Time research only</th><th>Time + search</th></tr>{arow}</table>",
                 "<b>Table 7.</b> Same stores, same factual-history questions, two read interfaces: the production research tool alone (arms retired on 27 Sep 2026) vs the merged interface with <code>search_context</code>. The merged interface is what Harnext now ships.") + "</section>")  # fmt: skip

    # 10. stores
    srow = "".join(f"<tr><td>{esc(s)}</td><td class='num'>{v['windows'] or '–'}</td><td class='num'>${v['cost']:,.2f}</td><td class='num'>{v['hours']:.1f} h</td><td class='num'>{v['failed']}</td></tr>" for s, v in sorted(stores.items()))
    # decided value still present in each store at the question's checkpoint (identifier answers)
    import subprocess
    dd_tasks = [t for t in private.values() if t.get("family") == "design_decision" and t.get("answer_type") in ("identifier", "number", "version") and len(str(t["expected"]["value"])) >= 4]
    kept_rows = []
    for arm in arms:
        s = STORE_OF(arm)
        tree = EVAL / "out/kafka-500-v2/stores-w2026" / s / "tree"
        if not tree.exists() or arm.endswith("-search") is False:
            continue
        hits = sum(bool(subprocess.run(["git", "-C", str(tree), "grep", "-l", "-i", "-F", str(t["expected"]["value"]), f"ckpt-{t['checkpoint']}"],
                                       capture_output=True, text=True).stdout.strip()) for t in dd_tasks)  # fmt: skip
        v = by[(arm, "Design decisions")]
        kept_rows.append(f"<tr><td>{esc(s)}</td><td class='num'>{hits}/{len(dd_tasks)}</td><td class='num'>{sum(r['ok'] for r in v)}/{len(v)}</td></tr>")
    retention = (fig(f"<table><tr><th>Store</th><th>Decided value present in the store</th><th>Design decisions answered (arm)</th></tr>{''.join(kept_rows)}</table>",
                     "<b>Table 9.</b> Is the decided value (the exact identifier, number or version a discussion settled on) still anywhere in the store at the question's checkpoint? Entity-organised layouts (F1 templated, F4-entity, F5) keep issues and PRs but lose most mailing-list discussion content; the arms' accuracy follows the retention.") if kept_rows else "")  # fmt: skip
    P.append("<section><h2>9 Stores</h2>" + retention + fig(f"<table><tr><th>Store</th><th>Builder windows</th><th>Build cost</th><th>Builder time</th><th>Failed windows</th></tr>{srow}</table>",
             "<b>Table 8.</b> Store builds over the window (31,566 events). F0/F1 are rendered by code, R-hyb embeds locally (no model cost). LLM stores: Claude Haiku 4.5, 250 events per builder run, rebuilt from zero on 26 Sep 2026 after a guard bug was fixed.") + "</section>")  # fmt: skip

    # 11. limitations
    P.append("""<section><h2>10 Limitations and deviations</h2><ol>
<li><b>F5 builder cadence.</b> Production Harnext runs its builder per entity (20-event windows); the eval hands it 250-event slices with many entity batches per instruction. The F5 store holds an entity for only ~45 % of the window's issues and ~59 % of its PRs, so F5 results understate the product; a rebuild with a per-slice completion check is planned.</li>
<li><b>Coding workspaces start at the PR's merge parent.</b> Commits merged while the PR was open are in the repository, and Claude Code shows the last five in its system prompt; for 12 of 54 coding tasks these include sibling PRs of the same ticket. It affects all arms equally. A smoke test on one task rebased to the PR's start (all 12 arms) found no difference (12/12 solved both ways, same cost), which rules out a large effect on that task only.</li>
<li><b>Sample sizes.</b> 12 coding tasks, 1–7 questions per history family (reported only per group), 29 design decisions, 14 implicit links: intervals are wide; read differences through the paired table and its p-values.</li>
<li><b>Design-decision gold</b> was proposed by an LLM from the thread text and verified automatically (quotes present verbatim in the cited messages), not by a second human annotator; strict and lenient scores are both reported.</li>
<li><b>Factual-history questions are, by construction, answerable with the systems-of-record CLIs</b> (the answerability audit requires it), so S-tools is an upper bound there; the categories where memory can add value are design decisions and implicit links.</li>
<li>One agent (Claude Code, Haiku 4.5), one trial per task and arm; token refreshes interrupted a few trials, which were rerun (never recorded as failures).</li></ol></section>""")  # fmt: skip
    P.append("</main>")
    out = out or EVAL / "reports/e2" / config.run_id / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(P))
    return out, total_trials


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--extra-config", type=Path, action="append", default=[],
                        help="another run's config whose arms are shown beside this run's (repeatable)")  # fmt: skip
    args = parser.parse_args()
    out, n = build(args.config, args.out, args.extra_config)
    print(out, n, "trials")


if __name__ == "__main__":
    main()
