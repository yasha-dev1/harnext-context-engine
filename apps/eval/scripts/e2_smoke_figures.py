"""Export measured merged-E2 smoke charts and data; never run models or rewrite scores.

Run with the engine's workspace Python and --bundle <fresh-smoke directory>.
HTML is authored separately through apply_patch. Charts export as SVG and PNG.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from harnext_eval.e2.context import snapshot_files
from harnext_eval.stores.base import StoreHandle
from harnext_eval.types import SnapshotRef
from matplotlib.colors import ListedColormap

ARMS = ["S3", "S1", "S0"]
COLORS = ["#2563eb", "#d97706", "#0d9488"]
FAMILIES = ["extraction", "temporal", "update", "multisource", "abstention"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#334155",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "axes.edgecolor": "#cbd5e1",
        "axes.axisbelow": True,
        "grid.color": "#e2e8f0",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        "svg.hashsalt": "e2-smoke-report-v1",
        "savefig.bbox": "tight",
        "savefig.dpi": 180,
    }
)


def read(path):
    return json.loads(path.read_text())


def export(bundle: Path):
    figures = bundle / "figures"
    figures.mkdir(exist_ok=False)
    series, manifests, results, audits, probes, configs, stores = {}, {}, {}, {}, {}, {}, {}
    for arm in ARMS:
        root = bundle / "runs" / arm
        manifests[arm] = read(root / "manifest.json")
        results[arm] = read(root / "results.json")
        audits[arm] = read(bundle / "audits" / arm / "audit.json")
        configs[arm] = read(root / "resolved-config.json")
        assert manifests[arm]["status"] == "completed" and audits[arm]["audit_passed"]
        series[arm] = [
            read(path) for path in sorted((root / f"{arm}-repeat-1/answers").glob("*.json"))
        ]
        assert len(series[arm]) == 10
        for line in (root / "probes.gold.jsonl").read_text().splitlines():
            probe = json.loads(line)
            probes[probe["probe_id"]] = probe
        store = StoreHandle(arm, "smoke", root / f"{arm}-repeat-1/store")
        stores[arm] = []
        for build in results[arm]["builds"]:
            files = snapshot_files(store, SnapshotRef.model_validate(build["snapshot"]))
            stores[arm].append(
                {
                    "sha": build["snapshot"]["sha"],
                    "files": len(files),
                    "bytes": sum(len(text.encode()) for text in files.values()),
                    "event_count": len(
                        store.delivered_event_ids(SnapshotRef.model_validate(build["snapshot"]))
                    ),
                }
            )
    assert len({m["replay_sha256"] for m in manifests.values()}) == 1
    assert len({m["gold_sha256"] for m in manifests.values()}) == 1
    assert all(not build["reused"] for result in results.values() for build in result["builds"])
    previous = {}
    for arm in ARMS:
        root = bundle.parent / f"codex-luna-medium-20260912-run5-{arm}"
        if root.exists():
            assert read(root / "manifest.json")["gold_sha256"] == manifests[arm]["gold_sha256"]
            previous[arm] = [read(path) for path in sorted(root.glob("*/answers/*.json"))]
    charts = []

    def save(fig, slug, title, section, description, data):
        index = len(charts) + 1
        stem = f"{index:02}-{slug}"
        fig.tight_layout(pad=1.5)
        for suffix in ("svg", "png"):
            fig.savefig(figures / f"{stem}.{suffix}", format=suffix)
        plt.close(fig)
        charts.append(
            {
                "id": stem,
                "title": title,
                "section": section,
                "description": description,
                "data": data,
                "svg_path": str(figures / f"{stem}.svg"),
                "png_path": str(figures / f"{stem}.png"),
            }
        )

    def bar(ax, values, label=None, offset=0, width=0.65, color=None):
        bars = ax.bar(np.arange(3) + offset, values, width, label=label, color=color or COLORS)
        ax.set_xticks(np.arange(3), ARMS)
        ax.grid(axis="y", alpha=0.65)
        return bars

    def heat(matrix, ax, labels, *, text=None, vmax=1):
        ax.imshow(
            matrix,
            cmap=ListedColormap(["#fee2e2", "#fef3c7", "#ccfbf1"]),
            vmin=0,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_xticks(range(3), ARMS)
        ax.set_yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    str(text[i][j]) if text else f"{matrix[i][j]:.0%}",
                    ha="center",
                    va="center",
                    color="#0f172a",
                    fontweight="bold",
                )

    accuracy = [sum(r["grade"]["value_correct"] for r in series[a]) for a in ARMS]
    evidence = [sum(r["grade"]["evidence_correct"] for r in series[a]) for a in ARMS]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    b1 = bar(ax, accuracy, "Exact value", -0.18, 0.34, "#2563eb")
    b2 = bar(ax, evidence, "Strict evidence", 0.18, 0.34, "#0d9488")
    ax.bar_label(b1, padding=3)
    ax.bar_label(b2, padding=3)
    ax.set_ylim(0, 11.5)
    ax.set_ylabel("Correct probes / 10")
    ax.legend(loc="upper right")
    save(
        fig,
        "overall",
        "Answer correctness and strict evidence",
        "quality",
        "Each arm answers the same ten probes. Evidence checks required source IDs and actual retrieval exposure; it is separate from value correctness. No confidence intervals are estimated from this tiny smoke.",
        {"values": accuracy, "evidence": evidence, "denominator": 10},
    )

    for metric, title in [
        ("value_correct", "Exact answers by question family"),
        ("evidence_correct", "Strict evidence by question family"),
    ]:
        matrix = [
            [sum(r["grade"][metric] for r in series[a] if r["family"] == f) / 2 for a in ARMS]
            for f in FAMILIES
        ]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        heat(matrix, ax, [f.title() for f in FAMILIES])
        save(
            fig,
            metric,
            title,
            "quality",
            "Two probes per cell: possible scores are 0%, 50%, and 100%. Equal family sizes make overall micro accuracy equal to the five-family macro mean in this fixture.",
            matrix,
        )

    labels = [f"{p['probe_id'].replace('smoke-', 'Q')} · {p['family']}" for p in probes.values()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.5))
    matrices = {}
    for ax, metric, title in zip(
        axes,
        ["value_correct", "evidence_correct"],
        ["Exact answer", "Strict evidence"],
        strict=True,
    ):
        matrix = [[int(series[a][i]["grade"][metric]) for a in ARMS] for i in range(10)]
        matrices[metric] = matrix
        heat(
            matrix,
            ax,
            labels if ax is axes[0] else [f"Q{i + 1:02}" for i in range(10)],
            text=[["PASS" if v else "FAIL" for v in row] for row in matrix],
        )
        ax.set_title(title)
    save(
        fig,
        "probe-grid",
        "Every probe, every store",
        "quality",
        "All 30 answers are retained, including wrong values and evidence failures. Q01–Q10 map to the expandable question table below.",
        matrices,
    )

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.7))
    confusion = {}
    for ax, arm in zip(axes, ARMS, strict=True):
        matrix = np.zeros((2, 2), dtype=int)
        for row in series[arm]:
            gold_unknown = probes[row["probe_id"]]["gold"] is None
            predicted_unknown = row["answer"]["unknown"]
            matrix[int(gold_unknown), int(predicted_unknown)] += 1
        confusion[arm] = matrix.tolist()
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=8)
        ax.set_xticks([0, 1], ["Answered", "Unknown"])
        ax.set_yticks(
            [0, 1], ["Answerable (8)", "Absent (2)"] if arm == "S3" else ["Answerable", "Absent"]
        )
        ax.set_title(arm)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(matrix[i, j]),
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > 4 else "#0f172a",
                    fontweight="bold",
                )
    save(
        fig,
        "abstention",
        "Answer versus abstention decisions",
        "quality",
        "Rows are the gold answerability; columns are the model's decision. Answering an answerable question is not necessarily correct—the exact-value plots score that separately.",
        confusion,
    )

    if len(previous) == 3:
        old = [sum(r["grade"]["value_correct"] for r in previous[a]) for a in ARMS]
        fig, ax = plt.subplots(figsize=(8, 4.4))
        ax.bar_label(bar(ax, old, "Previous smoke", -0.18, 0.34, "#94a3b8"), padding=3)
        ax.bar_label(bar(ax, accuracy, "Fresh smoke", 0.18, 0.34, "#2563eb"), padding=3)
        ax.set_ylim(0, 11.5)
        ax.set_ylabel("Exact answers / 10")
        ax.legend()
        save(
            fig,
            "previous-smoke",
            "Fresh run versus the previous smoke",
            "quality",
            "Same fixture and nominal model/effort; this run rebuilds the stores and resamples readers. Changes can come from either builder or reader stochasticity. Two runs do not establish a trend or a significant improvement.",
            {"previous": old, "fresh": accuracy},
        )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    byte_data = {}
    for i, arm in enumerate(ARMS):
        byte_data[arm] = [r["bytes_read"] for r in series[arm]]
        ax.bar(
            np.arange(10) + (i - 1) * 0.25,
            np.array(byte_data[arm]) / 32768 * 100,
            0.24,
            color=COLORS[i],
            label=arm,
        )
    ax.axhline(100, color="#dc2626", linestyle="--", label="32,768-byte cap")
    ax.set_xticks(range(10), [f"Q{i + 1:02}" for i in range(10)])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Retrieval budget used (%)")
    ax.legend(ncol=4, fontsize=9)
    ax.grid(axis="y", alpha=0.6)
    save(
        fig,
        "read-budget",
        "Retrieval budget use for every probe",
        "retrieval",
        "Every returned UTF-8 byte counts, including paths, snippets, errors and repeated reads. This is a byte budget, not a model-token cap. Unused budget does not imply that enough relevant evidence was retrieved.",
        byte_data,
    )

    tools = {
        a: Counter(x["action"]["action"] for r in series[a] for x in r["retrieval"]) for a in ARMS
    }
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bottom = np.zeros(3)
    for kind, color in zip(
        ["list", "read", "search"], ["#94a3b8", "#2563eb", "#0d9488"], strict=True
    ):
        vals = [tools[a][kind] for a in ARMS]
        bars = ax.bar(ARMS, vals, bottom=bottom, label=kind, color=color)
        ax.bar_label(bars, label_type="center", color="white")
        bottom += vals
    ax.set_ylabel("Retrieval actions across ten probes")
    ax.legend(ncol=3)
    ax.set_ylim(0, max(bottom) * 1.18)
    ax.grid(axis="y", alpha=0.6)
    save(
        fig,
        "tool-mix",
        "Which tools did the reader use?",
        "retrieval",
        "Host list/read/search actions only. The final answer and provider completions are not counted as retrieval actions. More tool calls alone are not evidence of better retrieval.",
        {a: dict(v) for a, v in tools.items()},
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    calls = {}
    for i, arm in enumerate(ARMS):
        calls[arm] = [r["provider_calls"] for r in series[arm]]
        ax.plot(range(10), calls[arm], "o-", color=COLORS[i], label=arm, alpha=0.85)
    ax.axhline(6, color="#dc2626", linestyle="--", label="Six-call maximum")
    ax.set_xticks(range(10), [f"Q{i + 1:02}" for i in range(10)])
    ax.set_ylim(0.5, 6.8)
    ax.set_ylabel("Reader completions, including final answer")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.6)
    save(
        fig,
        "reader-calls",
        "Reader call depth",
        "retrieval",
        "The sixth completion is reserved for an answer or unknown. A response at the cap may still be incomplete. Calls are measured independently of the retrieved-byte budget.",
        calls,
    )

    fig, ax = plt.subplots(figsize=(8, 4.4))
    latency = {}
    for i, arm in enumerate(ARMS):
        vals = [r["latency_s"] for r in series[arm]]
        latency[arm] = vals
        ax.scatter(
            np.repeat(i, 10) + np.linspace(-0.12, 0.12, 10), vals, color=COLORS[i], alpha=0.8
        )
        ax.hlines(np.median(vals), i - 0.25, i + 0.25, color=COLORS[i], linewidth=3)
    ax.set_xticks(range(3), ARMS)
    ax.set_ylabel("End-to-end reader seconds / probe")
    ax.grid(axis="y", alpha=0.6)
    save(
        fig,
        "reader-latency",
        "Observed reader latency",
        "runtime",
        "Dots are the ten probes; horizontal lines are medians. The three arm processes run concurrently. Values include repeated Codex startup/inference and local tool handling, and are not a controlled serial latency benchmark.",
        latency,
    )

    usage = {a: results[a]["total_reader_usage"] for a in ARMS}
    fig, ax = plt.subplots(figsize=(8, 4.4))
    cached = np.array([usage[a].get("cached_input_tokens", 0) for a in ARMS])
    total = np.array([usage[a]["input_tokens"] for a in ARMS])
    assert np.all(total >= cached)
    ax.bar(ARMS, (total - cached) / 1000, label="Not reported cached", color="#2563eb")
    ax.bar(
        ARMS, cached / 1000, bottom=(total - cached) / 1000, label="Cached input", color="#93c5fd"
    )
    ax.set_ylabel("Reported reader input tokens (thousands)")
    ax.legend()
    ax.grid(axis="y", alpha=0.6)
    save(
        fig,
        "input-cache",
        "Reader input and cache accounting",
        "tokens",
        "Cached input is a subset of total input, never added twice. Counts include instructions, repeated history and retrieved material across every completion. Token totals are not USD prices.",
        {a: {"input": int(total[i]), "cached": int(cached[i])} for i, a in enumerate(ARMS)},
    )

    fig, ax = plt.subplots(figsize=(8, 4.4))
    output = [usage[a]["output_tokens"] for a in ARMS]
    ax.bar_label(bar(ax, output), padding=3)
    ax.set_ylim(0, max(output) * 1.18)
    ax.set_ylabel("Provider-reported output tokens")
    save(
        fig,
        "output-tokens",
        "Reader generation volume",
        "tokens",
        "Reported output tokens across ten probes. Separately reported reasoning tokens are not added to this total. The CLI's output-token target is advisory; the enforced controls are calls, time and retrieval bytes.",
        dict(zip(ARMS, output, strict=True)),
    )

    fig, ax = plt.subplots(figsize=(9, 4.4))
    inputs = {}
    for i, arm in enumerate(ARMS):
        inputs[arm] = [r["usage"].get("input_tokens", 0) for r in series[arm]]
        ax.plot(range(10), np.array(inputs[arm]) / 1000, "o-", color=COLORS[i], label=arm)
    ax.set_xticks(range(10), [f"Q{i + 1:02}" for i in range(10)])
    ax.set_ylabel("Input tokens / probe (thousands)")
    ax.legend()
    ax.grid(axis="y", alpha=0.6)
    save(
        fig,
        "probe-input",
        "Input footprint by question",
        "tokens",
        "Input is summed over all reader completions for that probe, including cache hits. High input can reflect repeated protocol overhead rather than more unique context read.",
        inputs,
    )

    fig, ax = plt.subplots(figsize=(8, 4.4))
    build_secs = [sum(b["latency_s"] for b in results[a]["builds"]) for a in ARMS]
    read_secs = [sum(latency[a]) for a in ARMS]
    ax.bar(ARMS, build_secs, color="#0d9488", label="Builder folds")
    ax.bar(ARMS, read_secs, bottom=build_secs, color="#2563eb", label="Reader probes")
    ax.set_ylabel("Sum of measured operation seconds")
    ax.legend()
    ax.grid(axis="y", alpha=0.6)
    save(
        fig,
        "phase-time",
        "Measured work by phase",
        "runtime",
        "Within-arm builder and reader operation durations are summed. Setup/report overhead is excluded. Do not sum all three bars to estimate elapsed time: the arms run concurrently.",
        {"builder_s": build_secs, "reader_s": read_secs},
    )

    fig, ax = plt.subplots(figsize=(8, 4.4))
    fold_times = {}
    for i, arm in enumerate(ARMS):
        vals = [b["latency_s"] for b in results[arm]["builds"]]
        fold_times[arm] = vals
        ax.bar(np.arange(2) + (i - 1) * 0.25, vals, 0.24, label=arm, color=COLORS[i])
    ax.set_xticks([0, 1], ["First window · 3 events", "Second window · 3 events"])
    ax.set_yscale("log")
    ax.set_ylabel("Fold seconds (log scale)")
    ax.legend()
    ax.grid(axis="y", alpha=0.6)
    save(
        fig,
        "build-latency",
        "Fresh build latency per window",
        "runtime",
        "All six snapshots were built fresh. S3 includes real model maintenance; S0 and S1 are deterministic local writers. The log axis keeps small local fold times visible without implying they have equal cost.",
        fold_times,
    )

    builder_path = bundle / "runs/S3/S3-repeat-1/store/usage.jsonl"
    builder_rows = [json.loads(line) for line in builder_path.read_text().splitlines()]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    binputs = [r["input_tokens"] for r in builder_rows]
    boutputs = [r["output_tokens"] for r in builder_rows]
    for ax, vals, title in zip(
        axes, [binputs, boutputs], ["Input tokens", "Output tokens"], strict=True
    ):
        bars = ax.bar(["Window 1", "Window 2"], vals, color=["#2563eb", "#60a5fa"])
        ax.bar_label(bars, padding=3)
        ax.set_ylim(0, max(vals) * 1.2)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.6)
    save(
        fig,
        "builder-tokens",
        "S3 builder token work",
        "tokens",
        "Each bar aggregates the Codex completions required for one successful fold. Deterministic S0/S1 writers make no builder model calls. These are new builds, not reused snapshots.",
        {"input": binputs, "output": boutputs},
    )

    for key, title, unit in [
        ("bytes", "Reader-visible store growth", "UTF-8 bytes"),
        ("files", "Reader-visible file count", "Visible files"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.4))
        for i, arm in enumerate(ARMS):
            vals = [s[key] for s in stores[arm]]
            ax.plot([1, 2], vals, "o-", label=arm, color=COLORS[i], linewidth=2)
        ax.set_xticks(
            [1, 2], ["After window 1\n3 delivered events", "After window 2\n6 delivered events"]
        )
        ax.set_ylabel(unit)
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.grid(axis="y", alpha=0.6)
        save(
            fig,
            f"store-{key}",
            title,
            "storage",
            "Measured from the exact reader-visible text export for each committed SHA. Git objects, evaluator ledgers, builder instructions and transient source mounts are excluded. Size alone is not semantic retention.",
            {a: [s[key] for s in stores[a]] for a in ARMS},
        )

    fig, ax = plt.subplots(figsize=(8, 4.4))
    for i, arm in enumerate(ARMS):
        ax.scatter(total[i] / 1000, accuracy[i] / 10 * 100, color=COLORS[i], s=140)
        ax.annotate(
            arm,
            (total[i] / 1000, accuracy[i] / 10 * 100),
            xytext=(8, 8),
            textcoords="offset points",
            fontweight="bold",
        )
    ax.set_xlabel("Reported reader input tokens (thousands)")
    ax.set_ylabel("Exact-answer accuracy (%)")
    ax.set_ylim(0, 110)
    ax.margins(x=0.18)
    ax.grid(alpha=0.6)
    save(
        fig,
        "quality-input",
        "Observed quality versus input footprint",
        "tokens",
        "One point per store on one ten-probe workload. This is not a cost-optimal frontier: USD prices, independent build repetitions and held-out measurements are absent.",
        {a: {"input": int(total[i]), "accuracy": accuracy[i] / 10} for i, a in enumerate(ARMS)},
    )

    gates = {
        "Completed answers": sum(
            r["status"] == "completed" for rows in series.values() for r in rows
        ),
        "Consistent schema": sum(
            r["grade"]["schema_valid"] for rows in series.values() for r in rows
        ),
        "Within byte cap": sum(r["bytes_read"] <= 32768 for rows in series.values() for r in rows),
        "Snapshot prefix": sum(
            check["passed"] for result in results.values() for check in result["checks"]
        ),
        "Tool replay": sum(
            r["tool_replay_passed"] for audit in audits.values() for r in audit["rows"]
        ),
    }
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bars = ax.barh(list(gates), list(gates.values()), color="#0d9488")
    ax.bar_label(bars, labels=[f"{v}/30" for v in gates.values()], padding=3)
    ax.set_xlim(0, 35)
    ax.set_xlabel("Passing probe checks")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.6)
    save(
        fig,
        "audit-gates",
        "Infrastructure verification",
        "audit",
        "These gates verify execution, access and accounting. They do not require correct answers. Tool replay recomputes every returned string from the selected SHA, including byte truncation.",
        gates,
    )

    flat = []
    for arm in ARMS:
        for row in series[arm]:
            flat.append(
                {
                    "arm": arm,
                    "probe": row["probe_id"],
                    "family": row["family"],
                    "value_correct": row["grade"]["value_correct"],
                    "evidence_correct": row["grade"]["evidence_correct"],
                    "bytes_read": row["bytes_read"],
                    "tool_calls": row["tool_calls"],
                    "provider_calls": row["provider_calls"],
                    "latency_s": row["latency_s"],
                    "input_tokens": row["usage"].get("input_tokens", 0),
                    "output_tokens": row["usage"].get("output_tokens", 0),
                    "unknown": row["answer"]["unknown"],
                    "gold": json.dumps(probes[row["probe_id"]]["gold"]),
                    "answer": json.dumps(row["answer"]["value"]),
                }
            )
    with (bundle / "probe-metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    payload = {
        "bundle": str(bundle.resolve()),
        "arms": ARMS,
        "manifests": manifests,
        "configs": configs,
        "results": results,
        "audits": audits,
        "probes": probes,
        "answers": series,
        "store_sizes": stores,
        "charts": charts,
        "previous_values": {
            a: sum(r["grade"]["value_correct"] for r in rows) for a, rows in previous.items()
        },
        "values": dict(zip(ARMS, accuracy, strict=True)),
        "evidence": dict(zip(ARMS, evidence, strict=True)),
    }
    (bundle / "chart-data.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {
                "charts": len(charts),
                "data": str(bundle / "chart-data.json"),
                "figures": str(figures),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    export(parser.parse_args().bundle.resolve())
