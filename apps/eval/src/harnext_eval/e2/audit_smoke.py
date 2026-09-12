"""Replay and audit saved merged-E2 smoke evidence without any model calls.

Preserves the original scores. Separately reports the narrow source#ID citation
alias used by the engine's operating manual, with no fuzzy/value normalization.
This is an engineering diagnostic, never post-hoc thesis confirmation scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from harnext_eval.e2.context import (
    FAMILIES,
    Action,
    Profile,
    SnapshotTools,
    grade,
    sha,
    snapshot_files,
    write_json,
)
from harnext_eval.stores.base import StoreHandle
from harnext_eval.types import Probe


def citation_diagnostic(probe: Probe, result: dict[str, Any], ids: set[str]) -> dict[str, Any]:
    """Accept only a raw delivered ID or its exact, actually-read source#ID wrapper."""
    normalized = json.loads(json.dumps(result))
    answer = normalized.get("answer")
    aliases = {}
    exposed = "\n".join(row["result"] for row in result.get("retrieval", []))
    if answer:
        for citation in answer["evidence_ids"]:
            raw = citation.rsplit("#", 1)[-1]
            if citation not in ids and "#" in citation and raw in ids and citation in exposed:
                aliases[citation] = raw
        answer["evidence_ids"] = [aliases.get(value, value) for value in answer["evidence_ids"]]
    return {"grade": grade(probe, normalized, ids), "exact_citation_aliases": aliases}


def audit(run_dir: Path, output: Path) -> dict[str, Any]:
    if not __debug__:
        raise RuntimeError("Do not run the evidence auditor with Python assertion checks disabled")
    if output.exists() and any(output.iterdir()):
        raise ValueError("audit output must be new or empty")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["status"] != "completed" or manifest["thesis_evidence"] is not False:
        raise ValueError("audit requires a completed smoke run")
    profile = Profile.model_validate_json((run_dir / "resolved-config.json").read_text())
    probes_path = run_dir / "probes.gold.jsonl"
    assert sha(probes_path.read_bytes()) == manifest["gold_sha256"]
    assert sha((run_dir / "replay.jsonl").read_bytes()) == manifest["replay_sha256"]
    assert sha(profile.model_dump_json()) == manifest["config_sha256"]
    probes = [Probe.model_validate_json(line) for line in probes_path.read_text().splitlines()]
    rows = []
    builder_usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    for repeat in range(1, profile.repeats + 1):
        for layout in profile.layouts:
            arm = run_dir / f"{layout}-repeat-{repeat}"
            assert (arm / "store/git/smoke/.git").is_dir()
            store = StoreHandle(layout, "smoke", arm / "store")
            for probe in probes:
                row = json.loads((arm / "answers" / f"{probe.probe_id}.json").read_text())
                ref = store.snapshot(probe.T)
                files = snapshot_files(store, ref)
                assert row["snapshot_sha"] == ref.sha
                assert row["snapshot_content_sha256"] == sha(json.dumps(files, sort_keys=True))
                tools = SnapshotTools(files, budget=profile.read_budget_bytes,
                                      response_cap=profile.response_bytes)
                for retrieval in row["retrieval"]:
                    assert tools.invoke(Action.model_validate(retrieval["action"])) == retrieval["result"]
                    assert tools.log[-1] == retrieval
                assert tools.used == row["bytes_read"] <= profile.read_budget_bytes
                ids = set(store.delivered_event_ids(ref))
                original = grade(probe, row, ids)
                assert original == row["grade"]
                rows.append({"layout": layout, "repeat": repeat, "probe_id": probe.probe_id,
                    "family": probe.family, "original_grade": original,
                    **citation_diagnostic(probe, row, ids), "tool_replay_passed": True})
            transcripts = arm / "reader-transcripts.json"
            if transcripts.exists():
                for call in json.loads(transcripts.read_text()):
                    assert call["model"] == profile.model
                    assert call["usage"]["reasoning_effort"] == profile.reasoning_effort
                    assert call["stop_reason"] == "completed" and call["error"] is None
                    assert not any(event.get("type") == "item.completed" and
                        event.get("item", {}).get("type") not in {"agent_message", "reasoning"}
                        for event in call["usage"]["events"])
            usage = arm / "store/usage.jsonl"
            if usage.exists():
                for line in usage.read_text().splitlines():
                    build = json.loads(line)
                    assert build["status"] == "success" and build["files_touched"]
                    assert build["model"] == profile.model
                    assert build["usage"]["reasoning_effort"] == profile.reasoning_effort
                    assert not any("_event/" in path for path in store.list_files(store.snapshot(probes[-2].T)))
                    for key in builder_usage:
                        builder_usage[key] += build["usage"].get(key, 0)
                    for call in build["usage"].get("calls", []):
                        assert call["model"] == profile.model
                        assert call["usage"]["reasoning_effort"] == profile.reasoning_effort
                        assert call["stop_reason"] == "completed" and call["error"] is None
    table = []
    for layout in profile.layouts:
        group = [row for row in rows if row["layout"] == layout]
        table.append({"layout": layout, "n": len(group),
            "values_correct": sum(row["original_grade"]["value_correct"] for row in group),
            "strict_citations_correct": sum(row["original_grade"]["evidence_correct"] for row in group),
            "exact_wrapper_citations_correct": sum(row["grade"]["evidence_correct"] for row in group)})
    results = json.loads((run_dir / "results.json").read_text())
    report = {"source_run": str(run_dir.resolve()), "source_manifest_sha256": sha((run_dir / "manifest.json").read_bytes()),
        "audit_version": "tool-replay-v1/exact-citation-wrapper-v1", "thesis_evidence": False,
        "audit_passed": True, "probe_count": len(rows), "families": list(FAMILIES),
        "model": profile.model, "reasoning_effort": profile.reasoning_effort,
        "builder_harness": profile.builder_harness, "reader_provider": profile.reader_provider,
        "builder_tool_policy": profile.builder_tool_policy, "table": table, "rows": rows,
        "builder_usage": builder_usage, "reader_usage": results["total_reader_usage"],
        "total_cost_usd": None,
        "interpretation": "Original scores preserved. Wrapper normalization is a separate engineering diagnostic, not a replacement confirmation score.",
        "metadata_correction": ("File-tool mode uses the host path allowlist; native-isolation label was stale."
            if profile.builder_tool_policy == "files" and str(manifest.get("builder_isolation", "")).startswith("native")
            else None)}
    write_json(output / "audit.json", report)
    text = ["# Merged E2: Codex smoke verification", "", "Snapshot/tool replay audit: **PASS**.", "",
        f"Model: `{profile.model}`, effort: `{profile.reasoning_effort}`, builder tools: `{profile.builder_tool_policy}`.",
        "", "| Store | Values correct | Strict event IDs | Exact source#ID wrapper | Probes |",
        "|---|---:|---:|---:|---:|"]
    text += [f"| {row['layout']} | {row['values_correct']} | {row['strict_citations_correct']} | "
             f"{row['exact_wrapper_citations_correct']} | {row['n']} |" for row in table]
    text += ["", "The original grades and model outputs remain untouched. The last citation column",
        "accepts a source#event-id wrapper only when that exact wrapper was retrieved and its",
        "event ID belongs to the snapshot. It performs no fuzzy answer matching. Freeze this",
        "citation contract before the real-data pilot; do not treat it as post-hoc confirmation scoring.",
        "", "Every recorded tool response was reproduced from its exact snapshot SHA, including",
        "truncation and byte accounting. Model/effort and completed provider turns were checked.",
        "", f"Builder input/output tokens: {builder_usage['input_tokens']:,} / {builder_usage['output_tokens']:,}.",
        f"Reader input/output tokens: {report['reader_usage'].get('input_tokens', 0):,} / "
        f"{report['reader_usage'].get('output_tokens', 0):,}.",
        "Cache hits are included within input totals, not added twice. USD cost is unknown.",
        "", "This is a synthetic infrastructure smoke, not a configuration-selection result.",
        "Native shell mode remains unverified on this host because bwrap fails at setup.",
        "Next: freeze the citation schema and exact tokenizer, then run a small real causal-data",
        "pilot before the file-template, semantic-retrieval, graph, and harness matrices."]
    (output / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.out)


if __name__ == "__main__":
    main()
