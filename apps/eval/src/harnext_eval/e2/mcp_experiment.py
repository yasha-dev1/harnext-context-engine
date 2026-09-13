"""External-harness handoff and deterministic grading of actual MCP receipts.

All inputs are selected by the same strategy YAML. This module never calls an
answering model. It stages public inputs, emits client connection config, and
grades externally produced answers/patches plus server-side retrieval evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from harnext_builder.strategies.config import canonical, digest, load_config
from harnext_builder.strategies.engine import load_artifact

from harnext_eval.e2.coding_sandbox import resolve_image, run_tests, safe_path
from harnext_eval.e2.retrieval_metrics import EvidenceSpan, overlap


def audit_receipts(artifact: dict, log_path: Path, gold: dict | None = None):
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    previous = ""
    for row in rows:
        body = {key: value for key, value in row.items() if key != "hash"}
        if row["previous"] != previous or digest(body) != row["hash"]:
            raise ValueError("receipt hash chain mismatch")
        previous = row["hash"]
    if not rows or rows[0]["kind"] != "start" or rows[0]["snapshot"] != artifact["sha256"]:
        raise ValueError("receipt belongs to another snapshot")
    trial = rows[0]["trial"]
    if rows[0]["identity"] != digest([trial, artifact["sha256"]]):
        raise ValueError("receipt identity mismatch")
    docs = artifact["payload"]["documents"]
    gold = gold or {}
    if gold and gold.get("snapshot_sha256") != artifact["sha256"]:
        raise ValueError("evidence annotations must be bound to this exact snapshot hash")
    spans = [EvidenceSpan.model_validate(row) for row in gold.get("spans", [])]
    groups = [set(group) for group in gold.get("required_groups", [])]
    known = {span.unit_id for span in spans}
    for span in spans:
        if (
            span.start >= span.end
            or docs.get(span.path, "").encode()[span.start : span.end] != span.quote.encode()
        ):
            raise ValueError("stale evidence annotation")
    if any(not group or not group <= known for group in groups):
        raise ValueError("gold groups contain unknown evidence units")
    relevant = set().union(*groups) if groups else set()
    coverage, exposed, curve, mix = {}, set(), [], {}
    total = relevant_bytes = repeat_bytes = errors = 0
    first = None
    for index, row in enumerate(rows[1:], 1):
        if row["kind"] != "call" or row["step"] != index:
            raise ValueError("receipt sequence mismatch")
        size = len(row["output"].encode())
        total += size
        if (
            size != row["bytes"]
            or total != row["cumulative_bytes"]
            or total > trial["budget_bytes"]
            or size > trial["response_bytes"]
        ):
            raise ValueError("receipt byte accounting mismatch")
        if row["tool"] not in trial["tools"] and row["error"] is None:
            raise ValueError("disabled tool returned successful material")
        if (index > trial["max_calls"] or row["elapsed_s"] > trial["timeout_s"]) and row[
            "error"
        ] is None:
            raise ValueError("expired trial returned successful material")
        mix[row["tool"]] = mix.get(row["tool"], 0) + 1
        errors += row["error"] is not None
        output = json.loads(row["output"]) if row["output"] else {"items": []}
        for item in output["items"]:
            if "text" not in item:
                continue
            path, start, end = item["path"], item["start"], item["end"]
            if (
                start < 0
                or end < start
                or path not in docs
                or docs[path].encode()[start:end] != item["text"].encode()
            ):
                raise ValueError("MCP returned content differs from immutable snapshot bytes")
            repeat_bytes += overlap((start, end), coverage.get(path, []))
            support = [
                (span.start, span.end)
                for span in spans
                if span.path == path and span.unit_id in relevant
            ]
            relevant_bytes += overlap((start, end), support)
            coverage.setdefault(path, []).append((start, end))
        for span in spans:
            if (
                overlap((span.start, span.end), coverage.get(span.path, []))
                == span.end - span.start
            ):
                exposed.add(span.unit_id)
        if first is None and exposed & relevant:
            first = index
        curve.append(
            {
                "call": index,
                "bytes": total,
                "recall": sum(bool(group & exposed) for group in groups) / len(groups)
                if groups
                else None,
            }
        )
    exhaustive = gold.get("annotations_exhaustive", False)
    return {
        "snapshot_sha256": artifact["sha256"],
        "receipt_hash": previous,
        "task_id": trial["task_id"],
        "declared_harness": trial["harness"],
        "declared_model": trial["model"],
        "declared_effort": trial["effort"],
        "audit_passed": True,
        "context_calls": len(rows) - 1,
        "tool_mix": mix,
        "error_calls": errors,
        "bytes_returned": total,
        "required_group_recall": curve[-1]["recall"] if curve else (0.0 if groups else None),
        "unit_precision": len(exposed & relevant) / len(exposed)
        if exhaustive and exposed
        else None,
        "relevant_byte_precision": relevant_bytes / total if exhaustive and total else None,
        "repeated_byte_fraction": repeat_bytes / total if total else None,
        "first_relevant_call": first,
        "exposed_units": sorted(exposed),
        "recall_curve": curve,
        "annotation_coverage": "exhaustive" if exhaustive else "partial_or_absent",
    }


def stage(config):
    spec = config.evaluation
    if spec.task_file is None or spec.workspace_dir is None:
        raise ValueError("stage requires evaluation.task_file and workspace_dir")
    if spec.workspace_dir.exists() and any(spec.workspace_dir.iterdir()):
        raise ValueError("task workspace must be new or empty")
    task = json.loads(spec.task_file.read_text())
    if task.get("task_id") != config.trial.task_id:
        raise ValueError("public task ID differs from the trial")
    spec.workspace_dir.mkdir(parents=True, exist_ok=True)
    (spec.workspace_dir / "TASK.md").write_text(task["prompt"])
    for name, content in task.get("base_files", {}).items():
        target = spec.workspace_dir / safe_path(name)
        if target.name in {"TASK.md", "test_visible.py"}:
            raise ValueError("implementation path collides with a protected task file")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    if "visible_tests" in task:
        (spec.workspace_dir / "test_visible.py").write_text(task["visible_tests"])
    return {
        "workspace": str(spec.workspace_dir),
        "task_id": task["task_id"],
        "public_task_sha256": digest(task),
        "gold_exposed": False,
    }


def grade(config):
    artifact = load_artifact(config.artifact_dir / "snapshot.json")
    spec = config.evaluation
    gold = json.loads(spec.gold_file.read_text()) if spec.gold_file else {}
    if (
        spec.task_file is not None
        and json.loads(spec.task_file.read_text()).get("task_id") != config.trial.task_id
    ):
        raise ValueError("public task ID differs from configured trial")
    result = audit_receipts(
        artifact,
        config.artifact_dir / "trials" / f"{config.trial.run_id}.jsonl",
        gold.get("retrieval"),
    )
    if spec.grading == "qa_exact":
        answer = json.loads(spec.answer_file.read_text())
        expected, actual = gold["value"], answer.get("value")
        consistent = type(answer.get("unknown")) is bool and answer["unknown"] == (actual is None)
        if isinstance(expected, list):
            correct = (
                isinstance(actual, list)
                and all(isinstance(item, str) for item in actual)
                and len(actual) == len(set(actual))
                and set(actual) == set(expected)
            )
        else:
            correct = type(actual) is type(expected) and actual == expected
        result["value_correct"] = bool(correct and consistent)
        result["answer_sha256"] = digest(answer)
    elif spec.grading == "coding_tests":
        if spec.task_file is None:
            raise ValueError("coding_tests requires a public task file")
        task = json.loads(spec.task_file.read_text())
        candidate = json.loads(spec.candidate_files.read_text())
        if set(candidate) != set(task["base_files"]) or not all(
            isinstance(v, str) for v in candidate.values()
        ):
            raise ValueError("candidate must contain exactly the allowed implementation files")
        image = resolve_image(spec.test_image)
        base = run_tests(
            task["base_files"], task["visible_tests"], gold["hidden_tests"], image=image
        )
        reference = run_tests(
            gold["reference"], task["visible_tests"], gold["hidden_tests"], image=image
        )
        if (
            base["status"] != "completed"
            or not reference["passed"]
            or set(base["tests"]) != set(reference["tests"])
        ):
            raise ValueError("base/reference task admission failed")
        f2p = [name for name, value in base["tests"].items() if value == "failed"]
        p2p = [name for name, value in base["tests"].items() if value == "passed"]
        if (
            not f2p
            or not p2p
            or any(value not in {"passed", "failed"} for value in base["tests"].values())
        ):
            raise ValueError(
                "task requires reproduced assertion failures and preserved regressions"
            )
        final = run_tests(candidate, task["visible_tests"], gold["hidden_tests"], image=image)
        result.update(
            final_tests=final,
            f2p={name: final["tests"].get(name) for name in f2p},
            p2p={name: final["tests"].get(name) for name in p2p},
            resolved=final["passed"] and set(final["tests"]) == set(reference["tests"]),
            candidate_sha256=digest(candidate),
            reference_sha256=digest(gold["reference"]),
        )
    result["config_sha256"] = digest(config.model_dump(mode="json"))
    result["agent_trace_sha256"] = (
        digest(spec.agent_trace.read_bytes()) if spec.agent_trace else None
    )
    destination = config.artifact_dir / "trials" / f"{config.trial.run_id}.report.json"
    if destination.exists():
        raise ValueError("report already exists; do not silently replace frozen grades")
    destination.write_text(canonical(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["client-config", "stage", "grade"])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "client-config":
        if config.trial.transport == "stdio":
            result = {
                "mcpServers": {
                    "harnext": {
                        "command": sys.executable,
                        "args": [
                            "-m",
                            "harnext_mcp.experiment",
                            "--config",
                            str(args.config.resolve()),
                        ],
                    }
                }
            }
        else:
            result = {
                "mcpServers": {
                    "harnext": {
                        "url": f"http://{config.trial.host}:{config.trial.port}/mcp",
                        "authorization_env": config.trial.token_env,
                    }
                },
                "note": "Map authorization_env to your client's bearer-token setting; it is a handoff field, not a universal MCP config key.",
            }
    elif args.command == "stage":
        result = stage(config)
    else:
        result = grade(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
