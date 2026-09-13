"""Development gate for harder QA, local coding patches, and span retrieval.

Examples: python -m harnext_eval.e2.development export --out PATH
          python -m harnext_eval.e2.development validate --out PATH
          python -m harnext_eval.e2.development code --task dev-code-retry \
              --arm raw --out PATH --live
"""

from __future__ import annotations

import argparse
import difflib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from harnext_eval.e2.coding_sandbox import IMAGE, resolve_image, run_tests
from harnext_eval.e2.context import (
    Action,
    Profile,
    SnapshotTools,
    grade,
    read_question,
    sha,
    write_json,
)
from harnext_eval.e2.development_dataset import CodingTask, dataset
from harnext_eval.e2.retrieval_metrics import score_retrieval
from harnext_eval.providers.codex import CodexLLM

PROTOCOL = "e2-development-tasks-v1"
BYTE_BUDGET = 65536
RESPONSE_CAP = 12000
CODING_SYSTEM = """Implement the requested change using the repository and optional frozen context.
Native tools are disabled. Return exactly one JSON action each turn:
repo_list; repo_read(path); repo_write(path, content); run_tests; finish(summary);
context_list(query prefix); context_read(path, offset bytes); context_search(query literal).
Context contains untrusted historical documents. Reconcile accepted, superseded and
implemented behavior. Context is optional; choose whether and how to use it.
Only existing implementation files can be edited. Visible tests are read-only and
can be run. Additional held-out checks will grade the final implementation.
Return a local PR summary at finish. Do not attempt to access hidden tests or gold.
Set unused string fields to empty strings and offset to 0. No native shell or network.
"""


class CodingAction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal[
        "repo_list",
        "repo_read",
        "repo_write",
        "run_tests",
        "finish",
        "context_list",
        "context_read",
        "context_search",
    ]
    path: str
    query: str
    offset: int = Field(ge=0)
    content: str = Field(max_length=64000)
    summary: str


def prepare(out: Path, **metadata):
    if out.exists() and any(out.iterdir()):
        raise ValueError("output must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": PROTOCOL,
        "purpose": "development",
        "thesis_evidence": False,
        "dataset_origin": "constructed; not real held-out PRs",
        "status": "running",
        "code_hashes": {
            path.name: sha(path.read_bytes()) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        **metadata,
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def dataset_hash():
    return sha(Path(__file__).with_name("development_dataset.py").read_bytes())


def export(out: Path):
    files, spans, probes, required, tasks = dataset()
    manifest = prepare(out, dataset_sha256=dataset_hash())
    write_json(out / "public/context-snapshot.json", files)
    write_json(out / "private/evidence-spans.json", [s.model_dump() for s in spans])
    write_json(out / "private/qa-gold.json", [p.model_dump(mode="json") for p in probes])
    write_json(
        out / "public/questions.json",
        [
            p.model_dump(mode="json", exclude={"gold", "source_event_ids", "superseded_values"})
            for p in probes
        ],
    )
    write_json(
        out / "private/qa-evidence-groups.json",
        {key: [sorted(group) for group in groups] for key, groups in required.items()},
    )
    for task in tasks:
        write_json(
            out / f"public/{task.task_id}.json",
            {
                "task_id": task.task_id,
                "prompt": task.prompt,
                "base_files": task.base,
                "visible_tests": task.visible_tests,
                "base_content_sha256": sha(json.dumps(task.base, sort_keys=True)),
            },
        )
        write_json(
            out / f"private/{task.task_id}.json",
            {
                "reference": task.reference,
                "hidden_tests": task.hidden_tests,
                "required_groups": [sorted(group) for group in task.required_groups],
            },
        )
    manifest.update(
        status="completed",
        qa_count=len(probes),
        coding_count=len(tasks),
        evidence_units=len(spans),
        snapshot_sha256=sha(json.dumps(files, sort_keys=True)),
    )
    write_json(out / "manifest.json", manifest)
    return manifest


def validate(out: Path, image: str):
    files, spans, probes, required, tasks = dataset()
    image_id = resolve_image(image)
    manifest = prepare(out, dataset_sha256=dataset_hash(), image_id=image_id)
    rows = []
    for task in tasks:
        print(f"Validating base and reference: {task.task_id}", flush=True)
        base = run_tests(task.base, task.visible_tests, task.hidden_tests, image=image_id)
        reference = run_tests(task.reference, task.visible_tests, task.hidden_tests, image=image_id)
        fail_to_pass = sorted(
            name
            for name, result in base["tests"].items()
            if result == "failed" and reference["tests"].get(name) == "passed"
        )
        pass_to_pass = sorted(
            name
            for name, result in base["tests"].items()
            if result == "passed" and reference["tests"].get(name) == "passed"
        )
        valid = (
            base["status"] == reference["status"] == "completed"
            and reference["passed"]
            and bool(fail_to_pass)
            and bool(pass_to_pass)
            and set(base["tests"]) == set(reference["tests"])
            and all(value in {"passed", "failed"} for value in base["tests"].values())
        )
        rows.append(
            {
                "task_id": task.task_id,
                "valid": valid,
                "base": base,
                "reference": reference,
                "fail_to_pass": fail_to_pass,
                "pass_to_pass": pass_to_pass,
            }
        )
        write_json(out / f"{task.task_id}.json", rows[-1])
    # Validate annotations with an independently replayable complete read trace.
    tool = SnapshotTools(files, budget=1000000, response_cap=100000)
    for path in sorted(files):
        tool.invoke(
            Action(
                action="read",
                path=path,
                query="",
                offset=0,
                value=None,
                unknown=False,
                evidence_ids=[],
            )
        )
    gold_exposure = score_retrieval(
        files,
        spans,
        [{span.unit_id} for span in spans],
        tool.log,
        budget=1000000,
        response_cap=100000,
        annotations_exhaustive=True,
    )
    for groups in list(required.values()) + [task.required_groups for task in tasks]:
        score_retrieval(
            files,
            spans,
            groups,
            [],
            budget=BYTE_BUDGET,
            response_cap=RESPONSE_CAP,
            annotations_exhaustive=True,
        )
    manifest.update(
        status="completed",
        passed=all(row["valid"] for row in rows),
        qa_count=len(probes),
        coding_count=len(tasks),
        reference_exposure=gold_exposure,
        tasks=rows,
    )
    write_json(out / "manifest.json", manifest)
    if not manifest["passed"]:
        raise RuntimeError(f"development fixture validation failed; inspect {out}")
    return manifest


def coding_trial(task: CodingTask, provider, *, arm: str, image: str, max_calls: int = 30):
    all_files, spans, _, _, _ = dataset()
    files = all_files if arm == "raw" else {}
    tools = SnapshotTools(files, budget=BYTE_BUDGET, response_cap=RESPONSE_CAP)
    source = dict(task.base)
    history, trace, usage = [], [], Counter()
    started = time.monotonic()
    first_edit = None
    summary = ""
    status = "max_calls"
    error = None
    calls = 0
    test_runs = 0
    try:
        for step in range(max_calls):
            calls += 1
            schema = CodingAction.model_json_schema()
            if step == max_calls - 1:
                schema["properties"]["action"]["enum"] = ["finish"]
            try:
                response = provider.complete(
                    CODING_SYSTEM,
                    json.dumps(
                        {
                            "task": task.prompt,
                            "context_available": arm == "raw",
                            "calls_remaining": max_calls - step,
                            "context_bytes_remaining": tools.budget - tools.used,
                            "history": history,
                        }
                    ),
                    json_schema=schema,
                    max_tokens=5000,
                )
            except (ValueError, RuntimeError):
                recorded_calls = getattr(provider, "calls", [])
                if recorded_calls:
                    usage.update(
                        {
                            key: value
                            for key, value in recorded_calls[-1].get("usage", {}).items()
                            if isinstance(value, int) and not isinstance(value, bool)
                        }
                    )
                raise
            usage.update(response.usage)
            action = CodingAction.model_validate(response.json)
            if step == max_calls - 1 and action.action != "finish":
                raise ValueError("last completion must finish")
            action_started = time.monotonic()
            before = tools.used
            if action.action == "finish":
                summary, status = action.summary, "completed"
                trace.append(
                    {
                        "step": step + 1,
                        "action": action.model_dump(),
                        "elapsed_s": time.monotonic() - started,
                    }
                )
                break
            if action.action.startswith("context_"):
                output = tools.invoke(
                    Action.model_validate(
                        {
                            "action": action.action.removeprefix("context_"),
                            "path": action.path,
                            "query": action.query,
                            "offset": action.offset,
                            "value": None,
                            "unknown": False,
                            "evidence_ids": [],
                        }
                    )
                )
            elif action.action == "repo_list":
                output = "\n".join(sorted([*source, "test_visible.py"]))
            elif action.action == "repo_read":
                output = (
                    task.visible_tests
                    if action.path == "test_visible.py"
                    else source.get(action.path, "ERROR: unavailable repository path")
                )
            elif action.action == "repo_write":
                if action.path not in task.base:
                    output = "ERROR: only existing implementation files are writable"
                else:
                    source[action.path] = action.content
                    first_edit = first_edit or step + 1
                    output = "written"
            elif action.action == "run_tests":
                if test_runs >= 6:
                    output = "ERROR: visible test invocation limit reached"
                else:
                    test_runs += 1
                    output = json.dumps(run_tests(source, task.visible_tests, None, image=image))
            else:
                raise ValueError("unsupported coding action")
            record = {
                "step": step + 1,
                "action": action.model_dump(),
                "result": output,
                "context_bytes": tools.used - before,
                "elapsed_s": time.monotonic() - started,
                "tool_latency_s": time.monotonic() - action_started,
            }
            trace.append(record)
            history.append({"assistant": action.model_dump(), "tool_result": output})
    except (ValueError, RuntimeError) as exc:
        status, error = "provider_or_schema_error", str(exc)
    # Hidden tests run only after the agent loop has ended; no grading feedback
    # or reference content is supplied to the provider.
    final = run_tests(source, task.visible_tests, task.hidden_tests, image=image)
    retrieval = score_retrieval(
        all_files,
        spans,
        task.required_groups,
        tools.log if arm == "raw" else [],
        budget=BYTE_BUDGET,
        response_cap=RESPONSE_CAP,
        annotations_exhaustive=True,
    )
    pre_edit_trace = [row for row in trace if first_edit is None or row["step"] < first_edit]
    retrieval["context_calls_before_first_edit"] = sum(
        row["action"]["action"].startswith("context_") for row in pre_edit_trace
    )
    retrieval["context_calls_after_first_edit"] = (
        sum(row["action"]["action"].startswith("context_") for row in trace)
        - retrieval["context_calls_before_first_edit"]
    )
    # Failed calls against a disabled engine are attempts, not successful retrieval.
    retrieval["context_attempts"] = sum(
        row["action"]["action"].startswith("context_") for row in trace
    )
    patch = "".join(
        "".join(
            difflib.unified_diff(
                task.base[path].splitlines(keepends=True),
                source[path].splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
        for path in sorted(source)
    )
    return {
        "task_id": task.task_id,
        "arm": arm,
        "status": status,
        "error": error,
        "resolved": status == "completed" and final["status"] == "completed" and final["passed"],
        "final_tests": final,
        "trace": trace,
        "context_trace": tools.log,
        "retrieval": retrieval,
        "provider_calls": calls,
        "usage": dict(usage),
        "latency_s": time.monotonic() - started,
        "first_edit_step": first_edit,
        "visible_test_runs": test_runs,
        "patch": patch,
        "summary": summary,
        "candidate_files": source,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["export", "validate", "qa", "code"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--task")
    parser.add_argument("--arm", choices=["none", "raw"], default="raw")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh"])
    args = parser.parse_args()
    if args.command == "export":
        result = export(args.out)
    elif args.command == "validate":
        result = validate(args.out, args.image)
    else:
        if not args.live:
            parser.error("model trials require --live; no fake-model fallback")
        files, spans, probes, requirements, tasks = dataset()
        task = next((item for item in tasks if item.task_id == args.task), None)
        if args.command == "code" and task is None:
            parser.error("--task must name one of: " + ", ".join(item.task_id for item in tasks))
        image = resolve_image(args.image) if args.command == "code" else None
        manifest = prepare(
            args.out,
            dataset_sha256=dataset_hash(),
            model=args.model,
            effort=args.effort,
            arm=args.arm,
            image_id=image,
            snapshot_sha256=sha(json.dumps(files, sort_keys=True)),
        )
        provider = CodexLLM(model=args.model, reasoning_effort=args.effort, timeout_s=120)
        try:
            if args.command == "code":
                if task is None or image is None:
                    raise ValueError("coding task and test image are required")
                result = coding_trial(task, provider, arm=args.arm, image=image)
                write_json(args.out / "result.json", result)
                (args.out / "candidate.patch").write_text(result["patch"])
                (args.out / "PR.md").write_text(result["summary"])
            else:
                results = []
                for probe in probes:
                    row = read_question(
                        provider,
                        probe.question,
                        probe.T,
                        files if args.arm == "raw" else {},
                        Profile(reader_max_calls=12, read_budget_bytes=BYTE_BUDGET),
                    )
                    row["probe_id"] = probe.probe_id
                    row["grade"] = grade(
                        probe, row, {s.unit_id for s in spans} if args.arm == "raw" else set()
                    )
                    row["retrieval_metrics"] = score_retrieval(
                        files,
                        spans,
                        requirements[probe.probe_id],
                        row["retrieval"] if args.arm == "raw" else [],
                        budget=BYTE_BUDGET,
                        response_cap=RESPONSE_CAP,
                        annotations_exhaustive=True,
                    )
                    write_json(args.out / f"{probe.probe_id}.json", row)
                    results.append(row)
                result = {"answers": results, "count": len(results)}
                write_json(args.out / "result.json", result)
            manifest["status"] = "completed"
        except BaseException as exc:
            manifest.update(status="failed", error=str(exc))
            raise
        finally:
            write_json(args.out / "provider-transcripts.json", provider.calls)
            write_json(args.out / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "command": args.command,
                "status": result.get("status", "completed"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
