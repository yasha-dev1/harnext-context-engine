"""Small, explicit real-Codex pilot on the frozen benchmark's development tasks.

This is a native external MCP client. It does not use an evaluator-owned agent
loop, leak gold into prompts, or execute coding tasks without test admission.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import secrets
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal

import yaml
from harnext_builder.harness.codex import command
from harnext_builder.strategies.config import Strict, canonical, load_config
from harnext_builder.strategies.engine import load_artifact
from pydantic import Field

from harnext_eval.e2.benchmark import read, save, sha, timestamp
from harnext_eval.e2.benchmark_audit import score_answer
from harnext_eval.e2.mcp_experiment import audit_receipts


class PilotConfig(Strict):
    protocol: Literal["harnext-benchmark-pilot-v1"]
    benchmark: Path
    engine_profile: Path
    selection_seed: str
    split: Literal["development"] = "development"
    families: list[str] = Field(min_length=1, max_length=8)
    harness: Literal["codex"] = "codex"
    model: str
    effort: Literal["low", "medium", "high", "xhigh"]
    agent_timeout_s: int = Field(default=240, ge=1, le=600)
    build_timeout_s: int = Field(default=300, ge=1)
    startup_timeout_s: int = Field(default=120, ge=1)
    transcript_bytes: int = Field(default=16777216, ge=1024)
    capability_probe: bool = True
    context_access: Literal["mcp", "none"] = "mcp"


def trial_prompt(task, context_access):
    if context_access == "mcp":
        return task["agent_prompt"]
    instruction = "Use only the Harnext context-engine MCP tools."
    if task["agent_prompt"].count(instruction) != 1:
        raise ValueError("expected exactly one historical MCP availability instruction")
    return task["agent_prompt"].replace(
        instruction,
        "No context-engine MCP tools or stored context are provided in this trial. "
        "Answer from the question and your existing knowledge only. "
        "Return evidence_ids=[] because no sources are available.",
    )


def select_tasks(bundle, config):
    if sha(canonical(bundle["tasks"])) != bundle["manifest"]["tasks_sha256"]:
        raise ValueError("benchmark hash mismatch")
    if len(config.families) != len(set(config.families)):
        raise ValueError("duplicate pilot families")
    tasks = []
    for family in config.families:
        candidates = [
            t
            for t in bundle["tasks"]
            if t["kind"] == "historical" and t["split"] == config.split and t["family"] == family
        ]
        if not candidates:
            raise ValueError(f"no eligible historical development task: {family}")
        task = min(candidates, key=lambda t: sha(config.selection_seed + "|" + t["id"]))
        if task["tools"]["shell"] or task["tools"]["native_file_access"]:
            raise ValueError("historical task does not have an MCP-only contract")
        tasks.append(task)
    return tasks


def answer_schema(family):
    # Derive the type from the public task family, never the hidden answer value.
    if family.endswith("history"):
        value = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "at": {"type": "string"},
                    "from": {"type": ["string", "null"]},
                    "to": {"type": ["string", "null"]},
                },
                "required": ["at", "from", "to"],
                "additionalProperties": False,
            },
        }
    elif family == "closed_at_snapshot":
        value = {"type": "boolean"}
    else:
        value = {"type": ["string", "null"]}
    return {
        "type": "object",
        "properties": {
            "value": value,
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["value", "evidence_ids"],
        "additionalProperties": False,
    }


def bind_evidence(task, artifact):
    required = set(task["required_evidence_ids"])
    evidence = {e["id"]: e for e in task["evidence"] if e["id"] in required}
    spans = []
    for unit, entry in evidence.items():
        # Equivalent exact encodings: raw source text or a JSON-escaped string
        # inside a graph/ledger document. Both expose the complete source record.
        variants = {entry["text"], canonical(entry["text"])[1:-1]}
        for path, text in artifact["payload"]["documents"].items():
            for quote in variants:
                offset = 0
                while (index := text.find(quote, offset)) >= 0:
                    start = len(text[:index].encode())
                    spans.append(
                        {
                            "unit_id": unit,
                            "path": path,
                            "start": start,
                            "end": start + len(quote.encode()),
                            "quote": quote,
                        }
                    )
                    offset = index + len(quote)
    covered = {s["unit_id"] for s in spans}
    if covered != required:
        raise ValueError(
            "required evidence missing from stored view: " + str(sorted(required - covered))
        )
    return {
        "snapshot_sha256": artifact["sha256"],
        "required_groups": [[u] for u in sorted(required)],
        "spans": spans,
        "annotations_exhaustive": False,
        "binding": "exact complete source record, raw or reversibly JSON-escaped; no ID-only credit",
    }


def verify_native_item(item, allowed):
    kind = item.get("type")
    if kind == "mcp_tool_call":
        if item.get("server") != "harnext" or item.get("tool") not in allowed:
            raise ValueError(
                "unapproved MCP server/tool: "
                + str(item.get("server"))
                + "/"
                + str(item.get("tool"))
            )
    elif kind not in {"agent_message", "reasoning", "error"}:
        raise ValueError("forbidden native tool/item: " + str(kind))


async def run_codex(config_path, pilot, prompt, schema, output, *, workspace):
    config = load_config(config_path) if pilot.context_access == "mcp" else None
    allowed_tools = config.trial.tools if config else []
    output.mkdir(parents=True, exist_ok=True)
    schema_path = output / "answer-schema.json"
    save(schema_path, schema)
    (output / "prompt.txt").write_text(prompt)
    argv = command(pilot.model, pilot.effort, str(workspace), tools_enabled=False)
    overrides: dict[str, object] = {
        "agents.enabled": False,
        "features.multi_agent_v2": False,
        "features.in_app_browser": False,
        "features.unified_exec": False,
    }
    if config:
        overrides.update(
            {
                "mcp_servers.harnext.command": sys.executable,
                "mcp_servers.harnext.args": [
                    "-m",
                    "harnext_mcp.experiment",
                    "--config",
                    str(config_path),
                ],
                "mcp_servers.harnext.enabled_tools": allowed_tools,
                "mcp_servers.harnext.startup_timeout_sec": pilot.startup_timeout_s,
                "mcp_servers.harnext.tool_timeout_sec": 60,
                "mcp_servers.harnext.required": True,
                "mcp_servers.harnext.default_tools_approval_mode": "approve",
            }
        )
    for key, value in overrides.items():
        argv.extend(["-c", key + "=" + json.dumps(value)])
    argv.extend(
        [
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output / "answer.json"),
            "-",
        ]
    )
    started = time.monotonic()
    metadata = {
        "argv": argv,
        "model": pilot.model,
        "effort": pilot.effort,
        "codex_version": (
            await asyncio.to_thread(subprocess.check_output, ["codex", "--version"], text=True)
        ).strip(),
        "prompt_sha256": sha(prompt),
        "native_tools_enabled": False,
        "context_access": pilot.context_access,
        "allowed_mcp_tools": allowed_tools,
        "capability_enforcement": "CLI feature controls + empty workspace + live native-event allowlist",
        "returncode": None,
    }
    save(output / "launch.json", metadata)
    events = []
    failure = None
    with (
        (output / "codex.stderr").open("w") as stderr,
        (output / "codex.jsonl").open("w") as transcript,
    ):
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=stderr,
            start_new_session=True,
            limit=2**22,
        )
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(prompt.encode())
        await proc.stdin.drain()
        proc.stdin.close()

        async def consume():
            size = 0
            assert proc.stdout is not None
            while line := await proc.stdout.readline():
                size += len(line)
                transcript.write(line.decode())
                transcript.flush()
                if size > pilot.transcript_bytes:
                    raise ValueError("native transcript byte cap exceeded")
                event = json.loads(line)
                events.append(event)
                if event.get("type") in {"item.started", "item.completed"}:
                    verify_native_item(event["item"], allowed_tools)
            await proc.wait()

        try:
            await asyncio.wait_for(consume(), pilot.agent_timeout_s)
        except (TimeoutError, ValueError) as exc:
            failure = type(exc).__name__ + ": " + str(exc)
            if proc.returncode is None:
                os.killpg(proc.pid, signal.SIGKILL)
                await proc.wait()
    metadata.update(
        returncode=proc.returncode, elapsed_s=time.monotonic() - started, failure=failure
    )
    save(output / "launch.json", metadata)
    completed = any(e.get("type") == "turn.completed" for e in events)
    errors = [e for e in events if e.get("type") in {"turn.failed", "error"}]
    answer = read(output / "answer.json") if (output / "answer.json").exists() else None
    usage = next(
        (e.get("usage", {}) for e in reversed(events) if e.get("type") == "turn.completed"), {}
    )
    return {
        "completed": completed and proc.returncode == 0 and not failure and not errors,
        "failure": failure or (str(errors)[:1000] if errors else None),
        "answer": answer,
        "usage": usage,
        "elapsed_s": metadata["elapsed_s"],
        "native_events": events,
        "trace_sha256": sha((output / "codex.jsonl").read_bytes()),
    }


def compare_receipts(events, log):
    native = [
        e["item"]
        for e in events
        if e.get("type") == "item.completed" and e.get("item", {}).get("type") == "mcp_tool_call"
    ]
    calls = [json.loads(line) for line in log.read_text().splitlines()][1:]
    if len(native) != len(calls):
        raise ValueError(f"native/server tool count differs: {len(native)} / {len(calls)}")
    for item, row in zip(native, calls, strict=True):
        if item["tool"] != row["tool"] or any(
            row["arguments"].get(k) != v for k, v in item["arguments"].items()
        ):
            raise ValueError("native/server tool arguments differ")
        texts = [
            c["text"]
            for c in (item.get("result") or {}).get("content", [])
            if c.get("type") == "text"
        ]
        if row["output"] not in texts:
            raise ValueError("native result differs from server receipt")
    return {"matched_calls": len(calls), "exact_response_match": True, "forbidden_native_calls": 0}


async def capability_probe(base_config, pilot, root):
    out = root / "capability-probe"
    out.mkdir()
    config = base_config.model_copy(deep=True)
    config.trial.run_id = root.name + "-capability"
    config.trial.task_id = "capability-native-access"
    path = out / "condition.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    with (
        tempfile.TemporaryDirectory(prefix="harnext-native-canary-") as canary_dir,
        tempfile.TemporaryDirectory(prefix="harnext-reader-empty-") as work,
    ):
        nonce = secrets.token_hex(24)
        canary = Path(canary_dir) / "native-read-canary.txt"
        canary.write_text(nonce)
        prompt = (
            f"Capability verification, not a benchmark question. Try to read the file {canary} using a native shell or native file tool if one is available. "
            'This file is outside Harnext MCP. If no native read/shell capability is available, return value="unavailable" and evidence_ids=[]. '
            "Do not guess file contents or substitute retrieved context."
        )
        result = await run_codex(
            path, pilot, prompt, answer_schema("capability"), out, workspace=work
        )
        leaked = nonce in canonical(result)
        report = {k: v for k, v in result.items() if k != "native_events"}
        report.update(
            passed=result["completed"]
            and not leaked
            and (result["answer"] or {}).get("value") == "unavailable",
            canary_leaked=leaked,
            limitation="Behavioral negative check plus configured/observed tool boundaries; not a formal OS-isolation proof.",
        )
        save(out / "report.json", report)
        if not report["passed"]:
            raise ValueError("native capability check failed; benchmark runs stopped")
        print(canonical({"capability_probe": "passed"}), flush=True)
        return report


async def run(pilot, root):
    root.mkdir(parents=True, exist_ok=False)
    bundle = read(pilot.benchmark)
    tasks = select_tasks(bundle, pilot)
    selection = {
        "benchmark_sha256": bundle["manifest"]["tasks_sha256"],
        "pilot": pilot.model_dump(mode="json"),
        "task_ids": [t["id"] for t in tasks],
        "scope": "development pilot only; no confirmation/coding tasks",
        "source_commit": (
            await asyncio.to_thread(
                subprocess.check_output, ["git", "rev-parse", "HEAD"], text=True
            )
        ).strip(),
        "runner_sha256": sha(await asyncio.to_thread(Path(__file__).read_bytes)),
        "context_access": pilot.context_access,
    }
    save(root / "selection.json", selection)
    results = []
    gate = None
    for index, task in enumerate(tasks, 1):
        folder = root / f"{index:02}-{task['family']}"
        folder.mkdir()
        config = load_config(pilot.engine_profile)
        config.artifact_dir = folder / "store"
        config.observed_at = config.valid_at = timestamp(task["cutoff"])
        config.trial.run_id = root.name + f"-{index:02}"
        config.trial.task_id = task["id"]
        config.trial.harness = "codex-cli-native-mcp"
        config.trial.model = pilot.model
        config.trial.effort = pilot.effort
        config.trial.timeout_s = pilot.agent_timeout_s + pilot.startup_timeout_s
        path = folder / "condition.yaml"
        path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
        save(
            folder / "task.public.json",
            {k: task[k] for k in ["id", "family", "agent_prompt", "tools", "cutoff"]},
        )
        if pilot.context_access == "none":
            if pilot.capability_probe and gate is None:
                gate = await capability_probe(config, pilot, root)
            print(
                canonical({"step": index, "task": task["id"], "phase": "native-codex-no-mcp"}),
                flush=True,
            )
            with tempfile.TemporaryDirectory(prefix="harnext-reader-empty-") as work:
                execution = await run_codex(
                    None,
                    pilot,
                    trial_prompt(task, "none"),
                    answer_schema(task["family"]),
                    folder / "reader",
                    workspace=work,
                )
            result = {
                "task_id": task["id"],
                "family": task["family"],
                "cutoff": task["cutoff"],
                "context_access": "none",
                "completed": execution["completed"],
                "failure": execution["failure"],
                "actual": execution["answer"],
                "expected": task["expected_answer"],
                "usage": execution["usage"],
                "reader_seconds": execution["elapsed_s"],
                "build_seconds": 0.0,
                "snapshot_sha256": None,
                "source_count": 0,
                "trace_sha256": execution["trace_sha256"],
                "value_correct": score_answer(task, execution["answer"])["value_correct"]
                if execution["completed"]
                else False,
                "retrieval": None,
                "native_receipt_match": None,
                "context_calls": 0,
                "returned_context_bytes": 0,
                "required_support_exposure": 0.0,
                "retrieval_note": "No retrieval system present; algorithm recall/precision are not applicable. Required evidence exposure is zero.",
            }
            save(folder / "report.json", result)
            results.append(result)
            save(
                root / "report.json",
                {
                    "selection": selection,
                    "capability_probe": gate,
                    "results": results,
                    "finished": len(results) == len(tasks),
                    "completed": sum(r["completed"] for r in results),
                    "correct": sum(r["value_correct"] for r in results),
                },
            )
            print(
                canonical(
                    {
                        "step": index,
                        "correct": result["value_correct"],
                        "completed": result["completed"],
                        "calls": 0,
                    }
                ),
                flush=True,
            )
            if execution["failure"]:
                raise ValueError("no-MCP execution failed; inspect native trace before continuing")
            continue
        print(canonical({"step": index, "task": task["id"], "phase": "build"}), flush=True)
        started = time.monotonic()
        with (
            (folder / "build.stdout").open("w") as stdout,
            (folder / "build.stderr").open("w") as stderr,
        ):
            built = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "harnext_builder.strategies",
                "build",
                "--config",
                str(path),
                stdout=stdout,
                stderr=stderr,
            )
            try:
                await asyncio.wait_for(built.wait(), pilot.build_timeout_s)
            except TimeoutError:
                built.kill()
                await built.wait()
                raise
        if built.returncode:
            raise ValueError("snapshot build failed; inspect " + str(folder / "build.stderr"))
        build_seconds = time.monotonic() - started
        artifact = load_artifact(config.artifact_dir / "snapshot.json")
        gold = bind_evidence(task, artifact)
        save(
            folder / "gold.private.json",
            {"value": task["expected_answer"]["value"], "retrieval": gold},
        )
        if pilot.capability_probe and gate is None:
            gate = await capability_probe(config, pilot, root)
        print(
            canonical(
                {
                    "step": index,
                    "task": task["id"],
                    "phase": "native-codex",
                    "sources": len(artifact["payload"]["source_ids"]),
                }
            ),
            flush=True,
        )
        with tempfile.TemporaryDirectory(prefix="harnext-reader-empty-") as work:
            execution = await run_codex(
                path,
                pilot,
                trial_prompt(task, pilot.context_access),
                answer_schema(task["family"]),
                folder / "reader",
                workspace=work,
            )
        log = config.artifact_dir / "trials" / f"{config.trial.run_id}.jsonl"
        result = {
            "task_id": task["id"],
            "family": task["family"],
            "cutoff": task["cutoff"],
            "completed": execution["completed"],
            "failure": execution["failure"],
            "actual": execution["answer"],
            "expected": task["expected_answer"],
            "usage": execution["usage"],
            "reader_seconds": execution["elapsed_s"],
            "build_seconds": build_seconds,
            "snapshot_sha256": artifact["sha256"],
            "source_count": len(artifact["payload"]["source_ids"]),
            "trace_sha256": execution["trace_sha256"],
            "value_correct": score_answer(task, execution["answer"])["value_correct"]
            if execution["completed"]
            else False,
        }
        if log.exists():
            result["retrieval"] = audit_receipts(artifact, log, gold)
            result["native_receipt_match"] = compare_receipts(execution["native_events"], log)
        else:
            result["retrieval"] = None
            result["native_receipt_match"] = None
        save(folder / "report.json", result)
        results.append(result)
        save(
            root / "report.json",
            {
                "selection": selection,
                "capability_probe": gate,
                "results": results,
                "finished": len(results) == len(tasks),
                "completed": sum(r["completed"] for r in results),
                "correct": sum(r["value_correct"] for r in results),
            },
        )
        print(
            canonical(
                {
                    "step": index,
                    "correct": result["value_correct"],
                    "completed": result["completed"],
                    "calls": (result["retrieval"] or {}).get("context_calls"),
                    "recall": (result["retrieval"] or {}).get("required_group_recall"),
                }
            ),
            flush=True,
        )
        del artifact
        gc.collect()
        if execution["failure"] and "forbidden" in execution["failure"]:
            raise ValueError("native capability violation; stopping remaining pilot tasks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required for real external harness execution")
    pilot = PilotConfig.model_validate(yaml.safe_load(args.config.read_text()))
    for field in ["benchmark", "engine_profile"]:
        value = getattr(pilot, field)
        if not value.is_absolute():
            setattr(pilot, field, (args.config.resolve().parent / value).resolve())
    asyncio.run(run(pilot, args.out.resolve()))


if __name__ == "__main__":
    main()
