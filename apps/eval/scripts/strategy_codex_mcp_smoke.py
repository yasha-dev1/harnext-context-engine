"""One explicitly opted-in Codex agent trial using native MCP, not an internal reader."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from harnext_builder.harness.codex import command
from harnext_builder.strategies.config import load_config
from harnext_builder.strategies.engine import load_artifact
from harnext_eval.e2.mcp_experiment import grade


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("output must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config.trial.run_id = out.name
    config.trial.harness = "codex-cli-native-mcp"
    config.trial.model = "gpt-5.6-luna"
    config.trial.effort = "medium"
    config.trial.transport = "stdio"
    question = "At the provided Harnext snapshot, who is assigned to CTX-41? Use its MCP context tools to obtain evidence. Return the canonical person key as value and unknown=false."
    (out / "task.json").write_text(
        json.dumps({"task_id": config.trial.task_id, "prompt": question})
    )
    artifact = load_artifact(config.artifact_dir / "snapshot.json")
    quote = "CTX-41 is now assigned to Noah."
    spans = []
    for path, text in artifact["payload"]["documents"].items():
        start = text.encode().find(quote.encode())
        if start >= 0:
            spans.append(
                {
                    "unit_id": "current-owner",
                    "path": path,
                    "start": start,
                    "end": start + len(quote.encode()),
                    "quote": quote,
                }
            )
    (out / "gold.json").write_text(
        json.dumps(
            {
                "value": "person:Noah",
                "retrieval": {
                    "snapshot_sha256": artifact["sha256"],
                    "required_groups": [["current-owner"]],
                    "spans": spans,
                    "annotations_exhaustive": False,
                },
            }
        )
    )
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}, "unknown": {"type": "boolean"}},
        "required": ["value", "unknown"],
        "additionalProperties": False,
    }
    (out / "answer-schema.json").write_text(json.dumps(schema))
    config.evaluation.grading = "qa_exact"
    config.evaluation.task_file = out / "task.json"
    config.evaluation.gold_file = out / "gold.json"
    config.evaluation.answer_file = out / "answer.json"
    config.evaluation.agent_trace = out / "codex.jsonl"
    config_path = out / "condition.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    with tempfile.TemporaryDirectory(prefix="harnext-external-agent-") as workspace:
        argv = command(config.trial.model, config.trial.effort, workspace, tools_enabled=False)
        overrides = {
            "mcp_servers.harnext.command": sys.executable,
            "mcp_servers.harnext.args": [
                "-m",
                "harnext_mcp.experiment",
                "--config",
                str(config_path),
            ],
            "mcp_servers.harnext.startup_timeout_sec": 60,
            "mcp_servers.harnext.tool_timeout_sec": 60,
            "mcp_servers.harnext.required": True,
            "mcp_servers.harnext.default_tools_approval_mode": "approve",
        }
        for key, value in overrides.items():
            argv.extend(["-c", key + "=" + json.dumps(value)])
        argv.extend(
            [
                "--output-schema",
                str(out / "answer-schema.json"),
                "--output-last-message",
                str(out / "answer.json"),
                "-",
            ]
        )
        with (out / "codex.jsonl").open("w") as stdout, (out / "codex.stderr").open("w") as stderr:
            result = subprocess.run(
                argv, input=question, text=True, stdout=stdout, stderr=stderr, timeout=240
            )
        (out / "launch.json").write_text(
            json.dumps(
                {
                    "argv": argv,
                    "returncode": result.returncode,
                    "codex_version": subprocess.run(
                        ["codex", "--version"], capture_output=True, text=True, check=True
                    ).stdout.strip(),
                    "question": question,
                },
                indent=2,
            )
        )
        if result.returncode != 0:
            raise RuntimeError(f"Codex failed; inspect {out / 'codex.stderr'}")
    report = grade(config)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    if (
        report["context_calls"] < 1
        or not report["value_correct"]
        or report["required_group_recall"] != 1
    ):
        raise RuntimeError("external Codex MCP verification did not pass")
    print(
        json.dumps(
            {
                "passed": True,
                "report": str(out / "report.json"),
                "context_calls": report["context_calls"],
                "bytes": report["bytes_returned"],
            }
        )
    )


if __name__ == "__main__":
    main()
