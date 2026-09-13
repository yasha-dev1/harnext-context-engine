"""Standalone native benchmark reports, built from saved measurements only."""

from __future__ import annotations

import base64
import gzip
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml


def read_run(root: Path) -> dict:
    report = json.loads((root / "report.json").read_text())
    if not report.get("finished"):
        raise ValueError("cannot present an incomplete pilot as a completed report")
    if len(report["results"]) != len(report["selection"]["task_ids"]):
        raise ValueError("result count differs from selected tasks")
    access = report["selection"].get(
        "context_access", report["selection"]["pilot"].get("context_access", "mcp")
    )
    report.update(run_name=root.name, context_access=access, details=[])
    for index, row in enumerate(report["results"], 1):
        folder = root / row.get("artifact_folder", f"{index:02}-{row['family']}")
        calls = []
        logs = [
            p
            for p in (folder / "store/trials").glob("*.jsonl")
            if not p.stem.endswith("capability")
        ]
        logs.extend(
            p
            for p in (folder / "store/trials").glob("*.jsonl.gz")
            if not p.name.endswith("capability.jsonl.gz")
        )
        for log in logs:
            content = (
                gzip.decompress(log.read_bytes()).decode()
                if log.suffix == ".gz"
                else log.read_text()
            )
            calls.extend(json.loads(line) for line in content.splitlines()[1:])
        expected_calls = (row.get("retrieval") or {}).get("context_calls", 0)
        if len(calls) != expected_calls or (access == "none" and calls):
            raise ValueError(f"receipt count mismatch: {row['task_id']}")
        snapshot = folder / "store/snapshot.json"
        report["details"].append(
            {
                "prompt": (folder / "reader/prompt.txt").read_text(),
                "calls": calls,
                "snapshot_bytes": snapshot.stat().st_size if snapshot.exists() else None,
                "condition": yaml.safe_load((folder / "condition.yaml").read_text()),
            }
        )
    return report


def build_report(runs: list[Path], output: Path) -> Path:
    if not runs:
        raise ValueError("at least one run is required")
    reports = [read_run(root) for root in runs]
    first = reports[0]
    for report in reports[1:]:
        if report["selection"]["benchmark_sha256"] != first["selection"]["benchmark_sha256"]:
            raise ValueError("cannot pair different benchmark hashes")
        if report["selection"]["task_ids"] != first["selection"]["task_ids"]:
            raise ValueError("cannot pair different task selections")
        if [(r["task_id"], r["cutoff"], r["expected"]) for r in report["results"]] != [
            (r["task_id"], r["cutoff"], r["expected"]) for r in first["results"]
        ]:
            raise ValueError("paired task identities, cutoffs, or gold differ")
    payload = {"generated_at": datetime.now(UTC).isoformat(), "runs": reports}
    encoded = base64.b64encode(
        gzip.compress(json.dumps(payload, ensure_ascii=False).encode(), mtime=0)
    ).decode()
    template = Path(__file__).with_name("benchmark_report.template.html").read_text()
    html = template.replace("__PILOT_DATA__", encoded)
    output = output.resolve()
    if output.exists():
        raise ValueError("use a new report path; existing reports are immutable")
    patch = (
        f"*** Begin Patch\n*** Add File: {output}\n"
        + "\n".join("+" + line for line in html.splitlines())
        + "\n*** End Patch\n"
    )
    subprocess.run(["apply_patch"], input=patch, text=True, capture_output=True, check=True)
    return output
