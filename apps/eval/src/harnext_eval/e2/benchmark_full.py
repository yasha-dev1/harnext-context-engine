"""Resumable historical phase of the 1,000-task benchmark; coding stays explicit."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import gzip
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Literal

import yaml
from harnext_builder.strategies.config import Strict, canonical, load_config
from pydantic import Field

from harnext_eval.e2.benchmark import read, sha
from harnext_eval.e2.benchmark_pilot import PilotConfig
from harnext_eval.e2.benchmark_report import build_report


class FullConfig(Strict):
    protocol: Literal["harnext-full-benchmark-v1"]
    pilot: Path
    run_id: str
    output: Path
    scratch: Path
    conditions: list[Literal["mcp", "none"]]
    workers: int = Field(default=2, ge=1, le=4)
    order_seed: str
    checkpoint_every_pairs: int = Field(default=20, ge=1)
    commit_results: bool = False
    minimum_free_gib: int = Field(default=15, ge=5)
    authorization: str
    coding_policy: Literal["require_executed_base_reference_admission"]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def freeze_population(bundle, seed):
    if sha(canonical(bundle["tasks"])) != bundle["manifest"]["tasks_sha256"]:
        raise ValueError("frozen benchmark task hash differs")
    tasks = bundle["tasks"]
    if Counter(t["kind"] for t in tasks) != {"historical": 800, "coding": 200}:
        raise ValueError("expected the complete 800+200 frozen population")
    historical = sorted(
        (t for t in tasks if t["kind"] == "historical"), key=lambda t: sha(seed + t["id"])
    )
    coding = [t for t in tasks if t["kind"] == "coding"]
    return historical, coding


def archive_trial(work, destination):
    """Publish audit material atomically; large reproducible stores stay out of Git."""
    if destination.exists():
        raise ValueError("trial archive already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(destination.name + ".staging")
    stage.mkdir(exist_ok=False)
    for source in sorted(work.rglob("*")):
        if (
            not source.is_file()
            or source.name in {"snapshot.json", "report.html"}
            or source.suffix == ".lock"
        ):
            continue
        relative = source.relative_to(work)
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".json" or source.suffix in {".yaml", ".txt"}:
            shutil.copyfile(source, target)
        else:
            target = target.with_name(target.name + ".gz")
            target.write_bytes(gzip.compress(source.read_bytes(), mtime=0))
    identities = {}
    for source in work.rglob("snapshot.json"):
        # Retain full frozen documents containing annotated evidence, sufficient
        # to inspect the scored support; the input corpus and recipe rebuild the rest.
        artifact = read(source)
        gold_file = source.parent.parent / "gold.private.json"
        gold = read(gold_file)["retrieval"] if gold_file.exists() else {"spans": []}
        paths = {span["path"] for span in gold["spans"]}
        support = {path: artifact["payload"]["documents"][path] for path in sorted(paths)}
        out = stage / "evidence-documents.json.gz"
        out.write_bytes(gzip.compress(canonical(support).encode(), mtime=0))
        identities[str(source.relative_to(work))] = {
            "sha256": artifact["sha256"],
            "bytes": source.stat().st_size,
        }
    atomic_json(stage / "store-identities.json", identities)
    files = {
        str(p.relative_to(stage)): sha(p.read_bytes()) for p in stage.rglob("*") if p.is_file()
    }
    atomic_json(
        stage / "archive-manifest.json",
        {
            "files_sha256": files,
            "snapshot_policy": "source/config/implementation pins and evidence documents retained; full transient store not versioned",
        },
    )
    stage.replace(destination)


def commit_checkpoint(output, message):
    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True).strip()
    if staged:
        raise ValueError(
            "checkpoint refused: unrelated staged changes need to be committed separately"
        )
    subprocess.run(["git", "add", "--", str(output)], check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        return
    subprocess.run(["git", "commit", "-m", message], check=True)


async def run(config):
    config.output.mkdir(parents=True, exist_ok=True)
    (config.output / ".gitignore").write_text("*.staging/\n*.tmp\n")
    config.scratch.mkdir(parents=True, exist_ok=True)
    lock = (config.scratch / "coordinator.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pilot = PilotConfig.model_validate(yaml.safe_load(config.pilot.read_text()))
    for name in ("benchmark", "engine_profile"):
        value = getattr(pilot, name)
        if not value.is_absolute():
            setattr(pilot, name, (config.pilot.parent / value).resolve())
    bundle = read(pilot.benchmark)
    historical, coding = freeze_population(bundle, config.order_seed)
    profile = load_config(pilot.engine_profile)
    source_hash = await asyncio.to_thread(lambda: sha(profile.source_jsonl.read_bytes()))
    if source_hash != bundle["manifest"]["context_sources_sha256"]:
        raise ValueError("context corpus differs from frozen manifest")
    if len(set(config.conditions)) != len(config.conditions) or set(config.conditions) != {
        "mcp",
        "none",
    }:
        raise ValueError("full comparison requires exactly mcp and none")
    manifest = {
        "run_id": config.run_id,
        "protocol": config.model_dump(mode="json"),
        "benchmark_sha256": bundle["manifest"]["tasks_sha256"],
        "source_sha256": source_hash,
        "engine_profile_sha256": sha(pilot.engine_profile.read_bytes()),
        "pilot": pilot.model_dump(mode="json"),
        "historical_task_ids": [t["id"] for t in historical],
        "coding_task_ids": [t["id"] for t in coding],
        "scope": "full benchmark requested; historical phase executes 800 tasks in two conditions; coding phase remains explicitly pending admission",
        "confirmation_note": "User authorized all tasks; confirmation candidates used here are no longer untouched for later configuration selection.",
    }
    manifest_path = config.output / "manifest.json"
    if manifest_path.exists() and read(manifest_path) != manifest:
        raise ValueError("cannot resume with a different protocol, input, or model")
    atomic_json(manifest_path, manifest)
    atomic_json(
        config.output / "coding-admission.json",
        {
            "status": "pending",
            "tasks": [
                {
                    "task_id": t["id"],
                    "base_sha": t["base_sha"],
                    "status": "not_run_pending_executed_test_admission",
                    "admission_at_freeze": t["admission"],
                }
                for t in coding
            ],
        },
    )
    provenance_path = config.output / "implementation.json"
    pins = {
        p.name: sha(p.read_bytes())
        for p in [Path(__file__), Path(__file__).with_name("benchmark_pilot.py")]
    }
    if provenance_path.exists() and read(provenance_path)["files"] != pins:
        raise ValueError(
            "runner changed since this run started; start a new run or adjudicate explicitly"
        )
    atomic_json(provenance_path, {"files": pins})
    results = {}
    for task in historical:
        for access in config.conditions:
            saved = config.output / "cases" / task["id"] / access / "report.json"
            if saved.exists():
                results[(task["id"], access)] = read(saved)["results"][0]
    stopped = asyncio.Event()
    checkpoint_lock = asyncio.Lock()
    fatal = []

    def summary(status):
        arms = {}
        for access in config.conditions:
            rows = [r for (tid, mode), r in results.items() if mode == access]
            arms[access] = {
                "recorded": len(rows),
                "target": 800,
                "completed": sum(r["completed"] for r in rows),
                "correct": sum(r["value_correct"] for r in rows),
                "by_family": {
                    family: {
                        "recorded": sum(r["family"] == family for r in rows),
                        "correct": sum(r["family"] == family and r["value_correct"] for r in rows),
                    }
                    for family in pilot.families
                },
            }
        return {
            "status": status,
            "updated_at_unix": time.time(),
            "historical": arms,
            "coding": {"pending_admission": 200, "executed": 0},
            "full_benchmark_complete": False,
            "errors": fatal,
        }

    async def checkpoint(status, commit=False):
        async with checkpoint_lock:
            atomic_json(config.output / "summary.json", summary(status))
            if config.commit_results and commit:
                await asyncio.to_thread(
                    commit_checkpoint,
                    config.output,
                    f"eval: checkpoint {config.run_id} ({len(results)}/1600 historical trials)",
                )

    async def trial(task, access, probe):
        destination = config.output / "cases" / task["id"] / access
        if (task["id"], access) in results:
            return
        if shutil.disk_usage(config.scratch).free < config.minimum_free_gib * 2**30:
            raise RuntimeError("minimum free disk guard reached")
        work = config.scratch / task["id"] / access
        if work.exists():
            raise RuntimeError(
                f"interrupted attempt retained at {work}; adjudicate before retrying"
            )
        attempt = pilot.model_copy(deep=True)
        attempt.split = "all_historical"
        attempt.task_ids = [task["id"]]
        attempt.context_access = access
        attempt.capability_probe = probe
        attempt.generate_html = False
        task_dir = work.parent
        task_dir.mkdir(parents=True, exist_ok=True)
        config_path = task_dir / f"{access}.yaml"
        config_path.write_text(yaml.safe_dump(attempt.model_dump(mode="json")))
        print(canonical({"task": task["id"], "condition": access, "phase": "starting"}), flush=True)
        with (
            (task_dir / f"{access}.stdout").open("w") as out,
            (task_dir / f"{access}.stderr").open("w") as err,
        ):
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "harnext_eval.e2.benchmark_pilot",
                "--config",
                str(config_path),
                "--out",
                str(work),
                "--live",
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            try:
                await asyncio.wait_for(
                    proc.wait(), attempt.build_timeout_s + 2 * attempt.agent_timeout_s + 300
                )
            except TimeoutError:
                os.killpg(proc.pid, signal.SIGKILL)
                await proc.wait()
                raise RuntimeError("trial process exceeded whole-attempt timeout") from None
        report_path = work / "report.json"
        if report_path.exists():
            report = read(report_path)
            await asyncio.to_thread(archive_trial, work, destination)
            result = report["results"][0]
            results[(task["id"], access)] = result
            print(
                canonical(
                    {
                        "task": task["id"],
                        "condition": access,
                        "completed": result["completed"],
                        "correct": result["value_correct"],
                        "recorded": len(results),
                    }
                ),
                flush=True,
            )
            await checkpoint("running")
            # Only our completed scratch directory is removed, after audit publication.
            await asyncio.to_thread(shutil.rmtree, work)
            if result["failure"]:
                raise RuntimeError(
                    "native execution failure; remaining trials paused for review: "
                    + str(result["failure"])[:500]
                )
        if proc.returncode or not report_path.exists() and not destination.exists():
            raise RuntimeError(
                f"trial failed with exit={proc.returncode}; logs retained in {task_dir}"
            )

    await checkpoint("starting", commit=True)
    # First paired task admits both native tool boundaries before fan-out.
    try:
        for access in config.conditions:
            await trial(historical[0], access, True)
        await checkpoint("running", commit=True)
        queue = asyncio.Queue()
        for index, task in enumerate(historical[1:], 1):
            queue.put_nowait((index, task))

        async def worker():
            while not stopped.is_set():
                try:
                    index, task = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    order = (
                        config.conditions if index % 2 == 0 else list(reversed(config.conditions))
                    )
                    for access in order:
                        if stopped.is_set():
                            return
                        await trial(task, access, False)
                    if len(results) % (2 * config.checkpoint_every_pairs) == 0:
                        await checkpoint("running", commit=True)
                except Exception as exc:
                    fatal.append({"task": task["id"], "error": str(exc)})
                    stopped.set()
                finally:
                    queue.task_done()

        await asyncio.gather(*(worker() for _ in range(config.workers)))
    except Exception as exc:
        fatal.append({"phase": "capability_preflight", "error": str(exc)})
    await checkpoint(
        "paused_on_error" if fatal else "historical_complete_coding_pending", commit=True
    )
    if not fatal:
        report_dirs = []
        for access in config.conditions:
            directory = config.output / "reports" / access
            report_dirs.append(directory)
            rows = []
            for task in historical:
                row = dict(results[(task["id"], access)])
                row["artifact_folder"] = f"../../cases/{task['id']}/{access}/01-{task['family']}"
                rows.append(row)
            atomic_json(
                directory / "report.json",
                {
                    "selection": {
                        "benchmark_sha256": manifest["benchmark_sha256"],
                        "task_ids": manifest["historical_task_ids"],
                        "pilot": pilot.model_dump(mode="json"),
                        "context_access": access,
                        "scope": "800 historical tasks, including confirmation candidates; 200 coding tasks remain pending admission",
                    },
                    "finished": True,
                    "results": rows,
                    "completed": sum(r["completed"] for r in rows),
                    "correct": sum(r["value_correct"] for r in rows),
                    "capability_probe": read(
                        config.output
                        / "cases"
                        / historical[0]["id"]
                        / access
                        / "capability-probe/report.json"
                    ),
                },
            )
        await asyncio.to_thread(build_report, report_dirs, config.output / "report.html")
        await checkpoint("historical_complete_coding_pending", commit=True)
    print(
        canonical(summary("paused_on_error" if fatal else "historical_complete_coding_pending")),
        flush=True,
    )
    lock.close()
    if fatal:
        raise RuntimeError("benchmark paused; inspect committed summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required")
    config = FullConfig.model_validate(yaml.safe_load(args.config.read_text()))
    for field in ("pilot", "output", "scratch"):
        value = getattr(config, field)
        if not value.is_absolute():
            setattr(config, field, (args.config.resolve().parent / value).resolve())
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
