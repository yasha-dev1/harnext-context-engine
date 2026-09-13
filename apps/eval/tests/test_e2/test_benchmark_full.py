import json
from pathlib import Path

import pytest
from harnext_builder.strategies.config import canonical
from harnext_eval.e2.benchmark import sha
from harnext_eval.e2.benchmark_full import archive_trial, freeze_population
from harnext_eval.e2.benchmark_pilot import PilotConfig, select_tasks


def test_full_population_never_silently_drops_coding_or_confirmation():
    tasks = [
        {"id": f"H-{i}", "kind": "historical", "split": "confirmation_candidate"}
        for i in range(800)
    ]
    tasks += [{"id": f"C-{i}", "kind": "coding"} for i in range(200)]
    bundle = {"tasks": tasks, "manifest": {"tasks_sha256": sha(canonical(tasks))}}
    hist, code = freeze_population(bundle, "seed")
    assert len(hist) == 800 and len(code) == 200
    assert freeze_population(bundle, "seed") == (hist, code)
    bundle["tasks"].pop()
    with pytest.raises(ValueError, match="hash"):
        freeze_population(bundle, "seed")


def test_explicit_selection_requires_declared_scope_and_rejects_coding():
    task = {
        "id": "H-1",
        "kind": "historical",
        "family": "status_at_snapshot",
        "split": "confirmation_candidate",
        "tools": {"shell": False, "native_file_access": False},
    }
    bundle = {"tasks": [task], "manifest": {"tasks_sha256": sha(canonical([task]))}}
    config = PilotConfig(
        protocol="harnext-benchmark-pilot-v1",
        benchmark=Path("x"),
        engine_profile=Path("y"),
        selection_seed="test",
        families=["status_at_snapshot"],
        model="test",
        effort="medium",
        task_ids=["H-1"],
    )
    with pytest.raises(ValueError, match="outside"):
        select_tasks(bundle, config)
    config.split = "all_historical"
    assert select_tasks(bundle, config) == [task]
    task["kind"] = "coding"
    bundle["manifest"]["tasks_sha256"] = sha(canonical([task]))
    with pytest.raises(ValueError, match="historical"):
        select_tasks(bundle, config)


def test_trial_archive_retains_json_and_compressed_native_evidence(tmp_path):
    work, destination = tmp_path / "work", tmp_path / "archive"
    work.mkdir()
    (work / "report.json").write_text('{"completed":true}')
    (work / "codex.jsonl").write_text('{"type":"turn.completed"}\n')
    archive_trial(work, destination)
    assert (destination / "report.json").read_text() == '{"completed":true}'
    assert (destination / "codex.jsonl.gz").exists()
    pins = json.loads((destination / "archive-manifest.json").read_text())["files_sha256"]
    assert pins["codex.jsonl.gz"] == sha((destination / "codex.jsonl.gz").read_bytes())
    with pytest.raises(ValueError, match="already exists"):
        archive_trial(work, destination)
