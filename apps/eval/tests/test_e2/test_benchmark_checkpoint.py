import copy
import json

import pytest
from harnext_eval.e2.benchmark import sha
from harnext_eval.e2.benchmark_checkpoint import (
    portable_manifest,
    validate_checkpoint,
    verify_files,
)


def manifest(root):
    return {
        "protocol": {
            "pilot": f"{root}/apps/eval/configs/pilot.yaml",
            "output": f"{root}/results",
            "scratch": f"{root}/scratch",
            "conditions": ["mcp", "none"],
        },
        "pilot": {
            "benchmark": f"{root}/data/tasks.json",
            "engine_profile": f"{root}/data/profile.yaml",
            "model": "gpt-5.6-luna",
            "effort": "medium",
        },
        "source_sha256": "frozen",
    }


def test_relocation_preserves_protocol_and_original_manifest():
    original = manifest("/old/checkout")
    unchanged = copy.deepcopy(original)
    assert portable_manifest(original) == portable_manifest(manifest("/new/checkout"))
    assert original == unchanged
    for section, key, value in [
        ("pilot", "model", "other"),
        ("pilot", "effort", "high"),
        ("protocol", "conditions", ["mcp"]),
    ]:
        changed = manifest("/new/checkout")
        changed[section][key] = value
        assert portable_manifest(original) != portable_manifest(changed)


def test_corruption_and_missing_evidence_rejected(tmp_path):
    path = tmp_path / "answer.json"
    path.write_text("original")
    hashes = {"answer.json": sha(path.read_bytes())}
    verify_files(tmp_path, hashes)
    path.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        verify_files(tmp_path, hashes)
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        verify_files(tmp_path, hashes)
    with pytest.raises(ValueError, match="escapes"):
        verify_files(tmp_path, {"../outside": "hash"})


def test_interrupted_attempt_cannot_replace_completed_report(tmp_path):
    interrupted = tmp_path / "interruptions/task/mcp"
    interrupted.mkdir(parents=True)
    (interrupted / "report.json").write_text("{}")
    (tmp_path / "checkpoint.json").write_text(
        json.dumps(
            {
                "implementation_files_sha256": {},
                "frozen_files_sha256": {},
                "completed_reports": ["cases/task/mcp/report.json"],
            }
        )
    )
    with pytest.raises(ValueError, match="completed checkpoint trials are missing"):
        validate_checkpoint(tmp_path, tmp_path)
