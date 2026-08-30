"""JSONL and checksum CLI tests for docs/evaluation-spec.md §5."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from harnext_eval.corpus.synthetic import generate_synthetic_corpus
from harnext_eval.types import Probe


def test_module_cli_writes_jsonl_and_sha256(tmp_path: Path) -> None:
    replay = tmp_path / "replay.jsonl"
    handle = generate_synthetic_corpus(
        replay, seed=9, event_count=600, entity_count=24
    )
    assert isinstance(handle.window, tuple)
    output = tmp_path / "probes.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "harnext_eval.probes.gen",
            "--replay",
            str(replay),
            "--out",
            str(output),
            "--per-family",
            "3",
            "--seed",
            "1",
            "--probe-start",
            handle.window[0].isoformat(),
            "--probe-end",
            handle.window[1].isoformat(),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    lines = output.read_text(encoding="utf-8").splitlines()
    assert "wrote 18 probes" in completed.stdout
    assert len(lines) == 18
    assert all(Probe.model_validate_json(line) for line in lines)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    assert Path(f"{output}.sha256").read_text(encoding="utf-8") == (
        f"{digest}  {output.name}\n"
    )
