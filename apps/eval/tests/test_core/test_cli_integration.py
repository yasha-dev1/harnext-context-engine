"""CLI integration coverage for docs/evaluation-spec.md §5, §7, and §11."""

from __future__ import annotations

import json
from pathlib import Path

from harnext_eval.cli import app
from harnext_eval.corpus.synthetic import generate_synthetic_corpus
from typer.testing import CliRunner


def test_probes_command_calls_six_family_generator(tmp_path: Path) -> None:
    replay = tmp_path / "replay.jsonl"
    generate_synthetic_corpus(replay, event_count=60, entity_count=12)
    output = tmp_path / "probes.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "probes",
            "--replay",
            str(replay),
            "--out",
            str(output),
            "--per-family",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(output.read_text(encoding="utf-8").splitlines()) == 5
    assert Path(f"{output}.sha256").is_file()


def test_run_builds_store_chart_checks_and_report(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--config",
            "apps/eval/configs/baseline-minimal.yaml",
            "--corpus",
            "synthetic",
            "--experiments",
            "e2",
            "--event-count",
            "40",
            "--entity-count",
            "8",
            "--per-family",
            "1",
            "--smoke",
            "--out",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    payload = json.loads((run_dir / "e2" / "seed-1" / "results.json").read_text())
    assert payload["checks"]
    assert {row["arm"] for row in payload["tables"]["metrics"]} >= {"A3", "A4"}
    assert [row["contrast"] for row in payload["tables"]["contrasts"]] == ["A4-A3"]
    assert payload["primary"]["contrast"] == "A4-A3"
    assert (run_dir / "e2" / "seed-1" / "charts" / "e2_family_bars.png").is_file()
    assert (run_dir / "stores" / "s0" / "seed-1" / "snapshots.csv").is_file()
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "Validity checks" in html
    assert "data:image/png;base64," in html
