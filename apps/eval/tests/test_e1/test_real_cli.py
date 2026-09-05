"""Real-corpus E1 CLI checks for docs/evaluation-spec.md §6/§7 E1."""

import json
from pathlib import Path

from harnext_eval.cli import app
from harnext_eval.corpus.synthetic import generate_synthetic_corpus
from typer.testing import CliRunner


def test_e1_only_skips_stores_and_probes_and_reports_harm_na(tmp_path, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("E1-only must not build stores or probes")

    monkeypatch.setattr("harnext_eval.cli._generate_probes", forbidden)
    monkeypatch.setattr("harnext_eval.cli._build_run_stores", forbidden)
    corpus = generate_synthetic_corpus(tmp_path / "replay.jsonl", seed=1, event_count=50, days=140, entity_count=4)
    result = CliRunner().invoke(app, [
        "run", "--config", "apps/eval/configs/e1-kafka.yaml", "--replay", str(corpus.replay_path),
        "--experiments", "e1", "--out", str(tmp_path / "out"),
        "--window", "2026-01-01", "2026-07-01",
    ])
    assert result.exit_code == 0, result.output + str(result.exception)
    files = list((tmp_path / "out").glob("*/e1/seed-1/results.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    check = payload["checks"]["harm_paired_coverage"]
    assert check["status"] == "not_applicable"
    assert check["required"] is False
    assert check["reason"] == "no real action provider / no S3 store in this profile"
    assert not list((tmp_path / "out").glob("*/stores"))
    assert not list((tmp_path / "out").glob("*/probes"))
    assert Path(files[0]).stat().st_size > 0


def test_missing_ci_is_zero_coverage_and_invalid_evidence(tmp_path) -> None:
    from dataclasses import replace

    from harnext_eval.cli import _handle_for_replay
    from harnext_eval.config import load_config
    from harnext_eval.corpus.build_replay import write_replay
    from harnext_eval.corpus.synthetic import generate_synthetic_events
    from harnext_eval.e1.run import E1Experiment

    events = generate_synthetic_events(1, event_count=30, days=140, entity_count=3)
    # Remove construction gold and model an API stream with no check/status events.
    events = [event.model_copy(update={
        "source": "github:apache/kafka", "type": "com.github.pull_request.opened",
        "data": {"title": "CI check failure discussed in PR text", "number": index},
    }) for index, event in enumerate(events)]
    replay = tmp_path / "real.jsonl"
    write_replay(events, replay)
    corpus = _handle_for_replay(replay, "kafka")
    corpus = replace(corpus, meta={**corpus.meta, "e1_only": True})
    result = E1Experiment().run(load_config("apps/eval/configs/e1-kafka.yaml").engine,
                                corpus, tmp_path / "result", 1)
    gate = result.check_details["lf.github_trunk_ci_failure_fix_6h.coverage"]
    assert gate["passed"] is False
    assert gate["required"] is True
    assert gate["value"] == 0.0
    assert result.primary["evidence_status"] == "non-evidentiary"
    assert "declared_outcome_agreement" in result.metrics


def test_corpus_command_avoids_hourly_world_states(tmp_path, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("replay-only generation must not materialise world states")

    monkeypatch.setattr("harnext_eval.cli.generate_synthetic_corpus", forbidden)
    path = tmp_path / "synthetic.jsonl"
    result = CliRunner().invoke(app, ["corpus", "--output", str(path),
                                    "--event-count", "40", "--days", "1600"])
    assert result.exit_code == 0, result.output + str(result.exception)
    assert len(path.read_text().splitlines()) == 40
    assert path.with_suffix(".jsonl.sha256").is_file()


def test_prereg_hash_window_and_read_only_chronology(tmp_path, monkeypatch) -> None:
    import subprocess
    from datetime import UTC, datetime

    from harnext_eval.config import load_config
    from harnext_eval.e1.prereg import parse_window, verify_prereg, write_prereg

    replay = tmp_path / "replay.jsonl"
    replay.write_text("{}\n")
    config = load_config("apps/eval/configs/e1-kafka.yaml")
    window = parse_window(("2022-01-01", "2026-07-01"))
    prereg = write_prereg(replay, config, tmp_path / "PREREG.md", window)
    content = prereg.read_text()
    calls = []

    def fake_git(*args, cwd):
        calls.append(args)
        output = {"rev-parse": str(tmp_path), "log": "abc 100", "show": content,
                  "merge-base": ""}[args[0]]
        return subprocess.CompletedProcess(args, 0, output, "")

    monkeypatch.setattr("harnext_eval.e1.prereg._git", fake_git)
    verified = verify_prereg(prereg, replay, config, window, datetime.now(UTC))
    assert verified["prereg_predates_evaluation"] is True
    assert verified["prereg_hash"]
    wrong = verify_prereg(prereg, replay, config, None, datetime.now(UTC))
    assert wrong["prereg_predates_evaluation"] is False
    assert "window" in wrong["prereg_reason"]
    assert all(call[0] in {"rev-parse", "log", "show", "merge-base"} for call in calls)
    replay.write_text("changed\n")
    assert not verify_prereg(prereg, replay, config, window, datetime.now(UTC))["prereg_predates_evaluation"]


def test_excluded_label_function_is_dropped_and_registered(tmp_path) -> None:
    from dataclasses import replace

    from harnext_eval.cli import _handle_for_replay
    from harnext_eval.config import load_config
    from harnext_eval.corpus.build_replay import write_replay
    from harnext_eval.corpus.synthetic import generate_synthetic_events
    from harnext_eval.e1.prereg import registration
    from harnext_eval.e1.run import E1Experiment

    events = generate_synthetic_events(1, event_count=30, days=140, entity_count=3)
    events = [event.model_copy(update={
        "source": "github:apache/kafka", "type": "com.github.pull_request.opened",
        "data": {"title": "PR", "number": index},
    }) for index, event in enumerate(events)]
    replay = tmp_path / "real.jsonl"
    write_replay(events, replay)
    cfg = load_config("apps/eval/configs/e1-kafka.yaml")
    assert cfg.e1.exclude_label_functions == ["github_trunk_ci_failure_fix_6h"]
    corpus = _handle_for_replay(replay, "kafka")
    corpus = replace(corpus, meta={
        **corpus.meta, "e1_only": True,
        "e1_exclude_label_functions": list(cfg.e1.exclude_label_functions),
    })
    result = E1Experiment().run(cfg.engine, corpus, tmp_path / "result", 1)
    assert "lf.github_trunk_ci_failure_fix_6h.coverage" not in result.check_details
    assert "github_trunk_ci_failure_fix_6h" not in set(result.tables["label_diagnostics"]["function"])
    assert result.check_details["excluded_label_functions"]["value"] == ["github_trunk_ci_failure_fix_6h"]
    reg = registration(replay, cfg, None)
    assert reg["excluded_label_functions"] == ["github_trunk_ci_failure_fix_6h"]
    assert any("excluded_label_functions" in rule for rule in reg["exclusion_rules"])

    bad = replace(corpus, meta={**corpus.meta, "e1_exclude_label_functions": ["nope"]})
    import pytest
    with pytest.raises(ValueError, match="unknown labeling functions"):
        E1Experiment().run(cfg.engine, bad, tmp_path / "bad", 1)
