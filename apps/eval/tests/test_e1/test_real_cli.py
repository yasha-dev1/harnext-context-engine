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
    assert set(cfg.e1.exclude_label_functions) == {"github_trunk_ci_failure_fix_6h", "jira_fix_version_in_flight_later", "dev_vote_cancelled_recast_later"}
    corpus = _handle_for_replay(replay, "kafka")
    corpus = replace(corpus, meta={
        **corpus.meta, "e1_only": True,
        "e1_exclude_label_functions": list(cfg.e1.exclude_label_functions),
    })
    result = E1Experiment().run(cfg.engine, corpus, tmp_path / "result", 1)
    assert "lf.github_trunk_ci_failure_fix_6h.coverage" not in result.check_details
    assert "github_trunk_ci_failure_fix_6h" not in set(result.tables["label_diagnostics"]["function"])
    assert "github_trunk_ci_failure_fix_6h" in result.check_details["excluded_label_functions"]["value"]
    reg = registration(replay, cfg, None)
    assert set(reg["excluded_label_functions"]) == set(cfg.e1.exclude_label_functions)
    assert any("excluded_label_functions" in rule for rule in reg["exclusion_rules"])

    bad = replace(corpus, meta={**corpus.meta, "e1_exclude_label_functions": ["nope"]})
    import pytest
    with pytest.raises(ValueError, match="unknown labeling functions"):
        E1Experiment().run(cfg.engine, bad, tmp_path / "bad", 1)


def test_amendment_rules_declared_transition_and_vote_thread_start() -> None:
    from datetime import UTC, datetime

    from harnext_eval.e1.policies import RuleSettings, match_rule
    from harnext_eval.types import EvalEvent

    def ev(type_: str, data: dict) -> EvalEvent:
        return EvalEvent(id="x", source="jira:KAFKA", type=type_, subject="issue:KAFKA-1", mgtenant="kafka",
                         time=datetime(2024, 3, 1, tzinfo=UTC), data=data)

    critical = ev("org.apache.jira.issue.transition", {"field": "priority", "from": "Major", "to": "Critical"})
    assert match_rule(critical) == "declared_priority"
    downgraded = ev("org.apache.jira.issue.transition", {"field": "priority", "from": "Blocker", "to": "Major"})
    assert match_rule(downgraded) is None
    reply = ev("org.apache.mail.message", {"subject": "Re: [VOTE] KIP-1", "in_reply_to": "<a@b>", "body": "+1"})
    root = ev("org.apache.mail.message", {"subject": "[VOTE] KIP-1", "body": "please vote"})
    assert match_rule(reply) == "vote"  # registered run-1 behaviour
    strict = RuleSettings(vote_thread_start_only=True)
    assert match_rule(reply, strict) is None
    assert match_rule(root, strict) == "vote"


def test_fix_version_lf_abstains_without_release_state_fields() -> None:
    from datetime import UTC, datetime, timedelta

    from harnext_eval.e1.labels import ABSTAIN, DEFAULT_LABELING_FUNCTIONS, apply_labeling_functions
    from harnext_eval.types import EvalEvent

    t0 = datetime(2024, 3, 1, tzinfo=UTC)
    events = [
        EvalEvent(id="c", source="jira:KAFKA", type="org.apache.jira.issue.created", subject="issue:KAFKA-9",
                  mgtenant="kafka", time=t0, data={"priority": "Major", "summary": "s"}),
        EvalEvent(id="f", source="jira:KAFKA", type="org.apache.jira.issue.transition", subject="issue:KAFKA-9",
                  mgtenant="kafka", time=t0 + timedelta(hours=2), data={"field": "fixVersion", "from": None, "to": "3.9.0"}),
    ]
    fn = [f for f in DEFAULT_LABELING_FUNCTIONS if f.name == "jira_fix_version_in_flight_later"]
    votes = apply_labeling_functions(events, fn, observation_end=t0 + timedelta(days=30))
    assert (votes["jira_fix_version_in_flight_later"] == ABSTAIN).all()
    events[1] = events[1].model_copy(update={"data": {**events[1].data, "in_flight_release": "3.9.0"}})
    votes = apply_labeling_functions(events, fn, observation_end=t0 + timedelta(days=30))
    assert (votes["jira_fix_version_in_flight_later"] != ABSTAIN).any()
