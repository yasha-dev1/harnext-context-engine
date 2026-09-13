"""Deterministic merged-E2 infrastructure gates, evaluation-spec §4/§7."""

import json
from pathlib import Path

import pytest
from harnext_eval.e2.context import (
    Action,
    FixtureReader,
    Profile,
    SnapshotTools,
    grade,
    read_question,
    run,
    smoke_dataset,
    snapshot_files,
    windows,
)
from harnext_eval.stores.base import StoreHandle


def action(kind="answer", **kwargs):
    return Action.model_validate(dict(action=kind, path="", query="", offset=0,
        value=None, unknown=kind == "answer", evidence_ids=[]) | kwargs)


def test_window_order_boundaries_count_and_duplicate_guard():
    events, _ = smoke_dataset()
    batches = windows(list(reversed(events)), Profile())
    assert [[event.id for event in batch] for batch in batches] == [
        ["ev-001", "ev-002", "ev-003"], ["ev-004", "ev-005", "ev-006"]]
    assert len(windows(events, Profile(window_max_events=2))) == 4
    with pytest.raises(ValueError, match="duplicate"):
        windows(events + events[:1], Profile())
    with pytest.raises(ValueError, match="byte cap"):
        windows(events, Profile(window_max_bytes=1))
    with pytest.raises(ValueError, match="tenant"):
        windows([events[0], events[1].model_copy(update={"mgtenant": "other"})], Profile())


def test_retrieval_budget_charges_search_paths_errors_and_repeats():
    tools = SnapshotTools({"INDEX.md": "é" * 100}, budget=40, response_cap=20)
    assert tools.invoke(action("read", path="../../gold.json")) == "ERROR: path unavaila"
    assert tools.used == 20
    assert tools.invoke(action("read", path="INDEX.md")) == "é" * 10
    assert tools.used == 40
    assert tools.invoke(action("search", query="é")) == ""
    assert tools.used == 40
    assert all(row["truncated"] for row in tools.log)


def test_snapshot_export_excludes_future_git_and_evaluator_metadata(tmp_path):
    events, probes = smoke_dataset()
    store = StoreHandle("S1", "test", tmp_path)
    for batch in windows(events, Profile()):
        store.fold(batch, "batch")
    early = store.snapshot(probes[0].T)
    files = snapshot_files(store, early)
    assert not any(name.startswith(".git") or name.startswith("_meta/input")
                   or "delivered" in name or name == "CLAUDE.md" for name in files)
    payload = json.dumps(files)
    assert "Mira" in payload
    assert "Noah" not in payload
    assert "ev-004" not in payload
    assert set(store.delivered_event_ids(early)) == {"ev-001", "ev-002", "ev-003"}


def test_deterministic_grading_requires_typed_values_and_exposed_evidence():
    _, probes = smoke_dataset()
    probe = probes[0]
    result = {"answer": action(value="Mira", unknown=False, evidence_ids=["ev-001"]).model_dump(),
              "retrieval": [{"result": "[ev-001] assignee=Mira"}]}
    assert grade(probe, result, {"ev-001"}) == {
        "value_correct": True, "evidence_correct": True, "schema_valid": True}
    result["retrieval"] = []
    assert not grade(probe, result, {"ev-001"})["evidence_correct"]
    result["answer"]["value"] = "Mira is the assignee."
    assert not grade(probe, result, {"ev-001"})["value_correct"]
    result["answer"]["value"] = ["PR-71", "PR-72", "PR-71"]
    assert not grade(probes[6], result, {"ev-002", "ev-005"})["value_correct"]
    result["answer"] = action().model_dump()
    assert grade(probes[-1], result, set())["value_correct"]
    assert not grade(probe, result, set())["value_correct"]


def test_reader_does_not_receive_gold_or_entire_store_implicitly():
    _, probes = smoke_dataset()
    result = read_question(FixtureReader(), probes[0].question, probes[0].T,
                           {"INDEX.md": "ev-001 public data"}, Profile())
    assert result["status"] == "completed"
    assert result["tool_calls"] == 1
    assert "Mira" not in json.dumps(result["history"])


def test_final_call_reserves_an_answer_instead_of_more_search():
    from harnext_eval.providers.llm import LLMResult

    class GreedyReader:
        def complete(self, system, user, *, json_schema=None, max_tokens):
            final = json_schema["properties"]["action"]["enum"] == ["answer"]
            response = action("answer" if final else "search", query="missing")
            return LLMResult(text=response.model_dump_json(), json=response.model_dump(), usage={})

    _, probes = smoke_dataset()
    result = read_question(GreedyReader(), probes[-1].question, probes[-1].T,
                           {"INDEX.md": "no such fact"}, Profile(reader_max_calls=3))
    assert result["status"] == "completed" and result["answer"]["unknown"]
    assert result["provider_calls"] == 3 and result["tool_calls"] == 2


def test_offline_end_to_end_and_live_guard(tmp_path: Path):
    with pytest.raises(ValueError, match="--live"):
        run(Profile(reader_provider="codex"), tmp_path / "denied")
    result = run(Profile(), tmp_path / "run")
    assert result["infrastructure_passed"]
    assert result["answer_count"] == 30
    assert len(result["builds"]) == 6
    assert result["macro_accuracy"] == {"S0": 0.2, "S1": 0.2, "S3": 0.2}
    assert result["thesis_evidence"] is False
    assert result["total_cost_usd"] is None
    with pytest.raises(ValueError, match="overwrite"):
        run(Profile(), tmp_path / "run")
    reused = run(Profile(), tmp_path / "reread", reuse_builds=tmp_path / "run")
    assert reused["new_builder_calls"] == 0
    assert [b["snapshot"] for b in reused["builds"]] == [b["snapshot"] for b in result["builds"]]
    assert all(b["reused"] and b["latency_s"] is None for b in reused["builds"])


def test_stale_provider_identity_cannot_disable_offline_guard(monkeypatch):
    from harnext_eval.config import load_config
    from harnext_eval.providers import factory

    config = load_config(Path(__file__).parents[2] / "configs" / "baseline-minimal.yaml")
    monkeypatch.setitem(factory._OFFLINE_BY_ENGINE, id(config.engine), False)
    monkeypatch.setitem(factory._SUMMARIES, id(config), {"offline_enforced": False})
    monkeypatch.delitem(factory._OWNERS, id(config), raising=False)
    monkeypatch.delitem(factory._OWNERS, id(config.engine), raising=False)
    assert factory._offline(config.engine) is True
    factory.make_llm(config)
    assert factory.provider_summary(config)["offline_enforced"] is True


@pytest.mark.parametrize("stop_reason,changed", [("timeout", True), ("completed", False)])
def test_failed_codex_fold_rolls_back_and_does_not_publish(tmp_path, monkeypatch, stop_reason, changed):
    from harnext_builder.agentfs.backend import RunResult
    from harnext_builder.harness.base import ConversationTranscript
    from harnext_eval.stores.layouts import configure_store

    events, _ = smoke_dataset()
    store = StoreHandle("S3", "test", tmp_path)
    configure_store(store, harness="codex", model="gpt-5.6-luna", reasoning_effort="medium")

    def failed_build(org_id, command, env, timeout_s):
        if changed:
            store.write("entities/partial.md", "uncommitted partial answer")
        transcript = ConversationTranscript(harness="codex", stop_reason=stop_reason,
            files_changed=["A entities/partial.md"] if changed else [])
        Path(env["RESULT_PATH"]).write_text(transcript.model_dump_json())
        return RunResult(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(store.backend, "run_build", failed_build)
    with pytest.raises(RuntimeError):
        store.fold(events[:1], "batch")
    assert not (store.worktree / "entities/partial.md").exists()
    assert not (store.worktree / "_meta/input.json").exists()
    assert not store.snapshots_csv.exists()
    assert not store.delivered_jsonl.exists()
