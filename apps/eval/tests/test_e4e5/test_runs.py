"""Offline E4/E5 end-to-end tests for docs/evaluation-spec.md §7."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from harnext_eval.config import load_config
from harnext_eval.corpus import CorpusHandle
from harnext_eval.e4.run import aggregate_seed_outputs, run_e4
from harnext_eval.e4.tasks import select_fast_tasks
from harnext_eval.e5.run import run_cadences
from harnext_eval.providers.llm import FakeLLM, LLMResult
from harnext_eval.providers.tokenizer import count_tokens
from harnext_eval.stores.base import StoreHandle
from harnext_eval.types import EvalEvent, Probe, Task


class EnvelopeAwareProvider:
    """Hand-calculable provider whose V3 action differs from raw/all-files."""

    model_id = "oracle-family-a"
    tokenizer_revision = "test-v1"

    @staticmethod
    def count_tokens(text: str) -> int:
        return count_tokens(text)

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict | None = None,
        max_tokens: int,
    ) -> LLMResult:
        del system, json_schema, max_tokens
        issue = next(iter(__import__("re").findall(r"HNX-\d+", user)), "HNX-1")
        trigger = next(iter(__import__("re").findall(r"trigger-\d+", user)), "trigger")
        correct = "## overview" in user and "## all_entity_files" not in user
        payload = {
            "assignee_candidates": [],
            "reviewer_candidates": [],
            "component": "runtime" if correct else "wrong",
            "duplicate_of": None,
            "priority_change": None,
            "suspected_locations": [],
            "draft_reply": "Investigating now.",
            "cited_ids": [issue, trigger],
            "action": "route_and_reply",
        }
        text = json.dumps(payload, sort_keys=True)
        return LLMResult(text=text, json=payload, usage={"input_tokens": 100, "output_tokens": 20})


class BatchReadOracle:
    model_id = "fake-batch-reader-oracle"

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict | None = None,
        max_tokens: int,
    ) -> LLMResult:
        del system, user, json_schema, max_tokens
        return LLMResult(text="Open", json=None, usage={"input_tokens": 10, "output_tokens": 1})


class StableDifferentFamilyJudge:
    model_id = "fake-family-b-judge"

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict | None = None,
        max_tokens: int,
    ) -> LLMResult:
        del system, json_schema, max_tokens
        first, second = user.split("Response B:\n", 1)
        winner = "A" if "GOOD" in first else "B" if "GOOD" in second else "tie"
        payload = {"winner": winner}
        return LLMResult(text=winner, json=payload, usage={"input_tokens": 20, "output_tokens": 1})


def _event(
    event_id: str,
    minute: int,
    subject: str,
    *,
    event_type: str = "jira.event",
    data: dict | None = None,
) -> EvalEvent:
    return EvalEvent(
        id=event_id,
        source="jira:test",
        type=event_type,
        subject=subject,
        time=datetime(2026, 2, 1, tzinfo=UTC) + timedelta(minutes=minute),
        mgtenant="test",
        data=data or {},
    )


def _three_task_events() -> list[EvalEvent]:
    events: list[EvalEvent] = []
    for index, marker in enumerate(("Critical", "[VOTE] release", "CVE-2026-1001")):
        subject = f"issue:HNX-{index + 1}"
        start = index * 20
        trigger_data = {"issue_key": f"HNX-{index + 1}", "title": marker}
        if marker == "Critical":
            trigger_data["priority"] = "Critical"
        trigger_type = "org.apache.mail.message" if index in {1, 2} else "jira.issue.created"
        trigger_source = "dev@kafka.apache.org" if index in {1, 2} else "jira:test"
        events.extend(
            [
                _event(
                    f"trigger-{index}",
                    start,
                    subject,
                    event_type=trigger_type,
                    data=trigger_data,
                ).model_copy(update={"source": trigger_source}),
                _event(
                    f"assignee-{index}",
                    start + 1,
                    subject,
                    data={"field": "assignee", "to": f"user-{index}", "actor": "human"},
                ),
                _event(
                    f"component-{index}",
                    start + 2,
                    subject,
                    data={"field": "component", "to": "runtime", "actor": "human"},
                ),
                _event(
                    f"reply-{index}",
                    start + 3,
                    subject,
                    event_type="jira.comment",
                    data={
                        "body": f"Handling HNX-{index + 1}",
                        "is_committer": True,
                        "author": "human",
                    },
                ),
                _event(
                    f"merged-{index}",
                    start + 4,
                    subject,
                    event_type="github.pull_request.merged",
                    data={
                        "title": f"HNX-{index + 1} implementation",
                        "issue_key": f"HNX-{index + 1}",
                        "number": 100 + index,
                        "changed_files": [f"src/builder/file_{index}.py"],
                    },
                ),
            ]
        )
    return sorted(events, key=lambda event: event.time)


def test_e4_three_tasks_three_variants_three_runs(tmp_path: Path) -> None:
    events = _three_task_events()
    tasks = select_fast_tasks(events, corpus="synthetic", limit=3)
    assert len(tasks) == 3

    def fold(store: StoreHandle, batch: list[EvalEvent], lane: str) -> None:
        del lane
        for event in batch:
            base = f"entities/issue/{event.subject.split(':', 1)[1]}"
            store.write(
                f"{base}/OVERVIEW.md",
                f"assignee: prior-user\ncomponent: old-component\n[{event.id}]\n",
            )
            store.write(
                f"{base}/timeline.md",
                "\n".join(f"- history line {value} [{event.id}]" for value in range(40)),
            )
            store.write(
                f"{base}/facts.md",
                "\n".join(
                    f"- {event.subject} component: old-component fact {value}" for value in range(40)
                ),
            )
            store.write(
                f"{base}/archive.md",
                "\n".join(f"archived supporting context {value}" for value in range(400)),
            )

    store = StoreHandle("S3", "test", tmp_path / "store")
    triggers = {task.trigger_event_id for task in tasks}
    for event in events:
        if event.id in triggers:
            fold(store, [event], "fast")
            store.fold([event], "fast")
    cfg = load_config("apps/eval/configs/baseline-minimal.yaml").engine
    result = run_e4(
        tasks,
        store,
        cfg,
        tmp_path / "e4",
        provider=FakeLLM(),
        variants=("V1", "V3", "V5", "V6"),
        runs=3,
        events=events,
    )

    rows = [json.loads(line) for line in (tmp_path / "e4" / "runs.jsonl").read_text().splitlines()]
    assert len(rows) == 45
    assert {row["variant"] for row in rows} == {"V1-N20", "V1-N100", "V3", "V5", "V6"}
    assert all(len(row["prediction"]["assignee_candidates"]) <= 3 for row in rows)
    assert all(len(row["prediction"]["suspected_locations"]) <= 5 for row in rows)
    assert all(row["tool_calls"] == 3 for row in rows if row["variant"] == "V5")
    assert all(
        len({row["snapshot_sha"] for row in rows if row["task_id"] == task_id}) == 1
        for task_id in {row["task_id"] for row in rows}
    )
    assert result.metrics["checks.leakage_gate_100_pct"] is True
    assert result.metrics["gate_exclusion_count"] == 0
    assert result.primary["evidence_status"] == "plumbing-only"
    assert "sample_cells" in result.primary["invalid_reasons"]
    assert (tmp_path / "e4" / "metrics.csv").exists()
    assert (tmp_path / "e4" / "contrasts.csv").exists()
    assert (tmp_path / "e4" / "sizes.csv").exists()


def _manual_fast_task(trigger: EvalEvent, *, component: str = "runtime") -> Task:
    action_time = trigger.time + timedelta(minutes=5)
    return Task(
        task_id=f"real:fast:{trigger.id}",
        corpus="real",
        T=trigger.time,
        trigger_event_id=trigger.id,
        entity=trigger.subject,
        kind="fast",
        gold={
            "people": {"assignees": [], "reviewers": [], "decision_times": [], "event_ids": []},
            "category": {
                "components": [component],
                "duplicate_of": [],
                "priority_changes": [],
                "required_ids": [trigger.subject.split(":", 1)[1]],
                "decision_times": [action_time.isoformat()],
                "event_ids": [f"gold-{trigger.id}"],
            },
            "place": {"files": [], "modules": [], "decision_times": [], "event_ids": []},
            "text": {"replies": [], "decision_times": [], "event_ids": []},
            "_trigger_event": trigger.model_dump(mode="json"),
            "_gold_source": "derived-corpus-r",
            "_archetype": "declared_priority",
        },
        gold_coverage={"people": False, "category": True, "place": False, "text": False},
    )


def test_e4_literal_q_and_clustered_inference(tmp_path: Path) -> None:
    triggers = [
        _event(
            f"trigger-{index}",
            index * 10,
            f"issue:HNX-{index + 1}",
            event_type="jira.issue.created",
            data={"priority": "Critical", "issue_key": f"HNX-{index + 1}"},
        )
        for index in range(2)
    ]
    store = StoreHandle("S3", "inference", tmp_path / "store")
    for trigger in triggers:
        store.fold([trigger], "fast")
    cfg = load_config("apps/eval/configs/baseline-minimal.yaml").engine

    tasks = [
        task.model_copy(
            update={
                "gold": {
                    **task.gold,
                    "_join_audit": {
                        "expected_pr_ids": [f"PR-{index + 1}"],
                        "observed_pr_ids": [f"PR-{index + 1}"],
                    },
                }
            }
        )
        for index, task in enumerate(_manual_fast_task(trigger) for trigger in triggers)
    ]
    result = run_e4(
        tasks,
        store,
        cfg,
        tmp_path / "e4-inference",
        provider=EnvelopeAwareProvider(),
        variants=("V1-N20", "V3", "V6"),
        runs=1,
        events=triggers,
        expected_fast_tasks=2,
        expected_batch_tasks=0,
        seed=17,
    )

    contrasts = result.tables["contrasts"].set_index("contrast")
    assert contrasts.loc["V3-V1-N20", "mean_delta_Q"] == pytest.approx(0.5)
    assert contrasts.loc["V3-V6", "mean_delta_Q"] == pytest.approx(0.5)
    assert contrasts.loc["V3-V6", "bca_resamples"] == 10_000
    assert contrasts.loc["V3-V6", "entities"] == 2
    assert {"ci_low", "ci_high", "mcnemar_p", "practical_threshold"}.issubset(contrasts.columns)
    assert result.metrics["checks.pr_join_precision"] == 1.0
    assert result.metrics["checks.pr_join_recall"] == 1.0
    assert result.primary["valid_primary"] is False  # one-run fixture cannot publish pass^3.


def test_one_variant_leak_excludes_whole_paired_task(tmp_path: Path) -> None:
    trigger = _event(
        "trigger-0",
        0,
        "issue:HNX-1",
        event_type="jira.issue.created",
        data={"priority": "Critical", "issue_key": "HNX-1"},
    )
    prior = _event(
        "prior-secret",
        -30,
        "issue:HNX-1",
        data={"field": "component", "to": "future-secret", "issue_key": "HNX-1"},
    )
    fillers = [
        _event(f"filler-{index}", index - 25, "issue:HNX-1", data={"note": index})
        for index in range(21)
    ]
    all_events = [prior, *fillers, trigger]
    store = StoreHandle("S3", "leak", tmp_path / "store")
    store.fold(all_events, "fast")
    cfg = load_config("apps/eval/configs/baseline-minimal.yaml").engine

    result = run_e4(
        [_manual_fast_task(trigger, component="future-secret")],
        store,
        cfg,
        tmp_path / "e4-leak",
        provider=EnvelopeAwareProvider(),
        variants=("V1-N20", "V3"),
        runs=1,
        events=all_events,
        expected_fast_tasks=1,
        expected_batch_tasks=0,
    )

    gate = result.tables["gate"]
    assert set(gate["result"]) == {"PASS", "FAIL"}
    assert result.metrics["gate_exclusion_count"] == 1
    assert (tmp_path / "e4-leak" / "runs.jsonl").read_text() == ""


def test_batch_fold_uses_scratch_s3_and_e2_grading(tmp_path: Path) -> None:
    events = [
        _event(
            "event-0",
            0,
            "issue:HNX-1",
            data={"field": "status", "to": "Open", "issue_key": "HNX-1"},
        ),
        _event("event-1", 1, "issue:HNX-1", data={"note": "window close"}),
    ]
    store = StoreHandle("S3", "batch", tmp_path / "store")
    snapshot = store.fold(events, "batch")
    probe = Probe(
        probe_id="batch-status",
        family="extraction",
        entity="issue:HNX-1",
        T=events[-1].time,
        question="What is the current status of issue:HNX-1?",
        gold="Open",
        gold_type="exact",
        source_event_ids=["event-0"],
    )
    task = Task(
        task_id="synthetic:batch:event-1",
        corpus="synthetic",
        T=events[-1].time,
        trigger_event_id="event-1",
        entity="issue:HNX-1",
        kind="batch",
        gold={"probes": [probe.model_dump(mode="json")]},
        gold_coverage={"people": False, "category": False, "place": False, "text": False},
    )
    before = {path: store.read(snapshot, path) for path in store.list_files(snapshot)}
    cfg = load_config("apps/eval/configs/baseline-minimal.yaml").engine

    run_e4(
        [task],
        store,
        cfg,
        tmp_path / "e4-batch",
        provider=BatchReadOracle(),
        variants=("V3", "V6"),
        runs=1,
        events=events,
        expected_fast_tasks=0,
        expected_batch_tasks=1,
    )

    rows = [json.loads(line) for line in (tmp_path / "e4-batch" / "runs.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["batch_e2_acc"] == 1.0 for row in rows)
    assert all(row["batch_result_sha"] != snapshot.sha for row in rows)
    assert all(row["batch_delta_files"] for row in rows)
    assert before == {path: store.read(snapshot, path) for path in store.list_files(snapshot)}


def test_e4_refuses_non_s3_store(tmp_path: Path) -> None:
    cfg = load_config("apps/eval/configs/baseline-minimal.yaml").engine
    store = StoreHandle("S0", "wrong-layout", tmp_path / "store")
    with pytest.raises(ValueError, match="requires a fixed S3"):
        run_e4([], store, cfg, tmp_path / "e4", provider=FakeLLM())


def test_seed_spread_aggregation_marks_partial_runs_supported(tmp_path: Path) -> None:
    root = tmp_path / "e4"
    for seed, effect in ((1, 0.1), (2, 0.3)):
        directory = root / f"seed-{seed}"
        directory.mkdir(parents=True)
        pd.DataFrame([{"contrast": "V3-V6", "mean_delta_Q": effect}]).to_csv(
            directory / "contrasts.csv", index=False
        )

    partial = aggregate_seed_outputs(root)
    assert partial.iloc[0]["status"] == "supported-not-run"
    assert partial.iloc[0]["seed_spread_sd"] == pytest.approx(2**0.5 / 10)

    third = root / "seed-3"
    third.mkdir()
    pd.DataFrame([{"contrast": "V3-V6", "mean_delta_Q": 0.2}]).to_csv(
        third / "contrasts.csv", index=False
    )
    complete = aggregate_seed_outputs(root)
    assert complete.iloc[0]["status"] == "complete"
    assert complete.iloc[0]["seeds"] == 3


def test_judge_requires_200_dual_human_labels_and_swaps_order(tmp_path: Path) -> None:
    trigger = _event(
        "trigger-0",
        0,
        "issue:HNX-1",
        event_type="jira.issue.created",
        data={"priority": "Critical", "issue_key": "HNX-1"},
    )
    store = StoreHandle("S3", "judge", tmp_path / "store")
    store.fold([trigger], "fast")
    calibration = [
        {
            "candidate": "GOOD response" if index % 2 == 0 else "BAD response",
            "baseline": "BAD response" if index % 2 == 0 else "GOOD response",
            "human_a": index % 2 == 0,
            "human_b": index % 2 == 0,
        }
        for index in range(200)
    ]
    cfg = load_config("apps/eval/configs/baseline-minimal.yaml").engine

    result = run_e4(
        [_manual_fast_task(trigger)],
        store,
        cfg,
        tmp_path / "e4-judge",
        provider=EnvelopeAwareProvider(),
        variants=("V3",),
        runs=1,
        events=[trigger],
        expected_fast_tasks=1,
        expected_batch_tasks=0,
        judge_provider=StableDifferentFamilyJudge(),
        judge_calibration=calibration,
        judge_model_family="family-b",
    )

    judge = result.tables["judge_kappa"].iloc[0]
    assert judge["n_calibration"] == 200
    assert judge["judge_kappa"] == 1.0
    assert bool(judge["used"]) is True
    assert result.metrics["checks.position_swapped"] is True


def test_e5_two_cadences_produce_cost_and_freshness(tmp_path: Path) -> None:
    events = [
        EvalEvent(
            id=f"event-{index}",
            source="jira:test",
            type="jira.event",
            subject="issue:HNX-1",
            time=datetime(2026, 2, 1, tzinfo=UTC) + timedelta(seconds=second),
            mgtenant="test",
            data={"field": "status", "to": f"state-{index}", "issue_key": "HNX-1"},
        )
        for index, second in enumerate((0, 1, 2, 20))
    ]
    replay = tmp_path / "replay.jsonl"
    replay.write_text("".join(event.model_dump_json() + "\n" for event in events), encoding="utf-8")
    probes = [
        Probe(
            probe_id="p-cutoff",
            family="extraction",
            entity="issue:HNX-1",
            T=events[2].time,
            question="What is the current status of issue:HNX-1 at the snapshot time?",
            gold="stale-on-purpose",
            gold_type="exact",
            source_event_ids=["event-2"],
        )
    ]
    corpus = CorpusHandle(
        name="synthetic",
        replay_path=replay,
        probes_path=None,
        tasks_path=None,
        window="test",
        meta={
            "prices": {"input_per_million": 2.0, "output_per_million": 8.0},
            "injected_situations": [
                {"event_id": "event-0", "onset": events[0].time.isoformat(), "cost_weight": 3}
            ],
            "bca_resamples": 200,
            "smoke": True,
        },
    )
    cfg = load_config("apps/eval/configs/baseline-minimal.yaml").engine
    result = run_cadences(
        cfg,
        corpus,
        tmp_path / "e5",
        seed=1,
        cadences=("W1", "W5"),
        probes=probes,
    )

    costs = pd.read_csv(tmp_path / "e5" / "cost.csv")
    freshness = pd.read_csv(tmp_path / "e5" / "freshness.csv")
    assert set(costs["cadence"]) == {"W1", "W5"}
    assert (costs["cost_1k"] > 0).all()  # deterministic fake-harness projection is billed
    assert (costs["runs_1k"] > 0).all()
    assert costs.set_index("cadence").loc["W1", "classifier_folds"] == 4
    assert costs.set_index("cadence").loc["W5", "classifier_folds"] == 2
    w1 = freshness[freshness["cadence"] == "W1"]
    w5 = freshness[freshness["cadence"] == "W5"]
    assert w1["freshness_s"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert w5["freshness_s"].tolist() == [9.5, 8.5, 7.5, 7.5]
    assert w5["urgent"].tolist() == [True, False, False, False]
    assert set(w5["urgency_provenance"]) == {"constructed-injected"}
    assert result.metrics["checks.cost_from_usage"] == 1
    assert result.metrics["checks.builder_run_count"] == 1
    assert result.primary["evidence_status"] == "plumbing-only"
    assert result.primary["valid_primary"] is False
    assert "claim_profile" in result.primary["invalid_reasons"]
    assert "probe_families" in result.primary["invalid_reasons"]
    assert (tmp_path / "e5" / "pareto.csv").exists()
    assert (tmp_path / "e5" / "pareto.png").exists()
    assert (tmp_path / "e5" / "gate.csv").exists()
    assert {"input_tokens", "output_tokens", "reader_latency_s"}.issubset(costs.columns)

    with pytest.raises(FileExistsError, match="not empty"):
        run_cadences(
            cfg,
            corpus,
            tmp_path / "e5",
            seed=1,
            cadences=("W1", "W5"),
            probes=probes,
        )
