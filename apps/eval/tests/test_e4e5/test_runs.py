"""Offline E4/E5 end-to-end tests for docs/evaluation-spec.md §7."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from harnext_eval.config import load_config
from harnext_eval.corpus import CorpusHandle
from harnext_eval.e4.run import run_e4
from harnext_eval.e4.tasks import select_fast_tasks
from harnext_eval.e5.run import run_cadences
from harnext_eval.providers.llm import FakeLLM
from harnext_eval.stores.base import StoreHandle, register_layout
from harnext_eval.types import EvalEvent, Probe


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
        events.extend(
            [
                _event(
                    f"trigger-{index}",
                    start,
                    subject,
                    event_type="jira.issue.created",
                    data=trigger_data,
                ),
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
                    data={"field": "component", "to": "builder", "actor": "human"},
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
                f"assignee: user-{event.subject[-1]}\ncomponent: builder\n[{event.id}]\n",
            )
            store.write(
                f"{base}/timeline.md",
                "\n".join(f"- history line {value} [{event.id}]" for value in range(40)),
            )
            store.write(
                f"{base}/facts.md",
                "\n".join(
                    f"- {event.subject} component: builder fact {value}" for value in range(40)
                ),
            )
            store.write(
                f"{base}/archive.md",
                "\n".join(f"archived supporting context {value}" for value in range(400)),
            )

    register_layout("E4RUN", fold)
    store = StoreHandle("E4RUN", "test", tmp_path / "store")
    triggers = {task.trigger_event_id for task in tasks}
    for event in events:
        if event.id in triggers:
            store.fold([event], "fast")
    cfg = load_config("apps/eval/configs/baseline-minimal.yaml").engine
    result = run_e4(
        tasks,
        store,
        cfg,
        tmp_path / "e4",
        provider=FakeLLM(),
        variants=("V1", "V3", "V6"),
        runs=3,
        events=events,
    )

    rows = [json.loads(line) for line in (tmp_path / "e4" / "runs.jsonl").read_text().splitlines()]
    assert len(rows) == 27
    assert {row["variant"] for row in rows} == {"V1", "V3", "V6"}
    assert all(len(row["prediction"]["assignee_candidates"]) <= 3 for row in rows)
    assert all(len(row["prediction"]["suspected_locations"]) <= 5 for row in rows)
    assert result.metrics["checks.leakage_gate_failed"] == 0
    assert (tmp_path / "e4" / "metrics.csv").exists()
    assert (tmp_path / "e4" / "contrasts.csv").exists()
    assert (tmp_path / "e4" / "sizes.csv").exists()


def test_e5_two_cadences_produce_cost_and_freshness(tmp_path: Path) -> None:
    events = [
        _event(
            f"event-{index}",
            index,
            f"issue:HNX-{index % 3}",
            data={"field": "status", "to": f"state-{index}", "issue_key": f"HNX-{index % 3}"},
        )
        for index in range(30)
    ]
    replay = tmp_path / "replay.jsonl"
    replay.write_text("".join(event.model_dump_json() + "\n" for event in events), encoding="utf-8")
    probes = [
        Probe(
            probe_id="p-final",
            family="extraction",
            entity="issue:HNX-2",
            T=events[-1].time,
            question="What state was observed?",
            gold="state-29",
            gold_type="exact",
            source_event_ids=["event-29"],
        )
    ]
    corpus = CorpusHandle(
        name="synthetic",
        replay_path=replay,
        probes_path=None,
        tasks_path=None,
        window="test",
        meta={"prices": {"input_per_million": 2.0, "output_per_million": 8.0}},
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
    assert (costs["cost_1k"] > 0).all()
    assert (costs["runs_1k"] > 0).all()
    assert not freshness.empty
    assert (freshness["freshness_s"] >= 0).all()
    assert result.metrics["checks.cost_from_usage"] == 1
    assert result.metrics["checks.builder_run_count"] == 1
    assert (tmp_path / "e5" / "pareto.csv").exists()
