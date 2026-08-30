"""Synthetic task/envelope tests for docs/evaluation-spec.md §7 E4."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from harnext_eval.agents.envelope import STATIC_PREFIX, build, execute_tools
from harnext_eval.config import WindowConfig
from harnext_eval.e4.tasks import (
    build_batch_tasks,
    build_constructed_tasks,
    select_fast_tasks,
)
from harnext_eval.providers.tokenizer import count_tokens
from harnext_eval.stores.base import StoreHandle, register_layout
from harnext_eval.types import EvalEvent, Probe


def _event(
    event_id: str,
    minute: int,
    *,
    event_type: str = "jira.event",
    subject: str = "issue:HNX-1",
    data: dict | None = None,
) -> EvalEvent:
    return EvalEvent(
        id=event_id,
        source="jira:test",
        type=event_type,
        subject=subject,
        time=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=minute),
        mgtenant="test",
        data=data or {},
    )


def _gold_events() -> list[EvalEvent]:
    return [
        _event(
            "trigger",
            0,
            event_type="jira.issue.created",
            data={"priority": "Critical", "issue_key": "HNX-1"},
        ),
        _event(
            "bot-assignee",
            1,
            data={"field": "assignee", "to": "dependabot[bot]", "actor": "dependabot[bot]"},
        ),
        _event("assignee", 2, data={"field": "assignee", "to": "alice", "actor": "maintainer"}),
        _event("component", 3, data={"field": "component", "to": "builder", "actor": "maintainer"}),
        _event("priority", 4, data={"field": "priority", "to": "Blocker", "actor": "maintainer"}),
        _event(
            "duplicate", 5, data={"field": "duplicate_of", "to": "HNX-9", "actor": "maintainer"}
        ),
        _event(
            "reply",
            6,
            event_type="jira.issue.comment",
            data={"body": "I will investigate HNX-1.", "author": "committer", "is_committer": True},
        ),
        _event(
            "review",
            7,
            event_type="github.pull_request.review",
            data={"issue_key": "HNX-1", "reviewer": "bob"},
        ),
        _event(
            "merged",
            8,
            event_type="github.pull_request.merged",
            data={
                "title": "HNX-1 implement the fix",
                "issue_key": "HNX-1",
                "number": 42,
                "changed_files": ["src/builder/engine.py", "tests/builder/test_engine.py"],
            },
        ),
    ]


def test_task_selection_derives_four_gold_groups_and_excludes_bots() -> None:
    tasks = select_fast_tasks(_gold_events(), corpus="synthetic")

    assert len(tasks) == 1  # A later priority transition is not a new-issue trigger.
    task = tasks[0]
    assert all(task.gold_coverage.values())
    assert task.gold["people"]["assignees"] == ["alice"]
    assert task.gold["people"]["reviewers"] == ["bob"]
    assert "dependabot[bot]" not in task.gold["people"]["assignees"]
    assert task.gold["category"]["components"] == ["builder"]
    assert task.gold["category"]["duplicate_of"] == ["HNX-9"]
    assert task.gold["place"]["files"] == [
        "src/builder/engine.py",
        "tests/builder/test_engine.py",
    ]
    assert task.gold["text"]["replies"] == ["I will investigate HNX-1."]
    assert all(
        datetime.fromisoformat(value) > task.T
        for group in ("people", "category", "place", "text")
        for value in task.gold[group]["decision_times"]
    )


def test_batch_tasks_pair_only_probes_at_window_close() -> None:
    events = [_event("one", 0), _event("two", 1)]
    close = events[-1].time
    probe = Probe(
        probe_id="p1",
        family="extraction",
        entity="issue:HNX-1",
        T=close,
        question="What is the status?",
        gold="Open",
        gold_type="exact",
    )
    tasks = build_batch_tasks(
        events,
        [probe],
        corpus="synthetic",
        window=WindowConfig(gap_s=30, max_events=2, max_age_s=120),
    )

    assert len(tasks) == 1
    assert tasks[0].kind == "batch"
    assert tasks[0].T == close
    assert tasks[0].gold["probes"][0]["probe_id"] == "p1"


def test_envelopes_have_exact_sections_and_v6_is_larger(tmp_path: Path) -> None:
    def fold(store: StoreHandle, events: list[EvalEvent], lane: str) -> None:
        del lane
        event = events[-1]
        base = "entities/issue/HNX-1"
        store.write(f"{base}/OVERVIEW.md", f"assignee: alice\ncomponent: builder\n[{event.id}]\n")
        store.write(
            f"{base}/timeline.md",
            "\n".join(f"- line {index} [history-{index}]" for index in range(60)),
        )
        store.write(
            f"{base}/facts.md",
            "\n".join(f"- HNX-1 fact {index} component: builder" for index in range(60)),
        )
        store.write(
            f"{base}/notes.md",
            "\n".join(
                f"long archived note number {index} with supporting context" for index in range(500)
            ),
        )
        store.write(f"{base}/actions.md", "\n".join(f"action {index}" for index in range(20)))
        store.write(f"{base}/superseded.md", "old priority: Major\nold assignee: carol\n")
        store.write(
            "entities/issue/HNX-10/superseded.md",
            "SECRET FROM A DIFFERENT ENTITY\n",
        )

    register_layout("E4TEST", fold)
    store = StoreHandle("E4TEST", "test", tmp_path / "store")
    events = _gold_events()
    snapshot = store.fold([events[0]], "fast")
    task = select_fast_tasks(events, corpus="synthetic")[0]

    envelopes = {
        variant: build(task, snapshot, variant, {"store_handle": store, "events": events})
        for variant in ("V0", "V1-N20", "V1-N100", "V2", "V3", "V4", "V5", "V6", "V7", "V8")
    }
    assert list(envelopes["V0"].sections) == ["triggering_event"]
    assert "raw_entity_events_last_20" in envelopes["V1-N20"].sections
    assert "raw_entity_events_last_100" in envelopes["V1-N100"].sections
    assert list(envelopes["V2"].sections) == ["triggering_event", "overview"]
    assert list(envelopes["V3"].sections) == [
        "triggering_event",
        "overview",
        "timeline_tail_20",
        "matched_facts_top_10",
    ]
    assert "action_log_last_10" in envelopes["V4"].sections
    assert envelopes["V5"].tools == ["read_state", "search_facts", "recent_events"]
    assert "all_entity_files" in envelopes["V6"].sections
    assert "superseded_bodies" in envelopes["V7"].sections
    assert "SECRET FROM A DIFFERENT ENTITY" not in envelopes["V7"].text
    assert list(envelopes["V8"].sections)[2] == "overview"
    assert envelopes["V6"].token_count > envelopes["V3"].token_count
    assert all(envelope.token_count == count_tokens(envelope.text) for envelope in envelopes.values())
    assert {envelope.prefix for envelope in envelopes.values()} == {STATIC_PREFIX}
    assert len(envelopes["V3"].sections["timeline_tail_20"].splitlines()) == 20
    assert len(envelopes["V3"].sections["matched_facts_top_10"].splitlines()) == 10
    used = execute_tools(
        envelopes["V5"],
        queries={"read_state": "HNX-1", "search_facts": "component", "recent_events": "20"},
        budget_tokens=12_000,
    )
    assert used.observed_tool_calls == 3
    assert all(name in used.text for name in ("read_state", "search_facts", "recent_events"))
    assert used.token_count <= 12_000


def test_constructed_s_gold_comes_from_injected_meta_and_world_state() -> None:
    trigger = _event(
        "situation-1",
        0,
        event_type="jira.issue.created",
        data={"priority": "Critical", "issue_key": "HNX-1"},
    )
    action_time = trigger.time + timedelta(minutes=5)
    meta = {
        "injected_situations": [
            {
                "event_id": trigger.id,
                "onset": trigger.time.isoformat(),
                "entity": trigger.subject,
                "archetype": "incident",
                "action_time": action_time.isoformat(),
                "scripted_handling": {
                    "owner": "oncall-alice",
                    "required_ids": ["fact-owner-1", "fact-incident-1"],
                    "action": "page_owner",
                },
            }
        ],
        "world_state": {trigger.subject: {"owner": "wrong-fallback"}},
    }

    tasks = build_constructed_tasks([trigger], corpus="synthetic", corpus_meta=meta)

    assert tasks[0].gold["people"]["assignees"] == ["oncall-alice"]
    assert tasks[0].gold["category"]["required_ids"] == [
        "fact-owner-1",
        "fact-incident-1",
    ]
    assert tasks[0].gold["_gold_source"] == "constructed-corpus-s-meta"


def test_title_only_pr_join_horizons_and_committer_identity() -> None:
    trigger = _event(
        "trigger-title",
        0,
        event_type="jira.issue.created",
        data={"priority": "Critical", "issue_key": "HNX-1"},
    )
    events = [
        trigger,
        _event(
            "assignee-title",
            1,
            data={"field": "assignee", "to": "alice", "actor": "maintainer"},
        ),
        _event(
            "ordinary-reply",
            2,
            event_type="jira.issue.comment",
            data={"body": "ordinary user", "author": "visitor"},
        ),
        _event(
            "docs-pr",
            3,
            event_type="github.pull_request.merged",
            data={
                "title": "HNX-1 document behavior",
                "number": 77,
                "changed_files": ["docs/behavior.md"],
            },
        ),
        _event(
            "body-only-pr",
            4,
            event_type="github.pull_request.merged",
            data={
                "title": "unrelated change",
                "body": "mentions HNX-1",
                "number": 78,
                "changed_files": ["src/wrong.py"],
            },
        ),
        _event(
            "committer-reply",
            5,
            event_type="jira.issue.comment",
            data={"body": "authoritative reply", "author": "alice"},
        ),
    ]

    task = select_fast_tasks(events, corpus="real", committer_accounts=["alice"])[0]

    assert task.gold["text"]["replies"] == ["authoritative reply"]
    assert task.gold["place"]["files"] == ["docs/behavior.md"]
    assert "src/wrong.py" not in task.gold["place"]["files"]
    assert "PR-77" in task.gold["category"]["required_ids"]


def test_gold_horizon_boundaries_multi_pr_union_and_format_exclusion() -> None:
    trigger = _event(
        "trigger-horizons",
        0,
        event_type="jira.issue.created",
        data={"priority": "Critical", "issue_key": "HNX-1"},
    )
    events = [
        trigger,
        _event(
            "assignee-at-24h",
            24 * 60,
            data={"field": "assignee", "to": "at-boundary", "actor": "human"},
        ),
        _event(
            "assignee-after-24h",
            24 * 60 + 1,
            data={"field": "assignee", "to": "too-late", "actor": "human"},
        ),
        _event(
            "duplicate-at-7d",
            7 * 24 * 60,
            data={"field": "duplicate_of", "to": "HNX-7", "actor": "human"},
        ),
        _event(
            "duplicate-after-7d",
            7 * 24 * 60 + 1,
            data={"field": "duplicate_of", "to": "HNX-8", "actor": "human"},
        ),
        _event(
            "pr-one",
            10 * 24 * 60,
            event_type="github.pull_request.merged",
            data={"title": "HNX-1 first fix", "number": 1, "changed_files": ["src/one.py"]},
        ),
        _event(
            "pr-format",
            11 * 24 * 60,
            event_type="github.pull_request.merged",
            data={
                "title": "HNX-1 formatting only",
                "number": 2,
                "formatting_only": True,
                "changed_files": ["src/formatted.py"],
            },
        ),
        _event(
            "pr-two-at-14d",
            14 * 24 * 60,
            event_type="github.pull_request.merged",
            data={"title": "HNX-1 docs", "number": 3, "changed_files": ["docs/real.md"]},
        ),
        _event(
            "pr-after-14d",
            14 * 24 * 60 + 1,
            event_type="github.pull_request.merged",
            data={"title": "HNX-1 too late", "number": 4, "changed_files": ["src/late.py"]},
        ),
    ]

    task = select_fast_tasks(events, corpus="real")[0]

    assert task.gold["people"]["assignees"] == ["at-boundary"]
    assert task.gold["category"]["duplicate_of"] == ["HNX-7"]
    assert task.gold["place"]["files"] == ["src/one.py", "docs/real.md"]
    assert {"PR-1", "PR-3"}.issubset(task.gold["category"]["required_ids"])
