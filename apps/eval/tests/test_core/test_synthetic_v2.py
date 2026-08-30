"""Scenario-engine regressions for synthetic corpus v2 (§4.1 and E1–E6)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from harnext_eval.corpus.synthetic import (
    generate_synthetic_corpus,
    generate_synthetic_events,
)
from harnext_eval.stores.fake_usage import (
    ensure_fake_fold_usage,
    estimate_fake_fold_usage,
)
from harnext_eval.types import EvalEvent


@pytest.fixture(scope="module")
def scenario_events() -> list[EvalEvent]:
    return generate_synthetic_events(seed=11, event_count=800, entity_count=24)


def test_injected_situations_are_balanced_and_have_detectable_outcomes(
    scenario_events: list[EvalEvent],
) -> None:
    positives = [
        (index, event)
        for index, event in enumerate(scenario_events)
        if (event.data or {}).get("injected_positive")
    ]
    archetypes = Counter((event.data or {}).get("situation_archetype") for _, event in positives)

    assert 0.01 <= len(positives) / len(scenario_events) <= 0.06
    assert set(archetypes) == {
        "declared-critical",
        "security-cve",
        "vote-thread",
        "silent-burst",
    }
    assert max(archetypes.values()) / len(positives) <= 0.4
    for index, onset in positives:
        outcomes = scenario_events[index + 1 : index + 10]
        assert any(
            event.time - onset.time <= timedelta(hours=1)
            and (event.data or {}).get("is_committer")
            for event in outcomes
        )
        assert any((event.data or {}).get("urgent_outcome") for event in outcomes)
        assert any(
            (event.data or {}).get("outcome_for")
            and (event.data or {}).get("field") in {"status", "priority"}
            for event in outcomes
        )

    hard_negatives = [
        event for event in scenario_events if (event.data or {}).get("hard_negative")
    ]
    assert hard_negatives
    assert all(event.type.endswith("flash_crowd") for event in hard_negatives)
    assert all(not (event.data or {}).get("outcome_for") for event in hard_negatives)
    assert all(not (event.data or {}).get("changelog") for event in hard_negatives)


def test_state_and_cross_source_gold_are_non_trivial(
    scenario_events: list[EvalEvent],
) -> None:
    transition_counts: Counter[tuple[str, str]] = Counter()
    issue_gaps: dict[str, list[float]] = defaultdict(list)
    previous_time: dict[str, datetime] = {}
    for event in scenario_events:
        data = event.data or {}
        issue = str(data.get("issue_key"))
        field = data.get("field")
        if isinstance(field, str):
            transition_counts[(issue, field)] += 1
        if issue in previous_time:
            issue_gaps[issue].append((event.time - previous_time[issue]).total_seconds())
        previous_time[issue] = event.time

    active_issues = {issue for issue, _ in transition_counts}
    repeatedly_mutated = {
        issue
        for (issue, _field), count in transition_counts.items()
        if count >= 2
    }
    assert len(repeatedly_mutated) >= len(active_issues) * 0.75
    assert any(len(set(gaps)) >= 3 for gaps in issue_gaps.values())

    pull_requests = [event for event in scenario_events if "pull_request" in event.type]
    mail = [event for event in scenario_events if event.source.startswith("mail:")]
    assert pull_requests and mail
    for event in pull_requests:
        data = event.data or {}
        assert str(data["issue_key"]) in str(data["title"])
        assert "KIP-" in str(data["title"])
        assert all(len(path.split("/")) >= 3 for path in data["changed_files"])
    assert all(
        str((event.data or {})["issue_key"]) in str((event.data or {})["subject"])
        and "KIP-" in str((event.data or {})["subject"])
        for event in mail
    )


def test_corpus_meta_records_exact_situations_and_actor_roles(tmp_path: Path) -> None:
    corpus = generate_synthetic_corpus(
        tmp_path,
        seed=3,
        event_count=400,
        entity_count=16,
    )
    meta = corpus.meta

    assert meta["generator"] == "synthetic-v2"
    assert meta["injected_situations"]
    assert all(
        {"event_id", "onset", "archetype", "cost_weight", "entity"} <= set(item)
        for item in meta["injected_situations"]
    )
    actors = meta["actor_catalog"]
    assert len(actors["committers"]) / len(actors["humans"]) == pytest.approx(0.2)
    assert len(actors["bots"]) == 2


def test_fake_fold_usage_rewards_batching(scenario_events: list[EvalEvent]) -> None:
    sample = scenario_events[:20]
    per_event = [estimate_fake_fold_usage([event]) for event in sample]
    batched = estimate_fake_fold_usage(sample)

    assert sum(row.input_tokens for row in per_event) > batched.input_tokens
    assert sum(row.cost_usd for row in per_event) > batched.cost_usd


def test_fake_fold_usage_writer_does_not_duplicate_provider_rows(
    scenario_events: list[EvalEvent], tmp_path: Path
) -> None:
    path = tmp_path / "usage.jsonl"

    assert ensure_fake_fold_usage(path, 0, scenario_events[:2], "batch", "S0")
    assert not ensure_fake_fold_usage(path, 0, scenario_events[:2], "batch", "S0")
    row = json.loads(path.read_text())
    assert row["input_tokens"] > 0
    assert row["output_tokens"] > 0
    assert row["cost_usd"] > 0
