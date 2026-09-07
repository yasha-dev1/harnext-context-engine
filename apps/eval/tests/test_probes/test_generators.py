"""Five-family generator tests for docs/evaluation-spec.md §7 E2 and E4 Place."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta

import pytest
from harnext_eval.probes.common import changed_files, module_for_file
from harnext_eval.probes.gen import (
    generate_probe_set,
    validate_a0_prior,
    validate_evidentiary_population,
)
from harnext_eval.probes.gen_abstention import fact_absent_for_window
from harnext_eval.probes.gen_code_location import (
    code_location_gold,
    generate_code_location_probes,
)
from harnext_eval.probes.gen_multisource import JoinAuditTrail, regex_join_keys
from harnext_eval.types import EvalEvent, Probe

FAMILIES = {
    "extraction",
    "temporal",
    "update",
    "multisource",
    "abstention",
}


def _generate(
    events: list[EvalEvent], period: tuple[datetime, datetime], *, seed: int = 7
) -> list[Probe]:
    return generate_probe_set(
        events,
        per_family=12,
        seed=seed,
        probe_start=period[0],
        probe_end=period[1],
    )


def test_each_family_has_stratified_valid_gold(
    synthetic_events: list[EvalEvent], probe_period: tuple[datetime, datetime]
) -> None:
    probes = _generate(synthetic_events, probe_period)
    counts = Counter(probe.family for probe in probes)

    assert counts == {family: 12 for family in FAMILIES}
    assert len({probe.probe_id for probe in probes}) == len(probes)
    assert all(probe_period[0] <= probe.T <= probe_period[1] for probe in probes)
    for family in FAMILIES:
        family_probes = [probe for probe in probes if probe.family == family]
        assert len({probe.entity for probe in family_probes}) >= 2
        assert all(probe.question and probe.gold is not None for probe in family_probes)

    assert all(
        probe.gold_type == "exact"
        for probe in probes
        if probe.family in {"extraction", "temporal", "update", "abstention"}
    )
    assert all(probe.gold_type == "links" for probe in probes if probe.family == "multisource")


def test_generation_is_deterministic_by_seed(
    synthetic_events: list[EvalEvent], probe_period: tuple[datetime, datetime]
) -> None:
    first = _generate(synthetic_events, probe_period, seed=31)
    second = _generate(synthetic_events, probe_period, seed=31)
    different = _generate(synthetic_events, probe_period, seed=32)

    assert [probe.model_dump_json() for probe in first] == [
        probe.model_dump_json() for probe in second
    ]
    assert [probe.probe_id for probe in first] != [probe.probe_id for probe in different]


def test_updates_have_two_transitions_and_superseded_values(
    synthetic_events: list[EvalEvent], probe_period: tuple[datetime, datetime]
) -> None:
    by_id = {event.id: event for event in synthetic_events}
    updates = [
        probe
        for probe in _generate(synthetic_events, probe_period)
        if probe.family == "update"
    ]

    assert updates
    for probe in updates:
        transitions = [by_id[event_id] for event_id in probe.source_event_ids]
        assert len(transitions) >= 2
        # Synthetic v2's silent-burst preflight is telemetry-typed while still
        # carrying a real changelog transition consumed by both gold readers.
        assert all((event.data or {}).get("changelog") for event in transitions)
        assert all(event.time <= probe.T for event in transitions)
        assert probe.superseded_values
        assert str(probe.gold) not in probe.superseded_values


def test_abstention_gold_is_unknown(
    synthetic_events: list[EvalEvent], probe_period: tuple[datetime, datetime]
) -> None:
    abstentions = [
        probe
        for probe in _generate(synthetic_events, probe_period)
        if probe.family == "abstention"
    ]

    assert abstentions
    assert all(probe.gold == "UNKNOWN" for probe in abstentions)
    assert all(not probe.source_event_ids for probe in abstentions)
    assert all("MISSING-" not in probe.question for probe in abstentions)
    assert all("absent_field_" not in probe.question for probe in abstentions)
    assert all(
        fact_absent_for_window(
            synthetic_events,
            probe.entity,
            probe.question.split("current ", 1)[1].split(" of ", 1)[0],
            probe_period[1],
        )
        for probe in abstentions
    )


def _event(
    event_id: str,
    day: int,
    *,
    event_type: str,
    subject: str,
    data: dict[str, object],
    source: str = "jira:kafka",
) -> EvalEvent:
    return EvalEvent(
        id=event_id,
        source=source,
        type=event_type,
        subject=subject,
        time=datetime(2026, 5, 1, tzinfo=UTC) + timedelta(days=day),
        mgtenant="test",
        data=data,
    )


def test_code_location_is_post_merge_issue_trigger_union_with_modules() -> None:
    events = [
        _event(
            "issue",
            0,
            event_type="org.apache.jira.issue.created",
            subject="issue:KAFKA-1",
            data={"issue_key": "KAFKA-1", "status": "Open"},
        ),
        _event(
            "pr-1",
            3,
            source="github:apache/kafka",
            event_type="com.github.pull_request.merged",
            subject="pr:10",
            data={
                "number": 10,
                "title": "First fix",
                "head_ref": "KAFKA-1-first-fix",
                "merged_at": "2026-05-04T00:00:00Z",
                "changed_files": ["core/src/Main.java"],
            },
        ),
        _event(
            "pr-2",
            10,
            source="github:apache/kafka",
            event_type="com.github.pull_request.merged",
            subject="pr:11",
            data={
                "number": 11,
                "title": "KAFKA-1 follow-up",
                "merged_at": "2026-05-11T00:00:00Z",
                "changed_files": ["clients/Dockerfile"],
            },
        ),
        _event(
            "late",
            15,
            source="github:apache/kafka",
            event_type="com.github.pull_request.merged",
            subject="pr:12",
            data={
                "number": 12,
                "title": "KAFKA-1 too late",
                "merged_at": "2026-05-16T00:00:00Z",
                "changed_files": ["late/Excluded.java"],
            },
        ),
    ]
    locations = generate_code_location_probes(
        events,
        count=1,
        seed=3,
        probe_start=datetime(2026, 5, 11, 0, 0, 1, tzinfo=UTC),
        probe_end=datetime(2026, 5, 14, tzinfo=UTC),
    )

    assert locations
    for probe in locations:
        expected_gold, expected_ids = code_location_gold(events, probe.entity, probe.T)
        assert probe.gold == expected_gold
        assert probe.source_event_ids == expected_ids
        assert probe.family == "multisource"
        assert probe.gold_type == "files"
        source_events = [event for event in events if event.id in probe.source_event_ids]
        assert all(event.time < probe.T for event in source_events)
        expected_files = sorted(
            {path for event in source_events for path in changed_files(event)}
        )
        assert probe.gold["files"] == expected_files
        assert probe.gold["modules"] == sorted(
            {module_for_file(path) for path in expected_files}
        )
        assert "late/Excluded.java" not in probe.gold["files"]


def test_regex_join_uses_only_pr_title_thread_subject_and_commit_message() -> None:
    pr = _event(
        "pr",
        1,
        source="github:apache/kafka",
        event_type="com.github.pull_request.opened",
        subject="pr:1",
        data={
            "title": "KAFKA-1 KIP-2 fix",
            "head_ref": "KAFKA-3-follow-up",
            "body": "KAFKA-999",
        },
    )
    mail = _event(
        "mail",
        1,
        source="mail:dev@kafka.apache.org",
        event_type="org.apache.mail.message",
        subject="thread:r1",
        data={"subject": "[DISCUSS] KIP-2", "body": "KAFKA-999"},
    )
    commit = _event(
        "commit",
        1,
        source="github:apache/kafka",
        event_type="com.github.push",
        subject="contributor:a",
        data={"commit_message": "finish KAFKA-1", "title": "KIP-999"},
    )
    assert regex_join_keys(pr) == ["KAFKA-1", "KIP-2", "KAFKA-3"]
    assert regex_join_keys(mail) == ["KIP-2"]
    assert regex_join_keys(commit) == ["KAFKA-1"]


def test_join_audit_reports_precision_and_recall() -> None:
    audit = JoinAuditTrail(expected={"a": ["KAFKA-1", "KIP-2"]})
    audit.record("a", ["KAFKA-1", "KAFKA-9"])
    report = audit.report()
    assert report["join_precision"] == 0.5
    assert report["join_recall"] == 0.5


def test_evidentiary_population_rejects_under_150_entities(
    synthetic_events: list[EvalEvent], probe_period: tuple[datetime, datetime]
) -> None:
    template = _generate(synthetic_events, probe_period)[0]
    probes = [
        template.model_copy(
            update={
                "probe_id": f"{family}-{index}",
                "family": family,
                "entity": f"ENTITY-{(family_index * 60 + index) % 149}",
            }
        )
        for family_index, family in enumerate(sorted(FAMILIES))
        for index in range(60)
    ]
    with pytest.raises(ValueError, match="at least 150 entities"):
        validate_evidentiary_population(probes)
    complete = [
        probe.model_copy(update={"entity": f"ENTITY-{index % 150}"})
        for index, probe in enumerate(probes)
    ]
    validate_evidentiary_population(complete)


def test_a0_prior_audits_named_families_and_reports_abstention_separately(
    synthetic_events: list[EvalEvent], probe_period: tuple[datetime, datetime]
) -> None:
    probes = _generate(synthetic_events, probe_period)
    answers = {
        probe.probe_id: "UNKNOWN"
        for probe in probes
        if probe.family in {"temporal", "update", "multisource", "abstention"}
    }
    report = validate_a0_prior(probes, answers)

    assert report["population"] == ["temporal", "update", "multisource"]
    assert report["combined_accuracy"] == 0.0
    assert report["per_family_accuracy"] == {
        "temporal": 0.0,
        "update": 0.0,
        "multisource": 0.0,
    }
    assert report["abstention_diagnostic_accuracy"] == 1.0

    temporal = [probe for probe in probes if probe.family == "temporal"]
    for probe in temporal[:4]:
        answers[probe.probe_id] = str(probe.gold)
    with pytest.raises(ValueError, match="temporal=33.333%"):
        validate_a0_prior(probes, answers)


def test_a0_code_probe_uses_files_gold_type_and_partial_set_f1() -> None:
    when = datetime(2026, 5, 2, tzinfo=UTC)
    probes = [
        Probe(
            probe_id="temporal",
            family="temporal",
            entity="KAFKA-1",
            T=when,
            question="historical status?",
            gold="Open",
            gold_type="exact",
        ),
        Probe(
            probe_id="update",
            family="update",
            entity="KAFKA-1",
            T=when,
            question="latest status?",
            gold="Closed",
            gold_type="exact",
        ),
        Probe(
            probe_id="code",
            family="multisource",
            entity="KAFKA-1",
            T=when,
            question="files?",
            gold={"files": ["src/one.py", "src/two.py"], "modules": ["src"]},
            gold_type="files",
        ),
    ]
    report = validate_a0_prior(
        probes,
        {
            "temporal": "UNKNOWN",
            "update": "UNKNOWN",
            "code": '{"files":["src/one.py"]}',
        },
        maximum_accuracy=1.0,
    )

    assert report["per_family_accuracy"] == {
        "temporal": 0.0,
        "update": 0.0,
        "multisource": pytest.approx(2 / 3),
    }
