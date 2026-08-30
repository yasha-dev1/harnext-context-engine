"""Six-family generator tests for docs/evaluation-spec.md §7 E2 and E4 Place."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

from harnext_eval.probes.common import changed_files, module_for_file
from harnext_eval.probes.gen import generate_probe_set
from harnext_eval.probes.gen_code_location import code_location_gold
from harnext_eval.types import EvalEvent, Probe

FAMILIES = {
    "extraction",
    "temporal",
    "update",
    "multisource",
    "code_location",
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
    assert all(
        probe.gold_type == "links" and isinstance(probe.gold, list) and probe.gold
        for probe in probes
        if probe.family == "multisource"
    )
    assert all(
        probe.gold_type == "files"
        and isinstance(probe.gold, dict)
        and probe.gold["files"]
        and probe.gold["modules"]
        for probe in probes
        if probe.family == "code_location"
    )


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
        assert all("transition" in event.type for event in transitions)
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


def test_code_location_is_future_14_day_union_with_modules(
    synthetic_events: list[EvalEvent], probe_period: tuple[datetime, datetime]
) -> None:
    by_id = {event.id: event for event in synthetic_events}
    locations = [
        probe
        for probe in _generate(synthetic_events, probe_period)
        if probe.family == "code_location"
    ]

    assert locations
    assert any(len(probe.source_event_ids) > 1 for probe in locations)
    for probe in locations:
        expected_gold, expected_ids = code_location_gold(
            synthetic_events, probe.entity, probe.T
        )
        assert probe.gold == expected_gold
        assert probe.source_event_ids == expected_ids
        source_events = [by_id[event_id] for event_id in probe.source_event_ids]
        assert all(probe.T < event.time <= probe.T + timedelta(days=14) for event in source_events)
        expected_files = sorted(
            {path for event in source_events for path in changed_files(event)}
        )
        assert probe.gold["files"] == expected_files
        assert probe.gold["modules"] == sorted(
            {module_for_file(path) for path in expected_files}
        )
