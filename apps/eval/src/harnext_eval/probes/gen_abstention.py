"""Whole-window abstention probes for docs/evaluation-spec.md §7 E2."""

from __future__ import annotations

import random
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from harnext_eval.probes.common import ISSUE_KEY_RE, canonical_entity, uniform_time, validate_period
from harnext_eval.probes.gold import PythonGold
from harnext_eval.probes.schema import ProbeCandidate, stratified_sample
from harnext_eval.types import EvalEvent, Probe


def generate_abstention_probes(
    events: list[EvalEvent],
    *,
    count: int,
    seed: int,
    probe_start: datetime,
    probe_end: datetime,
) -> list[Probe]:
    """Ask natural questions whose entity/field fact is absent from the whole window."""

    start, end = validate_period(probe_start, probe_end)
    rng = random.Random(f"abstention:{seed}")
    oracle = PythonGold(events)
    histories = oracle.histories()
    observed_entities = sorted({history[-1].entity for history in histories.values() if history})
    valid_fields = sorted({history[-1].field for history in histories.values() if history})
    if not observed_entities or not valid_fields:
        raise ValueError("abstention probes require observed state fields and entities")

    candidates: list[ProbeCandidate] = []
    entity_first_seen = {
        entity.casefold(): min(
            history[0].time
            for history in histories.values()
            if history and history[0].entity.casefold() == entity.casefold()
        )
        for entity in observed_entities
    }
    field_first_seen = {
        field_name.casefold(): min(
            history[0].time
            for (_, history_field), history in histories.items()
            if history and history_field == field_name.casefold()
        )
        for field_name in valid_fields
    }
    for entity in observed_entities:
        for field_name in valid_fields:
            if oracle.field_value(entity, field_name, end) is not None:
                continue
            lower = max(
                start,
                entity_first_seen[entity.casefold()],
                field_first_seen[field_name.casefold()],
            )
            if lower > end:
                continue
            candidates.append(
                ProbeCandidate(
                    family="abstention",
                    entity=entity,
                    T=uniform_time(rng, lower, end),
                    question=f"What is the current {field_name} of {entity}?",
                    gold="UNKNOWN",
                    gold_type="exact",
                    stratum=f"missing-field:{entity.casefold()}",
                )
            )

    window_entities = {
        entity.casefold()
        for event in events
        if start <= event.time <= end
        for entity in _event_entities(event, ignore_catalogue=True)
    }
    catalogued = _catalogued_entities(events, at_or_before=start)
    for entity in sorted(catalogued):
        if entity.casefold() in window_entities:
            continue
        if any(
            oracle.field_value(entity, field_name, end) is not None
            for field_name in valid_fields
        ):
            continue
        field_name = valid_fields[len(candidates) % len(valid_fields)]
        candidates.append(
            ProbeCandidate(
                family="abstention",
                entity=entity,
                T=uniform_time(rng, start, end),
                question=f"What is the current {field_name} of {entity}?",
                gold="UNKNOWN",
                gold_type="exact",
                stratum=f"missing-entity:{entity.casefold()}",
            )
        )
    return stratified_sample(candidates, count, seed=seed)


def fact_absent_for_window(
    events: list[EvalEvent], entity: str, field_name: str, end: datetime
) -> bool:
    """Return whether retrieve-everything through the window end lacks the fact."""

    return PythonGold(events).field_value(entity, field_name, end) is None


def _catalogued_entities(events: list[EvalEvent], *, at_or_before: datetime) -> set[str]:
    entities: set[str] = set()
    for event in events:
        if event.time > at_or_before:
            continue
        data = event.data or {}
        raw = data.get("known_entities", [])
        if isinstance(raw, list):
            entities.update(str(value) for value in raw if str(value).strip())
    return entities


def _event_entities(event: EvalEvent, *, ignore_catalogue: bool) -> set[str]:
    entities = {canonical_entity(event)}

    def walk(value: Any, *, key: str = "") -> None:
        if ignore_catalogue and key in {"known_entities", "known_kips"}:
            return
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                walk(child, key=str(child_key))
        elif isinstance(value, list):
            for child in value:
                walk(child, key=key)
        elif isinstance(value, str):
            entities.update(ISSUE_KEY_RE.findall(value))

    walk(event.data or {})
    return entities


generate = generate_abstention_probes
