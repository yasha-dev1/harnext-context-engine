"""Code-location gold for docs/evaluation-spec.md §4 and §7 E4 Place."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from harnext_eval.probes.common import (
    canonical_entity,
    changed_files,
    is_formatting_only,
    is_merged_pr,
    issue_keys_for_pr,
    module_for_file,
    uniform_time,
    validate_period,
)
from harnext_eval.probes.schema import ProbeCandidate, stratified_sample
from harnext_eval.types import EvalEvent, Probe

_MERGE_HORIZON = timedelta(days=14)


def code_location_gold(
    events: list[EvalEvent], entity: str, at: datetime
) -> tuple[dict[str, list[str]], list[str]]:
    """Union PR files merged within 14 days of the issue trigger and visible by T."""

    files: set[str] = set()
    source_ids: list[str] = []
    trigger = issue_trigger_time(events, entity)
    if trigger is None:
        return {"files": [], "modules": []}, []
    horizon = trigger + _MERGE_HORIZON
    for event in events:
        if event.time < trigger or event.time > horizon or event.time > at:
            continue
        if not is_merged_pr(event) or is_formatting_only(event):
            continue
        if entity.casefold() not in {key.casefold() for key in issue_keys_for_pr(event)}:
            continue
        event_files = changed_files(event)
        if not event_files:
            continue
        files.update(event_files)
        source_ids.append(event.id)
    sorted_files = sorted(files)
    modules = sorted({module_for_file(path) for path in sorted_files})
    return {"files": sorted_files, "modules": modules}, source_ids


def issue_trigger_time(events: list[EvalEvent], entity: str) -> datetime | None:
    """Return the issue's creation/trigger time, falling back to its first event."""

    exact = [
        event.time
        for event in events
        if canonical_entity(event).casefold() == entity.casefold()
        and "jira.issue.created" in event.type.casefold()
    ]
    if exact:
        return min(exact)
    fallback = [
        event.time
        for event in events
        if canonical_entity(event).casefold() == entity.casefold()
    ]
    return min(fallback) if fallback else None


def generate_code_location_probes(
    events: list[EvalEvent],
    *,
    count: int,
    seed: int,
    probe_start: datetime,
    probe_end: datetime,
) -> list[Probe]:
    """Place E2's T after every qualifying fixing PR merge."""

    start, end = validate_period(probe_start, probe_end)
    rng = random.Random(f"code-location:{seed}")
    candidates: list[ProbeCandidate] = []
    entities: set[str] = set()
    for event in events:
        if not is_merged_pr(event) or is_formatting_only(event) or not changed_files(event):
            continue
        entities.update(issue_keys_for_pr(event))
    by_id = {event.id: event for event in events}
    for entity in sorted(entities):
        trigger = issue_trigger_time(events, entity)
        if trigger is None:
            continue
        qualifying = [
            event
            for event in events
            if trigger <= event.time <= trigger + _MERGE_HORIZON
            and is_merged_pr(event)
            and not is_formatting_only(event)
            and bool(changed_files(event))
            and entity.casefold() in {key.casefold() for key in issue_keys_for_pr(event)}
        ]
        if not qualifying:
            continue
        lower = max(start, max(event.time for event in qualifying) + timedelta(microseconds=1))
        if lower > end:
            continue
        snapshot_time = uniform_time(rng, lower, end)
        gold, source_ids = code_location_gold(events, entity, snapshot_time)
        if not gold["files"] or any(by_id[event_id].time >= snapshot_time for event_id in source_ids):
            continue
        candidates.append(
            ProbeCandidate(
                family="multisource",
                entity=entity,
                T=snapshot_time,
                question=(
                    f"Which files and modules were changed by fixing pull requests for "
                    f"{entity} merged within 14 days of the issue trigger?"
                ),
                gold=gold,
                gold_type="files",
                source_event_ids=tuple(source_ids),
                stratum=f"code:{entity.casefold()}",
            )
        )
    return stratified_sample(candidates, count, seed=seed)


generate = generate_code_location_probes
