"""Code-location gold for docs/evaluation-spec.md §4 and §7 E4 Place."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from harnext_eval.probes.common import (
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
    """Union files/modules over qualifying PRs merged in (T, T + 14 days]."""

    files: set[str] = set()
    source_ids: list[str] = []
    horizon = at + _MERGE_HORIZON
    for event in events:
        if event.time <= at or event.time > horizon:
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


def generate_code_location_probes(
    events: list[EvalEvent],
    *,
    count: int,
    seed: int,
    probe_start: datetime,
    probe_end: datetime,
) -> list[Probe]:
    """Sample T before a merge and derive future 14-day localisation gold."""

    start, end = validate_period(probe_start, probe_end)
    rng = random.Random(f"code-location:{seed}")
    candidates: list[ProbeCandidate] = []
    for event in events:
        if not is_merged_pr(event) or is_formatting_only(event) or not changed_files(event):
            continue
        upper = min(end, event.time - timedelta(microseconds=1))
        lower = max(start, event.time - _MERGE_HORIZON)
        if lower > upper:
            continue
        for entity in issue_keys_for_pr(event):
            snapshot_time = uniform_time(rng, lower, upper, exclude_end=True)
            gold, source_ids = code_location_gold(events, entity, snapshot_time)
            if not gold["files"]:
                continue
            candidates.append(
                ProbeCandidate(
                    family="code_location",
                    entity=entity,
                    T=snapshot_time,
                    question=(
                        f"Which files and modules are changed by pull requests carrying "
                        f"{entity} that merge within 14 days after the snapshot?"
                    ),
                    gold=gold,
                    gold_type="files",
                    source_event_ids=tuple(source_ids),
                    stratum=entity.casefold(),
                )
            )
    return stratified_sample(candidates, count, seed=seed)


generate = generate_code_location_probes
