"""Knowledge-update probe generator for docs/evaluation-spec.md §7 E2."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from harnext_eval.probes.common import string_value, uniform_time, unique, validate_period
from harnext_eval.probes.gold import (
    GoldAuditTrail,
    GoldRequest,
    PythonGold,
    RawJiraInput,
    SqlGold,
)
from harnext_eval.probes.schema import ProbeCandidate, stratified_sample
from harnext_eval.types import EvalEvent, Probe


def generate_update_probes(
    events: list[EvalEvent],
    *,
    count: int,
    seed: int,
    probe_start: datetime,
    probe_end: datetime,
    raw_jira: RawJiraInput | None = None,
    gold_audit: GoldAuditTrail | None = None,
) -> list[Probe]:
    """Ask for latest state after at least two transitions and retain old values."""

    start, end = validate_period(probe_start, probe_end)
    python = PythonGold(events)
    rng = random.Random(f"update:{seed}")
    candidates: list[ProbeCandidate] = []
    with SqlGold(raw_jira if raw_jira is not None else events) as sql:
        for history in python.histories().values():
            if not history or any(item.source_kind != "jira" for item in history):
                continue
            for index in range(1, len(history)):
                latest = history[index]
                lower = max(start, latest.time)
                next_time = history[index + 1].time if index + 1 < len(history) else end
                upper = min(end, next_time - timedelta(microseconds=1))
                if lower > upper or latest.new_value is None:
                    continue
                snapshot_time = uniform_time(rng, lower, upper)
                py_value = python.field_value(latest.entity, latest.field, snapshot_time)
                sql_value = sql.field_value(latest.entity, latest.field, snapshot_time)
                request = GoldRequest(latest.entity, latest.field, snapshot_time)
                if gold_audit is not None:
                    py_value = gold_audit.compare(request, py_value, sql_value)
                elif py_value != sql_value:
                    py_value = None
                if py_value is None:
                    continue
                gold = string_value(py_value)
                raw_superseded = [history[0].old_value]
                raw_superseded.extend(item.new_value for item in history[:index])
                superseded = tuple(
                    value
                    for value in unique(
                        string_value(item) for item in raw_superseded if item is not None
                    )
                    if value != gold
                )
                if not superseded:
                    continue
                candidates.append(
                    ProbeCandidate(
                        family="update",
                        entity=latest.entity,
                        T=snapshot_time,
                        question=(
                            f"After all updates through the snapshot, what is the latest "
                            f"{latest.field} of {latest.entity}?"
                        ),
                        gold=gold,
                        gold_type="exact",
                        superseded_values=superseded,
                        source_event_ids=tuple(item.event_id for item in history[: index + 1]),
                        stratum=latest.entity.casefold(),
                    )
                )
    return stratified_sample(candidates, count, seed=seed)


generate = generate_update_probes
