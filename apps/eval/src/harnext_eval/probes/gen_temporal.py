"""Temporal probe generator for docs/evaluation-spec.md §7 E2."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from harnext_eval.probes.common import display_time, string_value, uniform_time, validate_period
from harnext_eval.probes.gold import (
    GoldAuditTrail,
    GoldRequest,
    PythonGold,
    RawJiraInput,
    SqlGold,
)
from harnext_eval.probes.schema import ProbeCandidate, stratified_sample
from harnext_eval.types import EvalEvent, Probe


def generate_temporal_probes(
    events: list[EvalEvent],
    *,
    count: int,
    seed: int,
    probe_start: datetime,
    probe_end: datetime,
    raw_jira: RawJiraInput | None = None,
    gold_audit: GoldAuditTrail | None = None,
) -> list[Probe]:
    """Ask for a historical value T' while material is frozen at later Probe.T."""

    start, end = validate_period(probe_start, probe_end)
    python = PythonGold(events)
    rng = random.Random(f"temporal:{seed}")
    candidates: list[ProbeCandidate] = []
    with SqlGold(raw_jira if raw_jira is not None else events) as sql:
        for history in python.histories().values():
            if not history or any(item.source_kind != "jira" for item in history):
                continue
            for index in range(len(history) - 1):
                target = history[index]
                later = history[index + 1]
                snapshot_lower = max(start, later.time)
                if snapshot_lower > end or target.new_value is None:
                    continue
                snapshot_time = uniform_time(rng, snapshot_lower, end)
                as_of_upper = min(later.time - timedelta(microseconds=1), snapshot_time)
                if target.time > as_of_upper:
                    continue
                as_of = uniform_time(rng, target.time, as_of_upper)
                py_value = python.field_value(target.entity, target.field, as_of)
                sql_value = sql.field_value(target.entity, target.field, as_of)
                request = GoldRequest(target.entity, target.field, as_of)
                if gold_audit is not None:
                    py_value = gold_audit.compare(request, py_value, sql_value)
                elif py_value != sql_value:
                    py_value = None
                if py_value is None:
                    continue
                candidates.append(
                    ProbeCandidate(
                        family="temporal",
                        entity=target.entity,
                        T=snapshot_time,
                        question=(
                            f"What was the {target.field} of {target.entity} as of "
                            f"{display_time(as_of)}?"
                        ),
                        gold=string_value(py_value),
                        gold_type="exact",
                        source_event_ids=(target.event_id,),
                        stratum=target.entity.casefold(),
                    )
                )
    return stratified_sample(candidates, count, seed=seed)


generate = generate_temporal_probes
