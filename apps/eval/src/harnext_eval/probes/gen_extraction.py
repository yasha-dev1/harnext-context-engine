"""Extraction probe generator for docs/evaluation-spec.md §7 E2."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from harnext_eval.probes.common import string_value, uniform_time, validate_period
from harnext_eval.probes.gold import (
    GoldAuditTrail,
    GoldRequest,
    PythonGold,
    RawJiraInput,
    SqlGold,
)
from harnext_eval.probes.schema import ProbeCandidate, stratified_sample
from harnext_eval.types import EvalEvent, Probe


def generate_extraction_probes(
    events: list[EvalEvent],
    *,
    count: int,
    seed: int,
    probe_start: datetime,
    probe_end: datetime,
    raw_jira: RawJiraInput | None = None,
    gold_audit: GoldAuditTrail | None = None,
) -> list[Probe]:
    """Ask for current Jira, PR, KIP, thread, or world-state values."""

    start, end = validate_period(probe_start, probe_end)
    python = PythonGold(events)
    rng = random.Random(f"extraction:{seed}")
    candidates: list[ProbeCandidate] = []
    with SqlGold(raw_jira if raw_jira is not None else events) as sql:
        for history in python.histories().values():
            for index, transition in enumerate(history):
                lower = max(start, transition.time)
                next_time = history[index + 1].time if index + 1 < len(history) else end
                upper = min(end, next_time - timedelta(microseconds=1))
                if lower > upper or transition.new_value is None:
                    continue
                snapshot_time = uniform_time(rng, lower, upper)
                py_value = python.field_value(
                    transition.entity, transition.field, snapshot_time
                )
                if transition.source_kind == "jira":
                    sql_value = sql.field_value(
                        transition.entity, transition.field, snapshot_time
                    )
                    request = GoldRequest(
                        transition.entity, transition.field, snapshot_time
                    )
                    if gold_audit is not None:
                        py_value = gold_audit.compare(request, py_value, sql_value)
                    elif py_value != sql_value:
                        py_value = None
                if py_value is None:
                    continue
                candidates.append(
                    ProbeCandidate(
                        family="extraction",
                        entity=transition.entity,
                        T=snapshot_time,
                        question=(
                            f"What is the current {transition.field} of {transition.entity} "
                            f"at the snapshot time?"
                        ),
                        gold=string_value(py_value),
                        gold_type="exact",
                        source_event_ids=(transition.event_id,),
                        stratum=transition.entity.casefold(),
                    )
                )
    return stratified_sample(candidates, count, seed=seed)


generate = generate_extraction_probes
