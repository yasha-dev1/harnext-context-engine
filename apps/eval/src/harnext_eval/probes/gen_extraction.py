"""Extraction probe generator for docs/evaluation-spec.md §7 E2."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from harnext_eval.probes.common import string_value, uniform_time, validate_period
from harnext_eval.probes.gold import PythonGold, SqlGold
from harnext_eval.probes.schema import ProbeCandidate, stratified_sample
from harnext_eval.types import EvalEvent, Probe


def generate_extraction_probes(
    events: list[EvalEvent],
    *,
    count: int,
    seed: int,
    probe_start: datetime,
    probe_end: datetime,
) -> list[Probe]:
    """Ask for a current field value at a sampled snapshot time."""

    start, end = validate_period(probe_start, probe_end)
    python = PythonGold(events)
    rng = random.Random(f"extraction:{seed}")
    candidates: list[ProbeCandidate] = []
    with SqlGold(events) as sql:
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
                sql_value = sql.field_value(transition.entity, transition.field, snapshot_time)
                if py_value != sql_value or py_value is None:
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
                        stratum=f"{transition.entity.casefold()}:{transition.field.casefold()}",
                    )
                )
    return stratified_sample(candidates, count, seed=seed)


generate = generate_extraction_probes
