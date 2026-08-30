"""Abstention probe generator for docs/evaluation-spec.md §7 E2."""

from __future__ import annotations

import random
from datetime import datetime

from harnext_eval.probes.common import canonical_entity, uniform_time, validate_period
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
    """Generate balanced missing-entity and missing-field questions with UNKNOWN gold."""

    start, end = validate_period(probe_start, probe_end)
    rng = random.Random(f"abstention:{seed}")
    entities = sorted({canonical_entity(event) for event in events})
    if not entities:
        raise ValueError("abstention probes require at least one observed entity")
    candidates: list[ProbeCandidate] = []
    for index in range(max(count * 3, len(entities) * 3, 6)):
        snapshot_time = uniform_time(rng, start, end)
        missing_entity = f"MISSING-{seed}-{index:05d}"
        candidates.append(
            ProbeCandidate(
                family="abstention",
                entity=missing_entity,
                T=snapshot_time,
                question=f"What is the status of {missing_entity}?",
                gold="UNKNOWN",
                gold_type="exact",
                stratum="missing-entity",
            )
        )
        entity = entities[index % len(entities)]
        missing_field = f"absent_field_{index:05d}"
        candidates.append(
            ProbeCandidate(
                family="abstention",
                entity=entity,
                T=uniform_time(rng, start, end),
                question=f"What is the {missing_field} of {entity}?",
                gold="UNKNOWN",
                gold_type="exact",
                stratum="missing-field",
            )
        )
    return stratified_sample(candidates, count, seed=seed)


generate = generate_abstention_probes
