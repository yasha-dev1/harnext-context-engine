"""Internal probe candidates and stratified sampling for spec §7 E2."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from harnext_eval.types import Probe

ProbeFamily = Literal[
    "extraction", "temporal", "update", "multisource", "code_location", "abstention"
]
GoldType = Literal["exact", "links", "files"]


@dataclass(frozen=True)
class ProbeCandidate:
    """An unhashed candidate; the public record remains ``harnext_eval.types.Probe``."""

    family: ProbeFamily
    entity: str
    T: datetime
    question: str
    gold: Any
    gold_type: GoldType
    superseded_values: tuple[str, ...] = ()
    source_event_ids: tuple[str, ...] = ()
    stratum: str = field(default="all", compare=False)

    def stable_key(self) -> str:
        payload = {
            "family": self.family,
            "entity": self.entity,
            "T": self.T.isoformat(),
            "question": self.question,
            "gold": self.gold,
            "gold_type": self.gold_type,
            "superseded_values": self.superseded_values,
            "source_event_ids": self.source_event_ids,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)

    def to_probe(self) -> Probe:
        digest = hashlib.sha256(self.stable_key().encode()).hexdigest()[:24]
        return Probe(
            probe_id=f"{self.family}-{digest}",
            family=self.family,
            entity=self.entity,
            T=self.T,
            question=self.question,
            gold=self.gold,
            gold_type=self.gold_type,
            superseded_values=list(self.superseded_values),
            source_event_ids=list(self.source_event_ids),
        )


def stratified_sample(
    candidates: list[ProbeCandidate], count: int, *, seed: int
) -> list[Probe]:
    """Sample round-robin across strata, deterministically within each stratum."""

    if count < 0:
        raise ValueError("count must be non-negative")
    unique = {candidate.stable_key(): candidate for candidate in candidates}
    if len(unique) < count:
        family = candidates[0].family if candidates else "requested family"
        raise ValueError(
            f"cannot sample {count} {family} probes from {len(unique)} unique candidates"
        )
    if count == 0:
        return []

    groups: dict[str, list[ProbeCandidate]] = {}
    for candidate in unique.values():
        groups.setdefault(candidate.stratum, []).append(candidate)

    rng = random.Random(seed)
    strata = sorted(groups)
    rng.shuffle(strata)
    for stratum in strata:
        groups[stratum].sort(key=ProbeCandidate.stable_key)
        rng.shuffle(groups[stratum])

    sampled: list[ProbeCandidate] = []
    while len(sampled) < count:
        progressed = False
        for stratum in strata:
            group = groups[stratum]
            if group:
                sampled.append(group.pop())
                progressed = True
                if len(sampled) == count:
                    break
        if not progressed:  # guarded by the unique-count check above
            break
    return [candidate.to_probe() for candidate in sampled]
