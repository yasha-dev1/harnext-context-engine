"""Probe generation for docs/evaluation-spec.md §4, §5, §7 E2 and E4."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from harnext_eval.types import EvalEvent, Probe


def generate_probe_set(
    events: list[EvalEvent],
    *,
    per_family: int,
    seed: int,
    probe_start: datetime,
    probe_end: datetime,
    **options: Any,
) -> list[Probe]:
    """Lazily import the generator so ``python -m ...probes.gen`` stays warning-free."""

    from harnext_eval.probes.gen import generate_probe_set as generate

    return generate(
        events,
        per_family=per_family,
        seed=seed,
        probe_start=probe_start,
        probe_end=probe_end,
        **options,
    )


def write_probe_set(
    probes: Sequence[Probe], output: str | Path
) -> tuple[Path, Path, str]:
    from harnext_eval.probes.gen import write_probe_set as write

    return write(probes, output)


__all__ = ["generate_probe_set", "write_probe_set"]
