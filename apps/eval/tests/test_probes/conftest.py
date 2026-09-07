"""Synthetic fixtures for docs/evaluation-spec.md §7 E2 probe tests."""

from __future__ import annotations

from datetime import datetime

import pytest
from harnext_eval.corpus.synthetic import generate_synthetic_events
from harnext_eval.types import EvalEvent


@pytest.fixture(scope="session")
def synthetic_events() -> list[EvalEvent]:
    return generate_synthetic_events(seed=23, event_count=1_000, entity_count=30)


@pytest.fixture(scope="session")
def probe_period(synthetic_events: list[EvalEvent]) -> tuple[datetime, datetime]:
    return synthetic_events[0].time, synthetic_events[-1].time
