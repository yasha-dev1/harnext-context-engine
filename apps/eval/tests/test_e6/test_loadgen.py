"""Load-generator tests for docs/evaluation-spec.md §7 E6."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from harnext_eval.corpus.synthetic import generate_synthetic_events
from harnext_eval.e6.loadgen import (
    calibrate_pareto_on_off,
    fit_workload,
    generate_schedule,
    kim_jo_burstiness,
)


def _fit():
    return fit_workload(
        generate_synthetic_events(seed=7, event_count=240, days=1, entity_count=6)
    )


def test_kim_jo_burstiness_on_known_sequences() -> None:
    assert kim_jo_burstiness([1.0] * 99) == pytest.approx(-1.0)

    rng = np.random.default_rng(4)
    poisson_intervals = rng.exponential(1.0, size=20_000)
    assert kim_jo_burstiness(poisson_intervals.tolist()) == pytest.approx(0.0, abs=0.025)

    bursty = np.concatenate([np.full(999, 0.01), np.array([100.0])])
    assert kim_jo_burstiness(bursty.tolist()) > 0.8


def test_pareto_on_off_calibration_converges() -> None:
    calibrated = calibrate_pareto_on_off(
        _fit(),
        target_b=0.5,
        duration_s=45.0,
        rate_hz=25.0,
        seed=19,
        tolerance=0.08,
    )

    assert calibrated.converged
    assert calibrated.realised_b == pytest.approx(0.5, abs=0.08)
    assert calibrated.tail_index > 1.0
    assert calibrated.schedule


@pytest.mark.parametrize(
    "shape",
    [
        "steady",
        "poisson",
        "pareto_on_off",
        "benign_flash",
        "anomalous_burst",
        "zipf_hot",
    ],
)
def test_all_schedules_are_monotone(shape: str) -> None:
    schedule = generate_schedule(
        _fit(),
        shape=shape,  # type: ignore[arg-type]
        duration_s=2.0,
        burst_duration_s=0.5,
        rate_hz=20.0,
        target_b=0.5,
        start=datetime(2026, 1, 1, tzinfo=UTC),
        seed=3,
        worker_kill=shape == "benign_flash",
    )
    timestamps = [entry.intended_send_ts for entry in schedule]

    assert timestamps == sorted(timestamps)
    assert all(
        entry.event is None or entry.event.intended_send_ts == entry.intended_send_ts
        for entry in schedule
    )
    if shape == "benign_flash":
        assert any(entry.marker == "worker_kill" for entry in schedule)
