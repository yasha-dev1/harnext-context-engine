"""Load-generator claim tests for evaluation spec §7 E6."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

import numpy as np
import pytest
from harnext_eval.corpus.synthetic import generate_synthetic_events
from harnext_eval.e6.loadgen import (
    build_schedule,
    calibrate_pareto_on_off,
    fit_workload,
    generate_schedule,
    generate_steady_schedule,
    kim_jo_burstiness,
    schedule_burstiness,
    situations_from_meta,
)


def _fit():
    return fit_workload(
        generate_synthetic_events(seed=7, event_count=240, days=1, entity_count=6)
    )


def test_kim_jo_burstiness_on_known_sequences() -> None:
    assert kim_jo_burstiness([1.0] * 99) == pytest.approx(-1.0)
    rng = np.random.default_rng(4)
    assert kim_jo_burstiness(rng.exponential(1.0, size=20_000).tolist()) == pytest.approx(
        0.0, abs=0.025
    )
    bursty = np.concatenate([np.full(999, 0.01), np.array([100.0])])
    assert kim_jo_burstiness(bursty.tolist()) > 0.8


def test_pareto_on_off_calibration_converges_and_records_parameters() -> None:
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
    assert calibrated.iterations > 0


def test_non_converged_pareto_schedule_is_invalid() -> None:
    with pytest.raises(ValueError, match="did not converge"):
        build_schedule(
            _fit(),
            shape="pareto_on_off",
            duration_s=0.01,
            rate_hz=5.0,
            target_b=0.95,
            seed=3,
        )


@pytest.mark.parametrize(
    "shape",
    ["steady", "poisson", "pareto_on_off", "benign_flash", "anomalous_burst", "zipf_hot"],
)
def test_all_schedules_are_monotone_and_propagate_intended_time(shape: str) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    schedule = generate_schedule(
        _fit(),
        shape=shape,  # type: ignore[arg-type]
        duration_s=2.0,
        burst_duration_s=0.5,
        rate_hz=20.0,
        target_b=0.5,
        start=start,
        seed=3,
        worker_kill=shape == "benign_flash",
        require_convergence=False,
    )
    timestamps = [entry.intended_send_ts for entry in schedule]
    assert timestamps == sorted(timestamps)
    assert all(
        entry.event is None or entry.event.intended_send_ts == entry.intended_send_ts
        for entry in schedule
    )
    if shape == "benign_flash":
        marker = next(entry for entry in schedule if entry.marker == "worker_kill")
        assert (marker.intended_send_ts - start).total_seconds() == pytest.approx(1.0)


def test_same_seed_freezes_ids_and_intended_timestamps() -> None:
    fit = _fit()
    left = build_schedule(
        fit,
        shape="anomalous_burst",
        duration_s=4,
        burst_duration_s=2,
        rate_hz=30,
        start=datetime(2026, 1, 1, tzinfo=UTC),
        seed=41,
    )
    right = build_schedule(
        fit,
        shape="anomalous_burst",
        duration_s=4,
        burst_duration_s=2,
        rate_hz=30,
        start=datetime(2026, 1, 1, tzinfo=UTC),
        seed=41,
    )
    assert left.schedule_id == right.schedule_id
    assert [(row.event_id, row.intended_send_ts) for row in left.entries] == [
        (row.event_id, row.intended_send_ts) for row in right.entries
    ]


def test_urgent_gold_is_sidecar_only_and_identical_across_shapes() -> None:
    fit = _fit()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    poisson = build_schedule(
        fit, shape="poisson", duration_s=4, rate_hz=30, start=start, seed=9
    )
    anomalous = build_schedule(
        fit,
        shape="anomalous_burst",
        duration_s=4,
        burst_duration_s=2,
        rate_hz=30,
        start=start,
        seed=9,
    )
    poisson_gold = [
        (row.event_id, row.entity, row.intended_send_ts, row.situation_id)
        for row in poisson.entries
        if row.urgent
    ]
    anomalous_gold = [
        (row.event_id, row.entity, row.intended_send_ts, row.situation_id)
        for row in anomalous.entries
        if row.urgent
    ]
    assert poisson_gold == anomalous_gold
    assert poisson_gold
    for plan in (poisson, anomalous):
        for row in plan.entries:
            if row.event is not None:
                payload = row.event.model_dump()
                assert "e6_urgent" not in str(payload)
                assert "situation_id" not in str(payload)


def test_corpus_meta_situations_override_default_catalogue() -> None:
    fit = _fit()
    catalogue = situations_from_meta(
        fit,
        {
            "injected_situations": [
                {
                    "situation_id": "meta-1",
                    "entity": fit.entities[2],
                    "onset_fraction": 0.4,
                    "archetype": "security_report",
                    "cost_weight": 7,
                    "pulses": 2,
                }
            ]
        },
    )
    assert len(catalogue) == 1
    assert catalogue[0].situation_id == "meta-1"
    assert catalogue[0].cost_weight == 7


def test_declared_shape_invariants_quantitatively() -> None:
    fit = _fit()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    steady = generate_steady_schedule(
        fit, duration_s=6, rate_hz=100, start=start, seed=12
    )
    assert len(steady) == 600

    benign = build_schedule(
        fit,
        shape="benign_flash",
        duration_s=6,
        burst_start_s=2,
        burst_duration_s=2,
        rate_hz=100,
        start=start,
        seed=12,
    )
    ordinary = [row for row in benign.entries if not row.urgent and row.event is not None]
    counts = Counter(
        "burst" if 2 <= (row.intended_send_ts - start).total_seconds() < 4 else "base"
        for row in ordinary
    )
    assert counts["burst"] / 2 == pytest.approx(5 * counts["base"] / 4, rel=0.02)
    base_types = Counter(
        row.event.type
        for row in ordinary
        if not 2 <= (row.intended_send_ts - start).total_seconds() < 4
    )
    burst_types = Counter(
        row.event.type
        for row in ordinary
        if 2 <= (row.intended_send_ts - start).total_seconds() < 4
    )
    base_total = sum(base_types.values())
    burst_total = sum(burst_types.values())
    type_l1 = sum(
        abs(base_types[name] / base_total - burst_types[name] / burst_total)
        for name in set(base_types) | set(burst_types)
    )
    assert type_l1 < 0.12

    anomalous = build_schedule(
        fit,
        shape="anomalous_burst",
        duration_s=6,
        burst_start_s=2,
        burst_duration_s=2,
        rate_hz=500,
        start=start,
        seed=12,
    )
    shifted_type = min(fit.type_mix, key=fit.type_mix.get)  # type: ignore[arg-type]
    by_entity: dict[str, list[int]] = {}
    for row in anomalous.entries:
        if row.urgent or row.event is None:
            continue
        stats = by_entity.setdefault(row.entity, [0, 0, 0, 0])
        in_burst = 2 <= (row.intended_send_ts - start).total_seconds() < 4
        stats[0 if in_burst else 2] += row.event.type == shifted_type
        stats[1 if in_burst else 3] += 1
    shifted_entities = {
        entity
        for entity, (burst_shift, burst_n, base_shift, base_n) in by_entity.items()
        if burst_shift / max(burst_n, 1) - base_shift / max(base_n, 1) > 0.5
    }
    assert shifted_entities == set(fit.entities[:3])

    poisson = build_schedule(
        fit, shape="poisson", duration_s=30, rate_hz=200, start=start, seed=5
    )
    assert abs(schedule_burstiness(poisson.entries)) < 0.12

    zipf = build_schedule(
        fit,
        shape="zipf_hot",
        duration_s=6,
        burst_start_s=2,
        burst_duration_s=2,
        rate_hz=100,
        start=start,
        seed=12,
    )
    hot = set(fit.entities[:3])
    burst_rows = [
        row
        for row in zipf.entries
        if not row.urgent and 2 <= (row.intended_send_ts - start).total_seconds() < 4
    ]
    assert sum(row.entity in hot for row in burst_rows) / len(burst_rows) > 0.8
