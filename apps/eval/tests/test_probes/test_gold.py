"""Dual-gold validity tests for docs/evaluation-spec.md §4 and §7 E2."""

from __future__ import annotations

from harnext_eval.probes.gold import (
    GoldRequest,
    PythonGold,
    cross_check_field_value,
    cross_check_gold,
    field_value_python,
    field_value_sql,
)
from harnext_eval.types import EvalEvent


def test_python_and_sql_gold_agree_on_synthetic_transitions(
    synthetic_events: list[EvalEvent],
) -> None:
    python = PythonGold(synthetic_events)
    requests = [
        GoldRequest(
            entity=history[-1].entity,
            field=history[-1].field,
            T=transition.time,
        )
        for history in python.histories().values()
        for transition in history
    ]

    assert requests
    assert cross_check_gold(synthetic_events, requests) == []

    example = requests[len(requests) // 2]
    expected = field_value_python(
        synthetic_events, example.entity, example.field, example.T
    )
    assert expected is not None
    assert field_value_sql(synthetic_events, example.entity, example.field, example.T) == expected
    assert (
        cross_check_field_value(
            synthetic_events, example.entity, example.field, example.T
        )
        is None
    )
