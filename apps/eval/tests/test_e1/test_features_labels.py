"""Temporal-firewall tests for docs/evaluation-spec.md §4.1 and §7 E1."""

from datetime import UTC, datetime, timedelta

import pandas as pd
from harnext_eval.corpus.synthetic import generate_synthetic_events
from harnext_eval.e1.features import extract_features
from harnext_eval.e1.labels import ABSTAIN, POSITIVE, apply_labeling_functions, build_labels
from harnext_eval.types import EvalEvent


def _event(
    event_id: str,
    when: datetime,
    *,
    event_type: str = "org.test.issue",
    data: dict[str, object] | None = None,
) -> EvalEvent:
    return EvalEvent(
        id=event_id,
        source="jira:test",
        type=event_type,
        subject="issue:X-1",
        time=when,
        mgtenant="test",
        baseline_keys=["component:test"],
        data=data or {},
    )


def test_features_are_deterministic_and_causal_under_truncation() -> None:
    events = generate_synthetic_events(seed=9, event_count=120, days=3, entity_count=8)
    full = extract_features(events)
    repeated = extract_features(events)
    pd.testing.assert_frame_equal(full, repeated)

    cutoff = 67
    prefix = extract_features(events[:cutoff])
    expected = full[full["event_id"].isin([event.id for event in events[:cutoff]])].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(prefix.reset_index(drop=True), expected)

    # An arbitrarily urgent-looking post-t payload cannot alter an earlier row.
    events[-1].data = {"priority": "Blocker", "amount": 10_000_000, "actor": "new"}
    mutated = extract_features(events)
    prior_id = events[cutoff - 1].id
    pd.testing.assert_frame_equal(
        full[full["event_id"] == prior_id].reset_index(drop=True),
        mutated[mutated["event_id"] == prior_id].reset_index(drop=True),
    )


def test_labels_change_when_strict_post_t_outcome_is_removed() -> None:
    start = datetime(2026, 1, 1, 12, tzinfo=UTC)
    candidate = _event("candidate", start)
    resolution = _event(
        "resolution",
        start + timedelta(hours=2),
        event_type="org.test.issue.transition",
        data={"field": "status", "from": "Open", "to": "Resolved"},
    )
    with_outcome = apply_labeling_functions([candidate, resolution])
    without_outcome = apply_labeling_functions([candidate])
    assert with_outcome.loc["candidate", "resolved_24h"] == POSITIVE
    assert without_outcome.loc["candidate", "resolved_24h"] == ABSTAIN

    fused = build_labels([candidate, resolution])
    truncated = build_labels([candidate])
    assert fused.probabilities.loc["candidate"] != truncated.probabilities.loc["candidate"]
    assert {"accuracy", "coverage", "conflict"} <= set(fused.diagnostics.columns)


def test_event_at_t_is_never_visible_to_its_own_label_functions() -> None:
    start = datetime(2026, 1, 1, 12, tzinfo=UTC)
    declared_now = _event("now", start, data={"priority": "Blocker"})
    votes = apply_labeling_functions([declared_now])
    assert (votes.loc["now"] == ABSTAIN).all()
