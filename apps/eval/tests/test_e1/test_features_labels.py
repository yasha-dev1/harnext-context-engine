"""Temporal-firewall regressions for docs/evaluation-spec.md §4.1 and §7 E1."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from harnext_eval.corpus.synthetic import generate_synthetic_events
from harnext_eval.e1.features import CausalFeatureExtractor, extract_features
from harnext_eval.e1.labels import (
    ABSTAIN,
    DEFAULT_LABELING_FUNCTIONS,
    NEGATIVE,
    POSITIVE,
    apply_labeling_functions,
    build_labels,
)
from harnext_eval.types import EvalEvent


def _event(
    event_id: str,
    when: datetime,
    *,
    source: str = "jira:test",
    event_type: str = "org.apache.jira.issue.created",
    subject: str = "issue:KAFKA-1",
    data: dict[str, object] | None = None,
) -> EvalEvent:
    return EvalEvent(
        id=event_id,
        source=source,
        type=event_type,
        subject=subject,
        time=when,
        mgtenant="test",
        baseline_keys=["component:test"],
        data=data or {},
    )


def _lf_cases() -> list[tuple[str, EvalEvent, list[EvalEvent], datetime]]:
    start = datetime(2026, 1, 1, 12, tzinfo=UTC)
    jira = _event("jira-candidate", start)
    dev = _event(
        "dev-candidate",
        start,
        source="dev@kafka.apache.org",
        event_type="org.apache.mail.message",
        subject="thread:root",
        data={"message_id": "root"},
    )
    github = _event(
        "github-candidate",
        start,
        source="github:apache/kafka",
        event_type="com.github.pull_request.opened",
        subject="pr:10",
        data={"number": 10},
    )
    ci = github.model_copy(
        update={
            "id": "ci-candidate",
            "type": "com.github.check.failed",
            "data": {"number": 10, "branch": "main", "conclusion": "failed"},
        }
    )
    return [
        (
            "jira_committer_comment_1h",
            jira,
            [_event("jira-comment", start + timedelta(minutes=30), event_type="org.apache.jira.issue.comment", data={"author": "alice", "role": "committer"})],
            start + timedelta(days=2),
        ),
        (
            "jira_priority_raised_later",
            jira,
            [_event("jira-priority", start + timedelta(days=45), event_type="org.apache.jira.issue.transition", data={"field": "priority", "from": "Major", "to": "Critical"})],
            start + timedelta(days=46),
        ),
        (
            "jira_fix_version_in_flight_later",
            jira,
            [_event("jira-fix-version", start + timedelta(days=2), event_type="org.apache.jira.issue.transition", data={"field": "fixVersion", "to": "3.8.0", "current_release": "3.8.0"})],
            start + timedelta(days=3),
        ),
        (
            "jira_resolved_24h",
            jira,
            [_event("jira-resolved", start + timedelta(hours=2), event_type="org.apache.jira.issue.transition", data={"field": "status", "from": "Open", "to": "Resolved"})],
            start + timedelta(days=2),
        ),
        (
            "jira_linked_pr_24h",
            jira,
            [_event("linked-pr", start + timedelta(hours=3), source="github:apache/kafka", event_type="com.github.pull_request.opened", subject="pr:11", data={"title": "Fix KAFKA-1", "action": "opened"})],
            start + timedelta(days=2),
        ),
        (
            "declared_blocker_critical",
            jira,
            [_event("jira-declared", start + timedelta(days=2), event_type="org.apache.jira.issue.transition", data={"field": "priority", "to": "Blocker"})],
            start + timedelta(days=3),
        ),
        (
            "dev_committer_reply_1h",
            dev,
            [_event("dev-reply", start + timedelta(minutes=20), source="dev@kafka.apache.org", event_type="org.apache.mail.message", subject="thread:root", data={"author": "alice", "role": "committer", "in_reply_to": "root"})],
            start + timedelta(days=2),
        ),
        (
            "dev_three_responders_2h",
            dev,
            [_event(f"dev-{index}", start + timedelta(minutes=10 * index), source="dev@kafka.apache.org", event_type="org.apache.mail.message", subject="thread:root", data={"author": actor, "in_reply_to": "root"}) for index, actor in enumerate(("alice", "bob", "carol"), start=1)],
            start + timedelta(days=2),
        ),
        (
            "dev_vote_cancelled_recast_later",
            dev,
            [
                _event("dev-cancel", start + timedelta(hours=3), source="dev@kafka.apache.org", event_type="org.apache.mail.message", subject="thread:root", data={"body": "vote cancelled", "in_reply_to": "root"}),
                _event("dev-recast", start + timedelta(hours=4), source="dev@kafka.apache.org", event_type="org.apache.mail.message", subject="thread:root", data={"subject": "[VOTE] re-cast", "in_reply_to": "root"}),
            ],
            start + timedelta(days=2),
        ),
        (
            "dev_cve_blocker_later",
            dev,
            [_event("dev-cve", start + timedelta(hours=3), source="dev@kafka.apache.org", event_type="org.apache.mail.message", subject="thread:root", data={"body": "CVE-2026-1234 is a blocker", "in_reply_to": "root"})],
            start + timedelta(days=2),
        ),
        (
            "github_reverted_48h",
            github,
            [_event("gh-revert", start + timedelta(hours=5), source="github:apache/kafka", event_type="com.github.push", subject="commit:abc", data={"message": "Revert #10"})],
            start + timedelta(days=3),
        ),
        (
            "github_hotfix_reference_24h",
            github,
            [_event("gh-hotfix", start + timedelta(hours=5), source="github:apache/kafka", event_type="com.github.pull_request.opened", subject="pr:12", data={"title": "hotfix for #10"})],
            start + timedelta(days=2),
        ),
        (
            "github_trunk_ci_failure_fix_6h",
            ci,
            [_event("gh-fix", start + timedelta(hours=2), source="github:apache/kafka", event_type="com.github.push", subject="commit:def", data={"message": "fix #10", "branch": "main"})],
            start + timedelta(days=2),
        ),
    ]


@pytest.mark.parametrize("name,candidate,outcomes,observation_end", _lf_cases())
def test_every_spec_label_function_has_positive_negative_and_wrong_source(
    name: str,
    candidate: EvalEvent,
    outcomes: list[EvalEvent],
    observation_end: datetime,
) -> None:
    positive = apply_labeling_functions(
        [candidate, *outcomes], observation_end=observation_end
    )
    assert positive.loc[candidate.id, name] == POSITIVE

    negative = apply_labeling_functions([candidate], observation_end=observation_end)
    assert negative.loc[candidate.id, name] == NEGATIVE

    wrong_source = candidate.model_copy(
        update={"id": f"wrong-{candidate.id}", "source": "other:test", "type": "org.other"}
    )
    wrong = apply_labeling_functions(
        [wrong_source, *outcomes], observation_end=observation_end
    )
    assert wrong.loc[wrong_source.id, name] == ABSTAIN


def test_finite_horizon_is_unknown_until_fully_observed() -> None:
    start = datetime(2026, 1, 1, 12, tzinfo=UTC)
    candidate = _event("candidate", start)
    votes = apply_labeling_functions(
        [candidate], observation_end=start + timedelta(hours=23)
    )
    assert votes.loc["candidate", "jira_resolved_24h"] == ABSTAIN
    observability = votes.attrs["observability"]
    assert not observability.loc["candidate", "jira_resolved_24h"]

    observed = apply_labeling_functions(
        [candidate], observation_end=start + timedelta(hours=24)
    )
    assert observed.loc["candidate", "jira_resolved_24h"] == NEGATIVE


def test_label_model_reports_votes_coverage_conflicts_and_declared_agreement() -> None:
    candidate, outcomes, end = _lf_cases()[0][1:]
    declared = _event(
        "declared",
        candidate.time + timedelta(hours=2),
        event_type="org.apache.jira.issue.transition",
        data={"field": "priority", "to": "Critical"},
    )
    result = build_labels([candidate, *outcomes, declared], observation_end=end)
    assert {
        "accuracy",
        "coverage",
        "conflict",
        "positive_votes",
        "negative_votes",
        "unknown",
    } <= set(result.diagnostics.columns)
    assert result.declared_outcome_comparable > 0
    assert np.isfinite(result.declared_outcome_agreement)


def test_features_are_deterministic_causal_and_use_independent_count_baselines() -> None:
    events = generate_synthetic_events(seed=9, event_count=120, days=3, entity_count=8)
    full = extract_features(events)
    repeated = extract_features(events)
    pd.testing.assert_frame_equal(full, repeated)
    prefix = extract_features(events[:67])
    expected = full[full["event_id"].isin([event.id for event in events[:67]])].reset_index(drop=True)
    pd.testing.assert_frame_equal(prefix.reset_index(drop=True), expected)

    start = datetime(2026, 1, 1, tzinfo=UTC)
    extractor = CausalFeatureExtractor()
    vector = None
    for index in range(30):
        vector = extractor.update(_event(f"bucket-{index}", start + timedelta(minutes=5 * index)))[0]
    assert vector is not None
    assert vector.values["count_5m_ratio"] == vector.context["count_5m"] / max(
        vector.context["baseline_median_5m"], 1.0
    )
    assert vector.values["count_1h_ratio"] == vector.context["count_1h"] / max(
        vector.context["baseline_median_1h"], 1.0
    )
    assert {"robust_z_5m", "robust_z_1h", "baseline_mad_1h"} <= set(vector.context)


def test_all_spec_functions_are_registered_once() -> None:
    assert len(DEFAULT_LABELING_FUNCTIONS) == 13
    assert len({function.name for function in DEFAULT_LABELING_FUNCTIONS}) == 13
