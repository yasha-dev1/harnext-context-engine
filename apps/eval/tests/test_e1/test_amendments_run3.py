"""Run-3 amendments after the independent E1 review (apps/eval/STATUS/REVIEW-E1-KAFKA.md)."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from harnext_eval.config import load_config
from harnext_eval.corpus.jira import parse_issue
from harnext_eval.e1.policies import (
    GUARDED_POLICIES,
    RULE_EXEMPT_POLICIES,
    GuardedHBOSPolicy,
    RuleSettings,
    RulesOnlyPolicy,
    budgeted_decisions,
    make_policy,
)
from harnext_eval.e1.score import label_situations, swap_labels
from harnext_eval.types import EvalEvent


def _event(event_id: str, when: datetime, *, subject: str = "issue:KAFKA-1", text: str = "") -> EvalEvent:
    return EvalEvent(
        id=event_id,
        source="jira:KAFKA",
        type="org.apache.jira.issue.comment",
        subject=subject,
        time=when,
        mgtenant="test",
        baseline_keys=["component:a"],
        data={"body": text},
    )


def test_rule_dedup_fires_once_per_subject_and_is_idempotent_per_event() -> None:
    at = datetime(2026, 3, 1, tzinfo=UTC)
    policy = RulesOnlyPolicy(rules=RuleSettings(dedup_per_subject=True))
    first = _event("a", at, text="Fix CVE-2024-0001 in jline")
    second = _event("b", at + timedelta(hours=1), text="CVE-2024-0001 still open")
    other = _event("c", at + timedelta(hours=2), subject="issue:KAFKA-2", text="cve in deps")
    assert policy.rules(first) == "cve"
    assert policy.rules(first) == "cve"  # repeated lookups never flip the verdict
    assert policy.rules(second) is None
    assert policy.rules(other) == "cve"
    plain = RulesOnlyPolicy()
    assert plain.rules(second) == "cve"


def test_rule_exempt_admissions_sit_outside_the_capacity() -> None:
    event_ids = [f"e-{index:02d}" for index in range(100)]
    scores = np.linspace(0, 1, 100)
    mandatory = [index < 15 for index in range(100)]
    exempt = budgeted_decisions(
        event_ids,
        scores,
        budget_pct=10,
        tuning_scores=[0.5],
        eligible=[True] * 100,
        mandatory=mandatory,
        exempt=True,
    )
    assert exempt["admitted"].sum() == 25  # 15 rule hits + 10 deviation slots
    assert exempt.loc[~np.asarray(mandatory), "admitted"].sum() == 10
    assert exempt["rules_outside_budget"].iloc[0] == 15
    assert exempt["budget_feasible"].iloc[0]
    assert exempt["rules_over_budget"].iloc[0] == 0
    shared = budgeted_decisions(
        event_ids, scores, budget_pct=10, tuning_scores=[0.5], eligible=[True] * 100, mandatory=mandatory
    )
    assert shared["admitted"].sum() == 10
    assert shared["rules_outside_budget"].iloc[0] == 0


def test_r8_and_r9_are_guarded_variants_with_registered_names() -> None:
    cfg = load_config("apps/eval/configs/e1-kafka.yaml").engine
    assert GUARDED_POLICIES == {"R5", "R8", "R9"}
    assert RULE_EXEMPT_POLICIES == {"R8", "R9"}
    r8 = make_policy("R8", cfg.router, seed=1, budget_pct=2.0)
    r9 = make_policy("R9", cfg.router, seed=1, budget_pct=2.0)
    assert isinstance(r8, GuardedHBOSPolicy) and r8.name == "R8" and not r8.any_key
    assert isinstance(r9, GuardedHBOSPolicy) and r9.name == "R9" and r9.any_key
    assert r8.rule_settings.dedup_per_subject and r8.rule_settings.vote_thread_start_only


def test_any_key_eligibility_uses_a_lower_scoring_key_that_passes_the_guards() -> None:
    policy = GuardedHBOSPolicy(any_key=True, multi_window=False)
    start = datetime(2026, 2, 1, tzinfo=UTC)

    def multi(event_id: str, when: datetime) -> EvalEvent:
        return EvalEvent(
            id=event_id, source="test:stream", type="org.test.event", subject="entity:1", time=when,
            mgtenant="test", baseline_keys=["component:busy", "contributor:quiet"], data={},
        )

    def quiet_only(event_id: str, when: datetime) -> EvalEvent:
        return EvalEvent(
            id=event_id, source="test:stream", type="org.test.event", subject="entity:2", time=when,
            mgtenant="test", baseline_keys=["contributor:quiet"], data={},
        )

    history = [multi(f"h-{index}", start + timedelta(minutes=7 * index)) for index in range(40)]
    history += [quiet_only(f"q-{index}", start + timedelta(minutes=11 * index)) for index in range(40)]
    policy.fit(sorted(history, key=lambda event: event.time))
    policy.threshold = -1e9  # everything anomalous; the guards decide eligibility
    later = start + timedelta(days=2)
    for index in range(3):
        policy.score(multi(f"burst-{index}", later + timedelta(seconds=index)))
    fired = policy.features_fired
    assert fired["eligible"]
    assert fired["volume_guard"]


def test_swap_labels_preserves_prevalence_where_flip_does_not() -> None:
    labels = np.zeros(1000)
    labels[:20] = 1.0
    swapped = swap_labels(labels, fraction=0.5, seed=3)
    assert swapped.sum() == 20
    assert int((swapped[:20] == 0).sum()) == 10


def test_label_situations_merge_consecutive_positives_per_subject() -> None:
    base = datetime(2026, 4, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "event_id": ["a1", "a2", "a3", "b1", "n1"],
            "subject": ["issue:A", "issue:A", "issue:A", "issue:B", "issue:A"],
            "t": [base, base + timedelta(hours=2), base + timedelta(days=3), base + timedelta(hours=1), base + timedelta(hours=3)],
            "label": [1.0, 1.0, 1.0, 0.9, 0.0],
        }
    )
    situations = label_situations(frame, gap_hours=24)
    assert len(situations) == 3
    first = situations[situations["event_id"] == "a1"].iloc[0]
    assert first["n_events"] == 2
    assert pd.Timestamp(first["end"]) == pd.Timestamp(base + timedelta(hours=2))
    assert set(situations["entity"]) == {"issue:A", "issue:B"}
    assert label_situations(frame.iloc[:0]).empty


def test_jira_components_are_replayed_as_of_each_event() -> None:
    issue = {
        "key": "KAFKA-42",
        "id": "42",
        "fields": {
            "created": "2026-01-01T00:00:00.000+0000",
            "summary": "s",
            "description": "d",
            "status": {"name": "Open"},
            "priority": {"name": "Major"},
            "components": [{"name": "streams"}],  # export-time snapshot (after the change)
            "fixVersions": [],
            "creator": {"emailAddress": "a@example.org", "displayName": "A"},
            "comment": {
                "comments": [
                    {"id": "c1", "created": "2026-01-02T00:00:00.000+0000", "body": "before",
                     "author": {"emailAddress": "b@example.org"}},
                    {"id": "c2", "created": "2026-01-04T00:00:00.000+0000", "body": "after",
                     "author": {"emailAddress": "b@example.org"}},
                ]
            },
        },
        "changelog": {
            "histories": [
                {
                    "id": "h1",
                    "created": "2026-01-03T00:00:00.000+0000",
                    "author": {"emailAddress": "c@example.org"},
                    "items": [{"field": "components", "from": None, "fromString": "core", "to": None, "toString": "streams"}],
                }
            ]
        },
    }
    events = {event.id: event for event in parse_issue(issue)}
    assert events["jira:42:created"].data["components"] == ["core"]
    assert "component:core" in events["jira:42:created"].baseline_keys
    assert "component:streams" not in events["jira:42:created"].baseline_keys
    assert events["jira:42:comment:c1"].data["components"] == ["core"]
    assert events["jira:42:change:h1:0"].data["components"] == ["streams"]
    assert events["jira:42:comment:c2"].data["components"] == ["streams"]
    assert "component:streams" in events["jira:42:comment:c2"].baseline_keys
