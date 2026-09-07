"""Run-3 amendments after the independent E1 review (apps/eval/STATUS/REVIEW-E1-KAFKA.md)."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from harnext_eval.config import load_config
from harnext_eval.corpus.jira import parse_issue
from harnext_eval.e1.policies import (
    GUARDED_POLICIES,
    RULE_EXEMPT_POLICIES,
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


def test_run4_policy_set_registers_per_source_and_global_variants() -> None:
    from harnext_eval.e1.policies import GLOBAL_VARIANTS, GlobalPolicy, PerSourceGlobalPolicy
    from harnext_eval.e1.prereg import POLICIES, PRIMARY_CONTRASTS

    cfg = load_config("apps/eval/configs/e1-kafka.yaml").engine
    assert GUARDED_POLICIES == {"R5"}
    assert RULE_EXEMPT_POLICIES == frozenset()
    assert POLICIES == ("R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R10", "R11", "R12", "R13")
    assert GLOBAL_VARIANTS == {"R11": "iforest", "R12": "ecod", "R13": "lof"}
    assert all(contrast.split("-")[0] in POLICIES and contrast.split("-")[1] in POLICIES for contrast in PRIMARY_CONTRASTS)
    for name in ("R8", "R9"):
        import pytest

        with pytest.raises(ValueError, match="unknown E1 policy"):
            make_policy(name, cfg.router, seed=1)
    r10 = make_policy("R10", cfg.router, seed=1)
    assert isinstance(r10, PerSourceGlobalPolicy) and r10.name == "R10" and r10.global_features
    for name, method in GLOBAL_VARIANTS.items():
        policy = make_policy(name, cfg.router, seed=1)
        assert isinstance(policy, GlobalPolicy) and policy.name == name and policy.method == method


def _source_event(event_id: str, when: datetime, source: str, value: float) -> EvalEvent:
    return EvalEvent(
        id=event_id, source=f"{source}:x", type=f"org.{source}.event", subject=f"{source}:{event_id}",
        time=when, mgtenant="test", baseline_keys=[], data={"amount": value},
    )


def test_global_variants_and_per_source_percentiles_score_deterministically() -> None:
    from harnext_eval.e1.policies import GLOBAL_VARIANTS, GlobalPolicy, PerSourceGlobalPolicy

    start = datetime(2026, 1, 1, tzinfo=UTC)
    history = [
        _source_event(f"{source}-{index}", start + timedelta(minutes=index * (3 if source == "jira" else 17)), source, index % 5)
        for source in ("jira", "github") for index in range(60)
    ]
    history.sort(key=lambda event: event.time)
    probe = history[-1].model_copy(update={"id": "probe", "time": start + timedelta(days=2)})
    for method in GLOBAL_VARIANTS.values():
        first = GlobalPolicy(method=method, seed=3).fit(history)
        second = GlobalPolicy(method=method, seed=3).fit(history)
        assert first.score(probe) == second.score(probe)
        assert first.features_fired["scorer"] == f"global_{method}"
    per_source = PerSourceGlobalPolicy(seed=3).fit(history)
    assert set(per_source.source_models) == {"jira", "github"}
    value = per_source.score(probe)
    assert 0.0 <= value <= 1.0
    assert per_source.features_fired["source_model"] is True
    unseen = per_source.score(_source_event("m", start + timedelta(days=2), "mail", 1.0))
    assert per_source.features_fired["source_model"] is False and unseen >= 0.0


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
