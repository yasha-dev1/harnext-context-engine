"""Discriminating policy and admission tests for docs/evaluation-spec.md §7 E1."""

from datetime import UTC, datetime, timedelta

import numpy as np
from harnext_eval.config import load_config
from harnext_eval.corpus.synthetic import generate_synthetic_events
from harnext_eval.e1.policies import (
    GlobalPolicy,
    GuardedHBOSPolicy,
    RouterPolicy,
    RuleSettings,
    budgeted_decisions,
    make_policy,
    match_rule,
)
from harnext_eval.types import EvalEvent


def _event(
    event_id: str,
    when: datetime,
    *,
    data: dict[str, object] | None = None,
    keys: list[str] | None = None,
) -> EvalEvent:
    return EvalEvent(
        id=event_id,
        source="test:stream",
        type="org.test.event",
        subject="entity:1",
        time=when,
        mgtenant="test",
        baseline_keys=keys or ["component:a"],
        data=data or {},
    )


def test_r1_is_exact_field_and_boundary_aware_with_configurable_dispute_floor() -> None:
    at = datetime(2026, 1, 1, tzinfo=UTC)
    settings = RuleSettings(dispute_amount=5_000)
    assert match_rule(_event("priority", at, data={"priority": "Critical"}), settings) == "declared_priority"
    assert match_rule(_event("vote", at, data={"subject": "[VOTE] KIP-1"}), settings) == "vote"
    assert match_rule(_event("cve", at, data={"body": "CVE-2026-1000"}), settings) == "cve"
    assert match_rule(_event("word", at, data={"body": "this is a blocker"}), settings) == "blocker_word"
    assert match_rule(_event("page", at, data={"provider": "PagerDuty"}), settings) == "on_call_page"
    assert match_rule(_event("money", at, data={"dispute": {"amount": 5_001}}), settings) == "large_dispute"
    assert match_rule(_event("small", at, data={"dispute": {"amount": 4_999}}), settings) is None
    assert match_rule(_event("noncritical", at, data={"body": "noncritical path"}), settings) is None
    assert match_rule(_event("plural", at, data={"body": "blockers discussed"}), settings) is None


def test_r2_has_one_global_stream_and_never_reports_an_entity_key() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    training = [
        _event(f"train-{index}", start + timedelta(minutes=index), keys=[f"component:{index % 3}"])
        for index in range(20)
    ]
    policy = GlobalPolicy(method="z").fit(training)
    assert policy.score(_event("test", start + timedelta(hours=1), keys=["a", "b", "c"])) >= 0
    assert policy.baseline_key_used == "__global__"
    assert policy.features_fired["scorer"] == "global_z"


def test_r5_requires_absolute_volume_and_two_distinct_consecutive_windows() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    policy = GuardedHBOSPolicy(absolute_floor=2, multi_window=True, budget_pct=10)
    policy.threshold = 5.0
    policy._score_vector = lambda vector: 10.0  # type: ignore[method-assign]

    first = policy.score(_event("w1-1", start + timedelta(seconds=1)))
    assert first == 10.0
    assert not policy.features_fired["volume_guard"]
    assert not policy.features_fired["eligible"]
    policy.score(_event("w1-2", start + timedelta(seconds=2)))
    assert policy.features_fired["volume_guard"]
    assert not policy.features_fired["multi_window_confirmed"]

    policy.score(_event("w2-1", start + timedelta(minutes=5, seconds=1)))
    assert policy.features_fired["multi_window_confirmed"]
    assert not policy.features_fired["eligible"]
    policy.score(_event("w2-2", start + timedelta(minutes=5, seconds=2)))
    assert policy.features_fired["eligible"]

    policy.threshold = 100.0
    policy.score(_event("gap", start + timedelta(minutes=10, seconds=1)))
    policy.threshold = 5.0
    policy.score(_event("w4-1", start + timedelta(minutes=15, seconds=1)))
    policy.score(_event("w4-2", start + timedelta(minutes=15, seconds=2)))
    assert not policy.features_fired["multi_window_confirmed"]
    assert not policy.features_fired["eligible"]


def test_monthly_budget_is_global_and_declares_an_overfull_rule_floor_infeasible() -> None:
    event_ids = [f"a-{index:02d}" for index in range(50)] + [f"b-{index:02d}" for index in range(50)]
    scores = np.linspace(0, 1, len(event_ids))
    selected = budgeted_decisions(
        event_ids,
        scores,
        budget_pct=10,
        tuning_scores=np.linspace(-2, 2, 101),
    )
    assert selected["admitted"].sum() == 10
    assert set(selected.loc[selected["admitted"], "event_id"]) == set(event_ids[-10:])

    guarded = budgeted_decisions(
        event_ids,
        scores,
        budget_pct=10,
        tuning_scores=[0.5],
        eligible=[index < 2 for index in range(100)],
    )
    assert guarded["admitted"].sum() == 2
    assert guarded["unused_capacity"].iloc[0] == 8

    mandatory = [index < 15 for index in range(100)]
    floor = budgeted_decisions(
        event_ids,
        scores,
        budget_pct=10,
        tuning_scores=[0.5],
        eligible=[False] * 100,
        mandatory=mandatory,
    )
    assert floor["admitted"].sum() == 10
    assert floor.loc[~np.asarray(mandatory), "admitted"].sum() == 0
    assert floor["rules_over_budget"].iloc[0] == 5
    assert not floor["budget_feasible"].iloc[0]


def test_every_policy_runs_deterministically_and_exposes_distinct_semantics() -> None:
    events = generate_synthetic_events(seed=3, event_count=240, days=8, entity_count=8)
    cfg = load_config("apps/eval/configs/baseline-minimal.yaml").engine
    repeated: dict[str, list[float]] = {}
    for name in (f"R{index}" for index in range(8)):
        policy = make_policy(name, cfg.router, seed=11).fit(events[:180])
        assert isinstance(policy, RouterPolicy)
        repeated[name] = [policy.score(event) for event in events[180:]]
        second = make_policy(name, cfg.router, seed=11).fit(events[:180])
        assert repeated[name] == [second.score(event) for event in events[180:]]
    assert len(set(repeated["R0"])) > 1
    assert set(repeated["R1"]) <= {0.0, 1.0}
    assert set(repeated["R7"]) == {1.0}
    assert repeated["R2"] != repeated["R4"]
