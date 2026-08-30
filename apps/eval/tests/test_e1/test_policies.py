"""Policy and budget tests for docs/evaluation-spec.md §7 E1."""

import numpy as np
from harnext_eval.config import load_config
from harnext_eval.corpus.synthetic import generate_synthetic_events
from harnext_eval.e1.policies import RouterPolicy, budgeted_decisions, make_policy


def test_every_preregistered_policy_runs() -> None:
    events = generate_synthetic_events(seed=3, event_count=240, days=8, entity_count=8)
    cfg = load_config("apps/eval/configs/baseline-minimal.yaml").engine
    for name in (f"R{index}" for index in range(8)):
        policy = make_policy(name, cfg.router, seed=11).fit(events[:180])
        assert isinstance(policy, RouterPolicy)
        scores = [policy.score(event) for event in events[180:]]
        assert len(scores) == 60
        assert np.isfinite(scores).all()


def test_budget_capacity_is_exact_and_theta_uses_tuning_scores() -> None:
    event_ids = [f"event-{index:03d}" for index in range(1_000)]
    scores = np.linspace(0, 1, len(event_ids))
    tuning = np.linspace(-4, 4, 501)
    selected = budgeted_decisions(
        event_ids, scores.tolist(), budget_pct=2.0, tuning_scores=tuning.tolist()
    )
    assert selected["admitted"].mean() == 0.02
    assert selected["admitted"].sum() == 20
    assert np.isclose(selected["theta"].iloc[0], np.quantile(tuning, 0.98))
    assert set(selected.loc[selected["admitted"], "event_id"]) == set(event_ids[-20:])
