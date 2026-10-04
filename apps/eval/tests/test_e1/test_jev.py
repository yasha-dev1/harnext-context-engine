"""Run-5 Jev routing arms (R14/R15): event view, decision parsing, cache, and the E1 runner."""

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from harnext_eval.config import load_config
from harnext_eval.corpus.synthetic import generate_synthetic_corpus
from harnext_eval.e1 import jev
from harnext_eval.e1.policies import JevPolicy, make_policy
from harnext_eval.e1.prereg import (
    JEV_CONTRASTS,
    JEV_POLICIES,
    POLICIES,
    contrasts_for,
    policies_for,
)
from harnext_eval.e1.run import E1Experiment, _router_events
from harnext_eval.registry import get_experiment
from harnext_eval.types import EvalEvent


def _event(
    kind: str, data: dict, *, source: str = "jira:KAFKA", event_id: str = "e1", day: int = 1
) -> EvalEvent:
    return EvalEvent(
        id=event_id, source=source, type=kind, subject="issue:KAFKA-1",
        time=datetime(2024, 1, day, tzinfo=UTC), mgtenant="kafka", baseline_keys=[], data=data,
    )


class _Jev:
    """Answers like the decisions endpoint; ``fast`` ids get a fast-lane category."""

    def __init__(self, fast=(), fail=()):
        self.fast, self.fail, self.calls = set(fast), set(fail), 0

    async def decide(self, state, questions, purpose=""):
        self.calls += 1
        text = json.dumps(state)
        if any(marker in text for marker in self.fail):
            raise ValueError("decisions answer lacks ['category']")
        hot = any(marker in text for marker in self.fast)
        choice = "regression_or_outage" if hot else "code_review"
        probabilities = dict.fromkeys(questions["category"]["criteria"], 0.0) | {choice: 0.9, "feature_work": 0.1}
        answers = {"category": {"type": "choice", "choice": choice, "probabilities": probabilities, "confidence": 0.9},
                   "urgency": {"type": "noul", "noul": 0.8 if hot else 0.1}}
        return answers, SimpleNamespace(model="typesafe/jev-1.13-test", input_tokens=900, cost_usd=4e-5, seconds=0.2,
                                        attempts=1, unmetered_usd=0.0)


def test_event_state_carries_text_but_no_identifiers_or_gold() -> None:
    event = _event("org.apache.jira.issue.created", {
        "issue_key": "KAFKA-1", "summary": "Broker crashes", "description": "NPE on startup\n\n\n" + "x" * 5000,
        "priority": "Major", "creator_name": "A", "creator_account_id": "secret-id", "is_committer": True,
    })
    state = jev.event_state(_router_events([event])[0])["event"]
    assert state["event"] == "issue created" and state["priority"] == "Major"
    assert state["reporter"] == "person", "committer status is a current snapshot: not shown"
    assert len(state["description"]) <= 1500 and state["description"].endswith("…")
    assert "secret-id" not in json.dumps(state) and "KAFKA-1" not in json.dumps(state)

    gold = _event("org.apache.jira.issue.comment", {"body": "ping", "urgent": True, "is_urgent": True})
    assert "urgent" not in json.dumps(jev.event_state(_router_events([gold])[0]))


def test_mail_view_drops_quoted_text_and_marks_automation() -> None:
    event = _event("org.apache.mail.message", {
        "list": "dev@kafka.apache.org", "subject": "Re: [VOTE] 3.7.0 RC1", "author_name": "PoAn Yang (Jira)",
        "body": "+1 binding\n\nOn Mon, Jan 1, 2024 someone wrote:\n> old text\n> more",
        "in_reply_to": "x",
    }, source="mail:dev@kafka.apache.org")
    state = jev.event_state(event)["event"]
    assert state["body"] == "+1 binding"
    assert state["author"] == "automation" and state["event"] == "reply in thread"


def test_identifiers_are_scrubbed() -> None:
    event = _event("com.github.pull_request.opened", {
        "title": "KAFKA-16310: fix ListOffsets (#15621)", "number": 15621, "pull_request_number": 15621,
        "issue_keys": ["KAFKA-16310"], "body": "See https://github.com/apache/kafka/pull/15000 and "
        "commit d12fce2b0656ee61e2bebdcd7badc67195928b78, KIP-848 too.",
    }, source="github:apache/kafka")
    text = json.dumps(jev.event_state(event))
    for leaked in ("16310", "15621", "15000", "d12fce2b", "github.com"):
        assert leaked not in text
    assert "<issue>" in text and "#<pr>" in text and "<sha>" in text


def test_jira_text_is_rebuilt_as_of_the_event_time() -> None:
    created = _event("org.apache.jira.issue.created", {
        "issue_key": "KAFKA-9", "summary": "Regression in 3.7", "description": "Later edited text: confirmed regression",
        "labels": ["regression"], "priority": "Major"}, event_id="c", day=1)
    edit = _event("org.apache.jira.issue.transition", {
        "issue_key": "KAFKA-9", "field": "description", "from": "Original text", "to": "Later edited text: confirmed regression",
        "summary": "Regression in 3.7", "description": "Later edited text: confirmed regression"}, event_id="t1", day=5)
    rename = _event("org.apache.jira.issue.transition", {
        "issue_key": "KAFKA-9", "field": "summary", "from": "Odd behaviour", "to": "Regression in 3.7"}, event_id="t2", day=6)
    labels = _event("org.apache.jira.issue.transition", {
        "issue_key": "KAFKA-9", "field": "labels", "from": "", "to": "regression"}, event_id="t3", day=7)
    comment = _event("org.apache.jira.issue.comment", {
        "issue_key": "KAFKA-9", "summary": "Regression in 3.7", "description": "Later edited text: confirmed regression",
        "body": "looking"}, event_id="m", day=3)
    as_of = jev.AsOfText.from_events([created, comment, edit, rename, labels])
    first = jev.event_state(created, as_of)["event"]
    assert first["description"] == "Original text" and first["summary"] == "Odd behaviour" and "labels" not in first
    assert jev.event_state(comment, as_of)["event"]["issue_description"] == "Original text"
    after = jev.event_state(edit, as_of)["event"]
    assert after["issue_description"].startswith("Later edited") and after["summary"] == "Odd behaviour"
    assert jev.event_state(labels, as_of)["event"]["summary"] == "Regression in 3.7"


def test_decision_scores_and_cache_precedence(tmp_path) -> None:
    sha = jev.prompt_sha256()
    answers = {"category": {"choice": "security", "probabilities": {"security": 0.6, "feature_work": 0.2, "build_break": 0.2}},
               "urgency": {"noul": 0.7}}
    decision = jev._parse("e1", sha, answers, SimpleNamespace(model="m", input_tokens=10, cost_usd=1e-6, seconds=0.1))
    assert decision.ok and decision.fast_category_p == pytest.approx(0.8)
    assert decision.score("R14") == pytest.approx(0.8 + 0.0007)
    assert decision.score("R15") == pytest.approx(0.7)
    failed = jev.JevDecision("e1", sha, error="timeout")
    assert failed.score("R14") == failed.score("R15") == -1.0

    path = tmp_path / "scores.jsonl"
    from dataclasses import asdict
    rows = [asdict(decision), asdict(failed), asdict(jev.JevDecision("e2", "other-prompt", category="security"))]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + '{"event_id": "e3", "prompt')
    summary = jev.CacheSummary()
    loaded = jev.load_decisions(path, sha, summary)
    assert set(loaded) == {"e1"} and loaded["e1"].ok, "a later failure must not replace a success"
    assert summary.bad_lines == 1 and summary.records == 2, "a truncated write is skipped, not fatal"


def test_score_events_is_resumable_and_retries_failures(tmp_path) -> None:
    events = [_event("org.apache.jira.issue.comment", {"body": f"comment {index}"}, event_id=f"e{index}") for index in range(6)]
    out = tmp_path / "cache.jsonl"
    client = _Jev(fast={"comment 1"}, fail={"comment 2"})
    stats = asyncio.run(jev.score_events(events, out, client=client, concurrency=2, progress_every=0))
    assert stats["ok"] == 5 and stats["failed"] == 1
    client.fail.clear()
    stats = asyncio.run(jev.score_events(events, out, client=client, concurrency=2, progress_every=0))
    assert stats == {**stats, "todo": 1, "skipped": 5, "ok": 1, "failed": 0}
    decisions = jev.load_decisions(out, jev.prompt_sha256())
    assert all(decision.ok for decision in decisions.values()) and len(decisions) == 6
    assert decisions["e1"].category == "regression_or_outage" and decisions["e1"].score("R14") > decisions["e0"].score("R14")


def test_registration_of_configs_without_jev_is_unchanged() -> None:
    from harnext_eval.e1.prereg import _hashed_config
    from harnext_eval.manifest import sha256_json

    config = load_config("apps/eval/configs/e1-kafka.yaml")
    payload = config.model_dump(mode="json")
    for key in ("jev_scores", "jev_prompt_sha256", "jev_registration", "policies"):
        assert payload["e1"].pop(key) is None
    assert sha256_json(_hashed_config(config)) == sha256_json(payload)
    jev_config = load_config("apps/eval/configs/e1-kafka-jev.yaml")
    assert _hashed_config(jev_config)["e1"]["jev_scores"]
    assert "policies" not in _hashed_config(jev_config)["e1"], "run-5 registration unchanged by the subset knob"


def test_stop_on_refused_credit_and_on_outage(tmp_path) -> None:
    from harnext_eval.e2.v2.engine.clients import InfraError

    events = [_event("org.apache.jira.issue.comment", {"body": f"c {index}"}, event_id=f"e{index}") for index in range(50)]

    class Refused(_Jev):
        async def decide(self, state, questions, purpose=""):
            self.calls += 1
            raise InfraError("provider error 402: credit")

    client = Refused()
    with pytest.raises(InfraError):
        asyncio.run(jev.score_events(events, tmp_path / "a.jsonl", client=client, concurrency=2, progress_every=0))
    assert client.calls <= 2

    class Down(_Jev):
        async def decide(self, state, questions, purpose=""):
            self.calls += 1
            raise InfraError("https://x: timeout (after 5 attempts)")

    stats = asyncio.run(jev.score_events(events, tmp_path / "b.jsonl", client=Down(), concurrency=2,
                                         progress_every=0, max_consecutive_failures=5))
    assert stats["stopped_early"] and stats["failed"] < len(events)


def test_jev_policies_are_registered_only_with_a_cache() -> None:
    assert policies_for(False) == POLICIES and set(JEV_POLICIES) <= set(policies_for(True))
    assert all(set(c.split("-")) <= set(policies_for(True)) for c in contrasts_for(True))
    assert set(JEV_CONTRASTS) <= set(contrasts_for(True)) and not set(JEV_CONTRASTS) & set(contrasts_for(False))
    cfg = load_config("apps/eval/configs/baseline-minimal.yaml").engine
    policy = make_policy("R14", cfg.router)
    assert isinstance(policy, JevPolicy)
    with pytest.raises(RuntimeError):
        policy.score(_event("org.apache.jira.issue.comment", {}))


def test_e1_runs_jev_arms_from_the_cache(tmp_path) -> None:
    corpus = generate_synthetic_corpus(tmp_path / "replay.jsonl", seed=1, event_count=140, days=240, entity_count=10)
    events = _router_events(sorted(corpus.events(), key=lambda event: (event.time, event.id)))
    positives = {item["event_id"] for item in corpus.meta["injected_situations"]}
    cache = tmp_path / "jev.jsonl"
    client = _Jev()
    original = jev.event_state
    asyncio.run(jev.score_events(events, cache, client=client, concurrency=4, progress_every=0))
    # Mark the injected situations fast by rewriting their cached category (the fake sees no gold).
    rows = [json.loads(line) for line in cache.read_text().splitlines()]
    for row in rows:
        if row["event_id"] in positives:
            row["category"] = "security"
            row["probabilities"] = {"security": 1.0}
    cache.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert jev.event_state is original

    corpus = replace(corpus, meta={**corpus.meta, "smoke": True, "e1_jev_scores": str(cache),
                                   "e1_jev_prompt_sha256": jev.prompt_sha256()})
    cfg = load_config("apps/eval/configs/baseline-minimal.yaml").engine
    experiment = get_experiment("e1")
    assert isinstance(experiment, E1Experiment)
    result = experiment.run(cfg, corpus, tmp_path / "e1", seed=1)
    scores = result.tables["scores"]
    assert set(scores["policy"]) == set(policies_for(True))
    assert result.primary["by_policy"]["R14"] >= result.primary["by_policy"]["R0"]
    assert "r14_minus_r2" in result.primary
    jev_rows = scores[scores["policy"] == "R14"]
    assert (jev_rows["routing_dollars"] > 0).all() and (jev_rows["routing_tokens"] == 900).all()

    subset = ["R0", "R1", "R2", "R7", "R14", "R15"]
    only = replace(corpus, meta={**corpus.meta, "e1_policies": subset})
    narrow = experiment.run(cfg, only, tmp_path / "e1-subset", seed=1)
    assert set(narrow.tables["scores"]["policy"]) == set(subset)
    assert "r14_minus_r2" in narrow.primary and "r5_minus_r1" not in narrow.primary
    assert narrow.primary["by_policy"]["R14"] == result.primary["by_policy"]["R14"]
    with pytest.raises(ValueError, match="unknown or inactive"):
        experiment.run(cfg, replace(corpus, meta={**corpus.meta, "e1_policies": ["R99"]}), tmp_path / "x", seed=1)

    bad = replace(corpus, meta={**corpus.meta, "e1_jev_prompt_sha256": "0" * 64})
    with pytest.raises(ValueError, match="prompt changed"):
        experiment.run(cfg, bad, tmp_path / "e1-bad", seed=1)
