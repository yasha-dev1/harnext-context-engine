"""Hand-built E4 action and localisation grader cases."""

import re
from typing import Any

import pytest
from harnext_eval.grade.action import grade_action, grade_rouge_l, judge_pairwise_stable
from harnext_eval.grade.localisation import (
    grade_localisation,
    localisation_scores,
    module_for,
)
from harnext_eval.providers.llm import LLMResult


class LengthJudge:
    def complete(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int,
    ) -> LLMResult:
        del system, json_schema, max_tokens
        match = re.search(r"Response A:\n(.*?)\n\nResponse B:\n(.*)$", user, re.DOTALL)
        assert match is not None
        winner = "A" if len(match.group(1)) > len(match.group(2)) else "B"
        return LLMResult(text="", json={"winner": winner}, usage={})


class PositionBiasedJudge:
    def complete(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int,
    ) -> LLMResult:
        del system, user, json_schema, max_tokens
        return LLMResult(text="A", json={"winner": "A"}, usage={})


def test_agentless_localisation_requires_gold_superset_in_top_k() -> None:
    gold = ["src/kafka/state.py", "tests/kafka/test_state.py"]
    prediction = ["src/kafka/state.py", "docs/readme.md", "tests/kafka/test_state.py"]
    result = grade_localisation("t", prediction, gold, k=3)
    assert result.value == 1.0
    assert result.details["file_recall"] == 1.0
    assert result.details["file_precision"] == 2 / 3
    assert result.details["module_hit"] == 1.0
    assert module_for("src/kafka/state.py") == "src/kafka"

    missed_gold = localisation_scores(prediction[:1], gold, k=1)
    assert missed_gold["file_hit@1"] == 0.0
    loose = localisation_scores(prediction[:1], gold, k=1, agentless_superset=False)
    assert loose["file_hit@1"] == 1.0


def test_action_quality_averages_available_fields_then_id_coverage() -> None:
    gold = {
        "people": {"assignees": ["alice"], "reviewers": ["bob"]},
        "category": {
            "components": ["storage"],
            "duplicate_of": "KAFKA-2",
            "priority_changes": ["Critical"],
            "required_ids": ["event-1", "event-2"],
        },
    }
    prediction = {
        "assignee_candidates": ["mallory", "Alice"],
        "reviewer_candidates": ["Bob"],
        "component": "Storage",
        "duplicate_of": "KAFKA-99",
        "priority_change": "critical",
        "cited_ids": ["EVENT-1"],
    }
    result = grade_action("t", prediction, gold)
    assert result.details["field_em"] == 0.8
    assert result.details["id_cov"] == 0.5
    assert result.value == pytest.approx(0.65)


def test_action_quality_ignores_unavailable_groups() -> None:
    result = grade_action(
        "t",
        {"component": "api", "cited_ids": ["event-1"]},
        {
            "people": {"assignees": ["wrong"]},
            "category": {"component": "api"},
            "required_ids": ["event-1"],
        },
        gold_coverage={"people": False, "category": True},
    )
    assert result.details["available_fields"] == ["component_exact"]
    assert result.value == 1.0


def test_rouge_l_uses_lcs_and_pairwise_win_must_survive_both_orders() -> None:
    rouge = grade_rouge_l("t", "a b c", "a x b c")
    assert rouge.details["lcs_length"] == 3
    assert rouge.details["precision"] == 1.0
    assert rouge.details["recall"] == 0.75
    assert rouge.value == pytest.approx(6 / 7)

    stable = judge_pairwise_stable("t", "a much better response", "bad", LengthJudge())
    assert stable.value == 1.0
    biased = judge_pairwise_stable("t", "candidate", "baseline", PositionBiasedJudge())
    assert biased.value == 0.0
    assert biased.details["stable"] is False
