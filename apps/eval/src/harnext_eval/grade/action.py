"""Action-quality and reply grading for docs/evaluation-spec.md §7 E4."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from harnext_eval.grade.exact import normalize_exact
from harnext_eval.grade.links import normalise_keys
from harnext_eval.providers.llm import LLMProvider
from harnext_eval.types import GradeResult

_PAIRWISE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"winner": {"type": "string", "enum": ["A", "B", "tie"]}},
    "required": ["winner"],
    "additionalProperties": False,
}


def _as_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, Iterable):
        return [value]
    return list(value)


def _normalised_set(value: Any) -> set[str]:
    return {normalised for raw in _as_values(value) if (normalised := normalize_exact(raw))}


def _hit_at(prediction: Any, gold: Any, k: int = 3) -> float:
    predicted = [normalize_exact(value) for value in _as_values(prediction)[:k]]
    return float(bool(set(predicted) & _normalised_set(gold)))


def _set_exact(prediction: Any, gold: Any) -> float:
    return float(_normalised_set(prediction) == _normalised_set(gold))


def _get_group(gold: Mapping[str, Any], group: str) -> Mapping[str, Any]:
    value = gold.get(group, {})
    return value if isinstance(value, Mapping) else {}


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _covered(gold_coverage: Mapping[str, bool] | None, group: str) -> bool:
    return gold_coverage is None or gold_coverage.get(group, True)


def grade_action(
    item_id: str,
    prediction: Mapping[str, Any],
    gold: Mapping[str, Any],
    *,
    gold_coverage: Mapping[str, bool] | None = None,
) -> GradeResult:
    """Return E4's ``Q = mean(field_em, id_cov)`` over available inputs."""

    people = _get_group(gold, "people")
    category = _get_group(gold, "category")
    field_scores: dict[str, float] = {}

    assignees = _first(people, "assignees", "assignee")
    if _covered(gold_coverage, "people") and _as_values(assignees):
        field_scores["assignee_hit@3"] = _hit_at(
            _first(prediction, "assignee_candidates", "assignees", "assignee"), assignees
        )
    reviewers = _first(people, "reviewers", "reviewer")
    if _covered(gold_coverage, "people") and _as_values(reviewers):
        field_scores["reviewer_hit@3"] = _hit_at(
            _first(prediction, "reviewer_candidates", "reviewers", "reviewer"), reviewers
        )

    category_fields = (
        ("component", ("components", "component")),
        ("duplicate_of", ("duplicate_of",)),
        ("priority_change", ("priority_changes", "priority_change")),
    )
    if _covered(gold_coverage, "category"):
        for output_key, aliases in category_fields:
            gold_value = _first(category, *aliases)
            if _as_values(gold_value):
                predicted_value = _first(prediction, output_key, *aliases)
                field_scores[f"{output_key}_exact"] = _set_exact(predicted_value, gold_value)

    field_em = sum(field_scores.values()) / len(field_scores) if field_scores else None
    required_ids = normalise_keys(
        _first(gold, "required_ids", "required_id")
        or _first(category, "required_ids", "required_id")
    )
    id_cov: float | None = None
    cited_ids = normalise_keys(_first(prediction, "cited_ids", "evidence_event_ids"))
    if required_ids:
        id_cov = len(required_ids & cited_ids) / len(required_ids)

    available_components = [score for score in (field_em, id_cov) if score is not None]
    quality = sum(available_components) / len(available_components) if available_components else 0.0
    return GradeResult(
        item_id=item_id,
        metric="action_quality",
        value=quality,
        details={
            **field_scores,
            "field_em": field_em,
            "id_cov": id_cov,
            "required_ids": sorted(required_ids),
            "cited_ids": sorted(cited_ids),
            "available_fields": sorted(field_scores),
            "available_composite_components": len(available_components),
        },
    )


def _rouge_tokens(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    """Compute LCS length with O(min(n,m)) memory."""

    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def grade_rouge_l(item_id: str, prediction: str, gold: str) -> GradeResult:
    """Return token-level ROUGE-L F1 with precision and recall details."""

    predicted_tokens = _rouge_tokens(prediction)
    gold_tokens = _rouge_tokens(gold)
    common = lcs_length(predicted_tokens, gold_tokens)
    precision = common / len(predicted_tokens) if predicted_tokens else float(not gold_tokens)
    recall = common / len(gold_tokens) if gold_tokens else float(not predicted_tokens)
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return GradeResult(
        item_id=item_id,
        metric="rouge_l",
        value=score,
        details={"precision": precision, "recall": recall, "lcs_length": common},
    )


def _parse_winner(result_json: Any, text: str) -> str:
    value = result_json.get("winner") if isinstance(result_json, Mapping) else None
    if value is None:
        match = re.search(r"\b(A|B|tie)\b", text, flags=re.IGNORECASE)
        value = match.group(1) if match else "tie"
    folded = str(value).strip().casefold()
    return folded.upper() if folded in {"a", "b"} else "tie"


def _judge_once(
    provider: LLMProvider,
    candidate_a: str,
    candidate_b: str,
    criterion: str,
) -> str:
    result = provider.complete(
        "Choose the better response for the stated criterion. Return A, B, or tie.",
        f"Criterion:\n{criterion}\n\nResponse A:\n{candidate_a}\n\nResponse B:\n{candidate_b}",
        json_schema=_PAIRWISE_SCHEMA,
        max_tokens=16,
    )
    return _parse_winner(result.json, result.text)


def judge_pairwise_stable(
    item_id: str,
    candidate: str,
    baseline: str,
    provider: LLMProvider,
    *,
    criterion: str = "quality, correctness, and usefulness",
) -> GradeResult:
    """Count a candidate win only when it wins in both presentation orders."""

    candidate_first = _judge_once(provider, candidate, baseline, criterion)
    candidate_second = _judge_once(provider, baseline, candidate, criterion)
    stable_win = candidate_first == "A" and candidate_second == "B"
    return GradeResult(
        item_id=item_id,
        metric="judge_win",
        value=float(stable_win),
        details={
            "candidate_first_winner": candidate_first,
            "candidate_second_winner": candidate_second,
            "stable": (candidate_first, candidate_second) in {("A", "B"), ("B", "A")},
        },
    )
