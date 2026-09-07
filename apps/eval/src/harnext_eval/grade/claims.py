"""Claim entailment grading for docs/evaluation-spec.md §7 E2."""

from __future__ import annotations

import json
import re
from typing import Any

from harnext_eval.grade.exact import normalize_exact
from harnext_eval.providers.llm import FakeLLM, LLMProvider
from harnext_eval.types import GradeResult

_CLAIMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {"type": "string"}}},
    "required": ["claims"],
    "additionalProperties": False,
}
_ENTAILMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"entailed": {"type": "boolean"}},
    "required": ["entailed"],
    "additionalProperties": False,
}


def _fake_claims(text: str) -> list[str]:
    """Deterministically split prose for the offline FakeLLM path."""

    pieces = re.split(r"(?:\r?\n|;|(?<=[.!?])\s+)", text)
    claims: list[str] = []
    for piece in pieces:
        cleaned = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", piece).strip()
        cleaned = cleaned.rstrip(".!?").strip()
        if cleaned:
            claims.append(cleaned)
    return list(dict.fromkeys(claims))


def _claims_from_result(payload: Any, text: str) -> list[str]:
    if isinstance(payload, dict) and isinstance(payload.get("claims"), list):
        values = payload["claims"]
    else:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        values = parsed.get("claims", []) if isinstance(parsed, dict) else []
    return [str(value).strip() for value in values if str(value).strip()]


def decompose_claims(text: str, provider: LLMProvider) -> list[str]:
    """Decompose text in one provider operation (or deterministic fake operation)."""

    if isinstance(provider, FakeLLM):
        return _fake_claims(text)
    result = provider.complete(
        "Decompose the supplied text into minimal, independently verifiable atomic claims.",
        f"Text:\n{text}",
        json_schema=_CLAIMS_SCHEMA,
        max_tokens=1024,
    )
    return list(dict.fromkeys(_claims_from_result(result.json, result.text)))


def _fake_entails(evidence: str, claim: str) -> bool:
    canonical_evidence = normalize_exact(evidence)
    canonical_claim = normalize_exact(claim)
    if not canonical_claim:
        return True
    if canonical_claim in canonical_evidence:
        return True
    evidence_claims = {normalize_exact(value) for value in _fake_claims(evidence)}
    return canonical_claim in evidence_claims


def entails(evidence: str, claim: str, provider: LLMProvider) -> bool:
    """Ask the configured provider whether evidence entails one atomic claim."""

    if isinstance(provider, FakeLLM):
        return _fake_entails(evidence, claim)
    result = provider.complete(
        "Judge entailment using only the supplied evidence. Do not use outside knowledge.",
        f"Evidence:\n{evidence}\n\nClaim:\n{claim}",
        json_schema=_ENTAILMENT_SCHEMA,
        max_tokens=16,
    )
    if isinstance(result.json, dict) and isinstance(result.json.get("entailed"), bool):
        return result.json["entailed"]
    folded = result.text.strip().casefold()
    return folded in {"true", "yes", "entailed", "a"}


def grade_claims(
    item_id: str,
    answer: str,
    gold: str,
    provider: LLMProvider,
) -> GradeResult:
    """Grade bidirectional atomic-claim entailment and return claim F1.

    Gold is decomposed exactly once. Recall checks each gold claim against the
    answer; precision decomposes the answer once and checks each answer claim
    against the gold. This prevents an answer with many unsupported additions
    from receiving perfect precision.
    """

    gold_claims = decompose_claims(gold, provider)
    answer_claims = decompose_claims(answer, provider)
    recalled = [claim for claim in gold_claims if entails(answer, claim, provider)]
    supported = [claim for claim in answer_claims if entails(gold, claim, provider)]
    recall = len(recalled) / len(gold_claims) if gold_claims else float(not answer_claims)
    precision = len(supported) / len(answer_claims) if answer_claims else float(not gold_claims)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return GradeResult(
        item_id=item_id,
        metric="claim_f1",
        value=f1,
        details={
            "precision": precision,
            "recall": recall,
            "gold_claims": gold_claims,
            "answer_claims": answer_claims,
            "entailed_gold_claims": recalled,
            "supported_answer_claims": supported,
        },
    )


def compare_claim_runs(first: GradeResult, second: GradeResult) -> GradeResult:
    """Compare two claim-grade runs for the E2 reproducibility check."""

    first_gold = set(first.details.get("entailed_gold_claims", []))
    second_gold = set(second.details.get("entailed_gold_claims", []))
    first_answer = set(first.details.get("supported_answer_claims", []))
    second_answer = set(second.details.get("supported_answer_claims", []))
    universe = (
        set(first.details.get("gold_claims", []))
        | set(second.details.get("gold_claims", []))
        | set(first.details.get("answer_claims", []))
        | set(second.details.get("answer_claims", []))
    )
    disagreements = (first_gold ^ second_gold) | (first_answer ^ second_answer)
    disagreement_rate = len(disagreements) / len(universe) if universe else 0.0
    return GradeResult(
        item_id=first.item_id,
        metric="claim_run_agreement",
        value=1.0 - disagreement_rate,
        details={
            "disagreement_rate": disagreement_rate,
            "first_f1": first.value,
            "second_f1": second.value,
            "disagreed_claims": sorted(disagreements),
        },
    )


def grade_claims_twice(
    item_id: str,
    answer: str,
    gold: str,
    provider: LLMProvider,
) -> tuple[GradeResult, GradeResult, GradeResult]:
    """Run claim grading twice and return both grades plus their agreement."""

    first = grade_claims(item_id, answer, gold, provider)
    second = grade_claims(item_id, answer, gold, provider)
    return first, second, compare_claim_runs(first, second)
