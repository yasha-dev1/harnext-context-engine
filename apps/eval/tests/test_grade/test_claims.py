"""Claim-grader precision, recall, and reproducibility cases."""

from typing import Any

from harnext_eval.grade.claims import grade_claims, grade_claims_twice
from harnext_eval.providers.llm import FakeLLM, LLMResult


class PipeClaimsProvider:
    """Small deterministic provider that records decompositions and entailments."""

    def __init__(self) -> None:
        self.decomposed: list[str] = []

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int,
    ) -> LLMResult:
        del json_schema, max_tokens
        if system.startswith("Decompose"):
            text = user.removeprefix("Text:\n")
            self.decomposed.append(text)
            claims = [part.strip() for part in text.split("|") if part.strip()]
            return LLMResult(text="", json={"claims": claims}, usage={})
        evidence, claim = user.removeprefix("Evidence:\n").split("\n\nClaim:\n", 1)
        entailed = claim.casefold() in evidence.casefold()
        return LLMResult(text="", json={"entailed": entailed}, usage={})


def test_claims_fake_path_is_deterministic_and_penalises_unsupported_claims() -> None:
    provider = FakeLLM()
    result = grade_claims(
        "p",
        "Kafka is stable. The moon is green.",
        "Kafka is stable. Version 4 is current.",
        provider,
    )
    assert result.details["precision"] == 0.5
    assert result.details["recall"] == 0.5
    assert result.value == 0.5
    first, second, agreement = grade_claims_twice("p", "A. B.", "A. B.", provider)
    assert first == second
    assert agreement.value == 1.0
    assert agreement.details["disagreement_rate"] == 0.0


def test_gold_is_decomposed_once_per_claim_grade() -> None:
    provider = PipeClaimsProvider()
    result = grade_claims("p", "one | extra", "one | two", provider)
    assert provider.decomposed.count("one | two") == 1
    assert provider.decomposed.count("one | extra") == 1
    assert result.details["precision"] == 0.5
    assert result.details["recall"] == 0.5
