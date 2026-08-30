"""Fake-provider tests for docs/evaluation-spec.md §5."""

import numpy as np
import pytest
from harnext_eval.providers.embeddings import FakeEmbeddings
from harnext_eval.providers.llm import FakeLLM


def test_fake_llm_returns_last_lexical_value() -> None:
    result = FakeLLM().complete(
        "Answer only from the material.",
        "Question: What is the status of HNX-7?\n"
        "Material:\n"
        "HNX-7 status: Open\n"
        "HNX-8 status: Blocked\n"
        "HNX-7 status: Resolved\n",
        max_tokens=20,
    )

    assert result.text == "Resolved"
    assert result.json is None
    assert result.usage["input_tokens"] > 0


def test_fake_llm_abstains_without_matching_material() -> None:
    result = FakeLLM().complete(
        "",
        "Question: What is the assignee of HNX-7?\nMaterial:\nHNX-7 status: Open",
        max_tokens=20,
    )

    assert result.text == "UNKNOWN"


def test_fake_embeddings_are_normalised_and_deterministic() -> None:
    provider = FakeEmbeddings(dim=16)
    vectors = provider.embed(["alpha beta alpha", "", "alpha beta alpha"])

    np.testing.assert_array_equal(vectors[0], vectors[2])
    assert np.linalg.norm(vectors[0]) == pytest.approx(1)
    assert np.linalg.norm(vectors[1]) == 0
