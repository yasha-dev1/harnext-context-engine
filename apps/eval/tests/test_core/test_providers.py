"""Provider tests for docs/evaluation-spec.md §5."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from harnext_eval.providers.embeddings import FakeEmbeddings, OpenAIEmbeddings
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


def test_fake_llm_reads_raw_history_at_the_question_cutoff() -> None:
    material = "\n".join(
        json.dumps(record)
        for record in (
            {
                "id": "a" * 24,
                "time": "2026-01-01T00:00:00+00:00",
                "subject": "issue:KAFKA-7",
                "data": {"field": "status", "to": "Open"},
            },
            {
                "id": "b" * 24,
                "time": "2026-01-03T00:00:00+00:00",
                "subject": "issue:KAFKA-7",
                "data": {"field": "status", "to": "Resolved"},
            },
        )
    )

    result = FakeLLM().complete(
        "",
        "Question: What was the status of KAFKA-7 as of "
        f"2026-01-02T00:00:00Z?\nMaterial:\n{material}",
        max_tokens=20,
    )

    assert result.text == "Open"


def test_fake_llm_does_not_assign_state_to_incidentally_mentioned_entity() -> None:
    material = json.dumps(
        {
            "time": "2026-01-01T00:00:00+00:00",
            "subject": "issue:KAFKA-7",
            "data": {
                "issue_key": "KAFKA-7",
                "linked_kip": "KIP-900",
                "field": "status",
                "to": "Resolved",
            },
        }
    )

    result = FakeLLM().complete(
        "",
        f"Question: What is the current status of KIP-900?\nMaterial:\n{material}",
        max_tokens=20,
    )

    assert result.text == "UNKNOWN"


def test_fake_llm_uses_transition_value_not_embedded_state_snapshot() -> None:
    material = "\n".join(
        json.dumps(record)
        for record in (
            {
                "time": "2026-01-01T00:00:00+00:00",
                "subject": "issue:KAFKA-7",
                "data": {"issue_key": "KAFKA-7", "field": "state", "to": "merged"},
            },
            {
                "time": "2026-01-02T00:00:00+00:00",
                "subject": "issue:KAFKA-7",
                "data": {
                    "issue_key": "KAFKA-7",
                    "field": "status",
                    "to": "Resolved",
                    "state": {"status": "Resolved", "components": ["api"]},
                },
            },
        )
    )

    state = FakeLLM().complete(
        "", f"Question: What is the current state of KAFKA-7?\nMaterial:\n{material}", max_tokens=20
    )
    components = FakeLLM().complete(
        "",
        f"Question: What is the current components of KAFKA-7?\nMaterial:\n{material}",
        max_tokens=20,
    )

    assert state.text == "merged"
    assert components.text == "UNKNOWN"


def test_fake_llm_normalises_multivalue_transition_fields() -> None:
    material = json.dumps(
        {
            "time": "2026-01-01T00:00:00+00:00",
            "subject": "issue:KAFKA-7",
            "data": {
                "issue_key": "KAFKA-7",
                "field": "fixVersion",
                "to": "4.2",
            },
        }
    )

    result = FakeLLM().complete(
        "",
        f"Question: What is the current fixVersion of KAFKA-7?\nMaterial:\n{material}",
        max_tokens=20,
    )

    assert result.text == '["4.2"]'


def test_fake_llm_uses_vote_subject_tag_as_kip_owner() -> None:
    material = json.dumps(
        {
            "time": "2026-01-01T00:00:00+00:00",
            "type": "org.apache.mail.message",
            "subject": "issue:KAFKA-7",
            "data": {
                "subject": "[VOTE] KIP-900 required for KAFKA-7",
                "subject_tags": ["KAFKA-7", "KIP-900"],
            },
        }
    )

    result = FakeLLM().complete(
        "",
        f"Question: What is the current vote_outcome of KIP-900?\nMaterial:\n{material}",
        max_tokens=20,
    )
    issue_result = FakeLLM().complete(
        "",
        f"Question: What is the current vote_outcome of KAFKA-7?\nMaterial:\n{material}",
        max_tokens=20,
    )

    assert result.text == "open"
    assert issue_result.text == "UNKNOWN"


def test_fake_llm_reads_curated_fact_formats() -> None:
    material = """[file:entities/issue/KAFKA-7/OVERVIEW.md]
# issue:KAFKA-7
_Last updated: 2026-01-02T00:00:00+00:00 [jira#old]_
- status: Open
[file:entities/issue/KAFKA-7/facts.md]
- 2026-01-03 [jira#new] status=Resolved
"""

    result = FakeLLM().complete(
        "",
        f"Question: What is the current status of KAFKA-7?\nMaterial:\n{material}",
        max_tokens=20,
    )

    assert result.text == "Resolved"


def test_fake_llm_lists_all_related_links_and_changed_files() -> None:
    material = "\n".join(
        json.dumps(record)
        for record in (
            {
                "time": "2026-01-01T00:00:00+00:00",
                "subject": "pr:PR-7",
                "data": {
                    "issue_key": "KAFKA-7",
                    "pr_key": "PR-7",
                    "number": 7,
                    "changed_files": ["services/api/handler.py"],
                },
            },
            {
                "time": "2026-01-02T00:00:00+00:00",
                "subject": "thread:THREAD-9",
                "data": {
                    "issue_key": "KAFKA-7",
                    "thread_key": "THREAD-9",
                    "thread_id": "9",
                },
            },
        )
    )
    provider = FakeLLM()

    links = provider.complete(
        "",
        f"Question: Which pull requests or mail threads are related to KAFKA-7?\n"
        f"Material:\n{material}",
        max_tokens=50,
    )
    files = provider.complete(
        "",
        f"Question: Which files and modules are changed for KAFKA-7?\nMaterial:\n{material}",
        max_tokens=50,
    )

    assert links.text.splitlines() == ["pr:7", "thread:9"]
    assert files.text == "services/api/handler.py"


def test_fake_llm_builds_action_json_from_the_envelope() -> None:
    material = "\n".join(
        json.dumps(record)
        for record in (
            {
                "id": "a" * 24,
                "time": "2026-01-01T00:00:00+00:00",
                "subject": "issue:KAFKA-7",
                "data": {
                    "issue_key": "KAFKA-7",
                    "actor": "dev-a",
                    "components": ["api"],
                    "changed_files": ["services/api/handler.py"],
                    "linked_issues": ["KAFKA-9"],
                },
            },
            {
                "id": "b" * 24,
                "time": "2026-01-02T00:00:00+00:00",
                "subject": "issue:KAFKA-7",
                "data": {"issue_key": "KAFKA-7", "actor": "dev-a"},
            },
        )
    )
    schema = {
        "type": "object",
        "properties": {
            "assignee_candidates": {"type": "array", "items": {"type": "string"}},
            "reviewer_candidates": {"type": "array", "items": {"type": "string"}},
            "component": {"type": ["string", "null"]},
            "duplicate_of": {"type": ["string", "null"]},
            "priority_change": {"type": ["string", "null"]},
            "suspected_locations": {"type": "array", "items": {"type": "string"}},
            "draft_reply": {"type": "string"},
            "cited_ids": {"type": "array", "items": {"type": "string"}},
            "action": {"type": "string"},
        },
    }

    result = FakeLLM().complete(
        "Return a typed action.",
        f"Question: Route KAFKA-7.\nMaterial:\n{material}",
        json_schema=schema,
        max_tokens=200,
    )

    assert isinstance(result.json, dict)
    assert result.json["assignee_candidates"] == ["dev-a"]
    assert result.json["component"] == "api"
    assert result.json["duplicate_of"] == "KAFKA-9"
    assert result.json["suspected_locations"] == ["services/api/handler.py"]
    assert "KAFKA-7" in result.json["cited_ids"]


def test_fake_llm_smoke_action_uses_post_onset_situation_evidence() -> None:
    records = "\n".join(
        json.dumps(record)
        for record in (
            {
                "subject": "issue:KAFKA-7",
                "data": {
                    "issue_key": "KAFKA-7",
                    "situation_archetype": "vote-thread",
                },
            },
            {
                "subject": "issue:KAFKA-7",
                "data": {
                    "issue_key": "KAFKA-7",
                    "body": "I am taking KAFKA-7",
                    "outcome_for": "situation-3",
                },
            },
        )
    )
    schema = {
        "type": "object",
        "properties": {
            "assignee_candidates": {"type": "array", "items": {"type": "string"}},
            "reviewer_candidates": {"type": "array", "items": {"type": "string"}},
            "component": {"type": ["string", "null"]},
            "duplicate_of": {"type": ["string", "null"]},
            "priority_change": {"type": ["string", "null"]},
            "suspected_locations": {"type": "array", "items": {"type": "string"}},
            "draft_reply": {"type": "string"},
            "cited_ids": {"type": "array", "items": {"type": "string"}},
            "action": {"type": "string"},
        },
    }

    result = FakeLLM().complete(
        "",
        f"Question: Route KAFKA-7.\nMaterial:\n## overview\n{records}",
        json_schema=schema,
        max_tokens=200,
    )

    assert isinstance(result.json, dict)
    assert result.json["assignee_candidates"][0] == "responder-vote-thread-3"


def test_fake_embeddings_are_normalised_and_deterministic() -> None:
    provider = FakeEmbeddings(dim=16)
    vectors = provider.embed(["alpha beta alpha", "", "alpha beta alpha"])

    np.testing.assert_array_equal(vectors[0], vectors[2])
    assert np.linalg.norm(vectors[0]) == pytest.approx(1)
    assert np.linalg.norm(vectors[1]) == 0


def test_fake_embeddings_rank_an_exact_identifier_first() -> None:
    provider = FakeEmbeddings(dim=64)
    vectors = provider.embed(
        [
            "What is the current status of KAFKA-1042?",
            "KAFKA-1004 status Resolved",
            "KAFKA-1042 status In Review",
            "A generic status update without an issue key",
        ]
    )
    scores = vectors[1:] @ vectors[0]

    assert int(np.argmax(scores)) == 1


def test_openai_embeddings_calls_pinned_model_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class SpyEmbeddingsAPI:
        def create(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=1, embedding=[0.0, 3.0]),
                    SimpleNamespace(index=0, embedding=[4.0, 0.0]),
                ]
            )

    class SpyOpenAI:
        def __init__(self, api_key: str | None = None) -> None:
            assert api_key is None
            self.embeddings = SpyEmbeddingsAPI()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=SpyOpenAI))
    provider = OpenAIEmbeddings("text-embedding-3-large", "2024-01-25")

    vectors = provider.embed(["first", "second"])

    assert calls == [
        {"input": ["first", "second"], "model": "text-embedding-3-large"}
    ]
    np.testing.assert_array_equal(vectors, np.eye(2))
