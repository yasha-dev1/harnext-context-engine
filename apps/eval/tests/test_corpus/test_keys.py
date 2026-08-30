"""Key derivation checks for docs/evaluation-spec.md §3.1."""

import hashlib

from harnext_eval.corpus.keys import (
    component_key,
    contributor_key,
    derive_baseline_keys,
    derive_subject,
    extract_issue_keys,
    extract_kip_keys,
    extract_text_keys,
    issue_subject,
    kip_subject,
    pr_subject,
    thread_subject,
)


def test_extracts_canonical_issue_and_kip_references_without_duplicates() -> None:
    text = "kafka-19876 implements KIP-1150; KAFKA-19876 follows KIP-999."
    assert extract_issue_keys(text) == ["KAFKA-19876"]
    assert extract_kip_keys(text) == ["KIP-1150", "KIP-999"]
    assert extract_text_keys(text) == [
        "issue:KAFKA-19876",
        "kip:1150",
        "kip:999",
    ]


def test_canonical_subject_helpers() -> None:
    assert issue_subject("kafka-19876") == "issue:KAFKA-19876"
    assert kip_subject("KIP-1150") == "kip:1150"
    assert pr_subject("#20412") == "pr:20412"
    assert thread_subject("<root@example.org>") == "thread:root@example.org"


def test_contributor_hash_and_baseline_keys_are_stable_and_private() -> None:
    expected = hashlib.sha256(b"alice@apache.org").hexdigest()[:12]
    assert contributor_key("Alice Committer <ALICE@APACHE.ORG>") == f"contributor:{expected}"
    assert component_key("Stream Processing") == "component:stream-processing"
    assert derive_baseline_keys(
        author_emails=["alice@apache.org", "ALICE@APACHE.ORG"],
        components=["Streams", "streams"],
        thread_root="<root@example.org>",
    ) == [
        f"contributor:{expected}",
        "component:streams",
        "thread:root@example.org",
    ]


def test_subject_derivation_keeps_native_pr_and_uses_cross_source_key_for_push() -> None:
    assert (
        derive_subject(
            "com.github.pull_request.opened",
            {"number": 20412, "title": "KAFKA-19876: snapshot fix"},
        )
        == "pr:20412"
    )
    assert (
        derive_subject(
            "com.github.push",
            {"commits": [{"message": "KIP-1150 follow-up for KAFKA-19876"}]},
        )
        == "kip:1150"
    )

