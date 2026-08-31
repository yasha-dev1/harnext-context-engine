"""Synthetic-corpus coverage for store variants in evaluation spec §7 E3."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from harnext_eval.agents.reader import answer
from harnext_eval.config import load_config
from harnext_eval.corpus.synthetic import generate_synthetic_events
from harnext_eval.e2.arms import a3, a4, store_read
from harnext_eval.providers.embeddings import FakeEmbeddings
from harnext_eval.providers.factory import make_embeddings
from harnext_eval.providers.llm import FakeLLM
from harnext_eval.replay.driver import run_pipeline
from harnext_eval.stores.base import StoreHandle
from harnext_eval.stores.build_s3 import run_builder_harness
from harnext_eval.stores.layouts import configure_store
from harnext_eval.stores.vector_index import StoreVectorIndex, VectorIndex, search_store
from harnext_eval.types import EvalEvent, Probe


class _NamedEmbeddings:
    provider_id = "fixture-provider"
    model_revision = "2026-08-30"

    def __init__(self, model_id: str, dim: int = 32) -> None:
        self.model_id = model_id
        self._delegate = FakeEmbeddings(dim=dim)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._delegate.embed(texts)


def _state_event(
    event_id: str,
    at: datetime,
    *,
    status: str,
    old_status: str | None = None,
) -> EvalEvent:
    data: dict[str, object] = {
        "issue_key": "HNX-1",
        "field": "status",
        "from": old_status,
        "to": status,
        "linked_keys": ["HNX-2"],
    }
    if old_status is None:
        data["state"] = {
            "status": status,
            "assignee": "alice",
            "priority": "Critical",
            "components": ["builder", "api"],
            "fixVersion": "1.2.0",
        }
    return EvalEvent(
        id=event_id,
        source="jira:test",
        type="org.harnext.jira.issue.transition",
        subject="issue:HNX-1",
        time=at,
        mgtenant="test",
        baseline_keys=["component:builder"],
        data=data,
    )


def _store(tmp_path: Path, layout: str) -> StoreHandle:
    store = StoreHandle(layout, "synthetic", tmp_path / layout.lower())
    configure_store(store, harness="fake", embeddings=FakeEmbeddings(dim=128), timeout_s=30)
    return store


@pytest.mark.parametrize("layout", ["S0", "S1", "S2", "S3", "S4", "S5"])
def test_every_layout_folds_and_records_same_input(
    tmp_path: Path, layout: str
) -> None:
    events = generate_synthetic_events(seed=7, event_count=6, days=2, entity_count=3)
    store = _store(tmp_path, layout)
    cfg = load_config("apps/eval/configs/baseline-minimal.yaml")

    stats = run_pipeline(events, cfg.engine, store)

    assert stats.events == len(events)
    assert stats.snapshots
    with store.snapshots_csv.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == len(stats.snapshots)
    metadata = json.loads((store.worktree / "_meta" / "input.json").read_text())
    delivered = (store.worktree / "_meta" / "delivered_event_ids.jsonl").read_text()
    expected = hashlib.sha256(delivered.encode()).hexdigest()
    assert metadata["same_input_hash"] == expected
    assert metadata["event_count"] == len(events)
    assert set(delivered.splitlines()) == {event.id for event in events}

    if layout in {"S2", "S3", "S5"}:
        usage_rows = (store.root / "usage.jsonl").read_text().splitlines()
        assert len(usage_rows) == len(stats.snapshots)
        assert all(json.loads(row)["harness"] == "fake" for row in usage_rows)


def test_s0_is_exactly_one_dated_markdown_file_per_event_plus_minimal_index(
    tmp_path: Path,
) -> None:
    events = generate_synthetic_events(seed=2, event_count=4, days=2, entity_count=2)
    store = _store(tmp_path, "S0")
    store.write("STALE.md", "# stale curated document\n")
    store.write("entities/issue/stale/OVERVIEW.md", "# stale\n")
    store.write("_meta/fake-curator.md", "# stale fake curator marker\n")
    store.write("events/2020/01/01/rogue.md", "# Event rogue\n")

    ref = store.fold(events, "batch")

    expected_events = sorted(
        f"events/{event.time:%Y/%m/%d}/{event.id}.md" for event in events
    )
    files = store.list_files(ref)
    reader_files = [path for path in files if not path.startswith("_meta/")]
    assert reader_files == ["INDEX.md", *expected_events]
    assert [path for path in files if path.startswith("_meta/")] == [
        "_meta/delivered_event_ids.jsonl",
        "_meta/input.json",
    ]
    index = store.read(ref, "INDEX.md") or ""
    assert index == "# Event files\n\n" + "".join(
        f"- [{relpath}]({relpath})\n" for relpath in expected_events
    )
    assert all(store.read(ref, relpath) is not None for relpath in expected_events)
    assert "_meta/" not in index


def test_s1_templates_every_catalogue_and_world_state_field(tmp_path: Path) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)

    def event(
        event_id: str,
        minute: int,
        *,
        source: str,
        event_type: str,
        subject: str,
        data: dict[str, object],
    ) -> EvalEvent:
        return EvalEvent(
            id=event_id,
            source=source,
            type=event_type,
            subject=subject,
            time=start + timedelta(minutes=minute),
            mgtenant="test",
            data=data,
        )

    events = [
        event(
            "jira-created",
            0,
            source="jira:KAFKA",
            event_type="org.apache.jira.issue.created",
            subject="issue:KAFKA-1",
            data={
                "status": "Open",
                "assignee": "alice",
                "priority": "Critical",
                "components": ["core", "streams"],
                "fix_versions": ["4.0"],
            },
        ),
        event(
            "pr-merged",
            1,
            source="github:apache/kafka",
            event_type="com.github.pull_request.merged",
            subject="pr:17",
            data={
                "merged": True,
                "changed_files": ["core/src/Main.java", "core/src/MainTest.java"],
            },
        ),
        event(
            "pr-closed",
            2,
            source="github:apache/kafka",
            event_type="com.github.pull_request.closed",
            subject="pr:18",
            data={"state": "closed", "changed_files": ["docs/rejected.md"]},
        ),
        event(
            "mail-vote",
            3,
            source="mail:dev@kafka.apache.org",
            event_type="org.apache.mail.message",
            subject="thread:vote-root",
            data={
                "subject": "[VOTE] KIP-9",
                "body": "Please vote on KIP-9.",
                "in_reply_to": None,
                "author": "committer:bob",
            },
        ),
        event(
            "mail-result",
            4,
            source="mail:dev@kafka.apache.org",
            event_type="org.apache.mail.message",
            subject="thread:vote-root",
            data={
                "subject": "[RESULT] KIP-9 vote accepted",
                "body": "The vote passed and KIP-9 is accepted.",
                "in_reply_to": "root-message",
                "author": "committer:alice",
            },
        ),
        event(
            "world-state",
            5,
            source="orgforge:test",
            event_type="orgforge.world_state.dump",
            subject="world:snapshot-1",
            data={
                "world_state": {
                    "entities": {
                        "account:7": {
                            "plan": "enterprise",
                            "open_tickets": 2,
                            "unpaid_invoices": ["inv-3"],
                            "incident_status": "mitigated",
                            "owner": "alice",
                        }
                    }
                }
            },
        ),
    ]
    expected = {
        "entities/issue/KAFKA-1": {
            "status": "Open",
            "assignee": "alice",
            "priority": "Critical",
            "components": '["core","streams"]',
            "fixVersion": '["4.0"]',
        },
        "entities/pr/17": {
            "state": "merged",
            "changed_files": '["core/src/Main.java","core/src/MainTest.java"]',
        },
        "entities/pr/18": {
            "state": "closed",
            "changed_files": '["docs/rejected.md"]',
        },
        "entities/thread/vote-root": {"answered_by": "committer:alice"},
        "entities/entity/KIP-9": {"vote_outcome": "accepted"},
        "entities/account/7": {
            "plan": "enterprise",
            "open_tickets": "2",
            "unpaid_invoices": '["inv-3"]',
            "incident_status": "mitigated",
            "owner": "alice",
        },
    }
    store = _store(tmp_path, "S1")

    ref = store.fold(events, "batch")

    for entity_dir, fields in expected.items():
        overview = store.read(ref, f"{entity_dir}/OVERVIEW.md") or ""
        facts = store.read(ref, f"{entity_dir}/facts.md") or ""
        for field, value in fields.items():
            assert f"- {field}: {value}" in overview
            assert f" {field}={value}" in facts
    superseded = store.read(ref, "_meta/superseded.md") or ""
    assert "answered_by=UNANSWERED" in superseded
    assert "vote_outcome=open" in superseded


def test_s1_tracks_latest_fields_and_moves_superseded_values(tmp_path: Path) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    first = _state_event("state-open", start, status="Open")
    second = _state_event(
        "state-resolved",
        start + timedelta(days=1),
        status="Resolved",
        old_status="Open",
    )
    store = _store(tmp_path, "S1")

    store.fold([first], "batch")
    ref = store.fold([second], "fast")

    overview = store.read(ref, "entities/issue/HNX-1/OVERVIEW.md") or ""
    facts = store.read(ref, "entities/issue/HNX-1/facts.md") or ""
    timeline = store.read(ref, "entities/issue/HNX-1/timeline.md") or ""
    superseded = store.read(ref, "_meta/superseded.md") or ""
    index = store.read(ref, "INDEX.md") or ""
    assert "status: Resolved" in overview
    assert "assignee: alice" in overview
    assert "priority: Critical" in overview
    assert 'components: ["builder","api"]' in overview
    assert "fixVersion: 1.2.0" in overview
    assert 'linked_keys: ["HNX-2"]' in overview
    assert "status=Open" not in facts
    assert "status=Resolved" in facts
    assert "status=Open" in superseded
    assert "state-open" in timeline and "state-resolved" in timeline
    assert "entities/issue/HNX-1/OVERVIEW.md" in index


def test_s2_fake_curator_writes_flat_state_timeline_and_supersession(tmp_path: Path) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    first = _state_event("state-open", start, status="Open")
    second = _state_event(
        "state-resolved",
        start + timedelta(days=1),
        status="Resolved",
        old_status="Open",
    )
    store = _store(tmp_path, "S2")

    ref = store.fold([first, second], "batch")

    files = store.list_files(ref)
    assert "INDEX.md" not in files
    assert not any(path.startswith("topics/") for path in files)
    overview = store.read(ref, "entities/issue/HNX-1/OVERVIEW.md") or ""
    facts = store.read(ref, "entities/issue/HNX-1/facts.md") or ""
    timeline = store.read(ref, "entities/issue/HNX-1/timeline.md") or ""
    superseded = store.read(ref, "_meta/superseded.md") or ""
    label = store.read(ref, "_meta/fake-curator.md") or ""
    assert "status: Resolved" in overview
    assert "status=Resolved" in facts and "status=Open" not in facts
    assert "state-open" in timeline and "state-resolved" in timeline
    assert '"id": "state-open"' in timeline
    assert "status=Open" in superseded and "status=Resolved" in superseded
    assert "stand-in" in label
    usage = json.loads((store.root / "usage.jsonl").read_text().splitlines()[0])
    assert usage["model"] == "offline-fake-curator-v1"
    assert usage["tokens_in"] > 0 and usage["tokens_out"] > 0


def test_s3_fake_curator_adds_index_and_topics(tmp_path: Path) -> None:
    event = _state_event(
        "state-open",
        datetime(2026, 1, 1, tzinfo=UTC),
        status="Open",
    )
    store = _store(tmp_path, "S3")
    configure_store(store, harness="fake", seed=17)

    ref = store.fold([event], "batch")

    index = store.read(ref, "INDEX.md") or ""
    topic = store.read(ref, "topics/issue.md") or ""
    assert "offline fake-curator" in index.casefold()
    assert "entities/issue/HNX-1/OVERVIEW.md" in index
    assert "HNX-1" in topic
    metadata = json.loads(store.read(ref, "_meta/input.json") or "{}")
    usage = json.loads((store.root / "usage.jsonl").read_text().splitlines()[0])
    assert metadata["builder_seed"] == 17
    assert usage["seed"] == 17


def test_live_builder_request_carries_seed_and_usage_records_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event = _state_event(
        "seeded-live",
        datetime(2026, 1, 1, tzinfo=UTC),
        status="Open",
    )
    store = _store(tmp_path, "S3")
    configure_store(store, harness="claude_code", model="fixture-model", seed=23)
    captured: dict[str, object] = {}

    def fake_run_build(
        org_id: str,
        command: list[str],
        env: dict[str, str],
        timeout_s: int,
    ) -> SimpleNamespace:
        del org_id, command, timeout_s
        request = json.loads(Path(env["REQUEST_PATH"]).read_text())
        captured.update(request)
        Path(env["RESULT_PATH"]).write_text(
            json.dumps(
                {
                    "harness": "claude_code",
                    "model": "fixture-model",
                    "usage": {"input_tokens": 2, "output_tokens": 1},
                    "stop_reason": "completed",
                }
            )
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(store.backend, "run_build", fake_run_build)
    run_builder_harness(store, [event], "batch")

    assert captured["seed"] == 23
    usage = json.loads((store.root / "usage.jsonl").read_text().splitlines()[0])
    assert usage["seed"] == 23


def test_s4_exact_id_search_and_historical_snapshot(tmp_path: Path) -> None:
    events = generate_synthetic_events(seed=4, event_count=2, days=1, entity_count=2)
    store = _store(tmp_path, "S4")
    first = store.fold([events[0]], "batch")
    second = store.fold([events[1]], "batch")
    provider = FakeEmbeddings(dim=128)

    assert search_store(store, events[1].id, top_k=1) == [events[1].id]
    assert search_store(store, events[0].id, top_k=1, provider=provider, ref=first) == [
        events[0].id
    ]
    assert events[1].id not in search_store(
        store, events[1].id, top_k=10, provider=provider, ref=first
    )
    assert VectorIndex.from_store(store, provider=provider, ref=second).count == 2
    natural_hits = StoreVectorIndex(store, provider).query(
        f"What happened to {events[0].subject}?",
        10,
        at=first,
    )
    assert [hit.item_id for hit in natural_hits] == [events[0].id]
    assert events[0].id in natural_hits[0].document
    assert VectorIndex(store, provider).query(
        f"What happened to {events[0].subject}?", 10, at=first
    ) == natural_hits
    metadata = json.loads(store.read(second, "_vector/metadata.json") or "{}")
    assert metadata == {
        "chunking": "one-raw-event-per-document-v1",
        "dimension": 128,
        "document_count": 2,
        "embedding_model": "fake-feature-hash-blake2b-v1",
        "embedding_provider": "fake",
        "embedding_revision": "1",
        "indexed_event_count": 2,
    }
    counts = (store.worktree / "_vector" / "snapshot_counts.jsonl").read_text().splitlines()
    assert [json.loads(row)["indexed_events"] for row in counts] == [1, 2]


def test_configured_real_adapter_reaches_a3_and_s4_without_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, list[str]]] = []

    class SpyVoyageClient:
        def __init__(self, api_key: str | None = None) -> None:
            assert api_key is None

        def embed(self, texts: list[str], *, model: str) -> SimpleNamespace:
            calls.append((model, list(texts)))
            rows = [
                [1.0, float(len(text)), float(sum(map(ord, text)) % 101 + 1)]
                for text in texts
            ]
            return SimpleNamespace(embeddings=rows)

    monkeypatch.setitem(sys.modules, "voyageai", SimpleNamespace(Client=SpyVoyageClient))
    cfg = load_config(Path(__file__).parents[2] / "configs" / "s3-curated.yaml")
    adapter = make_embeddings(cfg)
    assert calls == []

    event = _state_event(
        "state-open",
        datetime(2026, 1, 1, tzinfo=UTC),
        status="Open",
    )
    probe = Probe(
        probe_id="configured-a3",
        family="extraction",
        entity="issue:HNX-1",
        T=event.time,
        question="What is the status of HNX-1?",
        gold="Open",
        gold_type="exact",
        source_event_ids=[event.id],
    )

    material = a3(probe, [event], cfg.engine, k=1)
    store = StoreHandle("S4", "configured-real", tmp_path / "configured-s4")
    configure_store(store, harness="fake", embeddings=adapter, timeout_s=30)
    store.fold([event], "batch")

    assert material.arm == "A3"
    assert len(calls) == 2
    assert calls[0][0] == calls[1][0] == "voyage-3-large"
    assert calls[0][1][0] == probe.question
    assert event.id in calls[1][1][0]
    metadata = json.loads((store.worktree / "_vector" / "metadata.json").read_text())
    assert metadata["embedding_provider"] == "voyage"
    assert metadata["embedding_model"] == "voyage-3-large"
    assert metadata["embedding_revision"] == "2025-01-07"


def test_vector_snapshot_requires_the_persisted_nonfake_model(tmp_path: Path) -> None:
    event = generate_synthetic_events(seed=8, event_count=1, days=1, entity_count=1)[0]
    root = tmp_path / "named-s4"
    provider = _NamedEmbeddings("fixture-embedding-v3")
    store = StoreHandle("S4", "named", root)
    configure_store(store, harness="fake", embeddings=provider)
    ref = store.fold([event], "batch")

    metadata = json.loads(store.read(ref, "_vector/metadata.json") or "{}")
    assert metadata["embedding_provider"] == "fixture-provider"
    assert metadata["embedding_model"] == "fixture-embedding-v3"
    assert metadata["embedding_revision"] == "2026-08-30"
    assert StoreVectorIndex(store, provider).query("synthetic event", 1, at=ref)

    reopened = StoreHandle("S4", "named", root)
    with pytest.raises(ValueError, match="requires its pinned embedding provider"):
        StoreVectorIndex(reopened).query("synthetic event", 1, at=ref)
    with pytest.raises(ValueError, match="embedding model mismatch"):
        StoreVectorIndex(reopened, _NamedEmbeddings("wrong-model")).query(
            "synthetic event", 1, at=ref
        )


def test_s5_search_returns_event_ids_from_curated_files(tmp_path: Path) -> None:
    events = generate_synthetic_events(seed=5, event_count=3, days=1, entity_count=2)
    store = _store(tmp_path, "S5")

    ref = store.fold(events, "batch")

    provider = FakeEmbeddings(dim=128)
    for event in events:
        assert search_store(store, event.id, top_k=1, provider=provider, ref=ref) == [event.id]
    assert "INDEX.md" in store.list_files(ref)
    metadata = json.loads(store.read(ref, "_vector/metadata.json") or "{}")
    assert metadata["embedding_model"] == "fake-feature-hash-blake2b-v1"
    assert metadata["chunking"] == "whole-durable-file-plus-event-citations-v1"


def test_fake_reader_can_read_pr_files_from_s1_and_curated_layout(
    tmp_path: Path,
) -> None:
    at = datetime(2026, 1, 1, tzinfo=UTC)
    event = EvalEvent(
        id="pr-merged",
        source="github:test",
        type="org.harnext.github.pull_request.merged",
        subject="issue:HNX-1",
        time=at,
        mgtenant="test",
        baseline_keys=["component:builder"],
        data={
            "issue_key": "HNX-1",
            "number": 42,
            "pr_key": "PR-42",
            "changed_files": ["src/context/store.py", "tests/test_store.py"],
            "state": {"status": "Resolved"},
        },
    )
    s1 = _store(tmp_path, "S1")
    s3 = _store(tmp_path, "S3")
    s1.fold([event], "batch")
    s3.fold([event], "batch")
    cfg = load_config("apps/eval/configs/baseline-minimal.yaml").engine
    probe = Probe(
        probe_id="code-location",
        family="code_location",
        entity="HNX-1",
        T=at,
        question="Which files changed for HNX-1?",
        gold=["src/context/store.py", "tests/test_store.py"],
        gold_type="files",
        source_event_ids=[event.id],
    )

    s1_answer = answer(probe, store_read(probe, s1, cfg), cfg, provider=FakeLLM())
    s3_answer = answer(probe, a4(probe, s3, cfg), cfg, provider=FakeLLM())

    expected_paths = {"src/context/store.py", "tests/test_store.py"}
    s1_score = float(all(path in s1_answer.text for path in expected_paths))
    s3_score = float(all(path in s3_answer.text for path in expected_paths))
    assert s1_score == 1.0
    assert s3_score == 1.0
