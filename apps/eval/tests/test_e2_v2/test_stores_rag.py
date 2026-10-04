"""R-hyb: token chunks, monthly shards, hard-linked as-of exports, search, leak safety —
offline neural stand-in."""

import json

import numpy as np
import pytest
from harnext_builder.strategies.retrieval import tokens
from harnext_eval.e2.v2.stores import synthetic
from harnext_eval.e2.v2.stores.rag import (
    MAX_TOKENS,
    HashNeural,
    RagStore,
    SparseBM25,
    best,
    chunk,
    id_rank,
    load,
    read_event,
    retrieval_config,
    search,
)
from rank_bm25 import BM25Okapi

EVENTS = synthetic.events(130, hours=12)


class Counting(HashNeural):
    def __init__(self):
        self.embedded = 0

    def embed(self, texts, *, query=False):
        self.embedded += 0 if query else len(texts)
        return super().embed(texts, query=query)


def test_blueprint_parameters():
    cfg = retrieval_config()
    assert (cfg.method, cfg.bm25_k1, cfg.bm25_b, cfg.rrf_k, cfg.candidates, cfg.top_k, cfg.reranker) == (
        "hybrid", 1.2, 0.75, 60, 40, 10, "cross_encoder")  # fmt: skip


def test_one_chunk_per_event_unless_longer_than_the_token_budget():
    neural = HashNeural()
    assert len(chunk(EVENTS[0], neural)) == 1
    parts = chunk(EVENTS[0] | {"text": " ".join(f"w{n}" for n in range(1200))}, neural)
    assert len(parts) == 3 and [p["part"] for p in parts] == [0, 1, 2]
    assert all(len(neural.spans(p["text"])) <= MAX_TOKENS - 2 for p in parts)
    assert "w0 " in parts[0]["text"] and parts[2]["text"].endswith("w1199")
    assert all(p["text"].startswith("[2023-01-02T09:00:00Z] jira.created") for p in parts)


def test_checkpoints_embed_only_new_events_and_exports_are_cumulative(tmp_path):
    neural = Counting()
    RagStore(builder="fake", neural=neural).build(iter(EVENTS), tmp_path, until="2023-01")
    january = sum(e["time"] < "2023-02" for e in EVENTS)
    assert neural.embedded == january
    rag = RagStore(builder="fake", neural=neural)
    rag.build(iter(EVENTS), tmp_path)
    assert neural.embedded == len(EVENTS)  # January was not embedded again
    target = rag.export(tmp_path, "2023-02")
    index = load(str(target))
    chunks = [index.record(n) for n in range(len(index))]
    assert {c["event_id"] for c in chunks} == {e["id"] for e in EVENTS if e["time"] < "2023-03"}
    assert [v.shape[1] for v in index.vectors] == [64, 64] and sum(map(len, index.vectors)) == len(
        chunks
    )
    meta = json.loads((target / "meta.json").read_text())
    assert meta["chunks"] == len(chunks) and meta["models"]["embedding"]["model"] == "fake-hash-64"
    assert [s["month"] for s in meta["shards"]] == ["2023-01", "2023-02"]
    manifest = json.loads((tmp_path / "export/2023-02.manifest.json").read_text())
    assert manifest["files"] == 7 == sum(p.is_file() for p in target.rglob("*"))
    # the export shares the index's files instead of copying them
    assert (target / "shards/2023-01.emb.f32").samefile(tmp_path / "index/2023-01.emb.f32")
    assert rag.export(tmp_path, "2023-01") and manifest["content_sha256"] != json.loads(
        (tmp_path / "export/2023-01.manifest.json").read_text())["content_sha256"]  # fmt: skip


def test_a_killed_build_resumes_at_the_unfinished_month(tmp_path):
    class Killed(HashNeural):
        def passages(self, texts):
            if any("2023-02" in t[:12] for t in texts):
                raise KeyboardInterrupt
            return super().passages(texts)

    with pytest.raises(KeyboardInterrupt):
        RagStore(builder="fake", neural=Killed()).build(iter(EVENTS), tmp_path)
    state = json.loads((tmp_path / "build-state.json").read_text())
    assert sorted(state["checkpoints"]) == ["2023-01"]
    (tmp_path / "index/2023-02.chunks.jsonl").write_text("debris of the killed month\n")
    RagStore(builder="fake").build(iter(EVENTS), tmp_path)
    RagStore(builder="fake").build(iter(EVENTS), tmp_path / "fresh")
    resumed, fresh = (json.loads((d / "build-state.json").read_text())["checkpoints"] for d in (tmp_path, tmp_path / "fresh"))  # fmt: skip
    assert {m: c["sha256"] for m, c in resumed.items()} == {
        m: c["sha256"] for m, c in fresh.items()
    }


def test_leak_safety_a_checkpoint_sees_and_counts_only_its_own_past(tmp_path):
    """Checkpoint X of the shared index == an index built from the events before X alone:
    same BM25 statistics (N, average length, df, idf per word), same hits; nothing from
    after X is in the mounted directory or can be returned."""
    full = RagStore(builder="fake")
    full.build(iter(EVENTS), tmp_path / "full")
    assert max(e["time"] for e in EVENTS) > "2023-03"
    for checkpoint in ("2023-01", "2023-02"):
        cutoff = f"{checkpoint[:5]}{int(checkpoint[5:]) + 1:02d}"
        past = [e for e in EVENTS if e["time"] < cutoff]
        only = RagStore(builder="fake")
        only.build(iter(past), tmp_path / checkpoint)
        shared = load(str(full.export(tmp_path / "full", checkpoint)))
        alone = load(str(only.export(tmp_path / checkpoint, checkpoint)))
        assert len(shared) == len(alone) == len(past) < len(EVENTS)
        assert shared.lexical.statistics() == alone.lexical.statistics()
        reference = SparseBM25([tokens(shared.record(n)["text"]) for n in range(len(shared))], 1.2, 0.75)  # fmt: skip
        assert shared.lexical.statistics() == reference.statistics()
        future = {e["id"] for e in EVENTS} - {e["id"] for e in past}
        words = {w for e in EVENTS if e["id"] in future for w in tokens(e["text"])}
        assert words - set(shared.lexical.vocabulary)  # words first seen later are unknown
        for query in ("StoreChangelogReader state restore", "KIP-501 fix", " ".join(sorted(words)[:20])):  # fmt: skip
            hits = shared.search(query, k=40)
            assert hits == alone.search(query, k=40)
            assert hits and all(h["time"] < cutoff and h["event_id"] not in future for h in hits)
        names = {p.name[:7] for p in (tmp_path / "full/export" / checkpoint / "shards").iterdir()}
        assert max(names) == checkpoint
        assert shared.dense(np.ones(64, dtype=np.float32)).shape == (len(past),)


def test_search_returns_ranked_chunks_from_the_exported_directory(tmp_path):
    rag = RagStore(builder="fake")
    rag.build(iter(EVENTS), tmp_path)
    hits = search(
        rag.export(tmp_path, "2023-03"), "issue:KAFKA-9003 StoreChangelogReader state restore", k=5
    )
    assert len(hits) == 5 and hits[0]["score"] >= hits[-1]["score"]
    assert "KAFKA-9003" in hits[0]["text"] and {"id", "event_id", "time", "text", "score"} <= set(
        hits[0]
    )
    assert search(tmp_path / "export/2023-03", "   ") == []
    whole = read_event(tmp_path / "export/2023-03", hits[0]["event_id"])
    assert whole[0]["id"] == hits[0]["event_id"] and hits[0]["text"] in whole[0]["text"]
    with pytest.raises(ValueError, match="no such event"):
        read_event(tmp_path / "export/2023-03", "jira:absent")


def test_sparse_bm25_scores_equal_rank_bm25():
    corpus = [tokens(e["text"]) for e in EVENTS]
    ours, theirs = SparseBM25(corpus, 1.2, 0.75), BM25Okapi(corpus, k1=1.2, b=0.75)
    for query in ("LogCleaner stale epoch retry retry", "the fix for KIP-501", "absent-word"):
        assert np.allclose(
            ours.get_scores(tokens(query)), theirs.get_scores(tokens(query)), atol=1e-5
        )


def test_best_breaks_ties_by_id_without_a_full_sort():
    assert best(np.array([1.0, 3.0, 3.0, 2.0, 3.0]), id_rank(["e", "d", "c", "b", "a"]), 2) == [
        4,
        2,
    ]
    assert best(np.zeros(3), id_rank(["b", "c", "a"]), 2) == [2, 0]


def test_a_cached_month_must_match_the_stream(tmp_path):
    RagStore(builder="fake").build(iter(EVENTS), tmp_path)
    with pytest.raises(RuntimeError, match="2023-02"):
        RagStore(builder="fake").build((e for e in EVENTS if e["id"] != EVENTS[70]["id"]), tmp_path)
