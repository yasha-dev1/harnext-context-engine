# Arm `R-hyb` — hybrid BM25 + dense retrieval, RRF, cross-encoder rerank

Source: thesis blueprint `06-e3-experiment-blueprint.html`, factors D08–D12 and §07 "Retrieval
regime" (exact starting contract table); `05-e3-store-design-plan.html` rows R-*. Implementation:
`stores/rag.py`, ranking reused from `apps/builder/src/harnext_builder/strategies/retrieval.py`.
Read over MCP tools `search(query, k≤50)` / `read_event(id)` (`mcp_server.py:29`, `325-331`).
Shared decisions: [README.md](README.md).

| blueprint says | what the code does | deviation (and why) |
|---|---|---|
| D08 / K0: *"One event per chunk when ≤512 tokens; longer events split into 512-token pieces with 64-token overlap. Every piece carries event/version ID and availability time."* | One chunk per event; longer events cut on token boundaries of the **embedding model's own tokenizer** (`rag.py:67-90` `LocalNeural.spans`, truncation off); budget = 512 − 2 special − header − 1 (`rag.py:46`, `147`); every piece repeats a one-line header `[time] type entities — title (actor)` capped at 96 tokens (`rag.py:144-146`); `id = <event_id>#<part>` (`rag.py:159`). | **Overlap is 32 tokens, not 64** (`rag.py:46 OVERLAP_TOKENS`). Tokens are bge-small WordPiece tokens, not the blueprint's "pinned audit tokenizer" — the blueprint left the tokenizer to be bound at pilot (§201). The header steals up to 96 of the 512 tokens. `chunking: "fixed"` in the config is informational only (`rag.py:53`); `chunk_bytes=2048` from the builder's config class is **not** used — chunking is by tokens. |
| K1 entity bundles (D08 second level). | Not implemented; `bundling: "none"` recorded in `meta.json` (`rag.py:516-521`). | Out of scope: only the K0 R-hyb cell is run. |
| BM25: *"Pinned tokenizer; k1=1.2, b=0.75 … No hidden exact-ID bonus."* | `k1=1.2, b=0.75` (`rag.py:57-58`) overriding the builder default 1.5 (`config.py:163`). Tokenizer = `re.findall(r"[\w:-]+", text.casefold())` (`retrieval.py:16-17`), so `KAFKA-1234` is one token. `SparseBM25` reproduces `rank_bm25.BM25Okapi` scoring incl. its `epsilon=0.25` negative-idf floor (`rag.py:173-234`). | Reimplemented as a sparse matrix for ~10⁶ chunks; same formula, not the same library. The `[\w:-]+` tokenizer is a de-facto exact-ID advantage (blueprint asks for none but names no tokenizer). |
| Dense: *"One pinned embedding model E0; normalized vectors; cosine."* | `BAAI/bge-small-en-v1.5` via fastembed ONNX (`config.py:140`), 384-d, local files under `.harnext/models`, content-hashed and pinned into `meta.json:models` (`rag.py:80-87`, `366-374`: a mismatched model refuses to load). Cosine = dot on normalised vectors (`rag.py:314-320`). | E0 was a role in the blueprint; bound here to bge-small (cheap, CPU). Same model family as Graphiti's embedder (see G-graphiti.md). |
| Hybrid: *"RRF with k=60 over eligible lexical and dense ranked lists; union then dedupe."* | `rrf_k=60`, top `candidates=40` per ranker, `1/(60+rank)` (`rag.py:56`, `340-347`). | none |
| D10 / depth: *"40 eligible candidates; optional pinned cross-encoder reranks these same candidates."* Reference R-hyb = *"K0 and RRF with no reranker."* | Fused list cut to 40, **all 40 reranked** by `Xenova/ms-marco-MiniLM-L-6-v2` (`config.py:167`, fastembed `TextCrossEncoder`), top 10 returned (`rag.py:347-353`). | **This arm is the blueprint's reranked cell, not its "R-hyb reference" (which had no reranker)**; E2-V2-DESIGN's arms table asks for the reranker explicitly. The blueprint's separate BM25-only / dense-only / no-rerank cells are not run. |
| D12 candidate pool 20/40/80. | 40 only. | Only the core value. |
| Reader: ≤20 rounds, typed answer. | MCP `search` returns k ≤ 50 chunks (`mcp_server.py:32`), default 10; `read_event` returns one full event. No round cap beyond the agent's own. | Rounds are not capped (the coding agent's turn budget applies). |
| Prefix-correct: *"cutoff before ranking"*, per-prefix df/length stats. | **One shared, append-only index, read as of a checkpoint.** `build` streams the events once and writes one immutable shard per month, `index/<YYYY-MM>.{chunks.jsonl,emb.f32,lex.npz}` (`rag.py:393-429`, `444-487`; one month in memory, a killed build redoes one month). The stream is time-ordered, so checkpoint X = the shards ≤ X; `export/<X>/` is `meta.json` (shard list with as-of row counts) plus **hard links** to those shards (`rag.py:489-540`) — 42 checkpoints cost the disk of one index. `Index` (`rag.py:258-302`) reads only the export's shards: postings are merged into one sparse matrix and **N, document frequencies, idf (incl. the epsilon floor's mean) and average length are computed from those shards alone**; vectors are memory-mapped and scored block-wise; chunk text is read by byte offset for the fused 40 only. | none — stronger than a filter: the directory mounted into the trial contains no text, vector, id or vocabulary entry from after the checkpoint, so neither the ranker nor an agent reading `/store` can see the future. `test_leak_safety_…` asserts that checkpoint X of the shared index has the same BM25 statistics and returns the same hits as an index built from the events before X only. Identity = sha256 over the shards' (month, rows, file hashes), hashed when each shard was written (`<X>.manifest.json`). |

## What an examiner should know

- `R-hyb` is a **code-built** store: no LLM writes anything; its cost is embedding time only
  (`build-state.json:seconds`). It is the "warehouse plus retriever" alternative of plan row 145.
- The ranking function is that of the thesis engine's `SearchIndex` (`rag.py:258-263` docstring)
  with scale substitutions (sparse BM25 from stored postings, partial sorts, memory-mapped
  vectors, passages embedded sorted by length in ONNX batches of 16 — padding only, same vectors up
  to float rounding); ties broken by chunk id in both.
- Two parameters differ from the blueprint text: overlap 32 vs 64 tokens, and the reranker is
  always on. Neither was tuned on tasks; both are fixed before any trial.
- Chunk text starts with the header line, so the reranker and the embedder see time, type,
  entities and actor — metadata the blueprint says every piece must carry, here inside the text.
- The `fake` builder (`HashNeural`, `rag.py:117-137`) exists for offline tests only; `meta.json:
  builder` distinguishes it and `load` refuses to mix them (`rag.py:366-374`).
- Query-time cost (embedding one query + reranking 40 pairs on CPU) is inside the MCP server and is
  not billed as model tokens; the blueprint asks to "count every reranking cost" — here it is wall
  time in `mcp/receipts.jsonl`, not dollars.
