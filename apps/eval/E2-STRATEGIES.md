# E2 merged experiment: configurable strategies and external MCP evaluation

The shared configuration is [configs/strategies/indexed.yaml](configs/strategies/indexed.yaml).
It controls an executable builder, immutable context snapshot, retrieval engine,
read-only experiment MCP server, and evaluator. Unknown keys and incompatible
combinations fail validation. This is the new experiment route in the production
builder/MCP packages; the existing streaming service entrypoint is not switched
automatically to these experimental strategies.

## Configuration contract

| YAML field | Supported settings and actual effect |
|---|---|
| `representation` | `files`, `graph`, `hybrid`: materialized context surfaces |
| `files.layout` | `verbatim_dump`: event files; `foldered_sessions`: unchanged events grouped by canonical entity; `flat_facts`: one assertion per file; `indexed_entities`: entity pages and index; `indexed_temporal`: entity pages plus ledger; `topic_hierarchy`: topic/entity directories and index; `agent_curated`: model-selected, validated Markdown paths |
| `files.prompt` | `preserve_supported`: include quote wrappers; `concise_supported`: concise assertions with source IDs; `domain_focused`: restrict configured predicates. These also select learned extraction instructions |
| `files.extraction` | `structured`: consume supplied facts deterministically; `atomic_llm`: extract source-backed assertions through the selected builder harness |
| `files.retention` | `all`: retain eligible history; `current`: latest eligible functional assertions; `window`: observed-time window of `retention_days` |
| `files.summarization` | `none`, `extractive` (bounded sentence count), `abstractive` (learned extraction instruction) |
| `files.update` | `append`: assertion history; `replace`: current functional assertions; `supersede`: history with temporal metadata |
| `retrieval.method` | `literal` occurrence ranking, actual `bm25`, actual embedding cosine `dense`, or `hybrid` reciprocal rank fusion |
| `retrieval.chunking` | `document`, UTF-8-safe `fixed` byte windows, or `entity_bundle` followed by bounded windows |
| `retrieval.embeddings.provider` | Local `fastembed` or `openai_compatible` embedding endpoint; model, revision, cache, threads and timeout configurable |
| `retrieval.reranker` | `none` or actual local `cross_encoder`; model configurable |
| `graph.schema_name` | `generic_attributed` or `typed_assertions` with domain vocabulary/type checks |
| `graph.extraction` | `broad` or `domain_predicates` |
| `graph.resolution` | `canonical_only` or `evidence_aliases`; aliases require source support and cannot merge conflicting canonical IDs |
| `graph.traversal` | `neighbors`, `paths`, hop-bounded `personalized_pagerank`, or `dual_level` entity/topic seed selection |
| `builder` | Harness, model, effort, provider/key environment variable, calls, time and input-byte allowance |
| `trial` | External harness/model/effort labels, run/task IDs, MCP tool allowlist, response/total bytes, calls, time and transport |
| `evaluation` | `retrieval_only`, `qa_exact`, or `coding_tests`; public task, private gold, answer/candidate, workspace and native trace paths |

Additional numeric controls include page size, summary length, functional
predicates, BM25 parameters, fusion constant, candidate count, top-k, chunk size
and overlap, graph hops/edges/expansions, damping and iterations. Generate the
complete schema with `python -m harnext_builder.strategies schema`.

Structured mode makes no builder model calls. Learned extraction, abstractive
summarization and agent-curated paths require a learned builder. Codex requires
explicit effort. The Harnext SDK adapter requires an explicit provider and
`effort: null`, because equivalent reasoning-effort control is not verified in
that SDK. Claude Code supports low/medium/high here. Only Codex has been live
verified for this new route.

Raw/foldered preservation disallows transformations that contradict unchanged
event text. `expose_sources: true` deliberately gives the reader a raw archive
escape route; use `false` when measuring whether a curated view alone retained
enough evidence. File and graph functional predicates are independently explicit;
multi-valued relationships are not collapsed like single-valued status/ownership.

## Relationship to the papers

The layout and writing factors adapt [Filesystem-Based Memory for LLM Agents:
Organization, Evolution, and Sustainability](https://arxiv.org/html/2607.26637v1).
Our foldered baseline uses deterministic entity grouping; it does not reproduce
the paper's learned move-only organization. Learned writing uses constrained
source-backed assertion extraction with previous assertions in context, rather
than unrestricted editing of arbitrary files.

Valid-time and known-time assertion filtering follows the motivation in
[Zep](https://arxiv.org/html/2501.13956v1). Entity/topic retrieval and personalized
PageRank are reference adaptations motivated by
[LightRAG](https://arxiv.org/html/2410.05779v1) and
[HippoRAG](https://arxiv.org/abs/2405.14831). These enums are working methods,
not claims to reproduce complete Graphiti, LightRAG or HippoRAG packages.

The original thesis blueprint remains the scientific planning reference. The
example file grid's third factor is **reader MCP navigation versus search tools**;
it is not the blueprint's builder-native-file-tools versus shell factor. Builder
native-tool comparisons require a separate treatment/adapter. Chunk budgets here
are **bytes**, not the blueprint's token counts. Do not label these cells as exact
paper replications or token-equivalent experiments.

## Build and connect an outside agent

Run from the engine repository root. Install optional real retrieval models:

```bash
uv sync --all-packages --extra retrieval
.venv/bin/python -m harnext_builder.strategies validate --config apps/eval/configs/strategies/indexed.yaml
.venv/bin/python -m harnext_builder.strategies build --config apps/eval/configs/strategies/indexed.yaml
.venv/bin/python -m harnext_eval.e2.mcp_experiment client-config --config apps/eval/configs/strategies/indexed.yaml
```

Use a fresh `artifact_dir` for every build; existing snapshots are not overwritten.
Input is normalized Source JSONL with IDs, entity, timezone-aware observed/valid
clocks, text and optional structured facts. See `sources.jsonl`. This is not an
automatic converter for every legacy ingestion corpus. Each fact needs an exact
source quote. Quote containment proves provenance, not semantic entailment;
extraction quality still requires reviewed labels.

The builder excludes events observed after the cutoff before constructing any
view or index. It saves documents, graph, model-content pins, source/config/prompt
hashes, build implementation hash, package versions and learned transcripts.
Local dense document vectors persist in the snapshot. Serving verifies model
content against the build pins; bootstrap `content-hash-at-build` records the
actual files, while a supplied content hash also constrains the build.

The handoff command prints a stdio `mcpServers` configuration. Any compatible
outside client may connect: the server never invokes an answering model. For
Codex, translate the handoff into its [native MCP configuration](https://developers.openai.com/codex/mcp/):

```toml
[mcp_servers.harnext]
command = "/absolute/path/to/harnext-context-engine/.venv/bin/python"
args = ["-m", "harnext_mcp.experiment", "--config", "/absolute/path/to/condition.yaml"]
startup_timeout_sec = 60
tool_timeout_sec = 60
```

For remote clients set `trial.transport: http`, appropriate bind host/port, and a
token environment variable containing at least 32 characters, then run
`python -m harnext_mcp.experiment --config condition.yaml`. The URL is `/mcp`.
The generated HTTP `authorization_env` is an adapter handoff field, not a
universal client key; map it to the client's bearer-token setting. Use the
reachable server address when its bind address is `0.0.0.0`.

Available read tools are `context_list`, `context_read`, `context_search`,
`find_entities`, `get_assertions`, `neighbors`, and `fetch_source`. The allowlist
actually changes what is registered. No gold answers, hidden tests, future events
or write tools are exposed by this server.

Assign a new run ID and fresh agent history for each independent trial. Restarting
the same trial restores its original budget and clock; it does not grant a new
allowance. Multiple server processes cannot claim one trial simultaneously.
Engine configuration must match the frozen artifact; trial/evaluation metadata
may change so independent readers can reuse the same built snapshot.

## Deterministic measurement and isolation

The server records executed tool calls, arguments, errors, returned document byte
spans and full response bytes in a hash-chained, flushed journal. Total bytes
include wrappers and repeated reads. Calls, response/total bytes and elapsed
retrieval allowance are enforced server-side. Hash chains detect inconsistency;
they are not signatures protecting against an administrator rewriting a journal.
Malformed requests rejected by MCP before tool execution are not retrieval calls.

Private gold binds an exact snapshot hash to reviewed evidence byte spans and
required groups; alternatives may satisfy the same group. The evaluator checks
that receipts match frozen text, unions partial exposures across calls, and grants
support only once the required span has actually been returned. Merely returning
an assertion ID earns no support credit. It reports required-support recall,
recall over calls, first relevant call, tool mix, repeated bytes, errors and byte
use. Precision is null without exhaustive relevance labels; recall is null
without gold. Seeing evidence is not proof the agent causally used it.

Set `evaluation.task_file`, `gold_file`, and the relevant `answer_file` or
`candidate_files` in the same YAML, then:

```bash
.venv/bin/python -m harnext_eval.e2.mcp_experiment stage --config condition.yaml
# Run the outside agent against the staged task/workspace and configured MCP.
.venv/bin/python -m harnext_eval.e2.mcp_experiment grade --config condition.yaml
```

`stage` exports public task/base files/visible tests into a new workspace. The
outside harness must be isolated from private gold, future commits, reference
patches and the evaluator checkout; MCP isolation alone cannot prevent another
client tool reading those locations. Run the server outside the agent sandbox.
QA grades use typed exact answers. The current coding adapter supports Python
fixture source files and pinned Docker tests: base/reference admission, reproduced
fail-to-pass tests, preserved pass-to-pass tests and pristine final grading tests.
Real repository build systems and historical task populations need their own
admitted task adapters; arbitrary real PRs are not yet turnkey.

Harness/model/effort fields are declarations, not proof. Preserve the native
agent trace (`evaluation.agent_trace`) and launch configuration. Actual provider
tokens, monetary cost, whole-agent deadlines and non-MCP tool usage require native
harness telemetry/control; the MCP byte budget does not enforce those quantities.

## Screening matrices and next study gate

```bash
.venv/bin/python -m harnext_eval.e2.strategy_matrix --config apps/eval/configs/strategies/file-grid.yaml
.venv/bin/python -m harnext_eval.e2.strategy_matrix --config apps/eval/configs/strategies/retrieval-grid.yaml
.venv/bin/python -m harnext_eval.e2.strategy_matrix --config apps/eval/configs/strategies/graph-grid.yaml
```

These generate 12 file, 12 retrieval and 8 graph configurations; generation does
not execute experiments. Dot-path factor lists can vary validated fields. All
cells validate before any are written. Start with separate screening blocks,
then cross the shortlisted settings to test interactions. Keep paired tasks,
source cutoffs, visible tests, budgets and independent agent histories fixed.
Separate development selection from held-out confirmation; repeat stochastic
builds/readers and predeclare the selection rule and uncertainty analysis.

The immediate next scientific gate is an admitted historical PR/QA population
with per-representation evidence mappings, followed by a small paired pilot.
Smoke success establishes working wiring, not an optimal configuration or an
adequately difficult thesis dataset. See the [verification record](STATUS/E2-STRATEGIES-20260912.md).
