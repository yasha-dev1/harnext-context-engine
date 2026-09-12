# Configurable E2 strategy gate — 2026-09-12

Implemented the [shared YAML/MCP route](../E2-STRATEGIES.md), not a completed
configuration-selection study. The example is
[indexed.yaml](../configs/strategies/indexed.yaml); all enums and constraints are
available in [schema.json](../configs/strategies/schema.json).

## Executed checks

| Check | Result | Evidence |
|---|---|---|
| Eval, builder and MCP regression suite | 395 passed in 165.80s; 10 existing statistical/multiprocessing warnings | `pytest apps/eval/tests apps/builder/tests apps/mcp/tests -q` |
| Strategy and Codex tests after strict-output correction | 31 passed in 2.91s | `pytest apps/eval/tests/test_e2/test_strategies.py apps/builder/tests/test_codex.py -q` |
| Static checks and configuration | Targeted Pyright: zero errors/warnings; Ruff passed; example YAML validated; diff whitespace check passed; locked optional-retrieval install succeeded | Commands use the repository `.venv` |
| Six retrieval transport conditions | All passed: BM25/dense/hybrid × no reranker/cross-encoder; real models and separate-process MCP clients | [Raw report](../out/strategies/mcp-neural-verification-001/report.json) |
| External native Codex reader | Correct answer; 3 MCP calls; 6,710 response bytes; required-support recall 1.0; zero tool errors | [Report](../out/strategies/codex-external-001/report.json), [native trace](../out/strategies/codex-external-001/codex.jsonl), [launch](../out/strategies/codex-external-001/launch.json) |
| Learned builder | Two source updates passed with `agent_curated`, `atomic_llm`, `abstractive`, `concise_supported` | [Configuration](../out/strategies/learned-builder-001/condition-retry.yaml), [transcripts](../out/strategies/learned-builder-001/artifact-retry/builder-transcripts.json), [snapshot](../out/strategies/learned-builder-001/artifact-retry/snapshot.json) |
| Matrix expansion | 12 file + 12 retrieval + 8 graph YAML cells validated/generated; not executed as a study | [File manifest](../out/strategies/file-grid/matrix.json), [retrieval manifest](../out/strategies/retrieval-grid/matrix.json), [graph manifest](../out/strategies/graph-grid/matrix.json) |

Both live agent checks used `gpt-5.6-luna` at `medium` effort. The reader called
`find_entities`, `context_search`, and `get_assertions`; the required supporting
span was first delivered on call 2. Precision is deliberately null because gold
is partial. The six protocol cases are scripted external clients, not six agent
performance trials. Learned-builder inputs include supplied facts; this check
establishes harness/schema/path execution, not independent extraction accuracy.

The first learned-builder attempt failed because Codex strict structured output
requires every object property in `required`, including nullable/defaulted fields.
The transport schema now makes those values explicit while retaining domain-model
defaults. A regression test covers this, and the live retry passed. Failed output
was preserved separately from the successful artifact.

The learned snapshot SHA is
`266b5f703aa3d44f1f8d557a7c7093ead8dd8469321652b3dda834a6e79a7614`.
The external reader snapshot SHA is
`d0a013f63b889f9d9f0b4e37cc710d1fddddf1c3b93b2a0dd75e28b57ff3a438`.
These are historical build hashes; later source edits do not rewrite their
provenance. Runtime server source is not separately sealed by these build hashes.

## Scope and remaining gates

The mechanisms work through configuration: materialized file layouts and policy
changes, actual BM25/dense/hybrid/reranking, temporal graph methods, and read-only
MCP serving with server-side receipts/budgets and external grading. The standalone
experiment entrypoint lives in the production packages; existing streaming
service deployment has not been migrated to it.

Local FastEmbed and ONNX cross-encoder execution are verified. The configurable
remote embedding adapter and Claude/Harnext learned-builder adapters have not
been live verified in this gate. Graph methods are paper-inspired reference
implementations, not full named-system reproductions. The example reader-tool
matrix is not the blueprint's builder-native-tools versus shell matrix.

Next: admit real historical QA/bugfix/feature tasks at frozen pre-change snapshots,
review evidence mappings for each representation, run a paired difficulty pilot,
then freeze screening/confirmation splits and the decision rule. The existing
synthetic Python coding adapter is useful for admission tests but does not provide
all repository build systems. Preserve external native traces and isolate agents
from gold/future data; byte receipts alone cannot establish whole-agent cost,
token use, identity or absence of leakage through other tools.
