# E2: merged context-engine evaluation

This runbook implements the first infrastructure gate for the merged E2/E3
blueprint in the thesis repository, `eval/artifacts/06-e3-experiment-blueprint.html`.
The blueprint's factor matrix remains the study plan. This smoke protocol is an
explicit, smaller engineering check. Its results must not enter selection or
confirmation statistics.

The next development extension is [E2-DEVELOPMENT-TASKS.md](E2-DEVELOPMENT-TASKS.md):
implementation-history questions and executable withheld-change tasks, with
independent code/test outcomes and span-based retrieval metrics. The original
ten-probe smoke remains unchanged for infrastructure regression checks.

## Starting harness and configuration

Use the existing **Harnext builder protocol and Git backend, with Codex CLI as
the builder harness**. The CLI adapter is `harnext_builder.harness.codex.CodexHarness`.
Its initial `builder_tool_policy: files` exposes host-controlled reads/writes,
with native shell tools disabled. It uses the same subprocess runner, event-file mounting, transcript,
rollback, and snapshot boundary as the production builder.

The native-shell trial on this machine failed before executing commands with
`bwrap: loopback: Failed RTM_NEWADDR`. File-tool mode does not disable or bypass
that sandbox: it gives the model no native shell and lets the host perform only
allowlisted file operations. The native adapter remains available via
`builder_tool_policy: native` for a later environment check. File-tool mode caps
each file at 64,000 bytes and each read/write batch at 256,000 bytes, rejects
symlinks and outside paths, and cannot edit evaluator metadata.

The first profile is `configs/e2-context-codex-smoke.yaml`:

| Choice | Initial setting | Reason |
|---|---|---|
| Builder agent | Codex CLI with host-controlled file tools; exact version recorded | Exercise the requested model through the production adapter interface |
| Builder model / reasoning | `gpt-5.6-luna` / `medium` | User-selected smoke model; no silent substitution |
| Reader model / reasoning | Same explicit model and effort | Keep the model fixed while testing infrastructure |
| Primary store | S3, current indexed entity files | Use the engine's actual operating manual and maintenance procedure |
| Cheap controls | S1 deterministic entity projection; S0 raw events | Distinguish builder losses from reader/protocol problems |
| Persistence | Git, one commit per successful fold | Recoverable builds and immutable snapshot references |
| Input | Six constructed events, one organization, three source types | Known causal gold; two updates; explicit cross-source relationships |
| Folding | UTC hour windows, at most 100 events / 64,000 serialized bytes | Avoid one paid build per isolated subject/event |
| Reader tools | Host-controlled `list`, `read`, literal `search` | Agent chooses material; all returned content is logged and charged |
| Retrieval cap | 32,768 UTF-8 bytes; 12,000 bytes per response | Exact local smoke accounting; explicitly not a model-token budget |
| Reader calls | At most six completions per probe; final schema permits only an answer | Bound retrieval and reserve a slot for supported output or abstention |
| Builder limit | 40 JSON-action completions; 180-second fold deadline | Reject incomplete builds; the CLI does not expose a verified output-token cap |
| Replication | One build per arm initially | Infrastructure smoke only; independent repeats configurable |
| Output | Ten probes × three stores = 30 answers | Two probes in each of five families |

This uses the Harnext **orchestration and store**, with the **Codex harness**.
It does not measure the separate `harness=harnext` SDK adapter. Comparing that
adapter with Codex and Claude Code is a later, matched harness factor. Neither
Kafka, the web application, nor the classifier is needed for this isolated
context-quality check. The same fixed input goes to every arm; E1 routing quality
is deliberately not a varying factor here.

## What runs, exactly

1. Validate the profile and explicit `--live` opt-in. Offline fixture mode never
   constructs a real provider. Reject reused output directories.
2. Write separate replay and gold files; hash the inputs, profile and source code.
   Gold is never sent in a builder or reader prompt.
3. Fold the identical ordered six-event stream into each store. The first window
   establishes facts; the second changes the assignee, priority and release freeze
   date, and adds a second linked PR. Changed code is mounted under `_event/` for
   the real builder, then removed before the snapshot.
4. On a completed build, commit the snapshot and delivery ledger. Nonzero exits,
   missing completion events, provider failures, timeouts, and tool-limit stops
   fail the build and roll back context edits.
5. Build both windows **before** reading either snapshot. Export reader-visible
   text by exact Git SHA. No Git object database, future branch, evaluator ledger,
   gold, or builder instruction file is passed to the reader tools.
6. For each question, Codex emits typed JSON actions. The host executes retrieval
   against that frozen text map, charges every returned byte, and sends the result
   back. Search snippets, file paths, repeated reads, errors and index text count.
   Intermediate history is replayed to the model; provider input tokens record
   this additional billed input separately from retrieved content.
7. Grade the final typed value and evidence deterministically, persist every
   answer, tool trace and provider transcript, and compute family means.

## The ten probes

| Family | Query target | Expected |
|---|---|---|
| Extraction | Earlier issue assignee | `Mira` |
| Extraction | Current issue priority | `P1` |
| Temporal | Issue status at the earlier snapshot | `Open` |
| Temporal | Freeze date at the earlier snapshot | `2026-10-01` |
| Update | Current assignee after reassignment | `Noah` |
| Update | Current freeze after rescheduling | `2026-10-08` |
| Multisource | All PRs linked to the issue | `{PR-71, PR-72}` |
| Multisource | Files changed across those linked PRs | `{src/cache.py, tests/test_cache.py}` |
| Abstention | Security-review approver | Unknown |
| Abstention | Actual production release timestamp | Unknown |

The temporal probes test earlier-snapshot lookup after newer state exists. They
do not yet test the full study's within-snapshot historical interval/reconstruction
panel. The multisource fixtures test explicit joins; alias ambiguity, missing
edges and long graph paths belong in the expanded development set.

## What is deterministic

The corpus, windows, snapshot prefix check, tool access, retrieval byte caps,
answer schema, exact-value/set grader, evidence checks and aggregation are
deterministic. **The model's answers and the files it builds are stochastic.**
Medium effort and pinned model names do not imply reproducible sampled output.
Codex exposes no verified sampling-seed control; use independent repeat IDs and
fresh stores. Do not change prompts by injecting arbitrary "random seeds".

Scalar values require the exact type and string. Sets ignore ordering but reject
duplicates. A missing value is JSON `null` with `unknown=true`; answering unknown
to an answerable question is wrong. No prose extraction, fuzzy matching or LLM
judge rescues an invalid output.

Value correctness and evidence correctness are separate. Correct evidence must
include required source IDs, cite no undelivered IDs, and cite only IDs that
appeared in material returned by a tool. This is provenance validation, not a
general semantic entailment grader for arbitrary prose.

Infrastructure passes only when every snapshot prefix check passes and every
probe completes with consistent typed output within its retrieval budget.
Quality is descriptive; the smoke has no winner-selection threshold. The
offline fixture intentionally retrieves once and abstains everywhere, producing
20% macro accuracy. That checks the protocol without pretending to emulate a model.

## Run and inspect

From the engine repository root:

```bash
codex --version
codex login status

.venv/bin/python -m harnext_eval.e2.context \
  --config apps/eval/configs/e2-context-offline-smoke.yaml \
  --out apps/eval/out/e2-context/offline-001

.venv/bin/python -m harnext_eval.e2.context \
  --config apps/eval/configs/e2-context-codex-smoke.yaml \
  --out apps/eval/out/e2-context/live-001 --live

.venv/bin/pytest apps/eval/tests apps/builder/tests -q
```

Each run stores:

- `manifest.json`: model, effort, CLI version, source/input hashes, limitations,
  running/completed/failed/interrupted status, and `thesis_evidence=false`.
- `resolved-config.json`, `replay.jsonl`, `probes.gold.jsonl`: frozen smoke inputs.
- `<arm>/store/`: Git snapshots, exact delivery ledger, builder usage and raw
  Codex transcript events. S0/S1 have no paid builder calls.
- `<arm>/answers/<probe>.json`: selected SHA, content hash, tool returns, byte
  accounting, typed answer, exact grades, provider usage and elapsed time.
- `<arm>/reader-transcripts.json`: raw completion events and provider failures.
- `results.json` and `report.md`: infrastructure verdict, per-family scores,
  macro accuracy, build timing and reader token totals. USD costs remain unknown
  when the CLI does not report them; cache hits are not added to total input twice.

The actual successful study runs need pinned billable prices and accurate model
token caps. A ChatGPT-authenticated CLI smoke must not be reported as costing $0.

## Readiness after this gate

| Component | State / next acceptance criterion |
|---|---|
| Codex builder + Git snapshots | Live smoke exercises production runner, event-file mounts, incremental maintenance and usage |
| Tool-using reader + typed grading | Runnable; unit tests cover gold separation, budgets, schema/value/evidence failures and earlier snapshots |
| Five family wiring + cheap controls | Runnable synthetic smoke; no statistical evidence |
| Exact 2k/8k/32k token-budget comparison | Pending pinned tokenizer, wrapper accounting, and parity tests; byte caps are not a replacement |
| Real development replay | Pending causal-field audit and frozen, manually checked development probes; current CLI intentionally accepts only the smoke fixture |
| F4/F5/F7 × extraction policy × tool policy | Pending exact template/prompt manifests and comparable writer implementations; current S3 is a starting point, not all 12 cells |
| Semantic/BM25/hybrid retrieval cells | Legacy components exist; merged tool interface, real embedding pins, reranking, index accounting and parity tests pending |
| Knowledge-graph cells | Pending schema/extractor/entity-resolution/traversal implementations and deterministic failure fixtures |
| Harness comparisons | Codex adapter is wired; equivalent budgets, tools and isolation must be frozen before comparing harnesses |
| Builder filesystem isolation | File-tool mode enforces a host path allowlist and rejects symlinks; native shell mode remains pending because this host's bwrap sandbox fails during setup |
| Scheduling policy | Smoke uses event time and serial completed folds. Real publication clocks, cap splits, duplicate/out-of-order events and latency evaluation need the registered contract |
| Full chart suite and statistics | Pending real result tables with paired workloads and repeated builds; the blueprint remains the chart specification |
| Selection and confirmation | Pending sample-size/cluster audit, frozen selection rules and untouched confirmation set |

The immediate next work is a small **real-data development pilot with pinned
token accounting**, followed by the file-regime screen. Add retrieval and graph
cells through the same reader and scoring contract, then run matched harness
comparisons on surviving configurations. Expand repeat counts only after fixture
failures and exposure/accounting checks pass. None of the smoke's ten questions
should become the held-out confirmation set.

## Retest readers and replay the evidence

To verify a reader fix without paying to rebuild the same context, use
`--reuse-builds`. The runner checks the builder configuration, replay/probe
hashes, snapshot delivery prefixes and layout availability. It copies the Git
stores, preserves their exact SHAs, records the parent manifest, reports zero new
builder calls, and leaves new-build latency undefined. A profile may select a
subset of the source run's stores for a focused retest.

```bash
.venv/bin/python -m harnext_eval.e2.context \
  --config apps/eval/configs/e2-context-codex-smoke.yaml \
  --out apps/eval/out/e2-context/reader-retest-001 \
  --reuse-builds apps/eval/out/e2-context/live-001 --live

# No LLM calls: reproduce every logged tool result from its exact snapshot.
.venv/bin/python -m harnext_eval.e2.audit_smoke \
  --run apps/eval/out/e2-context/reader-retest-001 \
  --out apps/eval/out/e2-context/reader-retest-001-audit
```

The reader contract now explicitly requests raw event IDs, matching the strict
grader. The auditor also reports a **separate** narrow formatting diagnostic:
`source#ID` is accepted only if that exact wrapper appeared in retrieved text and
the raw ID belongs to the snapshot. Original scores and model outputs remain
unchanged. This diagnoses provenance formatting; it is not fuzzy matching and
must not silently replace a preregistered confirmation metric.
