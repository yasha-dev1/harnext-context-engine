# harnext evaluation framework

This package implements [`docs/evaluation-spec.md`](../../docs/evaluation-spec.md).
The synthetic corpus, fake reader, fake embeddings, and fake builder harness run
fully offline: no model key, Kafka broker, or application stack is required.

## Merged E2 context experiment: start here

**New: [1,000-instance Kafka benchmark review SPA](benchmarks/kafka-1000-v1/review.html)**
and [dataset methodology / admission status](benchmarks/kafka-1000-v1/README.md).
Contains 800 deterministic historical questions with MCP-only tool contracts and
200 real future-PR implementation candidates with MCP + shell contracts. All gold,
proof and test/reference patches are available for human review. Coding test
execution admission and user approval remain pending; no agent experiment has run.

Current configurable route: [YAML strategies and external MCP evaluation](E2-STRATEGIES.md),
with [verification evidence](STATUS/E2-STRATEGIES-20260912.md). This adds real
BM25/dense/hybrid retrieval, reranking, file/graph strategies and outside-agent
MCP trials. Earlier smoke and synthetic development runners remain available below.

The new context-quality/store experiment is called **E2 (merged E2/E3)**.
Its initial executable gate is documented in
[E2-CONTEXT.md](E2-CONTEXT.md). The older `--experiments e2,e3` commands below
retain the original experiment cards; they do not execute the merged blueprint.

Latest repeat: [fresh smoke report with 20 graphs](../../.harnext/artifacts/e2-smoke-20260912T100821Z.html)
and [results / verification record](STATUS/E2-FRESH-SMOKE-20260912T100821Z.md).

Next development track: [implementation questions and held-out coding tasks](E2-DEVELOPMENT-TASKS.md).
Includes 32 constructed QA probes, five executable coding fixtures, isolated
base/reference validation, and deterministic retrieval span scoring. Real
historical PR selection and Harnext representation mappings remain pending.

```bash
# No network: protocol fixture, with deliberately non-scientific fixture answers.
.venv/bin/python -m harnext_eval.e2.context \
  --config apps/eval/configs/e2-context-offline-smoke.yaml \
  --out apps/eval/out/e2-context/offline-smoke

# Live smoke: requires an authenticated Codex CLI. New output directory per run.
.venv/bin/python -m harnext_eval.e2.context \
  --config apps/eval/configs/e2-context-codex-smoke.yaml \
  --out apps/eval/out/e2-context/codex-smoke --live
```

The live profile pins `gpt-5.6-luna` / `medium` for the builder and reader.
It compares S3 (Codex-curated indexed files), S1 (deterministic templates), and
S0 (raw event files) using Git snapshots, real reader tool decisions, and typed
deterministic grades. It is a synthetic infrastructure gate, **not readiness to
run the complete configuration-selection matrix or claim a winning engine**.

## End-to-end commands

Run these from the repository root:

```bash
uv sync

# Release-gate profile: E1-E6, 120 events, 10 probes/family, reduced E4/E5 matrix.
make eval-smoke

# Comparable S1 report over the same generated replay and frozen probe hash.
uv run harnext-eval run \
  --config apps/eval/configs/s1-templated.yaml \
  --corpus synthetic \
  --all

# Quality gates.
uv run pytest apps/eval/tests -q
uv run ruff check apps/eval
uv run pyright apps/eval/src
```

Both run commands print their run directory and write `manifest.json`, resolved
`config.yaml`, replay/probe JSONL and hashes, store registries, per-seed E1-E6
results, CSV/JSONL artifacts, PNG charts, and a self-contained `report.html`
beneath `apps/eval/out/<run-id>/`.

The other CLI stages can be run independently:

```bash
# Generate the full standalone 2,000-event synthetic corpus.
uv run harnext-eval corpus \
  --output apps/eval/out/corpus/synthetic.jsonl

# Validate/load an existing real EvalEvent JSONL without network access.
uv run harnext-eval corpus --replay /path/to/replay.jsonl

# Generate all six probe families (explicit times are optional and inferred).
uv run harnext-eval probes \
  --replay apps/eval/out/corpus/synthetic.jsonl \
  --out apps/eval/out/probes/synthetic.jsonl \
  --per-family 60 --seed 1

# Build one or more registered layouts through the shared replay driver.
uv run harnext-eval stores \
  --config apps/eval/configs/baseline-minimal.yaml \
  --replay apps/eval/out/corpus/synthetic.jsonl \
  --layouts S0,S1,S4 \
  --out apps/eval/out/stores

# Select experiments with a comma-separated option (repeatable -e also works).
uv run harnext-eval run \
  --config apps/eval/configs/baseline-minimal.yaml \
  --corpus synthetic \
  --experiments e1,e3

# Rebuild a report from completed artifacts.
uv run harnext-eval report apps/eval/out/<run-id>
```

For a generated synthetic `run`, the default comparison profile is 120 events,
12 entities, and 10 probes per family so the exact baseline and S1 commands use
identical replay/probe hashes. Override `--event-count`, `--entity-count`, and
`--per-family` for larger offline studies. Real `--replay` inputs are never
downsampled; the standalone `corpus` and `probes` commands retain their full
2,000-event and 60-probe-per-family defaults.

## What `baseline-minimal` means

`configs/baseline-minimal.yaml` is the lower anchor, not a toy no-op. It uses the
declared-priority/text rules floor with deviation scoring disabled, a 30-second
gap / 20-event / 120-second session window, and the S0 git-backed event dump.
The builder harness, read agent, and 64-dimensional hash embeddings are all
deterministic fakes. The reader still performs real budget truncation and
evidence matching at 2k/8k/32k tokens; stores still create immutable snapshots;
graders, statistics, checks, charts, and reports are the same code paths used by
larger runs. `s1-templated.yaml` changes the configured store to the no-LLM S1
entity projection, leaving the other nudges fixed for a controlled comparison.
