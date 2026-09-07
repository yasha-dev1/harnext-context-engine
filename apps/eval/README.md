# harnext evaluation framework

This package implements [`docs/evaluation-spec.md`](../../docs/evaluation-spec.md).
The synthetic corpus, fake reader, fake embeddings, and fake builder harness run
fully offline: no model key, Kafka broker, or application stack is required.

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
