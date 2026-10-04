# Evaluation checkpoint — 4 October 2026

This checkpoint preserves the observations needed to make new reports without rerunning model calls. The implementation and E2 trial records are on `eval/e1-jev`. The thesis is in `yasha-dev1/masters-streaming-ai-agent-architecture`, branch `eval/e2-benchmark-review-plan`, directory `thesis/`.

Release: https://github.com/yasha-dev1/harnext-context-engine/releases/tag/eval-checkpoint-2026-10-04

## What is stored

| Location | Contents | Supports |
|---|---|---|
| Git: `.harnext/evaluations/kafka-500-v2/` | 14,694 tracked files across completed, diagnostic and aborted E2 runs; trial outcomes, usage, grading and transcripts | New task/category/arm comparisons, paired outcomes, cost and failure analysis |
| Git: `apps/eval/benchmarks/kafka-50-w2026/` | Frozen task definitions, gold answers and run order | Reconstructing the thesis's 78 historical + first 12 coding tasks |
| Git: `apps/eval/reports/e1-kafka-run4/` and `e1-kafka-run5/` | Derived E1 reports, tables, validity records | Inspecting the published comparisons |
| Release: `e1-run4.tar.gz` | Full run 4, including 846 MiB scores parquet and label votes | Per-event/per-policy reanalysis of the original routers |
| Release: `e1-run5b.tar.gz` | Full Jev run 5b, scores parquet and label votes | Event-level R14/R15 comparisons and new aggregations |
| Release: `e1-jev-decisions.tar.gz` | Jev cached decisions, category/urgency scores, recorded usage and development outputs | Re-scoring semantic routing without new model calls |
| Release: `e1-kafka-corpus.tar.gz` | Original downloaded public Kafka source exports, parsed feeds and merged replay | Auditing source outcomes, text history and alternative label definitions |
| Release: `e2-stores-w2026.tar.gz` | Evaluated stores, build-state records and build artifacts; S-tools retains the five January–May 2026 snapshots used by the comparison | Construction-cost accounting and inspection of retained context |
| Release: `e2-stream-w2026.tar.gz` | Frozen E2 event stream | Auditing the input to the evaluated stores |

E2 headline runs are `w2026-claude-code-haiku-4-5`, `w2026-engine-claude-code-haiku-4-5`, and `w2026-engine-v2-claude-code-haiku-4-5`. Other runs are retained but must not be silently pooled into the 1,440-trial comparison. E1's validity gates remain unresolved. Preservation does not turn exploratory results into validated claims.

The assets contain original bytes under their repository-relative paths. `index.json` lists archive sizes and SHA-256 hashes. Compressed JSON manifests list every member's size and SHA-256. The corpus deliberately retains publicly posted Kafka test-certificate fixtures (including a self-signed “Test Broker 1” example in KAFKA-14237); these are source evidence, not local provider credentials.

## Download, verify, restore

Run from the implementation repository root, using Python 3.11+ and GitHub CLI. These commands do not call any model or overwrite differing local files.

```sh
mkdir -p /tmp/harnext-eval-assets
gh release download eval-checkpoint-2026-10-04 \
  --repo yasha-dev1/harnext-context-engine \
  --pattern '*.tar.gz' --dir /tmp/harnext-eval-assets

for manifest in apps/eval/archives/checkpoint-2026-10-04/*.manifest.json.gz; do
  python apps/eval/scripts/checkpoint_archive.py restore \
    --directory /tmp/harnext-eval-assets --manifest "$manifest" --destination .
done
```

Use `verify` instead of `restore` and omit `--destination` to check downloads without extracting them. A matching existing file is left alone; a differing file stops restoration. Download only the relevant assets if space is limited. To regenerate thesis numbers, E2 trial files from Git and `e2-stores-w2026.tar.gz` are required; the full E1 archives are needed for new event-level analyses, not to render the existing compact figures.

With both repositories checked out as sibling directories:

```sh
cd ../masters-streaming-ai-agent-architecture/thesis
python scripts/extract_results.py
./build.sh
```

Regeneration requires NumPy and Matplotlib; new parquet analyses also require pandas and PyArrow. PDF compilation requires Tectonic or TeX Live. Existing generated figures/tables compile without Python. The checkpoint does not archive local virtual environments, model weights, Docker images, live database services, or every temporary coding checkout; it preserves report/reanalysis evidence, not a complete execution-environment image.

## Recreating assets locally

`checkpoint_archive.py pack --group GROUP --directory EMPTY_OUTPUT_DIR` packages one allowlisted group. It never reruns an evaluation and refuses to overwrite an existing asset. Unit checks:

```sh
python apps/eval/tests/test_checkpoint_archive.py
```
