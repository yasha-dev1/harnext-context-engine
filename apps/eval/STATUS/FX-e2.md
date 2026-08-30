# FX-e2 status

## Built

- `src/harnext_eval/e2/arms.py`: exact entity/baseline ownership, A1-N20/A1-N100 through T, configured A3 embeddings, recursive budgeted S3 traversal, and immutable top-k S4/S5 vector reads.
- `src/harnext_eval/e2/run.py`: canonical leakage-gate integration, exact five-family macro primary, 10,000-draw entity-clustered BCa inference, per-family Holm-adjusted secondaries, exact code-set grading, validity gating, response cache, provider/input/cost accounting, and reviewable artifacts.
- `src/harnext_eval/agents/reader.py`: documented structured material lines and provider-tokenizer accounting for selected material and actual provider input.
- `src/harnext_eval/providers/tokenizer.py`: provider-bound counter seam, Anthropic count-tokens support, and explicitly smoke-only deterministic fake counting.
- `tests/test_e2e3/test_e2e3.py`: regression coverage for every assigned E2 defect.
- Review mappings appended to `REVIEW/R1-e2.md`, `R1-graders.md`, `R1-probes.md`, and `R1-stores.md`.

## Minimal integration changes outside ownership

- `configs/s3-curated.yaml:11`: selects the preregistered Anthropic reader instead of the fake reader.
- `src/harnext_eval/e5/run.py:1003`: consumes the canonical gate column `result` rather than the removed reduced column `status`.
- `tests/test_stores/test_layouts.py:16,326`: the S1 test calls the layout-labelled `store_read`; only S3 is called A4.

## How to run

```bash
uv run pytest apps/eval/tests/test_e2e3 -q
uv run pytest apps/eval/tests -q
uv run ruff check apps/eval
```

## Deferred / supported-not-run

- Real R-H1/OrgForge corpora, 300 probes across 150 entities, paid Sonnet calls, Anthropic token-count calls, and network-backed pinned embeddings were not executed.
- These are supported execution profiles. The fake provider/embedding/tokenizer path stays deterministic, offline, small, and explicitly `non-evidentiary-smoke`; it cannot yield `valid_primary=true`.
- Human pilot, independent-gold, and claim-audit thresholds are accepted through `validation_audit` and fail closed for evidentiary results when absent.

## Verification

- `uv run pytest apps/eval/tests/test_e2e3 -q` → 18 passed.
- `uv run pytest apps/eval/tests -q` → 220 passed in 55.40s.
- `uv run ruff check apps/eval` → all checks passed.

Open integration questions: none.
