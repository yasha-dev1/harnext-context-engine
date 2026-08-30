# FX-e3 status

## Built

- `src/harnext_eval/e3/run.py`: immutable store conditions; mandatory S0/S1/S3/S4 validation; layout-specific snapshot retrieval including real S4/S5 vector queries; replay-hash plus authoritative delivered-ID proof/diff; fixed family-stratified erosion panel with checkpoint gold re-derivation and leakage exclusions; exact five-family entity-clustered BCa inference at 10,000 resamples; single-seed McNemar; within-run Sonnet seed and health spread; optional Opus isolation; resource/floor/failure/fairness checks; and exact root-level E3 artifacts.
- `src/harnext_eval/e3/__init__.py`: exports `StoreCondition` for callers without changing shared interfaces.
- `src/harnext_eval/cli.py`: builds E3's mandatory baseline stores once, S3 once per configured seed, optional S2/S5, and optional Opus seed-1; invokes one aggregate E3 run instead of one E3 run per seed. `--e3-optional-stores` and `--e3-opus-model` expose the optional conditions.
- `tests/test_e2e3/test_e2e3.py`: real S0/S1/S3/S4 integration, natural-question S4 retrieval/recall, two primary contrasts, three explicit S3 seeds, 10,000-resample evidence, ledger-middle mismatch, checkpoint re-derivation, mandatory-arm failure, and CLI smoke/Opus registration tests.
- `REVIEW/R1-e3.md` and `REVIEW/R1-stores.md`: appended finding-to-code-to-test mappings under `## Fixes applied`.

One minimal compatibility change was made to the E2 smoke test in the otherwise shared `test_e2e3` file: its cast-only `ToyStore` was replaced by the existing real S3 fixture helper after the shared core leakage gate began requiring authoritative `StoreHandle` delivery provenance. No E2 production code or shared interface was changed.

## How to run

```bash
uv run pytest apps/eval/tests/test_e2e3 -q
uv run pytest apps/eval/tests -q
uv run ruff check apps/eval

uv run harnext-eval run \
  --config apps/eval/configs/baseline-minimal.yaml \
  --experiment e3 --smoke

# Optional scientific conditions (non-smoke, appropriate online config):
uv run harnext-eval run \
  --config /path/to/e3-online.yaml \
  --experiment e3 \
  --e3-optional-stores S2,S5 \
  --e3-opus-model claude-opus-5
```

## Supported but not run

- Real Kafka/OrgForge corpora and the preregistered 300-probe/150-entity population are accepted through the existing replay/probe inputs; no such corpus was supplied.
- Three paid Sonnet builds are driven by `seeds: [1, 2, 3]`; the separate Opus seed-1 condition is supported by config/`--e3-opus-model`. The offline smoke intentionally used one deterministic fake-provider seed and labelled seed reliability and Opus as `supported-not-run`.
- The fixed production erosion panel size is 60; smoke may use a smaller deterministic panel and records its IDs/hash plus `supported-not-run` status.
- Human pilot/S1-folder-review evidence and frozen real-provider prices/model revisions remain external preregistration inputs. Smoke does not claim those human/paid gates passed.

## Open integration questions

- A future multi-corpus orchestration layer can combine separately frozen Corpus R and Corpus S E3 tables into one cross-corpus report. Every table now carries a corpus column, and each per-corpus run writes the required named artifacts.
- Store writers own any provider-specific wall-clock or embedding-cost fields absent from usage transcripts; E3 reports missing applicable usage rather than silently treating it as a successful free build.

## Verification

- `uv run pytest apps/eval/tests/test_e2e3 -q` → **9 passed in 8.68s**.
- Offline E3 command with 40 events, 6 entities, and one probe per generator → completed successfully; wrote report plus `curve.png`, `erosion.png`, named CSVs, `ledger_diff.json`, and the frozen panel artifact.
- `uv run pytest apps/eval/tests -q` → **196 passed in 94.96s**.
- `uv run ruff check apps/eval` → **All checks passed**.
