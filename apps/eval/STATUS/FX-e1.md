# FX-e1 status

## Built

- `src/harnext_eval/e1/labels.py`: every §4.1 source-specific outcome function, strict post-t relationships, finite-horizon censoring, observed negative votes, and label-model diagnostics/agreement.
- `src/harnext_eval/e1/features.py`: causal four-week 5-minute-bucket baselines, independent 5-minute/hour count ratios and MAD statistics, rolling gap statistics, and window identity.
- `src/harnext_eval/e1/policies.py`: exact R1, truly global R2, causal R3, HBOS terms, guarded R5 with an absolute volume floor/consecutive-window confirmation, and explicit mandatory/eligible monthly admission.
- `src/harnext_eval/e1/score.py`: undefined zero-admission precision and timestamped entity/situation affiliation.
- `src/harnext_eval/e1/run.py`: exact Corpus S sidecar gold, gold stripping, rolling prior-month tuning, global monthly admission, exact report slices, paired clustered primary CIs, delay/onset jitter/affiliation, executable non-smoke gates, named PNGs, attribution, preflights, usage columns, and optional Phase-2 harm rows.
- `tests/test_e1/`: 28 discriminating offline tests covering the above semantics.
- `REVIEW/R1-e1.md`: finding-by-finding fix/defer mapping appended under `## Fixes applied`.

No shared interface in `harnext_eval.types`, `config`, `registry`, or `stores.base` was changed. No git state-changing command was run.

## How to run

```bash
uv run pytest apps/eval/tests/test_e1 -q
uv run pyright apps/eval/src/harnext_eval/e1
uv run ruff check apps/eval

# Smoke E1; generated corpus remains intentionally small.
uv run harnext-eval run \
  --config apps/eval/configs/baseline-minimal.yaml \
  --corpus synthetic --experiment e1 --smoke

# R-long / Flink / other corpora use the existing CLI replay flag; repeat per
# registered corpus so each gets its own manifest and reportable condition.
uv run harnext-eval run \
  --config apps/eval/configs/baseline-minimal.yaml \
  --corpus kafka-r-long --replay /path/to/kafka-long.jsonl --experiment e1
uv run harnext-eval run \
  --config apps/eval/configs/baseline-minimal.yaml \
  --corpus flink-r-long --replay /path/to/flink-long.jsonl --experiment e1
```

For reportable real runs, populate corpus metadata with `prereg_ref`; for Corpus S, include `injected_situations`/`situations` records with event IDs (or entity/onset/end), label/hard-negative, onset/end, entity, and cost weight. Optional completed Phase-2 harm comparisons are accepted as `harm_results`.

## Supported but not run

- Kafka R-long 2022-01 through 2026-06 and the Flink replication: supported via `--corpus` + `--replay`; data was not supplied.
- Corpus S ≥200 situations × three seeds: supported through exact corpus metadata and config seeds; smoke stays small.
- Real Parquet: supported when pandas has a parquet engine. This workspace has neither `pyarrow` nor `fastparquet`; dependency and lock files are outside FX-e1 ownership, so smoke writes the existing explicitly marked JSON-table fallback and does not advertise it as a complete artifact.
- S3/E4 harm execution: E1 consumes and reports completed paired harm rows but does not duplicate unowned store/E4/provider orchestration.
- Live replay-driver convergence, annotation workflow, and leakage-gate execution remain with their owning modules; E1 now exposes the required rule, selected-key, HBOS, volume, window, eligibility, and mandatory evidence.
- VUS remains the local buffered implementation because no pinned reference dependency exists in the allowed workspace. Timestamped situation-aware affiliation is implemented locally.

## Verification

- `uv run pytest apps/eval/tests/test_e1 -q` → **28 passed**.
- `uv run pyright apps/eval/src/harnext_eval/e1` → **0 errors, 0 warnings**.
- `uv run ruff check apps/eval` → **All checks passed**.
- Required `uv run pytest apps/eval/tests -q` → **collection blocked outside FX-e1 ownership**: `harnext_eval.e6.run` is missing, causing three E6 collection errors.
- Diagnostic `uv run pytest apps/eval/tests -q --ignore=apps/eval/tests/test_e6` → **161 passed, 3 unrelated failures**: CLI discovery still imports the missing E6 runner, plus two existing E2/E3 failures (`test_e2_tiny_end_to_end_writes_metrics_and_checks`, `test_e3_same_input_detects_changed_middle_event_with_same_last_id`).

## Open integration questions

- The E6 owner must restore `src/harnext_eval/e6/run.py` before the mandatory all-eval pytest command can collect.
- The replay owner must consume E1’s typed evidence to make live causal decisions match the fixed offline semantics.
- The foundation owner must choose and pin reference VUS/affiliation/Parquet dependencies if externally validated artifacts are mandatory in the base environment.
