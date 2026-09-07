# R3 probes fixer status

## Built

- `src/harnext_eval/corpus/jira.py`: creation-state reconstruction from changelog history and export-snapshot consistency checking.
- `src/harnext_eval/probes/gold.py`: independent SQLite initial-state derivation from earliest raw changelog values.
- `src/harnext_eval/corpus/gharchive.py`: optional PR-object files only; title/head-branch key extraction.
- `src/harnext_eval/probes/{common,gen_code_location,gen_multisource}.py`: exact merge-SHA push fallback, audited PR-key joins, and PR/push provenance.
- `src/harnext_eval/probes/gen.py`: A0 gating on temporal/update/multisource with exact, link-F1, and file-F1 dispatch by `gold_type`; abstention diagnostic and JSON report.
- Regression coverage under `tests/test_corpus/` and `tests/test_probes/`.
- `REVIEW/R2-probes.md`: appended Round 3 finding/change/test ledger.

## How to run

```bash
uv run pytest apps/eval/tests/test_probes apps/eval/tests/test_corpus -q
uv run ruff check apps/eval/src/harnext_eval/probes apps/eval/src/harnext_eval/corpus/jira.py apps/eval/src/harnext_eval/corpus/gharchive.py apps/eval/tests/test_probes apps/eval/tests/test_corpus
uv run pyright apps/eval/src/harnext_eval/probes apps/eval/src/harnext_eval/corpus/jira.py apps/eval/src/harnext_eval/corpus/gharchive.py
```

## Deferred / integration

- E2 scoring is intentionally untouched; code probes expose `family="multisource"` and `gold_type="files"` for the E2 owner.
- No real-corpus or network run was performed.

## Verification

- Scoped: `27 passed`; Ruff clean; Pyright `0 errors`.
- Required full pytest command: `241 passed, 4 failed`; all four failures are in concurrently modified, out-of-scope E1 files.
- Required full Ruff command: 16 out-of-scope E1 findings.
- Required full Pyright command: 3 out-of-scope E1 errors.
