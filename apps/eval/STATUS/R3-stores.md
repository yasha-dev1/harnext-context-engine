# R3-stores status

## Built

- `src/harnext_eval/stores/templated.py` now reduces the complete §4 state
  catalogue into S1: Jira state, PR state and changed files, mail answer state,
  KIP vote outcomes, and arbitrary fields from each Corpus-S world-state entity.
- `src/harnext_eval/stores/build_s0.py` now emits a path-only minimal event-file
  index and no curated reader-visible structure.
- `tests/test_stores/test_layouts.py` contains completeness and exact-layout
  regressions that failed against the pre-R3 implementations.
- `REVIEW/R2-stores.md` maps findings 5 and 7 to their changes and tests.

## How to run

```text
uv run pytest apps/eval/tests/test_stores -q
uv run pytest apps/eval/tests -q
uv run ruff check apps/eval
uv run pyright apps/eval/src
```

## Deferred / integration notes

- Real-corpus execution and the 20-folder human review are execution-stage
  evidence; this scoped fix supplies and tests the deterministic S1 projection
  that review will inspect.
- S0 retains the minimal linked `INDEX.md` entrypoint explicitly required by
  the Round 3 assignment. Its only other snapshot files are unlinked input
  provenance under `_meta`.

## Open questions

- None for the assigned store implementation.

## Verification

- `uv run pytest apps/eval/tests/test_stores -q` → **16 passed**.
- `uv run ruff check apps/eval/src/harnext_eval/stores apps/eval/tests/test_stores`
  → **All checks passed**.
- `uv run pyright apps/eval/src/harnext_eval/stores apps/eval/tests/test_stores`
  → **0 errors, 0 warnings**.
- Final owned rerun: **16 passed**, Ruff clean, Pyright **0 errors**.
- Latest `uv run pytest apps/eval/tests -q` → **245 passed, 1 failed**; the one
  failure is in the concurrently edited E1 admission implementation.
- Latest `uv run ruff check apps/eval` → **All checks passed**.
- Latest `uv run pyright apps/eval/src` → **3 errors**, all in concurrent E1
  files. No repository-wide failure points at an R3-stores file.
