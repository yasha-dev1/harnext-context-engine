# FX-core status

## Built

- `src/harnext_eval/stores/base.py`: cumulative event-time high-water snapshots; immutable `delivered.jsonl` fold ledger; exact SHA/fold/ancestry validation; cached `delivery_records(ref)` and `delivered_event_ids(ref)` provenance APIs; safe historical-call store lookup.
- `src/harnext_eval/replay/gate.py`: documented preferred `StoreHandle + T` gate API; exact ledger-boundary proof; replay/question/source/material checks; task-only structured gold checks; canonical audit rows; fail-closed compatibility adapter.
- `src/harnext_eval/replay/driver.py`: prefix-causal budget admission with per-decision audit, unchanged `RouterPolicy` seam, explicit rules-floor accounting, and same-entity batch flush before an immediate fast fold.
- `src/harnext_eval/replay/snapshots.py`: monotone/timezone-aware snapshot-index validation plus Git existence/ancestry checks.
- Regression tests in `tests/test_replay/` and `tests/test_core/test_store.py` cover the previously failing cumulative-commit, contaminated/unresolved ledger, structured leakage, prefix-causality, and fast-before-batch cases.
- Appended finding-to-change-to-test mappings to all six assigned review files.

## How to run

```text
uv run pytest apps/eval/tests/test_replay apps/eval/tests/test_core/test_store.py -q
uv run pytest apps/eval/tests -q
uv run ruff check apps/eval
uv run pyright apps/eval/src/harnext_eval/replay apps/eval/src/harnext_eval/stores/base.py
```

## Deferred / supported-not-run

- `R1-probes.md` item 9 (preregistered probe-hash acceptance) is deliberately not fixed: it belongs to `probes/gen.py`, outside FX-core ownership, and is unrelated to replay/store provenance.
- Real corpora, paid model tiers, large populations/seeds, Kafka, and separate load hosts are execution profiles, not these assigned code defects; the APIs support their callers without enlarging the deterministic smoke profile.

## Open questions

- None for the shared replay/store API. Callers should prefer `leakage_gate(item, store=store, T=item.T, all_events=replay, material=shown_input, ...)`; the positional compatibility form intentionally fails closed if its snapshot cannot be mapped uniquely to a live `StoreHandle`.

## Verification

- `uv run pytest apps/eval/tests/test_replay apps/eval/tests/test_core/test_store.py -q` → **19 passed**.
- `uv run pyright apps/eval/src/harnext_eval/replay apps/eval/src/harnext_eval/stores/base.py` → **0 errors, 0 warnings**.
- `uv run pytest apps/eval/tests -q` → **196 passed in 91.97s**.
- `uv run ruff check apps/eval` → **All checks passed**.
