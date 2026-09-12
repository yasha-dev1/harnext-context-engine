# E2 development extension: verified implementation state

Added September 12, 2026. Protocol: `e2-development-tasks-v1`.

The [protocol and next experiment](../E2-DEVELOPMENT-TASKS.md) add implementation
understanding, withheld-change coding tasks, and actual supporting-span retrieval
metrics. This is a constructed development package; no real historical PR or
new live model performance is claimed.

## Dataset and executable gate

- 32 QA questions across pagination, idempotency, retry policy and snapshot
  visibility: 16 contract/history/absence and 16 applied scenarios.
- 5 independent coding tasks: four fixes/policy implementations and a new batch
  planning feature using separate worker and planner modules.
- 29 annotated context paragraphs, including superseded decisions and 12
  distractor documents; separate public inputs and private evaluator artifacts.
- Native model tools disabled; host-mediated repository and immutable context
  tools; protected visible tests; final held-out tests after agent completion.
- Local `candidate.patch`, `PR.md`, source files, test results, provider traces,
  exact context responses and deterministic retrieval measurements are produced
  by the live trial entrypoint. That entrypoint has not been run against a live
  model as part of this change.

The final [dataset export](../out/e2-development/dataset-v3-20260912/manifest.json)
and [container validation](../out/e2-development/validation-v3-20260912/manifest.json)
are preserved. Earlier v1/v2 exports are intermediate development records.

| Coding fixture | Failing base tests fixed by reference | Base tests preserved | Reference tests passing |
|---|---:|---:|---:|
| Pagination | 3 | 2 | 5 |
| Idempotency | 3 | 1 | 4 |
| Retry policy | 4 | 1 | 5 |
| Snapshot visibility | 3 | 2 | 5 |
| Batch planning feature | 2 | 2 | 4 |

All five admissions passed. These are individual unittest method counts, not
counts of assertions or independent research samples. Base failures were test
assertion failures, not missing imports or broken test infrastructure.

Docker used a pinned Python image resolved to local immutable ID
`sha256:ec7d6c95cd3692a2e2d228a8b1ca74e4025b54121fcc4c5da6f09cfa473315ad`.
Each execution used a fresh non-root, network-disabled container, read-only
source/tests, dropped capabilities, and memory/process/time limits.

## Verification

- Evaluation + builder regression suite: **370 passed**, 10 existing warnings.
- Focused new tests after the final feature/runner edits: **7 passed**.
- All five base/reference pairs validated in fresh containers after final edits.
- Ruff passed for all new modules/tests; targeted Pyright passed using the
  workspace interpreter.
- Tests cover evidence IDs without supporting text, partial reads, repeated
  byte charging, search wrappers, stale annotations, corrupted replay, alternative
  support, absent gold, public/private export separation, test-edit rejection,
  and hidden-test grading only after the scripted agent finishes.

The scripted provider in a unit test is an infrastructure fixture, not a model
performance result. The full suite preceded the final small runner/type and
fifth-task additions; those were checked by the focused tests, final container
validation and targeted type/lint checks above.

## Remaining gates before the real PR study

Real repository/PR selection; historical source/cutoff audit; reproducible
repository-specific tests and dependencies; curated Harnext and retrieval/graph
snapshot adapters with reviewed evidence mappings; full trial/token/output
limits; test-report integrity hardening; measured live coding pilots; and frozen
paired development/confirmation populations. The runnable coding conditions
today are no context and raw immutable development context.
