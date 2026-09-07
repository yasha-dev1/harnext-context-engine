# FX-e4 — E4 review fixes

## Built

- `src/harnext_eval/agents/envelope.py`: exact canonical entity resolution, provider-bound rendered-token accounting, distinct V1 N=20/N=100 cells, bounded callable V5 tools, and entity-only V7 distractors.
- `src/harnext_eval/e4/tasks.py`: strict fast-trigger shapes, audited Corpus-R horizons/title joins/identity rules, seeded stratification, complete required IDs, and constructed Corpus-S metadata/world-state gold.
- `src/harnext_eval/e4/run.py`: fixed-S3 enforcement, canonical whole-task leakage gating, scratch builder/E2 batch evaluation, literal Q, evidence validation, costs, judge calibration, paired BCa/McNemar/Holm inference, claim-validity suppression, and seed-spread aggregation.
- `tests/test_e4e5/`: adversarial regression coverage for the reviewed E4 findings.
- `REVIEW/R1-e4.md` and `REVIEW/R1-graders.md`: finding-to-change/test mappings and supported-not-run declarations.

## Run

```bash
uv run pytest apps/eval/tests/test_e4e5 -q
uv run pytest apps/eval/tests -q
uv run ruff check apps/eval
```

Final required results: `223 passed in 55.51s`; Ruff: `All checks passed!`.

## Supported, not run

The offline smoke profile remains intentionally small and fake-provider based. Real Corpus-R/S 150-fast + 150-batch cells, three real S3 seeds, the Opus tier, and human judge/join-audit datasets are configurable and fail closed or report `supported-not-run` when absent; they were not executed in this fix pass.

## Integration notes

- E4 imports shared contracts rather than redefining them.
- The current S3 profile selects the real Anthropic reader; fake output is always marked plumbing-only and cannot publish a primary conclusion.
- No state-changing git command was run.
