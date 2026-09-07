# FX-probes status

## What was built

- Five equally weighted E2 macro families, with configurable link/code subtypes inside multi-source and an evidentiary 300-probe/150-entity validator.
- Corpus-aware state gold for Jira initial/current fields, PR state, KIP votes, answered threads, and OrgForge world-state fields.
- Independent raw-Jira SQLite replay plus a persisted comparison/disagreement/resolution report and the 98% agreement gate.
- Post-merge E2 code-location probes using the union of fixing PR files merged within 14 days of the issue trigger.
- Exact regex-key joins over PR titles, thread subjects, and commit messages; repository/namespace-qualified IDs; audited join precision/recall reports.
- Whole-window realistic abstention candidates and a required A0 non-triviality audit for evidentiary sets.
- Canonical set parsing for PR, thread, commit, and ticket links.

Changed implementation files:

- `apps/eval/src/harnext_eval/probes/{__init__,schema,gen,gold,gen_extraction,gen_temporal,gen_update,gen_multisource,gen_code_location,gen_abstention}.py`
- `apps/eval/src/harnext_eval/grade/links.py`
- Required one-line compatibility change outside ownership: `apps/eval/src/harnext_eval/e2/run.py` routes `multisource`/`files` probes to localisation rather than link grading (its import block was mechanically sorted for ruff).

Changed tests:

- `apps/eval/tests/test_probes/{test_cli,test_generators,test_gold}.py`
- `apps/eval/tests/test_grade/test_exact_links.py`
- Required one-line compatibility assertion outside ownership: `apps/eval/tests/test_core/test_cli_integration.py` expects five, not six, smoke families.

Review audit updates:

- `apps/eval/REVIEW/R1-probes.md`
- `apps/eval/REVIEW/R1-graders.md`
- `apps/eval/REVIEW/R1-e2.md`

## How to run

Smoke (small, deterministic, offline):

```bash
uv run python -m harnext_eval.probes.gen \
  --replay <replay.jsonl> --out <probes.jsonl> \
  --per-family 3 --probe-start <ISO-8601> --probe-end <ISO-8601>
```

Evidentiary profile (requires frozen real/audit inputs):

```bash
uv run python -m harnext_eval.probes.gen \
  --replay <R-H1.jsonl> --out <probes.jsonl> \
  --evidentiary --per-family 60 --minimum-entities 150 \
  --multisource-code-count 30 \
  --raw-jira <jira-search-export.json> \
  --gold-resolutions <gold-resolutions.json> \
  --join-audit <join-audit.json> \
  --a0-audit <a0-answers.json> \
  --probe-start <ISO-8601> --probe-end <ISO-8601>
```

The command writes the frozen JSONL/hash plus `.gold.json` and `.joins.json` audit reports, including reports on a failed evidentiary gate.

## Stubbed or deferred

- The smoke profile intentionally does not run the real 300-probe/150-entity corpus, real raw-Jira export, human join audit, or real-model A0 audit. These are supported through CLI/API inputs and reported as `supported-not-run`/`non-evidentiary-smoke`.
- Findings outside probe/link ownership remain with the E2/E4/core owners, as recorded in the review appendices.

## Open integration questions

- None. Audit formats are deterministic JSON mappings: join audit is `event_id -> expected keys`; A0 audit is `probe_id -> answer`; gold resolutions use the stable request keys emitted in `.gold.json`.

## Verification

- `uv run pytest apps/eval/tests -q` → **220 passed in 55.79s**.
- `uv run ruff check apps/eval` → **All checks passed**.
- Owned regression subset: `uv run pytest apps/eval/tests/test_probes apps/eval/tests/test_grade -q` → **27 passed**.
