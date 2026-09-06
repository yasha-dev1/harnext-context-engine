# Handoff — E1 on the real Kafka corpus (updated 2026-09-06 09:40)

Read this first when resuming on another machine. Branch: `eval-framework`. Everything below is
committed; `apps/eval/out/` (raw corpus, replay, run outputs) is git-ignored and lives only on the
laptop that produced it.

## 1. Where the thesis evaluation stands

| Stage | State |
|---|---|
| 1. Evaluation spec (`docs/evaluation-spec.md`) | done |
| 2. Eval framework (`apps/eval`, E1–E6, 295 tests, offline) | done |
| 3. Offline proof-out on the synthetic smoke corpus | done |
| 4a. Corpus R-long built (Kafka JIRA + dev@ + GitHub API, 2019-01 → 2026-06) | **done** — 629,458 events, SHA `397da1ac…f8a28` |
| 4b. E1 run 1 on the primary window 2022-01 → 2026-07 | **done, exploratory/invalid** — see `apps/eval/REPORT-E1-KAFKA.md` |
| 4c. Independent review of run 1 | **done** — `apps/eval/STATUS/REVIEW-E1-KAFKA.md`, 10 findings, 2 blockers |
| 4d. Amendments + run 2 (registered `apps/eval/PREREG-run2.md`) | **done** — verdict unchanged; `apps/eval/reports/e1-kafka-run2/`, §0 of `REPORT-E1-KAFKA.md` |
| 4e. Label-fusion sensitivity, human sanity-sample sheets | **done** — `reports/e1-kafka-label-sensitivity/`, `reports/e1-kafka-human-sample/annotation_sheet.csv` (200 items, two annotator columns) |
| 5. Evidentiary runs for E2–E6 | not started |

## 2. Run 1 result in one paragraph

Primary metric recall@2 % on rule-negative events (378 positives, 103 entity clusters):
R5 (ours) 0.000 = R1; R2 global HBOS 0.071; R0 random 0.024; R5 − R2 = −0.071 [−0.103, −0.047].
Mechanism: the rules floor flags 2.32 % of events and saturates b ≤ 2 % in 33/52 months (R5 ≡ R1);
above that the guards leave only 0.74 % of rule-negatives eligible and none is positive; the
per-entity HBOS scorer without guards (R4) is below random. Label prevalence is 0.11 %.
Published report: https://claude.ai/code/artifact/8fcf3f48-6f39-4d4f-a9e0-e8fe154d087e

The review reproduces every number but finds the evaluation not yet sound: (1) blocker —
`jira_fix_version_in_flight_later` needs release fields no event has and cast 145k negative
votes; (2) blocker — rule/feature/label text is export-time snapshot text (no proof it existed at
*t*); (3) the label model's symmetric negative voting makes urgency a conjunction of rare outcomes,
0.95 accuracy is a cap; (4) declared-priority rule missed `→ Critical` transitions, `[VOTE]` fired
on replies; (5) affiliation/NAB use row-index distance; (6) human sanity sample and metric
remediation record missing.

## 3. What was changed for run 2 (commits 84405fa, 9f49f3c)

- `e1/labels.py`: in-flight-release LF abstains unless release-state fields exist in the corpus.
- `e1/policies.py`: changelog `field=priority, to∈{Blocker,Critical}` → `declared_priority`;
  `RuleSettings.vote_thread_start_only` (config `engine.router.rules.vote_thread_start_only`).
- `configs/e1-kafka.yaml`: `vote_thread_start_only: true`; `e1.exclude_label_functions` =
  `[github_trunk_ci_failure_fix_6h, jira_fix_version_in_flight_later]`.
- `config.py`: `e1.exclude_label_functions` knob; recorded in PREREG and in `results.json`
  `check_details.excluded_label_functions`.
- `scripts/e1_label_sensitivity.py`: applies the LFs once and reports prevalence under
  alternative fusion rules (post-hoc; feeds the label-definition decision, §5).
- Label fusion model unchanged in run 2 on purpose.

## 4. Reproducing the corpus on a new machine (≈ 4 h wall, ≈ 3 GB disk)

```sh
# JIRA (~20 min) + dev@ mail (~40 min) + parse (~10 min)
UV_NO_SYNC=1 uv run python apps/eval/scripts/fetch_kafka_jira_mail.py jira mail parse
# GitHub GraphQL, resumable, ~3 h at 5 000 req/h (20 789 cached pages)
GH_TOKEN=$(gh auth token) UV_NO_SYNC=1 uv run --package harnext-eval python -m harnext_eval.corpus.github_api \
  --repo apache/kafka --since 2019-01-01 --until 2026-07-01 \
  --raw-dir apps/eval/out/corpus/kafka/raw --output apps/eval/out/corpus/kafka/parsed/github.jsonl
uv run python -m harnext_eval.corpus.build_replay \
  --input apps/eval/out/corpus/kafka/parsed/jira.jsonl --input apps/eval/out/corpus/kafka/parsed/mail.jsonl \
  --input apps/eval/out/corpus/kafka/parsed/github.jsonl --output apps/eval/out/corpus/kafka/replay/kafka-rlong.jsonl
sha256sum -c apps/eval/out/corpus/kafka/replay/kafka-rlong.jsonl.sha256   # must be 397da1ac…f8a28
```

Alternative: copy `apps/eval/out/corpus/kafka/{parsed,replay}` (≈ 2.6 GB) from the laptop.
The GitHub snapshot is time-dependent (PR bodies, associations), so a re-fetch may not reproduce
the SHA exactly; if it differs, register a new PREREG (the `prereg` command refuses to overwrite).

## 5. Run sequence

```sh
# run 2 (already registered; PREREG-run2.md is committed)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HARNEXT_E1_WORKERS=4 UV_NO_SYNC=1 \
uv run harnext-eval run --config apps/eval/configs/e1-kafka.yaml \
  --replay apps/eval/out/corpus/kafka/replay/kafka-rlong.jsonl \
  --experiments e1 --window 2022-01-01 2026-07-01 --prereg apps/eval/PREREG-run2.md
```

≈ 2 h, 15.3 GB peak RSS for 387k events; launch with `setsid nohup … < /dev/null & disown`
(two reboots killed runs on 2026-09-05). The secondary window 2019-01 → 2026-07
(`PREREG-full.md`) needs ≈ 24 GB free and has not been run.

Outputs: `apps/eval/out/<ts>-e1-kafka/e1/seed-1/{results.json,metrics.csv,label_diagnostics.csv,
validity.csv,calibration.csv,robustness.csv,scores.parquet}` and `report.html`. Copy the small
files into `apps/eval/reports/e1-kafka-run2/` and commit them; `scores.parquet` (100 MB) stays out.

## 6. Run 2 in one line

Rule share 2.32 % → 1.55 %, infeasible months at 2 % 33 → 13, positives 378 → 815; R5 still 0.000
(R2 0.108, random 0.010); R5 − R2 = −0.092 [−0.115, −0.072]; calibration inverted (ρ = −0.53).
The guards, not the budget, are the binding constraint. Non-evidentiary until the human sample
and remediation record exist.

## 7. Decisions still open (Yasha)

1. **Label definition.** Registered: weighted model, p ≥ 0.5 → 0.11 % prevalence. Sensitivity job
   output (`apps/eval/out/label-sensitivity/fusion_prevalence.csv` on the laptop) gives prevalence
   under: registered model 0.23 %; strict majority 0.81 %; ≥ 2 outcome LFs 3.2 %; ≥ half 3.9 %;
   any outcome LF 18.6 % (committed in `reports/e1-kafka-label-sensitivity/`). **Next action:**
   annotate `reports/e1-kafka-human-sample/annotation_sheet.csv` (two people, yes/no/unsure), then
   score κ with `annotation_key.csv` as described in its README; pick the label rule; register run 3.
2. **If run 2 still shows R5 ≤ random:** C1 as designed is falsified. Next are two *new*
   pre-registered policies: budgeted (non-mandatory) rule floor; guard-free scoring on
   low-density entities. Do not tune R5's guards on these months.
3. **Provenance blocker:** components as baseline keys should be rebuilt as-of-event from
   changelog transitions (`corpus/jira.py`); comment/PR body edits cannot be recovered from the
   APIs and must be stated as a limitation.

## 7. Files that matter

`apps/eval/REPORT-E1-KAFKA.md` · `apps/eval/reports/e1-kafka/` · `apps/eval/STATUS/REVIEW-E1-KAFKA.md`
· `apps/eval/STATUS/{K1,K2}.md`, `K2-parse-report.md` · `apps/eval/PREREG.md`, `PREREG-run2.md`,
`PREREG-full.md` · `apps/eval/PROMPTS/{K1,K2,K2-resume,REVIEW-E1-KAFKA}.md` (codex briefs; launch
pattern in §5 of the previous handoff, staggered by ~30 s).
