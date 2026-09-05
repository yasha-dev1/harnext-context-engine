# Handoff — E1 on the real Kafka corpus (written 2026-09-05)

Read this first when resuming on another machine. It summarises the 5 Sep session, the
decisions taken, what is running/blocked, and the exact next steps. Branch: `eval-framework`.

## 1. Where the thesis evaluation stands

| Stage | State |
|---|---|
| 1. Evaluation spec (`docs/evaluation-spec.md`, masters repo `eval/evaluation.md`) | done |
| 2. Eval framework (`apps/eval`, E1–E6, YAML-configured engine, 258 tests, offline) | done, pushed (`6b0b8ac`) |
| 3. Offline proof-out on the synthetic smoke corpus | done — see verdicts below |
| **4. First real run: E1 on Corpus R-long (Kafka)** | **in progress — this handoff** |
| 5. Evidentiary runs for E2–E6 (real model, Kafka R-H1, PREREG, human κ gates) | not started |

Smoke-run verdicts (fake providers, 140 synthetic events, nothing evidentiary):
E2 / E3 / E4 discriminate between arms (E3: S0 0.20 < S1 0.46 < S4 0.52 < S3 0.62 at 32k tokens);
E1 and E5 are degenerate at that size; E6's primary is correctly stamped INVALID.

Artifacts (claude.ai/code): eval proof-out with charts `cb233702-…`, experiments guide E1–E6
`882cefc9-…`, full E1 report of the smoke run `e0b9e726-…`, build report `09e1a896-…`.

## 2. Decision: E1 first, on real data, no LLM anywhere

Yasha's instruction: get real E1 results now ("prove the classifier for batch lane and fast
lane"), other experiments later. E1 has no model in the loop (its harm check is the only LLM
touchpoint and is reported N/A-with-reason in this phase). No provider API key is used.

Corpus window: **2019-01-01 → 2026-06-30**, a superset of the spec's 2022-01 → 2026-06.
The spec window stays the pre-registered primary; the longer window is an extra (more
evaluation months, more revealed positives). Spec amendment to record in PREREG.

## 3. Why the smoke E1 said nothing about C1 (analysis of run 20260831T005148Z)

- 110 evaluated events over 7 months → 10–20 events/month → 2 % budget rounds to **one slot
  per month**, and rule hits are mandatory, so the slot was gone before any deviation
  admission was considered. R5 ≡ R1 at every budget ≤ 5 %.
- 0.58 events/day → the volume guard (≥ 3 events in a 5-min bucket) was satisfied for 10 of
  105 rule-negative events. Real Kafka is ≈ 250 events/day.
- Exactly one rule-negative positive → every paired contrast NaN (n_entities = 1).
- **Harm check artefact:** all five harm deltas are exactly −0.5, Q(now) = 0 every time (fake
  provider; the state file likely does not exist at admission time). Must be fixed before any
  real-provider harm run.
- **Weak-label path never exercised on a replay:** the synthetic corpus supplies constructed
  labels, so the 13 revealed-urgency functions have coverage 0 (only unit-tested).
- R5 in the eval (`e1/policies.py:GuardedHBOSPolicy`) is a reference implementation; the
  production classifier (`apps/classifier/.../anomaly.py`) is arm R3 (gap-only robust-z).
  If R5 wins, the classifier must be upgraded to match.

## 4. Corpus acquisition — what exists, what is running

**GH Archive is dropped.** It is the whole GitHub firehose (~300 GB per 6-month window, most
discarded). GitHub comes from the GitHub GraphQL API for the single repo `apache/kafka`
(~1 GB). Spec amendment.

| Source | Mechanism | Status on the laptop (5 Sep) |
|---|---|---|
| KAFKA JIRA (changelog + comments) | `apps/eval/scripts/fetch_kafka_jira_mail.py` — one JQL window per month of `updated`, raw pages to `apps/eval/out/corpus/kafka/raw/jira/`, resumable (`.done` markers) | complete through 2026-06 |
| dev@kafka.apache.org | same script, Pony Mail `mbox.lua` per month → `raw/mail/dev-YYYY-MM.mbox` | was at 2023-09 when this handoff was written; resumable |
| GitHub apache/kafka | **to be built** by agent K1 (`corpus/github_api.py`) | not started (codex auth failed) |
| merge | `python -m harnext_eval.corpus.build_replay --input jira.jsonl --input mail.jsonl --input github.jsonl -o apps/eval/out/corpus/kafka/replay/kafka-rlong.jsonl` (existing) | after the above |

`apps/eval/out/` is git-ignored: **raw data is not in the repo.** On a new machine, re-run the
script (`UV_NO_SYNC=1 uv run python apps/eval/scripts/fetch_kafka_jira_mail.py jira mail`);
JIRA takes ~20 min, mail ~40 min. Disk: ~2 GB for raw + parsed + replay; E1 outputs a few GB
per run. GitHub API needs `GH_TOKEN=$(gh auth token)` (5 000 req/h).

Verified live: JIRA REST 200 (7 153 issues created 2022–2026; 9 138 updated since 2022),
Pony Mail 200 (~640 msgs/month), GitHub API 200 (23 359 PRs all-time).

**Known real-data failure already reproduced:** `corpus/jira.py::parse_search_page` raises
`Jira issue KAFKA-294 search snapshot disagrees with changelog final state: fixVersion` on
`raw/jira/2024-03-p000.json`. Real changelogs are lossy; the parser must record, not raise.
(K2 item 1.) The mail parser has not yet been run on a real month.

## 5. Work packages for codex agents (briefs are committed)

- `apps/eval/PLAN-E1-REAL.md` — the phase plan with disjoint ownership.
- `apps/eval/PROMPTS/K1.md` — GitHub API extractor (cached, resumable, same event shapes as
  `gharchive.py`, fixture tests only).
- `apps/eval/PROMPTS/K2.md` — E1 real-corpus readiness: tolerant JIRA parser + parse report
  over all raw pages; Kafka committer roster (`configs/kafka-committers.yaml`,
  `corpus/committers.py`, stamps `is_committer` — the "committer replied" label functions
  need it and nothing provides it today); E1-only run path (skip probes/stores, harm N/A,
  `--window`); scale to 350 k events (label token index restricted to identity tokens,
  trailing-12-month fit window, compact `scores.parquet`, multiprocessing); real-corpus label
  gates; `harnext-eval prereg` command + `prereg_chronology` gate; `make eval-e1-kafka`.

Launch pattern (from repo root, keys scrubbed, stdin closed — codex blocks on stdin otherwise):

```
for k in K1 K2; do
  nohup env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u OPENROUTER_API_KEY -u GH_TOKEN \
    codex exec --dangerously-bypass-approvals-and-sandbox -C "$PWD" \
    "$(cat apps/eval/PROMPTS/$k.md)" < /dev/null > .codex-logs/$k.log 2>&1 &
done
```

**Blocker on 5 Sep:** both launches died with `Your access token could not be refreshed
because your refresh token was already used. Please log out and sign in again.` Run
`codex logout && codex login` before relaunching. (Launching two agents in the same second
may have raced the token refresh; if it recurs, stagger launches by ~30 s.)

## 6. Run sequence once K1/K2 land

1. Parse: `UV_NO_SYNC=1 uv run python apps/eval/scripts/fetch_kafka_jira_mail.py parse`
   → `parsed/jira.jsonl`, `parsed/mail.jsonl` (needs K2's parser fix).
2. GitHub: `GH_TOKEN=$(gh auth token) uv run python -m harnext_eval.corpus.github_api --repo apache/kafka --since 2019-01-01 --until 2026-07-01 --raw-dir apps/eval/out/corpus/kafka/raw --output apps/eval/out/corpus/kafka/parsed/github.jsonl`
3. Merge → `replay/kafka-rlong.jsonl` + `.sha256`.
4. `uv run harnext-eval prereg --replay … --config apps/eval/configs/e1-kafka.yaml --out apps/eval/PREREG.md`; commit PREREG.
5. `uv run harnext-eval run --config apps/eval/configs/e1-kafka.yaml --replay … --experiments e1 --window 2022-01-01 2026-07-01`; then the full 2019–2026 window.
6. Fresh codex review agent over the E1 outputs; E1 report section with real curves
   (recall@2 % rule-negative R5 vs R1/R2 with entity-clustered CIs, calibration by decile,
   lift over rules, per-source recall, label-function diagnostics, declared-vs-outcome
   agreement, robustness).
7. Then: fix the harm-check artefact; Flink replication if time; E2–E6 real-model phase.

## 7. Files touched this session

- `apps/eval/PLAN-E1-REAL.md`, `apps/eval/PROMPTS/K1.md`, `apps/eval/PROMPTS/K2.md`,
  `apps/eval/scripts/fetch_kafka_jira_mail.py`, this file.
- Local only (not committed): `apps/eval/out/corpus/kafka/{raw,fetch.log}`, `.codex-logs/`.
