# E1 on the real Kafka corpus — phase plan

Goal: the first evidentiary E1 run (claim C1, routing under a budget) on Corpus R-long,
Apache Kafka, **2019-01-01 → 2026-06-30** (superset of the spec's 2022-01 → 2026-06;
the spec window is the pre-registered primary, the longer window is an extra).
Nothing else (E2–E6) is in scope for this phase. No LLM provider is used anywhere:
E1 has no model in the loop; its harm check is reported N/A-with-reason.

## Sources → replay

| Source | How | Owner | State |
|---|---|---|---|
| KAFKA JIRA (issues, changelog, comments) | `rest/api/2/search`, one JQL window per month of `updated`, raw pages saved to `apps/eval/out/corpus/kafka/raw/jira/` | orchestrator (running) | downloading |
| dev@kafka.apache.org | Pony Mail `mbox.lua` per month → `raw/mail/dev-YYYY-MM.mbox` | orchestrator (running) | downloading |
| GitHub apache/kafka | **GitHub GraphQL/REST API, not GH Archive** (disk: GH Archive is ~300 GB per 6 months; API for one repo is ~1 GB). Token via `GH_TOKEN` env (from `gh auth token`). | agent K1 | to build |
| merge | `python -m harnext_eval.corpus.build_replay --input jira.jsonl --input mail.jsonl --input github.jsonl -o replay/kafka-rlong.jsonl` (already supported) | orchestrator | after K1/K2 |

Event types must match what `corpus/gharchive.py` already emits so `e1/labels.py` predicates work unchanged:
`com.github.pull_request.{opened,merged,closed}`, `.review`, `.review_comment`, `.issue_comment`, `.push`.

## Work packages (disjoint paths; see AGENT_RULES.md)

### K1 — GitHub API extractor (owns: `src/harnext_eval/corpus/github_api.py`, `tests/test_corpus/test_github_api.py`, `tests/test_corpus/fixtures/github-*.json`, `STATUS/K1.md`)
- Resumable: raw GraphQL pages cached to `<raw_dir>/github/…json`; re-running skips fetched pages; parse step is pure (offline, fixture-tested).
- Data per PR: number, title, body, author login + `author_association`, created/merged/closed timestamps, `merged_by` login, base ref, labels, **changed files (path list, first 100+, paginated)**, reviews (author, association, state, submittedAt), review comments (author, association, createdAt, body), issue comments (same). Push/commit events for `trunk` via commit history (sha, author login/email, committedDate, message headline).
- Emit `EvalEvent`s exactly as `gharchive.py` does (same `type`, `source="github:apache/kafka"` or the existing convention, `subject=pr:N`, `data` keys incl. `issue_keys` from `corpus/keys.extract_issue_keys` on title/body/branch, `author_association`, `actor_login`). Keep `data` compact: bodies truncated to 4 000 chars.
- CLI: `python -m harnext_eval.corpus.github_api --repo apache/kafka --since 2019-01-01 --until 2026-07-01 --raw-dir … --output github.jsonl` writing via `build_replay.write_replay`.
- Never call the network in tests; live fetch is run by the orchestrator.

### K2 — E1 real-corpus readiness (owns: `src/harnext_eval/corpus/jira.py`, `corpus/pony_mail.py`, `corpus/committers.py` (new), `configs/kafka-committers.yaml` (new), `configs/e1-kafka.yaml` (new), `src/harnext_eval/cli.py`, `src/harnext_eval/e1/**`, `tests/test_e1/**`, `tests/test_corpus/test_jira.py`, `tests/test_corpus/test_pony_mail.py`, `Makefile`, `STATUS/K2.md`)
1. **JIRA parser vs live data.** `parse_search_page` raises `search snapshot disagrees with changelog final state: fixVersion` on real issues (e.g. KAFKA-294 in the 2024-03 window). Real JIRA changelogs are lossy (bulk edits, deleted versions, renamed components, legacy history). Replace the hard failure with: replay the changelog as the truth, record per-issue `state_inconsistencies: [field…]` in the created-event data, count them in a parse report, never drop the issue. Add a fixture reproducing the mismatch.
2. **Committer roster.** `_is_committer` (`e1/labels.py`) needs `is_committer` / `author_association` on events. Build `configs/kafka-committers.yaml` from https://kafka.apache.org/committers (names + Apache ids + GitHub handles where present; fetch once, commit the YAML, include the fetch date) and `corpus/committers.py` that stamps `is_committer` on JIRA events (by display name / Apache id), mail events (by email local-part / name), and GitHub events (by login or `author_association ∈ {MEMBER, OWNER}`). Report roster match counts in the parse report.
3. **E1-only run path.** `harnext-eval run --config configs/e1-kafka.yaml --replay <kafka-rlong.jsonl> --experiments e1` must: skip probe generation and every store build (E1 needs neither), mark the harm check N/A with reason `no real action provider / no S3 store in this profile`, and finish. Add `--e1-window START END` (or config keys) so the same replay can be evaluated on the pre-registered 2022-01 → 2026-06 window and on the full window.
4. **Scale.** Profile E1 on a 100 k-event synthetic replay (`harnext-eval corpus --event-count 100000 --entity-count 2000 --days 1600`). Known risks: `_OutcomeIndex.by_token` indexes every text token of every event (memory blow-up at 350 k events — restrict to identity-shaped tokens: `kafka-\d+`, `kip-\d+`, `#\d+`, `pr:\d+`, message ids); per-policy per-month refits over the growing tuning prefix (cap the fit window to the trailing 12 months, document it); `scores.parquet` with 8 policies × 4 budgets × 2 populations × 350 k rows (write once per policy×budget with population as a flag, features as JSON string, pyarrow compression). Target: full E1 on 350 k events in < 2 h on 20 cores; use multiprocessing across policies if needed.
5. **Real-corpus labels.** When the replay carries no constructed labels, `build_labels` must run (it does) and the label diagnostics must pass the spec gates (`accuracy ≥ 0.6`, `coverage ≥ 1 %` per function — report which functions fail, do not silently keep them). Make `declared_outcome_agreement` a reported number.
6. **PREREG.** Add `harnext-eval prereg --replay … --config … --out apps/eval/PREREG.md` that records replay SHA-256, config hash, policy list, budgets, primary metric, window, exclusion rules and the current git HEAD; `run` records the PREREG hash in `results.json` and the `prereg_chronology` gate passes when PREREG's git commit predates the run.
7. Keep `make eval-smoke` green, tests green, ruff + pyright clean.

## Run sequence (orchestrator)
1. Raw JIRA + mail complete → parse → `parsed/jira.jsonl`, `parsed/mail.jsonl`.
2. K1 done → live GitHub fetch → `parsed/github.jsonl`.
3. `build_replay` → `replay/kafka-rlong.jsonl` + sha256.
4. K2 done → `harnext-eval prereg` → commit → `harnext-eval run … --experiments e1` for the 2022–2026 window, then the 2019–2026 window.
5. Report: E1 section with real curves; review round by a fresh codex agent.
