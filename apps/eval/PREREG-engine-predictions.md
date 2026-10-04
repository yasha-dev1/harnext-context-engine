# Engine arms on w2026: predictions recorded before the experiment

Recorded 2026-09-28 (before any w2026 trial of the arms `H-engine` / `H-engine-mem`, apart from the six
sanity tasks named below). Source: second-round independent review of the engine at
`identity.code_sha256 = 1a21f39f930e9a0719fc9c8444f12b733b20136951d4ddb590c496bb2b747ef2`, based on the
design and on smoke-benchmark traces only. Purpose: compare with the outcome of the full run
(`w2026-engine-claude-code-haiku-4-5`). Do not edit after the run starts; add the comparison below the line
at the end.

Sanity tasks already run once on w2026 (09:29–09:33, 4 of 6 correct), to be reported separately:
`H-design_decision-074f443eea`, `H-design_decision-1bf0682184`, `H-implicit_link-2816fc05c3`,
`H-pr_approvers-728956e54a`, `H-reopen_count-db0c65201a`, `H-kip_thread-0bd81d2b36`.

## Predictions (reviewer's wording, lightly condensed)

1. **Exact state at an instant** (status, assignee, priority, resolution, fix version, closed): mostly right.
   Misses from choosing `jira_issues` and stopping at the creation row, and from verification declines
   (about 1 in 5). Fix-version sets right unless they exceed 25 rows.
2. **Change histories with timestamps:** timestamps copied verbatim and right when the history is short.
   Failures: items dropped after an "unsupported" rewrite, or findings that cite no rows. At most one of the
   two tasks misses.
3. **Joins across issue tracker / code review / mail:** weak, at best about half.
   - `issue_prs` (3 tasks): mostly wrong (the agent loses the "opened" qualifier).
   - `resolution_vs_merge_order` (3): about 50–70 %.
   - `pr_approvers`, `issue_changed_files`: mostly right.
4. **Aggregations:** counts come out of SQL correctly, but verification cannot check counts and often
   declines them. `author_activity`-type tasks likely declined. `review_hotspot` mostly right.
   `reopen_count` of 0 easy; nonzero counts hinge on exact status values.
5. **Decisions in long mail threads** (29 tasks): the weakest family, about 30–50 %. Expected failures:
   thread not found when there is no KIP id; decision beyond what the 6,000 read characters or the relevance
   selection show; value check strips paraphrased property names and the answer becomes "not determined";
   120 s deadline produces partial answers.
6. **Linking a mail thread to its tracking issue** (14 tasks): about 20–45 %. Branches run in parallel and
   share nothing, so the JIRA agent never sees the thread's subject.
7. **Across the board:** higher p95 latency on the larger w2026 exports and more partial answers.

## Context for the comparison

- Reference arm on the same tasks (main run, 27 Sep): S-tools factual history 97 %, design decisions 86 %,
  implicit links 100 %, coding 83 %; cost per trial $0.037 / $0.181 / $0.041 / $0.445.
- Smoke benchmark, engine alone, 19 tasks: 15–17 correct per run; with the outer agent, 6 of 6 historical
  tasks per engine arm (live smoke 3).
- Development used only the smoke benchmark, which shares 18 of 20 historical question templates with
  w2026; design decisions and implicit links (43 tasks) have no counterpart in it.
- If the engine code changes after this file was written, the new `code_sha256` and the reason for each
  change must be listed here before the run starts.

## Engine changes after these predictions were recorded

(none yet)

---

## Outcome (to be filled in after the run)

Run `w2026-engine-claude-code-haiku-4-5`, 180 trials (90 tasks × `H-engine`, `H-engine-mem`), 28 Sep 2026
17:15 – 29 Sep 01:15 local. Engine `code_sha256` `1a21f39f…` throughout (unchanged since the predictions).
Trials 1–24 on image `a75ed05ffc6e`, 25–180 on image `4615b2564319` (MCP server wrapper changed: it redoes a
research call degraded by provider stalls; runner: trial-scoped pauses). No research call was redone in the
recorded trials (provider-retry cost $0.0038 on one arm). Report of all 14 arms:
`apps/eval/reports/e2/w2026-all-arms/report.html`.

| Question kind | Predicted | H-engine | H-engine-mem | S-tools (main run) |
|---|---|---|---|---|
| Factual history, all (35) | – | 25/35 (71 %) | 25/35 (71 %) | 34/35 (97 %) |
| 1. State at an instant (8) | mostly right | 5/8 | 4/8 | 8/8 |
| 2. Change histories (5) | at most one miss | 4/5 | 5/5 | 5/5 |
| 3. Cross-source / multi-hop (20) | about half | 15/20 | 14/20 | 19/20 |
| 4. Aggregation (2) | often declined | 1/2 | 2/2 | 2/2 |
| 5. Design decisions (29) | 30–50 % | 19/29 (66 %) | 18/29 (62 %) | 25/29 (86 %) |
| 6. Implicit links (14) | 20–45 % | 12/14 (86 %) | 14/14 (100 %) | 14/14 (100 %) |
| Coding (12) | not predicted | 10/12 | 11/12 | 10/12 |

Without the six sanity tasks (72 historical tasks): H-engine 51/72, H-engine-mem 51/72, S-tools 68/72.
All six sanity tasks were answered correctly by H-engine-mem, five by H-engine (miss: `H-implicit_link-2816fc05c3`).

Where the predictions were right: state lookups are the weak spot of factual history (fix version 1/3 and
0/3, priority 0/1), `issue_links` 0/2 on both arms, p95 latency higher than S-tools on factual questions.
Where they were wrong: design decisions and implicit links came out well above the predicted ranges;
`resolution_vs_merge_order` 3/3 and `review_hotspot` 3/3 on both arms; `issue_prs` 3/3 with the memory store.

Mean cost per trial (agent + engine), S-tools in brackets: factual $0.022 / $0.033 [$0.037]; design decisions
$0.032 / $0.028 [$0.181]; implicit links $0.021 / $0.019 [$0.041]; coding $0.461 / $0.408 [$0.445].
Three tasks' questions had reached the engine once before their recorded trial, in trials discarded by the
first pause (`C-KAFKA-PR-22460`, `H-assignee_at_snapshot-26a89e8703`, `H-assignment_history-14e2f44596`).

---

## Second engine experiment (29–30 Sep 2026): NOT pre-registered, tasks already seen

Run `w2026-engine-v2-claude-code-haiku-4-5`, 180 trials, arms `H-engine-b8` (gpt-oss-20b) and `H-engine-b8-ds`
(DeepSeek V4 Flash 0731), engine code `ca68050ca3a0…`, image `1de715f951a3`. Changes against the engine of the
predictions above: tool-call budget feedback (8 calls per branch, the agent is told the count after every call
and must answer from what it has when the budget or the time is used up), research deadline 240 s instead of
120 s, branches run side by side, and for one arm a different chat model. These changes were designed after
analysing the failures of the first engine run on these same 90 tasks; the model was chosen on price. The
results below therefore show what the changes do on tasks used for diagnosis, not held-out performance.

| Category | H-engine-b8 | H-engine-b8-ds | S-tools (main run) | H-engine (first run) |
|---|---|---|---|---|
| Factual history (35) | 28/35 (80 %) | 34/35 (97 %) | 34/35 (97 %) | 25/35 (71 %) |
| Design decisions (29) | 20/29 (69 %) | 20/29 (69 %) | 25/29 (86 %) | 19/29 (66 %) |
| Implicit links (14) | 14/14 | 14/14 | 14/14 | 12/14 |
| Coding (12) | 9/12 | 9/12 | 10/12 | 10/12 |

Mean cost per trial (agent + engine), S-tools in brackets: factual $0.026 / $0.027 [$0.037]; design decisions
$0.029 / $0.029 [$0.181]; implicit links $0.020 / $0.021 [$0.041]; coding $0.472 / $0.468 [$0.445].
Report of all 16 arms: `apps/eval/reports/e2/w2026-all-arms-v2/report.html`.
