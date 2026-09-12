# Kafka Context Benchmark: 1,000 instances for review

Open [review.html](review.html). It is an offline SPA containing all 1,000 tasks,
with searchable categories, exact agent prompts, capability contracts, expected
answers, frozen source evidence, live proof links, test patches and reference PRs.
It includes local per-instance review decisions/notes and JSON export/import.
This is reviewer-only material: it contains the answers and solution patches.

The benchmark contains **1,000 distinct issues**, **998 issue/PR lineages**,
and no paraphrase multiplication. All instances come from **apache/kafka**.
The corpus is large enough for this first release; repository diversity has not
been established. Generation and selection use no LLM.

| Category | Instances | Capabilities | Deterministic target |
|---|---:|---|---|
| Assignee at snapshot | 100 | Harnext MCP only | Exact display name or null |
| Status at snapshot | 100 | Harnext MCP only | Exact status label |
| Resolution at snapshot | 100 | Harnext MCP only | Exact resolution label or null |
| Closed at snapshot | 100 | Harnext MCP only | Boolean; exactly 50 Closed and 50 other statuses |
| Priority at snapshot | 100 | Harnext MCP only | Exact priority label |
| Status transition history | 100 | Harnext MCP only | Ordered timestamp/from/to objects |
| Assignment history | 100 | Harnext MCP only | Ordered timestamp/from/to objects |
| Resolution transition history | 100 | Harnext MCP only | Ordered timestamp/from/to objects |
| Historical PR implementation | 200 | Harnext MCP + sandbox shell | Acceptance/regression tests, after admission |

## What is ready and what is pending

The full review dataset exists. Every historical answer has been independently
recomputed from raw Jira changelog entries. All 200 coding tasks have an actual
pre-PR base commit, reconstructed issue body, real visible test patch and separate
reference implementation patch. Both patches applied successfully to files fetched
at that base SHA. Diff line counts were checked against the API's additions and
deletions, and base files and patches have content hashes.

**The 200 coding tasks are candidates awaiting execution admission.** We have not
run their Gradle environments, demonstrated failing base tests, or shown that the
reference passes at the earlier base. Proposed commands are displayed for review;
they may need module/task/environment corrections. Eight historical development
tasks have since run through native Codex with and without MCP; see the
[paired pilot results](../../STATUS/E2-NO-MCP-PILOT-20260912.md). The frozen review
bundle remains unchanged and does not serve as a live execution ledger.
The old synthetic Python coding grader does not execute these Kafka Java/Scala
tasks. Do not treat patch applicability as compilation or behavioral validation.

The confirmation split is a **candidate** split: 694 development / 306 confirmation.
Connected issues mentioned in PR titles remain together, including multi-issue
PRs. This does not resolve every possible duplicate/backport/epic lineage; that
needs review. Because this SPA deliberately exposes all gold to the reviewer, it
does not claim a sealed confirmation set against reviewer-driven selection.

## Deterministic construction

The frozen inventory contains 12,670 issues, 17,795 PRs and 12,057 trunk-history
commits. Raw input hashes are recorded in `cache/inventory.json`.

1. Historical questions require complete changelogs and internally consistent
   field histories. Choose a source-recorded transition cutoff, compute the
   answer at that cutoff and retain exact proof entries. Exclude later entries
   from the task evidence and filter them before building a context snapshot.
2. Stratify candidate selection by year using the seed in `benchmark.yaml`.
   Status/resolution categories also stratify over answer labels; Closed has an
   explicit 50/50 quota. Prefer histories with at least three events where available.
   There is one benchmark task per issue across all categories.
3. Coding candidates require a merged trunk PR linked by title to one known Jira
   issue, 2–8 changed Java/Scala implementation/test files, both production and
   test changes, and at most 600 changed lines. This intentionally bounds the
   first release's change size; it excludes large architectural PRs and other
   languages. No claim of comprehensive Kafka change coverage is made.
4. Select the latest cached trunk-history commit strictly before PR creation,
   requiring that the issue already existed. Reverse later summary/description
   edits to reconstruct the issue text at that base timestamp. The actual solution
   PR is future information relative to the provided base.
5. Fetch the immutable merged-commit diff and base files. Separate test from
   implementation changes and verify patch application. Of 312 checked PRs, 91
   failed static admission: 78 patch conflicts, 11 missing base files, two missing
   textual patches/unsupported changes. Further repeated issues are skipped.
6. Freeze the 1,000-instance bundle and public/private exports. A rerun over the
   same frozen inputs and seed must reproduce the task hash. Network acquisition
   is a separate cached phase; transient failures must be reviewed before treating
   the acquisition pool as a scientific sampling frame.

Historical observation/ingestion times were not recorded. This release uses
source-recorded Jira/commit times as a declared proxy. It tests historical state
under that reconstruction, not a measured online-ingestion timeline. Current
issue pages may show later state; the frozen changelog text, history ID and item
index identify the evidence used. Live reference links are for human inspection.

## Tools and outside harnesses

`benchmark.yaml` declares two capability contracts:

- **Historical:** only the Harnext MCP endpoint. No shell, native filesystem,
  browser, network, other MCP servers or subagents. Restrict native harness tools
  in the launcher and isolate its workspace. Prompt wording alone is insufficient.
- **Coding:** Harnext MCP plus shell in an isolated source workspace. Stage a
  source export without `.git`, apply the visible test patch, and preload admitted
  build dependencies. No network during the trial. Keep gold, reference patches,
  review files and the evaluator outside that sandbox. Reapply pristine grading
  tests in the evaluator rather than trusting tests modified by the agent.

Different harnesses must receive the same task, source tree, visible tests and
capability contract. Record actual native launch configuration/tool traces;
reported model labels are not proof of execution identity. Additional MCP tool
subsets are experiment treatments within these contracts. Enabling shell on a
historical task violates its contract, regardless of harness.

`context-profile.yaml` is a shared engine starting profile; set both clocks and
the task/run IDs from a selected instance and choose an immutable artifact path.
`data/context-sources.jsonl` contains structured Jira transitions plus historical
commit/PR/file relationship records, including distractor issues. Do not replace
it with only the few gold evidence records. Future records must be filtered at
build time. File/graph/retrieval treatments still come from the shared strategy YAML.

This release supplies capability contracts and review data, not an already audited
launcher for every external harness. Launcher enforcement, negative capability
tests and exact representation-specific evidence mappings are admission gates.

## Scoring and interpretation

Historical correctness uses typed exact JSON. Arrays of history objects are
ordered; booleans do not equal integers. `benchmark_audit.score_answer` implements
this contract. The older QA prototype's string-set grading is not appropriate
for these nested history arrays.

Retrieval metrics require server-recorded returned bytes mapped to reviewed
evidence spans in each frozen engine representation. Task evidence IDs and
source quotes provide the annotation starting point; returning an ID alone earns
no evidence credit. Report storage coverage separately from reader retrieval.

Coding correctness is test behavior, not equality with the real PR. Admit base
and reference against the same test tree, derive fail-to-pass/pass-to-pass sets,
then grade submitted patches in a clean environment. Real-PR changed-file overlap
and diff statistics are secondary descriptive comparisons, never correctness gates.
Visible tests are explicitly supplied future information by task design. Additional
private tests would need separate curation; none are claimed here.

The issue body is already in each coding prompt. Merely retrieving it again does
not demonstrate useful extra context. Review history dependence and compare
repository/tests-only versus MCP-enabled coding conditions. Many state questions
are calibration items; retain their scores separately from history reconstruction
and coding difficulty. Public historical PRs may have appeared in model training.

## Review and reproduce

Use Approve / Needs changes / Exclude and notes on each detail page. Export reviews
when finished; decisions are bound to the exact dataset hash. Importing another
dataset's review file is rejected. No review action triggers execution.

Run from the engine repository root:

```bash
.venv/bin/python -m harnext_eval.e2.benchmark inventory
# Network acquisition, cached, no agent/model calls:
.venv/bin/python -m harnext_eval.e2.benchmark acquire
# Deterministic local generation:
.venv/bin/python -m harnext_eval.e2.benchmark generate
.venv/bin/python -m harnext_eval.e2.benchmark_audit validate
.venv/bin/python apps/eval/scripts/build_benchmark_review.py
```

The HTML builder writes through `apply_patch` and embeds compressed data. The SPA
has no remote dependencies and can be opened as a local file in a modern browser.
Browser checks and screenshots are in `data/`; the browser test script uses an
isolated temporary Playwright installation and does not alter your review state.

Key outputs: [manifest](data/manifest.json), [public tasks](data/tasks.public.jsonl),
[private reviewer bundle](data/review-bundle.json), [source audit](data/validation.json),
[browser verification](data/browser-validation.json). The review SPA, frozen
task exports, reports and compressed context/audit inputs are versioned with this
draft. Downloaded model weights and the full working API cache remain local.

On a fresh checkout, the SPA opens directly and the source audit runs offline
using `data/audit-inputs.json.gz` (selected raw Jira records, immutable commit
responses and base-file contents). Restore the context corpus before engine builds:

```bash
gzip -dk apps/eval/benchmarks/kafka-1000-v1/data/context-sources.jsonl.gz
.venv/bin/python -m harnext_eval.e2.benchmark_audit validate
```

The restored corpus must match `context_sources_sha256` in the manifest. Full
candidate resampling still needs the original Kafka raw corpus; the selected
audit archive is sufficient to verify the frozen 1,000 tasks, not to reproduce
the full acquisition pool. The archive contains public Apache source material
and is evaluator-only because it includes reference changes.

Browser checks require Playwright and Chromium. The script resolves the installed
`playwright` package by default; `HARNEXT_PLAYWRIGHT_MODULE` and
`HARNEXT_CHROMIUM_PATH` optionally select an existing local installation.
