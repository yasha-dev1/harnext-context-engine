# E2 v2 — kafka-500-v2: design contract

Decided with the author on 2026-09-20. This file is the interface contract between
modules; change it before changing code that crosses a module boundary.

## Fixed decisions

- 150 tasks: 50 coding (real merged Kafka PRs, graded by the PR's own tests against
  the golden patch) + 100 historical (the 8 v1 field-replay families at 4 each + the 11
  families of `v2/families.py` at 6 each, `issue_changed_files` and `pr_approvers` at 7; counts
  are the `COUNTS` dict at the top of `v2/tasks.py`). Cut from 250 + 250, then 100 + 250, on
  2026-09-20 to fit the run budget: a coding trial costs ~12 min (agent + Gradle grading).
  With ~5 tasks per historical family, results are reported for the historical part as a
  whole and for four family groups (state lookups, change histories, cross-source / multi-hop,
  aggregation), never per single family.
- Tasks are drawn from 2023-01-01 → 2026-06-30. The event stream is COMPLETE: every
  event in the corpus from its very first one (2011-07) up to the checkpoint is
  delivered to every store, in order. No sampling, no relevance scoping, no noise
  filter: managing the full volume is the system under test. Only the smoke scope
  (tests) uses a small slice. Monthly store checkpoints (see "Checkpoints").
- Every task runs under every arm (12: `N`, `S-tools`, `R-hyb`, and the nine file arms); 1,800 trials per agent (150 × 12). The graph arm `G-graphiti` was dropped on 2026-09-20: the official graphiti-core pipeline needs ~3 LLM calls per event, i.e. 3–6 months on the local GPU (or ~$9k by API) over the complete 629,458-event stream, measured at 12.6 s/event and rising. The adapter and its fidelity note stay in the repository as the record of that measurement. One trial = (task, arm, agent).
- Builder model for every LLM-built store: Claude Haiku 4.5. Stores are built once,
  frozen, hashed; all coding agents read identical stores.
- Every store arm is the Harnext context engine started with a different storage
  configuration, and the agent reaches every one of them over the read-only MCP server
  (`v2/mcp_server.py`). **Every file arm (F0, F1, F2, F3, F4-fact, F4-entity, F4-topic, F5,
  F6) is served through Harnext's real read interface**: MCP kind `research`, whose single
  tool is the production `context_research(question) -> {answer, files}` — the product's
  research agent (`apps/mcp/src/harnext_mcp/research.py`, shared function `research_dir`)
  runs read-only over the arm's checkpoint inside the trial container and returns a cited
  answer. F5 read this way IS the Harnext system arm. There are no plain file tools
  (list/read/grep) for any arm; R-hyb keeps kind `rag`, G-graphiti kind `graphiti`. Reader:
  Claude Code harness, model `claude-haiku-4-5-20251001`, 40 turns / 180 s per question,
  at most 25 questions per trial (server budget). Recorded deviation from the product: the
  production system prompt says *"Start from INDEX.md to navigate, then read the relevant
  entity files"* — an F5-layout hint that would bias the layout ablation — the eval uses
  *"Start from INDEX.md if present, otherwise from the top-level listing, then read the
  relevant files"* (`mcp_server.layout_neutral`; hashed into the receipts' `store.system_prompt`).
  (Decided 2026-09-20; the earlier native `/context` mount, the `files` MCP kind and the
  `F5-mcp` control are dropped.)
- **Gather context first (decided 2026-09-21).** Every arm that has a context source gets a
  standing instruction file, written to the agent's user-level instruction file in the trial
  home (`~/.claude/CLAUDE.md` for Claude Code, `~/.codex/AGENTS.md` for Codex; never in the
  workspace): *"DO NOT START UNTIL YOU GATHER CONTEXT FROM context_research mcp tool"* for the
  file arms, the same rule naming `search` for R-hyb, and naming `gh`/`jira`/`mail` for the
  S-tools baseline (so the instruction itself is not a confound). `N` has nothing to consult and
  gets no file. Texts: `GATHER_FIRST` in `v2/runner.py`, hashed into the run manifest and into
  every `result.json`. Reason: with consulting optional, 9 of 14 F-arm coding trials never
  called `context_research`, so the arms could not differ. Coding trials also run with a
  10-minute agent shell timeout (Claude Code backgrounds commands after 2 minutes; a Kafka test
  run takes 3–4).
- Trial results are saved atomically per trial. Quota/credit/auth/rate-limit/infra
  errors are never recorded as results: the run pauses and the trial is retried.

## Arms

| id | family | built by | source of definition |
|---|---|---|---|
| `N` | none | — | closed-book floor (Zhou et al. 2607.26637 §3), diagnostic |
| `S-tools` | sources | code | time-frozen `gh` / `jira` / `mail` CLIs over the raw systems of record; the genuine baseline |
| `R-hyb` | RAG | code | BM25 + dense, RRF k=60, cross-encoder rerank, token chunks (blueprint D08–D12); one shared append-only index, each checkpoint = its first rows (monthly shards), df/idf/avg-length as of the checkpoint |
| `F0` | file | code | verbatim dump, one file per event, `events/YYYY/MM/DD/<id>.md` + frontmatter |
| `F1` | file | code | templated per entity: `entities/<key>/{OVERVIEW,timeline}.md` |
| `F2` | file | LLM, move-only | foldered: LLM designs taxonomy, F0 files moved intact, zero byte edits |
| `F3` | file | LLM rewrite | reorganized (preserve): restructure with the keep-every-fact rule |
| `F4-fact` | file | LLM agent | curated, one file per fact/event digest |
| `F4-entity` | file | LLM agent | curated, one file per entity with headed sections |
| `F4-topic` | file | LLM agent | curated, one large file per topic with nested headings |
| `F5` | file | LLM agent (product builder) | **Harnext as shipped**: the product's batch lane — its seed (`CLAUDE.md`, `INDEX.md`, `_meta/schema.md`, `_meta/superseded.md`, `entities/<type>/<slug>/{OVERVIEW,facts,timeline}.md`, `topics/`), its `SYSTEM_PROMPT`, its per-entity batch instruction, its tool policy (`stores/product.py`) |
| `F6` | file | LLM agent | agent-curated from empty, free structure, taxonomy contract only |

Read access per arm (`arms.yaml`): `N` none; `S-tools` the offline `gh`/`jira`/`mail` CLIs;
`R-hyb` MCP `rag` (`search`, `read_event`);
every `F*` arm MCP `research` (`context_research` only — the Harnext product interface).

### Write side of F5 (decided 2026-09-20)

F5 is Harnext on the write side too, not the eval's Zhou-style builder. `stores/product.py`
imports (never paraphrases) the product: seed = `harnext_builder.agentfs.seed.SEED_FILES`,
system prompt = `harnext_builder.prompts.SYSTEM_PROMPT`, instruction = `_BATCH_TMPL` rendered
once per entity (events grouped by CloudEvent `subject`, e.g. `issue:KAFKA-14551`,
`pr:apache/kafka/13039`, `thread:<root message id>`), events as `CloudEvent` JSON, tools =
the builder's `ALLOWED_TOOLS`/`DENIED_TOOLS`. The store mechanics are those of every LLM arm
(git tree, one commit + cursor per window, checkpoint tags, rollback/retry, `_unprocessed/`,
PauseRun, transcripts). The one permitted deviation is cadence: production closes a window
per entity after 30 s of silence / 20 events / 120 s, i.e. ~629,000 builder runs over this
stream; the eval runs the builder once per slice of 250 events and hands it all per-entity
batches of the slice in one instruction. All deviations: `benchmarks/kafka-500-v2/fidelity/F5.md`.

## Genuine baseline (decided 2026-09-20)

An arm is only a fair comparator if its agent has a legitimate way to answer. So:

- **Every coding workspace is a real git repository** holding exactly the ancestors of
  the task's base commit (built per task from the bare clone by fetching `base_sha`
  only; verified with `git cat-file --batch-all-objects` that no object is newer than
  the base commit). `git log/blame/show/grep` work offline for ALL arms. No remote.
- **Arm `S-tools` (the normal agent)**: no store and no MCP, but time-frozen replicas of
  the systems of record, reachable with the usual CLIs inside the container: `gh`
  (pr/issue list, view, diff, search; `gh api` for the covered endpoints), `jira`
  (issue view with comments and change history, search), `mail` (search/read of the
  dev lists), all served offline from the stream as of the task's checkpoint. This is
  how an agent without a context engine would work in a company: query the sources
  at question time. The real github.com / issues.apache.org stay blocked for every
  arm because they hold the future (the golden PR, the final issue state).
- **Arm `N`** stays as the closed-book floor (contamination and guessing check); it is a
  diagnostic, not the baseline the stores are compared against. Primary contrasts are
  store arm − `S-tools`.
- Store arms get the store (+ git in coding tasks) and NOT the source CLIs, so a store
  cannot win by falling back to the sources (the v1 `fetch_source` escape).
- **Answerability audit** (must pass before any run): for every task, a scripted oracle
  using only the `S-tools` CLIs as of the checkpoint reproduces the gold answer
  (historical) or reaches the issue, its discussion and the touched files' history
  (coding); and no CLI output contains anything later than the checkpoint boundary.

Builder tools for LLM arms (Zhou §3): view, create, string-replace edit, insert,
delete, rename, regex search. F2 gets rename/move only plus view/search.

## Paths (all under `apps/eval/`)

- `benchmarks/kafka-500-v2/` — versioned: `coding-candidates.jsonl`,
  `coding-selection.json`, `coding-admission.json`, `tasks.public.jsonl`,
  `tasks.private.jsonl`, `manifest.json`, `arms.yaml`.
- `out/kafka-500-v2/stream/events.jsonl` — the context event stream (git-ignored).
- `out/kafka-500-v2/stores/<arm>/` — store working tree (a git repo; one commit per
  build window; tag `ckpt-YYYY-MM` per checkpoint). `build-state.json` = resume cursor.
- `out/kafka-500-v2/stores/<arm>/export/<YYYY-MM>/` — materialized read-only checkpoint
  for the non-git stores (R-hyb: `meta.json` + hard links to the monthly shards ≤ the
  checkpoint of the ONE cumulative index `stores/R-hyb/index/` — no copy, nothing from after the
  checkpoint in the mounted directory, BM25 statistics computed as of the checkpoint;
  Graphiti: db snapshot; S-tools: sqlite), plus
  `<YYYY-MM>.manifest.json` with its content hash. File arms (git trees) are NOT exported:
  a checkpoint is far too large × 42 checkpoints × 9 arms; trials extract on demand into
- `out/kafka-500-v2/checkouts/<arm>/<YYYY-MM>/` — read-only `git archive` of `ckpt-YYYY-MM`,
  leased by running trials, LRU-evicted beyond `keep_per_arm` (`v2/checkouts.py`).
- `.harnext/evaluations/kafka-500-v2/<agent-run-id>/` — versioned trial archives:
  `trials/<task_id>/<arm>/{result.json,…}`, `summary.json`, `manifest.json`.
- Source: `src/harnext_eval/e2/v2/` (`stream.py`, `tasks.py`, `stores/…`, `agents/…`,
  `runner.py`, `grade.py`, `report.py`); coding selection/admission stays in
  `src/harnext_eval/e2/kafka_coding.py`.
- The trial MCP server is `src/harnext_eval/e2/v2/mcp_server.py` (`--kind rag|graphiti|research`,
  receipts at `<trial>/mcp/receipts.jsonl`, archived as `mcp-receipts.jsonl`). Kind `research`
  imports the product's `harnext_mcp.research.research_dir` (the same code path
  `context_research` runs in production); the store checkout directory is the snapshot.

## Stream event (one JSON per line, sorted by `time`, then `id`)

```json
{"id": "...", "time": "2024-05-01T10:00:00Z", "type": "jira.comment|jira.created|jira.change|gh.pr_opened|gh.pr_merged|gh.pr_closed|gh.review|gh.review_comment|gh.issue_comment|gh.push|mail.message",
 "entities": ["issue:KAFKA-1", "pr:123"], "actor": "display name or null",
 "title": "short", "text": "markdown body shown to builders and readers", "meta": {}}
```

`text` must be self-contained and must not contain anything later than `time`.

## Checkpoints

`ckpt-YYYY-MM` contains exactly the events with `time < YYYY-(MM+1)-01T00:00:00Z`.
A task's `checkpoint` is the latest checkpoint whose boundary is ≤ the task's
context cutoff (coding: PR `createdAt` − 1 s; historical: chosen on a boundary).
The first checkpoint is `2023-01` (it already holds 2011-07 → 2023-01, eleven years of
history); the last is `2026-06`.

## Task records

Public (seen by agent code paths): `id, kind, family, checkpoint, prompt,
answer_schema` and for coding `base_sha, tests`. Private: `expected` (historical) or
`fail_to_pass, pass_to_pass, test_patch, reference_patch` (coding), `evidence_ids`.

## Store builder interface (`stores/base.py`)

```python
class StoreBuilder(Protocol):
    arm: str
    def build(self, events: Iterable[dict], out: Path, *, until: str | None) -> None
        """Resumable. Consumes events in order, commits per window, tags checkpoints."""
    def export(self, out: Path, checkpoint: str) -> Path
        """Materialize the read-only checkpoint directory (non-git stores; smoke/small only
        for git trees — the runner extracts those on demand, see Runner)."""
```

LLM builders raise `PauseRun` (from `v2/errors.py`) on quota/auth/rate-limit
errors after saving their cursor; a failed window is rolled back (`git reset --hard`).

## Agent adapter interface (`agents/base.py`)

```python
class AgentAdapter(Protocol):
    name: str            # "codex", "claude-code", …
    def run(self, trial: TrialSpec) -> AgentResult
```

`TrialSpec`: prompt, answer schema, workspace dir (coding) or empty dir, context dir
or None, MCP server spec or None, model, effort, timeout. `AgentResult`: `status`
in {`completed`, `model_failure`, `timeout`, `infra_error`}, answer, usage, trace
path, elapsed. `infra_error` (quota, credits, auth, 429/5xx, CLI crash, docker
failure) ⇒ runner raises `PauseRun`, nothing is recorded.

Agents run inside a Docker container (image `harnext-e2-agent`) that mounts only
the workspace (rw), `/store` (ro), the receipts directory and the agent's auth file — plus,
for `research` arms under EVERY agent (Codex too), the Claude Code login read-only, since
the research agent is the Claude Code harness running inside the same container; egress is
limited to the model APIs (api.openai.com, api.anthropic.com) through an allowlisting proxy. Gradle runs offline from a
preloaded cache. The container, not the agent's own sandbox, is the boundary.

## Runner

Frozen task order interleaves coding and historical and rotates families/years so
any prefix is balanced. Per task, arms run in an order rotated by task index.
After each trial: grade → `result.json` (atomic) → `summary.json` → optional git
checkpoint commit every N trials. After the agent finishes an MCP trial the runner reads the
receipts: a row with `"infra": true` (the research reader's provider refused — usage limit,
429, auth; `mcp_server.Infra`) raises `PauseRun`, nothing is recorded. A model failure of
the reader (max turns, timeout, garbage) is returned to the agent as an answer string and is
an ordinary trial. Config `reader: claude|fake` (default `claude`) selects the reader; `fake`
answers deterministically without a model so the offline smoke configs complete. `--check-resume` validates hashes and prints
remaining work without model calls. Trials with `result.json` are never rerun.

Store identity and checkouts. The run manifest pins, per (store, checkpoint) the run needs,
the tag's tree hash for git stores (`git rev-parse ckpt-X^{tree}`: exact content, no rehash of
600k files) and the export manifest's sha256 for the others (`store_hash: tree` hashes the
export's files instead — small stores and smoke only). Preflight and `--check-resume` fail
fast, listing every missing tag/export; a pinned hash that no longer matches refuses the
resume. At trial staging the runner leases the checkout for (arm, checkpoint) from
`checkouts:` (`Checkouts.acquire`): the first trial extracts `git archive ckpt-X` into a
staging directory, makes it read-only and renames it into place; concurrent trials (threads
or processes; fcntl locks) wait for it and share it; the lease is released after the archive.
Unleased checkouts beyond `keep_per_arm` (default 3) are evicted least-recently-acquired
first. Trade-off: the frozen `run_order` interleaves years for balanced prefixes, so
consecutive tasks want different checkpoints and an LRU of 3 thrashes (F0 ≈ 1 GB / 630k
files per extract); raising `keep_per_arm` trades disk (≈ 1 GB × arms × keep) for fewer
re-extractions, lowering it fits a small disk. Extraction cost is not measured here.

## Two-tool file arms (decided 2026-09-27)

Every file arm gets a variant `<arm>-search` over the same store, served by MCP kind
`research_search`: the production `context_research` (research agent, full cited reports, at most
25 calls per trial) plus the product's new `search_context(query, k)` (`harnext_mcp.search`:
BM25 over heading-sized chunks of the snapshot's markdown, no LLM, returns path + line range +
snippet; index built once per checkout in the background at server start). Standing
instructions (runner `GATHER_FIRST_EXTRA`): "Use context_research for full reports about the
codebase and use search_context for lightweight context retrievals." Purpose: cut the read
agent's cost; the research-only arms stay unchanged and every recorded trial is kept. The new
kind's texts live outside the run's recorded prompt-template hash and are pinned per arm
(`arm_pin`), so runs that began earlier resume unchanged.

**Merged 2026-09-27 (Yasha):** Harnext's read interface is `context_research` + `search_context`,
in production (`apps/mcp/src/harnext_mcp/main.py`, tested in `apps/mcp/tests/test_mcp.py`) and
in E2. The active arms are N, S-tools, R-hyb and the nine `-search` file arms (`F5-search` =
Harnext); the research-only F0-F6 arms are retired, and their trials so far are reported as an
ablation ("research agent only"), never mixed into the active arms.
