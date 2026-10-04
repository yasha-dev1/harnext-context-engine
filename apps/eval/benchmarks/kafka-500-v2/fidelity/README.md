# Fidelity notes — kafka-500-v2 arms

One note per arm: what the source paper / thesis blueprint prescribes, what the code
actually does (with `file:line` references into `apps/eval/src/harnext_eval/e2/v2/`), and
where the two differ. Written 2026-09-20 against branch `eval/e2-v2-kafka-500`; line numbers
are those of that day's tree. Sources of truth: the implementation; Zhou et al., "Filesystem-Based
Memory for LLM Agents" (arXiv 2607.26637v1, §2–3 and Appendix A/C); Rasmussen et al., "Zep: A
Temporal Knowledge Graph Architecture for Agent Memory" (arXiv 2501.13956v1) and the installed
`graphiti-core==0.30.2`; the thesis blueprint `masters-streaming-ai-agent-architecture/eval/artifacts/`
`05-e3-store-design-plan.html` (rows F0–F7, R-*) and `06-e3-experiment-blueprint.html` (D08–D12, §07).

| arm | note | built by | paper / blueprint anchor |
|---|---|---|---|
| `N` | [N.md](N.md) | — | Zhou §3 "Closed-book" |
| `S-tools` | [S-tools.md](S-tools.md) | code (SQLite replicas + CLIs) | E2-V2-DESIGN "Genuine baseline" (no paper) |
| `R-hyb` | [R-hyb.md](R-hyb.md) | code | blueprint D08–D12, §07 retrieval regime |
| `G-graphiti` | [G-graphiti.md](G-graphiti.md) | `graphiti-core` 0.30.2 | Zep §2–3 |
| `F0` | [F0.md](F0.md) | code | Zhou "Verbatim dump"; plan row F0 |
| `F1` | [F1.md](F1.md) | code | plan row F1 (not in Zhou) |
| `F2` | [F2.md](F2.md) | LLM, move-only | Zhou "Foldered sessions"; plan row F2 |
| `F3` | [F3.md](F3.md) | LLM rewrite | Zhou "Reorg. (preserve)", App. A.2; plan row F3 |
| `F4-fact` | [F4-fact.md](F4-fact.md) | LLM agent | Zhou "Agent-curated" A.1 + prescribed granularity |
| `F4-entity` | [F4-entity.md](F4-entity.md) | LLM agent | idem; plan row F4 |
| `F4-topic` | [F4-topic.md](F4-topic.md) | LLM agent | idem |
| `F5` | [F5.md](F5.md) | LLM agent (product builder) | Harnext as shipped: the product's batch lane (seed, prompt, per-entity batches, tools) |
| `F6` | [F6.md](F6.md) | LLM agent | Zhou "Agent-curated store"; plan row F6 |

Read access (decided 2026-09-20, supersedes any `/context` or `F5-mcp` wording in the notes'
cited line numbers): every `F*` arm is read through Harnext's production interface — the MCP
tool `context_research(question)` (`mcp_server.py` kind `research`), implemented by the product's
research agent (`apps/mcp/src/harnext_mcp/research.py:research_dir`, Claude Code harness on
`claude-haiku-4-5-20251001`, 40 turns / 180 s per question, 25 questions per trial). F5 read this
way is the Harnext system arm. **Recorded deviation**: the production research prompt says
*"Start from INDEX.md to navigate, then read the relevant entity files"*; the eval says *"Start
from INDEX.md if present, otherwise from the top-level listing, then read the relevant files"*
(`mcp_server.layout_neutral`), because the INDEX.md hint is F5's own layout and would bias the
layout ablation. This one sentence is the only difference from the product's read path.

## Decisions that apply to every arm

These are restated once here; each note only adds what is arm-specific.

1. **Complete stream, no filtering.** Every store ingests every event of the corpus from its
   first one up to the checkpoint. `stream.py:25` (`SCOPES["full"]`, window start `None` = first
   event); `out/kafka-500-v2/stream/stream-manifest.json` records `complete: true, events: 629458,
   filters: []`, first month `2011-07`. `base.py:328-359` (`Store.audit`) raises if the ids the
   build log accounts for are not exactly the stream's ids up to the cursor. Note: E2-V2-DESIGN.md
   says "2019-01" for the corpus start; the built stream begins 2011-07 (KAFKA-1). The doc is stale,
   the code is right.
2. **Monthly checkpoints.** `ckpt-YYYY-MM` = all events with `time < first instant of next month`
   (`base.py:108-126` windows never cross a month; tag at `last_of_month`, `base.py:312-314`;
   months without events still get a tag, `base.py:379-386`). Tasks use checkpoints 2023-01 …
   2026-06 (`stream.py:70`); a task's cutoff is boundary − 1 s (`tasks.py:211`). File-arm trees
   already carry 2011-07 … 2022-12 tags but no task reads them.
3. **Window size = batch-lane cadence and the cost lever.** Code-built stores commit every 1000
   events (`base.py:217`); LLM stores hand the builder 25 events or 60 kB of event text, whichever
   first (`llm_files.py:137-138`, `build.py:30-31`; overridable with `--window-events/--window-bytes`).
   Measured by the author with Haiku 4.5 on F5: $0.30 / 130 s per 25-event window and
   $0.58 / 251 s per 250-event window, i.e. a 10× larger window costs ~2× — the per-window fixed
   cost (survey the store, re-read INDEX) dominates. The trade-off: larger windows are cheaper
   per event but the builder sees a coarser stream and reconciles more at once; the paper's chunk
   is "at most eight turns, closed at 3,000 characters" (Zhou App. C.1), far smaller than either.
   The window size used for the full build is recorded in `build-state.json:config` and refused on
   resume if changed (`base.py:243-250`).
4. **Builder models.** Every LLM file arm: Claude Haiku 4.5 (`prompts.py:16`, hashed into the
   store manifest via `prompts.digest()`, `prompts.py:204-217`), through the Claude Code harness
   (`llm_files.py:51-54`; `apps/builder/.../harness/claude_code.py:81-96`, `permission_mode=dontAsk`,
   no shell: delete/rename are in-process tools, `stores/fs_tools.py`; F2 a move plan). Graphiti: local Qwen3-30B-A3B via Ollama (`graphiti.ollama.yaml`;
   `graphiti.py:223-231`). This is a **confound**: file arms and the graph arm are built by
   different models. Reason: the measured Graphiti cost with a paid model (6.3 LLM calls, ~11 s per
   event in the 21-episode smoke build: `out/.../G-graphiti-smoke/build-state.json`) projects to
   thousands of dollars over 629k events. Planned mitigation: a **second F5 build with the same
   Qwen model** so at least one file arm is model-matched to Graphiti. Until it exists, a
   G-graphiti − F* contrast mixes representation with builder strength.
5. **Failed windows lose nothing.** An LLM window gets two attempts (`llm_files.py:248`); after
   both fail the tree is rolled back and the window's events are written verbatim under
   `_unprocessed/YYYY/MM/DD/<id>.md` (`llm_files.py:173-176`, `base.py:79-81`), the builder is
   told never to touch that folder (`prompts.py:90`), and `audit` verifies they are there
   (`llm_files.py:178-181`, `base.py:342-353`). Three consecutive failures pause the build
   (`llm_files.py:260-265`). F2 and F3 have their own fallbacks (see their notes). Readers can grep
   `_unprocessed/`, so a failed window degrades the store to F0 quality for those events rather
   than deleting them; the count is reported in `build-state.json:audit.unprocessed`.
6. **Verification after every window** (`llm_files.py:208-234`): git HEAD untouched, no tool call
   addressed a path outside the tree, the store changed, `_unprocessed/` unchanged, plus per-arm
   invariants (F2 byte-identity, F5 seed files present). Every builder transcript is saved
   (`llm_files.py:236-243`).
7. **Store identity.** File arms: git tree hash of `ckpt-X` pinned in the run manifest; trials
   extract `git archive` on demand (E2-V2-DESIGN "Store identity and checkouts"; `base.py:401-429`).
   Non-git arms: sha256 of the export.
8. **Reading.** File arms: `/store` read-only + `mcp_server.py --kind research`, whose only tool
   is the production `context_research(question)` (`mcp_server.py` `ResearchBackend`, calling
   `harnext_mcp.research.research_dir`); the reader's answer and the files it read are in the
   trial's receipts (`mcp-receipts.jsonl`, `research` field). Paragraph `runner.py` `ACCESS["mcp:research"]`.
   R-hyb / G-graphiti: `/store` read-only + the tools in `KINDS` (`mcp_server.py`), paragraph
   `ACCESS["mcp"]`. S-tools: `/sources` read-only + `HARNEXT_SOURCES` (`container.py`), paragraph
   `ACCESS["sources"]`. Store arms never get the source CLIs (arms.yaml has no arm with both).
   The Claude Code login is mounted read-only in EVERY research trial's container (Codex too)
   because the reader runs inside it (`container.py:reader_auth`). Egress only to the model APIs
   through an allowlisting proxy; github.com / issues.apache.org blocked (`container.py`).
   A reader provider refusal is flagged `"infra": true` in the receipts and pauses the run
   (`runner.py:reader_infra`); a reader model failure is an answer string, not a pause.
9. **Coding tasks.** Workspace = a real git repository holding exactly the ancestors of the base
   commit (`workspace.py:1-5`, `142-158`; verified by `git cat-file --batch-all-objects`,
   `workspace.py:79-130`: no object newer than base, no remote, no tags, reflog empty). Every arm
   gets the same wording (`runner.py:67-70`). Grading applies the **hidden test patch** to a fresh
   base tree plus the agent's non-test changes, then runs `fail_to_pass`/`pass_to_pass` offline
   (`grade.py:195-208`; sections of the agent's patch touching hidden test files are dropped,
   `grade.py:143-146`). Agents are told "hidden tests decide" (`runner.py:74-75`) but never see them.
10. **Prompts never mention a benchmark question** (`prompts.py:8`); the reader's grep habits are
    described to the builder (`prompts.py:18-19`), which the paper does not do (see F4/F5/F6).
