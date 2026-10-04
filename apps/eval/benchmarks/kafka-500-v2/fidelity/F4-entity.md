# Arm `F4-entity` — curated, one file per entity with fixed sections

Source: Zhou et al. App. A.1 prompt (see [F4-fact.md](F4-fact.md) "A.1 prompt fidelity" — shared,
not repeated) with a prescribed granularity; plan row F4 *"Curated flat — LLM, per fold —
entities/<subject>/{OVERVIEW,facts,timeline,superseded}.md; no INDEX, no topics, no cross-links;
curation without global structure."* Implementation: `stores/llm_files.py:363-364`; prompt
`prompts.py:110-111`. Shared decisions: [README.md](README.md).

| plan / paper says | what the code does | deviation (and why) |
|---|---|---|
| Plan: four files per entity `{OVERVIEW,facts,timeline,superseded}.md`. | **One file per entity** with five `##` sections in fixed order: `Overview`, `Current state`, `Facts`, `Timeline`, `Superseded` (`prompts.py:111`; the same names in `fake.py:18 SECTIONS`), grouped `entities/<kind>/…` by instruction. | Sections instead of files: one `cat` gives the whole entity, matching the paper's "a file remains the unit a reader may take in whole" (SHAPE, `prompts.py:51`); F5 uses the identical section list so F5 − F4-entity isolates INDEX/topics/cross-links/line cap. |
| "No INDEX, no topics, no cross-links." | No seed files, no INDEX instruction. But the shared PRINCIPLES block still tells the builder to cross-reference by full path (`prompts.py:84`) and STRATEGY 3 to leave *"a brief self-contained note that also cross-references the home file's path"* (`64`). | **Cross-links are not forbidden**, only unrequested: A.1 is kept intact for comparability with the paper, so F4-entity may grow cross-references and even ad-hoc index files. The clean F5 − F4 contrast is therefore "seeded global structure + cap" vs "none prescribed", not "links vs no links". |
| Paper's agent-curated store: content and organization both free. | Organization fixed to entity files by kind; content free. | Intentional: D02 layout factor at entity granularity. |
| "Current state — rewrite to keep current"; "Superseded — facts that stopped being true, with when and what replaced them." | Instruction only (`111`). | Supersession is a prompt rule (A.1 temporal-awareness bullet), not a format rule as in plan row F7 (`facts.yaml` ledger, not run). |
| Reader: paper's read-only tools (view, grep, toc, section read), one prompt family. | Harnext's production research agent (`harnext_mcp/research.py`, Claude Code harness, Haiku 4.5, tools Read/Glob/Grep/LS, fixed system prompt) behind the one MCP tool `context_research(question)`; the coding agent only asks questions. Paragraph identical for every file arm. | **Reader is the product's dedicated search agent with a fixed prompt** — closer to the paper than a general coding agent; Read/Glob/Grep cover view/grep/section-read, there is no toc tool. One recorded prompt deviation (README "Read access"). Same for every file arm, so it does not bias between them. |
| One line per event in Timeline. | Instruction. Nothing verifies it. | With 629k events, entity timelines for hot entities (e.g. `component/core`, release KIPs) will be long; F4-entity has **no line cap** (F5 does) — this is the designed difference. |
| Tools: seven-tool write set (as F4-fact). | `fs_tools.ALLOWED_TOOLS` = `Read(./**)` (Read, Glob, Grep), `Edit(./**)` (Write, Edit) — both confined to the store by permission rule — plus `delete` and `rename`, real tools served in-process (`stores/fs_tools.py`: relative normalized paths inside the tree, never `.git/`/`_event/`/`_unprocessed/`, no overwrite). No shell: Bash, Skill, web, sub-agents denied. | The harness's sandboxed Bash cannot start on the build host (`bwrap: loopback: Failed RTM_NEWADDR`; every `rm`/`mv` of the first build failed), so delete/rename became dedicated tools — closer to the paper's seven-tool set than Bash was. Checked live: deletes/renames execute; `../x`, `/tmp/x`, writes and reads outside the tree are refused. |

## Read access

Read through Harnext's production interface: the MCP tool `context_research(question)` (kind
`research`), i.e. the product's research agent over this store's checkpoint — see README
"Read access" for the recorded INDEX.md-prompt deviation shared by every F arm.

## What an examiner should know

- F4-entity is the **direct ancestor of F5**: same sections, same prompt except the F5 layout block
  and seed files. Any F5 lift is attributable to `INDEX.md`, `topics/`, `_meta/schema.md`, the
  cross-link requirement and the 200-line cap — with the caveat above that F4-entity may grow some
  of these on its own.
- Compare also with F1 (code-templated entity files): F4-entity − F1 is the "does the LLM digest
  add anything at entity granularity" question of plan row F1.
- Verified per window: only the shared checks (`llm_files.py:208-234`); section order is not checked
  (the `fake` builder produces it, `fake.py`, for tests only).
- Failed windows → `_unprocessed/` (README §5).
