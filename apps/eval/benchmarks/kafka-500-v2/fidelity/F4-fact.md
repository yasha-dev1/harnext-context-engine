# Arm `F4-fact` — curated, one small file per fact / event digest

Source: Zhou et al. App. A.1 management-agent prompt (the "Agent-curated store" builder) with a
**prescribed granularity** added by this benchmark; plan row F4 *"Curated flat — entities/<subject>/
{OVERVIEW,facts,timeline,superseded}.md; no INDEX, no topics, no cross-links"*, split here into
three granularities (fact / entity / topic) to test D02 "file layout" without the F5 index.
Implementation: `stores/llm_files.py:355-360`; prompt `prompts.py:107-109`, `127-141`.
Shared decisions: [README.md](README.md). The A.1-fidelity table below is shared by F4-*/F5/F6.

## A.1 prompt fidelity (all curated arms)

| paper says (A.1) | what the code does | deviation (and why) |
|---|---|---|
| Role: *"Memory Management Agent … All memory files are markdown (.md) and live under /memories."* | ROLE (`prompts.py:95-97`), root = working directory; context described as *"one window of consecutive events of the Apache Kafka project"*. | Root path and the nature of the chunk. |
| Filesystem structure: five taxonomy properties; "A workable shape" example; naming guidance. | TAXONOMY (`22-28`), SHAPE (`30-51`), NAMING (`53-57`) — verbatim but `/memories/` → `./` in the tree drawing. | none |
| Strategy 1–4 (survey, capture faithfully, file where it belongs, maintain). | STRATEGY (`60-65`) verbatim but *"View /memories"* → *"View the store root"*. | none |
| Principles: one topic per file; frontmatter `name`/`description`; cross-references by full path; temporal awareness; resolve references; source attribution in benchmark-specific brackets. | PRINCIPLES (`77-87`): same six bullets. Cross-reference example uses store-relative paths; temporal bullet says *"Every event carries a timestamp"*; resolve-references example uses "this patch" → the PR; source attribution = `[evt:<id>]` with the paper's repeated/append rules folded in (`87`). Frontmatter bullet drops the parenthetical *"(via str_replace on the description: line)"* (the tool is called Edit here). | The paper appends a per-benchmark "Source Attribution" extension; here it is merged into the bullet with the `[evt:…]` schema — same content, one block. |
| Tools: seven-tool write set (view, create, str_replace, insert, delete, rename, grep) via function calling. | `fs_tools.ALLOWED_TOOLS` = `Read(./**)` (Read, Glob, Grep), `Edit(./**)` (Write, Edit) — both confined to the store by permission rule — plus `delete` and `rename`, real tools served in-process (`stores/fs_tools.py`: relative normalized paths inside the tree, never `.git/`/`_event/`/`_unprocessed/`, no overwrite). No shell: Bash, Skill, web, sub-agents denied. TOOLS paragraph of the prompt gives the mapping and boundaries. | **Different harness** (Claude Code agent loop, Haiku 4.5) than the paper's function-calling loop (gpt-5.4-mini). The harness's sandboxed Bash cannot start on the build host (`bwrap: loopback: Failed RTM_NEWADDR`; every `rm`/`mv` of the first build failed), so delete/rename became dedicated tools — closer to the paper's seven-tool set than Bash was. Checked live: deletes/renames execute; `../x`, `/tmp/x`, writes and reads outside the tree are refused. `max_turns=200`, `timeout 1800 s` per window. |
| Chunk = ≤ 8 turns / 3,000 chars (C.1). | Window = ≤ 25 events / 60 kB inlined into the instruction (`160-168`, `INLINE_LIMIT` 120k chars) or, if larger, provided as files `_event/0001-<id>.md` + `MANIFEST.md` (`58-68`). Both always exist (`200`); `_event/` is removed after the run (`205`). | Much larger chunks (README §3). |
| Not present in the paper. | READER block (`18-19`): who reads the store, with what tools, and *"keep identifiers literal"*. | **Addition.** Reason: the paper's reader used the same file tools on chat data; here the reader greps for `KAFKA-12345`; without this the builder paraphrases ids. It does not mention any question. |
| Output: brief summary. | OUTPUT verbatim (`92-93`). | none |

## F4-fact granularity (`prompts.py:108-109`)

| this benchmark says | what the code does | note |
|---|---|---|
| *"Each file holds exactly one fact or one event digest … Group the files in one folder per entity … folders grouped by kind. Never merge several facts into one file; when a fact is superseded, write the new fact as a new file and add one line to the old file pointing at it. This granularity overrides 'one topic per file'."* | Instruction only; no code check on file size or count. `F4Fact` adds nothing to `Curated` (`llm_files.py:359-360`). | Not a Zhou variant: an extreme of D02 (finest granularity) to measure navigation cost vs. locality. Expect the largest file count of the curated arms. |

## Read access

Read through Harnext's production interface: the MCP tool `context_research(question)` (kind
`research`), i.e. the product's research agent over this store's checkpoint — see README
"Read access" for the recorded INDEX.md-prompt deviation shared by every F arm.

## What an examiner should know

- Same system prompt as F4-entity/F4-topic/F5/F6 except the GRANULARITY block; the digest of all
  prompts is in the store manifest, so the arms differ by exactly that block (plus F5's seed files).
- No invariant is verified for F4-fact beyond the shared checks; "one fact per file" is a prompt rule.
- Failed windows → `_unprocessed/` (README §5).
