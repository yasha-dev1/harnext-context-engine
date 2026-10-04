# Arm `F4-topic` — curated, one large file per topic with nested headings

Source: Zhou et al. App. A.1 prompt (see [F4-fact.md](F4-fact.md) "A.1 prompt fidelity" — shared)
with a prescribed granularity; plan factor D02 (layout) at its coarsest setting. Not a Zhou variant:
the paper's SHAPE guidance says to *"promote sections into files or folders rather than deepening
the headings"* — F4-topic deliberately overrides that to measure the opposite end.
Implementation: `stores/llm_files.py:367-368`; prompt `prompts.py:112-113`.
Shared decisions: [README.md](README.md).

| prompt says (this benchmark) | what the code does | deviation from A.1 (and why) |
|---|---|---|
| *"Organize by topic (a subsystem, feature area, release, process or recurring theme), not by entity: few, large files at the store root or one level down, one per topic."* | Instruction only; `F4Topic` adds nothing to `Curated` (`llm_files.py:367-368`). | Overrides A.1 "one topic per file" only in the sense of *topic size*: a topic file is still one subject. |
| *"Inside a topic file use nested headings; entities … are `###` sub-sections under the `##` theme they belong to, each holding that entity's facts and dated history."* | Instruction only. | Consistent with A.1 SHAPE ("headings continue the taxonomy", `prompts.py:51`) — the taxonomy lives inside files rather than in folders. |
| *"An entity lives in exactly one topic file; cross-reference from other topics."* | Instruction only. | A.1 cross-reference syntax `see path > ## Section` (`84`) is what makes this workable. |
| *"Prefer extending an existing topic file over creating a new one; this granularity overrides the advice to promote sections into files."* | Instruction only. | **Explicit override of A.1 SHAPE's last sentence.** |
| A.1 everything else. | Unchanged (shared blocks). | See F4-fact.md. |
| Tools: seven-tool write set (as F4-fact). | `fs_tools.ALLOWED_TOOLS` = `Read(./**)` (Read, Glob, Grep), `Edit(./**)` (Write, Edit) — both confined to the store by permission rule — plus `delete` and `rename`, real tools served in-process (`stores/fs_tools.py`: relative normalized paths inside the tree, never `.git/`/`_event/`/`_unprocessed/`, no overwrite). No shell: Bash, Skill, web, sub-agents denied. | The harness's sandboxed Bash cannot start on the build host (`bwrap: loopback: Failed RTM_NEWADDR`; every `rm`/`mv` of the first build failed), so delete/rename became dedicated tools — closer to the paper's seven-tool set than Bash was. Checked live: deletes/renames execute; `../x`, `/tmp/x`, writes and reads outside the tree are refused. |

## Read access

Read through Harnext's production interface: the MCP tool `context_research(question)` (kind
`research`), i.e. the product's research agent over this store's checkpoint — see README
"Read access" for the recorded INDEX.md-prompt deviation shared by every F arm.

## What an examiner should know

- F4-topic is the arm most exposed to the **Edit-tool cost of large files**: every window the
  builder must find the right `###` in a long file and edit in place; `Edit` requires a unique
  `old_string`, which gets harder as files grow. Expect the highest per-window turn counts and the
  most "no change to the store" / timeout problems (`llm_files.py:209`, `229`) — the `build-log`
  `attempts[].turns/seconds/problem` fields are the evidence.
- The reader side is the mirror image: `grep -n` finds the line, but `cat` of a topic file may be
  thousands of lines; readers must use `sed -n` ranges. This is the navigation-cost hypothesis of
  blueprint D05 ("fine granularity × navigation cost") at the coarse end.
- No line cap and no verification of "one entity in one topic"; both are prompt rules.
- Failed windows → `_unprocessed/` (README §5).
- Shares embedding of nothing: like all file arms it is read by grep only, so a paraphrased topic
  name that the builder chose (e.g. `topics/tiered-storage.md` vs a task asking about "KIP-405")
  is only findable if the builder kept the identifier literal — the READER block asks for that.
