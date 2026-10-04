# Arm `S-tools` — time-frozen systems of record behind `gh` / `jira` / `mail`

Source: no paper. E2-V2-DESIGN "Genuine baseline (decided 2026-09-20)": *"time-frozen replicas of the
systems of record, reachable with the usual CLIs … This is how an agent without a context engine
would work in a company."* Implementation: `sources/index.py` (build), `sources/cli.py` (the three
programs, stdlib only, run inside the agent image). Shared decisions: [README.md](README.md).

| design says | what the code does | deviation (and why) |
|---|---|---|
| Replicas "as of the task's checkpoint"; nothing later reachable. | One cumulative SQLite is built from the stream; at every boundary `VACUUM INTO export/<YYYY-MM>/sources.sqlite` (`index.py:319-329`), a physically separate file that never saw a later event. Read-only, `immutable=1` (`cli.py:55`). | Stronger than an `as_of` filter, on purpose (`index.py:1-9`): `strings`, `python -c sqlite3`, anything the agent does cannot reach the future. Cost: 42 snapshots, hence zlib for texts > 200 chars and a contentless FTS5 index (`index.py:36`, `66`, `84-90`). |
| Entity state = state as of the boundary. | Rebuilt from events only: status/assignee/… are the `to` of the last change before the boundary (`index.py:211-226`); a PR merged after the boundary is still `OPEN` (`index.py:239`, `271-285`). Audit (`TIMED`, `index.py:76-81`) proves every timestamp column `< boundary`. | none. Issues created before the corpus start show `(none)` fields with a note (`cli.py:471-472`). |
| "`gh` (pr/issue list, view, diff, search; `gh api` for the covered endpoints)". | `cli.py:23-27` `SUPPORTED`. `gh pr list/view/diff/checks`, `gh search prs|commits`, `gh api` GET on `repos/apache/kafka[/pulls[/n[/comments|/reviews|/files]]|/issues/n/comments|/commits/sha]` (`cli.py:266-296`), `gh repo view`. | **`gh pr diff` prints no patch**: the replica holds changed-file paths and the merge commit, and tells the agent to `git show <sha>` in the workspace if it is an ancestor (`cli.py:217-225`); unmerged PRs have no diff at all. **`gh issue *` always exits 2** — Kafka tracks issues in JIRA, the message says so (`cli.py:342-343`). `gh pr checks` unavailable (no CI in the corpus, `cli.py:340-341`). `gh api` is GET-only (`cli.py:347-348`), no GraphQL, no pagination (responses are complete, `--paginate` accepted and ignored). `--repo` ignored (`cli.py:326`). |
| "`jira` (issue view with comments and change history, search)". | `jira issue view KEY [--comments N] [--history] [--raw]` (`cli.py:528-532`); `jira issue list [-q JQL-lite] [filters]` (`cli.py:533-550`). | **JQL-lite**, not JQL (`cli.py:28-34`, grammar `cli.py:372-427`): clauses joined by `AND` only, fields `project/key/status/assignee/reporter/priority/resolution/type/component/text/summary/description/created/updated`, `is EMPTY`, `ORDER BY created|updated|key`. No `OR`, `IN`, functions, `labels`, `fixVersion` clauses. `--comments N` shows the **latest** N (`cli.py:443`), not the first N as jira-cli does. Output format is this tool's own, not jira-cli's table. |
| "`mail` (search/read of the dev lists)". | `mail search <query> [--list --from --after --before --limit]`, `mail read <id>`, `mail thread <id> [--full]` (`cli.py:572-614`). | No real-world equivalent; modelled on `notmuch`-style verbs. Search = all words must occur (FTS5 AND) newest first; quoted phrases and punctuated tokens (`KAFKA-123`, `a.b.C`) are re-checked on the decompressed text (`cli.py:79-88`). No ranking. |
| Same PATH in every arm. | Without `$HARNEXT_SOURCES` every command prints "not available in this trial" and exits 2 (`cli.py:621-627`). Every invocation ends with `# offline replica, data as of <boundary>` on stderr (`cli.py:641`); `--version` says `0.0.0-offline-replica` (`cli.py:630-631`). | none — the agent is never misled into believing it reached the live services. |
| Answerability audit must pass. | `sources/audit.py` (scripted oracle over the CLIs) — required before any run; not inspected here. | — |

## What an examiner should know

- **This is the baseline**, not `N`. Store arms are compared to it (E2-V2-DESIGN).
- The CLIs mimic `gh` and `jira-cli` in command and flag names so habits work, but they are
  **reimplementations over the stream**, not the real programs; anything not replicated says so and
  exits 2 (`cli.py:105-107` prints the supported set on every argparse error). An agent that knows
  the real `gh` will hit: no `gh issue`, no diff text, no `--json` on `gh api`, GET-only, JQL-lite.
- The replica's content is bounded by the corpus: GitHub bodies are harvest-time text and GitHub
  edit history is absent (`stream-manifest.json:known_limits`); comments edited later are counted.
- Long changelog fields (description, comment, environment) are clipped at 500 chars in history
  rows (`index.py:37-38`, `125-130`); the full text is in the issue/comment rows.
- The FTS index is `detail=none` (no positions), so phrase search is emulated by substring
  re-check after retrieval (`cli.py:86-88`): exact but slower on very common words.
- Coding tasks: `S-tools` also has the pruned git workspace, like every arm.
- Time is UTC ISO in every CLI output; `created <= 2023-01-31` means through end of day
  (`cli.py:421-423`), as in JIRA.
