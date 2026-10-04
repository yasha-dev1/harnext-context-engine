# E2 engine — raw store + file catalog (stage 1 of 3)

Written 2026-09-28. Stage 1 = ingest side and read API (this file, `src/harnext_eval/e2/v2/stores/raw/`).
Stage 2 = the query-time engine (Jev 1.13 picks sources, gpt-oss-20b writes SQL per source,
aggregation, cited answer). Stage 3 = eval / MCP / docker integration. Change this file before
changing code that crosses those boundaries.

## Why

w2026 (Jan–May 2026, 31,566 events, 90 tasks): `S-tools` (offline `gh`/`jira`/`mail` over a
relational replica) answered 97 % of factual history; every LLM-curated file store 26–83 %. The
replica keeps the raw data with all fields; the curated stores lose fields. The engine therefore
keeps ALL raw data per source in an analytics database and tracks each source in the file store
as a catalog the query-time models read.

## Decisions

- **Engine: embedded ClickHouse (`chdb` 4.4.0, engine 26.7.2.1)** for build and eval exports; the
  same SQL dialect as a production ClickHouse server. Everything goes through one small interface
  (`raw/backend.py`: `command`, `insert`, `select`, `close`) with two implementations:
  `ChdbBackend` (used, tested) and `HttpBackend` (clickhouse-connect; written, NOT exercised — no
  server on this machine). The fallback (a ClickHouse server container on `harnext-e2-net`, one
  read-only database per checkpoint) is not needed.
- **Raw payload = the stream event line.** `out/corpus/kafka/replay/*.jsonl` holds rawer CloudEvents,
  but they carry harvest-time state (final summary/description/components, `merged_at`,
  `closed_at`, …; see `stream.py` docstring), i.e. the future. The stream renderer is the leak
  filter every arm shares, so `raw` is the complete stream record (`canonical(event)`,
  byte-identical to the events.jsonl line — checked for all w2026 and smoke lines).
- **Exports are Parquet, loaded into a private scratch database at open.** chdb cannot open a
  read-only database directory, and views over `file()` need `readonly=2`, which lets a query read
  any file and write files (`INSERT INTO FUNCTION file(…)` wrote a file in the spike). Loading the
  export into a scratch MergeTree lets every model query run with `readonly = 1` while the export
  itself is never written. `readonly = 1` alone is NOT a sandbox in chdb (see "Read-only
  sandbox"): a statement guard does the rest.
- **Sources = one table per event kind** (9 for Kafka), grouped by system (JIRA, GitHub, mail).
  Finer than "jira/github/mail" because each kind has its own fields: a table has few mostly-empty
  columns, and the SQL for the question families is a one-table filter or a two-table join.
  PR lifecycle steps share one table (`action` = opened/merged/closed) because state-at-time is
  then `argMax(action, time)`.

## Sources and tables

Every table has the base columns `event_id String, event_type LowCardinality(String), time
DateTime64(6,'UTC'), actor Nullable(String), title String, text String (full rendered text),
entities Array(String), raw String (full record)`, `ORDER BY (time, event_id)`. Extracted fields
(all columns after the base ones; counts incl. base, w2026 2026-05):

| table | events | rows | cols | extracted fields |
|---|---|---|---|---|
| `jira_issues` | jira.created | 465 | 19 | issue_key, issue_id, summary, description, reporter, at, status, priority, assignee, labels, components |
| `jira_changes` | jira.change | 8,382 | 24 | issue_key, issue_id, at (µs), field, from_value, to_value, history_id, item_index, issue_summary, author, components, body (description edits); links: link_action, link_relation, linked_issue, linked_pr |
| `jira_comments` | jira.comment | 945 | 18 | issue_key, issue_id, comment_id, issue_summary, author, at, body, body_issues, body_kips, components |
| `github_pull_requests` | gh.pr_opened/merged/closed | 2,275 | 20 | pr_number, action, pr_title, author, merged_by, base, head, merge_commit, changed_files, body, issues, kips (closed_by when recorded) |
| `github_reviews` | gh.review | 5,617 | 18 | pr_number, pr_title, review_id, body, body_issues, body_kips, issues, reviewer, state, kips |
| `github_review_comments` | gh.review_comment | 6,717 | 19 | pr_number, pr_title, comment_id, author, path, line, body, body_issues, body_kips, issues, kips |
| `github_pr_comments` | gh.issue_comment | 2,947 | 17 | pr_number, pr_title, comment_id, author, body, body_issues, body_kips, issues, kips |
| `github_commits` | gh.push | 777 | 16 | sha, branch, author, headline, message, prs, issues, kips (changed_files: none in the w2026 stream, so the column does not exist there) |
| `mail_messages` | mail.message | 3,441 | 19 | message_id, list, subject, sender, thread_id, in_reply_to, body, issues, kips (both from the SUBJECT), body_issues, body_kips |

`body_issues` / `body_kips` (store version 3, data QA): the JIRA keys and KIP ids written in the
item's own text, distinct, in order of first mention (case-insensitive, stored upper-case). The
subject/PR-linked `issues`/`kips` stay as they were and are described precisely: for mail they come
from the subject only (KIP-1071: 2 rows by subject, 91 by body), for comments and reviews they are
the pull request's. "All messages discussing KIP-n" is `has(kips, 'KIP-n') OR has(body_kips,
'KIP-n')` (a worked query).

Extraction (`raw/sources.py`): `meta`, the `- name: value` fact lines of `text` (parsed like
S-tools' `sources.index.parse`/`split_files`, except that `sources.page` finds the facts after a
title that spans lines — 6 w2026 `jira_changes` rows with field `Comment` had a multi-line value in
their headline and lost `at`, `author`, `issue_summary`; a field change carries `body` only for
`description` edits), the event id (issue/history/item ids) and entities.
Dropped only where the value is already a column: fact `event` (= event_type), fact `at` of
GitHub/mail events (= time; verified equal for every w2026/smoke event), facts duplicated from
`meta`. JIRA keeps `at`, the exact microsecond instant (task gold uses it). fixVersion/Version
lists are comma-joined in from_value/to_value (w2026 lists have ≤ 1 element; `raw` keeps the
list). Type mapping is exact (`sources.TABLES`): any other event type — also a near miss such as
`jira.changes` — gets its own table `other_<snake(type)>` (catalog system "unrecognised event
type"), never a known table. An unknown fact line or meta key becomes a column without code
changes, named by `Schema.column` (below). Fact lines that render an extracted field or a meta key
of the same name are not stored twice. "Who did it" is `actor` (always set); `jira_changes.author`
is the printed 'by' line and is NULL on link rows (68 %) — the catalog says so. `jira_issues.at`
equals `time` (whole seconds), and the catalog says that too. People are named per system with no
mapping (JIRA display names, GitHub logins, commit author names, mail sender names); no event in
the stream pairs a login with a name, so there is no `people` table — the catalog states it
(`catalog.json` / `sources.json` → `people`). Static meanings/joins/worked SQL per source:
`sources.COLUMNS`, `TABLE_COLUMNS`, `JOINS`, `EXAMPLES`, `ABOUT`, `KEYS`, `ENUMERATIONS`, `PEOPLE`.

**Time zone** (data QA): every function now runs in UTC, whatever the host zone —
`session_timezone = 'UTC'` on the builder's session and in `query.limits()` for every read. Before,
`toDateTime('2026-03-01 00:00:00')` and friends used the host zone (Europe/Bratislava: one hour
off). Tested in subprocesses with `TZ=UTC`, `Europe/Bratislava`, `America/Los_Angeles`: identical
conversion results, `toString` of time columns, `toStartOfMonth`/`toStartOfDay`, `dateDiff`,
`formatDateTime`, oracle totals, and byte-identical builds.

## Schema evolution (`raw/schema.py`)

- Type per value: bool→Bool, int→Int64 (an int outside Int64 → String, exact digits),
  float→Float64, ISO-8601 UTC instant→DateTime64(6,'UTC') (fractions beyond microseconds are
  truncated in the column; `raw` keeps them), other str→String, list→Array(elem) (mixed
  elements→Array(String)), dict or nested list→String (JSON). Scalars are `Nullable(…)`, arrays
  not (`[]`). `null`/`[]` carry no type (no column yet).
- Unseen field → `ALTER TABLE … ADD COLUMN … Nullable(T)`; earlier rows read NULL / `[]`.
- Column names: a field without a fixed column (unmapped fact line or meta key) is named
  `snake(key)`; a key without ASCII letters/digits gets `field_<sha256[:6]>`. If the name is taken
  (a base column such as `time`/`raw`, or another field, e.g. `My Field` then `my.field`) it
  becomes `<meta|fact>_<name>`, then `…_2`, … — first come, first named, never two fields in one
  column. The origin of each such column (`meta:<key>` / `fact:<line>`) is recorded in
  `schema.json` → `names` and printed in the catalog ("original meta key '…'"). w2026: no renames.
- Conflict → int vs float widens to `Float64` (arrays: `Array(Float64)`); any other conflict to
  `Nullable(String)` (array vs array → `Array(String)`). ClickHouse's own
  `MODIFY COLUMN` would print arrays as `['a','b']`, so a widening rebuilds the table from its `raw`
  column: every value of the column, old or new, is encoded by the same function (JSON for
  lists/dicts, `true`/`false`, `str()` for numbers).
- Every change is logged `{table, column, change: create|add|widen, type, from_type, event_id,
  time}` (`schema-log.jsonl`). Types depend only on values in stream order, so a rebuild gives
  the same schema; two w2026 (and smoke) builds into scratch roots gave byte-identical export
  trees. Real w2026 data: 9 creates, 88 adds, 0 widenings, 0 renames (all covered by tests).

## Build, checkpoints, export

`python -m harnext_eval.e2.v2.stores.build --arm RAW --scope smoke|w2026 [--until YYYY-MM]`
(`RawStore`, `raw/store.py`). One pass over the stream into a working chdb DB (`work/`, deleted
after the build); at each monthly boundary the build writes

```
export/<YYYY-MM>/tables/<table>.parquet   rows with time < boundary, zstd, ORDER BY (time, event_id)
export/<YYYY-MM>/schema.json              column types per table as of the boundary, rows, names (origins)
export/<YYYY-MM>/schema-log.jsonl         schema changes with time < boundary
export/<YYYY-MM>/catalog/                 SOURCES.md, sources.json, <table>.md, catalog.json
export/<YYYY-MM>.manifest.json            sha256 per file, content_sha256, rows, bytes (no timestamps)
```

An export is written to `.<ckpt>.tmp`, made read-only, renamed; it never saw a later event
(tested on every DateTime column and on `raw`'s own `time`: nothing ≥ boundary; no later column
or sentinel in the files). The runner's
`checkouts.identity(store, ckpt)` pins `<ckpt>.manifest.json` unchanged. Resume: `build-state.json`
moves only at a checkpoint; a rerun rebuilds `work/` from the last export (it holds every row and
the exact schema) and skips the stream up to that boundary (tested: kill after a checkpoint →
resume = straight build, byte for byte). Usage/cost 0. Every checkpoint is exported (the resume
point needs it; w2026 is small).

Disk: w2026 exports 3.3 / 6.3 / 9.9 / 13.3 / 16.8 MB (2026-01…05), 48 MB total; smoke 2.3 + 2.6 MB.
Build: w2026 23 s wall, 0.83 GB max RSS; smoke 6.4 s. Full-history estimate (629k events at
~0.53 kB/event) ≈ 330 MB for the last export, ~10 GB for 42 cumulative exports: export only the
checkpoints a run needs there (open issue 4).

## File catalog (`raw/catalog.py`)

- `SOURCES.md` — one paragraph per source: system, rows, time range, column count, what one row
  holds and what identifies it (`about`, `use_for` — the key keeps its name for API stability, its
  text is a description of the data), key columns; its header states that times are UTC, that time
  columns accept `'2026-03-01 00:00:00'` and `'2026-03-01T00:00:00Z'` (or
  `parseDateTime64BestEffort('…Z')`), that `actor` names who caused each event, plus the people
  note; each source paragraph ends with its window notes. `sources.json` — the same,
  machine-readable. Together ≈ 10 kB (≈ 2.6k tokens) at 2026-05: fits Jev's 32k context with room
  for the question.
- `<table>.md` — every column: type, meaning, null % (two decimals; never "0" for a non-zero
  share: 6 of 8,382 prints 0.07, below 0.01 % prints "<0.01"), distinct count, examples, notes,
  joins, worked queries. `catalog.json` — everything (≈ 81 kB).
- Examples (data QA): true enumerations (`sources.ENUMERATIONS`: event_type, field, action, state,
  status, priority, resolution, link_action, link_relation, list, branch, base, components, labels,
  item_index) show their 5 most frequent values, with the note that values are case-sensitive
  (`Reopened` vs `approved`); every other column shows its 3 earliest distinct values by event time
  — formats, never the busy entities tasks ask about (the most-frequent rule showed task subjects
  such as KAFKA-20134, KIP-1274 and coding-task titles). Regression test: examples are recomputed
  and must equal that rule; identifying examples that also occur in a w2026 prompt are only
  earliest values by chance — 5 at 2026-02/05 (KIP-1251 in four columns, KIP-848 in one) — and none
  equals a non-person gold value.
- Window notes, counted at build time from each export's own rows (`catalog.NOTES`), facts only —
  no advice: mail threads vs threads beginning in the export (the message whose `message_id`
  equals `thread_id` opens its thread), messages without `in_reply_to` that do not open their
  thread, PRs that lack an 'opened' row (opened before the export began), changed issues without a
  `jira_issues` row. 2026-05: 1,460 threads of which 394 begin in the export (11 such messages);
  1,341 PRs, 139 of them without an 'opened' row; 823 changed issues of which 396 were created
  before the export.
- Structured links (`catalog.json` → `links`, also `db.catalog()["links"]` and, per table,
  `db.schema(t)["links"]`), the prose joins as data (`sources.LINKS`, 23 at 2026-05): `{"from":
  {"table", "column", "kind": scalar|array}, "to": {…}, "meaning", "value_shape", "sql"}`. `kind`
  comes from the schema; `value_shape` is inferred from the column's values (`catalog.shape`: the
  first of digits→`<n>`, alphanumeric runs→`<w>`, runs between `@ / : #`→`<text>` that fits 90 % of
  the distinct values — `KAFKA-<n>`, `<n>`, `KIP-<n>`, `<text>@<text>`, `<w>`; nothing project-
  specific); `sql` is a runnable join (array sides via `arrayJoin`). A link whose columns an
  export lacks is left out. Tested: every link runs through the guarded `sql` and returns rows at
  2026-05. Example: `{"from": {"table": "github_pull_requests", "column": "issues", "kind":
  "array"}, "to": {"table": "jira_changes", "column": "issue_key", "kind": "scalar"}, "meaning":
  "issues a pull request is linked to", "value_shape": "KAFKA-<n>", "sql": "SELECT count() AS pairs
  FROM (SELECT arrayJoin(`issues`) AS k FROM github_pull_requests) AS a INNER JOIN (SELECT
  `issue_key` AS k FROM jira_changes) AS b ON a.k = b.k"}`.
- Key columns of `jira_changes` include `actor` (always set; `author` is NULL on link rows).
- The SQL the read API accepts is NOT part of an export (data QA: exports listed six functions the
  guard had since refused). `RawDB.sql_reference()` renders it from the guard in force
  (`guard.summary()`): statement shapes, function groups, combinators, time pitfalls (`IN` and
  function arguments need `parseDateTime64BestEffort('…Z')`; `dateDiff('day', …)` counts calendar
  boundaries, full 24-hour periods are `intDiv(dateDiff('second', a, b), 86400)`; exact instants print with
  `formatDateTime(at, '%Y-%m-%dT%H:%i:%S.%fZ')`), backslashes and case-sensitivity. Stage 2 shows
  it only with the fallback `sql` tool.
- Meanings of known columns are static (per-table overrides in `TABLE_COLUMNS`); an unknown column
  gets "Auto-detected: original meta key '<key>' (<type>, first seen in event <id> (<time>)); e.g.
  …" — no LLM.
- Worked queries (`sources.EXAMPLES`, 21 templates) demonstrate SQL technique over this schema and
  are titled by technique: text search (ILIKE, hasTokenCaseInsensitive), array membership (one or
  two columns, on paths), a time range with explicit instants, group and count, the latest row
  per key before an instant (argMax over the ordering columns), filter by key in order, a union of
  tables with the latest value per key, a join through a link, a thread in order, and "rows of one
  action vs all rows" (the two pull-request variants). Their values are picked from that export's
  own data: the FIRST qualifying entity (earliest time, then value) — stable across checkpoints
  and unrelated to how busy an entity is; words for text search are the export's most frequent
  6–9-letter words; dates are `{month}` (first day of the checkpoint month) and `{boundary}`,
  never later. A template is printed only if every pick finds a value and the query returns a row
  with a non-empty first value: 20 of 21 per export ("commits changing one file" is dropped:
  commit file lists are absent from the streams). Everything is ordered deterministically, so the
  same data gives the same bytes. No benchmark file is read by the build; a test checks that no
  identifying literal of any worked query (keys, numbers, paths, titles) occurs in a w2026
  benchmark prompt or non-person gold value.
- Benchmark-neutral prose (methodology review, 28 Sep): descriptions say what the data holds, not
  which questions it answers. Worked queries that existed only because a question family needed
  them were dropped (reopen count, assignee and fix-version histories, links held now, who
  approved, top approvers of an author, files changed by an issue's merged PRs, merges of an author
  since a date, file with most inline comments, a KIP [DISCUSS] thread, distinct senders of a
  thread) or replaced by a technique on a generic pick (field history → "filter by key, in order"
  with the field picked from the data). `test_catalog_prose_does_not_echo_the_benchmark` builds
  word 3-grams (stop words, entity ids and dates removed) from `families.py` and the public prompts
  and fails if catalog prose (descriptions, notes, joins, link meanings, worked-query titles, the
  SOURCES header) shares more than 3 of them; the SQL reference and refusal hints must share none.
  Result: 0 shared 3-grams (of 4,955 benchmark 3-grams) at 2026-05, 2026-01 and smoke; before the
  rewording there were 9 (e.g. "strictly checkpoint boundary", "changelog history id", "review state approved", "merged pull
  requests", "thread first message").

**Knowledge of the benchmark during design.** The tables and the catalog were designed after
reading the benchmark's family definitions (`families.py`, the task prompts) and an answerability
check was run against the final benchmark's gold (`raw/oracle.py`). What that influenced: the table
layout and column extraction are general for these three systems — one table per event kind, every
`meta` key and fact line kept, typed and named by generic rules, nothing selected per question
family (the `body_issues`/`body_kips` columns came from a data-QA finding, the `\r` repair from a
parse defect) — but the choice to keep exact JIRA instants (`at`) and link parsing
(`link_relation`, `linked_issue`, `linked_pr`) was informed by knowing such questions exist. The
catalog wording mirrored the families (use-for lines, worked queries) until it was neutralised on
28 Sep after review. The oracle's implicit-link rule (KIP in the subject, else nearest summary) was
written after seeing that family and its 14/14 is therefore optimistic. The store code reads no
benchmark file; gold is used only by the oracle and the tests.

## Query API for stage 2 (`raw/query.py`)

```python
from harnext_eval.e2.v2.stores.raw.query import open as open_raw   # also: RawDB, QueryError, over
db = open_raw(export_dir, scratch=None)  # RawDB; export may be read-only; scratch = dir for tmp DB
db.catalog() -> dict        # sources.json: checkpoint, boundary, events, dialect, sources[...]
db.schema(source) -> dict   # table entry of catalog.json: columns[{name,type,description,null_share,distinct,examples}], joins, examples[{question,sql}]; KeyError lists valid sources
db.tables -> dict[str, list[str]]  # table -> column names
db.sql(query, max_rows=200, max_bytes=200_000, timeout_s=10.0) -> dict
    # {"columns": [[name, type]], "rows": [dict], "truncated": bool, "elapsed_ms": float,
    #  "error": str | None, "note"?: str}
db.get(event_id) -> dict | None   # {"source": table, **full raw record}
db.sql_reference() -> str   # the accepted SQL (from the guard in force), for the fallback sql tool
db.close()  # or `with open_raw(...) as db:`
```

`db.catalog()` now also carries `people` (the naming note) and `links` (structured joins); each
`sources[…]` entry carries `notes`; `db.schema(t)` carries `links` touching `t`. New columns stage 2 must learn: `body_issues`, `body_kips` on `mail_messages`,
`jira_comments`, `github_pr_comments`, `github_review_comments`, `github_reviews`. No table was
added or renamed; `catalog/SQL.md` and `catalog.json → sql` are gone (use `sql_reference()`).

- `sql` accepts ONE `SELECT …` / `WITH …` statement over the export's tables, `DESCRIBE [TABLE]
  <export table>` or `SHOW TABLES` (a trailing `;` is fine; comments are rejected). The text that
  runs is exactly the text checked. See "Read-only sandbox" for what is refused where.
- Rows beyond `max_rows`, or beyond `max_bytes` of JSON, → `truncated` (+ `note`); both are also
  bounded inside the engine, so a huge result never reaches Python whole.
- Errors are short and actionable: `UNKNOWN_IDENTIFIER: Unknown expression identifier `x`. Valid
  columns: jira_changes(…)`, `UNKNOWN_TABLE: … Tables: …`, `TIMEOUT_EXCEEDED: …`, a type error with
  the hint how to write a time literal (also on ILLEGAL_TYPE_OF_ARGUMENT), or the guard's own
  message: `function uniqExactt is not available; closest allowed: uniqExact, …`, explicit
  synonyms (`lag` → `lagInFrame(x) OVER (…)`, `lead` → `leadInFrame`, `now`/`today` → "the current
  time is not available; compare with explicit dates", `string_agg` → `groupConcat`, …),
  `unknown table 'jira_issue'; did you mean jira_issues?; tables: …` (never glued to the next
  token), a reserved word used as a name → "quote it with backticks". A name after `AS` is an alias
  even when it is a reserved word (`SELECT from_value AS from, to_value AS to …` runs).
- `harden(memory_bytes=8 GiB, cpu_seconds=None, file_bytes=4 GiB)` — optional OS caps
  (`setrlimit`, soft AND hard: data segment, largest written file, no core dumps; total CPU if
  given, hard 10 s above soft) and a SIGTERM handler that removes this process's scratch dirs, for
  the process that serves `sql`; the MCP server calls it at start, importing `query` applies
  nothing. RLIMIT_AS is not used (chdb reserves far more virtual memory than it touches).
- Time literals: `time >= '2026-03-03T00:00:00Z'`, `'2026-03-03 00:00:00'` and
  `parseDateTime64BestEffort('…Z')` all work on every time column (scratch tables are sorted by
  `event_id`, so the strict key-condition cast that rejected `'…Z'` on the sort key `time` never
  applies; `cast_string_to_date_time_mode = best_effort` is set too).
- Values: DateTime as `'2026-03-01 12:00:00.123000'` (UTC), Int64 as numbers, arrays as lists.
- Scratch: `open` refuses a second export before creating anything; its scratch dir
  (`<tmp>/harnext-raw-<pid>-*`) holds `.lock`, flock-ed for the life of the open export (taken on a
  temporary name, then renamed, so no one sees it unlocked); it is removed on failure, on
  `close()`, at interpreter exit and — after `harden()` — on SIGTERM. `open` sweeps only dirs whose
  `.lock` it can lock (the kernel drops a flock when its owner dies, in any pid namespace sharing
  the filesystem); dirs without a lock file are left alone. Data QA saw the stage 2 agent's live
  scratch dirs disappear: the previous sweep removed a `harnext-raw-<pid>-*` dir when `kill(pid,
  0)` found no such pid, which is also true for a LIVE process in another pid namespace (a trial
  container) sharing the host's `/tmp`, and for a process whose dir name carried another pid — so
  yes, it could have removed them; the flock rule cannot. (SIGKILL still leaves a dir behind until
  the next sweep.)
- One export per process: chdb binds one embedded server to one directory (`close()` then `open()`
  another works), and each open/close cycle leaks ~15 `/proc` handles inside chdb (fds 83 → 218
  over 10 cycles, RSS 453 → 585 MB; queries themselves do not grow them, 1,000 queries: 79 → 79):
  a process should open one export and keep it. For concurrent checkpoints: one process each, or
  `over(backend, catalog_dir)` against a server.
- Typical patterns (all in the catalog): state at time `argMax(to_value, (at, history_id,
  item_index))`; history `ORDER BY at, history_id, item_index`; issue→PRs `has(issues, 'KAFKA-N')`
  (any action: a PR opened before the export has no 'opened' row); title→key over `issue_summary`/`summary`; KIP threads via
  `thread_id`; fuzzy title match `ngramDistanceCaseInsensitiveUTF8`; text `ILIKE`,
  `hasTokenCaseInsensitive`, `multiSearchAnyCaseInsensitive`.

## Read-only sandbox (`raw/guard.py`, `raw/query.py`, `raw/backend.py`)

Reviewed 2026-09-28: the first version appended `SETTINGS readonly=1, …` to the statement text, so
a trailing `--`/`#` commented the limits out (read `/etc/hostname` and the private gold file,
fetched a URL, wrote a file, ran 14 s past a 2 s limit), and `DESCRIBE file(…)`/`url(…)` read files
and the network even under readonly = 1. Now two independent layers:

1. **Engine** (cannot be changed by any statement): each `sql`/`get` call opens a fresh chdb
   connection (< 1 ms) whose URL carries `readonly=1, max_execution_time, timeout_overflow_mode=
   throw, limit / max_result_rows = max_rows+1, max_result_bytes = 2·max_bytes+1 MB,
   result_overflow_mode=break, max_memory_usage=1 GB, max_bytes_in_join=250 MB,
   join_overflow_mode=throw, max_threads=4`. With readonly = 1 a statement's own `SETTINGS` or `SET`
   is refused (tested by calling the backend with hostile text directly, bypassing the guard:
   `getSetting('readonly')` = 1, `SETTINGS readonly=0` / `SET` / `INSERT INTO FUNCTION file` /
   `file()` table function / `CREATE` → READONLY). The engine blocks writes, DDL, settings changes
   and table functions in FROM.
2. **Statement guard** (tokenizer: strings, quoted identifiers, numbers, words, operators). Blocks
   what readonly = 1 in chdb lets through — measured: scalar `file('/etc/hostname')` (chdb ignores
   `user_files_path` for absolute paths, so no engine setting confines it), `DESCRIBE file(…)`,
   `hostName()`, `filesystemAvailable()`, `SELECT … FROM system.*`, `numbers()` — and everything
   else outside the export: comments of any kind, heredoc `$$`, query parameters, more than one
   statement; any word from INSERT/ALTER/CREATE/DROP/SET/SETTINGS/FORMAT/INTO/OUTFILE/USE/SYSTEM/
   EXPLAIN/…; the identifiers `system` and `information_schema`; every FROM/JOIN target, also in
   subqueries, CTEs, IN and comma joins, must be an export table, a CTE name or a subquery (no
   qualified names, no table functions); DESCRIBE only of an export table; SHOW only `SHOW
   TABLES`; and **functions deny by default** (second review): every call — a bare, back- or
   double-quoted name followed by `(`, unless the word is SQL syntax (`IN`, `EXISTS`, `OVER`,
   `INTERVAL`, `FILTER`, types inside `CAST(x AS Nullable(…))`, …) — must be on `guard.ALLOW`
   (case-insensitive) or be an allowed aggregate plus combinators `If`, `Distinct`, `OrNull`,
   `OrDefault`, `Array` (`countIf`, `argMaxIf`, `groupArrayDistinct`, `uniqExactArray`). A
   function the list does not name — also one a chdb upgrade adds — is refused.

The allowlist (302 engine spellings): comparison and logic 16, arithmetic and rounding 28, strings
and text search 64, date and time 59, arrays and higher-order 30, JSON over `raw` 25, type
conversion 26, conditionals and null handling 9, aggregates 30, window 9, tuples and maps 6. Left
out on purpose: anything touching files, network (incl. chdb's `ai*` functions), environment,
server/host/process state, dictionaries, other databases, sleeping, randomness, the current time
(`now`, `today`: results stay time-frozen and deterministic), and data amplifiers (`repeat`,
`space`, `range`, `arrayWithConstant`, `leftPad`, …) — none was needed by the oracle, the worked
queries or 45 ordinary analytical statements. Tests read the engine's own registries through the
backend (1,825 functions incl. 228 aliases, 91 table functions, 15 combinators): all 1,611 names
not on the list, each in 5 case variants plus back- and double-quoted, in 4 statement forms
(42,856 statements) are refused; every allowlisted name exists in the engine and none is a table
function; the 10 non-allowed combinators on every allowed aggregate are refused; no allowlisted
name contains a denied part (`file`, `url`, `remote`, `s3`, `dict`, `sleep`, `host`, `env`,
`server`, `setting`, `user`, `database`, `rand`, `now`, `today`, `uuid`, `generate`, `merge`,
`input`, `view`, `joinget`, …) except `toDayOfMonth/Week/Year`, justified in the test.

**Lexer agreement and input hygiene** (third review). The guard is only right if it splits
literals and quoted identifiers where the engine does. The engine's lexer: a backslash escapes the
next character, a doubled quote is a quote. Decoding differs from the documentation: the docs say
`\c` (unknown c) is `c`, the engine keeps `\c`; `\N` gives `''`; `\xZZ` gives a garbage byte; `\x4`
is a syntax error. So the guard admits, inside `'…'`, only `''`, `\\ \' \n \t \r \0 \b \f \a \v`
and `\x00`–`\x7F` (where docs and engine agree) and refuses every other escape — a regex backslash is
written `\\` (`'\\s+'`, `'\\['`); quoted identifiers (`` `…` ``, `"…"`) take doubled quotes only, no
backslash. Two values in a row (`'a' 'b'`, `'a' x 'b'`, `'a'(`, `` `a` b `c` ``) are refused as
the engine refuses them, so stray quotes cannot shift the guard's reading. Outside literals and
quoted identifiers only ASCII; control characters other than tab/LF/CR are refused everywhere;
a statement is at most 20,000 characters (`guard.MAX_LENGTH`). Differential test (seeded): 2,400
generated literals (quotes, doubled quotes, backslash escapes incl. unknown ones and one before the
closing quote, `\x` forms, newlines, `;`, `(`, `file(`, `--`, `#`, non-ASCII letters): 914 admitted,
each tokenised as one literal and returned by the engine as exactly one column `s` whose value
equals the test's own decoding by the documented rules; every rejection is a case the reference
rules cannot decode or that contains a control character. Identifiers: 1,000 per quote style,
146 admitted, each the single column name the engine returns. No disagreement remained after the
fixes above (the first version accepted the unknown escapes and backslashes in identifiers).

Settings read back through `getSetting` in a hostile-free statement on the per-query connection:
`readonly` 1, `max_execution_time` 3, `timeout_overflow_mode` throw, `max_result_rows` 8,
`max_result_bytes` 1,002,000, `result_overflow_mode` break, `max_memory_usage` 1,000,000,000,
`max_bytes_in_join` 250,000,000, `join_overflow_mode` throw, `max_threads` 4,
`cast_string_to_date_time_mode` best_effort (for `limits(7, 1000, 3)`). `limit` is consumed by
chdb from the connection URL (getSetting shows 0) and is checked by its effect: 27 rows asked, 8
returned by the engine.

**Allowlist audit** — why each group is acceptable, and every function whose output can far exceed
its input or whose cost is super-linear, each either removed or driven to its limit by a test
(`test_amplifying_functions_are_bounded_by_the_engine_caps`: cross joins of all mail bodies, 4 kB
literals, quadratic lambdas; each stopped by the engine within the 3 s cap + 2 s, the process
growing < 1.3 GB):

| group | property | named functions: outcome |
|---|---|---|
| comparison and logic | pure; scalar output | — |
| arithmetic and rounding | pure; scalar output | — |
| strings and text search | pure; output ≤ input, or input × a factor set by a ≤ 20 kB literal | `replaceAll`, `replaceOne`, `replaceRegexpAll`, `replaceRegexpOne`, `concat`, `concatWithSeparator`: memory or time cap (0.6–3.0 s); `splitByChar/String/Regexp/Whitespace/NonAlpha` + `arrayJoin`: memory cap; `ngramDistance*`/`ngramSearch*` (linear, fixed-size n-gram table): time cap; `match`, `extract`, `regexpExtract` (RE2: linear time, no backtracking): time cap, `(a+)+$` 3.0 s; `multiSearch*`: engine limits 255 needles. **Removed**: `countMatches` (9.4–10.1 s past a 3 s cap), `extractAll` (6.1–6.3 s), `extractAllGroups` (4.0–4.5 s), `editDistance`/`levenshteinDistance` (O(n·m)), `multiMatchAny` (hyperscan compilation not interruptible) |
| date and time | pure; bounded output | `formatDateTime` (output = format length × rows): memory cap |
| arrays and higher-order | output ≤ sum of input arrays; lambdas linear per element | `arrayJoin` (rows × array length): time/memory cap; `arrayConcat`, `arrayFlatten`: memory cap; nested `arrayMap`/`arrayExists` over `groupArray` (quadratic): memory cap; `arraySort`: memory cap |
| JSON over `raw` | pure; output ≤ input | `JSONExtractKeysAndValues`: time cap |
| type conversion | pure; bounded output | — |
| conditionals and null handling | pure; output is one of the inputs | — |
| aggregates | output ≤ rows × value size | `groupArray`, `groupUniqArray`, `groupConcat`, window `groupArray … OVER (UNBOUNDED)`: memory cap; `topK(10^9)`: engine refuses the parameter |
| window | one value per row | — |
| tuples and maps | pure; output = inputs | — |

`EXPLAIN AST` was considered as a parser but chdb prints the tree truncated at depth ~4, so the
guard tokenizes itself. Results of the reviewer's statements now: every `… --` / `#` / `/* */`
form → "comments … are not allowed"; the same without the comment → "unknown table 'file('" /
"'url('" / "INSERT is not allowed"; `DESCRIBE file(…)` / `url(…)` → "DESCRIBE takes one export
table"; the triple cross join → `TIMEOUT_EXCEEDED` after 2.0 s; `SELECT a.raw, a.text, b.raw FROM
mail_messages a CROSS JOIN mail_messages b` with max_bytes 1 MB → truncated, RSS growth < 400 MB.

## Spike measurements (chdb 4.4.0, this machine)

| check | result |
|---|---|
| install | `chdb` 4.4.0 + `chdb-core` 26.7.3 (wheel 175 MB, 545 MB installed; needs pandas, pyarrow) |
| persistent on-disk DB, reopen | works; `import` 0.2–0.5 s, session open 0.03 s |
| `ALTER TABLE ADD COLUMN` | works (metadata only) |
| `MODIFY COLUMN` Int→String / Array→String | works; arrays become `['x','y']` → we rebuild from `raw` instead |
| `ILIKE`, `hasToken(CaseInsensitive)`, `multiSearchAnyCaseInsensitive` | work; 2026-05: ILIKE over 3,441 mail bodies 33–46 ms, hasToken over 6,717 review comments 11 ms. Text/skip indexes not evaluated (full scans suffice at this size) |
| open a read-only DB directory | fails: `Cannot open file …/status: Permission denied` |
| copy DB dir to scratch then open | works with `copytree(symlinks=True)` (Atomic DB uses relative symlinks): 48 MB dir 28 ms + 30 ms open |
| views over Parquet `file()` | work, but not under `readonly=1`; `readonly=2` allowed reading /etc files and writing a file |
| chosen: Parquet → scratch MergeTree | 2026-05 export (16.8 MB) opens in 0.9–1.0 s (+0.5 s import), scratch 33 MB, peak RSS ≈ 0.6 GB |
| settings in a chdb connection URL | apply to that connection only, readonly = 1 refuses `SET`/`SETTINGS`; connect < 1 ms, beside the loading session |
| readonly = 1 in chdb | still allows scalar `file()`, `DESCRIBE file()`, `hostName()`, `system.*`, `numbers()`; `user_files_path` does not confine absolute paths |
| same data as MergeTree dir vs Parquet | 48 MB vs 13 MB (single `events` table) |
| one embedded server per process | a second path while one is open fails; close + open another works |
| chdb quirk | a query failing mid-run (timeout) on the session leaves its partial output, prepended to the next result; the builder's session path drains it with a throwaway query; model queries use a fresh connection each |

## Answerability (w2026, `python -m harnext_eval.e2.v2.stores.raw.oracle`)

Hand-written SQL per family (`raw/oracle.py`), parameterised only from the public prompt, run
through the read API at the task's checkpoint, graded by `grade.grade_historical`. Report:
`out/kafka-500-v2/stores-w2026/RAW/answerability.json`.

- Factual history: **35/35 derived** (all 19 families: 8 v1 field replays, issue_prs,
  issue_changed_files, pr_approvers, review_hotspot, reopen_count, issue_links,
  fix_version_at_snapshot, resolution_vs_merge_order, author_activity, top_reviewer_for_author,
  kip_thread; 11 of them keyless, resolved by title in SQL).
- implicit_link: **14/14 derived** by one generic rule (issue whose summary names the thread's KIP,
  else the nearest summary by n-gram distance).
- design_decision: not SQL-derivable (a phrase chosen by reading the thread). **29/29** settling
  messages are in the export at the checkpoint; the gold value is verbatim in 21, paraphrased in 8.

## Open issues for stages 2–3

1. **Image** (stage 3): add `chdb==4.4.0`, `chdb-core==26.7.3`, `pandas==3.0.5`, `pyarrow==25.0.1`
   (+ their deps, e.g. python-dateutil, tzdata) to `docker/e2-agent/constraints.txt` and `chdb==4.4.0`
   to the Dockerfile's pip line (~0.8 GB larger). In the image (empty `harnext_eval` / `e2`
   package files) the read path imports only `stores.raw.{backend,guard,query,schema}` — tested
   by a copy of the tree laid out like the image. On the host, importing any `harnext_eval` module
   runs `harnext_eval/__init__.py` (→ `e2.arms`, providers, `harnext_builder`, `harnext_shared`);
   that is left as is, other modules' behaviour is not changed. `open` needs a writable scratch
   dir (~2× export) — container `/tmp` or `scratch=`.
2. **Arm/runner** (stage 3): no `arms.yaml` entry, MCP kind or instructions yet. Exports follow the
   non-git store layout, so `checkouts.identity` pins them unchanged.
   **Container requirements** for the process serving `sql`: benchmark private files
   (`tasks.private.jsonl`, `extra/`, `answerability.json`) absent from the image and from every
   mount; the export mounted read-only; scratch (`open(scratch=…)`, else `/tmp`) on a size-capped
   tmpfs (≥ 2× the export, e.g. 256 MB for w2026); non-root user; no network route except the
   model API through the allowlisting proxy; memory and pids limits (`--memory`, `--pids-limit`);
   `harden()` called at server start.
3. **Fairness note**: `stores-w2026/S-tools` is a symlink to the FULL-history store (2011→);
   RAW, like the w2026 file stores, ingests only Jan–May 2026. All w2026 factual gold is derivable
   from the window anyway (above), but an issue created before 2026 has no `jira_issues` row
   (its title is in `issue_summary`).
4. Full-history scope: ~10 GB for 42 cumulative exports; export only wanted checkpoints and resume
   from the last exported one (not implemented).
5. `HttpBackend` is untested (settings go as HTTP parameters beside the statement, like the chdb
   URL); `over()` needs the server database to hold one checkpoint, and a server would also need
   the guard (it is applied by `RawDB.sql` for every backend).
6. Stage 2 must return `event_id`s as evidence and use `get()` for full records; `text`/`raw` of
   mail can be large — select `substring(body, 1, N)` under `max_bytes`.
7. Functions are allowlisted (deny by default); a model needing a function not on `guard.ALLOW`
   gets a refusal with the closest allowed names — widening the list is a reviewed code change,
   re-checked by the registry tests.

## Stage 2: query-time engine

Written 2026-09-28. Package `src/harnext_eval/e2/v2/engine/`; tests `tests/test_e2_v2/test_engine_*.py`
(+ `engine_fixtures.py`); dev evaluation `scripts/e2_engine_dev.py`. No file of stage 1 was changed.

### Pipeline

```
research(question)
  route      1 Jev call: noul per source group (jira / github / mail, + memory if given) over the
             catalog's group paragraphs; every group >= 0.5, at least the top one, at most 3
  branches   one gpt-oss agent per chosen group, concurrently (<= 8 turns each)
               raw:    find(need, values) | read(event_id[, page]) | sql(query)* | done(finding, evidence)
               memory: search(need, keywords) | read_file(path, from_line) | done(...)
  judge      Jev: relevance of each finding to the question and whether it states facts at all
             (one call, when > 1 finding), and SUPPORT of each finding by the tool results its
             own branch retrieved (one call per finding); irrelevant or unsupported (< 0.3) ->
             dropped, the best one kept and flagged "NOT confirmed" when nothing passes; a
             "nothing found" finding (< 0.3) is dropped when another kept finding states facts
  report     1 gpt-oss call: direct answer, supporting facts with [citations], what is unknown
```

**Routing is per source group, not per table.** The nine tables of the catalog fall into three
systems whose tables share identifiers (issue key, PR number, thread id) and join to each other;
the catalog paragraphs of tables within one system overlap ("use for" of `jira_changes` and
`jira_issues` both answer "status of an issue"), so a table-level route splits one question over
branches that cannot join. The group is the table-name prefix (generic for new sources). The table
is chosen inside `find`, and `find` may restrict rows through a table of another group (link).
Measured over all 677 smoke research calls of the development runs: every family got the same
groups in every run, one group for single-system questions (jira 11 families, github 4, mail 1) and
github + jira for the three cross-system ones (issue_prs, issue_changed_files,
resolution_vs_merge_order); no question was routed to an unrelated group.

### `find`: Jev slot filling -> typed plan -> compiled SQL (`translate.py`, `plan.py`, `rawtools.py`)

- **Call 1** (state: need, values): table (choice over the branch's tables, with the catalog's
  about/use-for text), shape (lookup / list / count / timeline / search), link (none or a table the
  catalog joins to one of the branch's tables).
- **Call 2** (state: need, values, the chosen table and its columns): per value its column (choice
  over the table's and the linked table's columns, or "none"; a date binds only to time columns)
  and for a date the comparison (after / before / on); per column "needed in the output" (noul);
  group column (count) or order column + direction (lists). Jev never writes text: every slot is a
  catalog identifier or a fixed option; the literals are the agent's own values.
- If a value stays unbound, the runner-up table of call 1 is tried (one more call 2) and the plan
  that uses more of the given values wins.
- The **plan** (`Plan`: table, shape, filters[table, column, op, value], columns, group, order,
  desc, link, limit) is validated (identifiers from the catalog only, operator vs column type,
  literal parses as the column type, a link needs a join edge and a filter, non-count plans need a
  filter) and compiled to ONE SELECT: equality or `has()` for arrays, case-insensitive
  `positionCaseInsensitiveUTF8` for text search, plain UTC string literals for time ranges
  (period end computed in Python: no conversion function sees the host zone), every identifier and
  alias backticked, every literal typed and quoted (never query parameters; control characters
  become spaces, quotes and backslashes are escaped, statements over 20,000 characters are
  refused before execution). A column and its sibling variant in the same table (`x` and
  `<prefix>_x`, same type, e.g. subject-derived `kips` and body-derived `body_kips`) are offered
  to Jev as one extra option "either of", compiled to an OR. Link: `IN (SELECT …)`
  or `hasAny(arr, (SELECT groupUniqArray…))`, scalar or array on either side. Every function the
  compiler (and the value probe) can emit is listed in `plan.FUNCTIONS`; a test compiles every
  shape and column type and runs `guard.check` + the real smoke export on it.
- Join edges come from the catalog's prose join lines (parsed) plus columns of different tables
  whose example values share one identifier shape (`ABC-9`), e.g. `mail_messages.issues` <->
  `jira_issues.issue_key`.
- **Execution**: exact text equality first; 0 rows -> relaxed to case-insensitive containment;
  still 0 -> a probe counts where each value occurs (every text column of every table) and, when
  the value is found in a descriptive column, a lookup there is returned at once ("located": e.g.
  a title that only a change record still carries, for an issue created before the window).
- **Results**: rows with event ids, times printed as ISO `…Z`, values shared by every row printed
  once, long texts cut to 300 characters (around the searched phrase), order stated ("ascending,
  the last row is the latest"). Lists and searches with more than 25 rows: Jev scores up to 75 rows
  (batches of 25 nouls) and the top 25 are shown, plus value counts over ALL fetched rows for short
  repeated columns. Timelines and counts are never filtered (the latest rows are kept if the
  7,000-character budget is hit).
- **Uncertainty**: a table / column / used-link slot below confidence 0.6 -> the agent is told the
  slot and alternatives and asked to restate once (the next call runs regardless).
- Agent-side robustness: values quoted in the need are added, `column: value` / `name>=date`
  prefixes are stripped, identical repeated calls are refused, leaked harmony tool calls in the
  text are parsed, malformed JSON and unknown tools come back as tool errors.

**SQL fallback** (`sql_fallback`): `auto` = the `sql` tool appears after a failed find (invalid
plan, error, 0 rows not located, or an uncertain slot) together with the branch's schema and the
store's `db.sql_reference()` (the guard's allowlist and time pitfalls); `always` = offered from the start (schema in its description);
`never`. Every call is logged; guard refusals are recorded per call (`stats.rejected`); time
literals `'…T…Z'` in model SQL are normalised. `read(event_id)` pages the full record (6,000
characters per page).

**Memory branch** (`memory.py`, optional): the markdown tree is loaded once (only files whose
resolved path stays inside the root; symlinks out are ignored), `read_file` accepts only those
keys (no path from the model reaches the file system); `search` greps the keywords, ranks files
by distinct keywords matched, Jev picks the 10 hits worth reading out of up to 40.

The branch system prompt lists the group's tables (catalog "about"), the catalog's per-table
`notes` (window effects counted from the export) and its `people` note.

### Budgets, errors, concurrency

Per research call: 120 s (branches stop 20 s before, for judge + report), 8 turns per branch, 60
Jev calls, 250k chat tokens; every model call is also cut at the deadline (plus 10 s grace), and
each HTTP request has a total timeout (httpx bounds only single reads). On any budget the branch
gives a partial finding (one tool-less "state what you found" call, else its last tool results)
and the result is `partial: true`. `InfraError` = 401/402/403 at once, 429/5xx/timeouts/unreachable
after 3 retries with backoff (2, 4, 8 s), or no key: it propagates out of `research` so the runner
can pause. `ProviderError` = the provider rejected the request (a bug), or the local endpoint
truncated the prompt (`prompt_tokens >= 32768`). Branches run as asyncio tasks; model calls pass a
semaphore (chat 3, Jev 4); all database access goes through one worker thread (the shared handle:
chdb binds one embedded server per process). Prompts are trimmed below ~28k estimated tokens by
eliding the oldest tool results.

### Models and endpoints (`config.py`)

`EngineConfig` holds every threshold, budget and model setting; `config_hash(config)` = sha256 of
the behaviour settings + every prompt text (transport fields excluded), recorded in each trace as
`identity.config_sha256`. Endpoint presets `ENDPOINTS`: **`ollama` (default)** =
`http://localhost:11434/v1`, `gpt-oss-20b-ctx32k`, `reasoning_effort: low`, temperature 0, seed 7,
max_tokens 2048 always, no `response_format` (it hides reasoning tokens from usage), context
overflow check; `openrouter` = `openai/gpt-oss-20b`, `reasoning: {effort: low}`,
`usage: {include: true}`. Chat cost = the provider's reported cost, else tokens at list price
($0.018 / $0.09 per M, i.e. API-equivalent for the local model); Jev cost as reported
(~$0.042 per M input). Calls record whether reasoning text was present.

### Prompts

All in `engine/prompts.py` (routing, slot questions and option texts, row/hit/finding/support
questions, branch system prompts, sql tool, finish, report). They name no project, family or
benchmark. Tuned only on the smoke benchmark (19 historical tasks, checkpoint 2023-02) and
invented smoke questions; w2026 was used for one pre-registered 6-task sanity check, with no change
made after it.

### Trace (JSON)

`question, identity{config_sha256, checkpoint, boundary, models, endpoint, sql_fallback},
routing{probabilities, chosen}, branches[{source, kind, status, finding, evidence,
unverified_evidence, turns, malformed_calls, error, sql_unlocked, calls[{tool, need, values, plan,
slots{table, shape, link, uncertain, unused_values, switched, confidence}, sql, rows, total,
outcome (rows|located|empty|invalid|error|restate|repeat), relaxed, probe, relevance, rejected,
…}]}], aggregation{relevance, support, kept, flagged}, stats{find, find_rows, find_empty,
find_invalid, find_restate, sql, sql_ok, read, search, rejected[]}, answer, evidence, partial,
usage{jev{calls, input_tokens, output_tokens, cost_usd, cost_reported, seconds},
chat{…, with_reasoning}, cost_usd}, calls[{kind, model, purpose, input_tokens, output_tokens,
cost_usd, cost_reported, seconds, reasoning}], seconds`.

### Entry point for stage 3

```python
from harnext_eval.e2.v2.engine.config import EngineConfig, ENDPOINTS, for_endpoint, config_hash
from harnext_eval.e2.v2.engine.engine import Engine
from harnext_eval.e2.v2.engine.clients import InfraError   # -> mcp_server's Infra (pause)

engine = Engine(store: Path | None, memory: Path | None = None, config: EngineConfig | None = None,
                *, scratch: Path | None = None)   # opens the export once (one per process)
result = await engine.research(question: str)
# {"answer": str, "evidence": [event ids / file paths], "partial": bool,
#  "usage": {"jev": {...}, "chat": {...}, "cost_usd", "input_tokens", "output_tokens"},
#  "seconds": float, "turns": int, "trace": {...}}
engine.identity  # pin per arm (config_sha256, checkpoint, models, endpoint, sql_fallback)
await engine.aclose()
```

Open one export per process and keep it open (chdb leaks handles per open/close), pass a private
`scratch=` directory. Store v3 runs every session in UTC; `ask.py` and the dev script still set
`TZ=UTC` before chdb is imported (harmless, kept for older exports). CLI: `python -m harnext_eval.e2.v2.engine.ask --store <export>
[--memory <dir>] --question "…" [--endpoint ollama|openrouter] [--sql-fallback auto|always|never]
[--config overrides.json] [--trace out.json]`.

### Measured (smoke benchmark, final code, RAW store build version 3)

Store `stores-smoke/RAW/export/2023-02` (build version 3, content sha256 `6b8805b5…`), 19
historical tasks at checkpoint 2023-02, engine run directly (no outer agent), question = task
prompt without its reply-format tail, answer extracted by one extra gpt-oss call under the
runner's reply contract, graded by `grade_historical`. Two replicate runs per configuration
(rounds r8, r9; identical code, config and store). Cost is per research call (extraction excluded:
~$0.00003); local chat cost is API-equivalent at list price. Results:
`out/kafka-500-v2/engine-dev/smoke-<endpoint>-<mode>-r8|r9/` (results.jsonl, summary.json,
traces/).

| endpoint | fallback | correct r8 / r9 (of 19) | mean acc | mean $/question (Jev / chat) | mean s (p95) | turns | chat tokens in / out | find / sql (ok) | plan share | partial |
|---|---|---|---|---|---|---|---|---|---|---|
| ollama (local) | auto | 15 / 14 | 76.3% | $0.00113 (0.00084 / 0.00029) | 14.3 (29) | 4.2 | 12928 / 599 | 91 / 15 (15) | 0.86 | 3 |
| ollama (local) | always | 16 / 18 | 89.5% | $0.00142 (0.00096 / 0.00047) | 17.3 (35) | 4.7 | 22070 / 761 | 106 / 2 (1) | 0.98 | 6 |
| ollama (local) | never | 15 / 16 | 81.6% | $0.00134 (0.00106 / 0.00029) | 14.1 (27) | 4.6 | 12816 / 649 | 115 / 0 (0) | 1.00 | 1 |
| openrouter | auto | 15 / 17 | 84.2% | $0.00129 (0.00091 / 0.00038) | 26.1 (50) | 5.1 | 15928 / 753 | 95 / 14 (12) | 0.87 | 2 |
| openrouter | always | 18 / 17 | 92.1% | $0.00182 (0.00105 / 0.00077) | 44.0 (78) | 7.2 | 34217 / 1137 | 109 / 10 (9) | 0.92 | 13 |
| openrouter | never | 17 / 17 | 89.5% | $0.00148 (0.00107 / 0.00040) | 26.7 (49) | 5.6 | 16794 / 820 | 108 / 0 (0) | 1.00 | 3 |

- Jev: 1,927 decisions calls in these runs, latency p50 0.30 s, p95 0.46 s, max 3.6 s; median
  ~2,900 tokens per call (input; output is free); per research call median 6 calls (p10 4, p90
  19, max 31). Jev is the larger cost: median 71 % of a research call (p10-p90 52-82 %). One Jev call hung for 43 s early in development: the 12 s per-request timeout + retry
  handles that.
- Guard refusals in the final runs: 1 (openrouter/always: model SQL on an invented table
  `github_pr_events`, refused with a "did you mean" hint). No compiled plan was ever refused.
- Variance: the same configuration moves by up to 4 tasks of 19 between runs (local auto 18 in an
  earlier round on the previous store, 15 / 14 here); single-run differences of 1-2 tasks between
  modes are not meaningful. With two replicates, `always` was best on both endpoints and `auto`
  was not better than `never`: once unlocked, the fallback SQL written by the 20b model helps less
  than a restated `find`. Local Ollama is ~2x faster than OpenRouter's gpt-oss-20b providers.
- Development history (not comparable: code, prompts and store changed between rounds): OpenRouter
  auto 13/19 (r1), 16/19 (r2), 18/19 (r3, always 16, never 14; host time zone, store v2); r5
  (host TZ) and r6 (`TZ=UTC`) at 15-18/19 per configuration; r7 stopped (store swap).
- w2026 sanity check, pre-registered by sorted id before any gold was seen (config `03af6d2e` =
  round-6 openrouter/auto, store v3, raw sources only): 4/6 correct (kip_thread, pr_approvers,
  reopen_count, design_decision-074f443eea correct; implicit_link-2816fc05c3 and
  design_decision-1bf0682184 wrong). Nothing was changed in response. Later changes (the
  "factual" drop, siblings, catalog notes, literal hygiene) were motivated by smoke results and
  the store changes only; w2026 was not re-run.

### Failure analysis

33 misses in the 228 final smoke answers, by failing step:

| family | misses | step | what happens |
|---|---|---|---|
| issue_prs | 11/12 | benchmark semantics | the agent lists every PR whose `issues` holds the key, incl. 4 PRs that have only a `merged` row (opened before the window); the gold lists only the 2 PRs with an `opened` row, while the v3 catalog note says "do not require action = 'opened'". Flagged, not tuned. |
| top_reviewer_for_author | 7/12 | plan / agent | no "not equal" operator: `reviewer != author` is bound as `reviewer = author`; the author's login bound to `actor` of merged rows (= the merger). The SQL fallback path solves it when taken. |
| resolution_vs_merge_order | 5/12 | report / agent | one branch has the resolution time, the other the merge time; the agent computes days itself (7 instead of 0) or the report does not combine them. |
| fix_version_at_snapshot | 3/12 (local) | agent | two `fixVersion` add rows; the model reports only the latest instead of the set. |
| resolution_transition_history | 2/12 | agent | the finding summarises the changes without the microsecond timestamps the rows had. |
| issue_links, status_*, author_activity | 5/12 | plan / agent | link direction taken from the wrong side (`linked_issue` instead of `issue_key`), a creation row added to a history, a count off by one. |

w2026 misses: implicit_link: the jira branch never found the tracking issue (agent search
strategy); design_decision-1bf0…: the mail branch found the gold value, but the support check
(which sees only the last 24,000 characters of the branch's tool results) scored it 0.23 and it
was dropped; the report writer then invented a property name with a fake citation.

### Known limits

- Plans have no negation, arithmetic, or per-group top-k; those needs go to the fallback SQL
  (`auto`/`always`) or fail (`never`). Multi-step questions rely on the agent chaining finds.
- The 20b model often puts the final answer format into its finding (bare JSON, a computed
  number) instead of facts; the report and the benchmark extractor then work from less context.
- The support check is lossy (last 24k characters) and can drop a correct finding; the report
  writer can hallucinate when the kept findings say "not found".
- Time: 14-17 s mean locally (p95 ~30 s), 26-44 s on OpenRouter, max 97 s; the 120 s deadline
  never ended a final-run call; all 21 unfinished branches stopped at the 8-turn budget.
- One export per process (chdb); `research` calls on one Engine share one DB worker thread.
- Prompts were tuned on 19 smoke tasks at one checkpoint; there are no design_decision or
  implicit_link tasks in smoke, only invented questions (the memory branch hallucinated a decision
  in one of them, which motivated the support check).

### Requests to the raw store (none blocking)

1. Machine-readable joins in the catalog (`[{column, table, other_column, kind}]`); the engine
   parses the prose join lines and example shapes today.
2. The issue_prs gold vs the `github_pull_requests` note above: decide whether "opened by the
   snapshot" includes PRs opened before the window (then the gold is incomplete) or not (then the
   note should say so).
3. `jira_changes.key_columns` could include `actor` now that `author` is NULL on link rows.

## Stage 3: benchmark integration

Written 2026-09-28. No file of stages 1–2 was changed. Changed (additively): `v2/mcp_server.py` (kind
`engine`), `v2/runner.py`, `v2/agents/{base,container,codex,fake}.py`, `benchmarks/kafka-500-v2/arms.yaml`,
`docker/e2-agent/{Dockerfile,constraints.txt,allowlist.filter,verify-egress.sh}`, `scripts/e2_report.py`,
`scripts/e2_charts.py`. Added: `configs/e2-engine-smoke-offline{,-pause}.yaml`,
`configs/e2-engine-smoke-live{,-2}.yaml`, `tests/test_e2_v2/test_engine_arm.py`.

### What is wired

- **MCP kind `engine`**: one tool `context_research(question)`, the same function, name and docstring
  as kind `research`, so the calling agent's interface is identical; `EngineBackend` runs
  `Engine.research` over the RAW export mounted at `/store` (+ `/memory` for notes). Budget 25 calls
  per trial. The export is opened once at server start (`--scratch /scratch`), after
  `query.harden()`. The agent sees `{answer, files (= evidence ids), seconds, turns, usage}`.
- **Receipts** (`research` field of each call row): usage per model (`jev`, `chat`: calls, tokens,
  cost, seconds), totals, seconds, turns, partial, answer bytes, `config_sha256`, the engine identity,
  and `trace: {file, line, sha256}` pointing into `mcp/engine-trace.jsonl` (the full trace per call;
  runner `ARCHIVED` → `mcp-engine-trace.jsonl[.gz]` in the trial archive, hashed in its manifest).
  `result.json` of an engine arm carries `engine` = the trial's totals (`mcp_server.engine_usage`).
  `e2_report.py` / `e2_charts.py` read the receipts' `research.usage.cost_usd` as read-agent cost, so
  engine cost counts there (Jev as reported by OpenRouter; chat as reported, else list price).
- **Infra**: `InfraError` from the engine → `Infra` → receipt `infra: true` → the runner pauses. An
  engine that cannot start (no key, local chat endpoint unreachable) writes an `infra` row at start.
  Any key text in an error is redacted.
- **Arms**: `H-engine` (RAW) and `H-engine-mem` (RAW + `memory_store: F5`), `engine: {preset:
  openrouter|ollama}`. Access paragraph and standing instructions are the kind-`research` texts
  verbatim (`ACCESS_EXTRA["mcp:engine"]`, `GATHER_FIRST_EXTRA["engine"]`), pinned per arm by
  `arm_pin`; config `engine_preset` overrides every engine arm's preset and enters the arm pin (not the
  manifest config). `memory_store` checkpoints are pinned and leased like `store`
  (`<store>/<ckpt>`). Pins and `prompt_template_sha256` of every existing arm are unchanged (test
  `test_existing_pins_and_prompt_template_do_not_move` holds the 21 values).
- **Container** (`container.py`): `/store` ro, `/memory` ro, `--tmpfs /scratch` 512 MB (≥ 2× the
  w2026 export; tmpfs counts against `--memory 10g`), non-root `--user`, `--pids-limit 4096`,
  `--cap-drop ALL`. MCP server env: proxy variables, `NO_PROXY=localhost,127.0.0.1,<gateway>`
  (`docker network inspect` of `harnext-e2-net`), `TZ=UTC`. No Claude login is mounted for this kind.
- **Key** (`OPENROUTER_API_KEY`, host env or repo `.env`): written per trial to
  `<trial scratch>/out/engine-key/key` (0600, never archived, removed with the scratch), mounted at
  `/run/harnext-engine`; the server reads it first thing at start and deletes the file; it never
  enters argv, the MCP config, an environment variable, receipts, traces or the image. What this does
  NOT give: the agent runs as the same user in the same container, so until the server has started
  (Claude Code starts MCP servers before its first model turn) the file is readable; a sibling
  process can read the server's `/proc/<pid>/environ` and `cmdline` (no key there) but not its memory
  (EACCES: yama ptrace_scope 1, no CAP_SYS_PTRACE). A restarted server has no key → infra row → pause.
- **Egress**: `openrouter.ai` is added to the proxy allowlist. The allowlist is ONE list for every
  trial container, so every arm's agent can now reach openrouter.ai (without a key; CONNECT filtering
  cannot restrict paths). `verify-egress.sh` checks it is allowed and that look-alike hosts are not.
- **Image**: `chdb==4.4.0`, `chdb-core==26.7.3`, `pandas==3.0.5`, `pyarrow==25.0.1`,
  `python-dateutil==2.9.0.post0`, `six==1.17.0`, `tzdata==2026.2` pinned; 1.98 → 2.75 GB. The
  package set is otherwise identical to the previous image (pip freeze diff = the six new packages).

### Runbook

```
# 0. stores (RAW + the F5 notes store, which already exists as a git store with ckpt tags)
.venv/bin/python -m harnext_eval.e2.v2.stores.build --arm RAW --scope smoke
.venv/bin/python -m harnext_eval.e2.v2.stores.build --arm RAW --scope w2026

# 1. image (repo root; no run may be active: systemctl --user list-units 'harnext*'; docker ps)
free -g; df -h /                                   # >= 6 GB available, >= 12 GB disk
docker tag harnext-e2-agent:latest harnext-e2-agent:pre-<change>-<date>
docker build -f apps/eval/docker/e2-agent/Dockerfile -t harnext-e2-agent .
docker run --rm --network none harnext-e2-agent /opt/venv/bin/pip freeze   # compare with constraints.txt

# 2. proxy (only when allowlist.filter changed)
docker build -f apps/eval/docker/e2-agent/Dockerfile.proxy -t harnext-e2-proxy apps/eval/docker/e2-agent
docker rm -f harnext-e2-proxy && bash apps/eval/docker/e2-agent/verify-egress.sh   # recreates it

# 3. offline smoke (no network; pause, then resume with the plain config; same run_id)
.venv/bin/python -m harnext_eval.e2.v2.runner --config apps/eval/configs/e2-engine-smoke-offline-pause.yaml --live --no-wait   # exit 75
.venv/bin/python -m harnext_eval.e2.v2.runner --config apps/eval/configs/e2-engine-smoke-offline.yaml --live --no-wait         # exit 0

# 4. live smoke (smoke benchmark only; OpenRouter + the Claude Code login)
.venv/bin/python -m harnext_eval.e2.v2.runner --config apps/eval/configs/e2-engine-smoke-live-2.yaml --live --no-wait
.venv/bin/python apps/eval/scripts/e2_report.py --config apps/eval/configs/e2-engine-smoke-live-2.yaml
.venv/bin/python apps/eval/scripts/e2_charts.py --config apps/eval/configs/e2-engine-smoke-live-2.yaml
```

**Full w2026 experiment (not run).** Add `H-engine, H-engine-mem` to `arms:` of
`configs/e2-w2026-claude-haiku.yaml` (run `w2026-claude-code-haiku-4-5`: arms join a run as their
stores are ready, each is pinned on first use, recorded trials are untouched, and the tasks, agent and
S-tools baseline are the same run's). Needs `stores-w2026/RAW/export/2026-0{1..5}` and the F5 tags
(present; F5's pins are already in that run's manifest and must still match). Keep the workers
modest (each engine trial runs its own export + up to 3 chat / 4 Jev requests in flight):
```
.venv/bin/python -m harnext_eval.e2.v2.runner --config apps/eval/configs/e2-w2026-claude-haiku.yaml --check-resume
systemctl --user start harnext-w2026-run-claude-haiku.service     # = runner --live with that config
```
Cost from the smoke (per historical trial): engine $0.0005–0.0024 (Jev ≈ 45 %), agent $0.012–0.056.

**Switching the chat model to local Ollama** (once the host service listens on the docker gateway,
E2-ENGINE-OLLAMA.md "Service state and host setup"): check from the host `curl
http://$(docker network inspect -f '{{(index .IPAM.Config 0).Gateway}}' harnext-e2-net):11434/v1/models`
(the runner's preflight does the same, `container.local_model_unreachable`). Then either set
`engine_preset: ollama` in the run config or give the arms `engine: {preset: ollama}` — both change the
arms' pins, so do it for a NEW run id or as new arm ids (e.g. `H-engine-local`), never mid-run.
The MCP server gets `--chat-base-url http://<gateway>:11434/v1`; the gateway is in `NO_PROXY`; Ollama
is never added to the proxy allowlist. With 4 Ollama slots, run at most 1–2 engine trials at a time.

### Smoke results (2026-09-28)

Offline (`engine-smoke-offline`, fake agent + fake models, N / S-tools / H-engine / H-engine-mem × 1
coding + 4 historical): paused at the injected provider failure with 2 trials recorded and nothing for
the failing one, resumed to 20/20, `--check-resume` = 0 remaining, every archive and receipts chain
verified, every trace reference matches.

Live (Claude Code Haiku 4.5, 4 historical tasks, OpenRouter): the smoke RAW export was rebuilt at
11:14 while `engine-smoke-live` ran; the runner refused the changed pin (no mixed trials). Its engine
trials (old export, all finished before 11:14) are kept as a record; the engine arms were rerun on the
current export and code as `engine-smoke-live-2`; S-tools (does not read RAW) from `engine-smoke-live`.

| arm | correct | mean $/trial (agent + engine) | engine $/trial | median s |
|---|---|---|---|---|
| S-tools | 4/4 | 0.0316 | – | 16.5 |
| H-engine | 4/4 | 0.0248 | 0.0008 | 42 |
| H-engine-mem | 3/4 | 0.0251 | 0.0017 | 67 |

One `context_research` call per engine trial. The one coding trial (S-tools, C-KAFKA-PR-13453) resolved
in 1,606 s for $0.56; the engine coding trials were dropped for memory (the coding container peaked
at 9.7 of 10 GB with swap full). Engine trial containers stayed below 0.4 GB (10 s samples).

### Known limits

- Four historical tasks are a plumbing check, not a measurement; H-engine-mem's miss is one trial.
- The start receipt's `store.export`/`memory` are the mount names (`store`, `memory`) inside a
  container; the checkpoint is in each call's `identity`.
- The proxy allowlist is global (above); the key is protected only by deletion-at-start (above).
- A restarted MCP server cannot re-read the deleted key: it pauses the run instead of answering.
- The image holds a snapshot of `engine/` and `stores/raw/`: rebuild it whenever they change, and
  rebuild the RAW exports before a run pins them (a rebuilt export stops a run at its pin check).
- `e2_report.py` still reads store build costs from `stores-w2026` and says "January–May 2026" in its
  header for every config.

### Review fixes (2026-09-28, after the independent review)

Additive; existing arms' pins, texts and the single-run report output are unchanged (checked: all 21
pins, `prompt_template_sha256`, `e2_report.py` for the main run byte-identical).

- **Code and image pinned to a run (engine arms only).** `v2/engine_pin.py` hashes the engine package
  (`engine.config.code_hash`), `stores/raw/` (same scheme), `mcp_server.py` and the engine config per
  chat preset. The runner pipes that script into `python -` INSIDE the image that will run the trials
  (so any image is measured the same way) and, on first use, stores in `manifest.json → engine`: image
  ID, the three code hashes, config hashes, the allowlist read from the running proxy. First use refuses
  an image whose code differs from the working tree ("rebuild the image first"); a resume refuses a
  different code or config hash, and a different image ID unless `--allow-image-change` (the change is
  appended to `engine.image_overrides` with time and argv). Live runs start every container from the
  image ID resolved at preflight (not the tag); `launch.json`, `result.json` and the session file carry
  `image_id`. Quote it all with
  `python -m harnext_eval.e2.v2.runner --config <yaml> --identity` (image ID, in-image and tree hashes,
  whether they match, engine arms, every store/checkpoint hash of the plan, the recorded pin).
- **Failures never scored as answers.** A research call is infrastructure (receipt `infra: true`, the
  runner pauses) when (a) it ended partial after the time budget AND model calls were retried, failed
  or cancelled (`mcp_server.engine_trouble`; a partial answer from the turn/token budgets alone stays
  an answer), or a branch failed with an error other than a budget; (c) the engine raised anything
  unexpected (the receipt keeps the traceback, key redacted). After each engine trial the runner reads
  the agent's trace (`runner.mcp_trouble`: Claude Code stream-json and Codex items): (b) the MCP server
  not `connected` in the init line, the call count in the trace ≠ the call rows in the receipts (one
  extra allowed only when the agent was killed at its time/turn limit), a "connection closed" result,
  and (d) an agent-side timeout on `context_research` all pause the run; nothing is recorded.
- **Key**: removed 60 s after the container starts even if the server never ran
  (`container.ENGINE_KEY_GRACE_S`) and by the runner as soon as the agent returns, on every path.
  The repo-root `.env` is mode 0664 today: make it 0600 (`chmod 600 .env`).
- **Reports.** `e2_report.py` / `e2_charts.py --extra-config <later run yaml>` show another run's arms
  beside the main run; an arm id present in both (the S-tools drift control) is labelled with its run
  id, and a "Runs combined" table gives per run: arms, trial dates, image ID and proxy allowlist, plus
  the sentence that the engine arms ran on a later date, a different image and a proxy that also
  allows openrouter.ai. Costs that would print as $0 / $0.00 get two significant digits ($0.0010);
  when an engine arm is shown, Table 4 adds median and p10–p90 cost per trial and the engine's
  unmetered upper bound, and says how engine cost is measured. Single-run output is unchanged.
- **Charts deterministic.** The coding matrix sorted tasks by `run_index` only; in the main run 30
  `run_index` values are shared by several tasks (tasks appended to the run), so the order depended on
  the hash seed. Task id is now the tie-breaker, in the shared code path: the published page was one
  random draw of that order, so no output was reproducible to preserve; only tied rows move.

### Full w2026 experiment: plan (not run)

Config `configs/e2-w2026-engine-claude-haiku.yaml`, NEW run id `w2026-engine-claude-code-haiku-4-5`
(the main run stays untouched). Same benchmark, caps (12 coding, 78 historical), agent, model, turn cap
and stores as the main run. Arms H-engine, H-engine-mem on every task (2 × 90 = 180 trials) and
S-tools as drift control on a fixed subset: `every_nth: {S-tools: 3}` = positions 0, 3, 6 … of each
category in run order (4 coding, 12 factual, 10 design decisions, 5 implicit links = 31 trials). Total
211 trials (`--dry-run` confirms). Workers: `workers: 3`, `coding_workers: 1` — one coding container
(10 GiB limit; peaked at 9.7 GiB today) plus two historical containers (engine ones stayed < 0.4 GiB):
≈ 11 GiB peak against ≈ 17 GiB normally available; engine coding trials (24) run one at a time.
Check `free -g` ≥ 12 GiB before starting; if the machine is busy use `workers: 2`.

Expected, from the main run (F5-search as the proxy for an MCP arm; means per trial: coding $0.48 /
620 s median + 200 s grading, design $0.20 / 167 s, factual $0.065 / 63 s, implicit $0.017 / 67 s):
- time: coding 28 trials × ~14 min (p90 ~21 min) ≈ 6.5–10 h sequential; the 183 historical trials
  (~2–3 min each incl. engine calls, 2 at a time) ≈ 3–4 h alongside → **≈ 7–10 h wall**.
- Claude subscription (API-equivalent): engine arms ≈ $14 each, drift control ≈ $4 → **≈ $32**.
- OpenRouter: ~1.3 `context_research` calls per trial × $0.001–0.004 per call (smoke: $0.0005–0.0024;
  w2026 exports and catalogs are larger) → **≈ $0.3–1** for 180 engine trials; keep ≥ $3 credit.

Steps once the engine code is final:
```
free -g; df -h /                                 # >= 12 GiB available, >= 12 GB disk
docker tag harnext-e2-agent:latest harnext-e2-agent:pre-engine-final-<date>
docker build -f apps/eval/docker/e2-agent/Dockerfile -t harnext-e2-agent .
.venv/bin/python -m harnext_eval.e2.v2.runner --config apps/eval/configs/e2-w2026-engine-claude-haiku.yaml --identity   # tree_matches_image: true
.venv/bin/python -m harnext_eval.e2.v2.runner --config apps/eval/configs/e2-w2026-engine-claude-haiku.yaml --dry-run    # 211 trials
nohup .venv/bin/python -m harnext_eval.e2.v2.runner --config apps/eval/configs/e2-w2026-engine-claude-haiku.yaml --live \
  > apps/eval/out/kafka-500-v2/runs/w2026-engine-claude-code-haiku-4-5.log 2>&1 &
# report beside the main run:
.venv/bin/python apps/eval/scripts/e2_report.py --config apps/eval/configs/e2-w2026-claude-haiku.yaml \
  --extra-config apps/eval/configs/e2-w2026-engine-claude-haiku.yaml --out apps/eval/reports/e2/w2026-with-engine/report.html
.venv/bin/python apps/eval/scripts/e2_charts.py --config apps/eval/configs/e2-w2026-claude-haiku.yaml \
  --extra-config apps/eval/configs/e2-w2026-engine-claude-haiku.yaml --out apps/eval/reports/e2/w2026-with-engine/charts.html
```

**OpenRouter without credit.** Jev answers 402 → InfraError → the trial is not recorded, the run
pauses and, without `--no-wait`, sleeps 15 minutes (`errors.reset_delay` default) and retries — for
ever, one agent trial per attempt burning Claude usage until the engine is called. Stop it with
`kill -TERM <runner pid>` (or `systemctl --user stop <unit>` if started as a unit; `kill -USR1` lets the
trials in flight finish first); nothing unrecorded is lost. Check credit first:
`curl -s https://openrouter.ai/api/v1/credits -H "Authorization: Bearer $OPENROUTER_API_KEY"`, or start
with `--no-wait` so a pause exits with code 75 instead of waiting.

**Local Ollama preset.** Besides binding Ollama beyond 127.0.0.1 (E2-ENGINE-OLLAMA.md, `sudo
systemctl edit ollama` with `OLLAMA_HOST=0.0.0.0:11434`), the host firewall must admit the trial
network: the review found every gateway port closed from `harnext-e2-net`. For the user (root):
```
SUBNET=$(docker network inspect -f '{{(index .IPAM.Config 0).Subnet}}' harnext-e2-net)   # 172.19.0.0/16
sudo ufw allow in from "$SUBNET" to any port 11434 proto tcp comment 'harnext trials -> ollama'
sudo ufw deny in to any port 11434 proto tcp comment 'ollama: nobody else'
sudo ufw status numbered | grep 11434     # the allow rule must come before the deny
# one-line reachability test from a trial-like container (no proxy, as the MCP server connects):
docker run --rm --network harnext-e2-net harnext-e2-agent curl -sS --noproxy '*' -m 5 http://$(docker network inspect -f '{{(index .IPAM.Config 0).Gateway}}' harnext-e2-net):11434/api/version
```
Then use a new run id or new arm ids with `engine: {preset: ollama}` (see above); the runner's preflight
refuses to start if the host cannot reach `<gateway>:11434`.

### Known limits (review)

- The outer agent's cost varies up to 4× with its prompt-cache state (42k vs 63k cached input tokens
  in two smoke trials of the same task), which swamps arm differences over a handful of trials: cost
  comparisons need the full task set, and the report shows medians and p10–p90 beside the means.
- In one smoke trial the engine's answer was right but the outer agent derived another value from a
  supporting finding listed below it ("assignee removed …"): the report's structure matters — the
  direct answer must be unambiguous and the supporting facts must not contradict it (request to the
  engine author, below).
- Engine gap: a branch cancelled at the research deadline (the orchestrator cancels its task) does not
  meter its in-flight model call as `cancelled`; only calls cut by `Session.bounded` are. A slow
  provider without retries can therefore end a call partial without any infra signal and it is scored
  as an answer (request below).

### Requests to the engine author (stage 3 review)

1. Report structure: put the direct answer first and alone; supporting facts must not state a value
   that contradicts it (e.g. an intermediate "assignee removed" row under a final assignee).
2. Meter in-flight calls of branches cancelled at the deadline (`CancelledError` in `Session.bounded`)
   as `cancelled` with their upper-bound cost, so time-outs caused by a hanging provider are visible.
3. Keep `identity.code_sha256`; stage 3 pins it together with a hash of `stores/raw/`.

### Fix round 1 (2026-09-28): grounding, plan language, determinism

Corrections to the numbers above (re-derived with `scripts/e2_engine_numbers.py` from the r8/r9
traces): Jev input tokens per decisions call, median **1,574** (the "~2,900" above included the
output tokens); Jev calls per research call, median 6; latency p50 0.30 s / p95 0.46 s over 1,927
calls. The r8/r9 traces do not store tool results, so no grounding audit can be computed for them.

#### Changelog (each change: what motivated it)

| # | change | motivation (smoke trace or general principle) |
|---|---|---|
| 1 | Report verification: citation tokens not found in the rows shown to a model are removed; checkable values (generic definition, `verify.py`) found neither in shown rows, findings, the question nor the instructions trigger one regeneration naming them, then the lines stating them are dropped; Jev judges the report against the findings and their rows (supported / unsupported / declined); on "unsupported" one rewrite, kept only if it then verifies. Drafts and actions are in `trace.verification`. | Principle: no invented values or citations reach the caller (coordinator item 1; reviewer M1). The keep-only-if-verified rule: fix1-ollama-always-d4 `H-author_activity`, `H-review_hotspot` (a correct report was rewritten into a wrong one). |
| 2 | Support of a finding from the rows it cites (by event id), deterministic first: all its values in cited rows → kept; ≥ 80 % of the values it adds to the question found in anything its branch showed → kept; else Jev on the cited rows. The report gets each finding with its cited rows (or the branch's last result). | fix1-ollama-always-d1 `H-resolution_vs_merge_order` (a correct resolution time dropped at 0.15 because the finding also named other retrieved rows); d3 `H-resolution_transition_history` (a vague finding over correct rows became "not found"). |
| 3 | Plan language: `!=`/NOT has, empty / set checks, numeric ranges, column comparisons (same row, or against the linked row via `(key, col) [NOT] IN (SELECT …)`), count distinct, `values` / `latest` / `earliest` shapes, AND/OR of several values for one column (Jev choice). Role columns separated by the catalog's column meanings in the binding state. | Coordinator item 3; smoke `H-top_reviewer_for_author` (exclusion of the author), `H-fix_version_at_snapshot` (both added values). |
| 4 | Time operators `before` (strictly before the start of the named period) / `through` (up to and including its end) / `from` / `after` / `on`, at the precision given (day, second, sub-second); agents pass instants as written. | Reviewer M3 (`before` a cutoff dropped the last day in 8 r8/r9 plans). |
| 5 | `compare_times(a, b, unit, rounding)` tool and a combine step across branches (Jev picks the two events, the unit and the rounding from the question's wording; SQL orders them and computes the difference). No built-in unit or rounding convention. | Coordinator items 4, 5; reviewer H2. d1/d3 `H-resolution_vs_merge_order` (model computed days itself, wrongly). |
| 6 | Truthful result headers: total from a count query, how many shown and which, order, and how to get the rest (`page` = rows strictly in order, 25 per page); `seen`/records only for rows actually displayed; timelines newest- or oldest-first by a Jev choice; values used and NOT used are named. | Reviewer M1, M4. |
| 7 | JSON-shaped findings unwrapped; model SQL results repeat their query; unit/rounding spellings normalised; tool crashes returned to the model as errors; a restate reply may be repeated to run as is; shape doubt does not block (stated in the header); empty/set and field-comparison slots need a clear signal (> 0.8). | d3 `H-author_activity` (JSON finding lost its evidence); d5 `H-author_activity`; d5 `H-resolution_vs_merge_order` ("unit must be one of"); d2: 37 of 71 plans got spurious empty/set filters, 10 of 16 restates were shape-only and 14 identical retries were refused. |
| 8 | `read_thread(event_id, need)`: all records of a thread or item in time order (grouping column = first key column that is a catalog join column), Jev-relevant ones in full, others one line with ids. | Reviewer design item 3; invented smoke question on a mail discussion (stage 2). |
| 9 | `cross_source = branches | single`: several routed raw sources handled by one agent over all their tables (compared below). Plans may link into any group. | Reviewer design item 2; coordinator item 5. |
| 10 | Provider failures (incl. 400/404/422 and truncated prompts) leave `research` as `InfraError`, with `.usage` of what was spent; 408 retried; 413 → prompt shrunk once, then infra. Every request metered: retries, cancelled calls and failed attempts recorded, with an upper bound for requests that returned no usage (`usage.unmetered_upper_usd`). Deadline applies to judge and report (grace 2 s). | Reviewer M5, M6. |
| 11 | Structured catalog links used first for joins; `value_shape` shown in column meanings. | Coordinator (catalog additions). |
| 12 | Defaults: `sql_fallback = always`, `chat_concurrency = 1` (no batching effects), temperature 0 and seed on both endpoints, deterministic tie-breaks in row scoring. | Coordinator items 6, 7. |
| 13 | Identity pins the code: `identity.code_sha256` (sha256 over the package's sources); each dev run stores `source.json` (per-file hashes). | Reviewer H4. |
| 14 | Neutrality: `test_engine_prompts.py` checks every model-facing text (prompts, Jev instructions and options, tool descriptions) for word 3-grams shared with the benchmark's templates and public prompts (w2026 + smoke) — 0 shared. One phrase ("ids or file paths", from the reply contract) was reworded. The value definition and all operation descriptions were rewritten generically (reviewer H1, H2). | Reviewer H1–H3. |
| 15 | An endpoint's failure to parse the model's own tool call (Ollama HTTP 500 "error parsing tool call") is a model error: not retried (the output is deterministic), the model is told once, then its raw text is taken as its reply. | fix1-final-ollama-r1 `H-resolution_vs_merge_order` (the run stopped with a false InfraError after 3 identical retries). |
| 16 | Report structure: the direct answer in one sentence naming the instant or condition it refers to; supporting facts in time order with timestamps; a fact that could be read as contradicting the answer is placed relative to the asked instant; no speculation. | Coordinator (benchmark live smoke: an outer agent read a supporting fact as the answer). |
| 17 | Citations: only identifiers shown to a model in this call stay inside brackets, one per bracket; the result's `evidence` starts with the ids the report cites. | Coordinator (garbled citations in archived traces). |
| 18 | Model calls in flight when a branch is cancelled at the deadline are metered as `cancelled` with an upper-bound cost; `verify_report` switch (ablation of the report checks). | Coordinator (a merely slow provider must leave a signal); the analysis below. |
| 19 | Citation filtering repeats until stable (an inner bracket removed could leave an outer non-citation bracket). | fix1-final4-openrouter-r1 `H-reopen_count` ("[0 rows for ]" left in the answer). |

No change in this round was motivated by a w2026 task. Two of the reviewer's hygiene points
concerned value kinds I had enumerated after reading the w2026 sanity failures (command-line
flags, dotted identifiers); that enumeration was replaced by the generic definition above.

#### Results (smoke, 19 historical tasks, RAW store build 3, content sha256 `7404163d…`)

Numbers from `scripts/e2_engine_numbers.py` (`engine-dev/fix1-final4-*-numbers.json`,
`fix1-final5-numbers.json`). Evaluated code `3c61d4f0…` (runs r1–r3, single, no-verify,
OpenRouter); final code `1a21f39f…` differs only by change 19 and was confirmed by one run.
Default configuration = local Ollama `gpt-oss-20b-ctx32k`, `sql_fallback = always`,
`cross_source = branches`, config `2add4647…` (OpenRouter preset `101557f6…`).

| run | correct | $/question (Jev / chat) | p50 / p95 s | find / model SQL | plan share | verification changed / regenerated | "not found" | ungrounded answers |
|---|---|---|---|---|---|---|---|---|
| local default r1 | 15/19 | 0.00114 (0.00067 / 0.00047) | 16.4 / 56.6 | 49 / 1 | 0.98 | 9 / 7 | 1 | 0 |
| local default r2 | 16/19 | 0.00121 (0.00068 / 0.00053) | 17.4 / 64.2 | 54 / 4 | 0.93 | 9 / 7 | 0 | 0 |
| local default r3 | 16/19 | 0.00124 (0.00071 / 0.00053) | 16.9 / 63.1 | 54 / 3 | 0.95 | 8 / 6 | 0 | 0 |
| local default, final code | 17/19 | 0.00120 (0.00067 / 0.00054) | 15.2 / 53.9 | 54 / 7 | 0.89 | 7 / 4 | 0 | 0 |
| local, one agent over sources | 15/19 | 0.00090 (0.00050 / 0.00040) | 14.6 / 31.1 | 39 / 1 | 0.98 | 8 / 6 | 0 | 0 |
| local, report checks off | 17/19 | 0.00108 (0.00060 / 0.00048) | 15.6 / 50.6 | 54 / 1 | 0.98 | 1 / 0 | 0 | 3 |
| OpenRouter default | 16/19 | 0.00135 (0.00069 / 0.00066) | 34.8 / 72.4 | 51 / 0 | 1.00 | 9 / 6 | 0 | 2 |

- Spread of the default configuration over 4 runs: 15–17 of 19. Jev: median 6 decisions calls
  per research call, **1,459** input tokens per call (median), latency p50 0.31 s / p95 0.48 s.
  Guard refusals: 0. Unmetered cost upper bound: 0 (no failed or cancelled request).
- "Ungrounded answers" = answers stating a checkable value or citation not found in the rows
  shown to a model or in the findings (audit in the numbers script). With the report checks on:
  0 in every run. Without them: 3 (e.g. "#13065" where the rows show `pr-13065-merged`, and
  "fvaleri-authored").
- Single vs several sources: every configuration is 14–16 of the 17 single-source tasks; the 2
  multi-source smoke tasks score 0–1 of 2. One agent over both sources (`single`) did not help
  (1/2, 15/19 total) and is not the default; it is cheaper (fewer turns).

#### Why lower than the pre-fix 89.5 % (r8/r9 local always: 16, 18)

- **Run-to-run noise**: the same code scores 15, 16, 16, 17 (and the pre-fix configuration
  16, 18): about ±1–2 tasks. Temperature 0, a seed and one chat call at a time reduce it but do
  not remove it (Jev and the local server's batching are not bit-exact).
- **Neutral catalog text** (store v3 catalog, reworded; no code involved): for
  `status_at_snapshot` Jev's first table choice was `jira_changes` in 4 of 6 pre-fix runs (5/6
  correct) and is `jira_issues` in 3 of 3 final default runs (0/3): the removed "use for: state of
  a field at a time (latest change before it)" line of `jira_changes` was what steered it; the
  agent then reads the creation row's status and stops. Not compensated by any hint.
- **Grounding / verification steps**: with the report checks off, the same code scored 17/19.
  In the three default runs the checks changed 26 of 57 answers (20 regenerations): 14 of the
  regenerated answers ended correct, 6 wrong. They cost `author_activity` in 3 of 3 runs (the
  finding "three pull requests" is right, but its cited rows come from a query whose WHERE held
  the author, not a column showing it, and the Jev verdict / rewrite turned it into "not
  determined") and `assignment_history` once (a verified rewrite dropped four of five changes).
  They removed every ungrounded value and citation (3 in the unchecked run).
- **Plan-language changes**: no failure traced to them in the final runs; `top_reviewer_for_author`
  (exclusion of the author) and `fix_version_at_snapshot` (values shape) now pass in all final
  runs; `issue_prs` still fails (below).

#### Failing tasks (final runs) and the failing step

| task | failing runs (default r1–r3, confirm) | step |
|---|---|---|
| status_at_snapshot | 3 of 4 | table choice (neutral catalog) → agent stops at the creation row |
| author_activity | 3 of 4 | report verification (see above) |
| issue_prs | 3 of 4 | agent follows "linked", the question says "opened" (the rows with and without an `opened` row are both shown; the agent does not keep the qualifier) |
| assignment_history | 2 of 4 | report step: r1 a verified rewrite dropped changes; confirm run: the finding listed all five changes but cited no row, so the report writer got the branch's last result (one change) and reported only that one |
| resolution_vs_merge_order | 1 of 4 | one branch computed a wrong difference itself; the combine step was not reached and the report declined |

#### Known limits (additions)

- The Jev report verdict is often "unsupported" for correct reports (20 of 57); the rewrite is
  kept only when it verifies, but a verified rewrite can still drop facts. Grounding (0
  ungrounded answers) is bought with about 1–2 tasks of 19 on smoke.
- Agents still compute some answers themselves instead of calling `compare_times`.
- Latency p95 54–64 s locally (multi-source questions run their branches one chat call at a
  time); p95 72 s on OpenRouter.

#### Code status

Final for stage 3: `identity.code_sha256 = 1a21f39f930e9a0719fc9c8444f12b733b20136951d4ddb590c496bb2b747ef2`,
config (local default) `2add464747c8438925516898c4dae54bbf32ea8e0b506a92579dbd6118069de6`,
config (OpenRouter preset) `101557f661923bdf5e40d82729edbf846aa623149f1d16354cee095436f5e99e`.
Additive to the stage-2 interface: `Engine(...)`, `research()` and the result keys are unchanged;
new: `usage.unmetered_upper_usd`, `usage.{jev,chat}.{retries,cancelled}`, `trace.verification`,
`trace.branches[].results`, `identity.code_sha256`; `research()` raises `InfraError` (with
`.usage`, `.calls`) for every provider failure.

### After the final engine code (2026-09-28, afternoon)

- **Pause rule narrowed.** The final engine meters the call in flight at the deadline as `cancelled`
  every time the time budget runs out, so "time + any cancelled call" paused on ordinary slow answers
  (smoke-live-3, `issue_changed_files` × H-engine, image `2bc32a8f7039`: paused; its evidence was not
  kept, so the cause cannot be shown). Now a cancelled call counts only if it had already run its own
  per-request timeout (Jev 12 s, chat 60 s); retries and failed attempts still count. A paused engine
  trial's receipts, engine trace and agent trace are now kept under `<results>/paused/`.
- **Receipts** also carry `verification` (verdict, regenerations, counts of stripped values / removed
  citations); `result.json → engine` sums retries and cancelled calls per model.
- **Override scope.** `--allow-image-change` also accepts a change of `mcp_server.py` ALONE (engine and
  raw-store code and config identical); old and new hashes go into `engine.image_overrides`.
- **Completed runs are not extended.** A run whose `summary.json` says complete and whose pinned arms
  all have trials refuses a config that lists new arms, unless `--extend-run`
  (28 Sep incident: the retired watcher `runs/w2026-add-arms.sh` rewrote the main config with every arm
  of arms.yaml at boot; the script now exits at once). Limit: arms the run already pinned (e.g. the
  retired F0–F6) are not new and would still get trials if relisted.
- **Coding trial of smoke-live-3** (H-engine, C-KAFKA-PR-13453), alone, once `free -g` shows ≥ 14 GB
  available and no other container runs:
  `.venv/bin/python -m harnext_eval.e2.v2.runner --config apps/eval/configs/e2-engine-smoke-live-3-coding.yaml --live --no-wait`

**Coding trial done (15:00–15:30).** C-KAFKA-PR-13453 × H-engine resolved on image `a75ed05ffc6e`: 1,460 s
agent + 185 s grading, 31 turns, 5 `context_research` calls (4 partial by the engine's own time/turn
budgets, 2 chat calls cancelled 3–4 s into the deadline), engine $0.0132 (Jev $0.0076, chat $0.0056,
unmetered ≤ $0.0005), agent $0.31. The pre-narrowing rule would have paused this trial falsely (twice);
the current rule and the trace checks did not fire. Container peak 8.9 of 10 GiB, host ≥ 35 GiB
available, no swap. Report view (all 19 trials): `configs/e2-engine-smoke-live-3-report.yaml` — for
the report scripts only, never `--live`.

**Crash-safe scratch.** Anything that must survive a crash or reboot (memory samples, logs, evidence,
helper scripts of a run) belongs under `apps/eval/out/` (e.g. the run's directory), never under `/tmp`:
the 14:24 reboot wiped `/tmp`. Example: `runs/engine-smoke-live-3/memwatch.py` writes fsync'd samples
every 30 s to `runs/engine-smoke-live-3/memory-coding.log` and stops the runner below 5 GB available.

### Full-run pause of 28 Sep 13:45 UTC: diagnosis

The first full w2026 engine run (`w2026-engine-claude-code-haiku-4-5`) paused within minutes: in all
three trials Jev requests hit the 12 s request timeout (24 of 67 calls with 1–3 failed attempts), research
calls ended partial on time, and the infra rule paused the run (correctly; 0 trials recorded, evidence in
`<results>/paused/`). Reproduction and measurements: `out/kafka-500-v2/runs/w2026-engine/diagnosis/`
(invented questions only, no benchmark content).

- Not local. In trial-identical containers on the w2026 export: event-loop lag max 6–7 ms (one or three
  containers), 0.2 s with the F5 notes; a plain thread's lag ≤ 19 ms (no GIL holds; SQL runs in a worker
  thread); cgroup `nr_throttled` 0 in every run. Proxy: tunnel setup median 4 ms (max 1 s) during the failed
  run, proxy TCP retransmissions 0.1 %, four simultaneous 5-minute pacers without load identical through
  the proxy and from the host (0 of 600 calls > 2 s).
- Upstream. The same stalls appear from the HOST with no container and no proxy: 23–27 s waits for the
  first byte after a 4 ms connect and a 50 ms TLS handshake (16:51–17:08 local: 11 of 120 pinned calls
  > 2 s, keep-alive A/B 10 of 32 > 12 s, low-rate watch 3 of 20 at 23–26 s). They come and go (clean
  20 minutes earlier, clean again at 17:07) and hit individual requests/connections: a host pacer on one
  connection stayed at ≤ 0.5 s while trial containers stalled — which is why host checks looked healthy.
- Full stack (real agent + real MCP server, 3 containers, invented questions, 7 rounds): 5 Jev stalls
  in ~150 engine Jev calls, one pause-worthy (a 75 s chat report); with no stall, no pause.

**Change (integration side, host only):** a Jev health gate (`container.jev_unhealthy`, runner
`provider_gate`). Before any trial starts — at every start and every automatic resume — the runner
makes 8 sequential Jev calls from the host with an invented state; a call at or past the engine's own
request timeout, or an error, pauses the run before a paid trial (the runner then waits 15 min and
probes again; `--no-wait` exits 75). Engine, budgets, timeouts, pause rule, image and pins unchanged
(image `a75ed05ffc6e`, engine `1a21f39f…`, `mcp_server` `e24afd15…`, `tree_matches_image: true`).
It cannot prevent a stall that begins mid-run: that trial pauses as before and is retried later.

**Start (no pinned code or image changed; the run's manifest pins stay valid):**
```
free -g; df -h /                       # >= 14 GB available, >= 12 GB disk; no other harnext container
.venv/bin/python -c 'from harnext_eval.e2.v2.agents import container; print(container.jev_unhealthy() or "healthy")'
.venv/bin/python -m harnext_eval.e2.v2.runner --config apps/eval/configs/e2-w2026-engine-claude-haiku.yaml --identity   # tree_matches_image: true
nohup .venv/bin/python -m harnext_eval.e2.v2.runner --config apps/eval/configs/e2-w2026-engine-claude-haiku.yaml --live \
  >> apps/eval/out/kafka-500-v2/runs/w2026-engine-claude-code-haiku-4-5.log 2>&1 &
```
Without `--no-wait` the runner keeps waiting out provider stalls (15 min per retry, one probe each);
stop it with `kill -TERM <pid>`. If the image or pinned code ever changes before the first trial is
recorded, move the empty run directory aside instead of overriding
(`mv .harnext/evaluations/kafka-500-v2/w2026-engine-claude-code-haiku-4-5{,.paused-before-start-<date>}`,
which keeps `paused/`) and start fresh.

### Provider retry and trial-scoped pauses (designed 28 Sep evening; staged in `dev-src`, not yet live)

Why: coding trials make ~5 open-ended research calls that usually end on the engine's time budget;
with upstream Jev stalls on a few per cent of calls nearly every coding trial met "partial on time with
failed/retried calls", paused the whole run and waited 15 min (twice for `C-KAFKA-PR-22460/H-engine`).
Principle kept: no answer degraded by provider trouble reaches the agent or a record.

Design (code in `out/kafka-500-v2/runs/w2026-engine/dev-src/eval/`, engine package untouched):
- **Redo inside the MCP server** (kind engine only). An attempt that `engine_trouble` calls degraded
  (partial on time + failed/retried/timed-out calls), or a transient `InfraError` (timeouts, 408/425/429,
  5xx incl. Cloudflare 52x, unreachable), is discarded; the server probes Jev (3 sequential calls with an
  invented state; healthy = each < 5 s, 8 s cap), waiting 10, 20, 30, 30 s between probes, no new probe
  after 90 s (one wait ≤ 114 s), then redoes the research from scratch; at most 3 attempts. Only the
  clean attempt's answer reaches the agent. Non-transient failures (400/401/402/403/404/422, no key,
  rejected requests, engine exceptions, a failed branch) pause at once, receipt `infra_scope: "run"`;
  trouble that outlives every redo gives `infra_scope: "trial"`.
- **Receipts**: per call `attempts` (each discarded attempt's reason, usage, seconds, health wait with
  its probes, trace ref), `waited_s`, `provider_retry_usd`; `usage` is the clean attempt's. The engine
  trace file has every attempt (`attempt`, `redo`). `result.json → engine` adds `provider_retry_usd`,
  `discarded_attempts`, `waited_s` (over every call, also one with no clean attempt). Reports: Table 4
  column "Provider-retry $" (engine view), included in trial totals; charts include it in read cost.
- **Tool timeout per kind**: engine 3 × (120 + 10) + 2 × 114 + 60 = **678 s** (`MCP_TOOL_TIMEOUT`,
  Codex `tool_timeout_sec`); research kinds keep 240 s. The historical trial limit (900 s) leaves 222 s.
  A trial that hits its own time or turn limit after provider redos is a trial pause, never a result.
- **Trial-scoped pauses in the runner** (engine arms only): a receipt with `infra_scope: "trial"`, an
  agent-side tool timeout, a server that died mid-call, or the time-limit case above raise `TrialPause`:
  that trial is left unrecorded and re-queued, trials in flight finish and are recorded, the health gate
  runs once before the next trial (healthy → continue at once; unhealthy → the whole run pauses and waits
  as before). A trial paused 3 times in a session is `deferred` (session file: `trial_pauses`,
  `deferred`; printed) and retried once at the end; still failing → the run ends `paused` and the next
  automatic resume retries it. Everything run-wide (usage limit, auth, credit, disk, image/pin mismatch,
  server not started/connected, engine bug) pauses the run as before. Runs without engine arms: unchanged
  (`test_runner.py` 21 passed with the new code; the main run's `--check-resume` on a scratch copy:
  1,080 recorded, 0 remaining, 21/21 recorded arm pins and the prompt-template hash equal; its report
  regenerated byte-identical).

Files that change (copy from `dev-src/eval/` to `apps/eval/`):
`src/harnext_eval/e2/v2/mcp_server.py`, `src/harnext_eval/e2/v2/runner.py`,
`src/harnext_eval/e2/v2/agents/container.py`, `src/harnext_eval/e2/v2/agents/codex.py`,
`src/harnext_eval/e2/v2/agents/fake.py`, `tests/test_e2_v2/test_engine_arm.py`, `scripts/e2_report.py`,
`scripts/e2_charts.py`. Pinned hashes after the change: engine `1a21f39f…` and raw store `fa45209c…`
unchanged, engine config `101557f6…` unchanged, `mcp_server.py` `e24afd15…` → `994a9eb9…`.

**Switch-over** (from the repo root; the running process keeps its old code in memory until it exits):
```
R=apps/eval/out/kafka-500-v2/runs/w2026-engine-claude-code-haiku-4-5; DEV=apps/eval/out/kafka-500-v2/runs/w2026-engine/dev-src/eval
PID=$(ps -eo pid,args | awk '$2 ~ /python$/ && $4=="harnext_eval.e2.v2.runner" && $6=="apps/eval/configs/e2-w2026-engine-claude-haiku.yaml" {print $1}'); echo $PID
# 1. drain: no new trials, those in flight finish and are recorded. A drained runner reloads its config
#    and would start again, so hold the run's lock: the reloaded runner then exits with "another runner
#    holds this run_id" before starting any trial (tested on a fake run: in-flight trial recorded, exit 1).
kill -USR1 $PID
flock -x $R/runner.lock -c 'while kill -0 '$PID' 2>/dev/null; do sleep 5; done; echo runner gone' &
while kill -0 $PID 2>/dev/null; do sleep 10; done; wait        # coding trials can take ~30 min to finish
cat .harnext/evaluations/kafka-500-v2/w2026-engine-claude-code-haiku-4-5/summary.json | head -5   # paused, draining
# 2. back up the current files, then apply the new ones
B=apps/eval/out/kafka-500-v2/runs/w2026-engine/backup-tree-$(date +%Y%m%d-%H%M); mkdir -p $B
for f in src/harnext_eval/e2/v2/mcp_server.py src/harnext_eval/e2/v2/runner.py src/harnext_eval/e2/v2/agents/container.py \
         src/harnext_eval/e2/v2/agents/codex.py src/harnext_eval/e2/v2/agents/fake.py tests/test_e2_v2/test_engine_arm.py \
         scripts/e2_report.py scripts/e2_charts.py; do
  mkdir -p $B/$(dirname $f) && cp -a apps/eval/$f $B/$f && cp $DEV/$f apps/eval/$f; done
# 3. image: keep the old one tagged, rebuild (free -g >= 6 GB, df -h / >= 12 GB)
docker image inspect -f '{{.Id}}' harnext-e2-agent:latest        # expect sha256:a75ed05ffc6e...
docker tag harnext-e2-agent:latest harnext-e2-agent:pre-provider-retry-20260928
docker build -f apps/eval/docker/e2-agent/Dockerfile -t harnext-e2-agent .
# 4. verify: tree_matches_image true; in_image engine 1a21f39f..., raw fa45209c..., mcp_server 994a9eb9...;
#    pinned still shows image a75ed05f and mcp_server e24afd15 (changed at the resume below)
.venv/bin/python -m harnext_eval.e2.v2.runner --config apps/eval/configs/e2-w2026-engine-claude-haiku.yaml --identity
.venv/bin/python -m pytest apps/eval/tests/test_e2_v2/test_engine_arm.py -q      # 39 passed (no trial running)
# 5. resume; --allow-image-change records image a75ed05f -> new and mcp_server e24afd15 -> 994a9eb9 in
#    manifest.json -> engine.image_overrides (engine / raw-store code must not move, or it refuses)
nohup .venv/bin/python -m harnext_eval.e2.v2.runner --config apps/eval/configs/e2-w2026-engine-claude-haiku.yaml \
  --live --allow-image-change >> $R.log 2>&1 &
python3 -c "import json;e=json.load(open('.harnext/evaluations/kafka-500-v2/w2026-engine-claude-code-haiku-4-5/manifest.json'))['engine'];print(e['image_id'], e['image_overrides'][-1])"
```
Trials recorded before carry `image_id` `sha256:a75ed05ffc6e…` in `result.json`, later ones the new ID.

**Rollback** (same stop as step 1, then):
```
for f in $(cd $B && find . -type f); do cp $B/$f apps/eval/$f; done          # the backed-up files
docker tag harnext-e2-agent:pre-provider-retry-20260928 harnext-e2-agent:latest
.venv/bin/python -m harnext_eval.e2.v2.runner --config apps/eval/configs/e2-w2026-engine-claude-haiku.yaml --identity  # tree_matches_image true, mcp_server e24afd15
nohup .venv/bin/python -m harnext_eval.e2.v2.runner --config apps/eval/configs/e2-w2026-engine-claude-haiku.yaml \
  --live --allow-image-change >> $R.log 2>&1 &                                 # records the switch back
```
Nothing recorded is removed either way; trials of each image stay identifiable by `image_id`.

### Engine v2: tool-call budget feedback and DeepSeek preset (2026-09-29)

Motivation (disclosed: the w2026 benchmark is now a seen set for this design): in the full w2026
run of code `1a21f39f…` (arms `H-engine`, `H-engine-mem`), 69 % of research calls ended partial;
branch endings were 209 × `time budget (120 s)`, 111 × `turn budget (8 turns)`, 323 normal. A
cut-off branch's finding was "(unfinished: …) Last results: …": the model never wrote its best
answer and was never told its budget. Nothing in this round is derived from w2026 task content;
development used the smoke benchmark only.

#### What changed (only with `budget_feedback = true`; off = v1, byte-identical inputs)

- **Budget of tool calls** (`tool_budget`, default 8) per branch agent; `done` does not count.
  Every tool result ends with the line `Tool call budget: m/n`. The branch system prompt gets one
  sentence (`prompts.BUDGET_SYSTEM`):
  > You have 8 tool calls (done does not count); every tool result ends with how many you have
  > used. Plan them: when they are used up, or the time runs out, further calls are not run and
  > you must answer with done from what you already have.
- **Budget used up**: a further tool call is not executed; its result is
  > Not executed: the tool call budget is used up. Answer now from the results you already
  > have: call done with your finding and the evidence ids.
  >
  > Tool call budget: 8/8
- **Time**: once the time left is below `reserve_s + max(answer_reserve_s, 2 × the branch's
  slowest turn so far)`, a tool call is not executed and its result is
  > Not executed: the time for this research is almost up. Answer now from the results you
  > already have: call done with your finding and the evidence ids.
  (with the budget line). The slowest-turn term lets slow models wrap up earlier.
- **Calls rejected before execution count** (malformed arguments, unknown tool, invalid plan,
  refused repeat): each result says what was wrong. Otherwise a confused model could loop
  without end (the w2026 traces show 146 invalid, 43 repeat and 257 restate outcomes).
- **The branch ends with the model's own finding**: `done`, or a plain-text reply. A model that
  keeps calling tools after being told to answer gets `answer_turns` (3) more turns, then one
  tool-less "answer now" call (`forced`). A branch still running at the branch cut-off
  (`deadline_s - reserve_s`) is cancelled and answers in one tool-less call from its own
  conversation (`conclude_s`, all cut branches at once; `prompts.BUDGET_TIME_FINAL`):
  > The time for this research is up: no more tool calls. Answer now from the results you
  > already have, in plain text: your finding, with the evidence ids it comes from.
  An answer given as a `done` call counts too. Only when that call also fails or times out does
  the branch end "(unfinished …)"; the timed-out call is metered as cancelled.
- **Replies spent on reasoning** (`retry_truncated`): a chat reply with no text and no tool call
  that stopped at the output-token limit is asked once more with twice the limit (both calls
  metered). A report that is still empty is labelled "(report not written: empty reply)".
- Trace: per branch `tool_calls`, `ending` (normal | budget | time | forced | cutoff | turns |
  failed), `started_s`, `ended_s`; per call `reasoning_tokens`, `cached_tokens`; `identity`
  gains `budget_feedback`, `tool_budget`, `deadline_s`.
- v1 is kept: with `budget_feedback = false` the old turn loop runs, no budget text is sent, and
  the v2 settings and texts are left out of the config hash, so `config_hash(openrouter)` is
  still `101557f6…` and `test_engine_budget.py` checks the prompts, the config dump and two
  rendered branch prompts against a snapshot captured from code `1a21f39f…` before the change.

#### Presets

| preset | chat model | behaviour | list price $/M in / out |
|---|---|---|---|
| `openrouter` | openai/gpt-oss-20b | v1 (unchanged; hash `101557f6…`) | 0.018 / 0.09 |
| `openrouter-b8` | openai/gpt-oss-20b | v2 | 0.018 / 0.09 |
| `deepseek-flash-b8` (primary) | deepseek/deepseek-v4-flash-0731 | v2 | 0.018 / 0.32 |
| `deepseek-b8` (comparison) | deepseek/deepseek-v4.1-flash | v2 | 0.30 / 1.20 |

Same prompts, budget, deadline policy, temperature 0, seed 7 and `reasoning: {effort: low}` for
all b8 presets; only the model differs. DeepSeek on OpenRouter: reasoning is set with
`reasoning: {effort | enabled | exclude}` (also `reasoning_effort`), returned as
`message.reasoning` plus `reasoning_details`, and counted in `completion_tokens`
(`completion_tokens_details.reasoning_tokens`); cached input in
`prompt_tokens_details.cached_tokens`; cost in `usage.cost`. Both DeepSeek models call tools
reliably at `effort: low` (probes in `engine-dev/v2-probes/`), so no other setting was needed.

#### Deadline policy (all b8 presets)

`deadline_s 240` (the maximum allowed), `reserve_s 90` (conclusion ≤ 40 s + aggregation),
`answer_reserve_s 30`, `conclude_s 40`: tool calls are refused from 120 s at the latest (earlier
for slow branches), branches are cut at 150 s, a cut branch answers by 190 s, the rest is for
judge, report and verification. Measured on smoke (see below): the cap binds only for DeepSeek on
the two questions that route to two sources (their branches alternate on one chat slot, turns
grow to 40–65 s as the conversation grows). For the integration: with the redo wrapper,
attempts × (240 + 10) + waits + 60 < 900 s allows at most 3 attempts with ≤ 90 s of waits.

#### Results (smoke, 19 historical tasks, RAW store build 3, engine alone)

Code `e2e1a8757f734fbd0d0d288c066f945192fa649c4149f45a214486007dd1b1aa`, runs
`engine-dev/v2-final5-*` (numbers: `scripts/e2_engine_numbers.py`, `v2-final5-numbers.json`;
spend: `engine-dev/v2-spend.py`). All presets on OpenRouter (the local server was down).
Costs are provider-reported (`usage.cost`); "list" = tokens × list price. Seconds = one research
call, with up to four evaluation processes sharing the provider.

| preset | runs: correct | $/question: Jev / chat / total | p50 / p95 s | tool calls per branch (all 8 used) | branch endings | partial answers | verification changed / regenerated |
|---|---|---|---|---|---|---|---|
| openrouter (v1) | 16, 15 | 0.00076 / 0.00074 / 0.00150 | 27 / 58 | — (turn loop) | normal 39, turn budget 3 | 11/38 | 10 / 7 |
| openrouter-b8 | 16, 16 | 0.00070 / 0.00076 / 0.00146 | 29 / 69 | 2.51 (1/43) | normal 42, forced 1 | 8/38 | 12 / 9 |
| **deepseek-flash-b8** | 16, 17, 16 | 0.00056 / 0.00141 / 0.00196 | 53 / 240 | 2.99 (0/65) | normal 52, time 8, cut-off 5 | 21/57 | 16 / 16 |
| deepseek-b8 (V4.1) | 17 | 0.00064 / 0.00523 / 0.00587 | 27 / 70 | 4.19 (3/21) | normal 21 | 5/19 | 11 / 11 |

Per question (means over the runs):

| preset | chat input tokens (cached share) | chat output tokens (reasoning share) | branch turns | chat $ reported vs list | engine $ per correct answer |
|---|---|---|---|---|---|
| openrouter | 26,166 (0.26) | 872 (0.26) | 4.81 | 0.00074 vs 0.00055 (×1.34) | 0.00184 |
| openrouter-b8 | 26,566 (0.20) | 1,014 (0.28) | 4.81 | 0.00076 vs 0.00057 (×1.32) | 0.00173 |
| deepseek-flash-b8 | 22,467 (0.46) | 3,034 (0.67) | 4.00 | 0.00141 vs 0.00138 (×1.02) | 0.00229 |
| deepseek-b8 | 37,224 (0.79) | 4,381 (0.64) | 5.21 | 0.00523 vs 0.01642 (×0.32) | 0.00657 |

- **Per model turn** (branch turns): gpt-oss-20b median 96 output tokens (26 reasoning), 2.7 s
  (p95 11.9 s); DeepSeek V4 Flash 0731 median 260 (78 reasoning), 6.6 s (p95 27.2 s, max 45 s;
  late turns with large contexts up to ~65 s); DeepSeek V4.1 Flash median 280 (87 reasoning),
  1.8 s (p95 6.6 s).
- **Prompt caching**: the providers report cached input (20–79 % of input tokens), but only the
  V4.1 Flash provider's bill reflects a discount (reported cost 0.32 × list). For V4 Flash 0731
  reported ≈ list (×1.02) despite 46 % cached; gpt-oss-20b providers bill 1.32–1.34 × the listed
  price.
- **Cost hypothesis (a more capable model needs fewer turns and is cheaper overall): not
  confirmed on smoke.** V4 Flash 0731 used 17 % fewer branch turns (4.00 vs 4.81) and 15 % fewer
  input tokens than gpt-oss-20b with budget feedback, but 3.0 × the output tokens (two-thirds
  reasoning) at 3.6 × the output price: chat cost 1.85 ×, engine cost per question 1.34 ×, per
  correct answer 1.32 × (0.00229 vs 0.00173 US$). Break-even against gpt-oss-20b-b8 at the same
  input: its output would have to shrink 2.7 × (to ≤ 1,100 tokens per question); overall it used
  0.92 × gpt-oss's chat tokens and would have to cost 1.85 × less per token. V4.1 Flash costs
  3.8 × per correct answer.
- Jev is unchanged: median 6 calls and 1,461 input tokens per research call, latency p50 0.29 s /
  p95 0.47 s; Jev retries 1 in 8 runs. Chat retries 4 (all DeepSeek). Guard refusals 0.
  Ungrounded answers (audit): 1 / 4 / 8 / 4 over the rows above (answers keep a value the rows
  do not show verbatim).
- **DeepSeek tool calling**: 0 malformed JSON arguments. Unusual: in a tool-less call it may
  still write its tool call as text in DSML markup (`<｜DSML｜tool_calls><｜DSML｜invoke name=…>`),
  seen 2× in 12 runs; now parsed (a leaked `done` counts). It spends the whole output allowance
  on reasoning in some report calls (13 of 194 report calls before `retry_truncated`), which then
  come back empty.

#### Deadline measurements

At a 240 s deadline with the first policy (reserve 20 s, answer margin 45 s), branch end times
were: gpt-oss-20b-b8 p50 25 s, p95 69 s, max 70 s; DeepSeek V4 Flash p50 22 s, p90 89 s,
p95 165 s, max 220 s; aggregation after the branches p95 25 s (gpt-oss) and 47 s (DeepSeek),
max 51 s. The long DeepSeek branches are the two questions routed to two sources: their branches
alternate on the single chat slot and late turns take 40–65 s. With the final policy (branches
cut at 150 s; all runs of v2-final5): gpt-oss-20b-b8 branches ended by p50 21 s / p95 43 s /
max 59 s (no cut, no time ending),
DeepSeek V4.1 by p95 57 s / max 65 s (no cut), DeepSeek V4 Flash 0731 by p95 150 s: 60 of 65
branches (92 %) answered themselves (52 normally, 8 after an answer-now message or at the
cut-off), 5 (8 %, all on the two-source questions) ended unfinished because the final answer
call itself did not return within 40 s. The 95 % target is met for gpt-oss-20b and V4.1 Flash but
not for V4 Flash 0731 within the 240 s cap; options: a higher cap, two chat slots for the b8
presets, or a reasoning-free final answer call (not done: outside this round's scope).

Integration: deadline **240 s**; attempts × (240 + 10) + waits + 60 < 900 s allows ≤ 3 attempts
with ≤ 90 s of waits in total.

#### Version history of this round (all on smoke; every change generic)

| code | change | DeepSeek V4 Flash correct (3 runs) |
|---|---|---|
| `91fcb4e6` | budget feedback, presets, deadline policy 240/60/60 | 18, 17, 18 (runs v2-final-*) |
| `8f2a0503` | conclusion call for branches cut at the deadline; policy 240/90/30/30 | 18, 18, 18 (v2-final2-*) |
| `842e75fc` | adaptive answer-now margin (2 × slowest turn), robust conclusion, 40 s | 17, 18, 16 (v2-final3-*) |
| `e76c083f` | retry of replies spent on reasoning; "empty reply" label | 17, 17, 15 (v2-final4-*) |
| `e2e1a875` | DSML leak parsing; adaptive margin capped (a stalled call cannot close the tool window) | 16, 17, 16 (v2-final5-*) |

The spread between versions (15–18 of 19) is of the size of run-to-run noise; the misses are
`issue_prs` in every run of every version (the agent keeps pull requests linked but not opened),
the two-source `resolution_vs_merge_order` under time pressure, and scattered single-source misses
(1–2 per run). The later changes were each motivated by a traced failure (a stall closing the
tool window, DSML leaks, empty reasoning-only reports) and are kept.

#### Hashes (superseded by the follow-up below)

`identity.code_sha256 = e2e1a8757f734fbd0d0d288c066f945192fa649c4149f45a214486007dd1b1aa`;
config hashes: `openrouter` 101557f661923bdf5e40d82729edbf846aa623149f1d16354cee095436f5e99e
(unchanged v1), `openrouter-b8` 25d635b2bef5de276c8683ee8efc9777fd1d88fd7224f99a23f99f5286393d37,
`deepseek-flash-b8` 9955974d7b339e6de8954a73e8bbef954d7466ceb124cc5e0942ef9a7ad61bd0,
`deepseek-b8` ad10b62272f6f55e8503e29334edda3bca3c2a13c187ad0015e46fba2eed22a6.

#### Follow-up: branches side by side, answer-now calls (2026-09-29)

Why: on w2026 the hard questions route to two or three sources, and the primary preset could not
finish two-source questions on smoke (DeepSeek Flash branches alternated on the single chat slot;
5 of 65 branches ended unanswered). Development used smoke and 12 invented questions only.

Changes (budget-feedback presets only; `openrouter` stays byte-identical, hash `101557f6…`):
- **Chat concurrency** (`parallel_branches`, 4 in every b8 preset): a research call allows as
  many chat calls at once as it has branches (at most 4); the single chat slot remains for v1
  and Ollama (`chat_concurrency 1`). **Order**: with `parallel_branches` the branches are sorted
  by source name before they start, so judging, combine, the report prompt, evidence and rows
  (within a branch: in result order) do not depend on which branch finishes first
  (`test_findings_are_ordered_by_source_name_whatever_finishes_first`). The routing order stays
  in the trace.
- **Answer-now calls**: after the budget or time message, the forced finish and the conclusion
  at the cut-off use `answer_options` (the lowest reasoning each model accepts, probed in
  `engine-dev/v3-probes/lowest-reasoning.txt`: DeepSeek V4 Flash 0731 and V4.1 Flash accept
  `reasoning: {effort: none}` (0 reasoning tokens); gpt-oss-20b rejects `none` / `enabled:false`
  ("Reasoning is mandatory") and `minimal` equals `low`, so it keeps `low`) and a fixed
  `answer_tokens` allowance of 1,500 (findings on smoke: p99 ≈ 1,500 characters ≈ 500 tokens).
  After the budget or time message only `done` is offered, so the answer turn cannot start
  another search (before this, DeepSeek kept calling tools in up to 5 answer turns).
- **Replies spent on reasoning**: the one retry (`retry_truncated`) now also uses
  `answer_options`. Cause found in the traces: at `effort: low` DeepSeek Flash sometimes spends
  the whole 2,048-token allowance on reasoning (35–50 s); the old retry at twice the allowance
  repeated the spiral (34–75 s) and ran into the cut-off. For gpt-oss (`low` = its options)
  nothing changes.
- `ChatClient.complete(options=…)` merges per-call options over `chat_options`.
- `e2_engine_dev.py --questions FILE` runs invented questions (`{"id","checkpoint","question"}`)
  through the engine alone (no extraction, no grading); `e2_engine_numbers.py` adds per run
  `timing` and `two_or_more_sources`: branch end times, branches that reached the cut-off
  (answered by the conclusion call) or ended unanswered, endings by the budget / time message,
  answer-now calls (seconds, timeouts), chat seconds per branch wall second (above 1: branches
  ran side by side), reports not written.

Results, smoke (19 historical tasks), code `ca68050c…`, runs `engine-dev/v3b-final-*`
(numbers `v3b-numbers.json`):

| preset | runs: correct | $/question: Jev / chat / total | p50 / p95 s per run | tool calls per branch (all 8 used) | branch endings | partial answers | verification changed / regenerated |
|---|---|---|---|---|---|---|---|
| openrouter-b8 | 15, 18 | 0.00063 / 0.00071 / 0.00134 | 19/35, 22/86 | 2.65 (2/42) | normal 42 | 7/38 | 10 / 10 |
| **deepseek-flash-b8** | 17, 17, 18 | 0.00051 / 0.00147 / 0.00199 | 48/153, 61/170, 52/206 | 3.28 (5/64) | normal 59, time 5 | 12/57 | 15 / 11 |

| preset | chat input tokens (cached share) | chat output tokens (reasoning share) | branch turns | chat $ reported vs list | engine $ per correct answer |
|---|---|---|---|---|---|
| openrouter-b8 | 25,322 (0.21) | 919 (0.22) | 4.58 | 0.00071 vs 0.00054 | 0.00155 |
| deepseek-flash-b8 | 24,472 (0.54) | 2,959 (0.60) | 3.95 | 0.00147 vs 0.00139 | 0.00218 |

Questions routed to two or more sources (smoke; routing varies per run):

| preset | question-runs | branches reaching the cut-off / unanswered | seconds per research call (p50/max per run) | correct |
|---|---|---|---|---|
| deepseek-flash-b8, before (code `e2e1a875`, v2-final5) | 8 | 10 of 16 / 5 | 206–240 | 2/8 |
| deepseek-flash-b8, parallel only (code `8abdccde`, v3a-final) | 6 | 9 of 12 / 0 | 182–240 | 3/6 |
| **deepseek-flash-b8 (final)** | 7 | 2 of 14 / 0 | 141/153, 117/170, 195/240 | 3/7 |
| openrouter-b8 (final) | 4 | 0 / 0 | 30/35, 44/65 | 2/4 |

The two cut-off branches in the final runs: one model call that returned nothing for 116 s
(provider stall), and one branch whose turns grew to 40–55 s (answered by the conclusion call).

12 invented questions (`engine-dev/v3-invented-questions.jsonl`; mailing-list decisions with
their issue and pull request, fixes and backports, a committer announcement and the reviews;
all at checkpoint 2023-02; not taken from any benchmark file), engine alone, runs `v3b-inv-*`:

| preset | branches | branch end s p50 / p95 / max | reached the cut-off / unanswered | ended by the budget / time message | seconds per research call p50 / p95 / max | tool calls per branch (all 8 used) | $/question |
|---|---|---|---|---|---|---|---|
| deepseek-flash-b8 r1 | 28 | 61 / 87 / 119 | 0 / 0 | 1 / 1 | 124 / 205 / 226 | 7.68 (21/28) | 0.0079 |
| deepseek-flash-b8 r2 | 28 | 51 / 113 / 150 | 1 / 0 | 0 / 1 | 146 / 226 / 240 | 7.39 (18/28) | 0.0067 |
| openrouter-b8 | 28 | 30 / 54 / 78 | 0 / 0 | 0 / 0 (forced 2) | 56 / 115 / 143 | 5.71 (5/28) | 0.0059 |

The one cut-off branch on the invented set: its answer-now call (reasoning off, done only) got no
reply for 51 s (provider stall); the conclusion call then answered in 4.9 s. Answer-now calls
over all final runs: DeepSeek 13, of which 2 did not return (both stalls, counted above), p50
≈ 10 s; gpt-oss 2, none timed out; conclusion calls: 0 timeouts. Chat seconds per branch wall
second: 1.2–1.6 (1.0 before: one call at a time). Reports not written: 1 of 81 DeepSeek research
calls (a report call stalled until the deadline). Target (no branch cut off unless the provider
failed) met: every branch answered itself, the three branches that reached the cut-off include
two provider stalls. Retries: chat 8, Jev 1; guard refusals 0.

Deadline: keep **240 s**. DeepSeek Flash branches now end by p95 87–145 s (smoke and invented),
but its research calls still reach p95 205–226 s: judge, combine and a report of 15–80 s (the
report, not changed in this round, sometimes reasons for 2,500 tokens and is retried). A longer
deadline does not fit the redo formula (attempts × (deadline + 10) + waits + 60 < 900 s allows
3 attempts only up to 240 s). gpt-oss-20b-b8 needs at most 143 s.

Spend of this follow-up: US$0.80 (`engine-dev/v3-spend.py`, all v3* runs and probes).

Hashes of the final code: `identity.code_sha256 =
ca68050ca3a08d1fb8892f1da55c6dcd24ed97a75ed5efbb075370c61b557166`; config hashes: `openrouter`
101557f661923bdf5e40d82729edbf846aa623149f1d16354cee095436f5e99e (unchanged v1),
`openrouter-b8` 39c29f2b8bcb7c7f2954b9d91f00127e2f074682ced93c09ef8e368dd8898094,
`deepseek-flash-b8` 77ba18b522d25c3f142d65f2b93351490063ba80549653ded00dc386e4752932,
`deepseek-b8` fe13c7a9f4762e0fbda81ba77633beb15592f4be11a833fd08d7ccfbf2ba836f.

### Engine v2 in the benchmark (integration, 29 Sep; prepared, not started)

- **Arms** (`arms.yaml`, dated comment): `H-engine-b8` (preset `openrouter-b8`: gpt-oss-20b + tool-call
  budget feedback 8) and `H-engine-b8-ds` (`deepseek-flash-b8`: DeepSeek V4 Flash `deepseek-v4-flash-0731`
  + budget feedback 8); store RAW, no memory variants; access paragraph and standing instructions are
  H-engine's (kind engine). All 23 existing pins, the prompt-template hash and every recorded run pin
  unchanged; both finished run directories byte-unchanged (tree hashes `d8070c84…`, `76d328f4…`).
- **Redo policy from the preset's deadline** (`mcp_server.engine_policy`): attempts = the most (≤ 3)
  whose worst case `attempts × (deadline + 10) + (attempts − 1) × 114 + 60` leaves the agent ≥ 120 s of
  the 900 s historical limit. Deadline 120 s (H-engine): 3 attempts, tool timeout **678 s** (as before,
  agent keeps 222 s). Deadline 240 s (-b8 presets): 3 attempts would need 1,038 s, so **2 attempts**,
  tool timeout **674 s** (agent keeps 226 s). `MCP_TOOL_TIMEOUT` / Codex `tool_timeout_sec` per arm preset.
- **Classification with budget feedback**: a branch that used its 8 tool calls ("budget") or was made to
  answer ("forced") ended normally. Only time pressure makes a partial answer eligible: told to answer as
  time runs out ("time"), cut at the branch deadline ("cutoff"), "no answer at the deadline", or v1's
  "time budget"; and then only with failed, retried or over-timeout model calls. Engine v1 traces keep
  the v1 rule exactly.
- **Health gate**: Jev (8 calls, each < 12 s) plus, for every preset still to run, the chat model:
  2 short calls ("reply OK", 256 tokens max), each must answer within 30 s. Local presets skip the chat
  probe (preflight checks their endpoint).
- **Reports**: arm types "Raw store + decision model", descriptions naming the chat model and "tool-call
  budget feedback 8"; `--extra-config` is repeatable (three runs side by side). Chat cost = provider-
  reported cost (`usage.include`; reasoning tokens are part of the billed output). Existing single-run
  and two-run reports byte-identical; the two-run charts gain only the two new store rows.
- **Stages**: stage 1 `max_coding: 0`; stage 2 = raise `max_coding` to 12 in the same config. That is a
  cap change with the same arms, so the finished-run guard lets it through without `--extend-run`
  (tested). Resume with the same command.

**Start procedure** (only after the final engine hash is announced; repo root):
```
free -g; df -h /                    # >= 14 GB available, >= 12 GB disk; no harnext unit or container active
systemctl --user list-units 'harnext*' --no-pager; docker ps --format '{{.Names}}' | grep harnext-e2-
.venv/bin/python -c 'from harnext_eval.e2.v2.engine.config import code_hash; print(code_hash())'   # = the announced hash
for f in test_engine_arm test_mcp_server test_agents test_runner; do .venv/bin/python -m pytest apps/eval/tests/test_e2_v2/$f.py -q; done
# image: keep the current one, rebuild
docker tag harnext-e2-agent:latest harnext-e2-agent:pre-engine-v2-20260929
docker build -f apps/eval/docker/e2-agent/Dockerfile -t harnext-e2-agent .
.venv/bin/python -m harnext_eval.e2.v2.runner --config apps/eval/configs/e2-w2026-engine-v2-claude-haiku.yaml --identity
#   -> tree_matches_image: true; in_image engine_code = the announced hash, raw_store fa45209c...,
#      config_sha256 for openrouter-b8 and deepseek-flash-b8 as announced; pinned: null (new run)
# live check on the SMOKE benchmark: 4 trials, both new arms, historical only
.venv/bin/python -m harnext_eval.e2.v2.runner --config apps/eval/configs/e2-engine-v2-smoke-live.yaml --live --no-wait
.venv/bin/python apps/eval/out/kafka-500-v2/runs/w2026-engine-v2/prep/check_smoke.py
#   -> per call: chat provider-reported=True (DeepSeek and gpt-oss), reasoning/cached tokens present,
#      deadline=240.0 budget=8, no unexplained redo
.venv/bin/python apps/eval/scripts/e2_report.py --config apps/eval/configs/e2-engine-v2-smoke-live.yaml
# full run, stage 1 (historical), as a transient unit
L=$PWD/apps/eval/out/kafka-500-v2/runs/w2026-engine-v2; mkdir -p $L
systemd-run --user --unit=harnext-w2026-engine-v2-run --working-directory=$PWD \
  --property=Restart=on-failure --property=RestartSec=120 --setenv=PYTHONUNBUFFERED=1 \
  --property=StandardOutput=append:$L/run.log --property=StandardError=append:$L/run.log \
  $PWD/.venv/bin/python -m harnext_eval.e2.v2.runner --config apps/eval/configs/e2-w2026-engine-v2-claude-haiku.yaml --live
systemctl --user status harnext-w2026-engine-v2-run --no-pager; tail -f $L/run.log
# stage 2 later: stop the unit when stage 1 is complete (or drain), set max_coding: 12, start the same unit again
# three runs side by side:
.venv/bin/python apps/eval/scripts/e2_report.py --config apps/eval/configs/e2-w2026-claude-haiku.yaml \
  --extra-config apps/eval/configs/e2-w2026-engine-claude-haiku.yaml --extra-config apps/eval/configs/e2-w2026-engine-v2-claude-haiku.yaml \
  --out apps/eval/reports/e2/w2026-all-arms-v2/report.html
```
The new image differs from the one pinned in the finished run `w2026-engine-claude-code-haiku-4-5`;
that run is complete and must never be resumed (its pin check would refuse anyway without
`--allow-image-change`, which must not be used for it). Stop the new run with
`systemctl --user stop harnext-w2026-engine-v2-run` (unrecorded in-flight trials rerun on the next start).
