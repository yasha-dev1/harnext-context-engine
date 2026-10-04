"""Arm `S-tools`: the systems of record as of a checkpoint, one SQLite file per checkpoint.

    python -m harnext_eval.e2.v2.stores.build --arm S-tools --scope smoke|full --export all

The stream is consumed once, in order, into ONE cumulative working database; each time the
stream crosses a checkpoint boundary the working database is committed and snapshotted with
`VACUUM INTO export/<YYYY-MM>/sources.sqlite`. A snapshot is a physically separate file that
never saw a later event, so nothing an agent does with it (the CLIs, python's sqlite3, `strings`)
can reach the future. A shared database with an `as_of` filter would not have that property.

Every entity is rebuilt from events only: an issue's status is the `to` of its last status
change before the boundary, a PR merged after the boundary is OPEN. Long texts are stored
zlib-compressed and the FTS5 index is contentless with detail=none (token -> row, no
positions); that keeps 42 cumulative snapshots of a 730 M character corpus on one disk. The
CLIs decompress and re-check phrases themselves.

Resume: the working database commits only at boundaries, so after a kill it is exactly at its
last snapshot; the build skips the stream up to that boundary and goes on.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import zlib
from collections.abc import Iterable
from pathlib import Path

from harnext_eval.e2.v2 import stream

ARM = "S-tools"
DB = "sources.sqlite"
SCHEMA_VERSION = 1
INLINE = 200  # shorter texts stay plain TEXT; longer ones become zlib BLOBs
CLIPPED = {"description", "environment", "comment", "docs text"}  # long-text fields in history
CLIP = 500
AT = re.compile(r"at (\d{4}-\d\d-\d\dT[\d:.]+Z)\)\s*$")
MANY = {"component": "components", "fixversion": "fix_versions", "version": "versions"}
SINGLE = {"status", "priority", "assignee", "resolution", "issuetype", "reporter", "summary"}
SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE issues(key TEXT PRIMARY KEY, summary TEXT, description, status TEXT, priority TEXT,
  assignee TEXT, resolution TEXT, reporter TEXT, issuetype TEXT, components TEXT DEFAULT '[]',
  labels TEXT DEFAULT '[]', fix_versions TEXT DEFAULT '[]', versions TEXT DEFAULT '[]',
  created TEXT, updated TEXT);
CREATE TABLE issue_comments(id TEXT PRIMARY KEY, issue TEXT, author TEXT, at TEXT, body);
CREATE TABLE issue_changes(id TEXT PRIMARY KEY, issue TEXT, at TEXT, history_id INTEGER,
  item_index INTEGER, author TEXT, field TEXT, from_value TEXT, to_value TEXT);
CREATE TABLE issue_prs(issue TEXT, pr INTEGER, PRIMARY KEY(issue, pr));
CREATE TABLE prs(number INTEGER PRIMARY KEY, title TEXT, body, author TEXT, state TEXT,
  base TEXT, head TEXT, created TEXT, merged TEXT, closed TEXT, merged_by TEXT,
  merge_commit TEXT, files TEXT DEFAULT '[]', issues TEXT DEFAULT '[]', kips TEXT DEFAULT '[]',
  updated TEXT);
CREATE TABLE pr_reviews(id TEXT PRIMARY KEY, pr INTEGER, author TEXT, state TEXT, at TEXT, body);
CREATE TABLE pr_review_comments(id TEXT PRIMARY KEY, pr INTEGER, author TEXT, at TEXT,
  path TEXT, line TEXT, body);
CREATE TABLE pr_comments(id TEXT PRIMARY KEY, pr INTEGER, author TEXT, at TEXT, body);
CREATE TABLE commits(sha TEXT PRIMARY KEY, branch TEXT, author TEXT, at TEXT, message,
  files TEXT DEFAULT '[]', prs TEXT DEFAULT '[]', issues TEXT DEFAULT '[]');
CREATE TABLE mails(id TEXT PRIMARY KEY, list TEXT, thread TEXT, subject TEXT, sender TEXT,
  at TEXT, in_reply_to TEXT, message_id TEXT, body);
CREATE TABLE docs(id INTEGER PRIMARY KEY, kind TEXT, ref TEXT, parent TEXT, at TEXT,
  UNIQUE(kind, ref));
CREATE VIRTUAL TABLE fts USING fts5(title, body, content='', detail=none);
CREATE INDEX issue_comments_issue ON issue_comments(issue, at);
CREATE INDEX issue_changes_issue ON issue_changes(issue, at, history_id, item_index);
CREATE INDEX issue_prs_pr ON issue_prs(pr);
CREATE INDEX pr_reviews_pr ON pr_reviews(pr, at);
CREATE INDEX pr_review_comments_pr ON pr_review_comments(pr, at);
CREATE INDEX pr_comments_pr ON pr_comments(pr, at);
CREATE INDEX mails_thread ON mails(thread, at);
CREATE INDEX docs_kind ON docs(kind, at);
"""
TIMED = {  # every column that holds an instant: the audit proves each one is < boundary
    "issues": ("created", "updated"), "issue_comments": ("at",), "issue_changes": ("at",),
    "prs": ("created", "merged", "closed", "updated"), "pr_reviews": ("at",),
    "pr_review_comments": ("at",), "pr_comments": ("at",), "commits": ("at",), "mails": ("at",),
    "docs": ("at",),
}  # fmt: skip


def pack(text):
    text = text or ""
    return text if len(text) < INLINE else zlib.compress(text.encode(), 6)


def unpack(value):
    return zlib.decompress(value).decode() if isinstance(value, bytes) else (value or "")


def parse(text):
    """(title, facts, body) of a `stream.page` text."""
    lines = text.split("\n")
    facts, index = {}, 2
    while index < len(lines) and lines[index].startswith("- "):
        name, _, value = lines[index][2:].partition(": ")
        facts[name] = value
        index += 1
    return lines[0].removeprefix("# "), facts, "\n".join(lines[index + 1 :]).strip("\n")


FILES = re.compile(r"(?:^|\n\n)## Changed files\n\n((?:- `.+`(?:\n|$))+)$")


def split_files(body, heading):
    """(text under `## heading`, changed files) of a push / merged-PR body. Only the trailing
    file list and the leading heading are structure; `##` lines inside the text are content."""
    found = FILES.search(body)
    rest = body[: found.start()] if found else body
    files = re.findall(r"^- `(.+)`$", found.group(1), re.MULTILINE) if found else []
    return rest.removeprefix(f"## {heading}\n\n").strip("\n"), files


def tagged(event, prefix):
    return [e.split(":", 1)[1] for e in event["entities"] if e.startswith(prefix + ":")]


def many(value):
    """Multi-valued changelog sides arrive as lists (fixVersion) or single strings (Component)."""
    return [str(v) for v in value] if isinstance(value, list) else [] if value is None else [str(value)]  # fmt: skip


def clip(field, value):
    if isinstance(value, list):
        value = ", ".join(map(str, value)) or None
    if value is not None and field.lower() in CLIPPED and len(value) > CLIP:
        return value[:CLIP] + f"… [{len(value)} characters]"
    return value


class Index:
    def __init__(self, path: Path):
        fresh = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, isolation_level=None)
        self.db.executescript("PRAGMA journal_mode=DELETE; PRAGMA synchronous=NORMAL;")
        if fresh:
            self.db.executescript(SCHEMA)
            self.meta(schema=SCHEMA_VERSION, events=0)
        self.db.execute("BEGIN")

    # ---- plumbing -------------------------------------------------------------------------
    def meta(self, **values):
        self.db.executemany(
            "INSERT OR REPLACE INTO meta VALUES(?, ?)", [(k, str(v)) for k, v in values.items()]
        )

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def document(self, kind, ref, parent, at, title, body, old=None):
        """(Re)index one searchable text. A contentless FTS row is deleted by value."""
        row = self.db.execute("SELECT id FROM docs WHERE kind = ? AND ref = ?", (kind, ref)).fetchone()  # fmt: skip
        if row and old is not None:
            self.db.execute("INSERT INTO fts(fts, rowid, title, body) VALUES('delete', ?, ?, ?)", (row[0], *old))  # fmt: skip
            self.db.execute("UPDATE docs SET at = ? WHERE id = ?", (at, row[0]))
            number = row[0]
        elif row:
            return
        else:
            number = self.db.execute("INSERT INTO docs(kind, ref, parent, at) VALUES(?, ?, ?, ?)", (kind, ref, parent, at)).lastrowid  # fmt: skip
        self.db.execute("INSERT INTO fts(rowid, title, body) VALUES(?, ?, ?)", (number, title, body))  # fmt: skip

    # ---- JIRA -----------------------------------------------------------------------------
    def issue(self, key, event, summary=None, description=None, **fields):
        self.db.execute("INSERT OR IGNORE INTO issues(key) VALUES(?)", (key,))
        old = self.db.execute("SELECT summary, description FROM issues WHERE key = ?", (key,)).fetchone()  # fmt: skip
        before = (f"{key} {old[0] or ''}", unpack(old[1]))
        fields["updated"] = event["time"]
        fields["components"] = json.dumps(tagged(event, "component"))
        if summary is not None:
            fields["summary"] = summary
        if description is not None:
            fields["description"] = pack(description)
        names = ", ".join(f"{name} = ?" for name in fields)
        self.db.execute(f"UPDATE issues SET {names} WHERE key = ?", (*fields.values(), key))
        after = (
            f"{key} {summary if summary is not None else old[0] or ''}",
            description if description is not None else before[1],
        )
        if after != before or old == (None, None):
            self.document("issue", key, key, event["time"], *after, old=before)

    def jira(self, event):
        key, kind, text = event["meta"]["issue"], event["type"], event["text"]
        title = ""
        if kind == "jira.change" and not text.startswith("# "):  # link events: one plain line
            found = AT.search(text)
            at, summary, facts, body = found.group(1) if found else event["time"], None, {}, ""
        else:
            title, facts, body = parse(text)
            at = facts.get("at") or event["time"]
            shown = facts.get("issue", "")
            summary = shown.split(" — ", 1)[1] if " — " in shown else None
        if kind == "jira.created":
            summary = title.split(" created: ", 1)[1] if " created: " in title else None
            self.issue(
                key, event, summary, body, created=at, reporter=facts.get("reporter"),
                status=facts.get("status"), priority=facts.get("priority"),
                assignee=facts.get("assignee"),
                labels=json.dumps(facts["labels"].split(", ") if facts.get("labels") else []),
            )  # fmt: skip
        elif kind == "jira.comment":
            self.issue(key, event, summary)
            self.db.execute("INSERT OR REPLACE INTO issue_comments VALUES(?, ?, ?, ?, ?)", (event["id"], key, event["actor"], at, pack(body)))  # fmt: skip
            self.document("issue_comment", event["id"], key, event["time"], key, body)
        else:
            meta = event["meta"]
            field, before, after = meta["field"], meta.get("from"), meta.get("to")
            name, state = field.lower(), {}
            if name in SINGLE - {"summary"}:
                state[name] = clip(field, after)
            elif name == "labels":
                state["labels"] = json.dumps(after if isinstance(after, list) else (after or "").split())  # fmt: skip
            elif name in MANY:
                column = MANY[name]
                row = self.db.execute(f"SELECT {column} FROM issues WHERE key = ?", (key,)).fetchone()  # fmt: skip
                values = set(json.loads(row[0] if row else "[]")) - set(many(before))
                state[column] = json.dumps(sorted(values | set(many(after))))
            self.issue(
                key, event, after if name == "summary" else summary,
                body if name == "description" else None, **state,
            )  # fmt: skip
            parts = event["id"].split(":")
            self.db.execute(
                "INSERT OR REPLACE INTO issue_changes VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (event["id"], key, at, int(parts[-2]), int(parts[-1]), event["actor"], field,
                 clip(field, before), clip(field, after)),
            )  # fmt: skip
            for number in tagged(event, "pr"):
                self.db.execute("INSERT OR IGNORE INTO issue_prs VALUES(?, ?)", (key, int(number)))

    # ---- GitHub ---------------------------------------------------------------------------
    def pull(self, event, facts, **fields):
        number = event["meta"]["pr"]
        self.db.execute("INSERT OR IGNORE INTO prs(number, state) VALUES(?, 'OPEN')", (number,))
        fields["updated"] = event["time"]
        if issues := tagged(event, "issue"):
            fields["issues"] = json.dumps(issues)
        if kips := tagged(event, "kip"):
            fields["kips"] = json.dumps(kips)
        shown = facts.get("pull request", "")
        known = self.db.execute("SELECT title FROM prs WHERE number = ?", (number,)).fetchone()[0]
        if " — " in shown and "title" not in fields and known is None:
            fields["title"] = shown.split(" — ", 1)[1]
            self.document("pr", str(number), str(number), event["time"], f"#{number} {fields['title']}", "")  # fmt: skip
        fields = {k: v for k, v in fields.items() if v is not None}
        names = ", ".join(f"{name} = ?" for name in fields)
        self.db.execute(f"UPDATE prs SET {names} WHERE number = ?", (*fields.values(), number))
        for key in issues:
            self.db.execute("INSERT OR IGNORE INTO issue_prs VALUES(?, ?)", (key, number))
        return number

    def github(self, event):
        kind, at = event["type"], event["time"]
        title, facts, body = parse(event["text"])
        if kind == "gh.push":
            (message, files), sha = split_files(body, "Commit message"), event["meta"]["sha"]
            self.db.execute(
                "INSERT OR REPLACE INTO commits VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (sha, event["meta"].get("branch"), event["actor"], at, pack(message),
                 json.dumps(files),
                 json.dumps([int(n) for n in tagged(event, "pr")]),
                 json.dumps(tagged(event, "issue"))),
            )  # fmt: skip
            self.document("commit", sha, sha, at, f"{sha} {message.split(chr(10))[0]}", message)
            return
        if kind in {"gh.pr_opened", "gh.pr_merged", "gh.pr_closed"}:
            name = title.split(": ", 1)[1] if ": " in title else ""
            fields = {"title": name, "author": facts.get("author"), "base": facts.get("base")}
            if kind == "gh.pr_opened":
                fields |= {"created": at, "head": facts.get("head"), "state": "OPEN"}
            elif kind == "gh.pr_merged":
                body, files = split_files(body, "Description")
                fields |= {
                    "state": "MERGED", "merged": at, "closed": at,
                    "merged_by": facts.get("merged by"),
                    "merge_commit": event["meta"].get("merge_commit"),
                    "files": json.dumps(files),
                }  # fmt: skip
            else:
                fields |= {"state": "CLOSED", "closed": at}
            old = self.db.execute("SELECT title, body FROM prs WHERE number = ?", (event["meta"]["pr"],)).fetchone()  # fmt: skip
            number = self.pull(event, facts, body=pack(body), **fields)
            before = (f"#{number} {old[0] or ''}", unpack(old[1])) if old else None
            if before != (f"#{number} {name}", body):
                self.document("pr", str(number), str(number), at, f"#{number} {name}", body, old=before)  # fmt: skip
            return
        number = self.pull(event, facts)
        if kind == "gh.review":
            self.db.execute("INSERT OR REPLACE INTO pr_reviews VALUES(?, ?, ?, ?, ?, ?)", (event["id"], number, event["actor"], event["meta"].get("state"), at, pack(body)))  # fmt: skip
        elif kind == "gh.review_comment":
            self.db.execute("INSERT OR REPLACE INTO pr_review_comments VALUES(?, ?, ?, ?, ?, ?, ?)", (event["id"], number, event["actor"], at, event["meta"].get("path"), facts.get("line"), pack(body)))  # fmt: skip
        else:
            self.db.execute("INSERT OR REPLACE INTO pr_comments VALUES(?, ?, ?, ?, ?)", (event["id"], number, event["actor"], at, pack(body)))  # fmt: skip
        if body:
            label = event["meta"].get("path") or f"#{number}"
            self.document(kind.removeprefix("gh."), event["id"], str(number), at, label, body)

    # ---- mail -----------------------------------------------------------------------------
    def mail(self, event):
        subject, facts, body = parse(event["text"])
        ref = event["id"].removeprefix("mail:")
        self.db.execute(
            "INSERT OR REPLACE INTO mails VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ref, facts.get("list"), facts.get("thread"), subject, facts.get("from"),
             event["time"], facts.get("in reply to"), event["meta"].get("message_id"), pack(body)),
        )  # fmt: skip
        self.document("mail", ref, facts.get("thread"), event["time"], subject, body)

    def add(self, event):
        family = event["type"].split(".")[0]
        {"jira": self.jira, "gh": self.github, "mail": self.mail}[family](event)

    # ---- checkpoints ----------------------------------------------------------------------
    def snapshot(self, checkpoint, events, target: Path):
        self.meta(checkpoint=checkpoint, boundary=stream.boundary(checkpoint), events=events)
        self.db.execute("COMMIT")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.unlink(missing_ok=True)
        self.db.execute("VACUUM INTO ?", (str(temporary),))
        temporary.chmod(0o444)
        target.unlink(missing_ok=True)
        temporary.replace(target)
        self.db.execute("BEGIN")

    def close(self):
        self.db.execute("ROLLBACK")
        self.db.close()


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def counts(path):
    db = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    rows = {t: db.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in TIMED}
    db.close()
    return rows


def save_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class SourcesStore:
    """StoreBuilder for `S-tools` (stores/build.py dispatch). `wanted` limits the snapshots."""

    arm = ARM

    def __init__(self, wanted: Iterable[str] | None = None, first="2023-01", last="2026-06"):
        self.wanted = set(wanted) if wanted else None
        self.checkpoints = stream.checkpoints(first, last)

    def keep(self, checkpoint):
        return self.wanted is None or checkpoint in self.wanted

    def build(self, events: Iterable[dict], out: Path, *, until: str | None = None) -> None:
        out.mkdir(parents=True, exist_ok=True)
        state_path, work = out / "build-state.json", out / "build" / DB
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        ckpts = [c for c in self.checkpoints if not until or c <= until]
        done = state["checkpoints"] if state and work.exists() else []
        exported = lambda c: (out / "export" / f"{c}.manifest.json").exists()  # noqa: E731
        if any(self.keep(c) and not exported(c) for c in done) or not set(done) <= set(ckpts):
            done = []  # a wanted snapshot lies behind the cursor: the only way back is a rebuild
        if not done:
            for stale in (work, work.with_name(DB + "-journal")):
                stale.unlink(missing_ok=True)
            state = {"arm": ARM, "schema": SCHEMA_VERSION, "checkpoints": [], "exports": {}, "usage": {}}  # fmt: skip
        assert state is not None
        index = Index(work)
        consumed, resume_from = int(index.get("events") or 0), (stream.boundary(done[-1]) if done else "")  # fmt: skip
        todo = [c for c in ckpts if c not in done]

        def finish(checkpoint):
            if self.keep(checkpoint):
                target = out / "export" / checkpoint / DB
                index.snapshot(checkpoint, consumed, target)
                state["exports"][checkpoint] = {
                    "arm": ARM, "checkpoint": checkpoint, "boundary": stream.boundary(checkpoint),
                    "events": consumed, "bytes": target.stat().st_size,
                    "sha256": file_sha(target), "rows": counts(target),
                }  # fmt: skip
                save_json(out / "export" / f"{checkpoint}.manifest.json", state["exports"][checkpoint])  # fmt: skip
            else:
                index.meta(checkpoint=checkpoint, events=consumed)
                index.db.execute("COMMIT")
                index.db.execute("BEGIN")
            state["checkpoints"].append(checkpoint)
            save_json(state_path, state)
            print(f"{ARM}: {checkpoint} events={consumed}" + (f" bytes={state['exports'][checkpoint]['bytes']}" if self.keep(checkpoint) else ""), flush=True)  # fmt: skip

        try:
            for event in events:
                if event["time"] < resume_from:
                    continue
                while todo and event["time"] >= stream.boundary(todo[0]):
                    finish(todo.pop(0))
                if not todo:
                    break
                index.add(event)
                consumed += 1
            else:
                if todo and (until is None or todo[0] <= until):
                    # End of stream: the month holding the last event is complete by construction
                    # (smoke slices end mid-month and say so in their manifest).
                    finish(todo.pop(0))
                while until and todo:
                    finish(todo.pop(0))
        finally:
            index.close()
        state["done"] = until is None
        save_json(state_path, state)

    def export(self, out: Path, checkpoint: str) -> Path:
        target = out / "export" / checkpoint
        if not (target / DB).exists():
            raise ValueError(f"{ARM}: no snapshot for {checkpoint}; rebuild with --export {checkpoint}")  # fmt: skip
        return target
