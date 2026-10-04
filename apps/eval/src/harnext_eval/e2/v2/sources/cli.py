"""`gh`, `jira`, `mail`: offline, time-frozen replicas of the project's systems of record.

Thin argparse programs over the checkpoint's SQLite file ($HARNEXT_SOURCES, mounted read-only
in `S-tools` trials). Standard library only: this module runs inside the agent image. The
command and flag names follow the real GitHub CLI and jira-cli so an agent's habits work; what
is not replicated says so and exits 2. Without a database every command prints "not available
in this trial" and exits 2, so every arm has the same PATH but only `S-tools` has data.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import zlib

REPO = "apache/kafka"
URL = f"https://github.com/{REPO}"
TOKEN = re.compile(r"[^\W_]+")  # unicode61: runs of letters and digits
SUPPORTED = {
    "gh": "pr list|view|diff|checks, issue list|view, search prs|commits, api <path>, repo view",
    "jira": "issue view <KEY> [--comments N] [--history], issue list [-q JQL-lite] [filters]",
    "mail": "search <query> [filters], read <id>, thread <id>",
}
JQL = (
    "JQL-lite: clauses joined by AND, optional `ORDER BY created|updated|key [ASC|DESC]`.\n"
    "  project = KAFKA | key = KAFKA-123 | status|assignee|reporter|priority|resolution|type "
    '(= | !=) "value"\n  component = "name" | <field> is [not] EMPTY | resolution = Unresolved\n'
    '  text ~ "words" (summary, description, comments) | summary ~ "words" | description ~ "words"'
    '\n  created|updated (>= | <= | > | <) 2023-01-31   (a \\"quoted phrase\\" inside ~ is exact)'
)
PR_FIELDS = (
    "number", "title", "body", "state", "author", "createdAt", "mergedAt", "closedAt", "updatedAt",
    "baseRefName", "headRefName", "mergeCommit", "mergedBy", "files", "url", "issues", "comments",
    "reviews", "reviewComments",
)  # fmt: skip


class Unavailable(Exception):  # noqa: N818
    pass


def unpack(value):
    return zlib.decompress(value).decode() if isinstance(value, bytes) else (value or "")


class Replica:
    def __init__(self):
        path = os.environ.get("HARNEXT_SOURCES")
        if not path or not os.path.exists(path):
            raise Unavailable
        self.db = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
        self.db.row_factory = sqlite3.Row
        self.boundary = self.one("SELECT value FROM meta WHERE key = 'boundary'")[0]

    def all(self, sql, args=()):
        return self.db.execute(sql, tuple(args)).fetchall()

    def one(self, sql, args=()):
        return self.db.execute(sql, tuple(args)).fetchone()

    def search(self, query, kinds, limit=5000):
        """docs rows (newest first) whose text holds every word of `query`; quoted phrases and
        words with punctuation (KAFKA-123, a.b.C) are re-checked on the text by the caller."""
        tokens = [t for t in TOKEN.findall(query.lower())]
        if not tokens:
            return []
        marks = ", ".join("?" for _ in kinds)
        match = " ".join(f'"{t}"' for t in tokens)
        return self.all(
            f"SELECT d.* FROM fts JOIN docs d ON d.id = fts.rowid WHERE fts MATCH ? "
            f"AND d.kind IN ({marks}) ORDER BY d.at DESC LIMIT ?", (match, *kinds, limit),
        )  # fmt: skip


def exact_parts(query):
    """Substrings that must literally occur: quoted phrases and words with inner punctuation."""
    phrases = re.findall(r'"([^"]+)"', query)
    words = [w for w in re.sub(r'"[^"]*"', " ", query).split() if len(TOKEN.findall(w)) > 1]
    return [p.lower() for p in phrases + words]


def holds(parts, *texts):
    text = " ".join(" ".join(t.lower().split()) for t in texts)
    return all(" ".join(p.split()) in text for p in parts)


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def pick(row, fields, available):
    names = [f.strip() for f in fields.split(",") if f.strip()]
    if unknown := [n for n in names if n not in available]:
        raise SystemExit(f"Unknown JSON field: {unknown[0]!r}\nAvailable fields:\n  " + "\n  ".join(available))  # fmt: skip
    return {name: row[name] for name in names}


class Parser(argparse.ArgumentParser):
    tool = boundary = ""

    def error(self, message):
        self.exit(2, f"{self.prog}: {message}\noffline replica as of {self.boundary}: supported "
                  f"commands are: {SUPPORTED[self.tool]}\n")  # fmt: skip


# --- gh ------------------------------------------------------------------------------------


def pull(replica, row, *, deep=False):
    number = row["number"]
    value = {
        "number": number, "title": row["title"], "body": unpack(row["body"]),
        "state": row["state"], "author": {"login": row["author"]}, "createdAt": row["created"],
        "mergedAt": row["merged"], "closedAt": row["closed"], "updatedAt": row["updated"],
        "baseRefName": row["base"], "headRefName": row["head"],
        "mergeCommit": {"oid": row["merge_commit"]} if row["merge_commit"] else None,
        "mergedBy": {"login": row["merged_by"]} if row["merged_by"] else None,
        "files": [{"path": p} for p in json.loads(row["files"])],
        "url": f"{URL}/pull/{number}", "issues": json.loads(row["issues"]),
    }  # fmt: skip
    if deep:
        value["comments"] = [
            {"author": {"login": r["author"]}, "createdAt": r["at"], "body": unpack(r["body"])}
            for r in replica.all("SELECT * FROM pr_comments WHERE pr = ? ORDER BY at, id", (number,))
        ]  # fmt: skip
        value["reviews"] = [
            {"author": {"login": r["author"]}, "state": (r["state"] or "").upper(),
             "submittedAt": r["at"], "body": unpack(r["body"])}
            for r in replica.all("SELECT * FROM pr_reviews WHERE pr = ? ORDER BY at, id", (number,))
        ]  # fmt: skip
        value["reviewComments"] = [
            {"author": {"login": r["author"]}, "createdAt": r["at"], "path": r["path"],
             "line": r["line"], "body": unpack(r["body"])}
            for r in replica.all("SELECT * FROM pr_review_comments WHERE pr = ? ORDER BY at, id", (number,))
        ]  # fmt: skip
    return value


def pull_row(replica, number):
    number = str(number).lstrip("#").rstrip("/").rsplit("/", 1)[-1]
    row = replica.one("SELECT * FROM prs WHERE number = ?", (number,)) if number.isdigit() else None
    if not row:
        raise SystemExit(
            f"no pull request #{number} in the replica as of {replica.boundary} (it may not exist yet)"
        )
    return row


def pulls(replica, *, state="open", search="", author=None, limit=30):
    words = []
    for word in re.findall(r'"[^"]*"|\S+', search or ""):
        name, _, value = word.partition(":")
        if name in {"is", "state"} and value in {"open", "closed", "merged"}:
            state = value
        elif name == "author" and value:
            author = value
        elif name in {"is", "repo", "type", "in"}:
            continue
        else:
            words.append(word)
    query = " ".join(words)
    where, args = ["1"], []
    if state != "all":
        where.append("state IN (%s)" % ("'CLOSED', 'MERGED'" if state == "closed" else "?"))
        args += [] if state == "closed" else [state.upper()]
    if author:
        where.append("lower(author) = ?")
        args.append(author.lower().lstrip("@"))
    if TOKEN.search(query):
        numbers = {int(d["ref"]) for d in replica.search(query, ["pr"])}
        rows = replica.all(f"SELECT * FROM prs WHERE {' AND '.join(where)} ORDER BY number DESC", args)  # fmt: skip
        parts = exact_parts(query)
        rows = [r for r in rows if r["number"] in numbers
                and holds(parts, f"#{r['number']} {r['title'] or ''}", unpack(r["body"]))]  # fmt: skip
        return rows[:limit]
    return replica.all(f"SELECT * FROM prs WHERE {' AND '.join(where)} ORDER BY number DESC LIMIT ?", (*args, limit))  # fmt: skip


def print_pulls(replica, rows, fields):
    if fields:
        deep = bool({"comments", "reviews", "reviewComments"} & set(fields.split(",")))
        return emit([pick(pull(replica, r, deep=deep), fields, PR_FIELDS) for r in rows])
    for r in rows:
        print(f"{r['number']}\t{r['title'] or ''}\t{r['head'] or ''}\t{r['state']}\t{r['created'] or ''}")  # fmt: skip


def gh_pr_view(replica, args):
    row = pull_row(replica, args.number)
    value = pull(replica, row, deep=True)
    if args.json:
        return emit(pick(value, args.json, PR_FIELDS))
    lines = [
        ("title", value["title"]), ("state", value["state"]), ("author", row["author"]),
        ("number", value["number"]), ("url", value["url"]), ("base", row["base"]),
        ("head", row["head"]), ("created", row["created"]), ("merged", row["merged"]),
        ("merged by", row["merged_by"]), ("merge commit", row["merge_commit"]),
        ("closed", row["closed"] if value["state"] == "CLOSED" else None),
        ("jira issues", ", ".join(value["issues"])),
        ("comments", len(value["comments"])), ("reviews", len(value["reviews"])),
        ("review comments", len(value["reviewComments"])),
        ("changed files", len(value["files"]) or None),
    ]  # fmt: skip
    print("\n".join(f"{k}:\t{v}" for k, v in lines if v not in (None, "")))
    print("--\n" + value["body"])
    if args.comments:
        notes = [(c["createdAt"], f"comment by {c['author']['login']}", c["body"]) for c in value["comments"]]  # fmt: skip
        notes += [(r["submittedAt"], f"review ({r['state']}) by {r['author']['login']}", r["body"]) for r in value["reviews"]]  # fmt: skip
        notes += [(c["createdAt"], f"review comment by {c['author']['login']} on {c['path']}" + (f":{c['line']}" if c["line"] else ""), c["body"]) for c in value["reviewComments"]]  # fmt: skip
        for at, head, body in sorted(notes, key=lambda n: n[0]):
            print(f"\n--- {at} {head}\n{body}".rstrip())


def gh_pr_diff(replica, args):
    row = pull_row(replica, args.number)
    if row["state"] != "MERGED":
        raise SystemExit(f"PR #{row['number']} is {row['state']} as of {replica.boundary}: the offline replica holds no diff for unmerged pull requests")  # fmt: skip
    print(f"# The replica stores no patch text. PR #{row['number']} was merged as commit {row['merge_commit']}.")  # fmt: skip
    print(f"# If that commit is in the workspace history, run: git show {row['merge_commit']}")
    print("# Changed files:")
    for path in json.loads(row["files"]):
        print(path)


def gh_search(replica, args):
    query = " ".join(args.query)
    if args.what == "prs":
        return print_pulls(replica, pulls(replica, state=args.state or "all", search=query, author=args.author, limit=args.limit), args.json)  # fmt: skip
    rows, parts = [], exact_parts(query)
    for doc in replica.search(query, ["commit"]):
        r = replica.one("SELECT * FROM commits WHERE sha = ?", (doc["ref"],))
        if holds(parts, r["sha"], unpack(r["message"])):
            rows.append(r)
    rows = rows[: args.limit]
    if args.json:
        return emit([commit(r) for r in rows])
    for r in rows:
        print(f"{r['sha']}\t{unpack(r['message']).splitlines()[0] if r['message'] else ''}\t{r['author'] or ''}\t{r['at']}")  # fmt: skip


def commit(row):
    message = unpack(row["message"])
    return {
        "sha": row["sha"], "commit": {"message": message, "author": {"name": row["author"], "date": row["at"]}},
        "branch": row["branch"], "files": [{"filename": p} for p in json.loads(row["files"])],
        "pulls": json.loads(row["prs"]), "issues": json.loads(row["issues"]),
    }  # fmt: skip


def rest_pull(value):
    merged = value["state"] == "MERGED"
    return {
        "number": value["number"], "title": value["title"], "body": value["body"],
        "state": "open" if value["state"] == "OPEN" else "closed", "merged": merged,
        "user": {"login": value["author"]["login"]}, "created_at": value["createdAt"],
        "merged_at": value["mergedAt"], "closed_at": value["closedAt"],
        "merge_commit_sha": (value["mergeCommit"] or {}).get("oid"),
        "base": {"ref": value["baseRefName"]}, "head": {"ref": value["headRefName"]},
        "html_url": value["url"],
    }  # fmt: skip


def gh_api(replica, args):
    path, _, query = args.path.strip("/").partition("?")
    params = dict(p.partition("=")[::2] for p in query.split("&") if p)
    root = f"repos/{REPO}"
    if path == root:
        return emit({"full_name": REPO, "html_url": URL, "default_branch": "trunk", "replica_as_of": replica.boundary})  # fmt: skip
    if path == f"{root}/pulls":
        rows = pulls(replica, state=params.get("state", "open"), limit=int(params.get("per_page", 30)))  # fmt: skip
        return emit([rest_pull(pull(replica, r)) for r in rows])
    if found := re.fullmatch(rf"{root}/commits/([0-9a-f]{{7,40}})", path):
        rows = replica.all("SELECT * FROM commits WHERE sha LIKE ?", (found.group(1) + "%",))
        if len(rows) != 1:
            raise SystemExit(f"gh: commit {found.group(1)} is not in the replica (HTTP 404)")
        return emit(commit(rows[0]))
    found = re.fullmatch(rf"{root}/(pulls|issues)/(\d+)(/comments|/reviews|/files)?", path)
    if not found or (found.group(1) == "issues" and found.group(3) != "/comments"):
        raise SystemExit(
            f"gh: endpoint not in the offline replica (HTTP 404): {path}\nsupported: {root}, "
            f"{root}/pulls[?state=], {root}/pulls/<n>[/comments|/reviews|/files], "
            f"{root}/issues/<n>/comments, {root}/commits/<sha>"
        )
    value = pull(replica, pull_row(replica, found.group(2)), deep=True)
    kind, part = found.group(1), found.group(3)
    if part == "/files":
        return emit([{"filename": f["path"]} for f in value["files"]])
    if part == "/reviews":
        return emit([{"user": {"login": r["author"]["login"]}, "state": r["state"], "submitted_at": r["submittedAt"], "body": r["body"]} for r in value["reviews"]])  # fmt: skip
    if part == "/comments":
        rows = value["comments"] if kind == "issues" else value["reviewComments"]
        return emit([{"user": {"login": c["author"]["login"]}, "created_at": c["createdAt"], "body": c["body"]} | ({"path": c["path"], "line": c["line"]} if "path" in c else {}) for c in rows])  # fmt: skip
    emit(rest_pull(value))


def gh(replica, argv):
    parser = Parser(prog="gh", description=f"Offline, time-frozen replica of {REPO} on GitHub "
                    f"(data as of {replica.boundary}). Read-only; supported: {SUPPORTED['gh']}.")  # fmt: skip
    groups = parser.add_subparsers(dest="group", required=True, parser_class=Parser)
    pr = groups.add_parser("pr", help="pull requests").add_subparsers(dest="command", required=True, parser_class=Parser)  # fmt: skip
    listing = pr.add_parser("list", help="list pull requests (default: open)")
    listing.add_argument("-s", "--state", choices=("open", "closed", "merged", "all"), default="open")  # fmt: skip
    listing.add_argument("-S", "--search", default="", help='words, "phrases", author:x, is:merged')
    listing.add_argument("-A", "--author")
    listing.add_argument("-L", "--limit", type=int, default=30)
    listing.add_argument("--json", metavar="fields", help=",".join(PR_FIELDS))
    view = pr.add_parser("view", help="one pull request, optionally with its discussion")
    view.add_argument("number")
    view.add_argument("-c", "--comments", action="store_true", help="comments, reviews, review comments")  # fmt: skip
    view.add_argument("--json", metavar="fields", help=",".join(PR_FIELDS))
    pr.add_parser("diff", help="changed files + merge commit of a merged PR").add_argument("number")
    pr.add_parser("checks", help="not available offline").add_argument("number", nargs="?")
    issue = groups.add_parser("issue", help="Kafka tracks issues in JIRA: use `jira`").add_subparsers(dest="command", required=True, parser_class=Parser)  # fmt: skip
    for name in ("list", "view"):
        issue.add_parser(name).add_argument("rest", nargs="*")
    search = groups.add_parser("search", help="search prs|commits")
    search.add_argument("what", choices=("prs", "commits"))
    search.add_argument("query", nargs="+")
    search.add_argument("--state", choices=("open", "closed", "merged"))
    search.add_argument("--author")
    search.add_argument("-L", "--limit", type=int, default=30)
    search.add_argument("--json", metavar="fields")
    search.add_argument("--repo", "-R", help="ignored: the replica holds apache/kafka only")
    api = groups.add_parser("api", help=f"GET repos/{REPO}[/pulls[/<n>[/comments|/reviews|/files]]|/issues/<n>/comments|/commits/<sha>]")  # fmt: skip
    api.add_argument("path")
    api.add_argument("--paginate", action="store_true", help="accepted; responses are complete")
    api.add_argument("-X", "--method", default="GET")
    repo = groups.add_parser("repo", help="repo view").add_subparsers(dest="command", required=True, parser_class=Parser)  # fmt: skip
    repo.add_parser("view").add_argument("name", nargs="?")
    args = parser.parse_args(argv)
    if args.group == "pr" and args.command == "list":
        print_pulls(replica, pulls(replica, state=args.state, search=args.search, author=args.author, limit=args.limit), args.json)  # fmt: skip
    elif args.group == "pr" and args.command == "view":
        gh_pr_view(replica, args)
    elif args.group == "pr" and args.command == "diff":
        gh_pr_diff(replica, args)
    elif args.group == "pr":
        raise SystemExit("gh pr checks: not available offline (CI results are not part of the replica)")  # fmt: skip
    elif args.group == "issue":
        raise SystemExit("apache/kafka has GitHub issues disabled: the project tracks issues in JIRA. Use `jira issue view KAFKA-<n>` / `jira issue list --help`.")  # fmt: skip
    elif args.group == "search":
        gh_search(replica, args)
    elif args.group == "api":
        if args.method.upper() != "GET":
            raise SystemExit("gh api: the replica is read-only (GET only)")
        gh_api(replica, args)
    else:
        rows = replica.one("SELECT count(*), sum(state = 'OPEN') FROM prs")
        print(f"name:\t{REPO}\ndescription:\tMirror of Apache Kafka (offline replica, data as of {replica.boundary})\ndefault branch:\ttrunk\npull requests:\t{rows[0]} ({rows[1] or 0} open)\nissues:\tdisabled (JIRA project KAFKA: use `jira`)")  # fmt: skip


# --- jira ----------------------------------------------------------------------------------

CLAUSE = re.compile(
    r"""\s*(?P<field>[A-Za-z]+)\s*(?:(?P<op>!=|>=|<=|=|~|>|<)\s*(?P<value>"(?:[^"\\]|\\.)*"|'[^']*'|[^\s"']+)
    |(?P<empty>is\s+(?:not\s+)?(?:empty|null)))\s*$""",
    re.IGNORECASE | re.VERBOSE,
)
COLUMNS = {
    "status": "status", "assignee": "assignee", "reporter": "reporter", "priority": "priority",
    "resolution": "resolution", "type": "issuetype", "issuetype": "issuetype",
}  # fmt: skip


class JqlError(ValueError):  # noqa: N818
    pass


def jql(text):
    """(where, args, text filters [(scope, words)], order by). Raises JqlError with the grammar."""
    order = "created DESC"
    body, found = (
        text,
        re.search(r"\border\s+by\s+(\w+)(?:\s+(asc|desc))?\s*$", text, re.IGNORECASE),
    )
    if found:
        column = {"created": "created", "updated": "updated", "key": "CAST(substr(key, 7) AS INTEGER)"}.get(found.group(1).lower())  # fmt: skip
        if not column:
            raise JqlError(f"cannot order by {found.group(1)!r}")
        order, body = f"{column} {(found.group(2) or 'ASC').upper()}", text[: found.start()]
    where, args, texts = [], [], []
    for clause in re.split(r"\s+AND\s+", body.strip(), flags=re.IGNORECASE) if body.strip() else []:
        found = CLAUSE.match(clause)
        if not found:
            raise JqlError(f"unsupported clause: {clause.strip()!r}")
        field, op = found.group("field").lower(), found.group("op")
        value = found.group("value") or ""
        if value[:1] in "\"'" and value[:1] == value[-1:]:
            value = value[1:-1].replace('\\"', '"')
        if found.group("empty"):
            column = COLUMNS.get(field) or {"component": "components"}.get(field)
            if not column:
                raise JqlError(f"`is EMPTY` is not supported for {field!r}")
            negated = "not" in found.group("empty").lower()
            blank = f"({column} IS NULL OR {column} IN ('', '[]'))"
            where.append(f"NOT {blank}" if negated else blank)
        elif field == "project" and op == "=":
            if value.upper() != "KAFKA":
                raise JqlError("the replica holds project KAFKA only")
        elif field in {"key", "issuekey", "issue"} and op == "=":
            where.append("key = ?")
            args.append(value.upper())
        elif field in COLUMNS and op in {"=", "!="}:
            column = COLUMNS[field]
            if field == "resolution" and value.lower() == "unresolved" or value.lower() in {"empty", "null"}:  # fmt: skip
                where.append(f"{column} IS {'NOT ' if op == '!=' else ''}NULL")
            else:
                where.append(f"({column} IS NULL OR lower({column}) != ?)" if op == "!=" else f"lower({column}) = ?")  # fmt: skip
                args.append(value.lower())
        elif field == "component" and op == "=":
            where.append("EXISTS (SELECT 1 FROM json_each(issues.components) WHERE lower(value) = ?)")  # fmt: skip
            args.append(value.lower())
        elif field in {"text", "summary", "description"} and op == "~":
            texts.append((field, value))
        elif field in {"created", "updated"} and op in {">=", "<=", ">", "<"}:
            if not re.fullmatch(r"\d{4}-\d\d-\d\d(T[\d:.]+Z?)?", value):
                raise JqlError(f"dates look like 2023-01-31, not {value!r}")
            # `<= 2023-01-31` means through the end of that day, as in JIRA's day granularity
            bound = value + "T99" if op in {"<=", ">"} and "T" not in value else value
            where.append(f"{field} {op} ?")
            args.append(bound)
        else:
            raise JqlError(f"unsupported clause: {clause.strip()!r}")
    return where, args, texts, order


def issue(replica, row, comments=0, history=False):
    value = {
        k: row[k] for k in ("key", "summary", "status", "priority", "assignee", "resolution",
                            "reporter", "issuetype", "created", "updated")
    } | {
        "components": json.loads(row["components"]), "labels": json.loads(row["labels"]),
        "fixVersions": json.loads(row["fix_versions"]),
        "description": unpack(row["description"]),
        "pullRequests": [r[0] for r in replica.all("SELECT pr FROM issue_prs WHERE issue = ? ORDER BY pr", (row["key"],))],
    }  # fmt: skip
    total = replica.one("SELECT count(*) FROM issue_comments WHERE issue = ?", (row["key"],))[0]
    value["commentCount"] = total
    if comments:
        rows = replica.all("SELECT * FROM (SELECT * FROM issue_comments WHERE issue = ? ORDER BY at DESC, id DESC LIMIT ?) ORDER BY at, id", (row["key"], comments))  # fmt: skip
        value["comments"] = [{"author": r["author"], "created": r["at"], "body": unpack(r["body"])} for r in rows]  # fmt: skip
    if history:
        rows = replica.all("SELECT * FROM issue_changes WHERE issue = ? ORDER BY at, history_id, item_index", (row["key"],))  # fmt: skip
        value["history"] = [
            {"at": r["at"], "author": r["author"], "field": r["field"], "from": r["from_value"],
             "to": r["to_value"], "historyId": r["history_id"], "itemIndex": r["item_index"]}
            for r in rows
        ]  # fmt: skip
    return value


def shown(value):
    return "(none)" if value in (None, "", []) else ", ".join(value) if isinstance(value, list) else str(value)  # fmt: skip


def jira_view(replica, args):
    row = replica.one("SELECT * FROM issues WHERE key = ?", (args.key.upper(),))
    if not row:
        raise SystemExit(f"jira: {args.key.upper()} is not in the replica as of {replica.boundary} (it may not exist yet, or nothing about it was recorded)")  # fmt: skip
    value = issue(replica, row, args.comments, args.history)
    if args.raw:
        return emit(value)
    print(f"{value['key']}: {value['summary'] or '(summary not recorded)'}")
    for name in ("status", "resolution", "priority", "assignee", "reporter", "issuetype", "components", "labels", "fixVersions", "created", "updated"):  # fmt: skip
        print(f"{name}:\t{shown(value[name])}")
    print(f"pull requests:\t{shown([f'#{n}' for n in value['pullRequests']])}")
    print(f"comments:\t{value['commentCount']}")
    if row["created"] is None:
        print("note:\tcreated before the replica's first recorded event; fields never changed since show (none)")  # fmt: skip
    print("\n--- description\n" + value["description"])
    for c in value.get("comments", []):
        print(f"\n--- comment {c['created']} {c['author']}\n{c['body']}")
    if args.history:
        print("\n--- history (timestamp, author, field: from → to)")
        for h in value["history"]:
            print(f"{h['at']}\t{h['author'] or '(unknown)'}\t{h['field']}: {shown(h['from'])} → {shown(h['to'])}")  # fmt: skip


def jira_list(replica, args):
    try:
        where, values, texts, order = jql(args.jql or "")
    except JqlError as exc:
        raise SystemExit(f"jira: {exc}\n{JQL}") from None
    simple = (("status", args.status), ("assignee", args.assignee), ("priority", args.priority),
              ("reporter", args.reporter), ("resolution", args.resolution), ("issuetype", args.type))  # fmt: skip
    for column, wanted in simple:
        if wanted:
            where.append(f"lower({column}) = ?")
            values.append(wanted.lower())
    if args.component:
        where.append("EXISTS (SELECT 1 FROM json_each(issues.components) WHERE lower(value) = ?)")
        values.append(args.component.lower())
    for column, op, wanted in (("created", ">=", args.created_after), ("created", "<", args.created_before), ("updated", ">=", args.updated_after)):  # fmt: skip
        if wanted:
            where.append(f"{column} {op} ?")
            values.append(wanted)
    texts += [("text", t) for t in (args.text, " ".join(args.words)) if t]
    texts += [("summary", args.summary)] if args.summary else []
    rows = replica.all(f"SELECT * FROM issues WHERE {' AND '.join(where) or '1'} ORDER BY {order}, key", values)  # fmt: skip
    for scope, words in texts:
        parts = exact_parts(words)
        if scope == "summary":
            tokens = TOKEN.findall(words.lower())
            rows = [r for r in rows if all(t in (r["summary"] or "").lower() for t in tokens) and holds(parts, r["summary"] or "")]  # fmt: skip
            continue
        kinds = ["issue"] if scope == "description" else ["issue", "issue_comment"]
        hits = {d["parent"] for d in replica.search(words, kinds, limit=100000)}
        rows = [r for r in rows if r["key"] in hits]
        if parts:  # exact substrings: re-read the texts of the (few) remaining candidates
            rows = [r for r in rows if holds(parts, r["key"], r["summary"] or "", unpack(r["description"]), *(
                [unpack(c[0]) for c in replica.all("SELECT body FROM issue_comments WHERE issue = ?", (r["key"],))]
                if scope == "text" else []))]  # fmt: skip
    rows = rows[: args.limit]
    if args.raw:
        return emit([issue(replica, r) | {"description": None} for r in rows])
    for r in rows:
        print(f"{r['key']}\t{r['status'] or ''}\t{r['assignee'] or ''}\t{r['created'] or ''}\t{r['summary'] or ''}")  # fmt: skip


def jira(replica, argv):
    parser = Parser(prog="jira", description=f"Offline, time-frozen replica of the Apache JIRA "
                    f"project KAFKA (data as of {replica.boundary}). Supported: {SUPPORTED['jira']}.")  # fmt: skip
    groups = parser.add_subparsers(dest="group", required=True, parser_class=Parser)
    commands = groups.add_parser("issue", help="view | list").add_subparsers(dest="command", required=True, parser_class=Parser)  # fmt: skip
    view = commands.add_parser("view", help="one issue as of the cutoff: fields, description, comments, change history")  # fmt: skip
    view.add_argument("key")
    view.add_argument("--comments", type=int, default=0, metavar="N", help="show the latest N comments")  # fmt: skip
    view.add_argument("--history", action="store_true", help="every changelog item: full-precision UTC timestamp, author, field, from → to")  # fmt: skip
    view.add_argument("--raw", "--json", action="store_true", dest="raw", help="JSON output")
    listing = commands.add_parser("list", help="search issues", epilog=JQL, formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    listing.add_argument("words", nargs="*", help="free text (same as --text)")
    listing.add_argument("-q", "--jql", help="JQL-lite, see below")
    listing.add_argument("-s", "--status")
    listing.add_argument("-a", "--assignee")
    listing.add_argument("-r", "--reporter")
    listing.add_argument("-y", "--priority")
    listing.add_argument("-R", "--resolution")
    listing.add_argument("-t", "--type")
    listing.add_argument("-C", "--component")
    listing.add_argument("--created-after", metavar="DATE")
    listing.add_argument("--created-before", metavar="DATE")
    listing.add_argument("--updated-after", metavar="DATE")
    listing.add_argument("--text", help="words in summary, description or comments")
    listing.add_argument("--summary", help='words in the summary; "\\"quoted\\"" = exact phrase')
    listing.add_argument("--limit", "--paginate", type=int, default=50, dest="limit")
    listing.add_argument("--raw", "--json", action="store_true", dest="raw", help="JSON output")
    listing.add_argument("--plain", action="store_true", help="accepted; output is always plain")
    args = parser.parse_args(argv)
    (jira_view if args.command == "view" else jira_list)(replica, args)


# --- mail ----------------------------------------------------------------------------------


def message(row, body=True):
    print(f"id:\t{row['id']}\nlist:\t{row['list']}\nfrom:\t{row['sender']}\ndate:\t{row['at']}\nsubject:\t{row['subject']}\nthread:\t{row['thread']}" + (f"\nin-reply-to:\t{row['in_reply_to']}" if row["in_reply_to"] else ""))  # fmt: skip
    if body:
        print("\n" + unpack(row["body"]))


def mail_row(replica, ref):
    ref = ref.removeprefix("mail:").strip("<>")
    row = replica.one("SELECT * FROM mails WHERE id = ? OR message_id = ?", (ref, ref))
    if not row:
        raise SystemExit(f"mail: no message {ref!r} in the replica as of {replica.boundary}")
    return row


def mail(replica, argv):
    parser = Parser(prog="mail", description=f"Offline, time-frozen archive of the Apache Kafka "
                    f"mailing lists (data as of {replica.boundary}). Supported: {SUPPORTED['mail']}.")  # fmt: skip
    commands = parser.add_subparsers(dest="command", required=True, parser_class=Parser)
    search = commands.add_parser("search", help="full-text search, newest first")
    search.add_argument("query", help='words and "exact phrases"; all must occur')
    search.add_argument("--list", help="dev | users | jira | commits (or a full list address)")
    search.add_argument("--from", dest="sender", help="substring of the sender")
    search.add_argument("--after", metavar="DATE")
    search.add_argument("--before", metavar="DATE")
    search.add_argument("--limit", type=int, default=20)
    commands.add_parser("read", help="one message with its body").add_argument("id")
    thread = commands.add_parser("thread", help="every message of the thread holding <id>")
    thread.add_argument("id")
    thread.add_argument("--full", action="store_true", help="print the bodies too")
    args = parser.parse_args(argv)
    if args.command == "read":
        return message(mail_row(replica, args.id))
    if args.command == "thread":
        first = replica.one("SELECT thread FROM mails WHERE thread = ? LIMIT 1", (args.id,))
        name = first[0] if first else mail_row(replica, args.id)["thread"]
        for row in replica.all("SELECT * FROM mails WHERE thread = ? ORDER BY at, id", (name,)):
            if args.full:
                message(row)
                print("\n" + "=" * 72)
            else:
                print(f"{row['id']}\t{row['at']}\t{row['sender']}\t{row['subject']}")
        return None
    parts, shown_rows = exact_parts(args.query), 0
    for doc in replica.search(args.query, ["mail"], limit=100000):
        row = replica.one("SELECT * FROM mails WHERE id = ?", (doc["ref"],))
        wanted = args.list and (args.list if "@" in args.list else args.list + "@")
        if (wanted and not (row["list"] or "").startswith(wanted)) or (
            args.sender and args.sender.lower() not in (row["sender"] or "").lower()
        ) or (args.after and row["at"] < args.after) or (args.before and row["at"] >= args.before):  # fmt: skip
            continue
        if parts and not holds(parts, row["subject"] or "", unpack(row["body"])):
            continue
        print(f"{row['id']}\t{row['at']}\t{row['sender']}\t{row['subject']}")
        shown_rows += 1
        if shown_rows >= args.limit:
            break
    return None


TOOLS = {"gh": gh, "jira": jira, "mail": mail}
WHAT = {"gh": "GitHub pull requests", "jira": "JIRA project KAFKA", "mail": "mailing lists"}


def main(tool, argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        replica = Replica()
    except (Unavailable, sqlite3.Error, TypeError):
        print(f"{tool}: not available in this trial (no offline replica of the project's systems of record is provided)", file=sys.stderr)  # fmt: skip
        return 2
    Parser.tool, Parser.boundary = tool, replica.boundary
    try:
        if argv[:1] in (["--version"], ["version"]):
            print(f"{tool} version 0.0.0-offline-replica (time-frozen replica of {REPO}'s {WHAT[tool]}, data as of {replica.boundary})")  # fmt: skip
        else:
            TOOLS[tool](replica, argv)
        code = 0
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BrokenPipeError:
        code = 0
    print(f"# offline replica, data as of {replica.boundary}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2:]))
