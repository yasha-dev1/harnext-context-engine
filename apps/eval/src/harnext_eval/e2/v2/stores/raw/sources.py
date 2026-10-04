"""Sources of the raw store: which table an event lands in, the fields extracted from it, and
the static knowledge the catalog prints (meanings, joins, worked queries).

Fields come from the event's `meta`, from the `- name: value` fact lines of its rendered
`text` (parsed as the S-tools replica parses them, `sources.index.parse`, except that a title
spanning lines does not hide the facts: `page`), from its id and entities, and — for texts
people write (mail, comments, reviews) — the JIRA keys and KIP ids named in the body. A fact or meta key without a mapping here becomes a column under its
snake_cased name, so a new field of a known source, or a whole new source, is stored without
code changes; the catalog then describes it from its name and examples.
"""

from __future__ import annotations

import re

from harnext_eval.e2.benchmark import canonical
from harnext_eval.e2.v2.sources.index import AT, parse, split_files
from harnext_eval.e2.v2.stores.raw.schema import snake

TABLES = {  # event type -> table; an unknown type gets the table snake(type)
    "jira.created": "jira_issues", "jira.change": "jira_changes", "jira.comment": "jira_comments",
    "gh.pr_opened": "github_pull_requests", "gh.pr_merged": "github_pull_requests",
    "gh.pr_closed": "github_pull_requests", "gh.review": "github_reviews",
    "gh.review_comment": "github_review_comments", "gh.issue_comment": "github_pr_comments",
    "gh.push": "github_commits", "mail.message": "mail_messages",
}  # fmt: skip
SYSTEMS = {"jira": "JIRA (issues.apache.org)", "gh": "GitHub (apache/kafka)", "github": "GitHub (apache/kafka)", "mail": "mailing lists (lists.apache.org)"}  # fmt: skip
# Fact lines: rendered name -> column; None = the same value is already a column (`event` is
# the event type in words; `at` of GitHub/mail events equals `time`, checked for w2026/smoke).
FACTS = {
    "jira": {"event": None, "issue": None, "by": "author", "field": None, "from": None, "to": None,
             "at": "at", "reporter": "reporter", "status": "status", "priority": "priority",
             "assignee": "assignee"},
    "gh": {"event": None, "at": None, "pull request": None, "by": "author", "issues": None,
           "KIPs": None, "pull requests": None, "commit": None, "merge commit": None,
           "file": None, "merged by": "merged_by", "closed by": "closed_by", "author": "author",
           "base": "base", "head": "head", "state": None, "branch": None},
    "mail": {"event": None, "at": None, "from": "sender", "thread": "thread_id", "in reply to": "in_reply_to"},
}  # fmt: skip
REVIEWER = {"by": "reviewer"}  # gh.review: the one who reviewed
PULL = re.compile(r"Pull Request #(\d+)")
RELATION = re.compile(r'^This issue (.+?) ("(.*)"|([A-Z][A-Z0-9]*-\d+))$', re.DOTALL)
ACTION = {"gh.pr_opened": "opened", "gh.pr_merged": "merged", "gh.pr_closed": "closed"}
FACTS_START = re.compile(r"\n\n(?=- (?:issue|event|pull request): )")
ISSUE_ID = re.compile(r"\bKAFKA-\d+\b", re.IGNORECASE)
KIP_ID = re.compile(r"\bKIP-\d+\b", re.IGNORECASE)


def page(text: str):
    """(title, facts, body) of a rendered page. Like `sources.index.parse`, but the facts are
    found after the title even when the title spans lines (a changelog value with line breaks:
    6 w2026 events lost `at`, `author`, `issue_summary` to that)."""
    found = FACTS_START.search(text)
    if not text.startswith("# ") or not found:
        return parse(text)
    lines, facts, index = text[found.start() + 2 :].split("\n"), {}, 0
    while index < len(lines) and lines[index].startswith("- "):
        name, _, value = lines[index][2:].partition(": ")
        facts[name] = value
        index += 1
    return text[2 : found.start()], facts, "\n".join(lines[index + 1 :]).strip("\n")


def mentioned(body: str | None) -> dict:
    """Distinct JIRA keys and KIP ids named in a text, in order of first appearance."""
    ids = lambda pattern: list(dict.fromkeys(m.upper() for m in pattern.findall(body or "")))  # noqa: E731
    return {"body_issues": ids(ISSUE_ID), "body_kips": ids(KIP_ID)}


def table_of(event: dict) -> str:
    """Exact type mapping; any other type gets its own table `other_<type>` (never a known one)."""
    return TABLES.get(event["type"]) or f"other_{snake(event['type'])}"


def tagged(event, prefix):
    return [e.split(":", 1)[1] for e in event.get("entities") or [] if e.startswith(prefix + ":")]


def joined(value):
    """Changelog sides are strings, null or (fixVersion) lists; lists are kept comma-joined."""
    if isinstance(value, list):
        return ", ".join(map(str, value)) or None
    return value


def generic(event, facts, names, known=()):
    """Fields without a fixed column: mapped fact lines under their column, other fact lines as
    "fact:<name>", meta keys as "meta:<key>" (`Schema.column` names those). A fact line or meta
    key that is the rendering of an extracted field (its name is in `known`) and a fact line
    that renders a meta key of the same name are not stored twice."""
    meta = event.get("meta") or {}
    rendered = {snake(k) for k in meta}
    row = {}
    for name, value in facts.items():
        column = names.get(name, "")
        if column is None or (column or snake(name)) in known or (not column and snake(name) in rendered):  # fmt: skip
            continue
        row[column or f"fact:{name}"] = value
    for key, value in meta.items():
        if snake(key) not in known:
            row[f"meta:{key}"] = value
    return row


def jira(event):
    meta, kind, text = event["meta"], event["type"], event["text"]
    parts = event["id"].split(":")
    row = {
        "issue_key": meta.get("issue"),
        "issue_id": int(parts[1]) if parts[1].isdigit() else None,
    }
    if kind == "jira.change" and not text.startswith("# "):  # a link: one plain line
        found, target = AT.search(text), meta.get("to") if meta.get("to") is not None else meta.get("from")  # fmt: skip
        relation = RELATION.match(target or "")
        pulls = [int(n) for n in PULL.findall(target or "")]
        row |= {"at": found.group(1) if found else None, "field": meta.get("field"),
                "from_value": joined(meta.get("from")), "to_value": joined(meta.get("to")),
                "link_action": "added" if meta.get("to") is not None else "removed",
                "link_relation": relation.group(1) if relation else None,
                "linked_issue": relation.group(4) if relation else None,
                "linked_pr": pulls[0] if pulls else None, "components": tagged(event, "component"),
                "history_id": int(parts[3]), "item_index": int(parts[4])}  # fmt: skip
        return row | generic(event, {}, {}, set(row) | {"issue", "from", "to"})
    title, facts, body = page(text)
    shown = facts.get("issue", "")
    known = {"issue_key", "issue_id", "issue", "from", "to"}
    if kind == "jira.created":
        row |= {"summary": title.split(" created: ", 1)[1] if " created: " in title else None,
                "components": tagged(event, "component"),
                "labels": facts["labels"].split(", ") if facts.get("labels") else [],
                "description": body}  # fmt: skip
        facts = {k: v for k, v in facts.items() if k not in ("components", "labels")}
    else:
        row["issue_summary"] = shown.split(" — ", 1)[1] if " — " in shown else None
        row["components"] = tagged(event, "component")
        if kind == "jira.comment":
            row |= {"comment_id": int(parts[3]), "body": body, **mentioned(body)}
        else:
            row |= {"field": meta.get("field"), "from_value": joined(meta.get("from")),
                    "to_value": joined(meta.get("to")),
                    "body": (body or None) if str(meta.get("field")).lower() == "description" else None,
                    "history_id": int(parts[3]), "item_index": int(parts[4])}  # fmt: skip
    return row | generic(event, facts, FACTS["jira"], set(row) | known)


def github(event):
    meta, kind = event["meta"], event["type"]
    title, facts, body = page(event["text"])
    shown = facts.get("pull request", "")
    refs = {"issues": tagged(event, "issue"), "kips": tagged(event, "kip")}
    names = FACTS["gh"] | (REVIEWER if kind == "gh.review" else {})
    if kind == "gh.push":
        message, files = split_files(body, "Commit message")
        row = {"sha": meta.get("sha"), "branch": meta.get("branch"),
               "headline": message.split("\n", 1)[0], "message": message, "changed_files": files,
               "prs": [int(n) for n in tagged(event, "pr")], **refs}  # fmt: skip
    elif kind in ACTION:
        row = {"pr_number": meta.get("pr"), "action": ACTION[kind],
               "pr_title": title.split(": ", 1)[1] if ": " in title else None}  # fmt: skip
        if kind == "gh.pr_merged":
            body, files = split_files(body, "Description")
            row |= {"merge_commit": meta.get("merge_commit"), "changed_files": files}
        row |= {"body": body, **refs}
    else:
        tail = event["id"].rsplit(":", 1)[-1]
        row = {"pr_number": meta.get("pr"),
               "pr_title": shown.split(" — ", 1)[1] if " — " in shown else None,
               ("review_id" if kind == "gh.review" else "comment_id"): tail.split("-")[-1]}  # fmt: skip
        if kind == "gh.review_comment":
            row["path"] = meta.get("path")
            line = facts.pop("line", None)
            row["line"] = int(line) if line and line.isdigit() else line
        row |= {"body": body, **refs, **mentioned(body)}
    return row | generic(event, facts, names, set(row) | {"pr"})


def mail(event):
    title, facts, body = page(event["text"])
    row = {"message_id": event["meta"].get("message_id"), "list": event["meta"].get("list"),
           "subject": title, "body": body, "issues": tagged(event, "issue"),
           "kips": tagged(event, "kip"), **mentioned(body)}  # fmt: skip
    return row | generic(event, facts, FACTS["mail"], set(row))


def fields(event: dict) -> dict:
    """Extracted fields of one event (the base columns are added by `row`)."""
    family = event["type"].split(".")[0]
    if family == "jira" and event["type"] in TABLES:
        return jira(event)
    if family == "gh" and event["type"] in TABLES:
        return github(event)
    if event["type"] == "mail.message":
        return mail(event)
    text = event.get("text") or ""
    facts = page(text)[1] if text.startswith("# ") else {}
    return generic(event, facts, {})


def row(event: dict) -> dict:
    base = {"event_id": event["id"], "event_type": event["type"], "time": event["time"],
            "actor": event.get("actor"), "title": event.get("title") or "",
            "text": event.get("text") or "", "entities": list(event.get("entities") or []),
            "raw": canonical(event)}  # fmt: skip
    return base | {k: v for k, v in fields(event).items() if k not in base}


# ---- static knowledge for the catalog ---------------------------------------------------------

BASE = {
    "event_id": "Unique id of the source event; `get(event_id)` returns the full record.",
    "event_type": "Kind of source event, e.g. jira.change, gh.review, mail.message.",
    "time": "When the event happened: DateTime64(6, 'UTC'), whole seconds; literals '2026-03-01 00:00:00' and '2026-03-01T00:00:00Z' both compare with it. Every row lies before the export's cut-off instant.",
    "actor": "Who caused the event (display name or login; null for some bots and closures).",
    "title": "One-line headline of the event as rendered for readers.",
    "text": "The full rendered text of the event (headline, fact lines, complete body).",
    "entities": "Linked entity ids: issue:KAFKA-N, pr:N, kip:KIP-N, thread:<message id>, component:<name>.",
    "raw": "The complete original event record as one JSON string (never clipped).",
}  # fmt: skip
COLUMNS = {
    "issue_key": "JIRA issue key, e.g. KAFKA-12345.",
    "issue_id": "Numeric JIRA issue id (stable even if the key moves).",
    "issue_summary": "The issue's summary (title) as of this event.",
    "summary": "The issue's summary (title) when it was created.",
    "description": "Issue description when it was created (full text).",
    "reporter": "Who created the issue.",
    "status": "Workflow status at creation (later changes are rows of jira_changes with field='status').",
    "priority": "Priority at creation (later changes: jira_changes field='priority').",
    "assignee": "Assignee at creation (JIRA display name); null = unassigned. Later changes are jira_changes rows with field='assignee'.",
    "components": "JIRA components of the issue as of the event.",
    "labels": "JIRA labels at creation.",
    "at": "Exact instant with microseconds as recorded by JIRA, UTC; formatDateTime(at, '%Y-%m-%dT%H:%i:%S.%fZ') prints it in ISO form.",
    "author": "Who wrote it (JIRA display name, GitHub login or commit author name).",
    "comment_id": "Id of the comment in the source system.",
    "body": "Full body text (comment, review, PR description, mail body; description edits hold the new description).",
    "field": "Changed JIRA field: status, resolution, assignee, priority, fixVersion, Version, Component, labels, summary, description, issuetype, Link (issue links), RemoteIssueLink (web links such as GitHub PRs), ...",
    "from_value": "Value before the change (null = empty). fixVersion/Version lists are comma-joined.",
    "to_value": "Value after the change (null = cleared).",
    "history_id": "Number of the JIRA changelog entry the item belongs to; JIRA numbers entries in the order it writes them.",
    "item_index": "Position of the item inside its changelog entry.",
    "link_action": "For Link/RemoteIssueLink rows: added or removed.",
    "link_relation": "Link wording, e.g. 'is related to', 'blocks', 'duplicates', 'links to' (web link).",
    "linked_issue": "Other issue key of an issue link (field='Link').",
    "linked_pr": "GitHub PR number of a 'GitHub Pull Request #N' web link.",
    "pr_number": "GitHub pull request number.",
    "pr_title": "Pull request title.",
    "action": "Lifecycle step of the pull request: opened, merged or closed (closed without merge).",
    "merged_by": "Login that merged the pull request.",
    "closed_by": "Login that closed the pull request without merging.",
    "base": "Target branch.",
    "head": "Source branch.",
    "merge_commit": "Merge commit sha (merged rows).",
    "changed_files": "Repository paths changed by a merged pull request or by a pushed commit.",
    "issues": "JIRA issue keys linked to the item (for comments and reviews: those of its pull request).",
    "kips": "KIP ids linked to the item (for comments and reviews: those of its pull request), e.g. KIP-1071.",
    "body_issues": "JIRA keys written in this item's own body, distinct, in order of first mention.",
    "body_kips": "KIP ids written in this item's own body (any case, shown upper-case), distinct, in order of first mention.",
    "review_id": "GitHub review id.",
    "reviewer": "Login of the reviewer.",
    "state": "State of the review: approved, changes_requested, commented, dismissed.",
    "path": "File the inline comment is on.",
    "line": "Line the inline comment is on (when known).",
    "sha": "Commit sha.",
    "branch": "Branch pushed to.",
    "headline": "First line of the commit message.",
    "message": "Full commit message.",
    "prs": "Pull request numbers associated with the commit.",
    "message_id": "Mail Message-ID.",
    "list": "Mailing list address.",
    "subject": "Mail subject.",
    "sender": "Sender name as recorded.",
    "thread_id": "Message-ID that opens the thread; every message of one thread carries it.",
    "in_reply_to": "Message-ID this mail replies to (null for a thread start).",
}  # fmt: skip
ABOUT = {  # table -> (what one row holds, what identifies it); descriptions of the data only
    "jira_issues": ("One row per issue-created event: the issue's fields as they were at creation (summary, reporter, status, priority, assignee, components, labels, description). An issue created before the export's first event has no row.",
                    "A row is identified by issue_key (and issue_id); time is the creation instant."),
    "jira_changes": ("One row per item of a JIRA changelog entry: the field named in `field` changed from `from_value` to `to_value` at `at`. Items of the fields Link and RemoteIssueLink record an added or removed link (link_action, link_relation, linked_issue, linked_pr).",
                     "A row is identified by event_id; history_id numbers its changelog entry and item_index its position there; issue_key names the issue."),
    "jira_comments": ("One row per comment posted on a JIRA issue, with its full body.",
                      "A row is identified by comment_id; issue_key names the issue."),
    "github_pull_requests": ("One row per pull-request lifecycle event: opened (title, author, branches, description), merged (merged_by, merge commit, changed files) or closed without merge; `action` says which.",
                             "A row is identified by event_id; pr_number names the pull request, which has one row per lifecycle event."),
    "github_reviews": ("One row per review submitted on a pull request, with its state and body.",
                       "A row is identified by review_id; pr_number names the pull request."),
    "github_review_comments": ("One row per inline review comment, attached to a file (path) and line of a pull request.",
                               "A row is identified by comment_id; pr_number names the pull request."),
    "github_pr_comments": ("One row per conversation comment on a pull request (not attached to a line).",
                           "A row is identified by comment_id; pr_number names the pull request."),
    "github_commits": ("One row per commit pushed to a branch: message, author, associated pull requests and the issue keys and KIP ids it names.",
                       "A row is identified by sha."),
    "mail_messages": ("One row per mailing-list message: subject, sender, thread, reply target and full body.",
                      "A row is identified by message_id; all messages of a thread share thread_id."),
}  # fmt: skip
KEYS = {  # the columns a summary names first
    "jira_issues": ["issue_key", "summary", "reporter", "status", "assignee"],
    "jira_changes": ["issue_key", "field", "from_value", "to_value", "at", "actor"],
    "jira_comments": ["issue_key", "author", "at", "body"],
    "github_pull_requests": ["pr_number", "action", "pr_title", "author", "issues", "changed_files"],
    "github_reviews": ["pr_number", "reviewer", "state", "body"],
    "github_review_comments": ["pr_number", "author", "path", "line", "body"],
    "github_pr_comments": ["pr_number", "author", "body"],
    "github_commits": ["sha", "branch", "author", "headline", "changed_files", "prs"],
    "mail_messages": ["thread_id", "subject", "sender", "in_reply_to", "body", "kips"],
}  # fmt: skip
JOINS = {
    "jira_issues": ["issue_key = jira_changes.issue_key = jira_comments.issue_key", "issue_key IN github_pull_requests.issues (has(issues, issue_key))"],
    "jira_changes": ["issue_key = jira_issues.issue_key", "linked_pr = github_pull_requests.pr_number (field = 'RemoteIssueLink')", "linked_issue = another issue_key (field = 'Link')"],
    "jira_comments": ["issue_key = jira_issues.issue_key = jira_changes.issue_key"],
    "github_pull_requests": ["pr_number = github_reviews/github_review_comments/github_pr_comments.pr_number", "has(issues, jira key); jira_changes.linked_pr = pr_number", "merge_commit = github_commits.sha"],
    "github_reviews": ["pr_number = github_pull_requests.pr_number"],
    "github_review_comments": ["pr_number = github_pull_requests.pr_number"],
    "github_pr_comments": ["pr_number = github_pull_requests.pr_number"],
    "github_commits": ["sha = github_pull_requests.merge_commit", "has(prs, pr_number)", "has(issues, jira key)"],
    "mail_messages": ["thread_id groups a thread", "has(issues, jira key); has(kips, 'KIP-N') (from the subject)", "KIP ids also appear in jira/github titles and bodies (ILIKE '%KIP-1071%')"],
}  # fmt: skip
# Structured links between tables (the prose of JOINS as data): (from table, column, to table,
# column, meaning). The catalog adds each side's kind (scalar/array, from the schema), the value
# shape (inferred from the values) and the SQL of the join; a link whose columns an export lacks
# is left out of that export.
LINKS = [
    ("jira_changes", "issue_key", "jira_issues", "issue_key", "the issue a change belongs to; its creation row"),
    ("jira_comments", "issue_key", "jira_changes", "issue_key", "the issue a comment belongs to"),
    ("jira_changes", "linked_issue", "jira_changes", "issue_key", "the other issue of an issue link (field = 'Link')"),
    ("jira_changes", "linked_pr", "github_pull_requests", "pr_number", "the pull request of a web link (field = 'RemoteIssueLink')"),
    ("github_pull_requests", "issues", "jira_changes", "issue_key", "issues a pull request is linked to"),
    ("github_commits", "issues", "jira_changes", "issue_key", "issues a commit names"),
    ("github_commits", "prs", "github_pull_requests", "pr_number", "pull requests a commit belongs to"),
    ("github_pull_requests", "merge_commit", "github_commits", "sha", "the commit a merged pull request produced"),
    ("github_reviews", "pr_number", "github_pull_requests", "pr_number", "the pull request a review belongs to"),
    ("github_review_comments", "pr_number", "github_pull_requests", "pr_number", "the pull request an inline comment belongs to"),
    ("github_pr_comments", "pr_number", "github_pull_requests", "pr_number", "the pull request a conversation comment belongs to"),
    ("mail_messages", "issues", "jira_changes", "issue_key", "issues named in a mail subject"),
    ("mail_messages", "in_reply_to", "mail_messages", "message_id", "the message a mail replies to"),
    ("mail_messages", "thread_id", "mail_messages", "message_id", "the first message of the mail's thread (if in the export)"),
    ("mail_messages", "body_issues", "jira_changes", "issue_key", "issues named in a mail's text"),
    ("jira_comments", "body_issues", "jira_changes", "issue_key", "other issues named in a JIRA comment"),
    ("github_reviews", "body_issues", "jira_changes", "issue_key", "issues named in a review's text"),
    ("github_review_comments", "body_issues", "jira_changes", "issue_key", "issues named in an inline comment"),
    ("github_pr_comments", "body_issues", "jira_changes", "issue_key", "issues named in a PR conversation comment"),
    ("mail_messages", "body_kips", "mail_messages", "kips", "KIP ids in a mail's text and on mail subjects"),
    ("github_pull_requests", "kips", "mail_messages", "kips", "KIP ids on a pull request and on mail subjects"),
    ("github_reviews", "body_kips", "mail_messages", "kips", "KIP ids in a review's text and on mail subjects"),
    ("jira_comments", "body_kips", "mail_messages", "kips", "KIP ids in a JIRA comment and on mail subjects"),
]  # fmt: skip

# Worked queries: templates whose {placeholders} are filled from the export's own data by the
# pick queries (run in order; a pick may use earlier ones and {month} = first day of the
# checkpoint month, {boundary} = the checkpoint boundary). An entity pick takes the FIRST
# qualifying entity in the data (earliest time, then value): stable across checkpoints and
# unrelated to how busy an entity is (busy entities are what benchmark tasks ask about).
# A template is printed only if every pick finds a value and the filled query returns a row
# whose first value is not empty.
WORD = "SELECT w FROM (SELECT arrayJoin(splitByNonAlpha(lower({col}))) AS w FROM {table}) WHERE length(w) BETWEEN 6 AND 9 GROUP BY w ORDER BY count() DESC, w LIMIT 1"
HELD = "(SELECT issue_key, link_relation, linked_issue, min(time) AS time FROM jira_changes WHERE field = 'Link' GROUP BY issue_key, link_relation, linked_issue HAVING argMax(link_action, (at, history_id, item_index)) = 'added')"
TOP = "SELECT {value} FROM {source} WHERE {where} GROUP BY {value} ORDER BY min(time), {value} LIMIT 1"


def top(value, source, where="1"):
    return TOP.format(value=value, source=source, where=where)


EXAMPLES = {  # demonstrations of SQL technique over this schema, titled by technique
    "jira_issues": [
        ("Text search (ILIKE): rows whose summary contains a word", "SELECT issue_key, summary, reporter, at FROM jira_issues WHERE summary ILIKE '%' || {word} || '%' ORDER BY at",
         {"word": WORD.format(col="summary", table="jira_issues")}),
        ("Array membership and a time range: rows with one component created since an instant", "SELECT issue_key, summary FROM jira_issues WHERE has(components, {component}) AND time >= {month}",
         {"component": top("c", "jira_issues ARRAY JOIN components AS c", "time >= {month}")}),
        ("Union of tables and the latest value per key: the newest summary seen for each issue", "SELECT issue_key, argMax(s, time) AS summary FROM (SELECT issue_key, summary AS s, time FROM jira_issues UNION ALL SELECT issue_key, issue_summary, time FROM jira_changes WHERE issue_summary IS NOT NULL) WHERE s ILIKE '%' || {word} || '%' GROUP BY issue_key",
         {"word": WORD.format(col="issue_summary", table="jira_changes")}),
    ],
    "jira_changes": [
        ("Latest row per key before an instant (argMax over the ordering columns): one field of one issue", "SELECT argMax(to_value, (at, history_id, item_index)) AS value FROM jira_changes WHERE issue_key = {issue} AND field = {field} AND at < {boundary}",
         {"issue": top("issue_key", "jira_changes", "field NOT IN ('Link', 'RemoteIssueLink')"), "field": top("field", "jira_changes", "issue_key = {issue} AND field NOT IN ('Link', 'RemoteIssueLink')")}),
        ("Filter by key, in order: all rows of one field of one issue", "SELECT at, from_value, to_value, actor FROM jira_changes WHERE issue_key = {issue} AND field = {field} ORDER BY at, history_id, item_index",
         {"issue": top("issue_key", "jira_changes", "field NOT IN ('Link', 'RemoteIssueLink')"), "field": top("field", "jira_changes", "issue_key = {issue} AND field NOT IN ('Link', 'RemoteIssueLink')")}),
        ("Join through a link: the pull requests of an issue's web links, with their titles", "SELECT c.linked_pr, p.pr_title FROM jira_changes AS c INNER JOIN (SELECT pr_number, any(pr_title) AS pr_title FROM github_pull_requests GROUP BY pr_number) AS p ON c.linked_pr = p.pr_number WHERE c.issue_key = {issue}",
         {"issue": top("issue_key", "jira_changes", "linked_pr IN (SELECT pr_number FROM github_pull_requests)")}),
    ],
    "jira_comments": [
        ("Filter by key, in time order: the comments of one issue", "SELECT at, actor, body FROM jira_comments WHERE issue_key = {issue} ORDER BY at",
         {"issue": top("issue_key", "jira_comments")}),
        ("Token search (hasTokenCaseInsensitive): comments containing a word", "SELECT issue_key, actor, at FROM jira_comments WHERE hasTokenCaseInsensitive(body, {word}) ORDER BY at DESC LIMIT 20",
         {"word": WORD.format(col="body", table="jira_comments")}),
    ],
    "github_pull_requests": [
        ("Rows of one action vs all rows (1/2): with an 'opened' row in this export naming an issue, each pull request's latest action", "SELECT pr_number, argMax(action, time) AS state FROM github_pull_requests WHERE pr_number IN (SELECT pr_number FROM github_pull_requests WHERE action = 'opened' AND has(issues, {issue})) GROUP BY pr_number ORDER BY pr_number",
         {"issue": top("i", "github_pull_requests ARRAY JOIN issues AS i", "action = 'opened'")}),
        ("Rows of one action vs all rows (2/2): pull requests any of whose rows names an issue, including those opened before this export, with their latest action", "SELECT pr_number, argMax(action, time) AS state FROM github_pull_requests WHERE has(issues, {issue}) GROUP BY pr_number ORDER BY pr_number",
         {"issue": top("i", "github_pull_requests ARRAY JOIN issues AS i")}),
        ("Group and count over a time range with explicit instants: rows per action", "SELECT action, count() AS n FROM github_pull_requests WHERE time >= {month} AND time < {boundary} GROUP BY action ORDER BY action",
         {}),
        ("Text search (ILIKE) on a title", "SELECT pr_number, pr_title, time FROM github_pull_requests WHERE action = 'opened' AND pr_title ILIKE '%' || {word} || '%' ORDER BY time",
         {"word": WORD.format(col="pr_title", table="github_pull_requests")}),
    ],
    "github_reviews": [
        ("Join through a link: the reviews of one pull request with its title", "SELECT r.time, r.reviewer, r.state, p.pr_title FROM github_reviews AS r INNER JOIN (SELECT pr_number, any(pr_title) AS pr_title FROM github_pull_requests GROUP BY pr_number) AS p ON r.pr_number = p.pr_number WHERE r.pr_number = {pr} ORDER BY r.time",
         {"pr": top("pr_number", "github_reviews", "pr_number IN (SELECT pr_number FROM github_pull_requests)")}),
        ("Group and count: rows per value of an enumeration column", "SELECT state, count() AS n FROM github_reviews GROUP BY state ORDER BY n DESC",
         {}),
    ],
    "github_review_comments": [
        ("Filter by a person, newest first", "SELECT pr_number, path, line, substring(body, 1, 200) AS body FROM github_review_comments WHERE author = {author} ORDER BY time DESC LIMIT 20",
         {"author": top("author", "github_review_comments")}),
    ],
    "github_pr_comments": [
        ("Filter by key, in time order: the conversation of one pull request", "SELECT time, author, substring(body, 1, 300) AS body FROM github_pr_comments WHERE pr_number = {pr} ORDER BY time",
         {"pr": top("pr_number", "github_pr_comments")}),
    ],
    "github_commits": [
        ("Array membership: commits on one branch naming an issue key", "SELECT time, sha, headline FROM github_commits WHERE branch = 'trunk' AND has(issues, {issue}) ORDER BY time",
         {"issue": top("i", "github_commits ARRAY JOIN issues AS i", "branch = 'trunk'")}),
        ("Array membership on paths: commits changing one file", "SELECT time, sha, headline FROM github_commits WHERE has(changed_files, {path}) ORDER BY time",
         {"path": top("f", "github_commits ARRAY JOIN changed_files AS f")}),
    ],
    "mail_messages": [
        ("Array membership in either of two columns: messages naming a KIP id in the subject or the text", "SELECT time, sender, subject FROM mail_messages WHERE has(kips, {kip}) OR has(body_kips, {kip}) ORDER BY time",
         {"kip": top("k", "mail_messages ARRAY JOIN body_kips AS k")}),
        ("A thread in order: all messages sharing one thread_id, first message first", "SELECT time, sender, subject FROM mail_messages WHERE thread_id = {thread} ORDER BY time, event_id",
         {"thread": top("thread_id", "mail_messages", "message_id = thread_id")}),
        ("Token search (hasTokenCaseInsensitive) in message bodies", "SELECT time, sender, subject FROM mail_messages WHERE hasTokenCaseInsensitive(body, {word}) ORDER BY time",
         {"word": WORD.format(col="body", table="mail_messages")}),
    ],
}  # fmt: skip
# True enumerations: their catalog examples are the most frequent values (with their exact,
# case-sensitive spelling); every other column shows its earliest values.
ENUMERATIONS = {"event_type", "field", "action", "state", "status", "priority", "resolution",
                "link_action", "link_relation", "list", "branch", "base", "components", "labels",
                "item_index"}  # fmt: skip
PEOPLE = ("People are named per system with no mapping between them: JIRA display names "
          "(actor, author, reporter, assignee), GitHub logins (actor, author, reviewer, merged_by), "
          "commit author names (github_commits.author) and mail sender names; no event pairs a "
          "login with a name.")  # fmt: skip
TABLE_COLUMNS = {  # per-table meanings that differ from COLUMNS
    "mail_messages": {
        "issues": "JIRA keys named in the SUBJECT only; keys in the text are in body_issues.",
        "kips": "KIP ids named in the SUBJECT only; a message discussing a KIP in its text has it in body_kips.",
    },
    "jira_issues": {"at": "Creation instant as printed on the created event; equals `time` (whole seconds)."},
    "jira_changes": {"author": "The 'by' line of a field change; missing on link rows. `actor` is set on every row."},
    "jira_comments": {"author": "Comment author (same as `actor`)."},
}  # fmt: skip
