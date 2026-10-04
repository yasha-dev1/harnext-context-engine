"""Context event stream for kafka-500-v2: replay CloudEvents -> readable, leak-safe events.

The full stream is COMPLETE: every replay event is rendered, nothing is filtered, sampled
or trimmed. Renderers whitelist the fields they show, because replay payloads carry
harvest-time state (final JIRA summary/description/components, `merged_at`, `closed_at`,
`updated_at`, ...); JIRA state is rebuilt as of the event from the raw changelog instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

from harnext_eval.e2.benchmark import ROOT, canonical, inventory, iso, save, timestamp

REPLAY = ROOT / "out/corpus/kafka/replay/kafka-rlong.jsonl"
OUT = ROOT / "out/kafka-500-v2"
SCOPES = {
    "full": {"dir": "stream", "window": (None, "2026-07-01T00:00:00Z")},  # from the first event
    # Scoped benchmark (Yasha, 2026-09-26): stores ingest Jan-May 2026 only (after the reader
    # models' training cutoff); January is warm-up, tasks use the Feb-May checkpoints.
    "w2026": {"dir": "stream-w2026", "window": ("2026-01-01T00:00:00Z", "2026-06-01T00:00:00Z")},
    "smoke": {
        "dir": "stream-smoke",
        "window": ("2023-01-01T00:00:00Z", "2023-02-15T00:00:00Z"),
        "top": 40,
        "threads": 3,
        "per_field": 3,
    },
}
TYPES = {
    "org.apache.jira.issue.created": "jira.created",
    "org.apache.jira.issue.transition": "jira.change",
    "org.apache.jira.issue.comment": "jira.comment",
    "com.github.pull_request.opened": "gh.pr_opened",
    "com.github.pull_request.merged": "gh.pr_merged",
    "com.github.pull_request.closed": "gh.pr_closed",
    "com.github.review": "gh.review",
    "com.github.review_comment": "gh.review_comment",
    "com.github.issue_comment": "gh.issue_comment",
    "com.github.push": "gh.push",
    "org.apache.mail.message": "mail.message",
}
LINK_FIELDS = {"link", "remoteissuelink"}
LIMITS = [
    "GitHub PR titles/bodies and comment bodies are the harvest-time text; GitHub edit "
    "history is not in the corpus. State fields (merged, closed, labels, merged_by) are "
    "never rendered outside the event that records them.",
    "JIRA comment bodies are the harvest-time text; comments edited later are counted in "
    "`jira_comments_edited_later`.",
    "`time` is truncated to whole seconds so ISO strings compare correctly; the exact "
    "source timestamp is printed in the text of JIRA events.",
]
KEY = re.compile(r"\bKAFKA-\d+\b")
KIP = re.compile(r"\bKIP-\d+\b")
TIME = re.compile(r'"time": ?"([^"]+)"')


# --- checkpoints -----------------------------------------------------------------------


def _next(month):
    year, number = int(month[:4]), int(month[5:7])
    return f"{year + number // 12:04d}-{number % 12 + 1:02d}"


def checkpoints(first="2023-01", last="2026-06") -> list[str]:
    result = [first]
    while result[-1] < last:
        result.append(_next(result[-1]))
    return result


def boundary(ckpt) -> str:
    """ISO instant of the first moment NOT included in the checkpoint."""
    return f"{_next(ckpt)}-01T00:00:00Z"


def checkpoint_for(cutoff_iso, first="2023-01", last="2026-06") -> str | None:
    """Latest checkpoint whose boundary is <= the cutoff."""
    cutoff = timestamp(cutoff_iso)
    fitting = [c for c in checkpoints(first, last) if timestamp(boundary(c)) <= cutoff]
    return fitting[-1] if fitting else None


# --- JIRA state as of an instant ------------------------------------------------------


def _name(value):
    if isinstance(value, dict):
        return value.get("displayName") or value.get("name")
    return value


class Jira:
    """As-of views over raw JIRA exports; an unknown issue yields nothing, never a guess."""

    def __init__(self, issues=None):
        self.issues = issues or {}
        self.logs = {}

    def log(self, key):
        if key not in self.logs:
            histories = (self.issues.get(key) or {}).get("changelog", {}).get("histories", [])
            self.logs[key] = sorted(
                (
                    (timestamp(h["created"]), int(h["id"]), index, item)
                    for h in histories
                    for index, item in enumerate(h["items"])
                ),
                key=lambda row: row[:3],
            )
        return self.logs[key]

    def item(self, key, history, index, field):
        """Raw changelog item behind a replay transition (it has the display names)."""
        rows = [r[3] for r in self.log(key) if r[1] == history and r[3]["field"] == field]
        exact = [r[3] for r in self.log(key) if r[1] == history and r[2] == index]
        return exact[0] if exact and exact[0]["field"] == field else (rows[0] if rows else None)

    def value(self, key, field, when):
        """Single-valued field as of `when` (inclusive): undo every later change."""
        if key not in self.issues:
            return None
        value = _name(self.issues[key]["fields"].get(field))
        for at, _, _, item in reversed(self.log(key)):
            if at <= when:
                break
            if item["field"].lower() == field:
                value = item.get("fromString")
        return value

    def many(self, key, field, log_field, when):
        """Multi-valued field (components, labels) as of `when`."""
        if key not in self.issues:
            return []
        values = [_name(v) for v in self.issues[key]["fields"].get(field) or []]
        for at, _, _, item in reversed(self.log(key)):
            if at <= when:
                break
            if item["field"].lower() != log_field:
                continue
            if log_field == "labels":
                values = (item.get("fromString") or "").split()
                continue
            if item.get("toString") in values:
                values.remove(item["toString"])
            if item.get("fromString"):
                values.append(item["fromString"])
        return sorted({str(v) for v in values if v})


# --- rendering --------------------------------------------------------------------------


def clean(body):
    """Bodies are delivered whole (templates, quotes, bot text); only newlines are unified."""
    return (body or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def shown(value):
    if isinstance(value, list):
        return ", ".join(map(str, value)) or "(none)"
    return "(none)" if value in (None, "") else str(value)


def page(title, facts, body=""):
    lines = [f"# {title}", ""]
    lines += [f"- {name}: {shown(value)}" for name, value in facts if value not in (None, "", [])]
    return "\n".join(lines) + (f"\n\n{body}" if body else "") + "\n"


def files(paths):
    return "\n".join(f"- `{p}`" for p in dict.fromkeys(paths or []))


def section(heading, body):
    return f"## {heading}\n\n{body}" if body else ""


def join(*parts):
    return "\n\n".join(p for p in parts if p)


def refs(data):
    issues = list(data.get("issue_keys") or [])
    kips = list(data.get("kip_keys") or [])
    return issues, kips


def render_jira(kind, event, jira):
    data = event["data"]
    key = data["issue_key"]
    when = timestamp(event["time"])
    exact = iso(when)
    summary = jira.value(key, "summary", when)
    components = jira.many(key, "components", "component", when)
    entities = [f"issue:{key}"] + [f"component:{c}" for c in components]
    issue = f"{key} — {summary}" if summary else key
    meta = {"issue": key}
    if kind == "jira.created":
        actor = data.get("creator_name")
        facts = [
            ("issue", key), ("event", "issue created"), ("reporter", actor), ("at", exact),
            ("status", jira.value(key, "status", when)),
            ("priority", jira.value(key, "priority", when)),
            ("assignee", jira.value(key, "assignee", when)),
            ("components", components),
            ("labels", jira.many(key, "labels", "labels", when)),
        ]  # fmt: skip
        title = f"{key} created: {summary}" if summary else f"{key} created"
        text = page(title, facts, clean(jira.value(key, "description", when)))
        return entities, actor, title, text, meta
    if kind == "jira.comment":
        actor = data.get("author_name")
        title = f"{key} comment by {actor}"
        facts = [("issue", issue), ("event", "comment"), ("by", actor), ("at", exact)]
        return entities, actor, title, page(title, facts, clean(data.get("body"))), meta
    actor = data.get("actor_name")
    field = data["field"]
    before, after = data.get("from"), data.get("to")
    if field.lower() == "assignee":  # the replay hashes people; the raw changelog names them
        parts = event["id"].split(":")
        raw = jira.item(key, int(data["changelog_id"]), int(parts[-1]), field)
        before, after = (raw.get("fromString"), raw.get("toString")) if raw else (before, after)
    meta.update(field=field, **{"from": before, "to": after})
    if field.lower() in LINK_FIELDS:
        target = after if after is not None else before
        verb = "link added" if after is not None else "link removed"
        pulls = re.findall(r"Pull Request #(\d+)", target or "")
        entities += [f"pr:{n}" for n in pulls]
        entities += [f"issue:{k}" for k in KEY.findall(target or "") if k != key]
        title = f"{key} linked to PR #{pulls[0]}" if pulls else f"{key} {verb}"
        text = f"{key} {verb}: {target} (by {shown(actor)}, at {exact})\n"
        return entities, actor, title, text, meta
    facts = [("issue", issue), ("event", "field change"), ("by", actor), ("at", exact)]
    if field.lower() == "description":
        title = f"{key} description edited"
        return entities, actor, title, page(title, facts, clean(after)), meta
    title = f"{key} {field}: {shown(before)} → {shown(after)}"
    facts += [("field", field), ("from", shown(before)), ("to", shown(after))]
    return entities, actor, title, page(title, facts), meta


def render_github(kind, event):
    data = event["data"]
    when = event["time"]
    issues, kips = refs(data)
    if kind == "gh.push":
        commits = data.get("commits") or []
        pulls = data.get("associated_pr_numbers") or []
        actor = next((c.get("author_name") for c in commits if c.get("author_name")), None)
        actor = actor or data.get("actor_login")
        message = clean(data.get("message") or "\n\n".join(c.get("message", "") for c in commits))
        headline = message.split("\n")[0]
        title = f"Push to {data.get('branch')}: {headline}"
        changed = [p for c in commits for p in c.get("changed_files") or []]
        facts = [
            ("event", "commit pushed"), ("commit", data.get("sha")), ("branch", data.get("branch")),
            ("author", actor), ("at", when), ("pull requests", [f"#{n}" for n in pulls]),
            ("issues", issues), ("KIPs", kips),
        ]  # fmt: skip
        body = join(
            section("Commit message", message),
            section("Changed files", files(changed or data.get("changed_files"))),
        )
        entities = [f"pr:{n}" for n in pulls]
        meta = {"sha": data.get("sha"), "branch": data.get("branch")}
    else:
        number = data["number"]
        name = data.get("title") or ""
        pull = f"#{number} — {name}"
        entities = [f"pr:{number}"]
        meta = {"pr": number}
        links = [("issues", issues), ("KIPs", kips)]
        body = clean(data.get("body"))
        if kind == "gh.pr_opened":
            actor = data.get("author_login") or data.get("actor_login")
            title = f"PR #{number} opened: {name}"
            facts = [
                ("pull request", f"#{number}"), ("event", "pull request opened"),
                ("author", actor), ("at", when), ("base", data.get("base_ref")),
                ("head", data.get("head_ref")), *links,
            ]  # fmt: skip
        elif kind == "gh.pr_merged":
            actor = data.get("merged_by") or data.get("actor_login")
            title = f"PR #{number} merged: {name}"
            facts = [
                ("pull request", f"#{number}"), ("event", "pull request merged"),
                ("author", data.get("author_login")), ("merged by", actor), ("at", when),
                ("base", data.get("base_ref")), ("merge commit", data.get("merge_commit_sha")),
                *links,
            ]  # fmt: skip
            body = join(
                section("Description", body),
                section("Changed files", files(data.get("changed_files"))),
            )
            meta["merge_commit"] = data.get("merge_commit_sha")
        elif kind == "gh.pr_closed":
            actor = data.get("actor_login")
            title = f"PR #{number} closed without merge: {name}"
            facts = [
                ("pull request", f"#{number}"), ("event", "pull request closed without merge"),
                ("author", data.get("author_login")), ("closed by", actor), ("at", when), *links,
            ]  # fmt: skip
        elif kind == "gh.review":
            actor = data.get("author_login")
            state = (data.get("state") or "").replace("_", " ")
            title = f"PR #{number} review ({state}) by {actor}"
            facts = [
                ("pull request", pull), ("event", "review"), ("state", state),
                ("by", actor), ("at", when), *links,
            ]  # fmt: skip
            meta["state"] = data.get("state")
        elif kind == "gh.review_comment":
            actor = data.get("author_login")
            title = f"PR #{number} review comment by {actor} on {data.get('path')}"
            facts = [
                ("pull request", pull), ("event", "inline review comment"), ("by", actor),
                ("at", when), ("file", data.get("path")), ("line", data.get("line")), *links,
            ]  # fmt: skip
            meta["path"] = data.get("path")
        else:
            actor = data.get("author_login")
            title = f"PR #{number} comment by {actor}"
            facts = [("pull request", pull), ("event", "comment"), ("by", actor), ("at", when)]
            facts += links
    entities += [f"issue:{k}" for k in issues] + [f"kip:{k}" for k in kips]
    return entities, actor, title, page(title, facts, body), meta


def render_mail(event):
    data = event["data"]
    subject = " ".join((data.get("subject") or "(no subject)").split())
    actor = data.get("author_name") or data.get("author_email")
    thread = data.get("thread_root") or data.get("message_id")
    tags = " ".join(data.get("subject_tags") or []) + " " + subject
    entities = [f"thread:{thread}"]
    entities += [f"issue:{k}" for k in dict.fromkeys(KEY.findall(tags))]
    entities += [f"kip:{k}" for k in dict.fromkeys(KIP.findall(tags))]
    facts = [
        ("event", "mailing-list message"), ("list", data.get("list")), ("from", actor),
        ("at", event["time"]), ("thread", thread),
        ("in reply to", data.get("in_reply_to")),
    ]  # fmt: skip
    meta = {"list": data.get("list"), "message_id": data.get("message_id")}
    return entities, actor, subject, page(subject, facts, clean(data.get("body"))), meta


def render(event, jira) -> dict:
    """One stream event from one replay CloudEvent."""
    kind = TYPES[event["type"]]
    if kind.startswith("jira."):
        entities, actor, title, text, meta = render_jira(kind, event, jira)
    elif kind == "mail.message":
        entities, actor, title, text, meta = render_mail(event)
    else:
        entities, actor, title, text, meta = render_github(kind, event)
    return {
        "id": event["id"],
        "time": iso(timestamp(event["time"]).replace(microsecond=0)),
        "type": kind,
        "entities": list(dict.fromkeys(entities)),
        "actor": actor,
        "title": " ".join(title.split())[:200],
        "text": text,
        "meta": meta,
    }


def rendered(lines: Iterable[str], window, jira, notes: Counter) -> Iterator[dict]:
    """Every replay event inside the window; an unknown type is an error, never a drop."""
    start, end = window
    low, high = (start or "")[:10], end[:10]
    for line in lines:
        found = TIME.search(line)
        if found and not low <= found.group(1)[:10] <= high:
            continue  # cheap prefilter; the exact test follows
        event = json.loads(line)
        when = timestamp(event["time"])
        if (start and when < timestamp(start)) or when >= timestamp(end):
            continue
        if TYPES[event["type"]] == "jira.comment":
            updated = event["data"].get("updated")
            if updated and (timestamp(updated) - when).total_seconds() > 1:
                notes["jira_comments_edited_later"] += 1
        yield render(event, jira)


def focus(events, top, threads=0, per_field=0):
    """Events touching the `top` busiest issues/PRs, `threads` mail threads and, so every
    task family has material, the `per_field` issues with most changes of each gold field."""
    activity = Counter(e for event in events for e in event["entities"])
    edits = Counter(
        (event["meta"]["field"].lower(), f"issue:{event['meta']['issue']}")
        for event in events
        if event["type"] == "jira.change"
    )

    def busiest(rows, count):
        return {e for e, _ in sorted(rows, key=lambda kv: (-kv[1], kv[0]))[:count]}

    chosen = busiest([kv for kv in activity.items() if kv[0].startswith(("issue:", "pr:"))], top)
    chosen |= busiest([kv for kv in activity.items() if kv[0].startswith("thread:")], threads)
    for field in ("assignee", "priority", "resolution", "status"):
        chosen |= busiest([(e, n) for (f, e), n in edits.items() if f == field], per_field)
    return [e for e in events if chosen & set(e["entities"])], sorted(chosen)


def write(events, out: Path):
    """Sort by (time, id) through month buckets so the full stream never sits in memory."""
    buckets = out / ".buckets"
    shutil.rmtree(buckets, ignore_errors=True)
    buckets.mkdir(parents=True)
    handles, types, months, size = {}, Counter(), Counter(), 0
    for event in events:
        month = event["time"][:7]
        if month not in handles:
            handles[month] = (buckets / month).open("w")
        handles[month].write(f"{event['time']}\t{event['id']}\t{canonical(event)}\n")
        types[event["type"]] += 1
        size += len(event["text"])
        months[month] += 1
    for handle in handles.values():
        handle.close()
    digest = hashlib.sha256()
    with (out / "events.jsonl").open("wb") as target:
        for month in sorted(handles):
            rows = [line.split("\t", 2) for line in (buckets / month).open()]
            for row in sorted(rows, key=lambda r: (r[0], r[1])):
                target.write(row[2].encode())
                digest.update(row[2].encode())
    shutil.rmtree(buckets)
    return dict(sorted(types.items())), dict(sorted(months.items())), digest.hexdigest(), size


def source_sha(replay: Path):
    sidecar = replay.with_name(replay.name + ".sha256")
    if sidecar.exists():
        return sidecar.read_text().split()[0]
    digest = hashlib.sha256()
    with replay.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(scope="full", replay: Path = REPLAY, out: Path | None = None, issues=None) -> Path:
    config = SCOPES[scope]
    out = out or OUT / config["dir"]
    out.mkdir(parents=True, exist_ok=True)
    jira = Jira(inventory()[0] if issues is None else issues)
    notes, chosen = Counter(), None
    with replay.open() as lines:
        events = rendered(lines, config["window"], jira, notes)
        if "top" in config:
            events, chosen = focus(
                list(events), config["top"], config.get("threads", 0), config.get("per_field", 0)
            )
        types, months, digest, text_bytes = write(events, out)
    manifest = {
        "scope": scope,
        "window": {"from_inclusive": config["window"][0], "until_exclusive": config["window"][1]},
        "complete": "top" not in config,
        "source": str(replay.relative_to(ROOT)) if replay.is_relative_to(ROOT) else replay.name,
        "source_sha256": source_sha(replay),
        "events": sum(types.values()),
        "events_sha256": digest,
        "types": types,
        "months": months,
        "order": "time (whole seconds, UTC), then id",
        "filters": [],
        "text_bytes": text_bytes,
        "notes": dict(notes),
        "known_limits": LIMITS,
    }
    if chosen is not None:
        manifest["focus"] = {
            "top": config["top"],
            "threads": config.get("threads", 0),
            "per_field": config.get("per_field", 0),
            "entities": chosen,
        }
    save(out / "stream-manifest.json", manifest)
    return out / "events.jsonl"


def read_events(path: Path, until: str | None = None) -> Iterator[dict]:
    """Events in stream order; `until` is an exclusive ISO instant (e.g. `boundary(ckpt)`)."""
    limit = timestamp(until) if until else None
    with Path(path).open() as handle:
        for line in handle:
            event = json.loads(line)
            if limit and timestamp(event["time"]) >= limit:
                return
            yield event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("build")
    command.add_argument("--scope", choices=sorted(SCOPES), default="full")
    args = parser.parse_args()
    path = build(args.scope)
    manifest = json.loads((path.parent / "stream-manifest.json").read_text())
    print(canonical({k: manifest[k] for k in ("scope", "events", "text_bytes", "types")}))
    print(path)


if __name__ == "__main__":
    main()
