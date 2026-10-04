"""Eleven historical families beyond v1's field replays: pull requests, reviews, links, fix
versions, aggregates over issues and authors, and KIP discussion threads.

GitHub- and mail-side gold comes from ONE streaming pass over the context stream (`facts`,
cached beside events.jsonl under the stream's sha); JIRA-side gold replays the raw changelog
exactly as `tasks.candidates` does. Every gold is the state strictly before the checkpoint
boundary, and every family is answerable through the offline `gh`/`jira`/`mail` CLIs alone
(`sources/audit.py` holds the oracle). Nothing is padded: a generator yields what exists.
"""

from __future__ import annotations

import gzip
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

from harnext_eval.e2.benchmark import (
    ROOT,
    canonical,
    changes,
    iso,
    ordered,
    sha,
    stratified,
    timestamp,
)
from harnext_eval.e2.v2 import stream
from harnext_eval.e2.v2.sources.index import parse, split_files

REPO = ROOT / "out/kafka-repo/kafka.git"
NEW = (
    "issue_prs", "issue_changed_files", "pr_approvers", "review_hotspot", "reopen_count",
    "issue_links", "fix_version_at_snapshot", "resolution_vs_merge_order",
    "open_blockers_unassigned", "author_activity", "kip_thread",
)  # fmt: skip
VARIANTS = {"author_activity": ("author_activity", "top_reviewer_for_author")}
ISSUE_CENTRED = {"issue_prs", "issue_changed_files", "reopen_count", "issue_links", "fix_version_at_snapshot", "resolution_vs_merge_order"}  # fmt: skip
PR_CENTRED = {"pr_approvers", "review_hotspot"}
RELATIONS = ("duplicates", "is duplicated by", "blocks", "is blocked by", "relates to", "is related to", "is a clone of", "is cloned by")  # fmt: skip
LINK = re.compile(r"^This issue (.+?) ([A-Z][A-Z0-9]*-\d+)$")
DISCUSS = re.compile(r"\[DISCUSS\]\s*(KIP-\d+)(?!\d)", re.IGNORECASE)
GH = {"gh.pr_opened": "open", "gh.pr_merged": "merged", "gh.pr_closed": "closed"}
DAYS = {"author_activity": 90, "top_reviewer_for_author": 365}
MAX_FILES = 25  # an exact-match file list stays listable
MIN_THREAD = 5
BLOCKERS = (1, 8)  # matches per (component, checkpoint)
TRAP_TARGET = 0.7


# --- stream facts -------------------------------------------------------------------------


def facts(path: Path) -> dict:
    """Compact GitHub/mail facts of a stream: {"prs", "reviews", "review_comments", "mails"}.
    Cached as facts.json.gz beside the stream, valid for the manifest's events sha only."""
    manifest = json.loads((path.parent / "stream-manifest.json").read_text())
    cache = path.parent / "facts.json.gz"
    if cache.exists():
        found = json.loads(gzip.decompress(cache.read_bytes()))
        if found.get("sha") == manifest["events_sha256"]:
            return found
    prs, reviews, comments, mails = {}, defaultdict(list), defaultdict(list), []
    with path.open() as handle:
        for line in handle:
            if '"type":"jira.' in line or '"type":"gh.push"' in line or '"type":"gh.issue_comment"' in line:  # fmt: skip
                continue
            event = json.loads(line)
            kind, at = event["type"], event["time"]
            if kind == "mail.message":
                subject, shown, _ = parse(event["text"])
                mails.append([at, event["id"], shown.get("thread"), shown.get("from"), subject])
                continue
            if not kind.startswith("gh."):
                continue
            number = str(event["meta"]["pr"])
            pr = prs.setdefault(number, {"title": None, "author": None, "issues": [], "events": [], "tags": []})  # fmt: skip
            issues = [e.split(":", 1)[1] for e in event["entities"] if e.startswith("issue:")]
            pr["tags"] = sorted(set(pr["tags"]) | set(issues))
            if kind in GH:
                title, shown, body = parse(event["text"])
                pr["author"] = pr["author"] or shown.get("author")
                pr["events"].append([at, GH[kind], event["id"]])
                if kind == "gh.pr_opened":
                    pr["title"] = title.split(": ", 1)[1] if ": " in title else ""
                    pr["issues"] = issues
                elif kind == "gh.pr_merged":
                    pr["merge_commit"] = event["meta"].get("merge_commit")
                    pr["files"] = split_files(body, "Description")[1]
            elif kind == "gh.review":
                reviews[number].append(
                    [at, event["actor"], event["meta"].get("state"), event["id"]]
                )
            elif kind == "gh.review_comment":
                comments[number].append(
                    [at, event["actor"], event["meta"].get("path"), event["id"]]
                )
    found = {"sha": manifest["events_sha256"], "prs": prs, "reviews": dict(reviews), "review_comments": dict(comments), "mails": sorted(mails)}  # fmt: skip
    cache.write_bytes(gzip.compress(canonical(found).encode(), 6))
    return found


def state(pr, limit):
    """open|merged|closed as of `limit` (exclusive), None unless opened before it."""
    seen = [kind for at, kind, _ in pr["events"] if at < limit]
    return seen[-1] if seen and seen[0] == "open" else None


def merged_at(pr, limit=None):
    return next((at for at, kind, _ in pr["events"] if kind == "merged" and (limit is None or at < limit)), None)  # fmt: skip


def changed_files(sha_):
    """Paths the merge commit touches in the bare clone, or None when it is not there."""
    if not sha_ or not REPO.exists():
        return None
    done = subprocess.run(
        [
            "git",
            "--git-dir",
            str(REPO),
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            "-m",
            "--first-parent",
            sha_,
        ],  # fmt: skip
        capture_output=True,
        text=True,
    )
    return sorted(set(done.stdout.split())) if done.returncode == 0 else None


# --- candidates -------------------------------------------------------------------------------


class Context:
    """Everything a generator reads: raw issues, stream facts, checkpoints and the window.

    `start` is the first instant tasks are drawn from (the deciding event lies at or after it);
    smoke streams are slices: `since`/`end` bound them and `present` restricts the issues."""

    def __init__(self, issues, facts_, ckpts, start, *, since=None, end=None, present=None, seed="kafka-500-v2"):  # fmt: skip
        self.issues, self.facts, self.ckpts, self.seed = issues, facts_, ckpts, seed
        self.start, self.since, self.end, self.present = start, since, end, present
        self.jira = stream.Jira(issues)
        self.limits = [(c, stream.boundary(c), timestamp(stream.boundary(c))) for c in ckpts]
        self.by_issue = defaultdict(list)
        for number, pr in facts_["prs"].items():
            for key in pr["issues"]:
                self.by_issue[key].append(number)

    def keys(self):
        return [k for k in sorted(self.issues) if self.present is None or k in self.present]

    def log(self, key, field):
        """Complete raw changelog of one field with instants, or None when inconsistent."""
        try:
            rows = changes(self.issues[key], field)
        except (ValueError, KeyError):
            return None
        return [{**r, "when": timestamp(r["at"])} for r in rows]

    def visible(self, rows, when):
        """Rows strictly before `when`; None when a smoke slice would hide part of the story."""
        seen = [r for r in rows if r["when"] < when]
        if self.since and any(r["when"] < self.since for r in seen):
            return None
        if self.end and any(self.end <= r["when"] < when for r in rows):
            return None
        return seen

    def in_window(self, at):
        """A stream instant (whole-second ISO) lies at/after the task start."""
        return timestamp(at) >= self.start

    def task(self, family, key, ckpt, value, evidence, trap, **extra):
        return {
            "id": f"H-{family}-{sha(self.seed + key)[:10]}",
            "kind": "historical",
            "family": family,
            "checkpoint": ckpt,
            "cutoff": iso(timestamp(stream.boundary(ckpt)) - timedelta(seconds=1)),
            "issue": None,
            "expected": {"value": value},
            "evidence_ids": evidence,
            "temporal_trap": trap,
            "long_history": False,
            "grading": "typed_exact_json; strings casefolded and whitespace-normalised; arrays compared sorted",
            **extra,
        }

    def one_per_group(self, family, key, options):
        """`options`: {(group..., trap): [(ckpt, task)]} -> one hashed pick per group."""
        return [ordered(found, self.seed + family + key, lambda o: o[0])[0][1] for _, found in sorted(options.items())]  # fmt: skip


def issue_prs(ctx: Context):
    for key in ctx.keys():
        options = defaultdict(list)
        numbers = sorted(ctx.by_issue.get(key, []), key=int)
        prs = [(n, ctx.facts["prs"][n]) for n in numbers]
        if any(pr["tags"] != sorted(pr["issues"]) for _, pr in prs):
            continue  # an event disagrees on the PR's issues: the replica could differ
        for ckpt, limit, _ in ctx.limits:
            rows = [(n, state(pr, limit)) for n, pr in prs]
            rows = [(n, s) for n, s in rows if s]
            if not rows:
                continue
            times = [
                at for n, _ in rows for at, _, _ in ctx.facts["prs"][n]["events"] if at < limit
            ]
            if not ctx.in_window(max(times)):
                continue
            later = any(at >= limit for n, pr in prs for at, _, _ in pr["events"])
            value = [{"pr": int(n), "state": s} for n, s in rows]
            evidence = [
                i for n, _ in rows for at, _, i in ctx.facts["prs"][n]["events"] if at < limit
            ]
            options[(canonical(value), later)].append((ckpt, ctx.task(
                "issue_prs", key, ckpt, value, evidence, later, issue=key, preferred=len(rows) >= 2,
            )))  # fmt: skip
        yield from ctx.one_per_group("issue_prs", key, options)


def issue_changed_files(ctx: Context):
    """Files from the stream's merge events; `verify` re-derives them from the bare clone."""
    for key in ctx.keys():
        options = defaultdict(list)
        numbers = sorted(ctx.by_issue.get(key, []), key=int)
        prs = [(n, ctx.facts["prs"][n]) for n in numbers]
        if any(pr["tags"] != sorted(pr["issues"]) for _, pr in prs):
            continue
        for ckpt, limit, _ in ctx.limits:
            merged = [(n, pr) for n, pr in prs if merged_at(pr, limit)]
            if not merged or any(
                not pr.get("files") or not pr.get("merge_commit") for _, pr in merged
            ):
                continue
            if not ctx.in_window(max(merged_at(pr, limit) for _, pr in merged)):
                continue
            files = sorted({f for _, pr in merged for f in pr["files"]})
            if len(files) > MAX_FILES:
                continue
            later = any(merged_at(pr) and merged_at(pr) >= limit for _, pr in prs)
            evidence = [i for _, pr in merged for at, kind, i in pr["events"] if kind == "merged" and at < limit]  # fmt: skip
            options[(canonical(files), later)].append((ckpt, ctx.task(
                "issue_changed_files", key, ckpt, files, evidence, later, issue=key,
                prs=[int(n) for n, _ in merged], merge_commits=[pr["merge_commit"] for _, pr in merged],
                preferred=len(merged) == 1,
            )))  # fmt: skip
        yield from ctx.one_per_group("issue_changed_files", key, options)


def verify_files(task, cache=None):
    """The clone's diff-tree of every merge commit must reproduce the stream's file list."""
    cache = cache if cache is not None else {}
    union = set()
    for sha_ in task["merge_commits"]:
        if sha_ not in cache:
            cache[sha_] = changed_files(sha_)
        if cache[sha_] is None:
            return False
        union |= set(cache[sha_])
    return sorted(union) == task["expected"]["value"]


def approvers(rows, limit, author=None):
    return sorted({login for at, login, s, _ in rows if at < limit and s == "approved" and login and login != author})  # fmt: skip


def pr_approvers(ctx: Context):
    for number, rows in sorted(ctx.facts["reviews"].items(), key=lambda kv: int(kv[0])):
        pr, options = ctx.facts["prs"][number], defaultdict(list)
        for ckpt, limit, _ in ctx.limits:
            logins = approvers(rows, limit)
            if not logins or not state(pr, limit):
                continue
            last = max(at for at, _, s, _ in rows if at < limit and s == "approved")
            if not ctx.in_window(last):
                continue
            later = any(at >= limit and s == "approved" for at, _, s, _ in rows)
            evidence = [i for at, _, s, i in rows if at < limit and s == "approved"]
            options[(canonical(logins), later)].append((ckpt, ctx.task(
                "pr_approvers", f"pr:{number}", ckpt, logins, evidence, later, pr=int(number),
                issue=None, issues=pr["issues"], preferred=len(logins) >= 2,
            )))  # fmt: skip
        yield from ctx.one_per_group("pr_approvers", f"pr:{number}", options)


def hotspot(rows, limit):
    """(path, count) of the file with most review comments before `limit`; None on a tie."""
    counts = Counter(path for at, _, path, _ in rows if at < limit and path)
    top = counts.most_common(2)
    if not top or (len(top) > 1 and top[0][1] == top[1][1]):
        return None
    return top[0]


def review_hotspot(ctx: Context):
    for number, rows in sorted(ctx.facts["review_comments"].items(), key=lambda kv: int(kv[0])):
        pr, options = ctx.facts["prs"][number], defaultdict(list)
        for ckpt, limit, _ in ctx.limits:
            top = hotspot(rows, limit)
            if not top or not state(pr, limit) or not ctx.in_window(max(at for at, *_ in rows if at < limit)):  # fmt: skip
                continue
            later = any(at >= limit for at, *_ in rows)
            evidence = [i for at, _, path, i in rows if at < limit and path == top[0]]
            options[(top[0], later)].append((ckpt, ctx.task(
                "review_hotspot", f"pr:{number}", ckpt, top[0], evidence, later, pr=int(number),
                issue=None, issues=pr["issues"], preferred=top[1] >= 3,
            )))  # fmt: skip
        yield from ctx.one_per_group("review_hotspot", f"pr:{number}", options)


def jira_evidence(issue, rows):
    return [f"jira:{issue['id']}:change:{r['history_id']}:{r['item_index']}" for r in rows]


def reopen_count(ctx: Context):
    """Issues resolved at least once; `count` is the number of transitions to Reopened."""
    for key in ctx.keys():
        rows, options = ctx.log(key, "status"), defaultdict(list)
        if not rows:
            continue
        for ckpt, _, when in ctx.limits:
            seen = ctx.visible(rows, when)
            if not seen or not any(r["to"] in {"Resolved", "Closed"} for r in seen) or seen[-1]["when"] < ctx.start:  # fmt: skip
                continue
            reopened = [r for r in seen if r["to"] == "Reopened"]
            later = any(r["to"] == "Reopened" for r in rows if r["when"] >= when)
            options[(len(reopened), later)].append((ckpt, ctx.task(
                "reopen_count", key, ckpt, len(reopened), jira_evidence(ctx.issues[key], reopened or seen[-1:]),
                later, issue=key, preferred=len(reopened) >= 1,
            )))  # fmt: skip
        yield from ctx.one_per_group("reopen_count", key, options)


def link(row):
    """(relation, key, added) of a Link changelog item, or None for a foreign wording."""
    text, added = (row["to"], True) if row["to"] else (row["from"], False)
    found = LINK.match(text or "")
    return (*found.groups(), added) if found else None


def links(rows):
    """{relation: sorted keys} after replaying Link items in order (removals honoured)."""
    held = defaultdict(set)
    for parsed in filter(None, map(link, rows)):
        relation, key, added = parsed
        (held[relation].add if added else held[relation].discard)(key)
    return {k: sorted(v) for k, v in held.items() if v}


def issue_links(ctx: Context):
    for key in ctx.keys():
        rows, options = ctx.log(key, "Link"), defaultdict(list)
        if not rows:
            continue
        for ckpt, _, when in ctx.limits:
            seen = ctx.visible(rows, when)
            if not seen or seen[-1]["when"] < ctx.start:
                continue
            held = links(seen)
            for relation in RELATIONS:
                if relation not in held:
                    continue
                mine = [r for r in seen if (p := link(r)) and p[0] == relation]
                later = any((p := link(r)) and p[0] == relation for r in rows if r["when"] >= when)
                removed = any(not r["to"] for r in mine)
                options[(relation, canonical(held[relation]), later)].append((ckpt, ctx.task(
                    "issue_links", key, ckpt, held[relation], jira_evidence(ctx.issues[key], mine), later,
                    issue=key, relation=relation, preferred=removed or later,
                )))  # fmt: skip
        # one relation per issue: a hashed pick over the (relation, value, trap) groups
        picked = ctx.one_per_group("issue_links", key, options)
        if picked:
            yield ordered(picked, ctx.seed + "issue_links" + key, lambda t: t["relation"])[0]


def fix_versions(rows):
    """Forward replay from empty; the caller checks it meets the export's backward replay."""
    forward = set()
    for r in rows:
        forward.discard(r["from"])
        if r["to"]:
            forward.add(r["to"])
    return sorted(v for v in forward if v)


def fix_version_at_snapshot(ctx: Context):
    for key in ctx.keys():
        rows, options = ctx.log(key, "Fix Version"), defaultdict(list)
        if not rows:
            continue
        for ckpt, _, when in ctx.limits:
            seen = ctx.visible(rows, when)
            if not seen or seen[-1]["when"] < ctx.start:
                continue
            value = fix_versions(seen)
            backward = ctx.jira.many(
                key, "fixVersions", "fix version", when - timedelta(microseconds=1)
            )
            if not value or value != backward:
                continue  # empty answers or an inconsistent export are not asked
            later = any(r["when"] >= when for r in rows)
            options[(canonical(value), later)].append((ckpt, ctx.task(
                "fix_version_at_snapshot", key, ckpt, value, jira_evidence(ctx.issues[key], seen), later,
                issue=key, preferred=len(seen) >= 2,
            )))  # fmt: skip
        yield from ctx.one_per_group("fix_version_at_snapshot", key, options)


def order_and_days(resolved_at, merged):
    """({"first", "days"}) for a full-precision resolution instant and a merge instant."""
    delta = timestamp(merged) - resolved_at
    first = "resolution" if delta.total_seconds() > 0 else "merge"
    return {"first": first, "days": int(abs(delta.total_seconds()) // 86400)}


def resolution_vs_merge_order(ctx: Context):
    for key in ctx.keys():
        rows = ctx.log(key, "resolution")
        numbers = ctx.by_issue.get(key, [])
        prs = [(n, ctx.facts["prs"][n]) for n in numbers]
        if not rows or not prs or any(pr["tags"] != sorted(pr["issues"]) for _, pr in prs):
            continue
        options = defaultdict(list)
        for ckpt, limit, when in ctx.limits:
            seen = ctx.visible(rows, when)
            merged = [(n, pr) for n, pr in prs if merged_at(pr, limit)]
            if not seen or len(merged) != 1:
                continue
            setting = [r for r in seen if r["to"]]
            if len(setting) != 1:
                continue
            number, pr = merged[0]
            merge_time = merged_at(pr, limit)
            if timestamp(merge_time) == setting[0]["when"]:
                continue
            if max(setting[0]["when"], timestamp(merge_time)) < ctx.start:
                continue
            value = order_and_days(setting[0]["when"], merge_time)
            evidence = jira_evidence(ctx.issues[key], setting) + [i for at, kind, i in pr["events"] if kind == "merged" and at < limit]  # fmt: skip
            options[(value["first"], False)].append((ckpt, ctx.task(
                "resolution_vs_merge_order", key, ckpt, value, evidence, False, issue=key, pr=int(number), preferred=True,
            )))  # fmt: skip
        yield from ctx.one_per_group("resolution_vs_merge_order", key, options)


def blockers(ctx: Context, when, keys):
    """{component: [keys]} of the issues (among `keys`) that are Blocker, unassigned and not
    Resolved/Closed at `when`; a smoke slice is read as of its end, exactly like its replica."""
    found = defaultdict(list)
    when = min(when, ctx.end) if ctx.end else when
    for key in keys:
        created = timestamp(ctx.issues[key]["fields"]["created"])
        if created > when or (ctx.since and created < ctx.since):
            continue
        if ctx.jira.value(key, "priority", when) != "Blocker" or ctx.jira.value(
            key, "assignee", when
        ):
            continue
        if ctx.jira.value(key, "status", when) in {"Resolved", "Closed"}:
            continue
        for component in ctx.jira.many(key, "components", "component", when):
            found[component].append(key)
    return {c: sorted(v) for c, v in found.items()}


def ever_blocker(ctx: Context):
    """Issues that were Blocker at some point: the only ones `blockers` needs to replay."""
    keep = []
    for key in ctx.keys():
        fields = ctx.issues[key]["fields"]
        if stream._name(fields.get("priority")) == "Blocker" or any(
            "Blocker" in (item.get("fromString"), item.get("toString"))
            for h in ctx.issues[key]["changelog"]["histories"] for item in h["items"] if item["field"] == "priority"
        ):  # fmt: skip
            keep.append(key)
    return keep


def open_blockers_unassigned(ctx: Context):
    """One candidate per (component, year); the same component may recur in another year."""
    low, high = BLOCKERS
    options, narrow = defaultdict(list), ever_blocker(ctx)
    for ckpt, _, when in ctx.limits:
        for component, keys in blockers(ctx, when - timedelta(microseconds=1), narrow).items():
            if not low <= len(keys) <= high:
                continue
            options[(component, ckpt[:4])].append((ckpt, ctx.task(
                "open_blockers_unassigned", f"component:{component}:{ckpt}", ckpt, keys,
                [f"jira:{ctx.issues[k]['id']}:created" for k in keys], False, component=component,
                slot=f"blockers:{component}:{ckpt}", preferred=True,
            )))  # fmt: skip
    for (component, year), found in sorted(options.items()):
        yield from ctx.one_per_group(
            "open_blockers_unassigned", component + year, {(component, year): found}
        )


def author_prs(ctx: Context):
    by_author = defaultdict(list)
    for number, pr in ctx.facts["prs"].items():
        when = merged_at(pr)
        if pr["author"] and when:
            by_author[pr["author"]].append((when, number))
    return {a: sorted(v) for a, v in by_author.items()}


def author_activity(ctx: Context):
    for login, merges in sorted(author_prs(ctx).items()):
        options = defaultdict(list)
        for ckpt, limit, when in ctx.limits:
            since = iso(when - timedelta(days=DAYS["author_activity"]))
            mine = [(at, n) for at, n in merges if since <= at < limit]
            if not mine:
                continue
            evidence = [i for _, n in mine for at, kind, i in ctx.facts["prs"][n]["events"] if kind == "merged"]  # fmt: skip
            options[(len(mine), False)].append((ckpt, ctx.task(
                "author_activity", f"login:{login}:{ckpt}", ckpt, len(mine), evidence, False,
                login=login, since=since, preferred=len(mine) >= 2,
            )))  # fmt: skip
        yield from ctx.one_per_group("author_activity", login, options)


def top_reviewer(ctx: Context, login, merges, limit, since):
    """(login, prs approved) of the reviewer approving most of the author's merged PRs, or None."""
    counts = Counter()
    for at, number in merges:
        if not since <= at < limit:
            continue
        for approver in approvers(ctx.facts["reviews"].get(number, []), limit, author=login):
            counts[approver] += 1
    top = counts.most_common(2)
    if not top or (len(top) > 1 and top[0][1] == top[1][1]):
        return None
    return top[0]


def top_reviewer_for_author(ctx: Context):
    for login, merges in sorted(author_prs(ctx).items()):
        options = defaultdict(list)
        for ckpt, limit, when in ctx.limits:
            since = iso(when - timedelta(days=DAYS["top_reviewer_for_author"]))
            top = top_reviewer(ctx, login, merges, limit, since)
            if not top:
                continue
            evidence = [i for at, n in merges if since <= at < limit for a, who, s, i in ctx.facts["reviews"].get(n, []) if a < limit and s == "approved" and who == top[0]]  # fmt: skip
            options[(top[0], False)].append((ckpt, ctx.task(
                "top_reviewer_for_author", f"reviewer:{login}:{ckpt}", ckpt, top[0], evidence, False,
                login=login, since=since, preferred=top[1] >= 2,
            )))  # fmt: skip
        yield from ctx.one_per_group("top_reviewer_for_author", login, options)


def discussions(mails):
    """{KIP: messages of its DISCUSS thread}: the thread holding the earliest message (mails
    are time-sorted) whose subject names "[DISCUSS] KIP-N"."""
    threads, first = defaultdict(list), {}
    for m in mails:
        threads[m[2]].append(m)
        if found := DISCUSS.search(m[4]):
            first.setdefault(found.group(1).upper(), m[2])
    return {kip: threads[thread] for kip, thread in first.items()}


def discussion(mails, kip, limit):
    """(thread messages before `limit`, later messages exist) of the KIP's DISCUSS thread."""
    rows = discussions(mails).get(kip)
    if not rows:
        return None, False
    return [m for m in rows if m[0] < limit], any(m[0] >= limit for m in rows)


def kip_thread(ctx: Context):
    threads = discussions(ctx.facts["mails"])
    for kip in sorted(threads, key=lambda k: int(k[4:])):
        options = defaultdict(list)
        for ckpt, limit, _ in ctx.limits:
            rows = [m for m in threads[kip] if m[0] < limit]
            later = len(rows) < len(threads[kip])
            if len(rows) < MIN_THREAD or not ctx.in_window(rows[-1][0]):
                continue
            if not any((f := DISCUSS.search(m[4])) and f.group(1).upper() == kip for m in rows):
                continue
            value = {"starter": rows[0][3], "participants": len({m[3] for m in rows})}
            options[(canonical(value), later)].append((ckpt, ctx.task(
                "kip_thread", kip, ckpt, value, [m[1] for m in rows], later, kip=kip, thread=rows[0][2], preferred=True,
            )))  # fmt: skip
        yield from ctx.one_per_group("kip_thread", kip, options)


GENERATORS = {
    "issue_prs": issue_prs, "issue_changed_files": issue_changed_files, "pr_approvers": pr_approvers,
    "review_hotspot": review_hotspot, "reopen_count": reopen_count, "issue_links": issue_links,
    "fix_version_at_snapshot": fix_version_at_snapshot, "resolution_vs_merge_order": resolution_vs_merge_order,
    "open_blockers_unassigned": open_blockers_unassigned, "author_activity": author_activity,
    "top_reviewer_for_author": top_reviewer_for_author, "kip_thread": kip_thread,
}  # fmt: skip


# --- selection --------------------------------------------------------------------------------


def reserved(task):
    """Ids a task takes out of circulation: its issue(s), PR, login, component or KIP."""
    ids = {task["issue"]} if task.get("issue") else set()
    ids |= set(task.get("issues") or [])
    for name in ("pr", "login", "kip", "slot"):
        if task.get(name) is not None:
            ids.add(f"{name}:{task[name]}")
    return ids


def pick(pools, count, taken, accept=None):
    """First `count` candidates over quota'd pools; nothing reserved twice; `accept` may veto."""
    chosen = []
    for pool, quota in pools:
        for task in pool:
            if len(chosen) >= min(quota, count):
                break
            ids = reserved(task)
            if ids & taken or (accept and not accept(task)):
                continue
            chosen.append(task)
            taken |= ids
    return chosen


def select(family, found, count, taken, seed, accept=None):
    """Stratified by cutoff year; preferred and trapped candidates first (traps aim at 70 %)."""

    def pool(rows):
        return stratified(
            rows, seed + family, lambda t: t["id"] + t["checkpoint"], lambda t: (t["cutoff"][:4],)
        )

    if family == "reopen_count":
        many = pool([t for t in found if t["expected"]["value"] >= 2])
        some = pool([t for t in found if t["expected"]["value"] == 1])
        none = pool([t for t in found if t["expected"]["value"] == 0])
        half = math.ceil(count / 2)
        return pick([(many, min(3, half)), (some, half), (none, count), (many, count), (some, count)], count, taken, accept)  # fmt: skip
    quota = math.ceil(TRAP_TARGET * count)
    tiers = [
        (pool([t for t in found if t["preferred"] and t["temporal_trap"]]), quota),
        (pool([t for t in found if t["preferred"]]), count),
        (pool([t for t in found if t["temporal_trap"]]), quota),
        (pool(found), count),
    ]
    return pick(tiers, count, taken, accept)


def candidates(ctx: Context, counts):
    """{variant: (count, candidates)} for `counts` = {family: n}; author_activity splits its
    count over its two variants."""
    return {
        variant: (
            counts[family] // len(VARIANTS.get(family, (family,))),
            list(GENERATORS[variant](ctx)),
        )
        for family in counts
        for variant in VARIANTS.get(family, (family,))
    }


def choose(family, found, count, taken, seed, verify=None, cache=None):
    """Select `count` tasks of one family from its candidates; changed-file tasks must survive
    `verify` (the bare clone reproduces the stream's file list)."""
    found = [t for t in found if not reserved(t) & taken]
    verify = verify if verify is not None else verify_files
    accept = (lambda t: verify(t, cache if cache is not None else {})) if family == "issue_changed_files" else None  # fmt: skip
    return select(family, found, count, taken, seed, accept), len(found)


# --- wording ----------------------------------------------------------------------------------


def question(task, ref):
    family = task["family"]
    if family == "issue_prs":
        return (
            f"Which pull requests had been opened for {ref} by the snapshot, and what was each one's "
            'state then? Return an array of objects {"pr": <number as an integer>, "state": '
            '"open"|"closed"|"merged"} sorted by pr ascending; closed means closed without merging.'
        )
    if family == "issue_changed_files":
        return (
            f"Which files were changed by the pull requests merged for {ref} by the snapshot? Return "
            "the sorted array of repository-relative file paths (the union over those merged pull "
            "requests, each path once); pull requests not merged by the snapshot do not count."
        )
    if family == "pr_approvers":
        return f"Who had approved {ref} by the snapshot? Return the sorted array of distinct GitHub logins whose review state was approved."  # fmt: skip
    if family == "review_hotspot":
        return f"Which file received the most inline review comments on {ref} by the snapshot? Return its repository-relative path as a string."  # fmt: skip
    if family == "reopen_count":
        return f"How many times had {ref} been transitioned to the workflow status Reopened by the snapshot? Return an integer (0 if never)."  # fmt: skip
    if family == "issue_links":
        return (
            f'Which issues was {ref} linked to with the issue-link relation "{task["relation"]}" (as '
            f'JIRA words it: "This issue {task["relation"]} <KEY>") at the snapshot? Return the sorted '
            "array of issue keys; a link removed before the snapshot does not count."
        )
    if family == "fix_version_at_snapshot":
        return f'Which fix versions were set on {ref} at the snapshot? Return the sorted array of version names as strings (e.g. "3.6.0").'  # fmt: skip
    if family == "resolution_vs_merge_order":
        return (
            f"For {ref}, exactly one pull request had been merged and the resolution had been set "
            "by the snapshot. Which happened first, and how many whole days lay between the two "
            'instants? Return {"first": "resolution"|"merge", "days": <integer: the absolute '
            "difference floored to whole days>}."
        )
    if family == "open_blockers_unassigned":
        return (
            f'Which issues in component "{task["component"]}" were, at the snapshot, of priority '
            "Blocker, unassigned, and in a workflow status other than Resolved and Closed? Return "
            "the sorted array of issue keys."
        )
    if family == "author_activity":
        return (
            f"How many pull requests authored by the GitHub user {task['login']} were merged in the "
            f"90 days before the snapshot (merged at or after {task['since']}, through the cutoff)? "
            "Return an integer."
        )
    if family == "top_reviewer_for_author":
        return (
            f"Among the pull requests authored by the GitHub user {task['login']} and merged in the "
            f"365 days before the snapshot (merged at or after {task['since']}, through the cutoff), "
            "which GitHub login approved the largest number of them (approvals through the cutoff, "
            "one count per pull request, the author's own reviews excluded)? Return the login as a string."
        )
    if family == "kip_thread":
        return (
            f"Consider the mailing-list discussion of {task['kip']}: the thread holding the earliest "
            f'message whose subject contains "[DISCUSS] {task["kip"]}". Who sent that thread\'s first '
            "message, and how many distinct senders had posted in it by the snapshot? Return "
            '{"starter": <sender name exactly as recorded>, "participants": <integer>}.'
        )
    raise KeyError(family)


def answer_schema(family):
    string = {"type": "string"}
    strings = {"type": "array", "items": string}
    values = {
        "issue_prs": {"type": "array", "items": {"type": "object", "properties": {"pr": {"type": "integer"}, "state": {"type": "string", "enum": ["open", "closed", "merged"]}}, "required": ["pr", "state"], "additionalProperties": False}},
        "issue_changed_files": strings, "pr_approvers": strings, "issue_links": strings,
        "fix_version_at_snapshot": strings, "open_blockers_unassigned": strings,
        "review_hotspot": string, "top_reviewer_for_author": string,
        "reopen_count": {"type": "integer"}, "author_activity": {"type": "integer"},
        "resolution_vs_merge_order": {"type": "object", "properties": {"first": {"type": "string", "enum": ["resolution", "merge"]}, "days": {"type": "integer"}}, "required": ["first", "days"], "additionalProperties": False},
        "kip_thread": {"type": "object", "properties": {"starter": string, "participants": {"type": "integer"}}, "required": ["starter", "participants"], "additionalProperties": False},
    }  # fmt: skip
    return {
        "type": "object",
        "properties": {
            "value": values[family],
            "evidence": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["value", "evidence"],
        "additionalProperties": False,
    }
