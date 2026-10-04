"""Answerability of the RAW exports: hand-written SQL per task family, run through the read
API (`query.open`) at the task's checkpoint, graded by the benchmark's own grader.

    python -m harnext_eval.e2.v2.stores.raw.oracle [--bench benchmarks/kafka-50-w2026] \\
        [--stores out/kafka-500-v2/stores-w2026/RAW] [--out <report.json>]

The SQL sees only what the public prompt says (issue key or title, PR number or title, login,
dates, KIP, thread subject); the private gold is used for grading only. `design_decision` has
no SQL-only derivation (the answer is a phrase chosen by reading a thread): for it the report
checks that the settling message is in the export and contains the gold value (`present`).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from harnext_eval.e2.v2.grade import grade_historical
from harnext_eval.e2.v2.stores.base import ROOT
from harnext_eval.e2.v2.stores.raw import query

KEY = re.compile(r"\bKAFKA-\d+\b")
TITLED = re.compile(r'titled "(.+?)" \((?:created|opened) in (\d{4}-\d\d)\)')
FIELDS = {"assignee": "assignee", "status": "status", "resolution": "resolution", "priority": "priority"}  # fmt: skip
SNAPSHOT = {"assignee_at_snapshot": "assignee", "status_at_snapshot": "status", "resolution_at_snapshot": "resolution",
            "closed_at_snapshot": "status", "priority_at_snapshot": "priority"}  # fmt: skip
HISTORY = {"assignment_history": "assignee", "status_transition_history": "status", "resolution_transition_history": "resolution"}  # fmt: skip
SUMMARIES = (
    "SELECT issue_key, summary AS s, time FROM jira_issues UNION ALL "
    "SELECT issue_key, issue_summary, time FROM jira_changes WHERE issue_summary IS NOT NULL UNION ALL "
    "SELECT issue_key, to_value, time FROM jira_changes WHERE field = 'summary' UNION ALL "
    "SELECT issue_key, issue_summary, time FROM jira_comments WHERE issue_summary IS NOT NULL"
)
OPENED = "SELECT pr_number FROM github_pull_requests WHERE action = 'opened' AND has(issues, {k})"


def q(value) -> str:
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


class Run:
    """One task's SQL trail: every statement and its result rows."""

    def __init__(self, db):
        self.db, self.trail = db, []

    def rows(self, sql):
        found = self.db.sql(sql, max_rows=10_000, max_bytes=5_000_000)
        self.trail.append(sql)
        if found["error"]:
            raise LookupError(found["error"])
        return found["rows"]

    def one(self, sql):
        rows = self.rows(sql)
        return next(iter(rows[0].values())) if rows else None


def issue(run, prompt):
    """The issue a prompt names: its key, or the unique issue whose latest summary is the title."""
    if found := TITLED.search(prompt):
        title, month = found.groups()
        norm = "lower(trim(replaceRegexpAll(s, '\\\\s+', ' ')))"
        keys = [r["issue_key"] for r in run.rows(
            f"SELECT issue_key FROM ({SUMMARIES}) GROUP BY issue_key "
            f"HAVING argMax({norm}, time) = lower({q(' '.join(title.split()))}) ORDER BY issue_key")]  # fmt: skip
        if len(keys) > 1:  # several issues once bore that title: the one created in the named month
            created = run.rows(f"SELECT issue_key FROM jira_issues WHERE issue_key IN ({', '.join(map(q, keys))}) AND formatDateTime(time, '%Y-%m') = {q(month)}")  # fmt: skip
            keys = [r["issue_key"] for r in created] or keys
        if len(keys) != 1:
            raise LookupError(f"title resolves to {keys}")
        return keys[0]
    return KEY.search(prompt).group(0)


def pull(run, prompt):
    if found := TITLED.search(prompt):
        title, month = found.groups()
        return run.one(f"SELECT pr_number FROM github_pull_requests WHERE action = 'opened' AND lower(pr_title) = lower({q(title)}) AND formatDateTime(time, '%Y-%m') = {q(month)}")  # fmt: skip
    return int(re.search(r"pull request #(\d+)", prompt).group(1))


def latest(run, key, field):
    sql = f"SELECT argMax(to_value, (at, history_id, item_index)) AS v, count() AS n FROM jira_changes WHERE issue_key = {q(key)} AND field = {q(field)}"  # fmt: skip
    row = run.rows(sql)[0]
    if row["n"]:
        return row["v"]
    if field == "resolution":  # never set: unresolved
        return None
    rows = run.rows(f"SELECT `{field}` AS v FROM jira_issues WHERE issue_key = {q(key)}")
    if not rows:
        raise LookupError(f"no {field} change and no creation record of {key} in the store")
    return rows[0]["v"]


def replay(run, key, field, value, extra=""):
    """Values held after replaying adds (to) and removals (from) in changelog order."""
    return [r["v"] for r in run.rows(
        f"SELECT v FROM (SELECT {value[0]} AS v, 1 AS op, at, history_id, item_index FROM jira_changes WHERE issue_key = {q(key)} AND field = {q(field)} {extra} AND {value[0]} IS NOT NULL "
        f"UNION ALL SELECT {value[1]}, 0, at, history_id, item_index FROM jira_changes WHERE issue_key = {q(key)} AND field = {q(field)} {extra} AND {value[1]} IS NOT NULL) "
        "GROUP BY v HAVING argMax(op, (at, history_id, item_index, op)) = 1 ORDER BY v")]  # fmt: skip


def answer(db, task) -> tuple:
    """(value, SQL trail) for one public task; LookupError when the store cannot tell."""
    run, family = Run(db), task["family"]
    prompt = task["prompt"].split("\n\n")[1]
    if family in SNAPSHOT:
        value = latest(run, issue(run, prompt), SNAPSHOT[family])
        return (value == "Closed" if family == "closed_at_snapshot" else value), run.trail
    if family in HISTORY:
        rows = run.rows(f"SELECT formatDateTime(at, '%Y-%m-%dT%H:%i:%S.%fZ') AS at, from_value, to_value FROM jira_changes WHERE issue_key = {q(issue(run, prompt))} AND field = {q(HISTORY[family])} ORDER BY at, history_id, item_index")  # fmt: skip
        return [
            {"at": r["at"], "from": r["from_value"], "to": r["to_value"]} for r in rows
        ], run.trail
    if family == "reopen_count":
        return run.one(f"SELECT count() FROM jira_changes WHERE issue_key = {q(issue(run, prompt))} AND field = 'status' AND to_value = 'Reopened'"), run.trail  # fmt: skip
    if family == "fix_version_at_snapshot":
        return replay(run, issue(run, prompt), "fixVersion", ("to_value", "from_value")), run.trail
    if family == "issue_links":  # latest action per linked issue, in changelog order
        relation = re.search(r'relation "(.+?)"', prompt).group(1)
        return run.one(
            "SELECT arraySort(groupArray(linked_issue)) FROM (SELECT linked_issue, argMax(link_action, (at, history_id, item_index)) AS last "
            f"FROM jira_changes WHERE issue_key = {q(issue(run, prompt))} AND field = 'Link' AND link_relation = {q(relation)} "
            "GROUP BY linked_issue) WHERE last = 'added'"), run.trail  # fmt: skip
    if family == "issue_prs":
        key = issue(run, prompt)
        rows = run.rows(f"SELECT pr_number, argMax(action, time) AS state FROM github_pull_requests WHERE pr_number IN ({OPENED.format(k=q(key))}) GROUP BY pr_number ORDER BY pr_number")  # fmt: skip
        return [{"pr": r["pr_number"], "state": {"opened": "open"}.get(r["state"], r["state"])} for r in rows], run.trail  # fmt: skip
    if family == "issue_changed_files":
        key = issue(run, prompt)
        return run.one(f"SELECT arraySort(groupUniqArrayArray(changed_files)) FROM github_pull_requests WHERE action = 'merged' AND pr_number IN ({OPENED.format(k=q(key))})"), run.trail  # fmt: skip
    if family == "pr_approvers":
        number = pull(run, prompt)
        return run.one(f"SELECT arraySort(groupUniqArray(reviewer)) FROM github_reviews WHERE pr_number = {number} AND state = 'approved'"), run.trail  # fmt: skip
    if family == "review_hotspot":
        return run.one(f"SELECT path FROM github_review_comments WHERE pr_number = {pull(run, prompt)} GROUP BY path ORDER BY count() DESC, path LIMIT 1"), run.trail  # fmt: skip
    if family == "resolution_vs_merge_order":
        key = issue(run, prompt)
        row = run.rows(
            f"SELECT (SELECT any(at) FROM jira_changes WHERE issue_key = {q(key)} AND field = 'resolution' AND to_value IS NOT NULL) AS resolved, "
            f"(SELECT any(time) FROM github_pull_requests WHERE action = 'merged' AND pr_number IN ({OPENED.format(k=q(key))})) AS merged, "
            "dateDiff('microsecond', resolved, merged) AS delta")[0]  # fmt: skip
        return {"first": "resolution" if row["delta"] > 0 else "merge", "days": abs(row["delta"]) // 86_400_000_000}, run.trail  # fmt: skip
    if family == "author_activity":
        login, since = re.search(r"GitHub user (\S+) were merged .*?at or after (\S+?Z)", prompt).groups()  # fmt: skip
        return run.one(f"SELECT count() FROM github_pull_requests WHERE action = 'merged' AND author = {q(login)} AND time >= {q(since.rstrip('Z'))}"), run.trail  # fmt: skip
    if family == "top_reviewer_for_author":
        login, since = re.search(r"GitHub user (\S+) and merged .*?at or after (\S+?Z)", prompt).groups()  # fmt: skip
        rows = run.rows(
            "SELECT r.reviewer AS reviewer, uniqExact(r.pr_number) AS prs FROM github_reviews AS r INNER JOIN "
            f"(SELECT DISTINCT pr_number FROM github_pull_requests WHERE action = 'merged' AND author = {q(login)} AND time >= {q(since.rstrip('Z'))}) AS p "
            f"ON r.pr_number = p.pr_number WHERE r.state = 'approved' AND r.reviewer != {q(login)} GROUP BY reviewer ORDER BY prs DESC, reviewer LIMIT 2")  # fmt: skip
        if len(rows) > 1 and rows[0]["prs"] == rows[1]["prs"]:
            raise LookupError(f"tie between {rows[0]['reviewer']} and {rows[1]['reviewer']}")
        return (rows[0]["reviewer"] if rows else None), run.trail
    if family == "kip_thread":
        kip = re.search(r"discussion of (KIP-\d+)", prompt).group(1)
        thread = run.one(f"SELECT thread_id FROM mail_messages WHERE match(subject, {q('(?i)\\[DISCUSS\\]\\s*' + kip + '([^0-9]|$)')}) ORDER BY time, event_id LIMIT 1")  # fmt: skip
        row = run.rows(f"SELECT argMin(sender, (time, event_id)) AS starter, uniqExact(sender) AS participants FROM mail_messages WHERE thread_id = {q(thread)}")[0]  # fmt: skip
        return {"starter": row["starter"], "participants": row["participants"]}, run.trail
    if family == "implicit_link":
        return implicit(run, prompt), run.trail
    raise LookupError(f"no SQL oracle for {family}")


def implicit(run, prompt):
    """The JIRA issue a mail thread is about: an issue whose summary names the thread's KIP,
    else the issue whose summary is most similar to the stripped subject."""
    sender, subject, day = re.search(r'list, (.+?) started a thread with the subject "(.+)" on (\d{4}-\d\d-\d\d)', prompt).groups()  # fmt: skip
    latest_summaries = (
        f"SELECT issue_key, argMax(s, time) AS summary FROM ({SUMMARIES}) GROUP BY issue_key"
    )
    if kip := re.search(r"KIP-\d+", subject):
        keys = [r["issue_key"] for r in run.rows(f"SELECT issue_key FROM ({latest_summaries}) WHERE match(summary, {q(kip.group(0) + '([^0-9]|$)')}) ORDER BY issue_key")]  # fmt: skip
        if len(keys) == 1:
            return keys[0]
    topic = re.sub(r"(?i)^((re|fwd?):\s*|\[discuss\]\s*|kip-\d+:?\s*)+", "", subject).strip()
    return run.one(f"SELECT issue_key FROM ({latest_summaries}) ORDER BY ngramDistanceCaseInsensitiveUTF8(summary, {q(topic)}), issue_key LIMIT 1")  # fmt: skip


def plain(text):
    return " ".join(re.sub(r"[*_`]", "", text or "").split()).casefold()


def present(db, private) -> dict:
    """design_decision: the settling message(s) are in the export; is the gold value verbatim?"""
    ids = ", ".join(
        q(i if i.startswith("mail:") else f"mail:{i}") for i in private["final_evidence"]
    )
    rows = db.sql(f"SELECT event_id, body FROM mail_messages WHERE event_id IN ({ids})", max_bytes=5_000_000)["rows"]  # fmt: skip
    verbatim = any(plain(str(private["expected"]["value"])) in plain(r["body"]) for r in rows)
    reason = "" if rows and verbatim else "settling message not in the export" if not rows else "message present; gold value paraphrased, not verbatim"  # fmt: skip
    return {"mode": "present", "ok": bool(rows), "verbatim": verbatim, "reason": reason}


def check(export_root: Path, public: list[dict], private: dict) -> dict:
    """{task id: record} for every historical task, one export opened at a time."""
    found = {}
    by_ckpt = defaultdict(list)
    for task in public:
        if task["kind"] == "historical":
            by_ckpt[task["checkpoint"]].append(task)
    for ckpt, tasks in sorted(by_ckpt.items()):
        with query.open(export_root / "export" / ckpt) as db:
            for task in tasks:
                gold = private[task["id"]]
                record = {"family": task["family"], "checkpoint": ckpt, "expected": gold["expected"]["value"]}  # fmt: skip
                if task["family"] == "design_decision":
                    found[task["id"]] = record | present(db, gold)
                    continue
                try:
                    value, trail = answer(db, task)
                except LookupError as error:
                    found[task["id"]] = record | {"mode": "derived", "ok": False, "reason": str(error)}  # fmt: skip
                    continue
                ok = grade_historical(gold, {"value": value})["correct"]
                found[task["id"]] = record | {"mode": "derived", "ok": ok, "value": value, "sql": trail,
                                              "reason": "" if ok else "derived value differs from gold"}  # fmt: skip
    return found


def summary(found: dict) -> dict:
    families = defaultdict(Counter)
    for record in found.values():
        families[record["family"]]["tasks"] += 1
        families[record["family"]]["ok"] += record["ok"]
        families[record["family"]]["verbatim"] += record.get("verbatim", False)
    return {f: dict(c) for f, c in sorted(families.items())}


def load(bench: Path):
    public = [json.loads(line) for line in (bench / "tasks.public.jsonl").open()]
    private = {t["id"]: t for t in map(json.loads, (bench / "tasks.private.jsonl").open())}
    return public, private


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--bench", type=Path, default=ROOT / "benchmarks/kafka-50-w2026")
    parser.add_argument("--stores", type=Path, default=ROOT / "out/kafka-500-v2/stores-w2026/RAW")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    found = check(args.stores, *load(args.bench))
    report = {"bench": str(args.bench), "stores": str(args.stores), "families": summary(found), "tasks": found}  # fmt: skip
    text = json.dumps(report, indent=1, sort_keys=True, default=str)
    if args.out:
        args.out.write_text(text + "\n")
    for family, counts in report["families"].items():
        extra = (
            f" (gold verbatim in it: {counts.get('verbatim', 0)})"
            if family == "design_decision"
            else ""
        )
        print(f"{family:32s} {counts.get('ok', 0)}/{counts['tasks']}{extra}")
    for task_id, record in sorted(found.items()):
        if not record["ok"] or record.get("reason"):
            print(f"  MISS {task_id} ({record['family']}, {record['checkpoint']}): {record['reason']}; expected {json.dumps(record['expected'])[:120]}"  # fmt: skip
                  + (f", got {json.dumps(record.get('value'), default=str)[:120]}" if "value" in record else ""))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
