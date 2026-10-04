"""Answerability audit for arm `S-tools` (E2-V2-DESIGN.md "Genuine baseline").

    python -m harnext_eval.e2.v2.sources.audit --scope smoke|full

Historical: a scripted oracle that reads only the public prompt and shells out only to the
`jira`/`gh`/`mail` CLIs against the task's checkpoint database must reproduce the gold answer
through the run's own grader. Keyless tasks first find the key with `jira issue list --summary`
(issues) or `gh pr list -S "title"` (pull requests). Coding:
the issue must be viewable; the share of golden-patch files with history before the base
commit or a mention in the sources is reported. Every checkpoint database is also scanned for
an instant at or after its boundary. A task the oracle cannot answer is a BENCHMARK DEFECT: it
is listed with its reason in answerability.json, and the gold is never touched.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

from harnext_eval.e2 import kafka_coding as kc
from harnext_eval.e2.benchmark import ROOT, timestamp
from harnext_eval.e2.v2 import grade
from harnext_eval.e2.v2.sources import cli, index

OUT = ROOT / "out/kafka-500-v2"
SCOPES = {
    "smoke": (ROOT / "benchmarks/kafka-500-v2-smoke", OUT / "stores-smoke"),
    "full": (ROOT / "benchmarks/kafka-500-v2", OUT / "stores"),
    # The systems of record keep their whole history: w2026 audits against the full replicas.
    "w2026": (ROOT / "benchmarks/kafka-50-w2026", OUT / "stores"),
}
FIELD = {"assignee": "assignee", "status": "status", "resolution": "resolution", "closed": "status", "priority": "priority", "assignment": "assignee"}  # fmt: skip
ISSUE_FAMILIES = {"issue_prs", "issue_changed_files", "reopen_count", "issue_links", "fix_version_at_snapshot", "resolution_vs_merge_order"}  # fmt: skip
TITLED = re.compile(r'JIRA issue titled "(.*)" \(created in (\d{4}-\d\d)\)')
TITLED_PR = re.compile(r'pull request titled "(.*)" \(opened in (\d{4}-\d\d)\)')


class Defect(Exception):  # noqa: N818
    pass


def tool(name, database, *args):
    """One CLI call, exactly as an agent in the container makes it."""
    env = {**os.environ, "HARNEXT_SOURCES": str(database)}
    done = subprocess.run([sys.executable, cli.__file__, name, *args], env=env, capture_output=True, text=True)  # fmt: skip
    return done.returncode, done.stdout, done.stderr


def find_key(prompt, database):
    if found := re.search(r"\bKAFKA-\d+\b", prompt):
        return found.group(0)
    found = TITLED.search(prompt)
    if not found:
        raise Defect("the prompt names neither a key nor a title")
    title, month = found.groups()
    code, out, err = tool("jira", database, "issue", "list", "--summary", title, "--raw", "--limit", "200")  # fmt: skip
    if code:
        raise Defect(f"jira issue list failed: {err.strip()[:200]}")
    wanted = " ".join(title.split()).casefold()
    rows = [r for r in json.loads(out) if " ".join((r["summary"] or "").split()).casefold() == wanted]  # fmt: skip
    if len(rows) > 1:
        rows = [r for r in rows if (r["created"] or "")[:7] == month] or rows
    if len(rows) != 1:
        raise Defect(f"{len(rows)} issues carry the prompt's title as of the cutoff")
    return rows[0]["key"]


def find_pr(prompt, database):
    if found := re.search(r"pull request #(\d+)", prompt):
        return int(found.group(1))
    found = TITLED_PR.search(prompt)
    if not found:
        raise Defect("the prompt names neither a PR number nor a PR title")
    title, month = found.groups()
    rows = gh_json(database, "pr", "list", "-s", "all", "-S", f'"{title}"', "-L", "500", "--json", "number,title,createdAt")  # fmt: skip
    wanted = " ".join(title.split()).casefold()
    rows = [r for r in rows if " ".join((r["title"] or "").split()).casefold() == wanted]
    if len(rows) > 1:
        rows = [r for r in rows if (r["createdAt"] or "")[:7] == month] or rows
    if len(rows) != 1:
        raise Defect(f"{len(rows)} pull requests carry the prompt's title as of the cutoff")
    return rows[0]["number"]


def call(name, database, *args):
    code, out, err = tool(name, database, *args)
    if code:
        raise Defect(f"{name} {' '.join(args[:3])} failed: {err.strip()[:200]}")
    return out


def gh_json(database, *args):
    return json.loads(call("gh", database, *args))


def view_issue(key, database):
    return json.loads(call("jira", database, "issue", "view", key, "--history", "--raw"))


def issue_pulls(key, issue, database):
    """[(number, pr json)] of the issue's PRs that name it and were opened by the cutoff."""
    found = []
    for number in issue["pullRequests"]:
        pr = gh_json(database, "pr", "view", str(number), "--json", "number,state,createdAt,mergedAt,issues,author")  # fmt: skip
        if pr["createdAt"] and key in pr["issues"]:
            found.append((number, pr))
    return found


def merged_pr_files(number, database):
    return [line for line in call("gh", database, "pr", "diff", str(number)).splitlines() if line and not line.startswith("#")]  # fmt: skip


def parse_link(text):
    found = re.match(r"^This issue (.+?) ([A-Z][A-Z0-9]*-\d+)$", text or "")
    return found.groups() if found else None


def oracle_issue(task, key, database):
    """Issue-centred families, all from `jira issue view --history --raw` (+ `gh pr view/diff`)."""
    family, issue = task["family"], view_issue(key, database)
    if family == "issue_prs":
        return [
            {"pr": n, "state": pr["state"].lower()} for n, pr in issue_pulls(key, issue, database)
        ]
    if family == "issue_changed_files":
        merged = [n for n, pr in issue_pulls(key, issue, database) if pr["state"] == "MERGED"]
        return sorted({f for n in merged for f in merged_pr_files(n, database)})
    if family == "reopen_count":
        return sum(
            h["field"].lower() == "status" and h["to"] == "Reopened" for h in issue["history"]
        )
    if family == "issue_links":
        relation = re.search(r'issue-link relation "(.+?)"', task["prompt"]).group(1)  # type: ignore[union-attr]
        held = set()
        for h in issue["history"]:
            if h["field"] != "Link":
                continue
            parsed = parse_link(h["to"] or h["from"])
            if parsed and parsed[0] == relation:
                (held.add if h["to"] else held.discard)(parsed[1])
        return sorted(held)
    if family == "fix_version_at_snapshot":
        return sorted(issue["fixVersions"])
    if family == "resolution_vs_merge_order":
        setting = [h for h in issue["history"] if h["field"].lower() == "resolution" and h["to"]]
        merged = [pr for _, pr in issue_pulls(key, issue, database) if pr["state"] == "MERGED"]
        if len(setting) != 1 or len(merged) != 1:
            raise Defect(f"{len(setting)} resolution settings and {len(merged)} merged PRs visible")
        delta = timestamp(merged[0]["mergedAt"]) - timestamp(setting[0]["at"])
        return {"first": "resolution" if delta.total_seconds() > 0 else "merge", "days": int(abs(delta.total_seconds()) // 86400)}  # fmt: skip
    field = FIELD[family.split("_")[0]]
    if family.endswith("history"):
        return [{k: h[k] for k in ("at", "from", "to")} for h in issue["history"] if h["field"].lower() == field]  # fmt: skip
    if family == "closed_at_snapshot":
        return issue["status"] == "Closed"
    return issue[field]


def top(counter):
    best = counter.most_common(2)
    if not best or (len(best) > 1 and best[0][1] == best[1][1]):
        raise Defect("no unique maximum in the CLI output")
    return best[0][0]


def oracle_other(task, database):
    """PR-, author-, component- and KIP-centred families."""
    family, prompt = task["family"], task["prompt"]
    if family in {"pr_approvers", "review_hotspot"}:
        pr = gh_json(database, "pr", "view", str(find_pr(prompt, database)), "--json", "reviews,reviewComments")  # fmt: skip
        if family == "pr_approvers":
            return sorted({r["author"]["login"] for r in pr["reviews"] if r["state"] == "APPROVED" and r["author"]["login"]})  # fmt: skip
        return top(Counter(c["path"] for c in pr["reviewComments"] if c["path"]))
    if family == "open_blockers_unassigned":
        component = re.search(r'component "(.+?)"', prompt).group(1)  # type: ignore[union-attr]
        query = f'priority = Blocker AND assignee is EMPTY AND status != Resolved AND status != Closed AND component = "{component}"'  # fmt: skip
        rows = json.loads(call("jira", database, "issue", "list", "-q", query, "--limit", "100000", "--raw"))  # fmt: skip
        return sorted(r["key"] for r in rows)
    if family in {"author_activity", "top_reviewer_for_author"}:
        login = re.search(r"GitHub user (\S+) ", prompt).group(1)  # type: ignore[union-attr]
        since = re.search(r"merged at or after (\S+),", prompt).group(1)  # type: ignore[union-attr]
        rows = gh_json(database, "pr", "list", "-A", login, "-s", "merged", "-L", "100000", "--json", "number,mergedAt")  # fmt: skip
        mine = [r["number"] for r in rows if r["mergedAt"] and r["mergedAt"] >= since]
        if family == "author_activity":
            return len(mine)
        counts = Counter()
        for number in mine:
            pr = gh_json(database, "pr", "view", str(number), "--json", "reviews")
            counts.update({r["author"]["login"] for r in pr["reviews"] if r["state"] == "APPROVED" and r["author"]["login"] and r["author"]["login"] != login})  # fmt: skip
        return top(counts)
    if family == "kip_thread":
        kip = re.search(r"\bKIP-\d+\b", prompt).group(0)  # type: ignore[union-attr]
        pattern = re.compile(rf"\[DISCUSS\]\s*{kip}(?!\d)", re.IGNORECASE)
        hits = [line.split("\t", 3) for line in call("mail", database, "search", f"[DISCUSS] {kip}", "--limit", "100000").splitlines()]  # fmt: skip
        hits = sorted((at, id_) for id_, at, _, subject in hits if pattern.search(subject))
        if not hits:
            raise Defect(f"mail search finds no [DISCUSS] {kip} message")
        rows = [
            line.split("\t", 3)
            for line in call("mail", database, "thread", hits[0][1]).splitlines()
        ]
        return {"starter": rows[0][2], "participants": len({r[2] for r in rows})}
    raise Defect(f"no oracle for family {family}")


def oracle(task, database):
    """{"value", "evidence"} from CLI output alone."""
    if task["family"].split("_")[0] in FIELD or task["family"] in ISSUE_FAMILIES:
        key = find_key(task["prompt"], database)
        return {"value": oracle_issue(task, key, database), "evidence": [key]}
    return {"value": oracle_other(task, database), "evidence": []}


def leaks(database):
    """Instants at or after the boundary anywhere in the database (must be empty)."""
    db = sqlite3.connect(f"file:{database}?mode=ro&immutable=1", uri=True)
    limit = db.execute("SELECT value FROM meta WHERE key = 'boundary'").fetchone()[0]
    found = []
    for table, columns in index.TIMED.items():
        for column in columns:
            # full-precision JIRA instants and whole-second ones compare as strings up to seconds
            worst = db.execute(f"SELECT max(substr({column}, 1, 19)) FROM {table}").fetchone()[0]
            if worst and worst + "Z" >= limit:
                found.append(f"{table}.{column}={worst}")
    db.close()
    return found


def coding(task, private, database):
    """(issue viewable, [touched main files with prior history or a mention in the sources])."""
    key = private.get("issue") or re.search(r"KAFKA-\d+", task["prompt"]).group(0)  # type: ignore[union-attr]
    viewable = tool("jira", database, "issue", "view", key)[0] == 0
    paths = [p for p, _ in grade.split_patch(private["reference_patch"])]
    reachable = []
    for path in paths:
        prior = kc.git(
            "log", "-1", "--format=%H", task["base_sha"], "--", path, check=False
        ).strip()
        mentioned = tool("gh", database, "search", "prs", f'"{Path(path).name}"', "-L", "1")[1].strip()  # fmt: skip
        if prior or mentioned:
            reachable.append(path)
    return viewable, paths, reachable


def run(scope, stores=None, bench=None):
    bench_dir, stores_dir = SCOPES[scope]
    bench, stores = Path(bench or bench_dir), Path(stores or stores_dir)
    public = [json.loads(line) for line in (bench / "tasks.public.jsonl").open()]
    private = {t["id"]: t for t in map(json.loads, (bench / "tasks.private.jsonl").open())}
    families, defects, leaked, files = {}, [], {}, Counter()
    for task in public:
        database = stores / index.ARM / "export" / task["checkpoint"] / index.DB
        cell = families.setdefault(task["family"], {"tasks": 0, "pass": 0, "keyless": 0, "keyless_pass": 0})  # fmt: skip
        cell["tasks"] += 1
        cell["keyless"] += bool(task.get("keyless"))
        try:
            if not database.exists():
                raise Defect(f"no S-tools snapshot for checkpoint {task['checkpoint']}")
            if task["checkpoint"] not in leaked:
                leaked[task["checkpoint"]] = leaks(database)
            if task["kind"] == "coding":
                viewable, paths, reachable = coding(task, private[task["id"]], database)
                files.update(touched=len(paths), reachable=len(reachable), tasks_with_any=bool(reachable))  # fmt: skip
                if not viewable:
                    raise Defect("the issue is not viewable with `jira issue view`")
            else:
                answer = oracle(task, database)
                graded = grade.grade_historical(private[task["id"]], answer)
                if not graded["correct"]:
                    raise Defect(f"oracle answer differs from gold: got {json.dumps(answer['value'])[:300]}, expected {json.dumps(graded['expected'])[:300]}")  # fmt: skip
        except Defect as defect:
            defects.append({"task": task["id"], "family": task["family"], "checkpoint": task["checkpoint"], "keyless": bool(task.get("keyless")), "reason": str(defect)})  # fmt: skip
            continue
        cell["pass"] += 1
        cell["keyless_pass"] += bool(task.get("keyless"))
    report = {
        "benchmark": bench.name, "arm": index.ARM, "tasks": len(public),
        "pass": sum(c["pass"] for c in families.values()), "families": dict(sorted(families.items())),
        "defects": defects, "checkpoints_scanned": len(leaked),
        "instants_at_or_after_boundary": {c: v for c, v in sorted(leaked.items()) if v},
        "coding_files": dict(files),
        "oracle": "scripted; reads the public prompt and calls only the offline CLIs (`jira issue "
        "list`, `jira issue view --history --raw`, `gh pr list/view/diff`, `mail search/thread`), "
        "graded by grade.grade_historical",
    }  # fmt: skip
    (bench / "answerability.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")  # fmt: skip
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--scope", choices=sorted(SCOPES), required=True)
    parser.add_argument("--stores", type=Path)
    args = parser.parse_args(argv)
    report = run(args.scope, args.stores)
    print(json.dumps({k: v for k, v in report.items() if k != "defects"}, indent=2))
    for defect in report["defects"]:
        print(f"DEFECT {defect['task']}: {defect['reason']}")
    clean = not report["defects"] and not report["instants_at_or_after_boundary"]
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
