"""Deterministic kafka-500-v2 task records: 100 historical + up to 50 admitted coding.

Historical gold is recomputed from the raw JIRA changelog at a checkpoint boundary; coding
tasks are the admitted rows of `coding-candidates.jsonl`. Never runs an answering agent and
never pads: a shortfall is written down in the manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

from harnext_eval.e2.benchmark import (
    ROOT,
    canonical,
    changes,
    inventory,
    iso,
    ordered,
    read,
    save,
    sha,
    stratified,
    timestamp,
)
from harnext_eval.e2.v2 import families, stream

SEED = "kafka-500-v2"
TASKS_FROM = "2023-01-01T00:00:00Z"  # tasks are drawn from here on; the stream starts earlier
BENCH = ROOT / "benchmarks/kafka-500-v2"
ADMISSION = ROOT / "out/kafka-admission-v2/results"
# Historical part = 100 (cut from 250 on 2026-09-20 to fit the run budget): the 8 v1 field-replay
# families at 4 + the 11 families of families.py at 6 (issue_changed_files and pr_approvers, the
# two closest to coding work, at 7; author_activity = 3 counts + 3 top reviewers). Rebalance here.
COUNTS = {
    "assignee_at_snapshot": 4,
    "status_at_snapshot": 4,
    "resolution_at_snapshot": 4,
    "closed_at_snapshot": 4,
    "priority_at_snapshot": 4,
    "assignment_history": 4,
    "status_transition_history": 4,
    "resolution_transition_history": 4,
    "issue_prs": 6,
    "issue_changed_files": 7,
    "pr_approvers": 7,
    "review_hotspot": 6,
    "reopen_count": 6,
    "issue_links": 6,
    "fix_version_at_snapshot": 6,
    "resolution_vs_merge_order": 6,
    "open_blockers_unassigned": 6,
    "author_activity": 6,
    "kip_thread": 6,
}
FIELDS = {
    "assignee_at_snapshot": "assignee",
    "status_at_snapshot": "status",
    "resolution_at_snapshot": "resolution",
    "closed_at_snapshot": "status",
    "priority_at_snapshot": "priority",
    "assignment_history": "assignee",
    "status_transition_history": "status",
    "resolution_transition_history": "resolution",
}
FAMILIES = {family: (field, COUNTS[family]) for family, field in FIELDS.items()}
EXTRA = {family: COUNTS[family] for family in families.NEW}
VALUE_BALANCED = {"status_at_snapshot", "resolution_at_snapshot", "closed_at_snapshot"}
SCOPES = {
    "full": {
        "out": BENCH,
        "stream": "full",
        "first": "2023-01",
        "last": "2026-06",
        "coding": 50,
        # Hidden tests of an "API-dependent" task (admission base_failure == test_compile) call
        # symbols only the golden patch defines: passable only by an agent that lands on the
        # project's own names. Kept as a minority on purpose (decided 2026-09-20).
        "coding_api_dependent_max": 15,
        "families": FAMILIES,
        "extra": EXTRA,
    },
    # Scoped benchmark (Yasha, 2026-09-26): stores ingest Jan-May 2026 only, 50 tasks whose
    # whole evidence lies inside that window. Coding = every admitted task checkpointed in the
    # window (Jan included: the store then holds that month so far); history from Feb (Jan is
    # the stores' warm-up), 2 per new family and 1 per field-replay family (11*2 + 8*1 = 30,
    # plus 2 more for the two families closest to coding work).
    "w2026": {
        "out": ROOT / "benchmarks/kafka-50-w2026",
        "stream": "w2026",
        "first": "2026-02",
        "last": "2026-05",
        "coding_first": "2026-01",
        "tasks_from": "2026-01-01T00:00:00Z",
        "window": True,
        # The answerability audit (full-history replicas) found these issues' gold truncated by
        # the window: pre-2026 PRs/resolutions exist. Reserved, so their families pick others.
        "exclude_issues": {"KAFKA-19752", "KAFKA-19620"},
        "extra_files": True,  # <out>/extra/*.jsonl: design_decision (29) + implicit_link (14)
        # Every admitted task checkpointed Jan-May 2026 that passes the rules, including the
        # dependency-chain PRs admitted on 2026-09-26 (Yasha: "make sure all are in benchmark"),
        # and several PRs of one ticket (a multi-PR series is exactly such a chain).
        "coding": 100,
        "coding_api_dependent_max": 100,
        "coding_series": True,
        "families": {family: (field, 1) for family, (field, _) in FAMILIES.items()},
        # open_blockers_unassigned has no candidate inside the window: its 2 and the 3 the
        # admitted coding pool falls short (15 of 18) go to families with ample candidates.
        "extra": {family: 0 if family == "open_blockers_unassigned" else 3 if family in (
                      "issue_changed_files", "pr_approvers", "issue_prs", "review_hotspot",
                      "kip_thread", "fix_version_at_snapshot", "resolution_vs_merge_order",
                  ) else 2 for family in families.NEW},  # fmt: skip
    },
    "smoke": {
        "out": ROOT / "benchmarks/kafka-500-v2-smoke",
        "stream": "smoke",
        "first": "2023-02",
        "last": "2023-02",
        "coding": 2,
        "families": {family: (field, 1) for family, (field, _) in FAMILIES.items()},
        "extra": {family: len(families.VARIANTS.get(family, (family,))) for family in families.NEW},
    },
}
TRAP_SHARE = 0.6  # contract minimum for state families
TRAP_TARGET = 0.7  # what selection aims for, so the minimum survives scarce pools
RICH = 4  # visible changes that make a history task "long"
HIDDEN_TESTS = (  # name kept for the manifest key; the policy itself changed on 2026-09-20
    "Visible tests: the golden PR's test changes are applied to the agent's workspace and named "
    "in the prompt (a test-driven setting), so an agent can see which behaviour and which API "
    "the tests expect and can run them. Grading re-applies the pristine test patch to a fresh "
    "base plus the agent's non-test changes, so edits to those test files have no effect."
)


def answer_schema(family):
    """v1 value types; the evidence list holds event ids or store file paths."""
    if family == "pr_implementation":
        return None
    if family in families.GENERATORS:
        return families.answer_schema(family)
    if family.endswith("history"):
        item = {
            "type": "object",
            "properties": {
                "at": {"type": "string"},
                "from": {"type": ["string", "null"]},
                "to": {"type": ["string", "null"]},
            },
            "required": ["at", "from", "to"],
            "additionalProperties": False,
        }
        value = {"type": "array", "items": item}
    elif family == "closed_at_snapshot":
        value = {"type": "boolean"}
    else:
        value = {"type": ["string", "null"]}
    return {
        "type": "object",
        "properties": {"value": value, "evidence": {"type": "array", "items": {"type": "string"}}},
        "required": ["value", "evidence"],
        "additionalProperties": False,
    }


def field_log(issue, field):
    """Complete, internally consistent field history or None (never silently repaired)."""
    try:
        rows = changes(issue, field)
    except (ValueError, KeyError):
        return None
    if any(b["from"] != a["to"] for a, b in zip(rows, rows[1:], strict=False)):
        return None
    return [{**r, "when": timestamp(r["at"])} for r in rows]


def question(family, field, ref):
    if family == "assignee_at_snapshot":
        return f"Who was assigned to {ref} at the snapshot? Return the exact assignee display name in the latest recorded assignment, or null for unassigned."
    if family == "status_at_snapshot":
        return f"What was the recorded workflow status of {ref} at the snapshot? Return the exact status label."
    if family == "resolution_at_snapshot":
        return f"What was the recorded resolution of {ref} at the snapshot? Return the exact resolution label, or null if unresolved."
    if family == "closed_at_snapshot":
        return f"Was {ref} in the exact workflow status Closed at the snapshot? Return true only for Closed; Resolved is a distinct status and must return false."
    if family == "priority_at_snapshot":
        return f"What was the recorded priority of {ref} at the snapshot? Return the exact priority label."
    return (
        f"List every recorded {field} change for {ref} through the snapshot in chronological "
        "order. Return an array of objects with at, from, and to. Give `at` as the recorded UTC "
        "timestamp with microseconds (e.g. 2023-03-14T20:05:37.287000Z). Preserve null values "
        "and repeated states; break equal-time ties by numeric changelog history ID and item index."
    )


def prompt(cutoff, text):
    return (
        f"Context snapshot cutoff (inclusive): {cutoff}.\n\n{text}\n\n"
        'Return JSON {"value": <answer>, "evidence": [<ids or file paths of the sources you '
        "relied on>]}. Answer as of the cutoff; do not substitute a later or today's issue state."
    )


def candidates(issue, family, field, ckpts, start, end=None, whole=False):
    """One candidate per (answer value, trap, long) for this issue, at a hashed checkpoint.

    The deciding (latest visible) change must lie at or after `start`, the first instant
    tasks are drawn from. Smoke streams are slices: `whole` demands every visible change
    inside them and `end` is where the slice stops before the boundary.
    """
    rows = field_log(issue, field)
    if not rows:
        return []
    key, options = issue["key"], defaultdict(list)
    for ckpt in ckpts:
        limit = timestamp(stream.boundary(ckpt))
        visible = [r for r in rows if r["when"] < limit]
        if not visible or (end and any(end <= r["when"] < limit for r in rows)):
            continue
        history = family.endswith("history")
        first = visible[0] if whole and history else visible[-1]
        if first["when"] < start or (history and len(visible) < 2):
            continue
        if history:
            value = [{k: r[k] for k in ("at", "from", "to")} for r in visible]
            group = ("", len(visible) >= RICH)
        else:
            value = (
                visible[-1]["to"] == "Closed"
                if family == "closed_at_snapshot"
                else visible[-1]["to"]
            )
            group = (canonical(value), False)
        later = [r for r in rows if r["when"] >= limit]
        options[(*group, bool(later))].append((ckpt, visible, value, later))
    result = []
    for (_, rich, trap), found in sorted(options.items()):
        ckpt, visible, value, later = ordered(found, SEED + family + key, lambda o: o[0])[0]
        required = visible if family.endswith("history") else visible[-1:]
        result.append(
            {
                "id": f"H-{family}-{sha(SEED + key)[:10]}",
                "kind": "historical",
                "family": family,
                "checkpoint": ckpt,
                "cutoff": iso(timestamp(stream.boundary(ckpt)) - timedelta(seconds=1)),
                "issue": key,
                "field": field,
                "expected": {"value": value},
                "evidence_ids": [
                    f"jira:{issue['id']}:change:{r['history_id']}:{r['item_index']}"
                    for r in required
                ],
                "visible_change_count": len(visible),
                "long_history": rich,
                "temporal_trap": trap,
                "later_value": later[0]["to"] if later else None,
                "grading": "typed_exact_json; ordered history arrays",
            }
        )
    return result


def pick(pools, count, used, cap=None):
    """First `count` candidates over quota'd pools; one task per issue, optional value cap."""
    chosen, balance = [], Counter()
    for pool, quota in pools:
        for task in pool:
            if len(chosen) >= min(quota, count):
                break
            value = canonical(task["expected"]["value"])
            if task["issue"] in used or (cap and balance[value] >= cap):
                continue
            chosen.append(task)
            used.add(task["issue"])
            balance[value] += 1
    return chosen


def select(family, found, count, used):
    def strata(task):
        year = task["cutoff"][:4]
        return (canonical(task["expected"]["value"]), year) if family in VALUE_BALANCED else (year,)

    def pool(rows):
        return stratified(rows, SEED + family, lambda t: t["id"] + t["checkpoint"], strata)

    if family.endswith("history"):
        first = pool([t for t in found if t["long_history"]])
        rest = pool([t for t in found if not t["long_history"]])
        return pick([(first, count), (rest, count)], count, used)
    traps = pool([t for t in found if t["temporal_trap"]])
    plain = pool([t for t in found if not t["temporal_trap"]])
    cap = math.ceil(count / 2) if family == "closed_at_snapshot" else None
    quota = math.ceil(TRAP_TARGET * count)
    return pick([(traps, quota), (plain, count), (traps, count)], count, used, cap)


class Titles:
    """Issue summaries and PR titles as of a boundary, and whether they name one thing only."""

    def __init__(self, issues, facts=None):
        self.issues, self.facts, self.jira, self.seen = issues, facts, stream.Jira(issues), {}

    def summary(self, key, ckpt):
        when = timestamp(stream.boundary(ckpt)) - timedelta(microseconds=1)
        return " ".join((self.jira.value(key, "summary", when) or "").split())

    def unique(self, key, ckpt):
        limit = timestamp(stream.boundary(ckpt))
        if ("issue", ckpt) not in self.seen:
            self.seen[("issue", ckpt)] = Counter(
                self.summary(k, ckpt).casefold()
                for k, i in self.issues.items()
                if timestamp(i["fields"]["created"]) < limit
            )
        text = self.summary(key, ckpt)
        return len(text) >= 15 and "KAFKA-" not in text and self.seen[("issue", ckpt)][text.casefold()] == 1  # fmt: skip

    def pr_title(self, number):
        return " ".join((self.facts["prs"][str(number)]["title"] or "").split())

    def pr_unique(self, number, ckpt):
        """PR titles are harvest-time text (no edit history): unique among PRs opened by then."""
        limit = stream.boundary(ckpt)
        if ("pr", ckpt) not in self.seen:
            self.seen[("pr", ckpt)] = Counter(
                " ".join((pr["title"] or "").split()).casefold()
                for pr in self.facts["prs"].values()
                if families.state(pr, limit)
            )
        text = self.pr_title(number)
        return len(text) >= 15 and self.seen[("pr", ckpt)][text.casefold()] == 1

    def issue_ref(self, task):
        if not task["keyless"]:
            return task["issue"]
        created = self.issues[task["issue"]]["fields"]["created"][:7]
        return f'the Apache Kafka JIRA issue titled "{task["summary_at_cutoff"]}" (created in {created})'

    def pr_ref(self, task):
        if not task["keyless"]:
            return f"pull request #{task['pr']}"
        opened = self.facts["prs"][str(task["pr"])]["events"][0][0][:7]
        return (
            f'the apache/kafka pull request titled "{task["title_at_cutoff"]}" (opened in {opened})'
        )


def word(chosen, titles, index=0):
    """Prompt, keyless flag and schema for the new families: half of each issue- or PR-centred
    family (alternating single-task smoke families) hides the key behind the title."""
    groups = defaultdict(list)
    for task in chosen:
        groups[task["family"]].append(task)
    for offset, (family, mine) in enumerate(sorted(groups.items())):
        centred = family in families.ISSUE_CENTRED or family in families.PR_CENTRED
        keyless = (len(mine) // 2 if len(mine) > 1 else (index + offset) % 2) if centred else 0
        for task in ordered(mine, SEED + "keyless", lambda t: t["id"]):
            task["keyless"] = False
            if family in families.ISSUE_CENTRED:
                task["summary_at_cutoff"] = titles.summary(task["issue"], task["checkpoint"])
                task["keyless"] = keyless > 0 and titles.unique(task["issue"], task["checkpoint"])
                ref = titles.issue_ref(task)
            elif family in families.PR_CENTRED:
                task["title_at_cutoff"] = titles.pr_title(task["pr"])
                task["keyless"] = keyless > 0 and titles.pr_unique(task["pr"], task["checkpoint"])
                ref = titles.pr_ref(task)
            else:
                ref = None
            keyless -= task["keyless"]
            task["prompt"] = prompt(task["cutoff"], families.question(task, ref))
            task["answer_schema"] = answer_schema(family)
    return chosen


def historical(issues, config, excluded, present=None, end=None, facts=None, since=None):
    """Select and word the historical tasks; `present` restricts issues (smoke). All 19
    families (v1 field replays + families.py) pick in scarcity order: the pool with the fewest
    distinct entities goes first, so one-entity-per-task never starves a rare family."""
    ckpts = stream.checkpoints(config["first"], config["last"])
    start = timestamp(config.get("tasks_from", TASKS_FROM))
    titles = Titles(issues, facts)
    used, tasks, short, cache = set(excluded), [], {}, {}
    extra = config.get("extra") or {}
    pools = {
        family: (count, [
            task
            for key in sorted(issues)
            if key not in used and (present is None or key in present)
            for task in candidates(issues[key], family, field, ckpts, start, end, end is not None)
        ])
        for family, (field, count) in config["families"].items()
    }  # fmt: skip
    if extra and facts is not None:
        ctx = families.Context(issues, facts, ckpts, start, since=since, end=end, present=present, seed=SEED)  # fmt: skip
        pools.update(families.candidates(ctx, extra))
    elif extra:
        short.update({f: {"missing": n, "reason": "stream not built"} for f, n in extra.items()})

    def entities(family):
        return len({frozenset(families.reserved(t)) for t in pools[family][1]})

    plan = sorted(pools, key=lambda f: (entities(f), f))
    for index, family in enumerate(plan):
        count, found = pools[family]
        if family in families.GENERATORS:
            chosen, left = families.choose(family, found, count, used, SEED, cache=cache)
            if len(chosen) < count:
                short[family] = {"missing": count - len(chosen), "candidates": left}
            tasks.extend(word(chosen, titles, index))
            continue
        field = FIELDS[family]
        found = [t for t in found if t["issue"] not in used]
        chosen = select(family, found, count, used)
        if len(chosen) < count:
            short[family] = count - len(chosen)
        # Half of each family (alternating single-task smoke families) hides the key.
        keyless = count // 2 if count > 1 else index % 2
        for task in ordered(chosen, SEED + "keyless", lambda t: t["id"]):
            task["keyless"] = keyless > 0 and titles.unique(task["issue"], task["checkpoint"])
            keyless -= task["keyless"]
            task["summary_at_cutoff"] = titles.summary(task["issue"], task["checkpoint"])
            task["prompt"] = prompt(task["cutoff"], question(family, field, titles.issue_ref(task)))
            task["answer_schema"] = answer_schema(family)
        tasks.extend(chosen)
    return sorted(tasks, key=lambda t: t["id"]), short


def admitted(bench=BENCH, results=ADMISSION):
    rows = [json.loads(line) for line in (bench / "coding-candidates.jsonl").open()]
    found = []
    for row in rows:
        path = results / f"{row['pr']}.json"
        result = read(path) if path.exists() else {}
        if result.get("status") == "admitted":
            found.append(
                {
                    **row,
                    **{k: result[k] for k in ("fail_to_pass", "pass_to_pass")},
                    "base_failure": result.get("base_failure"),
                }
            )
    return rows, found


def coding_task(row, checkpoint):
    text = (
        f"Context snapshot cutoff (inclusive): {row['context_cutoff']}.\n\n"
        f"The workspace is the apache/kafka repository at commit {row['base_sha']}.\n\n"
        f"Issue {row['issue']}: {row['title']}\n\n{(row.get('description') or '').strip()}\n\n"
        "Implement the change this issue asks for by editing the workspace. The tests written for "
        "this change have already been added to the workspace (uncommitted; see `git status`): "
        + ", ".join(f"`{t.get('path', t['class'])}`" for t in row["tests"])
        + ". They fail or do not compile until the change is implemented. Make them pass with "
        + " ; ".join(
            sorted(
                {
                    f"`./gradlew {t['module']}:test --tests {t['class']} --offline`"
                    for t in row["tests"]
                }
            )
        )
        + ". Do not edit those test files: grading restores them to their original content and "
        "runs them against your other changes. Keep the change focused and do not substitute "
        "knowledge of how the project evolved after the cutoff."
    )
    return {
        "id": row["id"],
        "kind": "coding",
        "family": "pr_implementation",
        "checkpoint": checkpoint,
        "cutoff": row["context_cutoff"],
        "base_sha": row["base_sha"],
        "tests": row["tests"],
        "prompt": text,
        "answer_schema": None,
        "issue": row["issue"],
        "pr": row["pr"],
        "context_cutoff": row["context_cutoff"],
        "reference_commit": row["reference_commit"],
        "reference_patch": row["reference_patch"],
        "test_patch": row["test_patch"],
        "fail_to_pass": row["fail_to_pass"],
        "pass_to_pass": row["pass_to_pass"],
        "issue_at_checkpoint": row.get(
            "issue_at_checkpoint"
        ),  # False: the issue is younger than the checkpoint
        "evidence_ids": [],
    }


# An issue whose SUBJECT is the test code itself (mock-framework migrations, test kits, flaky
# tests) is not a coding task for this benchmark: the "fix" is a rewrite of the very tests that
# grade it, and any agent edit to a test file is dropped or breaks the hidden tests' compile.
TEST_SUBJECT = re.compile(r"easymock|mockito|powermock|junit|testkit|test kit|flaky", re.IGNORECASE)


def coding(rows, config, used, force=None, frozen=()):
    """Up to `config["coding"]` admitted rows, rotated over PR-creation quarters; rows whose
    issue already existed at their checkpoint come first, the others are flagged.

    `frozen` = ids already in the append-only coding order (a run has started on them): they
    are kept, in that order, whatever was admitted since; new admissions only fill what is
    left. The result is in run order (frozen first), not sorted by id."""

    def quarter(row):
        return f"{row['pr_created_at'][:4]}Q{(int(row['pr_created_at'][5:7]) + 2) // 3}"

    def ranked(candidates):
        return stratified(candidates, SEED + "coding", lambda r: r["id"], quarter)

    tasks, issues, api_count = [], set(used), 0
    rows = [r for r in rows if not TEST_SUBJECT.search(r["title"])]
    by_id = {r["id"]: r for r in rows}
    if gone := [i for i in frozen if i not in by_id]:
        raise ValueError(f"frozen coding tasks are no longer admitted: {gone}")
    kept = [by_id[i] for i in frozen]
    rows = [r for r in rows if r["id"] not in set(frozen)]
    first = [r for r in rows if r.get("issue_at_checkpoint", True)]
    rest = [r for r in rows if not r.get("issue_at_checkpoint", True)]
    for row in kept + ranked(first) + ranked(rest):
        checkpoint = (
            force
            or row.get("checkpoint")
            or stream.checkpoint_for(row["context_cutoff"], config["first"], config["last"])
        )
        lo = config.get("coding_first", config["first"])
        if checkpoint is not None and not lo <= checkpoint <= config["last"]:
            checkpoint = None  # outside the scope's checkpoints: no store snapshot exists
        if checkpoint is None or row["issue"] in issues or len(tasks) >= config["coding"]:
            continue
        api = row.get("base_failure") == "test_compile"
        if (
            api
            and row["id"] not in frozen
            and api_count >= config.get("coding_api_dependent_max", config["coding"])
        ):
            continue  # the API-dependent minority is full; behaviour-only tasks fill the rest
        api_count += api
        if not config.get("coding_series"):
            issues.add(row["issue"])
        tasks.append(coding_task(row, checkpoint))
    return tasks if frozen else sorted(tasks, key=lambda t: t["id"])


def rounds(values, stratum):
    """Round-robin over strata, keeping the given order inside each."""
    groups = defaultdict(list)
    for value in values:
        groups[stratum(value)].append(value)
    queues = [groups[k] for k in sorted(groups)]
    return [q[i] for i in range(max(map(len, queues), default=0)) for q in queues if i < len(q)]


def order(tasks, coding_lane=None) -> list[str]:
    """Frozen run order: coding/historical 1:1, families and years rotating in every prefix.
    `coding_lane` (the append-only coding order) replaces the computed coding lane, so tasks
    admitted later never move the ones a run has already started on."""

    def year(task):
        return task["checkpoint"][:4]

    lanes = []
    for kind in ("coding", "historical"):
        mine = ordered([t for t in tasks if t["kind"] == kind], SEED + "order", lambda t: t["id"])
        families = defaultdict(list)
        for task in rounds(mine, year):
            families[task["family"]].append(task)
        lanes.append(
            rounds([t for f in sorted(families) for t in families[f]], lambda t: t["family"])
        )
    if coding_lane:
        by_id = {t["id"]: t for t in tasks}
        lanes[0] = [by_id[i] for i in coding_lane]
    return [t["id"] for t in rounds([*lanes[0], *lanes[1]], lambda t: t["kind"])] if tasks else []


PUBLIC = ("id", "kind", "family", "checkpoint", "cutoff", "prompt", "answer_schema")


def public(task):
    extra = ("base_sha", "tests") if task["kind"] == "coding" else ("keyless",)
    return {k: task[k] for k in (*PUBLIC, *extra)}


def dump(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical(row) + "\n" for row in rows))
    return sha(path.read_bytes())


def build(
    scope="full", issues=None, out: Path | None = None, bench=BENCH, results=ADMISSION, events=None
) -> Path:
    config = SCOPES[scope]
    out = out or config["out"]
    issues = inventory()[0] if issues is None else issues
    rows, passed = admitted(bench, results)
    # Full scope reserves every candidate's issue, so admission progress never reshuffles
    # the historical half; smoke only keeps clear of the coding tasks it emits.
    smoke_code = coding(passed, config, set(), force=config["last"]) if scope == "smoke" else []
    reserved = {t["issue"] for t in smoke_code} if scope == "smoke" else {r["issue"] for r in rows}
    reserved |= config.get("exclude_issues", set())
    present = end = since = None
    path = events or stream.OUT / stream.SCOPES[config["stream"]]["dir"] / "events.jsonl"
    ids = {e["id"] for e in stream.read_events(path)} if path.exists() else None
    facts = families.facts(path) if ids is not None else None
    if scope == "smoke":
        if ids is None:
            raise SystemExit(f"build the smoke stream first: {path}")
        focus = read(path.parent / "stream-manifest.json")["focus"]["entities"]
        present = {e.removeprefix("issue:") for e in focus if e.startswith("issue:")}
        since, end = map(timestamp, stream.SCOPES["smoke"]["window"])
    elif config.get("window"):
        if ids is None:
            raise SystemExit(f"build the {config['stream']} stream first: {path}")
        since, end = map(timestamp, stream.SCOPES[config["stream"]]["window"])
    tasks, short = historical(issues, config, reserved, present, end, facts, since)
    if ids is not None and (scope == "smoke" or config.get("window")):
        tasks = [t for t in tasks if set(t["evidence_ids"]) <= ids]
    used = {t["issue"] for t in tasks}
    order_file = out / "coding-order.json"
    frozen = read(order_file)["order"] if scope == "full" and order_file.exists() else []
    tasks += smoke_code if scope == "smoke" else coding(passed, config, used, frozen=frozen)
    # Hand-built families (2026-09-27): design decisions from KIP discussions and implicit
    # mail -> JIRA links, kept as task records under <out>/extra/*.jsonl. They are appended
    # AFTER the generated tasks and at the END of the run order, so every generated task keeps
    # its record and its place (runs pin tasks, and max_coding/max_historical take prefixes).
    extra = [json.loads(line) for path in sorted((out / "extra").glob("*.jsonl")) for line in path.open()] if config.get("extra_files") else []  # fmt: skip
    for task in extra:  # bookkeeping fields every generated history task carries
        for key, default in (("temporal_trap", False), ("preferred", False), ("long_history", False), ("keyless", True)):
            task.setdefault(key, default)
    base_order_tasks = list(tasks)
    tasks += extra
    history = [t for t in tasks if t["kind"] == "historical"]
    code = [t for t in tasks if t["kind"] == "coding"]
    # Append-only coding lane: the first build freezes the computed (year-rotating) lane;
    # later builds keep it and append whatever was admitted since.
    kinds = {t["id"]: t["kind"] for t in tasks}
    lane = (
        frozen + [t["id"] for t in code if t["id"] not in set(frozen)]
        if frozen
        else [i for i in order(tasks) if kinds[i] == "coding"]
    )
    state = [t for t in history if t["family"] in FIELDS and not t["family"].endswith("history")]
    by_family = defaultdict(list)
    for task in history:
        by_family[task["family"]].append(task)
    manifest = {
        "benchmark": "kafka-500-v2" + ("-smoke" if scope == "smoke" else ""),
        "scope": scope,
        "seed": SEED,
        "task_count": len(tasks),
        "historical": len(history),
        "coding": len(code),
        "coding_target": config["coding"],
        "coding_candidates": len(rows),
        "coding_admitted": len(passed),
        "coding_issue_not_at_checkpoint": sorted(
            t["id"] for t in code if t["issue_at_checkpoint"] is False
        ),
        "families": dict(sorted(Counter(t["family"] for t in tasks).items())),
        "historical_shortfall": short,
        "checkpoints": dict(sorted(Counter(t["checkpoint"] for t in tasks).items())),
        "keyless": dict(sorted(Counter(t["family"] for t in history if t["keyless"]).items())),
        "temporal_trap_share_state_families": round(
            sum(t["temporal_trap"] for t in state) / max(len(state), 1), 3
        ),
        "temporal_trap_minimum": TRAP_SHARE,
        "temporal_trap_share_by_family": {
            f: round(sum(t["temporal_trap"] for t in mine) / len(mine), 3)
            for f, mine in sorted(by_family.items())
        },
        "preferred_share_by_family": {
            f: round(sum(t.get("preferred", False) for t in mine) / len(mine), 3)
            for f, mine in sorted(by_family.items())
            if f in families.GENERATORS
        },
        "long_history_tasks": sum(t["long_history"] for t in history),
        "evidence_in_stream": (
            "stream not built"
            if ids is None
            else {"missing": sorted(i for t in history for i in t["evidence_ids"] if i not in ids)}
        ),
        "test_patch_policy": HIDDEN_TESTS,
        "historical_rules": [
            "cutoff = checkpoint boundary - 1 s; gold = changelog items strictly before the boundary",
            "the asked field's latest visible change lies in the task window (from TASKS_FROM of the scope); "
            "earlier changes of a history answer are in the complete stream",
            "no issue appears twice; issues of every coding candidate are reserved for coding",
            "keyless tasks name the issue by its summary as of the cutoff (unique among issues "
            "existing then) and its creation month; the key stays private",
            "the families of families.py take GitHub/mail gold from the stream itself and JIRA gold "
            "from the raw changelog; strings are graded casefolded and whitespace-normalised, arrays "
            "sorted; keyless PR tasks name the PR by its (unique) title and opening month",
        ],
        "run_order": order(base_order_tasks, coding_lane=lane if scope == "full" else None) + [t["id"] for t in extra],
        "coding_order_policy": "append-only (coding-order.json): tasks admitted later are added at the end",
    }
    if len(code) < config["coding"]:
        manifest["coding_shortfall"] = {
            "missing": config["coding"] - len(code),
            "reason": "admission still running or admitted pool exhausted; never padded",
        }
    if scope == "smoke":
        manifest["relaxations"] = [
            "coding tasks are mapped to checkpoint 2023-02 regardless of their context cutoff",
            "historical gold ignores nothing, but issues with a change of the asked field between "
            "the smoke stream end (2023-02-15) and the 2023-02 boundary are excluded",
            "one task per family; keyless alternates by family",
            "only the emitted coding tasks' issues (not every candidate's) are kept out of "
            "the historical half",
        ]
    manifest["public_sha256"] = dump(out / "tasks.public.jsonl", map(public, tasks))
    manifest["private_sha256"] = dump(out / "tasks.private.jsonl", tasks)
    save(out / "manifest.json", manifest)
    if scope == "full":
        kept = read(order_file).get("excluded", {}) if order_file.exists() else {}
        failure = {r["id"]: r.get("base_failure") for r in passed}
        kind = {i: "api" if failure.get(i) == "test_compile" else "behaviour" for i in lane}
        save(order_file, {"order": lane, "kind": kind, "excluded": kept})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("build")
    command.add_argument("--scope", choices=sorted(SCOPES), default="full")
    args = parser.parse_args()
    out = build(args.scope)
    manifest = read(out / "manifest.json")
    print(canonical({k: v for k, v in manifest.items() if k != "run_order"}))
    print(out)


if __name__ == "__main__":
    main()
