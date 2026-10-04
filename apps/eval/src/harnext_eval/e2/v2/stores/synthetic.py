"""Synthetic stream in the contract's event schema, for tests and offline/live smoke
checks until `v2/stream.py` produces the real one. Deterministic; no real Kafka content."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

PEOPLE = ["Ana Ruiz", "Bo Chen", "Chidi Okafor", "Dana Levi", "Emil Novak"]
AREAS = [
    ("consumer rebalance", "ConsumerCoordinator", "clients"),
    ("log compaction", "LogCleaner", "core"),
    ("KRaft snapshot loading", "SnapshotReader", "metadata"),
    ("Streams state restore", "StoreChangelogReader", "streams"),
    ("Connect offset commit", "WorkerSourceTask", "connect"),
]
STEPS = ["jira.created", "jira.comment", "jira.change", "gh.pr_opened", "gh.review",
         "jira.comment", "gh.pr_merged", "jira.change", "mail.message"]  # fmt: skip


def event(n: int, when: datetime) -> dict:
    case, step = divmod(n, len(STEPS))
    kind = STEPS[step]
    issue, pr = f"KAFKA-{9000 + case}", 14000 + case
    topic, klass, module = AREAS[case % len(AREAS)]
    actor, other = PEOPLE[(case + step) % len(PEOPLE)], PEOPLE[(case + step + 2) % len(PEOPLE)]
    entities, meta = [f"issue:{issue}"], {}
    title = f"{issue}: {klass} fails during {topic} (case {case})"
    if kind == "jira.created":
        meta = {
            "status": "Open",
            "priority": ["Major", "Minor", "Blocker"][case % 3],
            "assignee": None,
        }
        text = (f"{actor} reported {issue}. While running {topic} on a 3-broker cluster, `{klass}` in the "
                f"`{module}` module throws IllegalStateException after {30 + case} seconds when "
                f"`max.poll.interval.ms={1000 * (case + 5)}`. Affects version 3.{case % 8}.0. "
                f"Expected: the operation retries; actual: the thread dies and the partition stalls.")  # fmt: skip
    elif kind == "jira.comment":
        title = f"{issue}: comment by {actor}"
        text = (f"{actor} commented on {issue}: I can reproduce this with {case % 4 + 2} partitions. "
                f"The root cause looks like a missing null check in `{klass}.maybeRetry()`; {other} "
                f"suggested guarding it with the epoch fence from KIP-{500 + case % 40}. "
                + ("A unit test in `" + klass + "Test` would cover it." if step == 1 else
                   "After the patch the stall no longer reproduces in a 2-hour soak run."))  # fmt: skip
    elif kind == "jira.change":
        final = step == 7
        meta = {"changes": [{"field": "status", "from": "Open", "to": "Resolved" if final else "In Progress"},
                            *([{"field": "resolution", "from": None, "to": "Fixed"}] if final else
                              [{"field": "assignee", "from": None, "to": actor}])]}  # fmt: skip
        title = f"{issue}: status → {'Resolved' if final else 'In Progress'}"
        text = (f"{actor} changed {issue}: status Open → {'Resolved, resolution Fixed, fix version 3.' + str(case % 8 + 1) + '.0' if final else 'In Progress; assignee → ' + actor}.")  # fmt: skip
    elif kind.startswith("gh."):
        entities = [f"pr:{pr}", f"issue:{issue}"]
        title = f"PR #{pr}: {issue} guard {klass}.maybeRetry against stale epoch"
        text = {
            "gh.pr_opened": f"{actor} opened PR #{pr} for {issue}. Adds an epoch check to `{klass}.maybeRetry()` in `{module}/src/main/java/org/apache/kafka/{module}/{klass}.java` and a regression test `{klass}Test.testRetryAfterStaleEpoch`.",
            "gh.review": f"{actor} reviewed PR #{pr}: requested changes — the new check should log at DEBUG not WARN, and the test must use MockTime instead of Thread.sleep. {other} agreed.",
            "gh.pr_merged": f"{actor} merged PR #{pr} ({issue}) into trunk; cherry-picked to the 3.{case % 8} branch.",
        }[kind]  # fmt: skip
    else:
        entities = [f"thread:dev-{case}", f"issue:{issue}"]
        title = f"[DISCUSS] backporting {issue} ({topic})"
        text = (f"From: {actor}\nTo: dev@kafka.apache.org\n\nShould the fix for {issue} be backported to "
                f"3.{case % 8}.x? {other} is +1 because {topic} regressions block the release; "
                f"the release manager wants it merged before the code freeze.")  # fmt: skip
    return {"id": f"syn-{n:05d}-{kind.replace('.', '-')}", "time": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "type": kind, "entities": entities, "actor": actor, "title": title, "text": text, "meta": meta}  # fmt: skip


def events(count: int, *, start: str = "2023-01-02T09:00:00Z", hours: float = 7.0) -> list[dict]:
    first = datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    return [event(n, first + timedelta(hours=hours * n)) for n in range(count)]


def dump(path: Path, count: int, **kwargs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in events(count, **kwargs)))
    return path
