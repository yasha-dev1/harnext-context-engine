"""Deterministic multi-source corpus for docs/evaluation-spec.md §3.2."""

from __future__ import annotations

import hashlib
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from harnext_eval.types import EvalEvent

if TYPE_CHECKING:
    from harnext_eval.corpus import CorpusHandle

_START = datetime(2026, 1, 1, tzinfo=UTC)
_STATUSES = ("Open", "In Progress", "In Review", "Resolved", "Reopened")
_PRIORITIES = ("Minor", "Major", "Critical")
_COMPONENTS = ("api", "builder", "classifier", "mcp", "web")


def _event(
    *,
    index: int,
    seed: int,
    source: str,
    event_type: str,
    issue_key: str,
    event_time: datetime,
    baseline_keys: list[str],
    data: dict[str, Any],
) -> EvalEvent:
    event_id = hashlib.sha256(f"synthetic:{seed}:{index}".encode()).hexdigest()[:24]
    return EvalEvent(
        id=event_id,
        source=source,
        type=event_type,
        subject=f"issue:{issue_key}",
        time=event_time,
        mgtenant="synthetic",
        baseline_keys=baseline_keys,
        data=data,
    )


def generate_synthetic_events(
    seed: int = 1,
    *,
    event_count: int = 2_000,
    days: int = 60,
    entity_count: int = 40,
) -> list[EvalEvent]:
    """Generate an ordered stream with Jira, mail, and GitHub gold signals."""

    if event_count <= 0 or days <= 0 or entity_count <= 0:
        raise ValueError("event_count, days, and entity_count must be positive")
    rng = random.Random(seed)
    issues = [f"HNX-{1000 + index}" for index in range(entity_count)]
    state = {
        issue: {
            "status": "Open",
            "assignee": f"user-{index % 12:02d}",
            "priority": "Major",
            "components": [_COMPONENTS[index % len(_COMPONENTS)]],
            "fixVersion": None,
        }
        for index, issue in enumerate(issues)
    }
    last_mail: dict[str, str] = {}
    mail_thread: dict[str, str] = {}
    events: list[EvalEvent] = []
    total_seconds = days * 24 * 60 * 60
    step_seconds = total_seconds / event_count
    pr_number = 500

    for index in range(event_count):
        issue_index = (index * 17 + rng.randrange(entity_count)) % entity_count
        issue = issues[issue_index]
        component = state[issue]["components"][0]
        contributor = f"contributor:{(issue_index * 7 + index) % 23:02d}"
        jitter = rng.random() * step_seconds * 0.35
        event_time = _START + timedelta(seconds=index * step_seconds + jitter)
        kind = index % 8

        if kind in {0, 4}:
            field = "status" if index % 16 else "priority"
            old_value = state[issue][field]
            if field == "status":
                new_value = _STATUSES[(_STATUSES.index(str(old_value)) + 1) % len(_STATUSES)]
            else:
                new_value = _PRIORITIES[(_PRIORITIES.index(str(old_value)) + 1) % len(_PRIORITIES)]
            state[issue][field] = new_value
            data = {
                "issue_key": issue,
                "field": field,
                "from": old_value,
                "to": new_value,
                "actor": contributor,
                "changelog": {
                    "items": [{"field": field, "from": old_value, "to": new_value}]
                },
                "state": dict(state[issue]),
            }
            event = _event(
                index=index,
                seed=seed,
                source="jira:harnext",
                event_type="org.harnext.jira.issue.transition",
                issue_key=issue,
                event_time=event_time,
                baseline_keys=[contributor, f"component:{component}"],
                data=data,
            )
        elif kind in {1, 6}:
            data = {
                "issue_key": issue,
                "comment_id": f"comment-{index}",
                "author": contributor,
                "body": f"Update for {issue}: current status is {state[issue]['status']}.",
                "state": dict(state[issue]),
            }
            event = _event(
                index=index,
                seed=seed,
                source="jira:harnext",
                event_type="org.harnext.jira.issue.comment",
                issue_key=issue,
                event_time=event_time,
                baseline_keys=[contributor, f"component:{component}"],
                data=data,
            )
        elif kind in {2, 5}:
            if issue not in mail_thread or index % 40 == 2:
                mail_thread[issue] = f"thread-{issue.lower()}-{index}"
                in_reply_to = None
            else:
                in_reply_to = last_mail.get(issue)
            message_id = f"<{seed}.{index}@synthetic.invalid>"
            last_mail[issue] = message_id
            thread = mail_thread[issue]
            data = {
                "issue_key": issue,
                "thread_id": thread,
                "message_id": message_id,
                "in_reply_to": in_reply_to,
                "from": f"user{issue_index % 12}@example.invalid",
                "subject": f"Re: {issue} {state[issue]['status']}",
                "body": f"The {issue} priority is {state[issue]['priority']}.",
            }
            event = _event(
                index=index,
                seed=seed,
                source="mail:dev",
                event_type="org.harnext.mail.message",
                issue_key=issue,
                event_time=event_time,
                baseline_keys=[contributor, f"component:{component}", f"thread:{thread}"],
                data=data,
            )
        else:
            pr_number += 1
            changed_files = [
                f"src/{component}/{issue.lower().replace('-', '_')}.py",
                f"tests/{component}/test_{issue.lower().replace('-', '_')}.py",
            ]
            data = {
                "issue_key": issue,
                "number": pr_number,
                "title": f"{issue}: update {component}",
                "state": "merged" if kind == 7 else "open",
                "author": contributor,
                "changed_files": changed_files,
                "merge_commit_sha": hashlib.sha1(  # noqa: S324 - fixture identifier only
                    f"{seed}:{pr_number}".encode()
                ).hexdigest(),
            }
            event = _event(
                index=index,
                seed=seed,
                source="github:harnext",
                event_type=f"org.harnext.github.pull_request.{data['state']}",
                issue_key=issue,
                event_time=event_time,
                baseline_keys=[contributor, f"component:{component}"],
                data=data,
            )
        events.append(event)
    return events


def events_hash(events: list[EvalEvent]) -> str:
    digest = hashlib.sha256()
    for event in events:
        digest.update(event.model_dump_json().encode())
        digest.update(b"\n")
    return digest.hexdigest()


def generate_synthetic_corpus(
    output: str | Path,
    seed: int = 1,
    *,
    event_count: int = 2_000,
    days: int = 60,
    entity_count: int = 40,
) -> CorpusHandle:
    """Write deterministic JSONL and return its corpus handle."""

    from harnext_eval.corpus import CorpusHandle

    path = Path(output)
    if path.suffix != ".jsonl":
        path = path / "synthetic.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    events = generate_synthetic_events(
        seed, event_count=event_count, days=days, entity_count=entity_count
    )
    path.write_text("".join(event.model_dump_json() + "\n" for event in events), encoding="utf-8")
    return CorpusHandle(
        name="synthetic",
        replay_path=path,
        probes_path=None,
        tasks_path=None,
        window=(events[0].time, events[-1].time),
        meta={
            "seed": seed,
            "event_count": event_count,
            "entity_count": entity_count,
            "days": days,
            "sha256": events_hash(events),
        },
    )
