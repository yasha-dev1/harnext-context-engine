"""Shared replay helpers for probe generators in docs/evaluation-spec.md §5."""

from __future__ import annotations

import random
import re
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

from harnext_eval.types import EvalEvent

ISSUE_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")


def load_replay(path: str | Path) -> list[EvalEvent]:
    """Load a chronologically ordered EvalEvent JSONL replay."""

    events: list[EvalEvent] = []
    with Path(path).open(encoding="utf-8") as replay:
        for line_number, line in enumerate(replay, start=1):
            if not line.strip():
                continue
            try:
                events.append(EvalEvent.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"invalid EvalEvent at {path}:{line_number}: {exc}") from exc
    if not events:
        raise ValueError(f"replay is empty: {path}")
    if any(left.time > right.time for left, right in zip(events, events[1:], strict=False)):
        raise ValueError("replay events must be sorted by event time")
    return events


def parse_time(value: str) -> datetime:
    """Parse an ISO-8601 CLI timestamp and require an explicit timezone."""

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must include a timezone: {value}")
    return parsed


def validate_period(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("probe period timestamps must be timezone-aware")
    if start >= end:
        raise ValueError("probe-start must be earlier than probe-end")
    return start, end


def canonical_entity(event: EvalEvent) -> str:
    data = event.data or {}
    issue_key = data.get("issue_key") or data.get("key")
    if isinstance(issue_key, str) and issue_key:
        return issue_key
    return event.subject.split(":", 1)[1] if event.subject.startswith("issue:") else event.subject


def uniform_time(
    rng: random.Random, start: datetime, end: datetime, *, exclude_end: bool = False
) -> datetime:
    """Sample a timezone-aware instant in [start, end] (or [start, end))."""

    if start > end or (exclude_end and start >= end):
        raise ValueError("cannot sample from an empty time interval")
    span = (end - start).total_seconds()
    if span == 0:
        return start
    fraction = rng.random()
    sampled = start + timedelta(seconds=span * fraction)
    if exclude_end and sampled >= end:
        return end - timedelta(microseconds=1)
    return sampled


def display_time(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def string_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        import json

        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def changed_files(event: EvalEvent) -> list[str]:
    raw = (event.data or {}).get("changed_files", [])
    if not isinstance(raw, list):
        return []
    files: list[str] = []
    for item in raw:
        if isinstance(item, str):
            path = item
        elif isinstance(item, dict):
            path = item.get("filename") or item.get("path") or item.get("name")
        else:
            continue
        if isinstance(path, str) and path.strip():
            files.append(path.strip().lstrip("/"))
    return unique(files)


def is_merged_pr(event: EvalEvent) -> bool:
    data = event.data or {}
    return (
        "pull_request" in event.type.casefold()
        and (
            event.type.casefold().endswith(".merged")
            or str(data.get("state", "")).casefold() == "merged"
            or bool(data.get("merged_at"))
        )
    )


def is_formatting_only(event: EvalEvent) -> bool:
    data = event.data or {}
    if bool(data.get("formatting_only")):
        return True
    labels = data.get("labels", [])
    label_text = " ".join(
        str(label.get("name", "") if isinstance(label, dict) else label) for label in labels
    ) if isinstance(labels, list) else str(labels)
    title = str(data.get("title", ""))
    return bool(re.search(r"\b(format(?:ting)?[- ]only|whitespace[- ]only)\b", f"{title} {label_text}", re.I))


def issue_keys_for_pr(event: EvalEvent) -> list[str]:
    """Return issue keys literally carried in the PR title."""

    data = event.data or {}
    return unique(ISSUE_KEY_RE.findall(str(data.get("title", ""))))


def module_for_file(path: str) -> str:
    parts = PurePosixPath(path).parts
    return "/".join(parts[:2])
