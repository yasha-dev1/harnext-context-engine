"""Deterministic S1 projection for docs/evaluation-spec.md §7 E3."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harnext_eval.stores.base import StoreHandle
from harnext_eval.stores.layouts import safe_component
from harnext_eval.types import EvalEvent

STRUCTURED_FIELDS = ("status", "assignee", "priority", "components", "fixVersion", "linked_keys")
_ISSUE_KEY = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")
_FACT = re.compile(
    r"^- (?P<date>\d{4}-\d{2}-\d{2}) \[(?P<provenance>.+)\] "
    r"(?P<field>status|assignee|priority|components|fixVersion|linked_keys)=(?P<value>.*)$"
)
_LAST_UPDATED = re.compile(r"^_Last updated: (?P<timestamp>.+) \[(?P<provenance>.+)\]_$")


@dataclass(slots=True)
class Fact:
    field: str
    value: str
    date: str
    provenance: str

    def render(self) -> str:
        return f"- {self.date} [{self.provenance}] {self.field}={self.value}"


def entity_relpath(subject: str) -> str:
    """Map ``type:value`` subjects to the seeded entity hierarchy."""

    if ":" in subject:
        entity_type, raw_slug = subject.split(":", 1)
    else:
        entity_type, raw_slug = "entity", subject
    entity_type = safe_component(entity_type, fallback="entity")
    raw_slug = raw_slug.replace("/", "__").replace(":", "_")
    slug = safe_component(raw_slug, fallback="unknown")
    return f"entities/{entity_type}/{slug}"


def _value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value.replace("\n", " ").strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalise_field(field: str) -> str | None:
    key = re.sub(r"[^a-z]", "", field.casefold())
    return {
        "status": "status",
        "assignee": "assignee",
        "priority": "priority",
        "component": "components",
        "components": "components",
        "fixversion": "fixVersion",
        "fixversions": "fixVersion",
        "linkedkey": "linked_keys",
        "linkedkeys": "linked_keys",
        "issuelink": "linked_keys",
        "issuelinks": "linked_keys",
    }.get(key)


def _key_candidates(value: Any) -> set[str]:
    if isinstance(value, str):
        return set(_ISSUE_KEY.findall(value))
    if isinstance(value, dict):
        found: set[str] = set()
        for nested in value.values():
            found.update(_key_candidates(nested))
        return found
    if isinstance(value, list):
        found = set()
        for nested in value:
            found.update(_key_candidates(nested))
        return found
    return set()


def structured_updates(event: EvalEvent) -> dict[str, str]:
    """Extract only the preregistered Customer-360 fields from an event."""

    data = event.data or {}
    updates: dict[str, Any] = {}
    state = data.get("state")
    if isinstance(state, dict):
        for raw_field, value in state.items():
            if field := _normalise_field(str(raw_field)):
                updates[field] = value

    for raw_field, value in data.items():
        if field := _normalise_field(str(raw_field)):
            updates[field] = value

    transition_field = data.get("field")
    if isinstance(transition_field, str) and (field := _normalise_field(transition_field)):
        updates[field] = data.get("to")

    changelog = data.get("changelog")
    if isinstance(changelog, dict):
        items = changelog.get("items")
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                raw_field = item.get("field")
                if isinstance(raw_field, str) and (field := _normalise_field(raw_field)):
                    updates[field] = item.get("to", item.get("toString"))

    if "components" in updates and not isinstance(updates["components"], list):
        updates["components"] = [updates["components"]]

    own_key = event.subject.split(":", 1)[-1]
    linked = _key_candidates(data)
    linked.discard(own_key)
    if linked:
        explicit = updates.get("linked_keys", [])
        linked.update(_key_candidates(explicit))
        updates["linked_keys"] = sorted(linked)

    return {field: _value(value) for field, value in updates.items() if field in STRUCTURED_FIELDS}


def _read_facts(path: Path) -> tuple[dict[str, Fact], list[str]]:
    facts: dict[str, Fact] = {}
    passthrough: list[str] = []
    if not path.exists():
        return facts, passthrough
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _FACT.match(line)
        if not match:
            if line and not line.startswith("#") and not line.startswith("_"):
                passthrough.append(line)
            continue
        facts[match.group("field")] = Fact(**match.groupdict())
    return facts, passthrough


def _write_facts(path: Path, facts: dict[str, Fact], passthrough: list[str]) -> None:
    lines = ["# Current Facts", "", "_One current, dated value per structured field._", ""]
    lines.extend(facts[field].render() for field in STRUCTURED_FIELDS if field in facts)
    lines.extend(passthrough)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _append_superseded(store: StoreHandle, subject: str, old: Fact, new: Fact) -> None:
    path = store.worktree / "_meta" / "superseded.md"
    if path.exists():
        content = path.read_text(encoding="utf-8").rstrip()
    else:
        content = "# Superseded Facts"
    line = (
        f"- {subject}: {old.field}={old.value} [{old.provenance}] was superseded by "
        f"{new.field}={new.value} on {new.date} [{new.provenance}]"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{content}\n{line}\n", encoding="utf-8")


def _timeline_summary(event: EvalEvent, updates: dict[str, str]) -> str:
    if updates:
        changes = ", ".join(f"{field}={updates[field]}" for field in STRUCTURED_FIELDS if field in updates)
        return changes
    return event.type


def _write_overview(path: Path, subject: str, event: EvalEvent, facts: dict[str, Fact]) -> None:
    lines = [
        f"# {subject}",
        "",
        f"_Last updated: {event.time.isoformat()} [{event.source}#{event.id}]_",
        "",
        "## Current structured fields",
        "",
    ]
    for field in STRUCTURED_FIELDS:
        display = facts[field].value if field in facts else "UNKNOWN"
        lines.append(f"- {field}: {display}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_event(store: StoreHandle, event: EvalEvent, lane: str) -> None:
    """Project one event, preserving current facts and append-only provenance."""

    entity_dir = store.worktree / entity_relpath(event.subject)
    entity_dir.mkdir(parents=True, exist_ok=True)
    facts_path = entity_dir / "facts.md"
    facts, passthrough = _read_facts(facts_path)
    updates = structured_updates(event)
    provenance = f"{event.source}#{event.id}"
    date = event.time.date().isoformat()
    for field in STRUCTURED_FIELDS:
        if field not in updates:
            continue
        new = Fact(field=field, value=updates[field], date=date, provenance=provenance)
        old = facts.get(field)
        if old is not None and old.value != new.value:
            _append_superseded(store, event.subject, old, new)
        if old is None or old.value != new.value:
            facts[field] = new
    _write_facts(facts_path, facts, passthrough)

    timeline = entity_dir / "timeline.md"
    if timeline.exists():
        content = timeline.read_text(encoding="utf-8").rstrip()
    else:
        content = "# Timeline"
    line = (
        f"- {event.time.isoformat()} [{event.source}#{event.id}] "
        f"({lane}) {_timeline_summary(event, updates)}"
    )
    timeline.write_text(f"{content}\n{line}\n", encoding="utf-8")
    _write_overview(entity_dir / "OVERVIEW.md", event.subject, event, facts)


def rebuild_index(store: StoreHandle) -> None:
    """Rebuild S1's complete entity index from the deterministic overview files."""

    rows: list[tuple[str, str, str, str]] = []
    entities = store.worktree / "entities"
    for overview in entities.glob("*/*/OVERVIEW.md"):
        lines = overview.read_text(encoding="utf-8").splitlines()
        subject = lines[0].removeprefix("# ") if lines else overview.parent.name
        updated = ""
        for line in lines:
            if match := _LAST_UPDATED.match(line):
                updated = match.group("timestamp")
                break
        relpath = overview.relative_to(store.worktree).as_posix()
        entity_type = relpath.split("/")[1]
        rows.append((subject, entity_type, updated, relpath))
    rows.sort(key=lambda row: (row[0], row[3]))

    lines = [
        "# Org Context Index",
        "",
        "| Entity | Type | Last updated | Overview |",
        "|--------|------|--------------|----------|",
    ]
    lines.extend(f"| {subject} | {kind} | {updated} | [{path}]({path}) |" for subject, kind, updated, path in rows)
    store.write("INDEX.md", "\n".join(lines) + "\n")

