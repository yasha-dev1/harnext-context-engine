"""S0 event-dump store from docs/evaluation-spec.md §7 E3."""

from __future__ import annotations

import json
import shutil

from harnext_eval.stores.base import StoreHandle
from harnext_eval.stores.layouts import (
    delivered_event_ids,
    record_input_metadata,
    safe_component,
    unseen_events,
)
from harnext_eval.types import EvalEvent


def _reset_seed_to_dump(store: StoreHandle, allowed_event_ids: set[str]) -> None:
    keep_top_level = {".git", "_meta", "events", "INDEX.md"}
    for path in store.worktree.iterdir():
        if path.name in keep_top_level:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    metadata = store.worktree / "_meta"
    if metadata.exists():
        keep_metadata = {"delivered_event_ids.jsonl", "input.json"}
        for path in metadata.iterdir():
            if path.name in keep_metadata:
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    events = store.worktree / "events"
    if events.exists():
        for path in events.rglob("*"):
            if not path.is_file():
                continue
            lines = path.read_text(encoding="utf-8").splitlines() if path.suffix == ".md" else []
            event_id = lines[0].removeprefix("# Event ") if lines else ""
            if path.suffix != ".md" or event_id not in allowed_event_ids:
                path.unlink()


def _event_markdown(event: EvalEvent, lane: str) -> str:
    payload = json.dumps(event.model_dump(mode="json"), indent=2, sort_keys=True, default=str)
    return (
        f"# Event {event.id}\n\n"
        f"- Time: {event.time.isoformat()}\n"
        f"- Source: {event.source}\n"
        f"- Type: {event.type}\n"
        f"- Subject: {event.subject}\n"
        f"- Lane: {lane}\n\n"
        f"```json\n{payload}\n```\n"
    )


def _rebuild_index(store: StoreHandle) -> None:
    rows: list[tuple[str, str]] = []
    root = store.worktree / "events"
    if root.exists():
        for path in root.rglob("*.md"):
            relpath = path.relative_to(store.worktree).as_posix()
            first_line = path.read_text(encoding="utf-8").splitlines()[0]
            event_id = first_line.removeprefix("# Event ")
            rows.append((event_id, relpath))
    rows.sort(key=lambda row: row[1])
    lines = ["# Event Dump", ""]
    lines.extend(f"- [{event_id}]({relpath})" for event_id, relpath in rows)
    store.write("INDEX.md", "\n".join(lines).rstrip() + "\n")


def fold_s0(store: StoreHandle, events: list[EvalEvent], lane: str) -> None:
    """Write one immutable markdown representation per newly delivered event."""

    accepted = unseen_events(store, events)
    _reset_seed_to_dump(
        store,
        {*delivered_event_ids(store), *(event.id for event in accepted)},
    )
    for event in accepted:
        date = event.time.date()
        event_id = safe_component(event.id, fallback="event")
        relpath = f"events/{date:%Y/%m/%d}/{event_id}.md"
        store.write(relpath, _event_markdown(event, lane))
    record_input_metadata(store, accepted)
    _rebuild_index(store)


fold = fold_s0
