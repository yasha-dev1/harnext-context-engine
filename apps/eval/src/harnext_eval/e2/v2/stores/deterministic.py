"""F0 (verbatim dump) and F1 (templated per entity): built by code, no model, no synthesis."""

from __future__ import annotations

from pathlib import Path

import yaml

from harnext_eval.e2.v2.stores.base import Store, event_path, render, safe_id, write

FIELDS = ("status", "assignee", "priority", "resolution")
PR_STATE = {"gh.pr_opened": "open", "gh.pr_merged": "merged", "gh.pr_closed": "closed"}


class F0(Store):
    """`events/YYYY/MM/DD/<safe-id>.md`: frontmatter + the event text verbatim."""

    arm = "F0"

    def apply(self, tree: Path, events: list[dict], index: int, last_of_month: bool) -> dict:
        for event in events:
            write(tree / event_path(event), render(event))
        return {}


def entity_dir(entity: str) -> str:
    kind, _, key = entity.partition(":")
    return f"entities/{safe_id(kind)}/{safe_id(key or kind)}"


def changes(event: dict) -> dict:
    """Field values an event sets on its primary entity, from `meta` only."""
    meta, kind = event.get("meta") or {}, event["type"]
    if kind in PR_STATE:
        return {"status": PR_STATE[kind]}
    if kind == "jira.created":
        return {f: meta[f] for f in FIELDS if meta.get(f) is not None}
    if kind != "jira.change":
        return {}
    items = meta.get("changes") or meta.get("items") or ([meta] if "field" in meta else [])
    found = {
        str(item["field"]).lower(): item.get("to", item.get("to_string"))
        for item in items
        if item.get("field")
    }
    found |= {f: meta[f] for f in FIELDS if f in meta and "field" not in meta}
    return {f: found[f] for f in FIELDS if f in found}


def overview(state: dict) -> str:
    head = yaml.safe_dump(state, sort_keys=False, allow_unicode=True, width=10**6)
    lines = [f"# {state['entity']} — {state['title']}", ""]
    lines += [f"- {f.capitalize()}: {state[f]}" for f in FIELDS if state.get(f) is not None]
    lines += [
        f"- Linked: {', '.join(state['linked']) or 'none'}",
        f"- First activity: {state['first_activity']}",
        f"- Last activity: {state['last_activity']}",
        f"- Events: {state['events']} (see timeline.md)",
    ]
    return f"---\n{head}---\n\n" + "\n".join(lines) + "\n"


def flat(value) -> str:
    return " ".join(str(value or "").split()).replace("|", "/")


class F1(Store):
    """`entities/<kind>/<key>/{OVERVIEW,timeline}.md`. OVERVIEW's frontmatter is the whole
    entity state, so the tree alone carries what the next window needs."""

    arm = "F1"

    def apply(self, tree: Path, events: list[dict], index: int, last_of_month: bool) -> dict:
        for event in events:
            entities = event.get("entities") or ["unlinked:events"]
            for position, entity in enumerate(entities):
                folder = tree / entity_dir(entity)
                path = folder / "OVERVIEW.md"
                state = (
                    yaml.safe_load(path.read_text().split("---\n")[1])
                    if path.exists()
                    else {
                        "entity": entity,
                        "title": flat(event.get("title")),
                        **dict.fromkeys(FIELDS),
                        "linked": [],
                        "first_activity": event["time"],
                        "events": 0,
                    }
                )
                if position == 0:
                    state |= changes(event)
                    if event["type"] in ("jira.created", "gh.pr_opened"):
                        state["title"] = flat(event.get("title"))
                state["linked"] = sorted({*state["linked"], *entities} - {entity})
                state["last_activity"] = event["time"]
                state["events"] += 1
                write(path, overview(state))
                line = " | ".join(
                    [event["time"], event["type"], flat(event.get("actor")) or "-", flat(event.get("title")), f"[evt:{event['id']}]"]
                )  # fmt: skip
                timeline = folder / "timeline.md"
                if not timeline.exists():
                    timeline.write_text(f"# Timeline of {entity}\n\n")
                with timeline.open("a") as handle:
                    handle.write(f"- {line}\n")
        return {}
