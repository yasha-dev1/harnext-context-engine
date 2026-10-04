"""Arm F5 write side = the product's batch lane (`harnext_builder`), imported, not paraphrased.

Seed, system prompt, batch-instruction wording, event shape (CloudEvent) and tool policy
come from the product modules. What the eval adds is recorded in `fidelity/F5.md`:
the cadence (one builder run per slice of the replayed stream instead of one per closed
per-entity window), hence one instruction carrying several per-entity batches; the mapping
of stream events to CloudEvents; and, when a slice is too large to inline, the events'
JSON mounted per entity under `_event/` (the product's mount for material too big for the prompt).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from harnext_builder.agentfs.seed import SEED_FILES
from harnext_builder.event_fs import MANIFEST, rel_for
from harnext_builder.harness.base import (
    ALLOWED_TOOLS,
    DENIED_TOOLS,
    ConversationTranscript,
    EventFile,
    HarnessRequest,
    TranscriptTurn,
)
from harnext_builder.prompts import _BATCH_TMPL, SYSTEM_PROMPT, _redact
from harnext_shared import CloudEvent

from harnext_eval.e2.v2.stores import fs_tools

ORG = "apache-kafka"
REPO = "apache/kafka"
INLINE_LIMIT = 120_000  # characters of one instruction the harness takes in practice
KEEP = [p for p in SEED_FILES if p.endswith(".md")]
# Connector conventions of the product: source `<provider>:<scope>`, reverse-DNS type.
TYPES = {"gh": "com.github", "jira": "com.atlassian.jira", "mail": "org.apache.mail"}

# Eval tool policy: the product's, minus Bash (its OS sandbox cannot start on the build host:
# `bwrap: loopback: Failed RTM_NEWADDR`), plus the eval's in-process delete/rename tools.
TOOLS = [t for t in ALLOWED_TOOLS if t != "Bash"] + [fs_tools.DELETE, fs_tools.RENAME]
DENIED = [*DENIED_TOOLS, "Bash"]
# Eval-written addendum of the INSTRUCTION (the product system prompt stays byte-identical).
FS_NOTE = """\
Environment note: there is no shell here. To delete a file call the `delete` tool (`path`); \
to move or rename a file or folder call the `rename` tool (`old_path`, `new_path`; parent \
folders are created). All paths are relative to the working directory.

"""

SLICE = """\
{batches} batch windows closed together for this organization ({count} events, \
{first} → {last}); each is given below exactly as the batch lane delivers it. \
Incorporate all of them.

"""
POINTERS = """\
The events of this build are too large to inline: each batch below lists its events and \
names the `events_file` under `_event/` that holds their full JSON (see \
`_event/MANIFEST.md`). Read every one.

"""


def subject(event: dict) -> str:
    """The product keys an entity as `<type>:<natural key>` (`repo:acme/web`). The stream's
    primary entity already has that form; a PR number is qualified by its repository the
    way the GitHub connector qualifies everything it emits."""
    kind, _, key = (event.get("entities") or ["stream:unlinked"])[0].partition(":")
    return f"{kind}:{REPO}/{key}" if kind == "pr" else f"{kind}:{key}"


def source(event: dict) -> str:
    provider = event["type"].split(".")[0]
    meta = event.get("meta") or {}
    if provider == "gh":
        return f"github:{REPO}"
    if provider == "jira":
        return f"jira:{str(meta.get('issue') or 'KAFKA').split('-')[0]}"
    return f"{provider}:{meta.get('list') or 'dev@kafka.apache.org'}"


def cloud_event(event: dict) -> CloudEvent:
    provider, _, name = event["type"].partition(".")
    data = {k: event.get(k) for k in ("title", "text", "actor", "meta")}
    return CloudEvent(id=event["id"], source=source(event), type=f"{TYPES.get(provider, provider)}.{name}",
                      subject=subject(event), time=event["time"], mgtenant=ORG,
                      data=data | {"related": (event.get("entities") or [])[1:]})  # fmt: skip


def entity_dir(subj: str) -> str:
    """Seed `CLAUDE.md`: `repo:acme/web` → `entities/repo/acme__web/`."""
    kind, _, key = subj.partition(":")
    return f"entities/{kind}/{key.replace('/', '__')}"


def batches(events: list[dict]) -> dict[str, list[CloudEvent]]:
    """Per-entity windows of one slice, in order of first appearance (the classifier
    keys its windows by `subject`)."""
    out: dict[str, list[CloudEvent]] = {}
    for event in map(cloud_event, events):
        out.setdefault(event.subject, []).append(event)
    return out


def mount(subj: str) -> str:
    return rel_for(subj, "events.json")


def pointer(event: CloudEvent, title: bool) -> str:
    head = {"id": event.id, "type": event.type, "time": event.time.isoformat()}
    return json.dumps(head | ({"title": (event.data or {}).get("title")} if title else {}))


def render(groups: dict[str, list[CloudEvent]], events: list[dict], mode: str) -> str:
    def body(batch: list[CloudEvent]) -> str:
        if mode == "inline":
            return json.dumps([_redact(e) for e in batch], indent=2, default=str)
        listed = ",\n".join(pointer(e, mode == "titles") for e in batch)
        return f'{{"events_file": {json.dumps(mount(batch[0].subject))}, "events": [\n{listed}\n]}}'

    head = SLICE.format(batches=len(groups), count=len(events), first=events[0]["time"],
                        last=events[-1]["time"])  # fmt: skip
    return head + FS_NOTE + ("" if mode == "inline" else POINTERS) + "\n".join(
        _BATCH_TMPL.format(n=len(batch), subjects=subj, events_json=body(batch))
        for subj, batch in groups.items()
    )  # fmt: skip


def request(events: list[dict]) -> tuple[str, list[EventFile]]:
    """One instruction per slice: the product's batch instruction once per entity. Inline
    when it fits; otherwise the events' JSON is mounted under `_event/`."""
    groups = batches(events)
    text = render(groups, events, "inline")
    if len(text) <= INLINE_LIMIT:
        return text, []
    files = [EventFile(path=mount(subj), content=json.dumps([_redact(e) for e in batch], indent=2, default=str))
             for subj, batch in groups.items()]  # fmt: skip
    listing = "\n".join(f"- `{subj}` — {len(batch)} event(s): [{mount(subj)}]({mount(subj)})"
                        for subj, batch in groups.items())  # fmt: skip
    files.append(EventFile(path=MANIFEST, content=f"# Events of this build\n\n{listing}\n"))
    text = render(groups, events, "titles")
    return (text if len(text) <= INLINE_LIMIT else render(groups, events, "bare")), files


def digest() -> str:
    payload = {"system": SYSTEM_PROMPT, "batch": _BATCH_TMPL, "slice": SLICE, "pointers": POINTERS,
               "seed": SEED_FILES, "tools": [TOOLS, DENIED], "note": FS_NOTE, "types": TYPES}  # fmt: skip
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class FakeProductBuilder:
    """Offline stand-in: files every event of the instruction in the product layout."""

    name = "fake"

    async def run(self, req: HarnessRequest) -> ConversationTranscript:
        tree = Path(req.working_dir)
        for block in re.findall(r"```json\n(.*?)\n```", req.instruction, re.S):
            batch = json.loads(block)
            for event in json.loads((tree / batch["events_file"]).read_text()) if isinstance(batch, dict) else batch:  # fmt: skip
                incorporate(tree, event)
        turns = [TranscriptTurn(role="assistant", content="fake F5 build")]
        return ConversationTranscript(harness=self.name, model="fake", turns=turns)


def incorporate(tree: Path, event: dict) -> None:
    home, day = tree / entity_dir(event["subject"]), event["time"][:10]
    home.mkdir(parents=True, exist_ok=True)
    data = event["data"]
    title = " ".join(str(data.get("title") or "").split())
    line = f"- {day} [{event['source']}#{event['id']}] {event['type']} by {data.get('actor') or 'unknown'}: {title}\n"
    for name in ("timeline.md", "facts.md"):
        with (home / name).open("a") as handle:
            handle.write(line)
    (home / "OVERVIEW.md").write_text(f"# {event['subject']}\n\nLatest: {title} ({day}).\n")
    index, rel = tree / "INDEX.md", home.relative_to(tree)
    kind = event["subject"].partition(":")[0]
    rows = [
        r for r in index.read_text().splitlines() if not r.startswith(f"| `{event['subject']}` |")
    ]
    index.write_text("\n".join([*rows, f"| `{event['subject']}` | {kind} | {day} | {rel}/OVERVIEW.md |"]) + "\n")  # fmt: skip
