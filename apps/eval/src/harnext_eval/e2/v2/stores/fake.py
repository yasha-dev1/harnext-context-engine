"""Fake builder: deterministic, arm-appropriate file writes with no model call.

Same idea as `harnext_builder.harness.fake`, but it honours each arm's layout so the
whole store pipeline (windows, verifiers, checkpoints, resume, export) runs offline.
It implements the harness protocol: `async run(HarnessRequest) -> ConversationTranscript`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from harnext_builder.harness.base import ConversationTranscript, HarnessRequest, TranscriptTurn

from harnext_eval.e2.v2.stores import product
from harnext_eval.e2.v2.stores.base import safe_id, write

SECTIONS = ("Overview", "Current state", "Facts", "Timeline", "Superseded")


def parse(path: Path) -> dict:
    _, head, body = path.read_text().split("---\n", 2)
    return {**yaml.safe_load(head), "text": body.strip("\n"), "path": path}


def head(name: str, description: str) -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n\n"


def slot(entity: str) -> tuple[str, str]:
    kind, _, key = entity.partition(":")
    return safe_id(kind), safe_id(key or kind)


def primary(event: dict) -> str:
    return (event.get("entities") or ["unlinked:events"])[0]


def digest_line(event: dict) -> str:
    title = " ".join(str(event.get("title") or "").split())
    return f"- {event['time'][:10]} {event['type']} by {event.get('actor') or 'unknown'}: {title} [evt:{event['id']}]"


def section_append(text: str, heading: str, line: str) -> str:
    """Append `line` at the end of the section opened by the exact `heading` line,
    creating the section at the end of the file when it is missing."""
    lines = text.rstrip("\n").split("\n")
    if heading not in lines:
        return "\n".join([*lines, "", heading, "", line]) + "\n"
    start = lines.index(heading)
    level = len(heading) - len(heading.lstrip("#"))
    end = next(
        (i for i in range(start + 1, len(lines)) if re.match(rf"#{{1,{level}}} ", lines[i])),
        len(lines),
    )
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    gap = [""] if end < len(lines) and lines[end].strip() else []
    return "\n".join([*lines[:end], line, *gap, *lines[end:]]) + "\n"


def entity_file(tree: Path, event: dict) -> Path:
    kind, key = slot(primary(event))
    path = tree / "entities" / kind / f"{key}.md"
    if not path.exists():
        body = "".join(f"## {s}\n\n" for s in SECTIONS)
        write(path, head(key, f"{primary(event)} — {event.get('title') or ''}") + body)
    text = section_append(path.read_text(), "## Facts", digest_line(event))
    path.write_text(section_append(text, "## Timeline", digest_line(event)))
    return path


def topic_file(tree: Path, event: dict) -> Path:
    topic = event["type"].split(".")[0]
    path = tree / f"{topic}.md"
    if not path.exists():
        write(path, head(topic, f"Everything that arrived through {topic}.") + f"# {topic}\n")
    path.write_text(section_append(path.read_text(), f"## {primary(event)}", digest_line(event)))
    return path


class FakeBuilder:
    name = "fake"

    def __init__(self, arm: str):
        self.arm = arm

    async def run(self, req: HarnessRequest) -> ConversationTranscript:
        if self.arm == "F5":  # the product lane: events arrive in the instruction
            return await product.FakeProductBuilder().run(req)
        tree = Path(req.working_dir)
        incoming = sorted((tree / "_event").glob("[0-9]*.md"))
        events = [parse(path) for path in incoming]
        getattr(self, self.arm.replace("-", "_").lower())(tree, events)
        return ConversationTranscript(
            harness=self.name,
            model="fake",
            turns=[TranscriptTurn(role="assistant", content=f"fake {self.arm} build")],
        )

    def f2(self, tree: Path, events: list[dict]) -> None:
        """Like the real builder: only writes the move plan (`stores/foldered.py`)."""
        plan = {}
        for path in sorted((tree / "events").rglob("*.md")):
            kind, key = slot(primary(parse(path)))
            plan[str(path.relative_to(tree))] = f"by-entity/{kind}/{key}/{path.name}"
        (tree / "_moves.json").write_text(json.dumps(plan))

    def f3(self, tree: Path, events: list[dict]) -> None:
        for path in sorted((tree / "_intake").rglob("*.md")):
            event = parse(path)
            kind, key = slot(primary(event))
            home = tree / "subjects" / kind / f"{key}.md"
            if not home.exists():
                write(home, head(key, f"Every stored fact about {primary(event)}."))
            block = f"{digest_line(event)[2:]}\n\n{event['text']}"
            home.write_text(section_append(home.read_text(), f"## {event['id']}", block))
            path.unlink()

    def f4_fact(self, tree: Path, events: list[dict]) -> None:
        for event in events:
            kind, key = slot(primary(event))
            name = f"{event['time'][:10]}-{safe_id(event['id'])}"
            write(
                tree / "entities" / kind / key / f"{name}.md",
                head(name, str(event.get("title") or "")) + digest_line(event) + "\n",
            )

    def f4_entity(self, tree: Path, events: list[dict]) -> None:
        for event in events:
            entity_file(tree, event)

    def f4_topic(self, tree: Path, events: list[dict]) -> None:
        for event in events:
            topic_file(tree, event)

    def f6(self, tree: Path, events: list[dict]) -> None:
        for event in events:
            path = tree / "log" / f"{event['time'][:7]}.md"
            if not path.exists():
                write(path, head(event["time"][:7], f"Events of {event['time'][:7]}."))
            with path.open("a") as handle:
                handle.write(digest_line(event) + "\n")
