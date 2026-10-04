"""Agent adapter contract (E2-V2-DESIGN.md "Agent adapter interface") and shared helpers."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from harnext_eval.e2.v2.errors import is_infra

Status = Literal["completed", "model_failure", "timeout", "turn_limit", "infra_error"]
CONTEXT_PATH = re.compile(r"/context(?:/[\w.@%+=:,~-]+)*")
# The only turn failures attributable to the model's own behaviour; every other failed turn
# pauses the run (a pause costs a retry, a wrongly recorded failure costs the comparison).
MODEL_CAUSED = re.compile(
    r"context (window|length)|maximum context|context_length|too (long|large)|max(imum)? turns|"
    r"max_turns|max_output_tokens|output token",
    re.IGNORECASE,
)


# A source CLI or git at a command position: start of the shell string, after a separator,
# or inside the quotes of `bash -lc '...'`; never a path component or a flag value.
CLI_CALL = re.compile(
    r"""(?:^|[;&|(`'"]|\$\(|\bxargs\s+(?:-\S+\s+)*|\btimeout\s+\S+\s)\s*(gh|jira|mail|git)\s+(?=[a-z-])"""
)


def cli_calls(commands):
    """{tool: invocations} over shell command strings."""
    found = Counter(m for command in commands for m in CLI_CALL.findall(command or ""))
    return dict(sorted(found.items()))


@dataclass(frozen=True)
class McpSpec:
    kind: str  # graphiti | rag | research | research_search | engine
    store: Path  # host path of the store export, mounted read-only at /store
    server: str = "history"
    tools: tuple[str, ...] = ()
    reader: str = "claude"  # research kind only: claude (Claude Code harness) | fake (offline)
    memory: Path | None = None  # engine kind: host path of the notes store, mounted read-only at /memory
    engine: str | None = None  # engine kind: chat endpoint preset (openrouter | ollama)


@dataclass(frozen=True)
class TrialSpec:
    task_id: str
    arm: str
    kind: str  # historical | coding
    prompt: str
    answer_schema: dict
    workspace: Path  # coding: extracted base tree; historical: empty directory
    out: Path  # trial scratch directory: traces, answer, launch metadata
    model: str
    effort: str
    timeout: int
    context: Path | None = None  # host path mounted read-only at /context
    mcp: McpSpec | None = None
    gradle_dist: str | None = None  # wrapper distribution the workspace needs, e.g. gradle-7.6
    sources: Path | None = None  # arm S-tools: the checkpoint's sources.sqlite, read-only
    # Cap on agent turns (one turn = one model response: a message or a tool action; the
    # model's private reasoning items do not count). None = unlimited. Enforced by the harness
    # from the live trace, identically for every agent CLI; reaching it is a model result.
    max_turns: int | None = None
    # Standing instructions for the agent CLI, written to its user-level instruction file in the
    # trial home (~/.claude/CLAUDE.md for Claude Code, ~/.codex/AGENTS.md for Codex): never in
    # the workspace, so they cannot end up in the agent's patch.
    instructions: str | None = None


@dataclass
class AgentResult:
    status: Status
    answer: dict | None = None
    usage: dict = field(default_factory=dict)  # input_tokens, cached_input_tokens, output_tokens
    tool_calls: dict = field(default_factory=dict)  # tool kind -> count
    context_paths: list[str] = field(default_factory=list)
    context_output_bytes: int = 0
    trace: Path | None = None
    elapsed_s: float = 0.0
    detail: str | None = None  # why the status is not `completed`
    model_turns: int = 0
    violations: list[str] = field(default_factory=list)  # e.g. a web search item in the trace
    cli_calls: dict = field(default_factory=dict)  # gh / jira / mail / git invocations

    def record(self):
        return {**asdict(self), "trace": self.trace.name if self.trace else None}


class AgentAdapter(Protocol):
    name: str

    def version(self) -> str: ...

    def run(self, trial: TrialSpec) -> AgentResult: ...


def events_of(path):
    """JSONL trace -> events; CLIs occasionally interleave a non-JSON line, which is kept out."""
    events = []
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                events.append(value)
    return events


def parse_answer(text):
    """The final message as a JSON object; tolerates a fenced block around it."""
    if not text:
        return None
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    for candidate in (text, fenced.group(1) if fenced else None):
        if candidate:
            try:
                value = json.loads(candidate)
            except ValueError:
                continue
            if isinstance(value, dict):
                return value
    return None


def classify(*, returncode, timed_out, turns, errors, stderr, answer, needs_answer) -> tuple:
    """(status, detail). Provider/machine trouble is never a model result; a timeout counts as
    the model's only when the transcript proves the model was actually working."""
    message = " | ".join(str(e) for e in errors)[:2000]
    if timed_out:
        if turns:
            return "timeout", "wall-clock limit reached"
        return "infra_error", "timeout with no model turn: " + (message or stderr[-500:])
    if is_infra(message) or (errors and not MODEL_CAUSED.search(message)):
        return "infra_error", message
    if returncode and not turns:
        return "infra_error", f"CLI exit {returncode} before any model turn: " + (
            message or stderr[-500:]
        )
    if returncode and is_infra(stderr[-4000:]):
        return "infra_error", f"CLI exit {returncode}: " + stderr[-500:]
    if errors or returncode:
        return "model_failure", message or f"CLI exit {returncode}"
    if needs_answer and answer is None:
        return "model_failure", "no parseable final answer"
    return "completed", None
