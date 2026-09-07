"""V0–V8 context envelopes from docs/evaluation-spec.md §7 E4."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast

from harnext_eval.config import EngineConfig
from harnext_eval.providers.tokenizer import count_tokens
from harnext_eval.stores.base import StoreHandle
from harnext_eval.types import EvalEvent, SnapshotRef, Task

type EnvelopeVariant = Literal[
    "V0", "V1-N20", "V1-N100", "V2", "V3", "V4", "V5", "V6", "V7", "V8"
]
type ToolCallback = Callable[[str], str]

STATIC_PREFIX = """You are the fast-lane context agent.
Use only the supplied event and state. Cite source/event IDs. Do not access a repository.
Available just-in-time tools, when enabled: read_state, search_facts, recent_events.
Return JSON matching this schema: assignee_candidates (at most 3), reviewer_candidates (at most 3), component, duplicate_of, priority_change, suspected_locations (at most 5), draft_reply, cited_ids, action.
"""
JIT_TOOLS = ("read_state", "search_facts", "recent_events")
_WORD_RE = re.compile(r"[A-Za-z0-9_./:-]+")


def _render_sections(sections: Mapping[str, str]) -> str:
    return "\n\n".join(f"## {name}\n{content}" for name, content in sections.items())


@dataclass(frozen=True)
class Envelope:
    """Rendered envelope plus snapshot-bounded callable tools and exact accounting."""

    sections: dict[str, str]
    tokens_by_section: dict[str, int]
    tools: list[str]
    prefix: str = STATIC_PREFIX
    tool_callbacks: Mapping[str, ToolCallback] = field(default_factory=dict, repr=False)
    observed_tool_calls: int = 0
    token_counter: Callable[[str], int] = field(default=count_tokens, repr=False)

    @property
    def text(self) -> str:
        body = _render_sections(self.sections)
        return f"{self.prefix}\n{body}" if body else self.prefix

    @property
    def token_count(self) -> int:
        """Count the exact text sent to the provider, including headings/separators."""

        return self.token_counter(self.text)

    def call_tool(self, name: str, query: str = "") -> str:
        """Execute one advertised snapshot-bounded tool."""

        if name not in self.tools or name not in self.tool_callbacks:
            raise KeyError(f"tool {name!r} is not enabled for this envelope")
        return self.tool_callbacks[name](query)


def _cfg_value(cfg: Any, name: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _store_from_cfg(cfg: Any) -> StoreHandle | None:
    if isinstance(cfg, StoreHandle):
        return cfg
    for name in ("store_handle", "snapshot_store", "store"):
        candidate = _cfg_value(cfg, name)
        if isinstance(candidate, StoreHandle) or (
            candidate is not None
            and all(hasattr(candidate, attr) for attr in ("read", "list_files"))
        ):
            return cast(StoreHandle, candidate)
    return None


def _mapping_files(cfg: Any) -> dict[str, str]:
    for name in ("snapshot_files", "files", "material"):
        candidate = _cfg_value(cfg, name)
        if isinstance(candidate, Mapping):
            return {str(path): str(content) for path, content in candidate.items()}
    return {}


def _all_files(snapshot: SnapshotRef, cfg: Any) -> dict[str, str]:
    store = _store_from_cfg(cfg)
    if store is not None:
        return {
            path: content
            for path in store.list_files(snapshot)
            if (content := store.read(snapshot, path)) is not None
        }
    files = _mapping_files(cfg)
    if files:
        return files
    possible_root = Path(snapshot.sha)
    if possible_root.is_dir():
        return {
            path.relative_to(possible_root).as_posix(): path.read_text(encoding="utf-8")
            for path in possible_root.rglob("*")
            if path.is_file() and ".git" not in path.parts
        }
    return {}


def _entity_prefix(entity: str) -> str:
    if ":" in entity:
        kind, slug = entity.split(":", 1)
        return f"entities/{kind}/{slug.replace('/', '__')}/"
    return f"entities/{entity}/"


def _entity_files(files: Mapping[str, str], entity: str) -> dict[str, str]:
    """Resolve only the canonical entity directory and fail closed on a miss."""

    prefix = _entity_prefix(entity).casefold()
    return {path: body for path, body in files.items() if path.casefold().startswith(prefix)}


def _find_named(files: Mapping[str, str], names: Iterable[str]) -> list[tuple[str, str]]:
    lowered = {name.casefold() for name in names}
    return [
        (path, body)
        for path, body in sorted(files.items())
        if Path(path).name.casefold() in lowered
    ]


def _join_files(named: Iterable[tuple[str, str]]) -> str:
    return "\n\n".join(f"### {path}\n{body.rstrip()}" for path, body in named).strip()


def _events_from_cfg(cfg: Any) -> list[EvalEvent]:
    candidate = _cfg_value(cfg, "events") or _cfg_value(cfg, "replay_events")
    if candidate is not None and not isinstance(candidate, (str, bytes)):
        return [
            event if isinstance(event, EvalEvent) else EvalEvent.model_validate(event)
            for event in candidate
        ]
    replay_path = _cfg_value(cfg, "replay_path")
    if replay_path:
        with Path(replay_path).open(encoding="utf-8") as source:
            return [EvalEvent.model_validate_json(line) for line in source if line.strip()]
    return []


def _trigger_text(task: Task, cfg: Any) -> str:
    embedded = task.gold.get("_trigger_event")
    if embedded is not None:
        return json.dumps(embedded, indent=2, sort_keys=True, default=str)
    for event in _events_from_cfg(cfg):
        if event.id == task.trigger_event_id:
            return event.model_dump_json(indent=2)
    return json.dumps(
        {"id": task.trigger_event_id, "subject": task.entity, "time": task.T.isoformat()},
        indent=2,
        sort_keys=True,
    )


def _raw_event_rows(task: Task, cfg: Any, limit: int) -> list[EvalEvent]:
    return sorted(
        (
            event
            for event in _events_from_cfg(cfg)
            if event.subject == task.entity
            and event.time <= task.T
            and event.id != task.trigger_event_id
        ),
        key=lambda event: (event.time, event.id),
    )[-limit:]


def _raw_events(task: Task, cfg: Any, limit: int) -> str:
    return "\n".join(event.model_dump_json() for event in _raw_event_rows(task, cfg, limit))


def _timeline_tail(entity_files: Mapping[str, str]) -> str:
    lines: list[str] = []
    for path, body in _find_named(entity_files, ["timeline.md"]):
        lines.extend(f"[{path}] {line}" for line in body.splitlines() if line.strip())
    return "\n".join(lines[-20:])


def _matched_facts(entity_files: Mapping[str, str], query: str, limit: int = 10) -> str:
    terms = {term.casefold() for term in _WORD_RE.findall(query) if len(term) >= 3}
    scored: list[tuple[int, int, str]] = []
    index = 0
    for path, body in _find_named(entity_files, ["facts.md"]):
        for line in body.splitlines():
            if not line.strip():
                continue
            score = sum(term in line.casefold() for term in terms)
            scored.append((score, -index, f"[{path}] {line}"))
            index += 1
    scored.sort(reverse=True)
    return "\n".join(line for _, _, line in scored[:limit])


def _all_entity_text(entity_files: Mapping[str, str]) -> str:
    return _join_files(sorted(entity_files.items()))


def _action_tail(entity_files: Mapping[str, str]) -> str:
    lines: list[str] = []
    for path, body in _find_named(
        entity_files, ["actions.md", "action_log.md", "action-log.md"]
    ):
        lines.extend(f"[{path}] {line}" for line in body.splitlines() if line.strip())
    return "\n".join(lines[-10:])


def _truncate_tokens(text: str, budget: int, counter: Callable[[str], int]) -> str:
    if budget <= 0:
        return ""
    if counter(text) <= budget:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if counter(text[:middle]) <= budget:
            low = middle
        else:
            high = middle - 1
    return text[:low].rstrip()


def _fit_sections(
    prefix: str,
    sections: Mapping[str, str],
    budget: int,
    counter: Callable[[str], int],
) -> dict[str, str]:
    """Keep the rendered small envelope within its configured provider budget."""

    fitted: dict[str, str] = {}
    for name, body in sections.items():
        fixed = f"{prefix}\n{_render_sections(fitted)}" if fitted else prefix
        heading = f"\n\n## {name}\n"
        remaining = budget - counter(fixed + heading)
        fitted[name] = _truncate_tokens(body, remaining, counter)
    return fitted


def _account(
    prefix: str,
    sections: Mapping[str, str],
    counter: Callable[[str], int],
) -> dict[str, int]:
    counts = {"prefix": counter(prefix)}
    for index, (name, body) in enumerate(sections.items()):
        separator = "\n" if index == 0 else "\n\n"
        counts[name] = counter(f"{separator}## {name}\n{body}")
    return counts


def _tool_callbacks(task: Task, entity_files: Mapping[str, str], cfg: Any) -> dict[str, ToolCallback]:
    visible_events = _raw_event_rows(task, cfg, 100)

    def read_state(_: str) -> str:
        return _join_files(_find_named(entity_files, ["overview.md", "facts.md"]))

    def search_facts(query: str) -> str:
        return _matched_facts(entity_files, query or task.entity, limit=10)

    def recent_events(query: str) -> str:
        match = re.search(r"\b(\d{1,3})\b", query)
        limit = min(100, int(match.group(1))) if match else 20
        return "\n".join(event.model_dump_json() for event in visible_events[-limit:])

    return {
        "read_state": read_state,
        "search_facts": search_facts,
        "recent_events": recent_events,
    }


def execute_tools(
    envelope: Envelope,
    *,
    queries: Mapping[str, str] | None = None,
    budget_tokens: int,
) -> Envelope:
    """Execute enabled tools and append their actual results within the total budget."""

    if not envelope.tools:
        return envelope
    query_by_name = dict(queries or {})
    counter = envelope.token_counter
    transcript_parts: list[str] = []
    calls = 0
    for name in envelope.tools:
        result = envelope.call_tool(name, query_by_name.get(name, ""))
        calls += 1
        candidate = "\n\n".join([*transcript_parts, f"### {name}\n{result}"])
        base_sections = {**envelope.sections, "tool_results": candidate}
        if counter(f"{envelope.prefix}\n{_render_sections(base_sections)}") > budget_tokens:
            fixed = f"{envelope.prefix}\n{_render_sections(envelope.sections)}"
            heading = "\n\n## tool_results\n"
            transcript_parts = [
                _truncate_tokens(candidate, budget_tokens - counter(fixed + heading), counter)
            ]
            break
        transcript_parts.append(f"### {name}\n{result}")
    sections = {**envelope.sections, "tool_results": "\n\n".join(transcript_parts)}
    updated = replace(
        envelope,
        sections=sections,
        tokens_by_section=_account(envelope.prefix, sections, counter),
        observed_tool_calls=calls,
    )
    if updated.token_count > budget_tokens:
        raise AssertionError("tool transcript exceeded the configured envelope budget")
    return updated


def build(
    task: Task,
    snapshot: SnapshotRef,
    variant: str,
    cfg: EngineConfig | StoreHandle | Mapping[str, Any] | Any,
) -> Envelope:
    """Build one exact V0–V8 envelope at ``snapshot`` for ``task``."""

    normalised = variant.upper().replace("V1-20", "V1-N20").replace("V1-100", "V1-N100")
    if normalised == "V1":
        normalised = f"V1-N{int(_cfg_value(cfg, 'raw_event_limit', 20))}"
    allowed = {"V0", "V1-N20", "V1-N100", *(f"V{index}" for index in range(2, 9))}
    if normalised not in allowed:
        raise ValueError(f"unknown envelope variant {variant!r}")
    files = _all_files(snapshot, cfg)
    entity_files = _entity_files(files, task.entity)
    trigger = _trigger_text(task, cfg)
    overview = _join_files(_find_named(entity_files, ["overview.md"]))
    timeline = _timeline_tail(entity_files)
    facts = _matched_facts(entity_files, trigger)
    base = {
        "triggering_event": trigger,
        "overview": overview,
        "timeline_tail_20": timeline,
        "matched_facts_top_10": facts,
    }
    if normalised == "V0":
        sections = {"triggering_event": trigger}
    elif normalised.startswith("V1-N"):
        raw_n = int(normalised.removeprefix("V1-N"))
        sections = {
            "triggering_event": trigger,
            f"raw_entity_events_last_{raw_n}": _raw_events(task, cfg, raw_n),
        }
    elif normalised == "V2":
        sections = {"triggering_event": trigger, "overview": overview}
    elif normalised in {"V3", "V5"}:
        sections = dict(base)
    elif normalised == "V4":
        sections = {**base, "action_log_last_10": _action_tail(entity_files)}
    elif normalised == "V6":
        sections = {"triggering_event": trigger, "all_entity_files": _all_entity_text(entity_files)}
    elif normalised == "V7":
        sections = {**base, "superseded_bodies": _join_files(_find_named(entity_files, ["superseded.md"]))}
    else:
        sections = {
            "matched_facts_top_10": facts,
            "triggering_event": trigger,
            "overview": overview,
            "timeline_tail_20": timeline,
        }
    budget = int(_cfg_value(cfg, "envelope_budget_tokens", 12_000))
    configured_counter = _cfg_value(cfg, "token_counter", count_tokens)
    raw_counter = (
        configured_counter.count if hasattr(configured_counter, "count") else configured_counter
    )
    if not callable(raw_counter):
        raise TypeError("token_counter must be callable or expose count(text)")
    counter = cast(Callable[[str], int], raw_counter)
    if normalised != "V6":
        sections = _fit_sections(STATIC_PREFIX, sections, budget, counter)
    callbacks = _tool_callbacks(task, entity_files, cfg) if normalised == "V5" else {}
    return Envelope(
        sections=sections,
        tokens_by_section=_account(STATIC_PREFIX, sections, counter),
        tools=list(JIT_TOOLS) if callbacks else [],
        tool_callbacks=callbacks,
        token_counter=counter,
    )


__all__ = ["Envelope", "EnvelopeVariant", "JIT_TOOLS", "STATIC_PREFIX", "build", "execute_tools"]
