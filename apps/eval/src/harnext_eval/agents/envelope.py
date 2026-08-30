"""V0–V8 context envelopes from docs/evaluation-spec.md §7 E4."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from harnext_eval.config import EngineConfig
from harnext_eval.providers.tokenizer import count_tokens
from harnext_eval.stores.base import StoreHandle
from harnext_eval.types import EvalEvent, SnapshotRef, Task

type EnvelopeVariant = Literal["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]

STATIC_PREFIX = """You are the fast-lane context agent.
Use only the supplied event and state. Cite source/event IDs. Do not access a repository.
Available just-in-time tools, when enabled: read_state, search_facts, recent_events.
Return JSON matching this schema: assignee_candidates (at most 3), reviewer_candidates (at most 3), component, duplicate_of, priority_change, suspected_locations (at most 5), draft_reply, cited_ids, action.
"""
JIT_TOOLS = ["read_state", "search_facts", "recent_events"]
_WORD_RE = re.compile(r"[A-Za-z0-9_./:-]+")


@dataclass(frozen=True)
class Envelope:
    """Rendered envelope with auditable per-section token counts."""

    sections: dict[str, str]
    tokens_by_section: dict[str, int]
    tools: list[str]
    prefix: str = STATIC_PREFIX

    @property
    def text(self) -> str:
        body = "\n\n".join(f"## {name}\n{content}" for name, content in self.sections.items())
        return f"{self.prefix}\n{body}" if body else self.prefix

    @property
    def token_count(self) -> int:
        return sum(self.tokens_by_section.values())


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
    prefix = _entity_prefix(entity)
    selected = {path: body for path, body in files.items() if path.startswith(prefix)}
    if selected:
        return selected
    needles = {
        entity.casefold(),
        entity.split(":", 1)[-1].casefold(),
        entity.replace(":", "/").casefold(),
    }
    return {
        path: body
        for path, body in files.items()
        if any(needle and needle in path.casefold() for needle in needles)
    }


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
        return [event if isinstance(event, EvalEvent) else EvalEvent.model_validate(event) for event in candidate]
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
        {
            "id": task.trigger_event_id,
            "subject": task.entity,
            "time": task.T.isoformat(),
        },
        indent=2,
        sort_keys=True,
    )


def _raw_events(task: Task, cfg: Any, limit: int) -> str:
    selected = [
        event
        for event in _events_from_cfg(cfg)
        if event.subject == task.entity and event.time <= task.T and event.id != task.trigger_event_id
    ][-limit:]
    return "\n".join(event.model_dump_json() for event in selected)


def _timeline_tail(entity_files: Mapping[str, str]) -> str:
    bodies = _find_named(entity_files, ["timeline.md"])
    lines: list[str] = []
    for path, body in bodies:
        lines.extend(f"[{path}] {line}" for line in body.splitlines() if line.strip())
    return "\n".join(lines[-20:])


def _matched_facts(entity_files: Mapping[str, str], query: str) -> str:
    terms = {term.casefold() for term in _WORD_RE.findall(query) if len(term) >= 3}
    scored: list[tuple[int, int, str]] = []
    index = 0
    for path, body in _find_named(entity_files, ["facts.md"]):
        for line in body.splitlines():
            if not line.strip():
                continue
            folded = line.casefold()
            score = sum(term in folded for term in terms)
            scored.append((score, -index, f"[{path}] {line}"))
            index += 1
    scored.sort(reverse=True)
    return "\n".join(line for _, _, line in scored[:10])


def _all_entity_text(entity_files: Mapping[str, str]) -> str:
    return _join_files(sorted(entity_files.items()))


def _action_tail(entity_files: Mapping[str, str]) -> str:
    lines: list[str] = []
    for path, body in _find_named(entity_files, ["actions.md", "action_log.md", "action-log.md"]):
        lines.extend(f"[{path}] {line}" for line in body.splitlines() if line.strip())
    return "\n".join(lines[-10:])


def build(
    task: Task,
    snapshot: SnapshotRef,
    variant: str,
    cfg: EngineConfig | StoreHandle | Mapping[str, Any] | Any,
) -> Envelope:
    """Build one exact V0–V8 envelope at ``snapshot`` for ``task``."""

    normalised = variant.upper()
    if normalised not in {f"V{index}" for index in range(9)}:
        raise ValueError(f"unknown envelope variant {variant!r}")
    files = _all_files(snapshot, cfg)
    entity_files = _entity_files(files, task.entity)
    trigger = _trigger_text(task, cfg)
    overview = _join_files(_find_named(entity_files, ["overview.md"]))
    timeline = _timeline_tail(entity_files)
    facts = _matched_facts(entity_files, trigger)
    raw_n = int(_cfg_value(cfg, "raw_event_limit", _cfg_value(cfg, "envelope_raw_n", 20)))

    base = {
        "triggering_event": trigger,
        "overview": overview,
        "timeline_tail_20": timeline,
        "matched_facts_top_10": facts,
    }
    if normalised == "V0":
        sections = {"triggering_event": trigger}
    elif normalised == "V1":
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
        superseded = _find_named(files, ["superseded.md"])
        sections = {**base, "superseded_bodies": _join_files(superseded)}
    else:  # V8: same V3 content, shuffled with current state in the middle.
        sections = {
            "matched_facts_top_10": facts,
            "triggering_event": trigger,
            "overview": overview,
            "timeline_tail_20": timeline,
        }
    tools = list(JIT_TOOLS) if normalised == "V5" else []
    tokens = {"prefix": count_tokens(STATIC_PREFIX)}
    tokens.update({name: count_tokens(body) for name, body in sections.items()})
    return Envelope(sections=sections, tokens_by_section=tokens, tools=tools)
