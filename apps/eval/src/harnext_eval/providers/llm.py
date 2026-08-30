"""Reader providers for docs/evaluation-spec.md §5, §7 E2, and §7 E4."""

from __future__ import annotations

import importlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from harnext_eval.providers.tokenizer import count_tokens

_FIELD_CUES = (
    ("changed_files", ("changed_files", "changed files", "files and modules")),
    ("fixVersion", ("fixversion", "fix version")),
    ("components", ("components", "component")),
    ("status", ("status",)),
    ("assignee", ("assignee",)),
    ("priority", ("priority",)),
    ("owner", ("owner",)),
    ("state", ("state",)),
    ("reviewer", ("reviewer",)),
)
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Z0-9]+-\d+|(?:issue|pr|thread|kip|component|contributor):[\w./-]+)\b",
    re.IGNORECASE,
)
_ISO_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})"
)
_DATE_RE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")
_PATH_RE = re.compile(r"(?<![\w/.-])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+")
_LINK_RE = re.compile(r"\b(?:KAFKA|KIP|PR|THREAD)-\d+\b", re.IGNORECASE)
_HEX_ID_RE = re.compile(r"\b[a-f0-9]{16,64}\b", re.IGNORECASE)


@dataclass(frozen=True)
class LLMResult:
    text: str
    json: dict[str, Any] | list[Any] | None
    usage: dict[str, int]


@runtime_checkable
class LLMProvider(Protocol):
    def complete(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int,
    ) -> LLMResult: ...


@dataclass(frozen=True)
class _Record:
    value: dict[str, Any]
    order: int
    time: datetime | None
    rendered: str


@dataclass(frozen=True)
class _Fact:
    value: str
    order: int
    time: datetime | None


def _question_and_material(system: str, user: str) -> tuple[str, str]:
    marker = re.search(r"(?im)^\s*material\s*:\s*", user)
    if marker:
        return user[: marker.start()], user[marker.end() :]
    marker = re.search(r"(?im)^\s*material\s*:\s*", system)
    if marker:
        return user, system[marker.end() :]
    lines = user.splitlines()
    return (lines[0] if lines else user), "\n".join(lines[1:]) or system


def _field_from_question(question: str) -> str | None:
    folded = question.casefold()
    for field, cues in _FIELD_CUES:
        if any(cue in folded for cue in cues):
            return field
    return None


def _normal_field(value: str) -> str:
    return re.sub(r"[^a-z]", "", value.casefold())


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=parsed.tzinfo or UTC)


def _time_in_text(text: str) -> datetime | None:
    if match := _ISO_RE.search(text):
        return _parse_time(match.group(0))
    if match := _DATE_RE.search(text):
        return datetime.fromisoformat(match.group(1)).replace(tzinfo=UTC)
    return None


def _cutoff_from_question(question: str) -> datetime | None:
    marker = re.search(r"\bas of\s+", question, re.IGNORECASE)
    if marker and (match := _ISO_RE.search(question, marker.end())):
        return _parse_time(match.group(0))
    return None


def _targets(question: str) -> list[str]:
    return list(dict.fromkeys(match.group(0) for match in _ENTITY_RE.finditer(question)))


def _target_matches(rendered: str, targets: list[str]) -> bool:
    if not targets:
        return True
    folded = rendered.casefold()
    return any(
        target.casefold() in folded
        or target.split(":", 1)[-1].casefold() in folded
        for target in targets
    )


def _json_records(material: str) -> list[_Record]:
    values: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(candidate: Any) -> None:
        if not isinstance(candidate, dict):
            return
        rendered = json.dumps(candidate, sort_keys=True, default=str)
        if rendered not in seen:
            values.append(candidate)
            seen.add(rendered)

    for block in re.findall(r"```(?:json)?\s*(.*?)```", material, flags=re.IGNORECASE | re.DOTALL):
        try:
            add(json.loads(block))
        except json.JSONDecodeError:
            pass
    for line in material.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            add(json.loads(line))
        except json.JSONDecodeError:
            pass
    return [
        _Record(
            value=value,
            order=order,
            time=_parse_time(value.get("time")),
            rendered=json.dumps(value, sort_keys=True, default=str),
        )
        for order, value in enumerate(values)
    ]


def _string_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip()


def _json_field(value: Any, field: str) -> Any | None:
    wanted = _normal_field(field)
    if isinstance(value, dict):
        raw_field = value.get("field")
        if isinstance(raw_field, str) and _normal_field(raw_field) == wanted and "to" in value:
            return value["to"]
        direct = [child for key, child in value.items() if _normal_field(str(key)) == wanted]
        if direct:
            return direct[-1]
        found = [_json_field(child, field) for child in value.values()]
        return next((item for item in reversed(found) if item is not None), None)
    if isinstance(value, list):
        found = [_json_field(child, field) for child in value]
        return next((item for item in reversed(found) if item is not None), None)
    return None


def _plain_facts(material: str, field: str, targets: list[str]) -> list[_Fact]:
    facts: list[_Fact] = []
    context_time: datetime | None = None
    context_entity = ""
    pattern = re.compile(
        rf"(?:^|[- (]){re.escape(field)}\s*(?::|=|\bis\b)\s*(.+?)(?=,\s*[A-Za-z][\w]*\s*=|\s+\[[^]]+\]\s*$|$)",
        flags=re.IGNORECASE,
    )
    for order, line in enumerate(material.splitlines()):
        if parsed := _time_in_text(line):
            context_time = parsed
        if line.startswith("[file:") or line.startswith("### ") or line.startswith("# issue:"):
            context_entity = line
        match = pattern.search(line)
        if not match:
            continue
        scope = f"{context_entity} {line}"
        if not _target_matches(scope, targets):
            continue
        value = match.group(1).strip().strip("`\"'. ")
        if value:
            facts.append(_Fact(value=value, order=order, time=_time_in_text(line) or context_time))
    return facts


def _latest_value(question: str, material: str) -> str:
    field = _field_from_question(question)
    if field is None:
        return "UNKNOWN"
    targets = _targets(question)
    cutoff = _cutoff_from_question(question)
    facts: list[_Fact] = []
    for record in _json_records(material):
        if cutoff is not None and record.time is not None and record.time > cutoff:
            continue
        if not _target_matches(record.rendered, targets):
            continue
        value = _json_field(record.value.get("data", record.value), field)
        if value is not None:
            facts.append(_Fact(_string_value(value), record.order, record.time))
    facts.extend(_plain_facts(material, field, targets))
    if cutoff is not None:
        facts = [fact for fact in facts if fact.time is not None and fact.time <= cutoff]
    if not facts:
        return "UNKNOWN"
    latest = max(
        facts,
        key=lambda fact: (
            fact.time or datetime.min.replace(tzinfo=UTC),
            fact.order,
        ),
    )
    return latest.value


def _records_for_question(question: str, material: str) -> list[_Record]:
    targets = _targets(question)
    cutoff = _cutoff_from_question(question)
    return [
        record
        for record in _json_records(material)
        if _target_matches(record.rendered, targets)
        and not (cutoff is not None and record.time is not None and record.time > cutoff)
    ]


def _links_answer(question: str, material: str) -> str:
    links: list[str] = []
    for record in _records_for_question(question, material):
        data = record.value.get("data", {})
        kind = str(record.value.get("type", "")).casefold()
        if isinstance(data, dict) and data.get("number") and (
            "pull_request" in kind or data.get("pr_key")
        ):
            links.append(f"pr:{str(data['number']).removeprefix('pr:')}")
        if isinstance(data, dict) and data.get("thread_id") and (
            "mail" in kind or "message" in kind or data.get("thread_key")
        ):
            links.append(f"thread:{str(data['thread_id']).removeprefix('thread:')}")
    if not links:
        links.extend(_LINK_RE.findall(material))
    unique = list(dict.fromkeys(links))
    return "\n".join(unique) if unique else "UNKNOWN"


def _files_answer(question: str, material: str) -> str:
    files: list[str] = []
    for record in _records_for_question(question, material):
        data = record.value.get("data", {})
        raw = data.get("changed_files", []) if isinstance(data, dict) else []
        if isinstance(raw, list):
            files.extend(str(item) for item in raw if isinstance(item, str))
    if not files:
        files.extend(_PATH_RE.findall(material))
    files = [path for path in dict.fromkeys(files) if not path.startswith(("entities/", "events/"))]
    return "\n".join(files) if files else "UNKNOWN"


def _schema_value(spec: dict[str, Any], answer: str) -> Any:
    choices = spec.get("enum")
    if isinstance(choices, list) and choices:
        return "tie" if "tie" in choices else choices[0]
    kind = spec.get("type")
    kinds = set(kind) if isinstance(kind, list) else {kind}
    if "array" in kinds:
        return [] if answer == "UNKNOWN" else [answer]
    if "boolean" in kinds:
        return False
    if "object" in kinds:
        return {}
    if kinds.intersection({"integer", "number"}):
        return 0
    return None if answer == "UNKNOWN" and "null" in kinds else answer


def _action_payload(material: str, properties: dict[str, Any]) -> dict[str, Any]:
    records = _json_records(material)
    entity = "the entity"
    for record in records:
        data = record.value.get("data", {})
        issue = data.get("issue_key") if isinstance(data, dict) else None
        subject = record.value.get("subject")
        if issue:
            entity = str(issue)
            break
        if isinstance(subject, str) and subject:
            entity = subject
            break
    scoped_records = [
        record for record in records if _target_matches(record.rendered, [entity])
    ] or records
    actors: Counter[str] = Counter()
    reviewers: Counter[str] = Counter()
    ids: list[str] = []
    for record in scoped_records:
        data = record.value.get("data", {})
        if isinstance(data, dict):
            actor = data.get("actor") or data.get("author") or data.get("from")
            if actor and "bot" not in str(actor).casefold():
                actors[str(actor)] += 1
                if "review" in str(record.value.get("type", "")).casefold():
                    reviewers[str(actor)] += 1
        if record.value.get("id"):
            ids.append(str(record.value["id"]))
    ids.extend(_HEX_ID_RE.findall(material))
    paths = [
        path
        for path in dict.fromkeys(_PATH_RE.findall(material))
        if not path.startswith(("entities/", "events/", "_meta/"))
    ]
    links = [key.upper() for key in dict.fromkeys(_LINK_RE.findall(material))]
    ids.extend(links)
    ids = list(dict.fromkeys(ids))
    component = _latest_value(f"What is the component of {entity}?", material)
    if component != "UNKNOWN":
        try:
            component_value = json.loads(component)
        except json.JSONDecodeError:
            component_value = component
        if isinstance(component_value, list):
            component = str(component_value[0]) if component_value else "UNKNOWN"
    priority = _latest_value(f"What is the priority of {entity}?", material)
    assignees = [name for name, _ in actors.most_common(3)]
    reviewer_names = [name for name, _ in reviewers.most_common(3)] or assignees
    base: dict[str, Any] = {
        "assignee_candidates": assignees[:3],
        "reviewer_candidates": reviewer_names[:3],
        "component": None if component == "UNKNOWN" else component,
        "duplicate_of": next(
            (
                key
                for key in links
                if key.startswith("KAFKA-")
                and key != entity.split(":", 1)[-1].upper()
            ),
            None,
        ),
        "priority_change": priority if priority.casefold() in {"critical", "blocker"} else None,
        "suspected_locations": paths[:5],
        "draft_reply": f"Reviewing {entity} from evidence {', '.join(ids[:3])}." if ids else f"Reviewing {entity} from the supplied envelope.",
        "cited_ids": ids,
        "action": "escalate_and_route" if re.search(r"\b(?:critical|blocker|cve)\b|\[vote\]", material, re.I) else "route_and_reply",
    }
    return {
        name: base.get(name, _schema_value(spec, "UNKNOWN"))
        for name, spec in properties.items()
    }


class FakeLLM:
    """Deterministic, time-aware structured reader over supplied evidence."""

    model_id = "fake-structured-v2"

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int,
    ) -> LLMResult:
        question, material = _question_and_material(system, user)
        folded = question.casefold()
        if "which pull requests" in folded or "mail threads" in folded:
            answer = _links_answer(question, material)
        elif "which files" in folded or "code location" in folded:
            answer = _files_answer(question, material)
        else:
            answer = _latest_value(question, material)

        payload: dict[str, Any] | None = None
        if json_schema is not None:
            properties = json_schema.get("properties", {})
            action_keys = {"assignee_candidates", "component", "suspected_locations", "cited_ids"}
            if action_keys.issubset(properties):
                payload = _action_payload(material, properties)
            else:
                payload = {
                    name: _schema_value(spec, answer)
                    for name, spec in properties.items()
                }
            rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        else:
            rendered = answer
        if max_tokens <= 0:
            rendered = ""
        elif json_schema is None:
            rendered = rendered[: max_tokens * 4]
        return LLMResult(
            text=rendered,
            json=payload,
            usage={
                "input_tokens": count_tokens(system) + count_tokens(user),
                "output_tokens": count_tokens(rendered),
            },
        )


class AnthropicLLM:
    """Thin Anthropic adapter whose optional SDK is imported only on use."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int,
    ) -> LLMResult:
        try:
            anthropic = importlib.import_module("anthropic")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "AnthropicLLM requires the optional 'anthropic' SDK; install it to use this provider"
            ) from exc
        client = anthropic.Anthropic(api_key=self.api_key)
        schema_instruction = ""
        if json_schema is not None:
            schema_instruction = f"\nReturn only JSON matching this schema: {json.dumps(json_schema)}"
        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,
            system=system + schema_instruction,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            str(getattr(block, "text", ""))
            for block in message.content
            if hasattr(block, "text")
        )
        payload: dict[str, Any] | list[Any] | None = None
        if json_schema is not None:
            parsed = json.loads(text)
            if isinstance(parsed, (dict, list)):
                payload = parsed
        return LLMResult(
            text=text,
            json=payload,
            usage={
                "input_tokens": int(getattr(message.usage, "input_tokens", 0)),
                "output_tokens": int(getattr(message.usage, "output_tokens", 0)),
            },
        )
