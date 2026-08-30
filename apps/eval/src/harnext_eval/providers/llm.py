"""Reader-model providers for docs/evaluation-spec.md §5 and D4."""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from harnext_eval.providers.tokenizer import count_tokens

_FIELD_CUES = (
    ("changed_files", ("changed_files", "changed files")),
    ("components", ("components",)),
    ("status", ("status",)),
    ("assignee", ("assignee",)),
    ("priority", ("priority",)),
    ("fixversion", ("fixversion", "fix version")),
    ("component", ("component",)),
    ("owner", ("owner",)),
    ("state", ("state",)),
    ("reviewer", ("reviewer",)),
)
_ENTITY_RE = re.compile(r"\b(?:[A-Z][A-Z0-9]+-\d+|[a-z]+:[\w./-]+)\b")


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


def _value_from_json(value: Any, field: str) -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.casefold() == field and not isinstance(child, (dict, list)):
                return str(child)
        if str(value.get("field", "")).casefold() == field and "to" in value:
            return str(value["to"])
        found = [_value_from_json(child, field) for child in value.values()]
        return next((item for item in reversed(found) if item is not None), None)
    if isinstance(value, list):
        found = [_value_from_json(child, field) for child in value]
        return next((item for item in reversed(found) if item is not None), None)
    return None


def _value_from_line(line: str, field: str) -> str | None:
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        value = _value_from_json(parsed, field)
        if value is not None:
            return value
    match = re.search(
        rf"\b{re.escape(field)}\b\s*(?::|=|\bis\b)\s*[\"']?([^,;|}}\n]+)",
        line,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip().strip("\"'.") if match else None


class FakeLLM:
    """A deterministic lexical reader over the supplied material.

    The last material line containing both the question's entity identifier and
    field name wins. Missing evidence returns ``UNKNOWN``.
    """

    model_id = "fake-lexical-v1"

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int,
    ) -> LLMResult:
        question, material = _question_and_material(system, user)
        field = _field_from_question(question)
        entities = [match.group(0) for match in _ENTITY_RE.finditer(question)]
        answer = "UNKNOWN"
        if field:
            for line in material.splitlines():
                folded = line.casefold()
                if field not in folded:
                    continue
                if entities and not any(entity.casefold() in folded for entity in entities):
                    continue
                value = _value_from_line(line, field)
                if value is not None:
                    answer = value
        answer = answer[: max(0, max_tokens * 4)] if max_tokens else ""
        payload: dict[str, Any] | None = None
        if json_schema is not None:
            properties = json_schema.get("properties", {})
            key = next(iter(properties), "answer")
            payload = {key: answer}
        return LLMResult(
            text=answer,
            json=payload,
            usage={
                "input_tokens": count_tokens(system) + count_tokens(user),
                "output_tokens": count_tokens(answer),
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
            str(getattr(block, "text", "")) for block in message.content if hasattr(block, "text")
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
