from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from .config import Strict


class Fact(Strict):
    id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    text: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    topic: str = "general"
    path: str | None = None
    aliases: list[str] = Field(default_factory=list)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    known_to: datetime | None = None


class Source(Strict):
    id: str = Field(min_length=1)
    entity: str = Field(min_length=1)
    observed_at: datetime
    valid_at: datetime
    text: str = Field(min_length=1)
    facts: list[Fact] = Field(default_factory=list)

    @model_validator(mode="after")
    def provenance(self):
        if self.observed_at.utcoffset() is None or self.valid_at.utcoffset() is None:
            raise ValueError("source clocks must be timezone-aware")
        for fact in self.facts:
            if fact.quote not in self.text:
                raise ValueError(f"fact {fact.id} has no exact supporting source quote")
            for value in (fact.valid_from, fact.valid_to, fact.known_to):
                if value is not None and value.utcoffset() is None:
                    raise ValueError("fact clocks must be timezone-aware")
            if fact.valid_to is not None and fact.valid_to <= (fact.valid_from or self.valid_at):
                raise ValueError("invalid fact validity interval")
            if fact.known_to is not None and fact.known_to <= self.observed_at:
                raise ValueError("invalid fact knowledge interval")
        return self


class ExtractionResult(Strict):
    facts: list[Fact]


def extraction_schema() -> dict:
    """Require explicit values (including nulls) for strict structured output."""
    schema = ExtractionResult.model_json_schema()

    def visit(node):
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object":
                node["required"] = list(node.get("properties", {}))
                node["additionalProperties"] = False
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(schema)
    return schema
