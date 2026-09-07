"""Fixed, budgeted read agent from docs/evaluation-spec.md §5 and §7 E2.

Material is line-oriented so the deterministic fake reader and real reader see
the same evidence shapes:

* raw events are one complete CloudEvent-shaped JSON object per line;
* files begin with ``[file:<relative path>]`` and retain their Markdown/JSONL;
* vector hits begin with ``[source:<index item id>]`` followed by the indexed
  raw-event or curated-file chunk.

These formats keep entity, timestamp, field/value, link, file, and provenance
facts parseable without adding answer-only hints.
"""

from __future__ import annotations

import time
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Any

from harnext_eval.providers.factory import make_llm
from harnext_eval.providers.llm import LLMProvider
from harnext_eval.providers.tokenizer import FakeTokenCounter, TokenCounter, tokenizer_for
from harnext_eval.types import Answer, Probe

SYSTEM_PROMPT = "answer only from the provided material; UNKNOWN if not present; cite IDs"


@dataclass(frozen=True)
class Material:
    """Text selected by an E2 arm plus its retrieval accounting."""

    arm: str
    text: str
    source_ids: list[str] = field(default_factory=list)
    tool_calls: int = 0
    original_tokens: int | None = None
    enforce_budget: bool = True

    def __post_init__(self) -> None:
        if self.tool_calls < 0:
            raise ValueError("tool_calls cannot be negative")


def truncate_to_tokens(
    text: str,
    budget_tokens: int,
    *,
    tokenizer: TokenCounter | None = None,
) -> str:
    """Return the longest text prefix whose provider-token count fits the budget."""

    counter = tokenizer or FakeTokenCounter()
    if budget_tokens < 0:
        raise ValueError("budget_tokens cannot be negative")
    if not text or budget_tokens == 0:
        return ""
    if counter.count(text) <= budget_tokens:
        return text
    low, high = 0, len(text)
    while low < high:
        midpoint = (low + high + 1) // 2
        if counter.count(text[:midpoint]) <= budget_tokens:
            low = midpoint
        else:
            high = midpoint - 1
    return text[:low].rstrip()


def _reader_cfg(cfg: Any) -> Any:
    return getattr(cfg, "reader", cfg)


def _budget(cfg: Any) -> int:
    reader_cfg = _reader_cfg(cfg)
    budget = getattr(reader_cfg, "budget_tokens", None)
    if budget is None and isinstance(reader_cfg, dict):
        budget = reader_cfg.get("budget_tokens")
    if budget is None:
        raise ValueError("reader configuration must define budget_tokens")
    return int(budget)


def _provider(cfg: Any) -> LLMProvider:
    return make_llm(cfg)


def _cited_ids(text: str, material: Material) -> list[str]:
    if text.strip().casefold() == "unknown":
        return []
    explicit = [event_id for event_id in material.source_ids if event_id in text]
    if explicit:
        return explicit
    folded_answer = text.casefold()
    evidence = [
        event_id
        for event_id in material.source_ids
        if event_id in material.text
        and any(
            folded_answer in line.casefold()
            for line in material.text.splitlines()
            if event_id in line
        )
    ]
    return evidence


def answer(
    probe: Probe,
    material: Material,
    cfg: Any,
    *,
    provider: LLMProvider | None = None,
    tokenizer: TokenCounter | None = None,
    accounting: MutableMapping[str, int | str | bool] | None = None,
) -> Answer:
    """Answer one probe using only budgeted material and record read costs."""

    selected_provider = provider or _provider(cfg)
    counter = tokenizer or tokenizer_for(selected_provider)
    budget = _budget(cfg)
    selected = (
        truncate_to_tokens(material.text, budget, tokenizer=counter)
        if material.enforce_budget
        else material.text
    )
    tokens_read = counter.count(selected)
    user_prompt = f"Question: {probe.question}\nMaterial:\n{selected}"
    provider_input_tokens = counter.count(f"{SYSTEM_PROMPT}\n{user_prompt}")
    started = time.perf_counter()
    result = selected_provider.complete(
        SYSTEM_PROMPT,
        user_prompt,
        max_tokens=max(1, min(512, budget)),
    )
    latency = time.perf_counter() - started
    response = result.text.strip() or "UNKNOWN"
    if accounting is not None:
        accounting.update(
            {
                "selected_material_tokens": tokens_read,
                "provider_input_tokens": provider_input_tokens,
                "reported_provider_input_tokens": int(result.usage.get("input_tokens", 0)),
                "reported_provider_output_tokens": int(result.usage.get("output_tokens", 0)),
                "tokenizer_id": counter.tokenizer_id,
                "tokenizer_revision": counter.tokenizer_revision,
                "smoke_only_tokenizer": counter.smoke_only,
            }
        )
    return Answer(
        probe_id=probe.probe_id,
        arm=material.arm,
        text=response,
        cited_ids=_cited_ids(response, material),
        tokens_read=tokens_read,
        tool_calls=material.tool_calls,
        latency_s=latency,
    )
