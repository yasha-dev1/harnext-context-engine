"""Fixed, budgeted read agent from docs/evaluation-spec.md §5 and §7 E2."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from harnext_eval.providers.factory import make_llm
from harnext_eval.providers.llm import LLMProvider
from harnext_eval.providers.tokenizer import count_tokens
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


def truncate_to_tokens(text: str, budget_tokens: int) -> str:
    """Return the longest text prefix whose provider-token count fits the budget."""

    if budget_tokens < 0:
        raise ValueError("budget_tokens cannot be negative")
    if not text or budget_tokens == 0:
        return ""
    if count_tokens(text) <= budget_tokens:
        return text
    low, high = 0, len(text)
    while low < high:
        midpoint = (low + high + 1) // 2
        if count_tokens(text[:midpoint]) <= budget_tokens:
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
) -> Answer:
    """Answer one probe using only budgeted material and record read costs."""

    budget = _budget(cfg)
    selected = (
        truncate_to_tokens(material.text, budget) if material.enforce_budget else material.text
    )
    tokens_read = count_tokens(selected)
    started = time.perf_counter()
    result = (provider or _provider(cfg)).complete(
        SYSTEM_PROMPT,
        f"Question: {probe.question}\nMaterial:\n{selected}",
        max_tokens=max(1, min(512, budget)),
    )
    latency = time.perf_counter() - started
    response = result.text.strip() or "UNKNOWN"
    return Answer(
        probe_id=probe.probe_id,
        arm=material.arm,
        text=response,
        cited_ids=_cited_ids(response, material),
        tokens_read=tokens_read,
        tool_calls=material.tool_calls,
        latency_s=latency,
    )
