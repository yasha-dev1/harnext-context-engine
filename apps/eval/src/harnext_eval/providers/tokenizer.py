"""Provider-bound token accounting for docs/evaluation-spec.md D8.

The deterministic regex tokenizer is deliberately limited to fake-provider
smoke runs. Evidentiary providers must expose a local ``count_tokens`` method
or be an Anthropic provider, in which case the provider's count-tokens endpoint
is used behind the already explicit online configuration flag.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


@runtime_checkable
class TokenCounter(Protocol):
    """Exact-or-declared token counter selected for one reader provider."""

    @property
    def tokenizer_id(self) -> str: ...

    @property
    def tokenizer_revision(self) -> str: ...

    @property
    def smoke_only(self) -> bool: ...

    def count(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class FakeTokenCounter:
    """Stable approximation paired only with the deterministic fake reader."""

    tokenizer_id: str = "fake-regex-unicode"
    tokenizer_revision: str = "1"
    smoke_only: bool = True

    def count(self, text: str) -> int:
        return len(_TOKEN_RE.findall(text))


@dataclass(frozen=True, slots=True)
class CallableTokenCounter:
    """Adapter for providers which publish their own local token counter."""

    function: Any
    tokenizer_id: str
    tokenizer_revision: str
    smoke_only: bool = False

    def count(self, text: str) -> int:
        value = int(self.function(text))
        if value < 0:
            raise ValueError("provider tokenizer returned a negative token count")
        return value


@dataclass(frozen=True, slots=True)
class AnthropicTokenCounter:
    """Anthropic's model-bound count-tokens endpoint (online runs only)."""

    model: str
    api_key: str | None = None
    tokenizer_revision: str = "anthropic-messages-count-tokens-v1"
    smoke_only: bool = False

    @property
    def tokenizer_id(self) -> str:
        return f"anthropic:{self.model}"

    def count(self, text: str) -> int:
        try:
            anthropic = importlib.import_module("anthropic")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "provider token counting requires the optional 'anthropic' SDK"
            ) from exc
        client = anthropic.Anthropic(api_key=self.api_key)
        result = client.messages.count_tokens(
            model=self.model,
            messages=[{"role": "user", "content": text}],
        )
        return int(result.input_tokens)


def tokenizer_for(provider: Any) -> TokenCounter:
    """Resolve the tokenizer paired with ``provider`` without silent fallback."""

    explicit = getattr(provider, "tokenizer", None)
    if isinstance(explicit, TokenCounter):
        return explicit
    local_counter = getattr(provider, "count_tokens", None)
    if callable(local_counter):
        model = str(
            getattr(provider, "model_id", None)
            or getattr(provider, "model", None)
            or type(provider).__qualname__
        )
        revision = str(getattr(provider, "tokenizer_revision", "provider-local"))
        return CallableTokenCounter(local_counter, model, revision)
    model_id = str(getattr(provider, "model_id", ""))
    if model_id.startswith("fake-") or type(provider).__name__ == "FakeLLM":
        return FakeTokenCounter()
    if type(provider).__name__ == "AnthropicLLM":
        model = str(getattr(provider, "model", ""))
        if not model:
            raise ValueError("Anthropic tokenizer requires a pinned reader model")
        return AnthropicTokenCounter(model=model, api_key=getattr(provider, "api_key", None))
    raise ValueError(
        f"reader provider {type(provider).__qualname__} has no provider tokenizer"
    )


def count_tokens(text: str) -> int:
    """Count tokens for explicitly smoke-only compatibility callers.

    E2/E3 reader budgets use :func:`tokenizer_for`; this helper remains
    deterministic for fake providers and non-reader accounting.
    """

    return FakeTokenCounter().count(text)


__all__ = [
    "AnthropicTokenCounter",
    "CallableTokenCounter",
    "FakeTokenCounter",
    "TokenCounter",
    "count_tokens",
    "tokenizer_for",
]
