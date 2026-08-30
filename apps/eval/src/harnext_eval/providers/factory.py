"""Offline-enforcing provider factory for docs/evaluation-spec.md §5."""

from __future__ import annotations

from typing import Any

from harnext_eval.providers.embeddings import EmbeddingsProvider, FakeEmbeddings
from harnext_eval.providers.llm import AnthropicLLM, FakeLLM, LLMProvider

_HARNESS_CLASSES = {"fake": "FakeHarness", "claude_code": "ClaudeCodeHarness"}
_SUMMARIES: dict[int, dict[str, str | bool]] = {}
_OFFLINE_BY_ENGINE: dict[int, bool] = {}


class OfflineViolation(RuntimeError):  # noqa: N818 - audit contract names this exception
    """Raised before an offline run can construct a network-capable component."""


def _value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _engine(cfg: Any) -> Any:
    return _value(cfg, "engine", cfg)


def _offline(cfg: Any) -> bool:
    configured = _value(cfg, "offline")
    if configured is not None:
        offline = bool(configured)
        _OFFLINE_BY_ENGINE[id(_engine(cfg))] = offline
        return offline
    return _OFFLINE_BY_ENGINE.get(id(cfg), True)


def _record(cfg: Any, key: str, value: str) -> None:
    summary = _SUMMARIES.setdefault(id(cfg), {"offline_enforced": _offline(cfg)})
    summary[key] = value


def assert_offline_ok(
    cfg: Any,
    *,
    transport: str | None = None,
    corpus_fetch: str | bool | None = None,
) -> None:
    """Reject every configured network-capable path while offline is enforced."""

    if not _offline(cfg):
        return
    engine = _engine(cfg)
    reader = _value(_value(engine, "reader", {}), "provider", "fake")
    embeddings = _value(_value(engine, "embeddings", {}), "provider", "fake")
    harness = _value(_value(engine, "builder", {}), "harness", "fake")
    transport = transport or _value(cfg, "transport")
    corpus_fetch = corpus_fetch or _value(cfg, "fetch")
    violations = []
    if reader != "fake":
        violations.append(f"reader.provider={reader}")
    if embeddings != "fake":
        violations.append(f"embeddings.provider={embeddings}")
    if harness != "fake":
        violations.append(f"builder.harness={harness}")
    if transport is not None and transport != "in-process":
        violations.append(f"transport={transport}")
    if corpus_fetch:
        violations.append(f"corpus fetch={corpus_fetch}")
    if violations:
        joined = ", ".join(violations)
        raise OfflineViolation(f"offline configuration forbids: {joined}")


def make_llm(cfg: Any) -> LLMProvider:
    """Construct the configured reader only after applying the offline guard."""

    assert_offline_ok(cfg)
    engine = _engine(cfg)
    reader = _value(engine, "reader", {})
    provider = _value(reader, "provider", "fake")
    if provider == "fake":
        resolved: LLMProvider = FakeLLM()
    elif provider == "anthropic":
        model = _value(reader, "model") or _value(_value(engine, "builder", {}), "model")
        resolved = AnthropicLLM(model=model or "claude-sonnet-5")
    else:
        raise ValueError(f"unsupported reader provider: {provider}")
    _record(cfg, "reader", type(resolved).__name__)
    return resolved


def make_embeddings(cfg: Any) -> EmbeddingsProvider:
    """Construct the configured embeddings adapter without silent fallback."""

    assert_offline_ok(cfg)
    embeddings = _value(_engine(cfg), "embeddings", {})
    provider = _value(embeddings, "provider", "fake")
    if provider != "fake":
        raise ValueError(f"unsupported embeddings provider: {provider}")
    resolved = FakeEmbeddings(dim=int(_value(embeddings, "dim", 64)))
    _record(cfg, "embeddings", type(resolved).__name__)
    return resolved


def make_harness_name(cfg: Any) -> str:
    """Resolve the builder harness name before its subprocess imports an adapter."""

    assert_offline_ok(cfg)
    harness = str(_value(_value(_engine(cfg), "builder", {}), "harness", "fake"))
    class_name = _HARNESS_CLASSES.get(harness)
    if class_name is None:
        raise ValueError(f"unsupported builder harness: {harness}")
    _record(cfg, "builder", class_name)
    return harness


def provider_summary(cfg: Any) -> dict[str, str | bool]:
    """Return a copy of the resolved provider names suitable for a manifest."""

    return dict(_SUMMARIES.get(id(cfg), {"offline_enforced": _offline(cfg)}))


__all__ = [
    "OfflineViolation",
    "assert_offline_ok",
    "make_embeddings",
    "make_harness_name",
    "make_llm",
    "provider_summary",
]
