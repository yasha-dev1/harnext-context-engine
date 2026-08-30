"""Retrieval arms and floors from docs/evaluation-spec.md §7 E2."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from harnext_eval.agents.reader import Material, truncate_to_tokens
from harnext_eval.providers.embeddings import EmbeddingsProvider, FakeEmbeddings
from harnext_eval.providers.tokenizer import count_tokens
from harnext_eval.stores.base import StoreHandle
from harnext_eval.types import EvalEvent, Probe

_WORD_RE = re.compile(r"[\w.-]+", flags=re.UNICODE)
_LINK_RE = re.compile(r"\[[^]]*]\(([^)#]+)(?:#[^)]+)?\)")


def _cfg_value(cfg: Any, name: str, default: int) -> int:
    candidates = (cfg, getattr(cfg, "e2", None), getattr(cfg, "reader", None))
    for candidate in candidates:
        if candidate is None:
            continue
        value = (
            candidate.get(name) if isinstance(candidate, dict) else getattr(candidate, name, None)
        )
        if value is not None:
            return int(value)
    return default


def _budget(cfg: Any) -> int:
    reader = getattr(cfg, "reader", cfg)
    if isinstance(reader, dict):
        return int(reader["budget_tokens"])
    return int(reader.budget_tokens)


def _matches_entity(event: EvalEvent, entity: str) -> bool:
    needle = entity.casefold()
    if needle in event.subject.casefold():
        return True
    return needle in json.dumps(event.data or {}, sort_keys=True, default=str).casefold()


def _event_text(event: EvalEvent) -> str:
    record = {
        "id": event.id,
        "time": event.time.isoformat(),
        "source": event.source,
        "subject": event.subject,
        "data": event.data,
    }
    return json.dumps(record, sort_keys=True, default=str, separators=(",", ":"))


def _material(
    arm: str,
    documents: Sequence[tuple[str, str]],
    budget: int,
    *,
    tool_calls: int = 0,
    enforce_budget: bool = True,
) -> Material:
    full_text = "\n".join(text for _, text in documents)
    text = truncate_to_tokens(full_text, budget) if enforce_budget else full_text
    included = [doc_id for doc_id, _ in documents if doc_id and doc_id in text]
    return Material(
        arm=arm,
        text=text,
        source_ids=included,
        tool_calls=tool_calls,
        original_tokens=count_tokens(full_text),
        enforce_budget=enforce_budget,
    )


def a0(probe: Probe, cfg: Any) -> Material:
    """A0: no material (the model-prior arm)."""

    del probe, cfg
    return Material(arm="A0", text="", original_tokens=0)


def a1(
    probe: Probe,
    events: Iterable[EvalEvent],
    cfg: Any,
    *,
    n: int | None = None,
) -> Material:
    """A1: raw last-N entity events strictly before the probe cutoff."""

    limit = n if n is not None else _cfg_value(cfg, "last_n", 20)
    eligible = sorted(
        (
            event
            for event in events
            if event.time < probe.T and _matches_entity(event, probe.entity)
        ),
        key=lambda event: (event.time, event.id),
    )[-limit:]
    documents = [(event.id, _event_text(event)) for event in eligible]
    budget = _budget(cfg)
    original_tokens = count_tokens("\n".join(text for _, text in documents))
    selected: list[tuple[str, str]] = []
    remaining = budget
    for event_id, text in reversed(documents):
        separator = 1 if selected else 0
        available = max(0, remaining - separator)
        if available == 0:
            break
        piece = truncate_to_tokens(text, available)
        if piece:
            selected.append((event_id, piece))
            remaining -= count_tokens(piece) + separator
        if piece != text:
            break
    selected.reverse()
    material = _material("A1", selected, budget)
    return Material(
        arm=material.arm,
        text=material.text,
        source_ids=material.source_ids,
        tool_calls=material.tool_calls,
        original_tokens=original_tokens,
    )


def _rankable_events(probe: Probe, events: Iterable[EvalEvent]) -> list[EvalEvent]:
    return sorted(
        (event for event in events if event.time <= probe.T),
        key=lambda event: (event.time, event.id),
    )


def a2(
    probe: Probe,
    events: Iterable[EvalEvent],
    cfg: Any,
    *,
    k: int | None = None,
) -> Material:
    """A2: BM25 top-k over all events visible at the cutoff."""

    candidates = _rankable_events(probe, events)
    if not candidates:
        return Material(arm="A2", text="", original_tokens=0)
    texts = [_event_text(event) for event in candidates]
    tokenized = [_WORD_RE.findall(text.casefold()) for text in texts]
    scores = BM25Okapi(tokenized).get_scores(_WORD_RE.findall(probe.question.casefold()))
    top_k = k if k is not None else _cfg_value(cfg, "top_k", 10)
    ranked = sorted(range(len(candidates)), key=lambda i: (scores[i], i), reverse=True)[:top_k]
    documents = [(candidates[index].id, texts[index]) for index in ranked]
    return _material("A2", documents, _budget(cfg))


def a3(
    probe: Probe,
    events: Iterable[EvalEvent],
    cfg: Any,
    *,
    embeddings: EmbeddingsProvider | None = None,
    k: int | None = None,
) -> Material:
    """A3: embedding-similarity top-k over all visible raw events."""

    candidates = _rankable_events(probe, events)
    if not candidates:
        return Material(arm="A3", text="", original_tokens=0)
    texts = [_event_text(event) for event in candidates]
    dim = int(getattr(getattr(cfg, "embeddings", None), "dim", 64))
    provider = embeddings or FakeEmbeddings(dim=dim)
    vectors = provider.embed([probe.question, *texts])
    scores = np.asarray(vectors[1:] @ vectors[0], dtype=float)
    top_k = k if k is not None else _cfg_value(cfg, "top_k", 10)
    ranked = sorted(range(len(candidates)), key=lambda i: (scores[i], i), reverse=True)[:top_k]
    documents = [(candidates[index].id, texts[index]) for index in ranked]
    return _material("A3", documents, _budget(cfg))


def _store_documents(root: Path, entity: str, budget: int) -> tuple[list[tuple[str, str]], int]:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.parts and path.name != "snapshots.csv"
    )
    relative = {path.relative_to(root).as_posix(): path for path in files}
    starts = [
        relpath
        for relpath in relative
        if Path(relpath).name.casefold() in {"index.md", "overview.md"}
        and len(Path(relpath).parts) == 1
    ]
    if not starts:
        starts = [
            relpath
            for relpath in relative
            if Path(relpath).name.casefold() in {"index.md", "overview.md"}
        ]
    entity_folded = entity.casefold()
    ordered: list[str] = []
    cached: dict[str, str] = {}

    def add(relpath: str) -> None:
        normalised = str(Path(relpath)).replace("\\", "/")
        if normalised in relative and normalised not in ordered:
            ordered.append(normalised)

    for relpath in starts:
        add(relpath)
    for relpath in starts:
        content = relative[relpath].read_text(encoding="utf-8", errors="replace")
        cached[relpath] = content
        for target in _LINK_RE.findall(content):
            resolved = (Path(relpath).parent / target).as_posix()
            if entity_folded in target.casefold():
                add(resolved)
    for relpath in relative:
        if entity_folded in relpath.casefold():
            add(relpath)
    documents: list[tuple[str, str]] = []
    loaded_tokens = 0
    for relpath in ordered:
        content = cached.get(relpath)
        if content is None:
            content = relative[relpath].read_text(encoding="utf-8", errors="replace")
        document = f"[file:{relpath}]\n{content}"
        documents.append((relpath, document))
        loaded_tokens += count_tokens(document)
        if loaded_tokens >= budget:
            break
    return documents, len(documents)


def a4(probe: Probe, store: StoreHandle, cfg: Any) -> Material:
    """A4: navigate INDEX/OVERVIEW and entity files in snapshot(T)."""

    try:
        ref = store.snapshot(probe.T)
    except LookupError:
        return Material(arm="A4", text="", original_tokens=0, tool_calls=1)
    checkout = store.materialise(ref)
    try:
        budget = _budget(cfg)
        documents, opened = _store_documents(checkout, probe.entity, budget)
        return _material("A4", documents, budget, tool_calls=opened + 1)
    finally:
        shutil.rmtree(checkout, ignore_errors=True)


def retrieve_everything(probe: Probe, events: Iterable[EvalEvent], cfg: Any) -> Material:
    """Answerability floor: the entity's full visible history without a read cap."""

    del cfg
    eligible = sorted(
        (
            event
            for event in events
            if event.time <= probe.T and _matches_entity(event, probe.entity)
        ),
        key=lambda event: (event.time, event.id),
    )
    documents = [(event.id, _event_text(event)) for event in eligible]
    return _material("retrieve_everything", documents, 0, enforce_budget=False)


def retrieve_nothing(probe: Probe, cfg: Any) -> Material:
    """No-retrieval floor, named separately from the A0 experimental arm."""

    del probe, cfg
    return Material(arm="retrieve_nothing", text="", original_tokens=0)


def build_arm(
    arm: str,
    probe: Probe,
    events: Iterable[EvalEvent],
    cfg: Any,
    *,
    store: StoreHandle | None = None,
    embeddings: EmbeddingsProvider | None = None,
) -> Material:
    """Build any named E2 arm through one integration-friendly entry point."""

    event_list = list(events)
    normalised = arm.casefold().replace("-", "_")
    if normalised == "a0":
        return a0(probe, cfg)
    if normalised == "a1":
        return a1(probe, event_list, cfg)
    if normalised == "a2":
        return a2(probe, event_list, cfg)
    if normalised == "a3":
        return a3(probe, event_list, cfg, embeddings=embeddings)
    if normalised == "a4":
        if store is None:
            return Material(arm="A4", text="", original_tokens=0)
        return a4(probe, store, cfg)
    if normalised == "retrieve_everything":
        return retrieve_everything(probe, event_list, cfg)
    if normalised == "retrieve_nothing":
        return retrieve_nothing(probe, cfg)
    raise ValueError(f"unknown E2 arm: {arm}")


# Descriptive aliases used by external runbooks.
build_a0 = a0
build_a1 = a1
build_a2 = a2
build_a3 = a3
build_a4 = a4
