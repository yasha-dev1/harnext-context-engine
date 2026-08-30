"""Retrieval arms and floors from docs/evaluation-spec.md §7 E2."""

from __future__ import annotations

import json
import posixpath
import re
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from pathlib import PurePosixPath
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from harnext_eval.agents.reader import Material, truncate_to_tokens
from harnext_eval.providers.embeddings import EmbeddingsProvider
from harnext_eval.providers.factory import make_embeddings
from harnext_eval.providers.tokenizer import FakeTokenCounter, TokenCounter
from harnext_eval.stores.base import StoreHandle
from harnext_eval.stores.vector_index import StoreVectorIndex
from harnext_eval.types import EvalEvent, Probe

_WORD_RE = re.compile(r"[\w.-]+", flags=re.UNICODE)
_LINK_RE = re.compile(r"\[([^]]*)]\(([^)#]+)(?:#[^)]+)?\)")
_SAFE_PATH_PART_RE = re.compile(r"^[A-Za-z0-9._-]+$")


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
    """Match canonical ownership keys exactly, never mentions in payload prose."""

    needle = entity.strip().casefold()
    keys = [event.subject, *event.baseline_keys]
    return any(key.strip().casefold() == needle for key in keys)


def _event_text(event: EvalEvent) -> str:
    record = {
        "id": event.id,
        "time": event.time.isoformat(),
        "source": event.source,
        "type": event.type,
        "subject": event.subject,
        "baseline_keys": event.baseline_keys,
        "data": event.data,
    }
    return json.dumps(record, sort_keys=True, default=str, separators=(",", ":"))


def _canonical_entity_relpath(entity: str) -> str | None:
    """Resolve ordinary canonical keys; unusual slugs are reached through INDEX links."""

    if ":" not in entity:
        return None
    kind, key = entity.split(":", 1)
    key = key.replace("/", "__").replace(":", "_")
    if not _SAFE_PATH_PART_RE.fullmatch(kind) or not _SAFE_PATH_PART_RE.fullmatch(key):
        return None
    return f"entities/{kind}/{key}"


def _material(
    arm: str,
    documents: Sequence[tuple[str, str]],
    budget: int,
    *,
    tool_calls: int = 0,
    enforce_budget: bool = True,
    tokenizer: TokenCounter | None = None,
) -> Material:
    counter = tokenizer or FakeTokenCounter()
    full_text = "\n".join(text for _, text in documents)
    text = (
        truncate_to_tokens(full_text, budget, tokenizer=counter) if enforce_budget else full_text
    )
    included = [doc_id for doc_id, _ in documents if doc_id and doc_id in text]
    return Material(
        arm=arm,
        text=text,
        source_ids=included,
        tool_calls=tool_calls,
        original_tokens=counter.count(full_text),
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
    tokenizer: TokenCounter | None = None,
) -> Material:
    """A1-N: raw last-N exact entity/baseline-key events through the cutoff."""

    limit = n if n is not None else _cfg_value(cfg, "last_n", 20)
    if limit <= 0:
        raise ValueError("A1 N must be positive")
    counter = tokenizer or FakeTokenCounter()
    eligible = sorted(
        (
            event
            for event in events
            if event.time <= probe.T and _matches_entity(event, probe.entity)
        ),
        key=lambda event: (event.time, event.id),
    )[-limit:]
    documents = [(event.id, _event_text(event)) for event in eligible]
    budget = _budget(cfg)
    original_tokens = counter.count("\n".join(text for _, text in documents))
    selected: list[tuple[str, str]] = []
    remaining = budget
    for event_id, text in reversed(documents):
        separator = 1 if selected else 0
        available = max(0, remaining - separator)
        if available == 0:
            break
        piece = truncate_to_tokens(text, available, tokenizer=counter)
        if piece:
            selected.append((event_id, piece))
            remaining -= counter.count(piece) + separator
        if piece != text:
            break
    selected.reverse()
    arm = f"A1-N{limit}"
    material = _material(arm, selected, budget, tokenizer=counter)
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
    tokenizer: TokenCounter | None = None,
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
    return _material("A2", documents, _budget(cfg), tokenizer=tokenizer)


def a3(
    probe: Probe,
    events: Iterable[EvalEvent],
    cfg: Any,
    *,
    embeddings: EmbeddingsProvider | None = None,
    k: int | None = None,
    tokenizer: TokenCounter | None = None,
) -> Material:
    """A3: embedding-similarity top-k over all visible raw events."""

    candidates = _rankable_events(probe, events)
    if not candidates:
        return Material(arm="A3", text="", original_tokens=0)
    texts = [_event_text(event) for event in candidates]
    provider = embeddings or make_embeddings(cfg)
    vectors = provider.embed([probe.question, *texts])
    scores = np.asarray(vectors[1:] @ vectors[0], dtype=float)
    top_k = k if k is not None else _cfg_value(cfg, "top_k", 10)
    ranked = sorted(range(len(candidates)), key=lambda i: (scores[i], i), reverse=True)[:top_k]
    documents = [(candidates[index].id, texts[index]) for index in ranked]
    return _material("A3", documents, _budget(cfg), tokenizer=tokenizer)


def _store_documents(
    files: Iterable[str],
    read: Callable[[str], str | None],
    entity: str,
    budget: int,
    tokenizer: TokenCounter,
) -> tuple[list[tuple[str, str]], int]:
    """Traverse a curated store from entry files, following visible links."""

    relative = sorted(
        {
            relpath
            for raw in files
            if (relpath := str(PurePosixPath(str(raw).replace("\\", "/"))))
            != "snapshots.csv"
            and ".git" not in PurePosixPath(relpath).parts
        }
    )
    relative_set = set(relative)
    starts = [name for name in ("INDEX.md", "OVERVIEW.md") if name in relative_set]
    if not starts:
        return [], 0

    canonical_dir = _canonical_entity_relpath(entity) or "__resolved_from_index__"
    paths_by_folded = {path.casefold(): path for path in relative}
    canonical_dir = paths_by_folded.get(canonical_dir.casefold(), canonical_dir)
    entity_overview = paths_by_folded.get(f"{canonical_dir}/OVERVIEW.md".casefold())
    entity_key = entity.split(":", 1)[-1].strip().casefold()
    queue: deque[str] = deque(starts)
    if entity_overview is not None:
        queue.append(entity_overview)
    queued = set(queue)
    opened_paths: set[str] = set()
    documents: list[tuple[str, str]] = []
    loaded_tokens = 0
    opened = 0

    def enqueue(relpath: str, *, first: bool = False) -> None:
        candidate = paths_by_folded.get(relpath.casefold())
        if candidate is None or candidate in queued or candidate in opened_paths:
            return
        queued.add(candidate)
        (queue.appendleft if first else queue.append)(candidate)

    while queue and loaded_tokens < budget:
        relpath = queue.popleft()
        opened_paths.add(relpath)
        content = read(relpath) or ""
        opened += 1
        document = f"[file:{relpath}]\n{content}"
        remaining = budget - loaded_tokens
        selected = truncate_to_tokens(document, remaining, tokenizer=tokenizer)
        if selected:
            documents.append((relpath, selected))
            loaded_tokens += tokenizer.count(selected)
        if selected != document:
            break

        discovered: list[str] = []
        links = [
            (line, label, target)
            for line in content.splitlines()
            for label, target in _LINK_RE.findall(line)
        ]
        for line, label, target in links:
            if "://" in target or target.startswith(("/", "#")):
                continue
            resolved = posixpath.normpath(
                posixpath.join(posixpath.dirname(relpath), target)
            )
            if resolved == ".." or resolved.startswith("../"):
                continue
            label_key = label.strip().casefold().split(":", 1)[-1]
            line_cells = {
                cell.strip().casefold()
                for cell in line.split("|")
                if cell.strip()
            }
            line_entity_match = entity.strip().casefold() in line_cells or any(
                cell.split(":", 1)[-1] == entity_key for cell in line_cells
            )
            in_entity_file = relpath == canonical_dir or relpath.startswith(f"{canonical_dir}/")
            exact_entity_link = (
                resolved == canonical_dir
                or resolved.startswith(f"{canonical_dir}/")
                or label.strip().casefold() in {entity.strip().casefold(), entity_key}
                or label_key == entity_key
                or line_entity_match
            )
            if in_entity_file or exact_entity_link:
                discovered.append(resolved)
                if exact_entity_link and PurePosixPath(resolved).name.casefold() == "overview.md":
                    canonical_dir = PurePosixPath(resolved).parent.as_posix()
        if relpath == entity_overview:
            siblings = [
                path
                for path in relative
                if path.startswith(f"{canonical_dir}/")
                and PurePosixPath(path).suffix.casefold() in {".md", ".txt", ".json", ".jsonl"}
            ]
            discovered.extend(siblings)
        for target in reversed(list(dict.fromkeys(discovered))):
            enqueue(target, first=True)
    return documents, opened


def _vector_store_material(
    probe: Probe,
    store: StoreHandle,
    cfg: Any,
    *,
    embeddings: EmbeddingsProvider | None,
    tokenizer: TokenCounter,
) -> Material:
    """Read S4/S5 only through top-k retrieval at the immutable snapshot."""

    try:
        ref = store.snapshot(probe.T)
    except LookupError:
        return Material(arm=store.layout, text="", original_tokens=0, tool_calls=1)
    top_k = _cfg_value(cfg, "top_k", 10)
    hits = StoreVectorIndex(store, embeddings).query(probe.question, top_k, at=ref)
    documents = [
        (hit.item_id, f"[source:{hit.item_id}]\n{hit.document}") for hit in hits
    ]
    return _material(
        store.layout,
        documents,
        _budget(cfg),
        tool_calls=1,
        tokenizer=tokenizer,
    )


def store_read(
    probe: Probe,
    store: StoreHandle,
    cfg: Any,
    *,
    embeddings: EmbeddingsProvider | None = None,
    tokenizer: TokenCounter | None = None,
) -> Material:
    """Dispatch a store condition to its actual filesystem or vector read path."""

    counter = tokenizer or FakeTokenCounter()
    if store.layout in {"S4", "S5"}:
        return _vector_store_material(
            probe,
            store,
            cfg,
            embeddings=embeddings,
            tokenizer=counter,
        )
    try:
        ref = store.snapshot(probe.T)
    except LookupError:
        return Material(
            arm="A4" if store.layout == "S3" else store.layout,
            text="",
            original_tokens=0,
            tool_calls=1,
        )
    budget = _budget(cfg)
    files = store.list_files(ref)
    documents, opened = _store_documents(
        files,
        lambda relpath: store.read(ref, relpath),
        probe.entity,
        budget,
        counter,
    )
    arm = "A4" if store.layout == "S3" else store.layout
    return _material(arm, documents, budget, tool_calls=opened + 1, tokenizer=counter)


def a4(
    probe: Probe,
    store: StoreHandle,
    cfg: Any,
    *,
    tokenizer: TokenCounter | None = None,
) -> Material:
    """A4: navigate INDEX/OVERVIEW and entity files in snapshot(T)."""

    material = store_read(probe, store, cfg, tokenizer=tokenizer)
    if material.arm != "A4":
        raise ValueError(f"A4 requires S3, received vector condition {material.arm}")
    return material


def retrieve_everything(
    probe: Probe,
    events: Iterable[EvalEvent],
    cfg: Any,
    *,
    tokenizer: TokenCounter | None = None,
) -> Material:
    """Answerability floor: every visible event without retrieval or read caps.

    Cross-source probes are keyed by mandated identifiers carried in PR titles,
    mail subjects, and commit messages rather than by the owning ``subject``.
    The ceiling therefore exposes the complete pre-cutoff replay; restricting it
    to exact subject ownership would turn a retrieval floor into another
    retriever and make valid link probes artificially unanswerable.
    """

    del cfg
    eligible = sorted(
        (event for event in events if event.time <= probe.T),
        key=lambda event: (event.time, event.id),
    )
    documents = [(event.id, _event_text(event)) for event in eligible]
    return _material(
        "retrieve_everything",
        documents,
        0,
        enforce_budget=False,
        tokenizer=tokenizer,
    )


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
    tokenizer: TokenCounter | None = None,
) -> Material:
    """Build any named E2 arm through one integration-friendly entry point."""

    event_list = list(events)
    normalised = arm.casefold().replace("-", "_")
    if normalised == "a0":
        return a0(probe, cfg)
    if normalised in {"a1", "a1_n20"}:
        return a1(probe, event_list, cfg, n=20, tokenizer=tokenizer)
    if normalised == "a1_n100":
        return a1(probe, event_list, cfg, n=100, tokenizer=tokenizer)
    if normalised == "a2":
        return a2(probe, event_list, cfg, tokenizer=tokenizer)
    if normalised == "a3":
        return a3(
            probe,
            event_list,
            cfg,
            embeddings=embeddings,
            tokenizer=tokenizer,
        )
    if normalised == "a4":
        if store is None:
            return Material(arm="A4", text="", original_tokens=0)
        return a4(probe, store, cfg, tokenizer=tokenizer)
    if normalised in {"s0", "s1", "s2", "s4", "s5"}:
        if store is None or store.layout != normalised.upper():
            raise ValueError(f"{normalised.upper()} requires its matching StoreHandle")
        return store_read(
            probe,
            store,
            cfg,
            embeddings=embeddings,
            tokenizer=tokenizer,
        )
    if normalised == "retrieve_everything":
        return retrieve_everything(probe, event_list, cfg, tokenizer=tokenizer)
    if normalised == "retrieve_nothing":
        return retrieve_nothing(probe, cfg)
    raise ValueError(f"unknown E2 arm: {arm}")


# Descriptive aliases used by external runbooks.
build_a0 = a0
build_a1 = a1
build_a2 = a2
build_a3 = a3
build_a4 = a4
