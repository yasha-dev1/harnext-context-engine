"""Layout registration and shared fold utilities for docs/evaluation-spec.md §7 E3."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

from harnext_builder.harness.base import ConversationTranscript

from harnext_eval.providers.embeddings import EmbeddingsProvider, FakeEmbeddings
from harnext_eval.stores.base import StoreHandle, register_layout
from harnext_eval.types import EvalEvent, SnapshotRef

_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")
_DELIVERED_PATH = "_meta/delivered_event_ids.jsonl"
_INPUT_META_PATH = "_meta/input.json"


@dataclass(slots=True)
class LayoutRuntime:
    """Providers used by layouts whose production implementation is configurable."""

    harness: str = "fake"
    model: str | None = None
    embeddings: EmbeddingsProvider | None = None
    max_turns: int = 40
    timeout_s: int = 300


_RUNTIMES: WeakKeyDictionary[StoreHandle, LayoutRuntime] = WeakKeyDictionary()


def configure_store(
    store: StoreHandle,
    *,
    harness: str = "fake",
    model: str | None = None,
    embeddings: EmbeddingsProvider | None = None,
    max_turns: int = 40,
    timeout_s: int = 300,
) -> None:
    """Attach offline/real provider choices without changing T0's ``StoreHandle`` API."""

    if max_turns <= 0 or timeout_s <= 0:
        raise ValueError("max_turns and timeout_s must be positive")
    _RUNTIMES[store] = LayoutRuntime(
        harness=harness,
        model=model,
        embeddings=embeddings,
        max_turns=max_turns,
        timeout_s=timeout_s,
    )


def runtime_for(store: StoreHandle) -> LayoutRuntime:
    runtime = _RUNTIMES.get(store)
    if runtime is None:
        runtime = LayoutRuntime(embeddings=FakeEmbeddings())
        _RUNTIMES[store] = runtime
    elif runtime.embeddings is None:
        runtime.embeddings = FakeEmbeddings()
    return runtime


def configured_embeddings(store: StoreHandle) -> EmbeddingsProvider | None:
    """Return an explicitly attached embedding provider without creating a default."""

    runtime = _RUNTIMES.get(store)
    return runtime.embeddings if runtime is not None else None


def safe_component(value: str, *, fallback: str = "item") -> str:
    """Return a traversal-safe path component while keeping ordinary IDs readable."""

    cleaned = _SAFE_COMPONENT.sub("_", value).strip("._")
    if not cleaned:
        cleaned = fallback
    if cleaned == value:
        return cleaned
    suffix = hashlib.sha256(value.encode()).hexdigest()[:8]
    return f"{cleaned}-{suffix}"


def ordered_events(events: Iterable[EvalEvent]) -> list[EvalEvent]:
    return sorted(events, key=lambda event: (event.time, event.id))


def delivered_event_ids(store: StoreHandle) -> list[str]:
    path = store.worktree / _DELIVERED_PATH
    if not path.exists():
        return []
    return [line for raw in path.read_text(encoding="utf-8").splitlines() if (line := raw.strip())]


def unseen_events(store: StoreHandle, events: Iterable[EvalEvent]) -> list[EvalEvent]:
    seen = set(delivered_event_ids(store))
    result: list[EvalEvent] = []
    for event in ordered_events(events):
        if event.id not in seen:
            result.append(event)
            seen.add(event.id)
    return result


def record_input_metadata(store: StoreHandle, events: Iterable[EvalEvent]) -> dict[str, object]:
    """Persist the delivered-ID ledger and hashes used by E3's same-input check."""

    existing = delivered_event_ids(store)
    seen = set(existing)
    accepted: list[EvalEvent] = []
    for event in ordered_events(events):
        if event.id not in seen:
            accepted.append(event)
            existing.append(event.id)
            seen.add(event.id)

    ids_payload = "".join(f"{event_id}\n" for event_id in existing).encode()
    event_digest = hashlib.sha256()
    prior_path = store.worktree / _INPUT_META_PATH
    prior: dict[str, object] = {}
    if prior_path.exists():
        try:
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            prior = {}
    prior_hash = str(prior.get("events_sha256", ""))
    if prior_hash:
        event_digest.update(bytes.fromhex(prior_hash))
    for event in accepted:
        event_digest.update(event.model_dump_json().encode())
        event_digest.update(b"\n")

    metadata: dict[str, object] = {
        "event_count": len(existing),
        "event_ids_sha256": hashlib.sha256(ids_payload).hexdigest(),
        "same_input_hash": hashlib.sha256(ids_payload).hexdigest(),
        "events_sha256": event_digest.hexdigest(),
        "last_event_id": existing[-1] if existing else None,
    }
    store.write(_DELIVERED_PATH, ids_payload.decode())
    store.write(_INPUT_META_PATH, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def append_usage(
    store: StoreHandle,
    transcript: ConversationTranscript,
    events: list[EvalEvent],
    lane: str,
) -> Path:
    """Write one stable accounting row beside the store from a harness transcript."""

    raw_usage = transcript.usage
    nested_candidate = raw_usage.get("usage")
    nested: dict[str, object] = (
        nested_candidate if isinstance(nested_candidate, dict) else raw_usage
    )
    def token_count(value: object) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float, str)):
            try:
                return int(value)
            except ValueError:
                return 0
        return 0

    input_tokens = sum(
        token_count(nested.get(key, 0))
        for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )
    output_tokens = token_count(nested.get("output_tokens", 0))
    cost_value = raw_usage.get("total_cost_usd")
    cost_usd = float(cost_value) if isinstance(cost_value, (int, float)) else 0.0
    row = {
        "event_ids": [event.id for event in ordered_events(events)],
        "event_count": len(events),
        "files_touched": transcript.files_changed,
        "harness": transcript.harness,
        "input_tokens": input_tokens,
        "lane": lane,
        "layout": store.layout,
        "model": transcript.model,
        "output_tokens": output_tokens,
        "status": "success" if transcript.ok else "failed",
        "stop_reason": transcript.stop_reason,
        "total_cost_usd": raw_usage.get("total_cost_usd"),
        "cost_usd": cost_usd,
        "tokens_in": input_tokens,
        "tokens_out": output_tokens,
        "usage": raw_usage,
    }
    path = store.root / "usage.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return path


def run_fold_loop(
    events: Iterable[EvalEvent],
    store: StoreHandle,
    *,
    lane: str = "batch",
    fold_size: int = 20,
) -> list[SnapshotRef]:
    """Minimal local replay loop used until/when T2's driver is available."""

    if fold_size <= 0:
        raise ValueError("fold_size must be positive")
    refs: list[SnapshotRef] = []
    batch: list[EvalEvent] = []
    for event in events:
        batch.append(event)
        if len(batch) == fold_size:
            refs.append(store.fold(batch, lane))
            batch = []
    if batch:
        refs.append(store.fold(batch, lane))
    return refs


def register_layouts() -> None:
    """Register all E3 layouts through T0's public hook."""

    from harnext_eval.stores.build_s0 import fold_s0
    from harnext_eval.stores.build_s1 import fold_s1
    from harnext_eval.stores.build_s2 import fold_s2
    from harnext_eval.stores.build_s3 import fold_s3
    from harnext_eval.stores.build_s4 import fold_s4
    from harnext_eval.stores.build_s5 import fold_s5

    for name, callback in (
        ("S0", fold_s0),
        ("S1", fold_s1),
        ("S2", fold_s2),
        ("S3", fold_s3),
        ("S4", fold_s4),
        ("S5", fold_s5),
    ):
        register_layout(name, callback)


register_layouts()
