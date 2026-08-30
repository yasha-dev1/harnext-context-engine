"""Snapshot leakage gate for docs/evaluation-spec.md §4.2."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from harnext_eval.types import EvalEvent, Probe, SnapshotRef, Task

_GATE_FIELDS: tuple[str, ...] = (
    "probe_id",
    "item_id",
    "T",
    "sha",
    "last_event_id",
    "result",
    "reasons",
)
_TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)
_EVENT_TOKEN_CACHE: dict[tuple[str, str], frozenset[str]] = {}
_EVENT_TOKEN_CACHE_LIMIT = 10_000


def leakage_gate(
    item: Probe | Task,
    snapshot: SnapshotRef,
    delivery_log: Iterable[EvalEvent | Mapping[str, Any]] | str | Path,
    *,
    question: str | None = None,
    gold_action_time: datetime | str | None = None,
    all_events: Iterable[EvalEvent | Mapping[str, Any]] | None = None,
    envelope: str | Iterable[str] | None = None,
    gold_action: str | None = None,
    out_csv: str | Path = Path("out/gate.csv"),
) -> bool:
    """Check one item, append an auditable gate row, and return PASS/FAIL."""

    cutoff = item.T
    deliveries = _load_records(delivery_log)
    delivered_before_snapshot = _records_before_snapshot(deliveries, snapshot.sha)
    reasons: list[str] = []

    delivered_times = [value for row in delivered_before_snapshot if (value := _event_time(row))]
    if any(value > cutoff for value in delivered_times):
        reasons.append("delivered_event_after_T")
    if snapshot.T_last_event > cutoff:
        reasons.append("snapshot_after_T")

    corpus_records = list(all_events) if all_events is not None else deliveries
    question_text = question if question is not None else getattr(item, "question", "")
    pre_tokens: set[str] = set()
    post_tokens: set[str] = set()
    for row in corpus_records:
        event_time = _event_time(row)
        if event_time is None:
            continue
        tokens = _record_tokens(row)
        (post_tokens if event_time > cutoff else pre_tokens).update(tokens)
    only_post = _tokens(question_text) & (post_tokens - pre_tokens)
    if only_post:
        reasons.append("question_token_only_post_T:" + "|".join(sorted(only_post)))

    action_time = _as_datetime(gold_action_time) or _gold_action_time(item)
    if action_time is None or action_time <= cutoff:
        reasons.append("gold_action_not_after_T")

    envelope_text = envelope if isinstance(envelope, str) else " ".join(envelope or ())
    action_text = gold_action or _gold_action_text(item)
    if action_text and action_text.casefold() in envelope_text.casefold():
        reasons.append("gold_action_in_envelope")

    passed = not reasons
    _append_gate_row(
        Path(out_csv),
        {
            "probe_id": _item_id(item),
            "item_id": _item_id(item),
            "T": cutoff.isoformat(),
            "sha": snapshot.sha,
            "last_event_id": snapshot.last_event_id,
            "result": "PASS" if passed else "FAIL",
            "reasons": ";".join(reasons),
        },
    )
    return passed


check_leakage = leakage_gate


def assert_leakage_safe(*args: Any, **kwargs: Any) -> None:
    """Run the logged gate and raise when the item fails."""

    if not leakage_gate(*args, **kwargs):
        raise AssertionError("snapshot leakage gate failed")


def _load_records(
    source: Iterable[EvalEvent | Mapping[str, Any]] | str | Path,
) -> list[EvalEvent | Mapping[str, Any]]:
    if not isinstance(source, (str, Path)):
        return list(source)
    path = Path(source)
    if path.suffix.casefold() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            csv_records: list[EvalEvent | Mapping[str, Any]] = list(csv.DictReader(handle))
            return csv_records
    records: list[EvalEvent | Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"delivery log row in {path} must be an object")
                records.append(value)
    return records


def _records_before_snapshot(
    rows: list[EvalEvent | Mapping[str, Any]], sha: str
) -> list[EvalEvent | Mapping[str, Any]]:
    matching_indexes = [
        index
        for index, row in enumerate(rows)
        if isinstance(row, Mapping)
        and str(row.get("snapshot_sha", row.get("commit_sha", row.get("sha", "")))) == sha
    ]
    if not matching_indexes:
        return rows
    return rows[: matching_indexes[-1] + 1]


def _event_time(row: EvalEvent | Mapping[str, Any]) -> datetime | None:
    if isinstance(row, EvalEvent):
        return row.time
    value = row.get("event_time", row.get("time", row.get("t")))
    if value is None and isinstance(row.get("event"), Mapping):
        value = row["event"].get("time")  # type: ignore[index]
    return _as_datetime(value)


def _record_text(row: EvalEvent | Mapping[str, Any]) -> str:
    if isinstance(row, EvalEvent):
        return row.model_dump_json()
    return json.dumps(row, sort_keys=True, default=str)


def _record_tokens(row: EvalEvent | Mapping[str, Any]) -> set[str] | frozenset[str]:
    if not isinstance(row, EvalEvent):
        return _tokens(_record_text(row))
    key = (row.source, row.id)
    cached = _EVENT_TOKEN_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_EVENT_TOKEN_CACHE) >= _EVENT_TOKEN_CACHE_LIMIT:
        _EVENT_TOKEN_CACHE.clear()
    tokens = frozenset(_tokens(row.model_dump_json()))
    _EVENT_TOKEN_CACHE[key] = tokens
    return tokens


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(text)}


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return None


def _gold_action_time(item: Probe | Task) -> datetime | None:
    gold = item.gold
    if not isinstance(gold, Mapping):
        return None
    for key in ("gold_action_time", "action_time", "time"):
        parsed = _as_datetime(gold.get(key))
        if parsed is not None:
            return parsed
    action = gold.get("action")
    if isinstance(action, Mapping):
        return _as_datetime(action.get("time"))
    return None


def _gold_action_text(item: Probe | Task) -> str | None:
    gold = item.gold
    if not isinstance(gold, Mapping):
        return None
    for key in ("gold_action", "action"):
        value = gold.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            for text_key in ("text", "value", "action"):
                text = value.get(text_key)
                if isinstance(text, str):
                    return text
    return None


def _item_id(item: Probe | Task) -> str:
    return item.probe_id if isinstance(item, Probe) else item.task_id


def _append_gate_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=_GATE_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
