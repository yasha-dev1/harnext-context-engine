"""Ledger-backed leakage firewall for docs/evaluation-spec.md §4.2.

Preferred API::

    leakage_gate(item, store=store, T=item.T, all_events=frozen_replay,
                 material=exact_reader_or_agent_input, out_csv=gate_csv)

The gate resolves ``store.snapshot(T)`` itself, then proves the chosen SHA's
exact cumulative boundary with ``store.delivery_records(ref)``. Callers may
still use the historical ``(item, snapshot, delivery_log)`` positional form,
but it now fails closed unless the log has a unique, ordered SHA boundary.
Probe checks do not invent E4 action metadata; task-only gold checks run only
against real structured task gold (or explicit real action metadata).
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from harnext_eval.stores.base import (
    DeliveryRecord,
    SnapshotLedgerError,
    StoreHandle,
    store_for_snapshot,
)
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


def leakage_gate(
    item: Probe | Task,
    snapshot: SnapshotRef | StoreHandle | None = None,
    delivery_log: Iterable[EvalEvent | Mapping[str, Any]] | str | Path | datetime | None = None,
    *,
    store: StoreHandle | None = None,
    T: datetime | None = None,  # noqa: N803 - notation fixed by the spec
    question: str | None = None,
    gold_action_time: datetime | str | None = None,
    all_events: Iterable[EvalEvent | Mapping[str, Any]] | None = None,
    envelope: str | Iterable[str] | None = None,
    material: str | Iterable[str] | None = None,
    gold_action: Any = None,
    out_csv: str | Path = Path("out/gate.csv"),
) -> bool:
    """Prove one item's cutoff against the chosen store commit and log the result."""

    explicit_cutoff = T
    if isinstance(snapshot, StoreHandle):
        if store is not None and store is not snapshot:
            raise TypeError("store was supplied both positionally and by keyword")
        store = snapshot
        snapshot = None
        if isinstance(delivery_log, datetime):
            if explicit_cutoff is not None and explicit_cutoff != delivery_log:
                raise TypeError("T was supplied twice with different values")
            explicit_cutoff = delivery_log
            delivery_log = None

    cutoff = explicit_cutoff or item.T
    reasons: list[str] = []
    if cutoff != item.T:
        reasons.append("item_cutoff_mismatch")

    chosen = snapshot if isinstance(snapshot, SnapshotRef) else None
    if store is None and chosen is not None:
        store = store_for_snapshot(chosen.sha)
    delivered: list[DeliveryRecord | EvalEvent | Mapping[str, Any]] = []
    if store is not None:
        try:
            resolved = store.snapshot(cutoff)
            if chosen is not None and chosen.sha != resolved.sha:
                reasons.append("snapshot_not_last_safe_commit")
            chosen = resolved
            delivered = list(store.delivery_records(resolved))
        except (LookupError, SnapshotLedgerError, ValueError) as exc:
            reasons.append(f"delivery_ledger_unverifiable:{_reason_text(exc)}")
    elif chosen is None:
        reasons.append("snapshot_unavailable")
    elif delivery_log is None or isinstance(delivery_log, datetime):
        reasons.append("delivery_ledger_unavailable")
    else:
        try:
            delivered.extend(_records_before_snapshot(_load_records(delivery_log), chosen.sha))
        except (OSError, ValueError, SnapshotLedgerError) as exc:
            reasons.append(f"delivery_ledger_unverifiable:{_reason_text(exc)}")

    delivered_times: list[datetime] = []
    for row in delivered:
        value = _event_time(row)
        if value is None:
            reasons.append("delivered_event_time_missing")
        else:
            delivered_times.append(value)
    if any(value > cutoff for value in delivered_times):
        reasons.append("delivered_event_after_T")
    if chosen is not None and chosen.T_last_event > cutoff:
        reasons.append("snapshot_after_T")
    if chosen is not None and delivered_times:
        delivered_max = max(delivered_times)
        if delivered_max != chosen.T_last_event:
            reasons.append("snapshot_watermark_mismatch")

    corpus_records = list(all_events) if all_events is not None else None
    if corpus_records is not None:
        reasons.extend(_verify_ledger_against_replay(delivered, corpus_records))

    question_text = question if question is not None else getattr(item, "question", "")
    if question_text:
        if corpus_records is None:
            reasons.append("question_replay_unavailable")
        else:
            pre_tokens: set[str] = set()
            post_tokens: set[str] = set()
            for row in corpus_records:
                event_time = _event_time(row)
                if event_time is None:
                    continue
                (post_tokens if event_time > cutoff else pre_tokens).update(
                    _tokens(_record_text(row))
                )
            only_post = _tokens(question_text) & (post_tokens - pre_tokens)
            if only_post:
                reasons.append("question_token_only_post_T:" + "|".join(sorted(only_post)))

    shown_text = _join_text(material if material is not None else envelope)
    if shown_text and corpus_records is not None:
        for row in corpus_records:
            event_id = _event_id(row)
            event_time = _event_time(row)
            if event_id and event_time is not None and event_time > cutoff and event_id in shown_text:
                reasons.append(f"post_T_event_in_material:{event_id}")

    if isinstance(item, Probe):
        reasons.extend(_probe_source_reasons(item, corpus_records, cutoff))
    else:
        action_times = _task_action_times(item, gold_action_time)
        if not action_times:
            reasons.append("gold_action_time_unavailable")
        elif any(action_time <= cutoff for action_time in action_times):
            reasons.append("gold_action_not_after_T")

        gold_value = item.gold if gold_action is None else gold_action
        leaked_values = [
            value for value in _gold_strings(gold_value) if value.casefold() in shown_text.casefold()
        ]
        if leaked_values:
            reasons.append("gold_action_in_envelope:" + "|".join(sorted(set(leaked_values))))

    passed = not reasons
    _append_gate_row(
        Path(out_csv),
        {
            "probe_id": _item_id(item),
            "item_id": _item_id(item),
            "T": cutoff.isoformat(),
            "sha": chosen.sha if chosen is not None else "",
            "last_event_id": chosen.last_event_id if chosen is not None else "",
            "result": "PASS" if passed else "FAIL",
            "reasons": ";".join(dict.fromkeys(reasons)),
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
            return list(csv.DictReader(handle))
    records: list[EvalEvent | Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"delivery log row {line_number} in {path} must be an object")
            records.append(value)
    return records


def _records_before_snapshot(
    rows: list[EvalEvent | Mapping[str, Any]], sha: str
) -> list[Mapping[str, Any]]:
    if not rows:
        raise SnapshotLedgerError("legacy delivery rows must include immutable SHA metadata")
    mapping_rows: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise SnapshotLedgerError("legacy delivery rows must include immutable SHA metadata")
        mapping_rows.append(row)
    sequences: list[int] = []
    row_shas: list[str] = []
    for row in mapping_rows:
        try:
            sequences.append(int(row["sequence"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise SnapshotLedgerError("delivery ledger requires contiguous sequence values") from exc
        row_shas.append(str(row.get("snapshot_sha", row.get("commit_sha", row.get("sha", "")))))
    if sequences != list(range(len(mapping_rows))):
        raise SnapshotLedgerError("delivery ledger sequence is not contiguous")
    matching_indexes = [index for index, value in enumerate(row_shas) if value == sha]
    if not matching_indexes:
        raise SnapshotLedgerError(f"snapshot SHA has no exact delivery boundary: {sha}")
    if matching_indexes != list(range(matching_indexes[0], matching_indexes[-1] + 1)):
        raise SnapshotLedgerError(f"snapshot SHA occurs in discontiguous ledger rows: {sha}")
    return mapping_rows[: matching_indexes[-1] + 1]


def _verify_ledger_against_replay(
    delivered: list[DeliveryRecord | EvalEvent | Mapping[str, Any]],
    replay: list[EvalEvent | Mapping[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    replay_by_id: dict[str, tuple[datetime | None, int]] = {}
    for row in replay:
        event_id = _event_id(row)
        if event_id is None:
            continue
        event_time = _event_time(row)
        if event_id in replay_by_id:
            prior_time, count = replay_by_id[event_id]
            replay_by_id[event_id] = (prior_time, count + 1)
        else:
            replay_by_id[event_id] = (event_time, 1)
    for row in delivered:
        event_id = _event_id(row)
        if event_id is None:
            reasons.append("delivered_event_id_missing")
            continue
        replay_entry = replay_by_id.get(event_id)
        if replay_entry is None:
            reasons.append(f"delivered_event_not_in_replay:{event_id}")
            continue
        replay_time, count = replay_entry
        if count != 1:
            reasons.append(f"replay_event_id_not_unique:{event_id}")
        if replay_time != _event_time(row):
            reasons.append(f"delivered_event_time_mismatch:{event_id}")
    return reasons


def _probe_source_reasons(
    probe: Probe,
    corpus_records: list[EvalEvent | Mapping[str, Any]] | None,
    cutoff: datetime,
) -> list[str]:
    if not probe.source_event_ids:
        return []
    if corpus_records is None:
        return ["source_event_replay_unavailable"]
    corpus_by_id = {_event_id(row): _event_time(row) for row in corpus_records}
    reasons: list[str] = []
    for event_id in probe.source_event_ids:
        event_time = corpus_by_id.get(event_id)
        if event_time is None:
            reasons.append(f"source_event_unresolved:{event_id}")
        elif event_time > cutoff:
            reasons.append(f"source_event_after_T:{event_id}")
    return reasons


def _event_id(row: DeliveryRecord | EvalEvent | Mapping[str, Any]) -> str | None:
    if isinstance(row, DeliveryRecord):
        return row.event_id
    if isinstance(row, EvalEvent):
        return row.id
    value = row.get("event_id", row.get("id"))
    if value is None and isinstance(row.get("event"), Mapping):
        value = row["event"].get("id")  # type: ignore[index]
    return str(value) if value is not None else None


def _event_time(row: DeliveryRecord | EvalEvent | Mapping[str, Any]) -> datetime | None:
    if isinstance(row, DeliveryRecord):
        return row.event_time
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


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(text)}


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return parsed if parsed.tzinfo is not None and parsed.utcoffset() is not None else None


def _task_action_times(item: Task, explicit: datetime | str | None) -> list[datetime]:
    parsed = _as_datetime(explicit)
    if parsed is not None:
        return [parsed]
    return _nested_action_times(item.gold)


def _nested_action_times(value: Any, *, parent_key: str = "") -> list[datetime]:
    if isinstance(value, Mapping):
        result: list[datetime] = []
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in {"time", "action_time", "gold_action_time", "timestamp"}:
                parsed = _as_datetime(child)
                if parsed is not None:
                    result.append(parsed)
            result.extend(_nested_action_times(child, parent_key=normalized))
        return result
    if isinstance(value, (list, tuple, set)):
        return [item for child in value for item in _nested_action_times(child, parent_key=parent_key)]
    return []


def _gold_strings(value: Any, *, parent_key: str = "") -> list[str]:
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized not in {"time", "action_time", "gold_action_time", "timestamp"}:
                result.extend(_gold_strings(child, parent_key=normalized))
        return result
    if isinstance(value, (list, tuple, set)):
        return [item for child in value for item in _gold_strings(child, parent_key=parent_key)]
    if isinstance(value, str) and len(value.strip()) >= 3:
        return [value.strip()]
    return []


def _join_text(value: str | Iterable[str] | None) -> str:
    return value if isinstance(value, str) else " ".join(value or ())


def _item_id(item: Probe | Task) -> str:
    return item.probe_id if isinstance(item, Probe) else item.task_id


def _reason_text(exc: Exception) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(exc)).strip("_") or type(exc).__name__


def _append_gate_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=_GATE_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
