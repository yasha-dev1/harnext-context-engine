"""Independent state-gold implementations for docs/evaluation-spec.md §4 and §7 E2."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from harnext_eval.probes.common import canonical_entity
from harnext_eval.types import EvalEvent


@dataclass(frozen=True)
class FieldTransition:
    entity: str
    field: str
    old_value: Any
    new_value: Any
    time: datetime
    event_id: str
    event_order: int


@dataclass(frozen=True)
class GoldRequest:
    entity: str
    field: str
    T: datetime


@dataclass(frozen=True)
class GoldDisagreement:
    request: GoldRequest
    python_value: Any
    sql_value: Any


class PythonGold:
    """Replay normalized changelog transitions in Python."""

    def __init__(self, events: Sequence[EvalEvent]) -> None:
        histories: dict[tuple[str, str], list[FieldTransition]] = {}
        for event_order, event in enumerate(events):
            data = event.data or {}
            entity = canonical_entity(event)
            changelog = data.get("changelog")
            raw_items = changelog.get("items", []) if isinstance(changelog, dict) else []
            items = raw_items if isinstance(raw_items, list) else []
            if not items and "field" in data and "to" in data:
                items = [data]
            for item in items:
                if not isinstance(item, dict):
                    continue
                field = item.get("field")
                if not isinstance(field, str) or "to" not in item:
                    continue
                transition = FieldTransition(
                    entity=entity,
                    field=field,
                    old_value=item.get("from"),
                    new_value=item.get("to"),
                    time=event.time,
                    event_id=event.id,
                    event_order=event_order,
                )
                histories.setdefault((entity.casefold(), field.casefold()), []).append(transition)
        self._histories = histories
        for history in self._histories.values():
            history.sort(key=lambda item: (item.time, item.event_order))

    def transitions(
        self, entity: str, field: str, at: datetime | None = None
    ) -> list[FieldTransition]:
        history = self._histories.get((_canonical_key(entity), field.casefold()), [])
        return list(history) if at is None else [item for item in history if item.time <= at]

    def field_value(self, entity: str, field: str, at: datetime) -> Any:
        history = self.transitions(entity, field, at)
        return history[-1].new_value if history else None

    def histories(self) -> dict[tuple[str, str], list[FieldTransition]]:
        return {key: list(value) for key, value in self._histories.items()}


class SqlGold:
    """Query raw EvalEvent JSON in an independent in-memory SQLite replay."""

    _QUERY = """
        WITH nested AS (
            SELECT
                e.ordinal AS ordinal,
                e.event_time AS event_time,
                CAST(item.key AS INTEGER) AS item_order,
                json_extract(item.value, '$.to') AS value,
                json_type(item.value, '$.to') AS value_type
            FROM raw_events AS e
            JOIN json_each(json_extract(e.payload, '$.data.changelog.items')) AS item
            WHERE lower(
                COALESCE(
                    json_extract(e.payload, '$.data.issue_key'),
                    CASE
                        WHEN json_extract(e.payload, '$.subject') LIKE 'issue:%'
                        THEN substr(json_extract(e.payload, '$.subject'), 7)
                        ELSE json_extract(e.payload, '$.subject')
                    END
                )
            ) = lower(?)
              AND lower(json_extract(item.value, '$.field')) = lower(?)
              AND json_type(item.value, '$.to') IS NOT NULL
              AND e.event_time <= ?
        ),
        direct AS (
            SELECT
                e.ordinal AS ordinal,
                e.event_time AS event_time,
                0 AS item_order,
                json_extract(e.payload, '$.data.to') AS value,
                json_type(e.payload, '$.data.to') AS value_type
            FROM raw_events AS e
            WHERE lower(
                COALESCE(
                    json_extract(e.payload, '$.data.issue_key'),
                    CASE
                        WHEN json_extract(e.payload, '$.subject') LIKE 'issue:%'
                        THEN substr(json_extract(e.payload, '$.subject'), 7)
                        ELSE json_extract(e.payload, '$.subject')
                    END
                )
            ) = lower(?)
              AND lower(json_extract(e.payload, '$.data.field')) = lower(?)
              AND json_type(e.payload, '$.data.to') IS NOT NULL
              AND COALESCE(
                    json_array_length(json_extract(e.payload, '$.data.changelog.items')), 0
                  ) = 0
              AND e.event_time <= ?
        )
        SELECT value, value_type
        FROM (SELECT * FROM nested UNION ALL SELECT * FROM direct)
        ORDER BY event_time DESC, ordinal DESC, item_order DESC
        LIMIT 1
    """

    def __init__(self, events: Sequence[EvalEvent]) -> None:
        self._connection = sqlite3.connect(":memory:")
        self._connection.execute(
            "CREATE TABLE raw_events (ordinal INTEGER PRIMARY KEY, event_time REAL, payload TEXT)"
        )
        self._connection.executemany(
            "INSERT INTO raw_events (ordinal, event_time, payload) VALUES (?, ?, ?)",
            (
                (ordinal, event.time.timestamp(), event.model_dump_json())
                for ordinal, event in enumerate(events)
            ),
        )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> SqlGold:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def field_value(self, entity: str, field: str, at: datetime) -> Any:
        canonical = _canonical_key(entity)
        args = (canonical, field, at.timestamp(), canonical, field, at.timestamp())
        row = self._connection.execute(self._QUERY, args).fetchone()
        if row is None:
            return None
        value, value_type = row
        if value_type in {"array", "object"} and isinstance(value, str):
            return json.loads(value)
        return value


def _canonical_key(entity: str) -> str:
    return entity.split(":", 1)[1].casefold() if entity.casefold().startswith("issue:") else entity.casefold()


def field_value_python(
    events: Sequence[EvalEvent], entity: str, field: str, at: datetime
) -> Any:
    """Return field value as of T by Python transition replay."""

    return PythonGold(events).field_value(entity, field, at)


def field_value_sql(
    events: Sequence[EvalEvent], entity: str, field: str, at: datetime
) -> Any:
    """Return field value as of T by a raw-JSON SQLite query."""

    with SqlGold(events) as gold:
        return gold.field_value(entity, field, at)


# Descriptive aliases used by callers that prefer the implementation first.
python_field_value = field_value_python
sql_field_value = field_value_sql


def cross_check_gold(
    events: Sequence[EvalEvent], requests: Iterable[GoldRequest]
) -> list[GoldDisagreement]:
    """Report every disagreement between the independent gold implementations."""

    python = PythonGold(events)
    disagreements: list[GoldDisagreement] = []
    with SqlGold(events) as sql:
        for request in requests:
            python_value = python.field_value(request.entity, request.field, request.T)
            sql_value = sql.field_value(request.entity, request.field, request.T)
            if python_value != sql_value:
                disagreements.append(
                    GoldDisagreement(
                        request=request,
                        python_value=python_value,
                        sql_value=sql_value,
                    )
                )
    return disagreements


def cross_check_field_value(
    events: Sequence[EvalEvent], entity: str, field: str, at: datetime
) -> GoldDisagreement | None:
    disagreements = cross_check_gold(events, [GoldRequest(entity=entity, field=field, T=at)])
    return disagreements[0] if disagreements else None
