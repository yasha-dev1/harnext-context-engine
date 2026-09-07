"""Independent state-gold implementations for docs/evaluation-spec.md §4 and §7 E2."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from harnext_eval.probes.common import canonical_entity, string_value, unique
from harnext_eval.types import EvalEvent

RawJiraInput = Mapping[str, Any] | Sequence[Mapping[str, Any]] | str | Path


@dataclass(frozen=True)
class FieldTransition:
    entity: str
    field: str
    old_value: Any
    new_value: Any
    time: datetime
    event_id: str
    event_order: int
    source_kind: str = "jira"


@dataclass(frozen=True)
class GoldRequest:
    entity: str
    field: str
    T: datetime

    def stable_key(self) -> str:
        return f"{_canonical_key(self.entity)}|{self.field.casefold()}|{self.T.isoformat()}"


@dataclass(frozen=True)
class GoldDisagreement:
    request: GoldRequest
    python_value: Any
    sql_value: Any
    resolution: Any | None = None


@dataclass
class GoldAuditTrail:
    """Collect the independent-oracle denominator and every reconciliation."""

    source: str
    resolutions: Mapping[str, Any] = field(default_factory=dict)
    comparisons: dict[str, dict[str, Any]] = field(default_factory=dict)

    def compare(self, request: GoldRequest, python_value: Any, sql_value: Any) -> Any | None:
        key = request.stable_key()
        agreed = python_value == sql_value
        resolved = key in self.resolutions
        resolution = self.resolutions.get(key)
        self.comparisons[key] = {
            "entity": request.entity,
            "field": request.field,
            "T": request.T.isoformat(),
            "python_value": python_value,
            "sql_value": sql_value,
            "agreed": agreed,
            "resolved": resolved,
            "resolution": resolution,
            "kept": agreed or resolved,
        }
        if agreed:
            return python_value
        return resolution if resolved else None

    @property
    def agreement_rate(self) -> float:
        if not self.comparisons:
            return 0.0
        agreements = sum(bool(row["agreed"]) for row in self.comparisons.values())
        return agreements / len(self.comparisons)

    def require_valid(self, *, evidentiary: bool, minimum: float = 0.98) -> None:
        if evidentiary and self.source != "raw-jira-export":
            raise ValueError("evidentiary temporal/update gold requires --raw-jira")
        if evidentiary and not self.comparisons:
            raise ValueError("evidentiary generation produced no dual-oracle comparisons")
        if self.comparisons and self.agreement_rate < minimum:
            raise ValueError(
                f"independent gold agreement {self.agreement_rate:.3%} is below {minimum:.0%}"
            )
        unresolved = [row for row in self.comparisons.values() if not row["kept"]]
        if evidentiary and unresolved:
            raise ValueError(f"{len(unresolved)} gold disagreements lack recorded resolutions")

    def report(self) -> dict[str, Any]:
        rows = sorted(
            self.comparisons.values(),
            key=lambda row: (row["entity"], row["field"], row["T"]),
        )
        return {
            "source": self.source,
            "status": (
                "evidentiary"
                if self.source == "raw-jira-export"
                else "non-evidentiary-smoke"
            ),
            "comparisons": len(rows),
            "agreements": sum(bool(row["agreed"]) for row in rows),
            "agreement_rate": self.agreement_rate,
            "disagreements": [row for row in rows if not row["agreed"]],
            "rows": rows,
        }


class PythonGold:
    """Replay EvalEvents in Python, including every state kind catalogued in §4."""

    def __init__(self, events: Sequence[EvalEvent]) -> None:
        histories: dict[tuple[str, str], list[FieldTransition]] = {}
        for event_order, event in enumerate(events):
            for transition in _event_transitions(event, event_order):
                key = (_canonical_key(transition.entity), transition.field.casefold())
                histories.setdefault(key, []).append(transition)
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
    """Query untouched Jira issue JSON through an independently defined SQLite replay."""

    def __init__(self, raw_jira: RawJiraInput | Sequence[EvalEvent]) -> None:
        event_sequence = _as_event_sequence(raw_jira)
        if event_sequence is not None:
            issues = _smoke_raw_jira_from_events(event_sequence)
            self.source = "normalised-smoke-adapter"
        else:
            issues = _load_raw_issues(cast(RawJiraInput, raw_jira))
            self.source = "raw-jira-export"
        self._issues = issues
        self._connection = sqlite3.connect(":memory:")
        self._connection.create_function("raw_identity", 3, _raw_identity)
        self._connection.create_function("jira_change_value", 3, _sql_changelog_value)
        self._connection.execute("CREATE TABLE raw_issues (payload TEXT NOT NULL)")
        self._connection.executemany(
            "INSERT INTO raw_issues (payload) VALUES (?)",
            ((json.dumps(issue, sort_keys=True),) for issue in issues),
        )
        self._connection.execute(
            """
            CREATE TABLE state_rows (
                entity TEXT NOT NULL,
                field TEXT NOT NULL,
                event_time TEXT NOT NULL,
                event_order INTEGER NOT NULL,
                value_json TEXT
            )
            """
        )
        self._populate_initial_rows()
        self._populate_changelog_rows()
        if self.source == "raw-jira-export":
            self._validate_export_snapshots()

    def _populate_initial_rows(self) -> None:
        self._connection.execute(
            """
            WITH changelog_items AS (
                SELECT json_extract(issue.payload, '$.key') AS entity,
                       CASE replace(lower(json_extract(item.value, '$.field')), ' ', '')
                           WHEN 'fixversion' THEN 'fixVersion'
                           WHEN 'fixversions' THEN 'fixVersion'
                           ELSE json_extract(item.value, '$.field')
                       END AS field,
                       json_extract(issue.payload, '$.fields.created') AS created,
                       json_extract(history.value, '$.created') AS changed_at,
                       CAST(history.key AS INTEGER) AS history_order,
                       CAST(item.key AS INTEGER) AS item_order,
                       json_extract(item.value, '$.fromString') AS display_value,
                       json_extract(item.value, '$.from') AS raw_value
                FROM raw_issues AS issue
                JOIN json_each(json_extract(issue.payload, '$.changelog.histories')) AS history
                JOIN json_each(json_extract(history.value, '$.items')) AS item
            ), earliest AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY lower(entity), lower(field)
                    ORDER BY julianday(changed_at), history_order, item_order
                ) AS state_order
                FROM changelog_items
                WHERE field IN ('status', 'assignee', 'priority', 'components', 'fixVersion')
            )
            INSERT INTO state_rows
            SELECT entity, field, created, -1,
                   jira_change_value(field, display_value, raw_value)
            FROM earliest
            WHERE state_order = 1
            """
        )
        scalar_fields = {
            "status": "$.fields.status.name",
            "priority": "$.fields.priority.name",
        }
        for field_name, json_path in scalar_fields.items():
            self._connection.execute(
                """
                INSERT INTO state_rows
                SELECT json_extract(payload, '$.key'), ?,
                       json_extract(payload, '$.fields.created'), -1,
                       json_quote(json_extract(payload, ?))
                FROM raw_issues
                WHERE json_type(payload, ?) IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM state_rows
                      WHERE lower(state_rows.entity) = lower(json_extract(raw_issues.payload, '$.key'))
                        AND lower(state_rows.field) = lower(?)
                  )
                """,
                (field_name, json_path, json_path, field_name),
            )
        self._connection.execute(
            """
            INSERT INTO state_rows
            SELECT json_extract(payload, '$.key'), 'assignee',
                   json_extract(payload, '$.fields.created'), -1,
                   json_quote(raw_identity(
                       json_extract(payload, '$.fields.assignee.emailAddress'),
                       COALESCE(
                           json_extract(payload, '$.fields.assignee.accountId'),
                           json_extract(payload, '$.fields.assignee.key'),
                           json_extract(payload, '$.fields.assignee.name')
                       ),
                       json_extract(payload, '$.fields.assignee.displayName')
                   ))
            FROM raw_issues
            WHERE json_type(payload, '$.fields.assignee') = 'object'
              AND NOT EXISTS (
                  SELECT 1 FROM state_rows
                  WHERE lower(state_rows.entity) = lower(json_extract(raw_issues.payload, '$.key'))
                    AND lower(state_rows.field) = 'assignee'
              )
            """
        )
        array_fields = {
            "components": "$.fields.components",
            "fixVersion": "$.fields.fixVersions",
        }
        for field_name, json_path in array_fields.items():
            self._connection.execute(
                """
                INSERT INTO state_rows
                SELECT json_extract(issue.payload, '$.key'), ?,
                       json_extract(issue.payload, '$.fields.created'), -1,
                       (SELECT json_group_array(json_extract(item.value, '$.name'))
                          FROM json_each(json_extract(issue.payload, ?)) AS item)
                FROM raw_issues AS issue
                WHERE json_type(issue.payload, ?) = 'array'
                  AND NOT EXISTS (
                      SELECT 1 FROM state_rows
                      WHERE lower(state_rows.entity) = lower(json_extract(issue.payload, '$.key'))
                        AND lower(state_rows.field) = lower(?)
                  )
                """,
                (field_name, json_path, json_path, field_name),
            )

    def _populate_changelog_rows(self) -> None:
        self._connection.execute(
            """
            INSERT INTO state_rows
            SELECT json_extract(issue.payload, '$.key'),
                   CASE replace(lower(json_extract(item.value, '$.field')), ' ', '')
                       WHEN 'fixversion' THEN 'fixVersion'
                       WHEN 'fixversions' THEN 'fixVersion'
                       ELSE json_extract(item.value, '$.field')
                   END,
                   json_extract(history.value, '$.created'),
                   CAST(history.key AS INTEGER) * 10000 + CAST(item.key AS INTEGER),
                   jira_change_value(
                       CASE replace(lower(json_extract(item.value, '$.field')), ' ', '')
                           WHEN 'fixversion' THEN 'fixVersion'
                           WHEN 'fixversions' THEN 'fixVersion'
                           ELSE json_extract(item.value, '$.field')
                       END,
                       json_extract(item.value, '$.toString'),
                       json_extract(item.value, '$.to')
                   )
            FROM raw_issues AS issue
            JOIN json_each(json_extract(issue.payload, '$.changelog.histories')) AS history
            JOIN json_each(json_extract(history.value, '$.items')) AS item
            """
        )

    def _validate_export_snapshots(self) -> None:
        for issue in self._issues:
            entity = issue.get("key")
            fields = issue.get("fields")
            changelog = issue.get("changelog")
            histories = changelog.get("histories", []) if isinstance(changelog, Mapping) else []
            if not isinstance(entity, str) or not isinstance(fields, Mapping):
                continue
            changed_fields = {
                _canonical_field(str(item.get("field")))
                for history in histories
                if isinstance(history, Mapping)
                for item in history.get("items", [])
                if isinstance(item, Mapping) and item.get("field") is not None
            }
            for field_name in changed_fields & {
                "status",
                "assignee",
                "priority",
                "components",
                "fixVersion",
            }:
                row = self._connection.execute(
                    """
                    SELECT value_json FROM state_rows
                    WHERE lower(entity) = lower(?) AND lower(field) = lower(?)
                    ORDER BY julianday(event_time) DESC, event_order DESC
                    LIMIT 1
                    """,
                    (entity, field_name),
                ).fetchone()
                actual = json.loads(row[0]) if row is not None and row[0] is not None else None
                expected = _raw_snapshot_value(fields, field_name)
                if actual != expected:
                    raise ValueError(
                        f"Jira issue {entity} search snapshot disagrees with changelog "
                        f"final state: {field_name}"
                    )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> SqlGold:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def field_value(self, entity: str, field: str, at: datetime) -> Any:
        row = self._connection.execute(
            """
            SELECT value_json
            FROM state_rows
            WHERE lower(entity) = lower(?)
              AND lower(field) = lower(?)
              AND julianday(event_time) <= julianday(?)
            ORDER BY julianday(event_time) DESC, event_order DESC
            LIMIT 1
            """,
            (_display_entity(entity), field, at.isoformat()),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return json.loads(row[0])


def _event_transitions(event: EvalEvent, event_order: int) -> list[FieldTransition]:
    data = event.data or {}
    transitions: list[FieldTransition] = []

    world_state = data.get("world_state")
    if isinstance(world_state, Mapping):
        entities = world_state.get("entities", world_state)
        if isinstance(entities, Mapping):
            for entity, fields in entities.items():
                if not isinstance(fields, Mapping):
                    continue
                for field_name, value in fields.items():
                    transitions.append(
                        _transition(
                            event,
                            event_order,
                            str(entity),
                            str(field_name),
                            None,
                            value,
                            "world-state",
                        )
                    )

    event_type = event.type.casefold()
    is_jira = "jira" in event.source.casefold() or "jira" in event_type
    entity = canonical_entity(event)
    if is_jira and "jira.issue.created" in event_type:
        initial_fields = {
            "status": data.get("status"),
            "assignee": data.get("assignee"),
            "priority": data.get("priority"),
            "components": data.get("components"),
            "fixVersion": data.get("fix_versions", data.get("fixVersion")),
        }
        for field_name, value in initial_fields.items():
            if value is not None:
                transitions.append(
                    _transition(event, event_order, entity, field_name, None, value, "jira")
                )

    if is_jira:
        changelog = data.get("changelog")
        raw_items = changelog.get("items", []) if isinstance(changelog, Mapping) else []
        items = raw_items if isinstance(raw_items, list) else []
        if not items and "field" in data and "to" in data:
            items = [data]
        for item in items:
            if not isinstance(item, Mapping):
                continue
            field_name = item.get("field")
            if not isinstance(field_name, str) or "to" not in item:
                continue
            transitions.append(
                _transition(
                    event,
                    event_order,
                    entity,
                    _canonical_field(field_name),
                    _normalise_event_change_value(
                        _canonical_field(field_name), item.get("from")
                    ),
                    _normalise_event_change_value(
                        _canonical_field(field_name), item.get("to")
                    ),
                    "jira",
                )
            )

    if "pull_request" in event_type:
        if event_type.endswith(".merged") or data.get("merged") or data.get("merged_at"):
            state = "merged"
        elif event_type.endswith(".closed") or str(data.get("state", "")).casefold() == "closed":
            state = "closed"
        else:
            state = str(data.get("state") or "open")
        transitions.append(
            _transition(event, event_order, event.subject, "state", None, state, "github")
        )

    if "mail" in event_type:
        subject = str(data.get("subject", ""))
        body = str(data.get("body", ""))
        for kip in unique(re.findall(r"\bKIP-\d+\b", subject, re.IGNORECASE)):
            outcome = _vote_outcome(data, subject, body)
            if outcome is not None:
                transitions.append(
                    _transition(
                        event,
                        event_order,
                        kip.upper(),
                        "vote_outcome",
                        None,
                        outcome,
                        "mail",
                    )
                )
        reply_to = data.get("in_reply_to")
        answered_by = data.get("author") or data.get("from")
        value = answered_by if reply_to and answered_by else "UNANSWERED"
        transitions.append(
            _transition(
                event,
                event_order,
                event.subject,
                "answered_by",
                None,
                value,
                "mail",
            )
        )
    return transitions


def _transition(
    event: EvalEvent,
    event_order: int,
    entity: str,
    field_name: str,
    old_value: Any,
    new_value: Any,
    source_kind: str,
) -> FieldTransition:
    return FieldTransition(
        entity=entity,
        field=field_name,
        old_value=old_value,
        new_value=new_value,
        time=event.time,
        event_id=event.id,
        event_order=event_order,
        source_kind=source_kind,
    )


def _vote_outcome(data: Mapping[str, Any], subject: str, body: str) -> str | None:
    explicit = data.get("vote_outcome") or data.get("vote_result")
    if explicit is not None:
        return string_value(explicit)
    text = f"{subject}\n{body}".casefold()
    if "[vote]" in text:
        return "open"
    if "[result]" not in text and "vote" not in text:
        return None
    for outcome in ("accepted", "passed", "rejected", "failed", "cancelled"):
        if outcome in text:
            return outcome
    return None


def _canonical_field(field_name: str) -> str:
    compact = field_name.casefold().replace(" ", "")
    return "fixVersion" if compact in {"fixversion", "fixversions"} else field_name


def _normalise_event_change_value(field_name: str, value: Any) -> Any:
    if field_name not in {"components", "fixVersion"}:
        return value
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [str(value)]


def _raw_identity(email: str | None, account: str | None, display: str | None) -> str | None:
    if account:
        digest = hashlib.sha256(str(account).encode()).hexdigest()[:12]
        return f"jira-user:{digest}"
    if email and "@" in email:
        digest = hashlib.sha256(email.strip().casefold().encode()).hexdigest()[:12]
        return f"contributor:{digest}"
    return display


def _sql_changelog_value(field: str, display: Any, raw: Any) -> str:
    if field == "assignee":
        value = _raw_identity(
            None,
            str(raw) if raw is not None else None,
            str(display) if display is not None else None,
        )
    elif field in {"components", "fixVersion"}:
        source = display if display is not None else raw
        if isinstance(source, str) and source.startswith("["):
            try:
                decoded = json.loads(source)
            except json.JSONDecodeError:
                decoded = source
            if isinstance(decoded, list):
                source = decoded
        if source is None or source == "":
            value = []
        elif isinstance(source, list):
            value = [str(item) for item in source]
        else:
            value = [part.strip() for part in str(source).split(",") if part.strip()]
    else:
        value = display if display is not None else raw
    return json.dumps(value, sort_keys=True)


def _raw_snapshot_value(fields: Mapping[str, Any], field_name: str) -> Any:
    if field_name in {"status", "priority"}:
        raw = fields.get(field_name)
        if isinstance(raw, Mapping):
            return raw.get("name") or raw.get("value")
        return raw
    if field_name == "assignee":
        raw = fields.get("assignee")
        if not isinstance(raw, Mapping):
            return None
        account = raw.get("accountId") or raw.get("key") or raw.get("name")
        email = raw.get("emailAddress")
        return _raw_identity(
            str(email) if email is not None else None,
            str(account) if account is not None else None,
            None,
        )
    raw_values = fields.get("fixVersions" if field_name == "fixVersion" else field_name)
    if not isinstance(raw_values, list):
        return []
    return [
        str(value.get("name") or value.get("value"))
        for value in raw_values
        if isinstance(value, Mapping) and (value.get("name") or value.get("value")) is not None
    ]


def _canonical_key(entity: str) -> str:
    if entity.casefold().startswith("issue:"):
        return entity.split(":", 1)[1].casefold()
    return entity.casefold()


def _display_entity(entity: str) -> str:
    return entity.split(":", 1)[1] if entity.casefold().startswith("issue:") else entity


def _as_event_sequence(value: object) -> Sequence[EvalEvent] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if value and not isinstance(value[0], EvalEvent):
        return None
    return cast(Sequence[EvalEvent], value)


def _load_raw_issues(raw_jira: RawJiraInput) -> list[Mapping[str, Any]]:
    payload: Any = raw_jira
    if isinstance(raw_jira, (str, Path)):
        payload = json.loads(Path(raw_jira).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        raw_issues = payload.get("issues", [payload])
    else:
        raw_issues = payload
    if not isinstance(raw_issues, Sequence) or isinstance(raw_issues, (str, bytes)):
        raise ValueError("raw Jira export must contain an issues array")
    issues = [item for item in raw_issues if isinstance(item, Mapping)]
    if len(issues) != len(raw_issues):
        raise ValueError("every raw Jira issue must be an object")
    return issues


def _smoke_raw_jira_from_events(events: Sequence[EvalEvent]) -> list[Mapping[str, Any]]:
    """Build a raw-shaped fixture adapter; never accepted for evidentiary generation."""

    grouped: dict[str, dict[str, Any]] = {}
    for event in events:
        if "jira" not in event.source.casefold() and "jira" not in event.type.casefold():
            continue
        entity = canonical_entity(event)
        issue = grouped.setdefault(
            entity,
            {
                "key": entity,
                "fields": {"created": event.time.isoformat()},
                "changelog": {"histories": []},
            },
        )
        data = event.data or {}
        if "jira.issue.created" in event.type.casefold():
            fields = issue["fields"]
            for name in ("status", "priority"):
                if data.get(name) is not None:
                    fields[name] = {"name": data[name]}
            if data.get("assignee") is not None:
                fields["assignee"] = {"displayName": data["assignee"]}
            fields["components"] = [{"name": value} for value in data.get("components", [])]
            fields["fixVersions"] = [
                {"name": value} for value in data.get("fix_versions", [])
            ]
        changelog = data.get("changelog")
        items = changelog.get("items", []) if isinstance(changelog, Mapping) else []
        if not items and "field" in data and "to" in data:
            items = [data]
        if not isinstance(items, list) or not items:
            continue
        raw_items = []
        for item in items:
            if not isinstance(item, Mapping) or "field" not in item or "to" not in item:
                continue
            raw_items.append(
                {
                    "field": item["field"],
                    "fromString": item.get("from"),
                    "toString": item.get("to"),
                }
            )
        if raw_items:
            issue["changelog"]["histories"].append(
                {"created": event.time.isoformat(), "items": raw_items}
            )
    return list(grouped.values())


def field_value_python(
    events: Sequence[EvalEvent], entity: str, field: str, at: datetime
) -> Any:
    return PythonGold(events).field_value(entity, field, at)


def field_value_sql(
    raw_jira: RawJiraInput | Sequence[EvalEvent],
    entity: str,
    field: str,
    at: datetime,
) -> Any:
    with SqlGold(raw_jira) as gold:
        return gold.field_value(entity, field, at)


python_field_value = field_value_python
sql_field_value = field_value_sql


def cross_check_gold(
    events: Sequence[EvalEvent],
    requests: Iterable[GoldRequest],
    *,
    raw_jira: RawJiraInput | None = None,
) -> list[GoldDisagreement]:
    python = PythonGold(events)
    disagreements: list[GoldDisagreement] = []
    with SqlGold(raw_jira if raw_jira is not None else events) as sql:
        for request in requests:
            history = python.transitions(request.entity, request.field, request.T)
            if not history or any(item.source_kind != "jira" for item in history):
                continue
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
    events: Sequence[EvalEvent],
    entity: str,
    field: str,
    at: datetime,
    *,
    raw_jira: RawJiraInput | None = None,
) -> GoldDisagreement | None:
    disagreements = cross_check_gold(
        events,
        [GoldRequest(entity=entity, field=field, T=at)],
        raw_jira=raw_jira,
    )
    return disagreements[0] if disagreements else None


def write_gold_report(audit: GoldAuditTrail, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit.report(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
