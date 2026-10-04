"""Dynamic mapping: per-table column types inferred from the values, evolved in stream order.

A value's type: bool -> Bool, int -> Int64, float -> Float64, an ISO-8601 UTC instant ->
DateTime64(6, 'UTC'), other strings -> String, a list -> Array(<element type>), a dict or a
list holding lists/dicts -> String (JSON). Scalars are Nullable; arrays are not (empty = []).
null and [] carry no type. An unseen field adds a column; a conflicting type widens the
column to Nullable(String) (arrays with differing elements to Array(String)). Every change is
logged with the event that caused it. Types depend only on the values in stream order, so a
rebuild from the same stream yields the same schema.
"""

from __future__ import annotations

import hashlib
import json
import re

DATETIME = "DateTime64(6, 'UTC')"
STRING = "Nullable(String)"
BASE = [
    ("event_id", "String"), ("event_type", "LowCardinality(String)"), ("time", DATETIME),
    ("actor", "Nullable(String)"), ("title", "String"), ("text", "String"),
    ("entities", "Array(String)"), ("raw", "String"),
]  # fmt: skip
ORDER = ("time", "event_id")
ISO = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,9})?Z")


INT64 = (-(2**63), 2**63 - 1)


def snake(name: str) -> str:
    """ASCII snake_case of a field name; a name without ASCII letters or digits gets a stable
    hash name instead of an empty one."""
    found = re.sub(r"[^0-9a-z]+", "_", re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()).strip("_")  # fmt: skip
    return found or f"field_{hashlib.sha256(name.encode()).hexdigest()[:6]}"


def columns_sql(columns: dict) -> str:
    return ", ".join(f"`{c}` {k}" for c, k in columns.items())


def scalar(value) -> str:
    if isinstance(value, bool):
        return "Bool"
    if isinstance(value, int):  # beyond Int64 the exact digits are kept as a string
        return "Int64" if INT64[0] <= value <= INT64[1] else "String"
    if isinstance(value, float):
        return "Float64"
    return DATETIME if ISO.fullmatch(value) else "String"


def infer(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return STRING
    if isinstance(value, list):
        items = [v for v in value if v is not None]
        if not items:
            return None
        if any(isinstance(v, list | dict) for v in items):
            return STRING
        kinds = {scalar(v) for v in items}
        return f"Array({kinds.pop() if len(kinds) == 1 else 'String'})"
    return f"Nullable({scalar(value)})"


NUMERIC = {"Int64", "Float64"}


def merge(old: str, new: str | None) -> str:
    if new is None or new == old or old == STRING:
        return old
    inner = lambda kind: kind.removeprefix("Nullable(").removeprefix("Array(").rstrip(")")  # noqa: E731
    if {inner(old), inner(new)} == NUMERIC and old.startswith("Array(") == new.startswith("Array("):
        return old.replace("Int64", "Float64")  # int -> float widens numerically
    if old.startswith("Array(") and new.startswith("Array("):
        return "Array(String)"
    return STRING


def text(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list | dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def encode(value, kind: str):
    """A Python value as JSONEachRow input for a column of type `kind`."""
    if kind.startswith("Array("):
        items = value if isinstance(value, list) else [] if value is None else [value]
        return [text(v) for v in items] if kind == "Array(String)" else items
    if value is None:
        return None
    return text(value) if kind in (STRING, "String") else value


class Schema:
    """{table: {column: type}} in creation order, the change log, and the names given to fields
    that came without a fixed column name ({table: {column: origin}}, origin "meta:<key>" or
    "fact:<line name>"). A generic field is named snake(key); if that name is taken (a base
    column, another field) it becomes "<meta|fact>_<name>", then "…_2", "…_3" — first come,
    first named, so the names follow the stream order and never merge two fields."""

    def __init__(self, tables: dict | None = None, log: list | None = None, names: dict | None = None):  # fmt: skip
        self.tables = {t: dict(cols) for t, cols in (tables or {}).items()}
        self.log = list(log or [])
        self.names = {t: dict(n) for t, n in (names or {}).items()}

    def column(self, table: str, key: str) -> str:
        """The column of a row key: itself, or the registered name of a generic origin."""
        if ":" not in key:
            return key
        named = self.names.setdefault(table, {})
        found = {origin: column for column, origin in named.items()}
        if key in found:
            return found[key]
        kind, field = key.split(":", 1)
        taken = set(self.tables.get(table, {})) | set(named) | {c for c, _ in BASE}
        name, count = snake(field), 1
        if name in taken:
            name = f"{kind}_{name}"
            while (candidate := name if count == 1 else f"{name}_{count}") in taken:
                count += 1
            name = candidate
        named[name] = key
        return name

    def resolve(self, table: str, row: dict) -> dict:
        return {self.column(table, k): v for k, v in row.items()}

    def observe(self, table: str, row: dict, event: dict) -> list[dict]:
        found = []
        where = {"event_id": event["id"], "time": event["time"]}
        if table not in self.tables:
            self.tables[table] = dict(BASE)
            found.append({"table": table, "change": "create", **where})
        columns = self.tables[table]
        for key, value in row.items():
            kind, name = infer(value), self.column(table, key)
            if kind is None or name in dict(BASE):
                continue
            origin = {"origin": key} if ":" in key else {}
            old = columns.get(name)
            if old is None:
                columns[name] = kind
                found.append({"table": table, "column": name, "change": "add", "type": kind, **origin, **where})  # fmt: skip
            elif (wider := merge(old, kind)) != old:
                columns[name] = wider
                found.append({"table": table, "column": name, "change": "widen", "type": wider, "from_type": old, "value_type": kind, **where})  # fmt: skip
        self.log += found
        return found

    def state(self) -> dict:
        return {t: [[c, k] for c, k in cols.items()] for t, cols in self.tables.items()}

    def encode(self, table: str, row: dict) -> str:
        columns, row = self.tables[table], self.resolve(table, row)
        return json.dumps({c: encode(row.get(c), k) for c, k in columns.items() if c in row}, ensure_ascii=False)  # fmt: skip
