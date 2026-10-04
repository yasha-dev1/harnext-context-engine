"""Shared fixtures of the engine tests: a synthetic catalog with every column type, and an
in-memory stand-in for RawDB that records the SQL it is asked to run."""

from __future__ import annotations

BASE = [("event_id", "String"), ("event_type", "LowCardinality(String)"),
        ("time", "DateTime64(6, 'UTC')"), ("actor", "Nullable(String)"), ("title", "String"),
        ("text", "String"), ("entities", "Array(String)"), ("raw", "String")]  # fmt: skip


def table(name, columns, *, joins=(), keys=(), examples=None):
    examples = examples or {}
    cols = [{"name": c, "type": k, "description": f"The {c} of a {name} row.",
             "examples": examples.get(c, []), "null_share": 0.0, "distinct": 3}
            for c, k in BASE + columns]  # fmt: skip
    return {"table": name, "system": name.split("_")[0].upper(), "about": f"One row per {name} event.",  # fmt: skip
            "use_for": f"Questions about {name}.", "rows": 3, "key_columns": list(keys),
            "columns": cols, "joins": list(joins), "examples": []}  # fmt: skip


def catalog_doc() -> dict:
    items = table(
        "tick_items",
        [("key", "Nullable(String)"), ("n", "Nullable(Int64)"), ("score", "Nullable(Float64)"),
         ("ok", "Nullable(Bool)"), ("at", "Nullable(DateTime64(6, 'UTC'))"),
         ("tags", "Array(String)"), ("nums", "Array(Int64)"), ("body", "Nullable(String)"),
         ("state", "Nullable(String)")],
        keys=["key", "state"], examples={"key": ["ABC-1", "ABC-2"]},
    )  # fmt: skip
    notes = table(
        "tick_notes",
        [("key", "Nullable(String)"), ("author", "Nullable(String)"), ("body", "Nullable(String)")],
        joins=["key = tick_items.key"], keys=["key", "author"],
    )  # fmt: skip
    mail = table(
        "mail_msgs",
        [("subject", "Nullable(String)"), ("refs", "Array(String)"), ("sender", "Nullable(String)"),
         ("body_refs", "Array(String)")],
        keys=["subject", "sender"], examples={"refs": ["ABC-5", "ABC-9"]},
    )  # fmt: skip
    tables = {t["table"]: t for t in (items, notes, mail)}
    return {"checkpoint": "2030-01", "boundary": "2030-02-01T00:00:00Z", "events": 9,
            "dialect": "ClickHouse SQL", "sources": [], "tables": tables}  # fmt: skip


class FakeDB:
    """RawDB stand-in: `answer(sql)` returns rows (list of dicts) or raises for an error."""

    def __init__(self, answer=None, records=None):
        self.catalog_doc = catalog_doc()
        self.answer = answer or (lambda sql: [])
        self.records = records or {}
        self.queries: list[str] = []
        self.closed = False

    def sql(self, query, max_rows=200, max_bytes=200_000, timeout_s=10.0):
        self.queries.append(query)
        try:
            rows = self.answer(query)
        except Exception as error:  # noqa: BLE001
            return {"columns": [], "rows": [], "truncated": False, "elapsed_ms": 0, "error": str(error)}  # fmt: skip
        columns = [[k, "DateTime64(6, 'UTC')" if k in ("time", "at") else "String"] for k in (rows[0] if rows else {})]  # fmt: skip
        return {"columns": columns, "rows": rows[:max_rows], "truncated": len(rows) > max_rows,
                "elapsed_ms": 0, "error": None}  # fmt: skip

    def get(self, event_id):
        record = self.records.get(event_id)
        return dict(record) if record else None

    def close(self):
        self.closed = True
