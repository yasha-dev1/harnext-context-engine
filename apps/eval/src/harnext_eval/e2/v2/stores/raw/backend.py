"""The one ClickHouse interface of the raw store: embedded chdb (eval exports, tests) or a
ClickHouse server over HTTP (production). Both speak the same SQL dialect.

    command(sql)                   DDL / INSERT … SELECT (build side)
    insert(table, columns, lines)  JSONEachRow lines into a table
    select(sql, settings, params)  {"columns": [(name, type)], "rows": [dict], "statistics": {}}

`select` with `settings` is the untrusted path: the settings travel beside the statement (a
chdb connection URL, an HTTP parameter), never in its text, so with readonly = 1 among them a
statement cannot change them. Without settings it is the builder's own trusted query.

chdb runs ONE embedded server per process and binds it to one directory; `ChdbBackend`
refuses a second directory while the first is open instead of failing obscurely.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol
from urllib.parse import urlencode


class Backend(Protocol):
    def command(self, sql: str) -> None: ...

    def insert(self, table: str, columns: list[str], lines: list[str]) -> None: ...

    def select(
        self, sql: str, settings: dict | None = None, params: dict | None = None
    ) -> dict: ...

    def close(self) -> None: ...


INSERT_SETTINGS = {"date_time_input_format": "best_effort", "input_format_skip_unknown_fields": 0}
JSON_SETTINGS = {"output_format_json_quote_64bit_integers": 0, "output_format_json_quote_decimals": 0, "session_timezone": "UTC"}  # fmt: skip


def quote(value) -> str:
    """A ClickHouse string literal."""
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def settings_sql(settings: dict) -> str:
    return ", ".join(f"{k} = {json.dumps(v) if not isinstance(v, str) else repr(v)}" for k, v in settings.items())  # fmt: skip


def parse(found: str) -> dict:
    found = json.loads(found) if found else {"meta": [], "data": [], "statistics": {}}
    return {"columns": [(m["name"], m["type"]) for m in found["meta"]], "rows": found["data"],
            "statistics": found.get("statistics", {})}  # fmt: skip


class ChdbBackend:
    current: Path | None = None  # the directory chdb's process-wide server is bound to

    def __init__(self, path: Path):
        from chdb import session

        path = Path(path).resolve()
        if ChdbBackend.current not in (None, path):
            raise RuntimeError(f"chdb is bound to {ChdbBackend.current} in this process; close it first")  # fmt: skip
        path.mkdir(parents=True, exist_ok=True)
        self.path, self.session = path, session.Session(str(path))
        self.session.query("SET session_timezone = 'UTC'")  # never the host's zone
        ChdbBackend.current = path

    def command(self, sql: str) -> None:
        self.session.query(sql)

    def insert(self, table: str, columns: list[str], lines: list[str]) -> None:
        if lines:
            names = ", ".join(f"`{c}`" for c in columns)
            self.session.query(f"INSERT INTO {table} ({names}) SETTINGS {settings_sql(INSERT_SETTINGS)} FORMAT JSONEachRow\n" + "\n".join(lines))  # fmt: skip

    def select(self, sql: str, settings: dict | None = None, params: dict | None = None) -> dict:
        if settings is not None:
            return self.guarded(sql, settings, params)
        try:
            found = self.session.query(f"{sql} SETTINGS {settings_sql(JSON_SETTINGS)}", "JSON", params=params or {}).data()  # fmt: skip
        except Exception:
            # chdb keeps the partial output of a query that failed mid-run (e.g. a timeout) and
            # prepends it to the next result; one throwaway query drains it.
            self.session.query("SELECT 1")
            raise
        return parse(found)

    def guarded(self, sql: str, settings: dict, params: dict | None) -> dict:
        """One fresh connection per statement, its settings in the connection URL (< 1 ms)."""
        import chdb

        connection = chdb.connect(f"{self.path}?{urlencode({**JSON_SETTINGS, **settings})}")
        try:
            return parse(connection.query(sql, "JSON", params=params or {}).data())
        finally:
            connection.close()

    def close(self) -> None:
        if ChdbBackend.current == self.path:
            self.session.close()
            ChdbBackend.current = None


class HttpBackend:
    """A ClickHouse server (clickhouse-connect, HTTP). Not exercised by the eval: no server runs
    on this machine; the eval uses chdb exports. Same contract, values normalised to JSON."""

    def __init__(self, host: str, port: int = 8123, database: str = "default", **options):
        import clickhouse_connect

        self.client = clickhouse_connect.get_client(host=host, port=port, database=database, **options)  # fmt: skip

    def command(self, sql: str) -> None:
        self.client.command(sql)

    def insert(self, table: str, columns: list[str], lines: list[str]) -> None:
        if lines:
            self.client.raw_insert(table, columns, "\n".join(lines).encode(), settings=INSERT_SETTINGS, fmt="JSONEachRow")  # fmt: skip

    def select(self, sql: str, settings: dict | None = None, params: dict | None = None) -> dict:
        found = self.client.query(sql, parameters=params, settings=settings)
        names = list(found.column_names)
        rows = [dict(zip(names, json.loads(json.dumps(row, default=str)), strict=True)) for row in found.result_rows]  # fmt: skip
        return {"columns": list(zip(names, [t.name for t in found.column_types], strict=True)),
                "rows": rows, "statistics": dict(found.summary or {})}  # fmt: skip

    def close(self) -> None:
        self.client.close()
