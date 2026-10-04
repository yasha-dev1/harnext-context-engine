"""Read side of the raw store, for the query-time engine:

    db = open(export_dir)            # an export of RAW, may be mounted read-only
    db.catalog()                     # sources.json: one summary per source
    db.schema("jira_changes")        # every column with meaning and examples, joins, worked SQL
    db.sql("SELECT …", max_rows=200, max_bytes=200_000, timeout_s=10)
    db.get("jira:1:change:2:0")      # the full raw record of one event
    db.close()                       # or use `with open(…) as db:`

The export's parquet files are loaded into a private scratch chdb database (the export itself
is never written; chdb cannot open a read-only directory). Two layers guard `sql`:
`guard.check` admits one SELECT / WITH over the export's tables (or DESCRIBE <table> / SHOW
TABLES) and nothing that reaches outside them, and the statement then runs, unchanged, on a
fresh connection whose URL carries readonly = 1 and the limits (time, rows, result bytes,
memory, joins) — settings a statement cannot change. Errors come back as short messages a model
can act on. The same object runs over a ClickHouse server (`over`).
"""

from __future__ import annotations

import atexit
import fcntl
import json
import os
import re
import shutil
import signal
import tempfile
import time
from pathlib import Path

from harnext_eval.e2.v2.stores.raw.backend import Backend, ChdbBackend, quote
from harnext_eval.e2.v2.stores.raw.guard import QueryError, check, summary
from harnext_eval.e2.v2.stores.raw.schema import columns_sql

KIND = re.compile(r"\(([A-Z][A-Z_]+)\)")
COLUMN_ERRORS = ("UNKNOWN_IDENTIFIER", "NO_SUCH_COLUMN_IN_TABLE", "THERE_IS_NO_COLUMN", "UNKNOWN_EXPRESSION_IDENTIFIER")  # fmt: skip
TIME_ERRORS = ("TYPE_MISMATCH", "CANNOT_PARSE_DATETIME", "CANNOT_PARSE_TEXT", "CANNOT_CONVERT_TYPE",
               "ILLEGAL_TYPE_OF_ARGUMENT", "NO_COMMON_TYPE")  # fmt: skip
TIME_HINT = (". Time columns are DateTime64(6, 'UTC'): compare with '2026-03-01 00:00:00' or "
             "'2026-03-01T00:00:00Z'; inside IN (…), dateDiff(…) and other functions convert first: "
             "parseDateTime64BestEffort('2026-03-01T00:00:00Z')")  # fmt: skip
PREFIX = "harnext-raw-"
MEMORY = 1_000_000_000
__all__ = ["QueryError", "RawDB", "harden", "open", "over"]
# chdb starts one embedded server per open() and leaks ~15 /proc handles when it stops (measured:
# fds 83 -> 218 over 10 open/close cycles; queries do not grow them): a process should open one
# export and keep it.


def harden(memory_bytes: int = 8 << 30, cpu_seconds: int | None = None, file_bytes: int = 4 << 30) -> dict:  # fmt: skip
    """Optional OS caps for a process that serves `sql` (the MCP server calls it at start; never
    applied on import): data segment, largest file written (scratch loads), no core dumps and,
    if given, total CPU seconds — soft and hard limits both (CPU: hard 10 s above soft, so
    SIGXCPU comes first). Also removes this process's scratch dirs on SIGTERM. Returns the limits
    now in force. RLIMIT_AS is not used: chdb reserves far more virtual memory than it touches."""
    import resource

    caps = {resource.RLIMIT_DATA: (memory_bytes, memory_bytes), resource.RLIMIT_FSIZE: (file_bytes, file_bytes), resource.RLIMIT_CORE: (0, 0)}  # fmt: skip
    if cpu_seconds is not None:
        caps[resource.RLIMIT_CPU] = (cpu_seconds, cpu_seconds + 10)
    for kind, (soft, hard) in caps.items():
        current = resource.getrlimit(kind)[1]
        if current != resource.RLIM_INFINITY:
            soft, hard = min(soft, current), min(hard, current)
        resource.setrlimit(kind, (soft, hard))
    previous = signal.getsignal(signal.SIGTERM)

    def terminated(number, frame):
        remove_all()
        if callable(previous):
            previous(number, frame)
        raise SystemExit(128 + number)

    signal.signal(signal.SIGTERM, terminated)
    return {kind: resource.getrlimit(kind) for kind in caps}


def limits(max_rows: int, max_bytes: int, timeout_s: float) -> dict:
    """Engine settings of one untrusted statement (readonly = 1 freezes all of them)."""
    return {"readonly": 1, "max_execution_time": timeout_s, "timeout_overflow_mode": "throw",
            "limit": max_rows + 1, "max_result_rows": max_rows + 1,
            "max_result_bytes": 2 * max_bytes + 1_000_000, "result_overflow_mode": "break",
            "max_memory_usage": MEMORY, "max_bytes_in_join": MEMORY // 4,
            "join_overflow_mode": "throw", "max_threads": 4,
            "cast_string_to_date_time_mode": "best_effort", "session_timezone": "UTC"}  # fmt: skip


def message(error: Exception) -> tuple[str, str | None]:
    """(short text, ClickHouse error name) of an engine error."""
    text = " ".join(str(error).split())
    kinds = KIND.findall(text)
    detail = text.split("DB::Exception: ", 1)[-1]
    detail = re.split(r"(?i) in scope | \(version | Stack trace|\. \([A-Z_]+\)$", detail)[0]
    detail = detail.removeprefix("default: ").rstrip(". ")[:300]
    if kinds and kinds[-1] == "READONLY":
        detail += " (the tool is read-only: writes, SETTINGS and table functions are disabled)"
    return (f"{kinds[-1]}: {detail}" if kinds else detail), kinds[-1] if kinds else None


class RawDB:
    def __init__(self, backend: Backend, catalog: dict, *, scratch: Path | None = None):
        self.backend, self.catalog_doc, self.scratch = backend, catalog, scratch
        self.tables = {t: [c["name"] for c in d["columns"]] for t, d in catalog["tables"].items()}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        self.backend.close()
        if self.scratch:
            remove(self.scratch)
            self.scratch = None

    # ---- catalog -----------------------------------------------------------------------------
    def catalog(self) -> dict:
        return {k: self.catalog_doc[k] for k in ("checkpoint", "boundary", "events", "dialect", "people", "sources", "links") if k in self.catalog_doc}  # fmt: skip

    def sql_reference(self) -> str:
        """The statements and functions `sql` accepts, from the guard in force (for the model,
        only when it falls back to the `sql` tool)."""
        return summary()

    def schema(self, source: str) -> dict:
        if source not in self.catalog_doc["tables"]:
            raise KeyError(f"unknown source {source!r}; sources: {', '.join(sorted(self.tables))}")
        return self.catalog_doc["tables"][source]

    # ---- queries -----------------------------------------------------------------------------
    def hint(self, query: str, kind: str | None) -> str:
        if kind in COLUMN_ERRORS:
            named = [t for t in self.tables if re.search(rf"\b{t}\b", query)] or sorted(self.tables)
            return ". Valid columns: " + "; ".join(f"{t}({', '.join(self.tables[t])})" for t in named)  # fmt: skip
        if kind == "UNKNOWN_TABLE":
            return f". Tables: {', '.join(sorted(self.tables))}."
        if kind in TIME_ERRORS:
            return TIME_HINT
        return ""

    def sql(self, query: str, max_rows: int = 200, max_bytes: int = 200_000, timeout_s: float = 10.0) -> dict:  # fmt: skip
        started = time.monotonic()
        result = {"columns": [], "rows": [], "truncated": False, "elapsed_ms": 0, "error": None}
        try:
            statement = check(query, self.tables)  # exactly this text runs
            found = self.backend.select(statement, limits(max_rows, max_bytes, timeout_s))
        except QueryError as error:
            result["error"] = str(error)
        except Exception as error:  # noqa: BLE001 - the engine's message goes back to the model
            text, kind = message(error)
            result["error"] = text + self.hint(query, kind)
        else:
            rows, used = [], 0
            for row in found["rows"][:max_rows]:
                size = len(json.dumps(row, ensure_ascii=False, default=str))
                if used + size > max_bytes:
                    result["note"] = f"stopped at {max_bytes} bytes; select fewer or shorter columns (e.g. substring(body, 1, 500)) or add LIMIT"  # fmt: skip
                    break
                rows.append(row)
                used += size
            result |= {"columns": [list(c) for c in found["columns"]], "rows": rows,
                       "truncated": len(rows) < len(found["rows"]) or len(found["rows"]) > max_rows}  # fmt: skip
        result["elapsed_ms"] = round(1000 * (time.monotonic() - started), 1)
        return result

    def get(self, event_id: str) -> dict | None:
        """The complete original record of one event, with the table it lives in."""
        union = " UNION ALL ".join(f"SELECT '{t}' AS source, raw FROM `{t}` WHERE event_id = {{id:String}}" for t in sorted(self.tables))  # fmt: skip
        rows = self.backend.select(union, limits(1, 50_000_000, 30), {"id": event_id})["rows"]
        return {"source": rows[0]["source"], **json.loads(rows[0]["raw"])} if rows else None


OPEN: dict[Path, int] = {}  # scratch dirs of this process -> the fd holding their lock


def remove(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    if (fd := OPEN.pop(path, None)) is not None:
        os.close(fd)


def remove_all() -> None:
    for path in list(OPEN):
        remove(path)


atexit.register(remove_all)


def claim(path: Path) -> int:
    """Hold an exclusive flock on `path/.lock` for the life of the process (or until close). The
    lock is taken on a temporary name and then renamed, so a sweep never sees an unlocked file."""
    fd = os.open(path / ".lock.new", os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.replace(path / ".lock.new", path / ".lock")
    return fd


def sweep(parent: Path) -> None:
    """Remove scratch dirs whose owner is gone: only a dir whose `.lock` can be locked now (the
    kernel drops a flock when its process dies, in any pid namespace sharing the filesystem). A
    dir without a lock file (being created, or from an older version) is left alone."""
    for path in parent.glob(f"{PREFIX}*"):
        lock = path / ".lock"
        if path in OPEN or not lock.exists():
            continue
        try:
            fd = os.open(lock, os.O_RDWR)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            continue
        shutil.rmtree(path, ignore_errors=True)
        os.close(fd)


def open(export: Path, scratch: Path | None = None) -> RawDB:  # noqa: A001 - the API's name
    """Load one RAW export into a private scratch chdb database (one per process at a time)."""
    export = Path(export).resolve()
    schema = json.loads((export / "schema.json").read_text())
    doc = json.loads((export / "catalog" / "catalog.json").read_text())
    if ChdbBackend.current is not None:
        raise RuntimeError(f"chdb is bound to {ChdbBackend.current} in this process; close it first")  # fmt: skip
    parent = Path(scratch or tempfile.gettempdir())
    sweep(parent)
    scratch = Path(tempfile.mkdtemp(prefix=f"{PREFIX}{os.getpid()}-", dir=parent))
    OPEN[scratch] = claim(scratch)
    backend = None
    try:
        backend = ChdbBackend(scratch / "db")
        for table, columns in schema["tables"].items():  # the default database: names unqualified
            cols = dict(columns)
            backend.command(f"CREATE TABLE `{table}` ({columns_sql(cols)}) ENGINE = MergeTree ORDER BY event_id")  # fmt: skip
            path = export / "tables" / f"{table}.parquet"
            backend.command(f"INSERT INTO `{table}` SELECT * FROM file({quote(path)}, Parquet, {quote(columns_sql(cols))})")  # fmt: skip
    except BaseException:
        if backend:
            backend.close()
        remove(scratch)
        raise
    return RawDB(backend, doc, scratch=scratch)


def over(backend: Backend, catalog_dir: Path) -> RawDB:
    """The same API over a live ClickHouse server database (production): the catalog files
    come from the file store, the rows from the server (its default database)."""
    return RawDB(backend, json.loads((Path(catalog_dir) / "catalog.json").read_text()))
