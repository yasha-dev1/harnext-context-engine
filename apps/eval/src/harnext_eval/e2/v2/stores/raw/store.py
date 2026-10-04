"""Arm `RAW`: every event's raw data in ClickHouse, one table per source, plus a file catalog.

    python -m harnext_eval.e2.v2.stores.build --arm RAW --scope smoke|w2026 [--until YYYY-MM]

The stream is consumed once, in order, into a working chdb database (`work/`, MergeTree tables
`raw.<source>` ordered by (time, event_id)); unseen fields become columns (`ALTER TABLE … ADD
COLUMN`), conflicting types widen a column (the table is then rebuilt from its `raw` column, so
old and new values are encoded alike). When the stream crosses a monthly boundary the build
exports the checkpoint:

    export/<YYYY-MM>/tables/<source>.parquet   rows with time < boundary, zstd, sorted
    export/<YYYY-MM>/schema.json               column types per table (as of the boundary)
    export/<YYYY-MM>/schema-log.jsonl          every schema change so far, with its event
    export/<YYYY-MM>/catalog/                  SOURCES.md, sources.json, <source>.md, catalog.json
    export/<YYYY-MM>.manifest.json             sha256 per file, rows, bytes (pinned by runs)

An export is a separate set of files that never saw a later event and is opened read-only
(`raw.query.open`). Resume: the working database is rebuilt from the last export (it holds
every row and the exact schema) and the stream is skipped up to that boundary.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Iterable
from pathlib import Path

from harnext_eval.e2.v2 import stream
from harnext_eval.e2.v2.stores.base import lock, next_month, save_json, unlock
from harnext_eval.e2.v2.stores.raw import catalog, sources
from harnext_eval.e2.v2.stores.raw.backend import ChdbBackend, quote
from harnext_eval.e2.v2.stores.raw.schema import ORDER, Schema, columns_sql

ARM = "RAW"
VERSION = 3  # bump when extraction or typing changes: a store built otherwise is refused
DB = "raw"
BATCH = 2000


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ch_time(iso: str) -> str:
    return iso.replace("T", " ").rstrip("Z")


def create(db, table: str, columns) -> None:
    db.command(f"CREATE TABLE {DB}.`{table}` ({columns_sql(columns)}) ENGINE = MergeTree ORDER BY ({', '.join(ORDER)})")  # fmt: skip


class RawStore:
    """StoreBuilder for `RAW` (stores/build.py dispatch). Built by code: usage is zero."""

    arm = ARM

    def __init__(self, batch: int = BATCH):
        self.batch = batch
        self.kill_after: str | None = None  # tests: raise after exporting this checkpoint

    def config(self) -> dict:
        return {"arm": ARM, "version": VERSION, "engine": "chdb", "order_by": list(ORDER)}

    # ---- state -----------------------------------------------------------------------------
    def load(self, out: Path) -> dict:
        path = out / "build-state.json"
        state = json.loads(path.read_text()) if path.exists() else None
        if state and state["config"] != self.config():
            raise ValueError(f"{ARM}: store was built with {state['config']}, now {self.config()}; rebuild from scratch")  # fmt: skip
        return state or {"config": self.config(), "checkpoints": [], "events": 0, "last": None,
                         "schema": {}, "schema_log": [], "names": {}, "exports": {}, "done": False,
                         "usage": {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0}}  # fmt: skip

    def restore(self, out: Path, state: dict):
        """A fresh working database: empty, or the last export loaded back."""
        work = out / "work"
        shutil.rmtree(work, ignore_errors=True)
        db = ChdbBackend(work)
        db.command(f"CREATE DATABASE {DB}")
        if state["checkpoints"]:
            source = out / "export" / state["checkpoints"][-1]
            for table, columns in json.loads((source / "schema.json").read_text())["tables"].items():  # fmt: skip
                cols = dict(columns)
                create(db, table, cols)
                path = (source / "tables" / f"{table}.parquet").resolve()
                db.command(f"INSERT INTO {DB}.`{table}` SELECT * FROM file({quote(path)}, Parquet, {quote(columns_sql(cols))})")  # fmt: skip
        return db

    # ---- build -----------------------------------------------------------------------------
    def build(self, events: Iterable[dict], out: Path, *, until: str | None = None) -> None:
        """Resumable; the state file only ever moves at a checkpoint (see `finish`)."""
        out.mkdir(parents=True, exist_ok=True)
        state = self.load(out)
        self.db, self.out, self.state, self.started = self.restore(out, state), out, state, time.monotonic()  # fmt: skip
        self.schema = Schema(state["schema"], state["schema_log"], state["names"])
        done = state["checkpoints"]
        resume = stream.boundary(done[-1]) if done else ""
        current = next_month(done[-1]) if done else None
        pending: list[tuple[str, dict]] = []
        previous = tuple(state["last"] or ("", ""))
        try:
            for event in events:
                key = (event["time"], event["id"])
                if event["time"] < resume:
                    continue
                if key <= previous:
                    raise ValueError(f"stream is not strictly sorted by (time, id) at {event['id']}")  # fmt: skip
                current = current or event["time"][:7]
                while event["time"] >= stream.boundary(current) and (not until or current <= until):
                    self.flush(pending)
                    self.finish(current)
                    current = next_month(current)
                if until and current > until:
                    break
                table, row = sources.table_of(event), sources.row(event)
                self.widen(table, self.schema.observe(table, row, event))
                pending.append((table, row))
                state["events"] += 1
                state["last"], previous = list(key), key
                if len(pending) >= self.batch:
                    self.flush(pending)
            else:
                state["done"] = until is None
            self.flush(pending)
            last = until or (state["last"][0][:7] if state["last"] else None)
            # the month still open, then the empty months up to `until`
            while current and last and current <= last:
                if current not in done:
                    self.finish(current)
                current = next_month(current)
            save_json(out / "build-state.json", state)
        finally:
            self.db.close()
            shutil.rmtree(out / "work", ignore_errors=True)  # a cache: `restore` rebuilds it

    def widen(self, table: str, changes: list[dict]) -> None:
        """Apply the DDL of the schema changes one event caused. A widening rebuilds the rows
        already stored; pending rows are encoded with the new types when they are flushed."""
        widened = False
        for change in changes:
            if change["change"] == "create":
                create(self.db, table, self.schema.tables[table])
            elif change["change"] == "add":
                self.db.command(f"ALTER TABLE {DB}.`{table}` ADD COLUMN IF NOT EXISTS `{change['column']}` {change['type']}")  # fmt: skip
            else:
                widened = True
        if widened:
            self.rebuild(table)

    def rebuild(self, table: str) -> None:
        """Re-encode every stored row of `table` from its raw record under the current types."""
        columns = self.schema.tables[table]
        self.db.command(f"DROP TABLE IF EXISTS {DB}.`{table}__new`")
        create(self.db, f"{table}__new", columns)
        rows = self.db.select(f"SELECT raw FROM {DB}.`{table}` ORDER BY {', '.join(ORDER)}")["rows"]  # fmt: skip
        lines = [self.schema.encode(table, sources.row(json.loads(r["raw"]))) for r in rows]
        self.db.insert(f"{DB}.`{table}__new`", list(columns), lines)
        self.db.command(f"EXCHANGE TABLES {DB}.`{table}` AND {DB}.`{table}__new`")
        self.db.command(f"DROP TABLE {DB}.`{table}__new`")

    def flush(self, pending: list) -> None:
        by_table: dict[str, list[str]] = {}
        for table, row in pending:
            by_table.setdefault(table, []).append(self.schema.encode(table, row))
        for table, lines in sorted(by_table.items()):
            self.db.insert(f"{DB}.`{table}`", list(self.schema.tables[table]), lines)
        pending.clear()

    # ---- checkpoints -----------------------------------------------------------------------
    def finish(self, checkpoint: str) -> None:
        """Export the working database as of `checkpoint` and move the cursor past it."""
        state, boundary = self.state, stream.boundary(checkpoint)
        root = self.out / "export"
        target, temporary = root / checkpoint, root / f".{checkpoint}.tmp"
        for stale in (temporary, target):
            if stale.exists():
                unlock(stale)
                shutil.rmtree(stale)
        (temporary / "tables").mkdir(parents=True)
        tables, rows = self.schema.tables, {}
        for table in sorted(tables):
            path = (temporary / "tables" / f"{table}.parquet").resolve()
            self.db.command(
                f"INSERT INTO FUNCTION file({quote(path)}, 'Parquet') SELECT * FROM {DB}.`{table}` "
                f"WHERE time < '{ch_time(boundary)}' ORDER BY {', '.join(ORDER)} "
                "SETTINGS output_format_parquet_compression_method = 'zstd', max_threads = 1, "
                "output_format_parquet_row_group_size = 100000"
            )  # fmt: skip
            rows[table] = self.db.select(f"SELECT count() AS n FROM {DB}.`{table}` WHERE time < '{ch_time(boundary)}'")["rows"][0]["n"]  # fmt: skip
        log = [e for e in self.schema.log if e["time"] < boundary]
        head = {"checkpoint": checkpoint, "boundary": boundary, "events": state["events"]}
        save_json(temporary / "schema.json", {**head, "order_by": list(ORDER), "rows": rows, "tables": self.schema.state(), "names": self.schema.names})  # fmt: skip
        (temporary / "schema-log.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in log))  # fmt: skip
        found = catalog.build(self.db, DB, tables, checkpoint, boundary, state["events"], log, self.schema.names)  # fmt: skip
        catalog.write(temporary / "catalog", found)
        files = {str(p.relative_to(temporary)): sha_file(p) for p in sorted(temporary.rglob("*")) if p.is_file()}  # fmt: skip
        manifest = {**head, "arm": ARM, "build": self.config(), "rows": rows, "files": files,
                    "content_sha256": hashlib.sha256(json.dumps(sorted(files.items())).encode()).hexdigest(),
                    "bytes": sum((temporary / f).stat().st_size for f in files)}  # fmt: skip
        lock(temporary)
        temporary.rename(target)
        save_json(root / f"{checkpoint}.manifest.json", manifest)
        state["checkpoints"].append(checkpoint)
        state["exports"][checkpoint] = {"bytes": manifest["bytes"], "rows": rows, "content_sha256": manifest["content_sha256"]}  # fmt: skip
        state["schema"], state["schema_log"], state["names"] = self.schema.state(), self.schema.log, self.schema.names  # fmt: skip
        state["seconds"] = round(state.get("seconds", 0) + time.monotonic() - self.started, 1)
        self.started = time.monotonic()
        save_json(self.out / "build-state.json", state)
        print(f"{ARM}: {checkpoint} events={state['events']} rows={sum(rows.values())} bytes={manifest['bytes']}", flush=True)  # fmt: skip
        if self.kill_after == checkpoint:
            raise RuntimeError(f"killed after {checkpoint}")

    def export(self, out: Path, checkpoint: str) -> Path:
        target = out / "export" / checkpoint
        if not (target / "schema.json").exists():
            raise ValueError(f"{ARM}: no export for {checkpoint}; the build writes one per checkpoint")  # fmt: skip
        return target
