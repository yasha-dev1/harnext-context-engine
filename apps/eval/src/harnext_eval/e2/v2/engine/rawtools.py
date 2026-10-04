"""Tools of a raw-source branch.

find(need, values)  Jev fills a plan (translate.py), the compiler writes the SQL (plan.py). No
                    rows: exact text equality is relaxed to case-insensitive containment, then
                    the values are probed in every table so the agent learns where they occur.
                    More rows than fit: Jev scores rows against the need (lists and searches;
                    timelines and counts are never filtered).
read(event_id)      the full record, paged.
sql(query)          fallback, per `sql_fallback`: auto = offered after a find failed or a slot
                    was uncertain, always, never. Its schema is shown only when it is offered.
Every call is logged for the trace (plan, SQL, row counts, fallback use).
"""

from __future__ import annotations

import asyncio
import collections
import json
import re

from harnext_eval.e2.v2.engine import prompts
from harnext_eval.e2.v2.engine.plan import (
    ROUNDINGS,
    UNITS,
    Filter,
    Plan,
    PlanError,
    compile_plan,
    count_plan,
    descending,
    inner,
    is_array,
    is_text,
    lit,
    relax,
    times_sql,
)
from harnext_eval.e2.v2.engine.session import BudgetExceeded, Session
from harnext_eval.e2.v2.engine.translate import clean, is_date, translate
from harnext_eval.e2.v2.stores.raw import guard

NOT_LOCATED = {
    "text",
    "title",
    "entities",
    "raw",
    "event_id",
}  # base columns: too generic to look up in
ISO_Z = re.compile(r"'(\d{4}-\d\d-\d\d)T(\d\d:\d\d(?::\d\d(?:\.\d{1,9})?)?)Z'")


def function(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}}  # fmt: skip


FIND = function(
    "find",
    "Find records. `need`: one full sentence saying which records you need and what you want "
    "to know from them. `values`: the literal data they must contain (keys, numbers, logins, "
    "names, exact titles, dates and instants as written in the question, states); data only, "
    "never column names. `page` (optional): show the matching rows strictly in order, 25 per "
    "page, instead of the most relevant ones.",
    {
        "need": {"type": "string"},
        "values": {"type": "array", "items": {"type": "string"}},
        "page": {"type": "integer", "minimum": 1},
    },  # fmt: skip
    ["need", "values"],
)
READ = function(
    "read",
    "The complete record of one event by its event id; long records come in pages.",
    {"event_id": {"type": "string"}, "page": {"type": "integer", "minimum": 1}},
    ["event_id"],
)


COMPARE_TIMES = function(
    "compare_times",
    "Order two events by their recorded instants and compute the time between them from the "
    "records. Choose the unit and the rounding from the wording of the question.",
    {
        "first_event_id": {"type": "string"},
        "second_event_id": {"type": "string"},
        "unit": {"type": "string", "enum": list(UNITS)},
        "rounding": {"type": "string", "enum": list(ROUNDINGS)},
    },  # fmt: skip
    ["first_event_id", "second_event_id", "unit", "rounding"],
)


UNIT_WORDS = {"ms": "milliseconds", "sec": "seconds", "s": "seconds", "min": "minutes", "h": "hours", "d": "days", "w": "weeks"}  # fmt: skip
ROUNDING_WORDS = {"floor": "down", "truncate": "down", "whole": "down", "round": "nearest", "ceil": "up", "ceiling": "up", "none": "exact", "no": "exact"}  # fmt: skip


def spelled(value, allowed, words: dict) -> str:
    """A unit or rounding as the tool names it ('Day' -> 'days', 'floor' -> 'down')."""
    text = str(value or "").strip().lower()
    for candidate in (text, text + "s", words.get(text, ""), words.get(text.rstrip("s"), "")):
        if candidate in allowed:
            return candidate
    return text


def describe_times(ids: list[str], row: dict, unit: str, rounding: str) -> str:
    a, b = iso(row["a_time"]), iso(row["b_time"])
    order = "at the same instant" if row["same_instant"] else f"{ids[0]} is earlier" if row["a_first"] else f"{ids[1]} is earlier"  # fmt: skip
    return (f"{ids[0]} at {a}; {ids[1]} at {b}; {order}; difference {row['amount']} {unit} "
            f"(rounding: {rounding}); exact difference {abs(int(row['milliseconds_a_to_b']))} milliseconds")  # fmt: skip


READ_THREAD = function(
    "read_thread",
    "All records of the thread or item an event belongs to (e.g. every message of a discussion "
    "thread, every comment on an item), in time order: the ones relevant to `need` in full, the "
    "others as one line each with their ids.",
    {"event_id": {"type": "string"}, "need": {"type": "string"}},
    ["event_id", "need"],
)


def as_int(value, default: int = 1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def iso(value) -> str:
    """'2026-03-01 12:00:00.123000' -> '2026-03-01T12:00:00.123000Z' (every time is UTC)."""
    return str(value).replace(" ", "T", 1) + "Z"


def show(value, kind: str) -> str:
    if value is None:
        return "null"
    if inner(kind).startswith("DateTime"):
        return json.dumps([iso(v) for v in value]) if is_array(kind) else iso(value)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return " ".join(str(value).split())


def lines(rows: list[dict], columns: list) -> tuple[str, list[str]]:
    """(values shared by every row, one line per row without them)."""
    types = dict(tuple(c) for c in columns)
    shared = {}
    if len(rows) > 2:
        shared = {k: v for k, v in rows[0].items() if k != "event_id" and v not in (None, "", []) and all(r.get(k) == v for r in rows)}  # fmt: skip
    common = "; ".join(f"{k}: {show(v, types.get(k, ''))}" for k, v in shared.items())
    found = []
    for row in rows:
        parts = [f"{k}: {show(v, types.get(k, ''))}" for k, v in row.items() if k not in shared and v not in (None, "", [])]  # fmt: skip
        found.append("- " + " | ".join(parts))
    return (f"every row has {common}" if common else ""), found


def facets(rows: list[dict], columns: list[str], most: int = 3, top: int = 8) -> str:
    """Value counts over ALL fetched rows for short, repeated columns (the rows shown may be a
    relevance-filtered subset)."""
    found = []
    for column in columns:
        values = [r.get(column) for r in rows if isinstance(r.get(column), str | int) and len(str(r.get(column))) <= 120]  # fmt: skip
        counts = collections.Counter(values)
        if len(values) == len(rows) and 1 < len(counts) <= 30 and len(counts) < len(rows):
            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[:top]
            found.append(f"{column}: " + ", ".join(f"{v} ({n})" for v, n in ranked))
        if len(found) == most:
            break
    return f"; counts over all {len(rows)} fetched rows: " + "; ".join(found) if found else ""


def fit(rendered: list[str], budget: int) -> list[int]:
    """Indices of the leading lines that fit `budget` characters (at least one)."""
    kept, used = [], 0
    for i, line in enumerate(rendered):
        if used + len(line) > budget and kept:
            break
        kept.append(i)
        used += len(line) + 1
    return kept


def schema_text(session: Session, tables: list[str]) -> str:
    """Tables and columns of the branch plus the SQL the guard allows (shown with `sql` only)."""
    cat = session.catalog
    columns = "\n".join(f"{t}({', '.join(f'{c} {k}' for c, k in cat.types[t].items())})" for t in tables)  # fmt: skip
    return columns + "\n\n" + session.db.reference()


def rejection(query: str, tables) -> str | None:
    """The guard's reason when it refuses a statement, else None (engine errors are not)."""
    try:
        guard.check(query, tables)
    except guard.QueryError as error:
        return str(error)
    return None


def explain(plan) -> str:
    ops = {"eq": "=", "ne": "!=", "contains": "contains", "before": "before", "through": "through", "from": "from", "after": "after", "on": "on", "null": "is empty", "notnull": "is set"}  # fmt: skip
    filters = "; ".join(f"{f.table}.{'|'.join((f.column, *f.also))} {ops[f.op]}" + (f" {f.value!r}" if f.value else "") for f in plan.filters) or "none"  # fmt: skip
    filters += "".join(f"; {c.left} {'=' if c.op == 'eq' else '!='} {plan.link.table + '.' if c.linked else ''}{c.right}" for c in plan.compares)  # fmt: skip
    extra = f"; {'values of' if plan.shape == 'values' else 'grouped by'} {plan.group}" if plan.group else ""  # fmt: skip
    extra += f"; counting distinct {plan.distinct}" if plan.distinct else ""
    extra += (
        f"; via {plan.link.table} ({plan.link.target} = {plan.link.column})" if plan.link else ""
    )
    return f"[{plan.shape} over {plan.table}; filters: {filters}{extra}]"


class RawTools:
    def __init__(self, session: Session, scope: list[str]):
        self.session, self.scope, self.config = session, scope, session.config
        self.unlocked = self.config.sql_fallback == "always"
        self.announce = False
        self.restated = False
        self.seen: set[str] = set()
        self.log: list[dict] = []
        self.asked: set[tuple] = set()
        self.records: dict[str, str] = {}  # event id -> compact rendering of what was retrieved

    # ---- tool table ----------------------------------------------------------------------
    def specs(self) -> list[dict]:
        tools = [FIND, READ, READ_THREAD, COMPARE_TIMES]
        if self.unlocked:
            text = prompts.SQL_TOOL
            if self.config.sql_fallback == "always":
                text += "\n" + schema_text(self.session, self.scope)
            tools.append(function("sql", text, {"query": {"type": "string"}}, ["query"]))
        return tools

    def handlers(self) -> dict:
        found = {"find": self.find, "read": self.read, "read_thread": self.read_thread, "compare_times": self.compare_times}  # fmt: skip
        if self.unlocked:
            found["sql"] = self.sql
        return found

    def failed(self) -> None:
        if self.config.sql_fallback == "auto" and not self.unlocked:
            self.unlocked = self.announce = True

    def tail(self) -> str:
        if not self.announce:
            return ""
        self.announce = False
        return "\n\n" + prompts.SQL_UNLOCKED.format(schema=schema_text(self.session, self.scope))

    def remember(self, rows: list[dict], common: str, rendered: list[str]) -> None:
        """The compact rendering of every row SHOWN to the model, by event id (support checks
        and the report's grounding read it)."""
        for row, line in zip(rows, rendered, strict=True):
            if isinstance(row.get("event_id"), str):
                self.seen.add(row["event_id"])
                self.records[row["event_id"]] = f"{line} | {common}" if common else line

    async def compare_times(self, first_event_id: str = "", second_event_id: str = "", unit: str = "seconds", rounding: str = "exact") -> str:  # fmt: skip
        """The order of two events and the time between them, computed in SQL."""
        unit, rounding = (
            spelled(unit, UNITS, UNIT_WORDS),
            spelled(rounding, ROUNDINGS, ROUNDING_WORDS),
        )
        record = {"tool": "compare_times", "events": [first_event_id, second_event_id], "unit": unit, "rounding": rounding}  # fmt: skip
        if unit not in UNITS or rounding not in ROUNDINGS:
            record["error"] = "bad unit or rounding"
            return f"error: unit must be one of {', '.join(UNITS)}; rounding one of {', '.join(ROUNDINGS)}"  # fmt: skip
        self.log.append(record)
        ids = [str(first_event_id).strip(), str(second_event_id).strip()]
        places = await self.session.db.sources(ids)
        if missing := [i for i in ids if i not in places]:
            record["error"] = "unknown id"
            return f"error: no event with id {', '.join(map(repr, missing))} (use event ids exactly as shown in rows)"  # fmt: skip
        sql = times_sql(self.session.catalog, (places[ids[0]], ids[0]), (places[ids[1]], ids[1]), unit, rounding)  # fmt: skip
        result = await self.session.db.sql(sql, max_rows=1)
        record |= {"sql": sql, "error": result["error"]}
        if result["error"] or not result["rows"]:
            return f"error: {result['error'] or 'no times found'}"
        text = describe_times(ids, result["rows"][0], unit, rounding)
        self.seen |= set(ids)
        for i in ids:
            self.records[i] = self.records.get(i, "") + f" | {text}"
        record["result"] = text
        return text

    # ---- find ----------------------------------------------------------------------------
    async def find(self, need: str = "", values=None, page=None) -> str:
        values = self.unprefix(clean(values, str(need)))
        page = as_int(page, 0) or None
        key = (" ".join(str(need).split()).lower(), tuple(values), page)
        if key in self.asked:
            self.log.append({"tool": "find", "need": need, "values": values, "outcome": "repeat"})
            return "You made this exact find call before; its result above is unchanged. Use it, restate the need differently, or finish with done."  # fmt: skip
        self.asked.add(key)
        return await self.plan_and_run(str(need), values, {"tool": "find", "need": need, "values": values, "page": page, "key": key})  # fmt: skip

    def unprefix(self, values: list[str]) -> list[str]:
        """'issue_key: X' -> 'X' when the prefix is a column name; 'merged_at>=2024-01-01' ->
        '2024-01-01' for any word prefix of a date (the comparison is Jev's to choose)."""
        names = {c for t in self.session.catalog.types.values() for c in t}
        found = []
        for value in values:
            parts = re.split(r"\s*(?:>=|<=|==|!=|[:=<>])\s*", value, maxsplit=1)
            named = len(parts) == 2 and parts[1] and re.fullmatch(r"\w+", parts[0])
            if named and (parts[0] in names or is_date(parts[1])):
                value = parts[1]
            found.append(value)
        return list(dict.fromkeys(found))

    async def plan_and_run(self, need: str, values: list[str], record: dict) -> str:
        self.log.append(record)
        if not need.strip():
            record["error"] = "empty need"
            return "error: say in one sentence which records you need (need), and pass identifying values"  # fmt: skip
        try:
            plan, info = await translate(self.session, need, values, self.scope)
        except BudgetExceeded as error:
            record["error"] = str(error)
            return f"error: {error}; finish with done"
        if record.get("page"):
            plan.limit, plan.offset = self.config.show_rows, (record["page"] - 1) * self.config.show_rows  # fmt: skip
        record |= {"plan": plan.to_dict(), "slots": info}
        blocking = [
            u for u in info["uncertain"] if u["slot"] != "shape"
        ]  # the rows barely depend on the shape
        if blocking and self.config.restate and not self.restated:
            self.restated = True
            record["outcome"] = "restate"
            self.asked.discard(record["key"])  # the same call may be made again: it will run
            self.failed()
            slots = "; ".join(f"{u['slot']}: {u['choice']} (alternatives {u['alternatives']})" for u in blocking)  # fmt: skip
            return f"The need is ambiguous ({slots}). Restate it once, more specifically: name the kind of record and what you want from it (or repeat the call to run it as it is).{self.tail()}"  # fmt: skip
        if any(u["slot"] == "shape" for u in info["uncertain"]):
            shape = next(u for u in info["uncertain"] if u["slot"] == "shape")
            record["values_note"] = f"; the kind of result was uncertain ({shape['alternatives']}): if this is not what you need, say so in the need"  # fmt: skip
        try:
            sql = self.compile(plan)
        except PlanError as error:
            record |= {"outcome": "invalid", "error": str(error)}
            self.failed()
            return f"error: the need could not be turned into a query: {error}. Restate the need or the values.{self.tail()}"  # fmt: skip
        result = await self.session.db.sql(sql, max_rows=self.config.fetch_rows)
        record |= {"sql": sql, "rows": len(result["rows"]), "error": result["error"]}
        if result["error"]:
            record["outcome"] = "error"
            record["rejected"] = rejection(sql, self.session.catalog.types)
            self.failed()
            return f"error: the query failed: {result['error']}{self.tail()}"
        note = ""
        if not result["rows"] and (loose := relax(plan, self.session.catalog)):
            again = await self.session.db.sql(self.compile(loose), max_rows=self.config.fetch_rows)
            if again["rows"] and not again["error"]:
                plan, result = loose, again
                note = " (no exact match; text values matched case-insensitively as substrings)"
                record |= {"relaxed": True, "sql": self.compile(loose), "rows": len(result["rows"])}
        if not result["rows"]:
            return await self.empty(plan, values, record)
        record["outcome"] = "rows"
        used = [f.value for f in plan.filters if f.value]
        record["values_note"] = record.get("values_note", "") + f"; values used as filters: {used}" + (f"; values NOT used: {info['unused_values']}" if info.get("unused_values") else "")  # fmt: skip
        return await self.show_rows(need, plan, result, note, record)

    def compile(self, plan) -> str:
        return compile_plan(plan, self.session.catalog, self.config.fetch_rows, self.config.cell_chars)  # fmt: skip

    async def empty(self, plan, values: list[str], record: dict) -> str:
        """No rows: say where the values occur; when a value is found in a descriptive column,
        look it up there right away (e.g. a title that only a change record still carries)."""
        where, found = await self.probe(values)
        record["probe"] = where
        cat = self.session.catalog
        tried = {(f.table, f.column) for f in plan.filters}
        spots = [(v, t, c) for v, t, c, _ in found if c not in NOT_LOCATED and (t, c) not in tried]
        if spots:
            value, table, column = spots[0]
            keys = [k for k in cat.tables[table].get("key_columns") or [] if k != column]
            spot = Plan(table, "lookup", [Filter(table, column, "contains", value)], [column, *keys], limit=10)  # fmt: skip
            located = await self.session.db.sql(self.compile(spot), max_rows=10)
            if located["rows"] and not located["error"]:
                record |= {"outcome": "located", "located_sql": self.compile(spot)}
                self.seen |= {r["event_id"] for r in located["rows"] if r.get("event_id")}
                common, rendered = lines(located["rows"], located["columns"])
                self.remember(located["rows"], common, rendered)
                head = f"0 rows for {explain(plan)}. {where} Looked up {value!r} in {table}.{column} ({common or 'first rows'}):"  # fmt: skip
                return head + "\n" + "\n".join(rendered[:10]) + "\nUse these identifiers in your next find."  # fmt: skip
        record["outcome"] = "empty"
        self.failed()
        return f"0 rows for {explain(plan)}. {where} Restate the need (another kind of record, other or fewer values).{self.tail()}"  # fmt: skip

    async def show_rows(self, need: str, plan, result: dict, note: str, record: dict) -> str:
        """The rows, with a truthful header: how many match in total, how many are shown and
        which, in which order, and how to get the others."""
        fetched = result["rows"]
        rowwise = plan.shape not in ("count", "values", "latest", "earliest")
        total = len(fetched) + plan.offset
        if rowwise and (len(fetched) >= plan.limit or plan.offset):
            counted = await self.session.db.sql(compile_plan(count_plan(plan), self.session.catalog, self.config.fetch_rows))  # fmt: skip
            total = counted["rows"][0]["n"] if counted["rows"] else total
        record["total"] = total
        order = f"ordered by {plan.order or 'time'}, {'newest / highest first' if descending(plan) else 'oldest / lowest first'}"  # fmt: skip
        if plan.shape == "count":
            head = f"{len(fetched)} groups{note} for {explain(plan)}; n = rows per group" if plan.group else f"count{note} for {explain(plan)}"  # fmt: skip
        elif plan.shape == "values":
            head = f"all distinct values{note} for {explain(plan)} (computed over every matching row; rows = how many matched)"  # fmt: skip
        elif plan.shape in ("latest", "earliest"):
            head = f"the {plan.shape} of {total} matching records{note}, {order}, for {explain(plan)}"  # fmt: skip
        else:
            head = f"{total} matching rows{note} for {explain(plan)}; {order}"
        rows, chosen = fetched, ""
        if plan.shape in ("lookup", "list", "search") and not record.get("page") and len(rows) > self.config.show_rows:  # fmt: skip
            rows, scored = await self.relevant(need, rows)
            record["relevance"] = scored
            chosen = f"; a relevance model chose these {len(rows)} from the first {scored['scored']} rows in this order"  # fmt: skip
            head += facets(fetched, plan.columns)
        common, rendered = lines(rows, result["columns"])
        kept = fit(rendered, self.config.show_chars)
        shown = [rows[i] for i in kept]
        self.seen |= {r["event_id"] for r in shown if r.get("event_id")}
        self.remember(shown, common, [rendered[i] for i in kept])
        if rowwise:
            first = plan.offset + 1
            where = f"rows {first}-{plan.offset + len(shown)} in this order" if not chosen else "selected rows"  # fmt: skip
            head += f"; showing {len(shown)} ({where}){chosen}"
            if len(shown) < total:
                pages = -(-total // self.config.show_rows)
                head += f"; to see the others in order call find again with page=1..{pages} ({self.config.show_rows} per page), or narrow the need"  # fmt: skip
        head += record.pop("values_note", "") + (f"; {common}" if common else "")
        record["shown"] = [r.get("event_id") for r in shown]
        return head + ":\n" + "\n".join(rendered[i] for i in kept) + self.tail()

    async def relevant(self, need: str, rows: list[dict]) -> tuple[list[dict], dict]:
        config = self.config
        candidates = rows[: config.relevance_rows]
        compact = [" | ".join(f"{k}: {' '.join(str(v).split())[:120]}" for k, v in r.items() if v not in (None, "", []))[:300] for r in candidates]  # fmt: skip
        scores = await self.score(need, compact)
        ranked = sorted(scores, key=lambda i: (-scores[i], i))
        keep = [i for i in ranked if scores[i] >= config.relevance_min][: config.show_rows] or ranked[:3]  # fmt: skip
        return [candidates[i] for i in sorted(keep)], {"scored": len(scores), "kept": len(keep)}

    async def probe(self, values: list[str]) -> tuple[str, list[tuple]]:
        """Where each value occurs (text, [(value, table, column, rows)]), over every table (one
        short statement per table and value)."""
        cat, found = self.session.catalog, []
        for table in sorted(cat.types):
            columns = [c for c, k in cat.types[table].items() if is_text(k) and c not in ("raw", "event_id")]  # fmt: skip
            for value in values if columns else []:
                parts = []
                for j, c in enumerate(columns):
                    test = f"arrayExists(x -> positionCaseInsensitiveUTF8(x, {lit(value)}) > 0, `{c}`)" if is_array(cat.types[table][c]) else f"positionCaseInsensitiveUTF8(`{c}`, {lit(value)}) > 0"  # fmt: skip
                    parts.append(f"countIf({test}) AS `c{j}`")
                result = await self.session.db.sql(f"SELECT {', '.join(parts)} FROM `{table}`", max_rows=1)  # fmt: skip
                for key, n in (result["rows"][0] if result["rows"] else {}).items():
                    if n:
                        found.append((value, table, columns[int(key[1:])], n))
        if not found:
            return ("None of the values occurs anywhere in the records (check spelling or use fewer words)." if values else ""), []  # fmt: skip
        found.sort(key=lambda f: (f[2] in NOT_LOCATED, -f[3]))
        return "The values occur in: " + "; ".join(f"{v!r} in {t}.{c} ({n} rows)" for v, t, c, n in found[:10]) + ".", found  # fmt: skip

    # ---- read_thread -------------------------------------------------------------------------
    def thread_column(self, table: str) -> str | None:
        """The column grouping a table's records into threads or items: the first key column
        that is a join column in the catalog's links (scalar), else the first key column."""
        cat = self.session.catalog
        keys = [k for k in cat.tables[table].get("key_columns") or [] if k in cat.types[table] and not is_array(cat.types[table][k])]  # fmt: skip
        joined = {e[c]["column"] for e in cat.doc.get("links") or [] for c in ("from", "to") if e[c]["table"] == table}  # fmt: skip
        return next((k for k in keys if k in joined), keys[0] if keys else None)

    async def read_thread(self, event_id: str = "", need: str = "") -> str:
        record = {"tool": "read_thread", "event_id": event_id, "need": need}
        self.log.append(record)
        event_id = str(event_id).strip()
        table = (await self.session.db.sources([event_id])).get(event_id)
        column = self.thread_column(table) if table else None
        if not column:
            record["error"] = "unknown id" if not table else "no thread column"
            return f"error: no thread for {event_id!r} (use an event id exactly as shown in rows)"
        types = self.session.catalog.types[table]
        body = next((c for c in ("body", "text") if c in types), "text")
        head = [c for c in ("actor",) if c in types]
        key = await self.session.db.sql(f"SELECT `{column}` AS `k` FROM `{table}` WHERE `event_id` = {lit(event_id)}", max_rows=1)  # fmt: skip
        if not key["rows"] or key["rows"][0]["k"] is None:
            record["error"] = "empty thread key"
            return f"error: {event_id} has no {column}"
        value = key["rows"][0]["k"]
        test = f"`{column}` = {lit(value) if isinstance(value, str) else int(value)}"
        select = ", ".join(
            ["`event_id`", "`time`", *(f"`{c}`" for c in head), f"`{body}` AS `body`"]
        )
        result = await self.session.db.sql(f"SELECT {select} FROM `{table}` WHERE {test} ORDER BY `time`, `event_id` LIMIT {self.config.fetch_rows}", max_rows=self.config.fetch_rows, max_bytes=2_000_000)  # fmt: skip
        rows = result["rows"]
        record |= {"table": table, "column": column, "rows": len(rows), "error": result["error"]}
        if result["error"] or not rows:
            return f"error: {result['error'] or 'no records'}"
        relevant = set(range(len(rows)))
        if len(rows) > 3:
            scores = await self.score(str(need or "the question"), [f"{r.get('actor', '')}: {' '.join(str(r.get('body') or '').split())[:280]}" for r in rows])  # fmt: skip
            relevant = {i for i, p in scores.items() if p >= self.config.relevance_min}
            record["relevant"] = len(relevant)
        budget, used, lines_ = self.config.read_chars * 2, 0, []
        each = max(400, budget // max(1, len(relevant)))
        for i, row in enumerate(rows):
            stamp = f"- event_id: {row['event_id']} | time: {iso(row['time'])}" + "".join(f" | {c}: {row.get(c)}" for c in head)  # fmt: skip
            text = " ".join(str(row.get("body") or "").split())
            if i in relevant and used < budget:
                line = f"{stamp}\n  {text[:each]}{'…' if len(text) > each else ''}"
                used += len(line)
            else:
                line = f"{stamp} | first words: {text[:100]}{'…' if len(text) > 100 else ''}"
            lines_.append(line)
            self.seen.add(row["event_id"])
            self.records[row["event_id"]] = line
        more = " (more exist: fetch limit reached)" if len(rows) >= self.config.fetch_rows else ""
        return (f"{len(rows)} records of {table} with {column} = {value!r}{more}, oldest first; "
                f"{len(relevant)} relevant to the need shown in full (up to {each} characters), the others as one line:\n" + "\n".join(lines_))  # fmt: skip

    async def score(self, need: str, texts: list[str]) -> dict[int, float]:
        """Jev relevance of each text to the need (batched)."""
        size = self.config.relevance_batch

        async def batch(start: int) -> dict[int, float]:
            part = range(start, min(start + size, len(texts)))
            state = {"need": need, "rows": {f"r{i}": texts[i] for i in part}}
            questions = {f"r{i}": {"type": "noul", "instructions": prompts.ROW.format(name=f"r{i}")} for i in part}  # fmt: skip
            answers = await self.session.decide(state, questions, "rows")
            return {i: float(answers[f"r{i}"].get("noul", 0)) for i in part}

        scores: dict[int, float] = {}
        try:
            for part in await asyncio.gather(*(batch(s) for s in range(0, len(texts), size))):
                scores |= part
        except BudgetExceeded:  # no decisions left: keep the leading rows, in order
            scores = {i: 1.0 if i < self.config.show_rows else 0.0 for i in range(len(texts))}
        return scores

    # ---- read / sql -----------------------------------------------------------------------
    async def read(self, event_id: str = "", page: int = 1) -> str:
        record = {"tool": "read", "event_id": event_id, "page": page}
        self.log.append(record)
        found = await self.session.db.get(str(event_id).strip())
        if not found:
            record["error"] = "unknown id"
            return f"error: no event with id {event_id!r} (use an event_id exactly as shown in rows)"  # fmt: skip
        self.seen.add(str(event_id).strip())
        body = found.pop("text", "") or ""
        whole = json.dumps({k: v for k, v in found.items() if k != "raw"}, ensure_ascii=False) + "\n\n" + body  # fmt: skip
        size = self.config.read_chars
        pages = max(1, -(-len(whole) // size))
        page = min(max(1, as_int(page)), pages)
        record["pages"] = pages
        more = f" (call read with page={page + 1} for more)" if page < pages else ""
        chunk = whole[(page - 1) * size : page * size]
        key = str(event_id).strip()
        self.records[key] = (self.records.get(key, "") + "\n" + chunk)[-self.config.read_chars :]
        return f"page {page}/{pages}{more}:\n" + chunk

    async def sql(self, query: str = "") -> str:
        record = {"tool": "sql", "query": query}
        self.log.append(record)
        query = ISO_Z.sub(r"'\1 \2'", str(query))  # the engine rejects 'YYYY-MM-DDThh:mm:ssZ'
        result = await self.session.db.sql(query, max_rows=50, max_bytes=4 * self.config.show_chars)  # fmt: skip
        record |= {"rows": len(result["rows"]), "error": result["error"]}
        if result["error"]:
            record["rejected"] = rejection(query, self.session.catalog.types)
            return f"error: {result['error']}"  # the guard's message unchanged: it names allowed functions
        common, rendered = lines(result["rows"], result["columns"])
        kept = fit(rendered, self.config.show_chars)
        shown = [result["rows"][i] for i in kept]
        self.remember(shown, common, [rendered[i] for i in kept])
        cut = len(rendered) - len(kept)
        rendered = [rendered[i] for i in kept]
        more = (f"; {cut} rows not shown (too long)" if cut else "") + (
            f"; {common}" if common else ""
        )
        truncated = "; more rows exist (add filters or LIMIT)" if result["truncated"] else ""
        return f"{len(result['rows'])} rows for the query {' '.join(query.split())}{more}{truncated}:\n" + "\n".join(rendered)  # fmt: skip
