"""The file side of the raw store: `catalog/` next to the tables of every export.

    SOURCES.md     one paragraph per source (what, when to use, rows, time range, key columns)
    sources.json   the same summaries, machine-readable (small: fits a decision model's context)
    <table>.md     every column (type, meaning, null share, distinct values, examples), joins and
                   worked SQL
    catalog.json   everything above in one document

Statistics are computed from the table as of the checkpoint; meanings of known columns come
from `sources.COLUMNS`, unknown ones are described from their name, type and examples. Every
number and example is ordered deterministically, so the same data yields the same bytes. The
SQL the read API accepts is not part of an export: `query.RawDB.sql_reference()` renders it from
the guard in force (`guard.summary()`), so an export never lists a function the guard refuses.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from harnext_eval.e2.v2.stores.raw import sources
from harnext_eval.e2.v2.stores.raw.backend import quote
from harnext_eval.e2.v2.stores.raw.schema import BASE

PLACEHOLDER = re.compile(r"\{(\w+)\}")

SHOW = 60  # characters of an example value printed in the catalog (the data is never cut)
KEY = 5  # key columns named in a summary
# Window effects, counted per export from its own rows: (table, SQL returning one row, text).
NOTES = [
    ("mail_messages", "SELECT uniqExact(thread_id) AS threads, uniqExactIf(thread_id, message_id = thread_id) AS started, "
     "countIf(in_reply_to IS NULL AND message_id != thread_id) AS orphans FROM mail_messages",
     "{threads:,} threads, of which {started:,} begin in this export (the message whose message_id equals "
     "thread_id opens its thread; the others began before the export). {orphans:,} messages without "
     "in_reply_to do not open their thread."),
    ("github_pull_requests", "SELECT uniqExact(pr_number) AS prs, uniqExactIf(pr_number, pr_number NOT IN "
     "(SELECT pr_number FROM github_pull_requests WHERE action = 'opened')) AS early FROM github_pull_requests",
     "{prs:,} pull requests; {early:,} of them lack an 'opened' row (opened before the export began)."),
    ("jira_changes", "SELECT uniqExact(issue_key) AS issues, uniqExactIf(issue_key, issue_key NOT IN "
     "(SELECT issue_key FROM jira_issues)) AS early FROM jira_changes",
     "{issues:,} issues changed, of which {early:,} were created before the export began (no jira_issues row; "
     "their title is issue_summary; changes before the export are not here)."),
]  # fmt: skip


def short(value) -> str:
    value = " ".join(str(value).split())
    return value if len(value) <= SHOW else value[: SHOW - 1] + "…"


def empty(name, kind):
    if kind.startswith("Array("):
        return f"empty(`{name}`)"
    if kind.startswith("Nullable("):
        return f"isNull(`{name}`)"
    return f"`{name}` = ''" if "String" in kind else "0"


def examples(db, table, name, kind, where, spread):
    """Enumerations (`spread` false): the 5 most frequent values; any other column: its 3
    earliest values, so examples show formats and never favour busy entities."""
    col = f"`{name}`"
    value = f"arrayJoin({col})" if kind.startswith("Array(") else col
    if spread:
        rows = db.select(f"SELECT v FROM (SELECT {value} AS v, time, event_id FROM {table} WHERE {where} AND NOT ({empty(name, kind)})) GROUP BY v ORDER BY min((time, event_id)), v LIMIT 3")["rows"]  # fmt: skip
    else:
        rows = db.select(f"SELECT {value} AS v, count() AS n FROM {table} WHERE {where} AND NOT ({empty(name, kind)}) GROUP BY v ORDER BY n DESC, v LIMIT 5")["rows"]  # fmt: skip
    return [short(r["v"]) for r in rows]


def describe(name, kind, table, first, found, origin=None):
    known = sources.BASE.get(name) or sources.TABLE_COLUMNS.get(table, {}).get(name) or sources.COLUMNS.get(name)  # fmt: skip
    if known and (origin is None or name == origin.split(":", 1)[1]):
        return known
    source = f"original {origin.split(':', 1)[0]} key '{origin.split(':', 1)[1]}'" if origin else f"field '{name.replace('_', ' ')}'"  # fmt: skip
    seen = f", first seen in event {first['event_id']} ({first['time']})" if first else ""
    return f"Auto-detected: {source} ({kind}{seen}); e.g. {', '.join(found[:3]) or 'no values yet'}."  # fmt: skip


def table_stats(db, database, table, columns, boundary, log, names):
    where = f"time < '{boundary.replace('T', ' ').rstrip('Z')}'"
    name = f"{database}.{table}"
    parts = ", ".join(f"countIf({empty(c, k)}) AS `e:{c}`, {'uniqExactArray' if k.startswith('Array(') else 'uniqExact'}(`{c}`) AS `d:{c}`" for c, k in columns.items())  # fmt: skip
    top = db.select(f"SELECT count() AS rows, toString(min(time)) AS first, toString(max(time)) AS last, {parts} FROM {name} WHERE {where}")["rows"][0]  # fmt: skip
    rows, cols = top["rows"], []
    firsts = {e["column"]: e for e in log if e.get("table") == table and e["change"] == "add"}
    for column, kind in columns.items():
        nonempty = rows - top[f"e:{column}"]
        distinct = top[f"d:{column}"]
        enumeration = column in sources.ENUMERATIONS
        found = examples(db, name, column, kind, where, not enumeration) if nonempty else []
        meaning = describe(column, kind, table, firsts.get(column), found, names.get(column))
        if enumeration:
            meaning += " Values are case-sensitive; the examples are the most frequent, as spelt."
        cols.append({"name": column, "type": kind, "description": meaning,
                     "null_share": round(1 - nonempty / rows, 6) if rows else None,
                     "distinct": distinct, "examples": found})  # fmt: skip
    return {"rows": rows, "first": top["first"] if rows else None, "last": top["last"] if rows else None, "columns": cols}  # fmt: skip


def system(table):
    if table.startswith("other_"):
        return "unrecognised event type"
    return sources.SYSTEMS.get(table.split("_")[0], table.split("_")[0])


def meaningful(value) -> bool:
    return value not in (None, "", 0, [], "0")


def worked(db, template, picks, values) -> str | None:
    """The template filled with values picked from this export, if it returns a non-empty
    first value; None otherwise (a missing column, no suitable data)."""
    values = dict(values)
    fill = lambda sql: PLACEHOLDER.sub(lambda m: values[m.group(1)], sql)  # noqa: E731
    try:
        for name, pick in picks.items():
            rows = db.select(fill(pick))["rows"]
            if not rows or not meaningful(value := next(iter(rows[0].values()))):
                return None
            values[name] = str(value) if isinstance(value, int) else quote(value)
        sql = fill(template)
        rows = db.select(f"SELECT * FROM ({sql}) LIMIT 1")["rows"]
    except Exception:  # noqa: BLE001 - any engine error disqualifies the example
        return None
    return sql if rows and meaningful(next(iter(rows[0].values()))) else None


SHAPES = (  # ever coarser generalisations of a value; the first that fits 90 % of the values wins
    lambda v: re.sub(r"\d+", "<n>", v),
    lambda v: re.sub(r"[A-Za-z0-9]+", "<w>", v),
    lambda v: re.sub(r"[^@/:#]+", "<text>", v),
)


def shape(values: list[str]) -> str:
    """The pattern of a column's values (KAFKA-<n>, <n>, <text>@<text>), inferred from them."""
    for generalise in SHAPES:
        counts: dict[str, int] = {}
        for value in values:
            key = generalise(value)
            counts[key] = counts.get(key, 0) + 1
        best = min(counts.items(), key=lambda kv: (-kv[1], kv[0])) if counts else ("<none>", 0)
        if best[1] >= 0.9 * len(values):
            return best[0]
    return "<text>"


def side(db, schema, table, column) -> dict:
    kind = schema[table][column]
    value = f"arrayJoin(`{column}`)" if kind.startswith("Array(") else f"`{column}`"
    found = db.select(f"SELECT DISTINCT toString({value}) AS v FROM {table} WHERE NOT ({empty(column, kind)}) ORDER BY v LIMIT 2000")["rows"]  # fmt: skip
    return {"table": table, "column": column, "kind": "array" if kind.startswith("Array(") else "scalar",
            "value_shape": shape([r["v"] for r in found])}  # fmt: skip


def keyed(sided) -> str:
    """One row per join value of a side: the column itself, or its array elements."""
    column = f"`{sided['column']}`"
    value = f"arrayJoin({column})" if sided["kind"] == "array" else column
    return f"(SELECT {value} AS k FROM {sided['table']})"


def links(db, schema) -> list[dict]:
    """`sources.LINKS` as data, for the columns this export has: both sides with kind and inferred
    value shape, and a runnable join."""
    found = []
    for source, column, target, other, meaning in sources.LINKS:
        if column not in schema.get(source, {}) or other not in schema.get(target, {}):
            continue
        a, b = side(db, schema, source, column), side(db, schema, target, other)
        found.append({"from": {k: a[k] for k in ("table", "column", "kind")}, "to": {k: b[k] for k in ("table", "column", "kind")},
                      "meaning": meaning, "value_shape": a["value_shape"] if a["value_shape"] == b["value_shape"] else f"{a['value_shape']} | {b['value_shape']}",
                      "sql": f"SELECT count() AS pairs FROM {keyed(a)} AS a INNER JOIN {keyed(b)} AS b ON a.k = b.k"})  # fmt: skip
    return found


def notes(db, table) -> list[str]:
    found = []
    for name, sql, text in NOTES:
        if name == table:
            try:
                found.append(text.format(**db.select(sql)["rows"][0]))
            except Exception:  # noqa: BLE001 - a table or column this export lacks
                continue
    return found


def build(db, database, schema, checkpoint, boundary, events, log, names=None) -> dict:
    tables, names = {}, names or {}
    db.command(f"USE {database}")  # worked queries name tables unqualified
    linked = links(db, schema)
    values = {"month": quote(f"{checkpoint}-01"), "boundary": quote(boundary.replace("T", " ").rstrip("Z"))}  # fmt: skip
    for table in sorted(schema):
        stats = table_stats(db, database, table, schema[table], boundary, log, names.get(table, {}))
        about, use = sources.ABOUT.get(table, (f"One row per event of an unrecognised type ({table}); its fields are auto-detected columns.", "A row is identified by event_id."))  # fmt: skip
        extracted = [c for c in stats["columns"] if c["name"] not in dict(BASE)]
        present = {c["name"] for c in stats["columns"]}
        key = [c for c in sources.KEYS.get(table, []) if c in present] or [c["name"] for c in extracted if c["null_share"] is not None and c["null_share"] < 0.5][:KEY]  # fmt: skip
        tables[table] = {"table": table, "system": system(table), "about": about, "use_for": use,
                         "rows": stats["rows"], "first": stats["first"], "last": stats["last"],
                         "key_columns": key, "columns": stats["columns"], "notes": notes(db, table),
                         "joins": sources.JOINS.get(table, []), "links": [x for x in linked if table in (x["from"]["table"], x["to"]["table"])],
                         "examples": [{"question": q, "sql": sql} for q, t, p in sources.EXAMPLES.get(table, []) if (sql := worked(db, t, p, values))]}  # fmt: skip
    summary = [{k: t[k] for k in ("table", "system", "about", "use_for", "rows", "first", "last", "key_columns", "notes")} | {"columns": len(t["columns"])} for t in tables.values()]  # fmt: skip
    head = {"checkpoint": checkpoint, "boundary": boundary, "events": events, "dialect": "ClickHouse SQL", "people": sources.PEOPLE}  # fmt: skip
    return {**head, "sources": summary, "tables": tables, "links": linked}


def sources_md(catalog) -> str:
    lines = [f"# Sources — checkpoint {catalog['checkpoint']} (every record is before {catalog['boundary']})", "",
             "Raw records of every event, one ClickHouse table per source; nothing is summarised or cut. "
             "Query with read-only ClickHouse SQL. Times are UTC: time columns are DateTime64(6, 'UTC') and accept "
             "'2026-03-01 00:00:00' or '2026-03-01T00:00:00Z' (or parseDateTime64BestEffort('…Z')). `actor` names who caused each event. "
             f"{catalog['people']} "
             "Every row has event_id, event_type, time, actor, "
             "title, text (full rendered text), entities and raw (the original record). Columns, joins and worked "
             "queries of a source: <table>.md.", ""]  # fmt: skip
    for s in catalog["sources"]:
        span = f"{s['first'][:10]} → {s['last'][:10]}" if s["rows"] else "empty"
        lines += [f"## {s['table']} ({s['system']})", "",
                  f"{s['rows']:,} rows, {span}, {s['columns']} columns. {s['about']} {s['use_for']} "
                  f"Key columns: {', '.join(s['key_columns']) or '-'}." + "".join(f" Note: {n}" for n in s["notes"]), ""]  # fmt: skip
    return "\n".join(lines)


def cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def percent(share) -> str:
    """A null share as a percentage that never prints a non-zero share as 0 (6 of 8,382 → 0.07)."""
    if share is None:
        return "-"
    if share in (0, 1):
        return str(int(100 * share))
    value = 100 * share
    return (
        "<0.01"
        if value < 0.01
        else ">99.99"
        if value > 99.99
        else f"{value:.2f}".rstrip("0").rstrip(".")
    )


def table_md(t) -> str:
    span = f"{t['first']} → {t['last']}" if t["rows"] else "no rows"
    lines = [f"# {t['table']} ({t['system']})", "", t["about"], "", t["use_for"], "",
             f"Rows: {t['rows']:,} ({span}). Sorted by (time, event_id).", "", "## Columns", "",
             "| column | type | meaning | null % | distinct | examples |", "|---|---|---|---|---|---|"]  # fmt: skip
    for c in t["columns"]:
        share = percent(c["null_share"])
        lines.append(f"| {c['name']} | {c['type']} | {cell(c['description'])} | {share} | {c['distinct']} | {cell('; '.join(c['examples']))} |")  # fmt: skip
    lines += ["", "## Notes", ""] + [f"- {n}" for n in t["notes"]] if t["notes"] else []
    lines += ["", "## Joins", ""] + [f"- {j}" for j in t["joins"] or ["(none known)"]]
    lines += ["", "## Example queries", ""]
    for e in t["examples"]:
        lines += [f"{e['question']}:", "", "```sql", e["sql"], "```", ""]
    return "\n".join(lines)


def write(target: Path, catalog: dict) -> None:
    target.mkdir(parents=True, exist_ok=True)
    dump = lambda value: json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n"  # noqa: E731
    (target / "catalog.json").write_text(dump(catalog))
    (target / "sources.json").write_text(dump({k: catalog[k] for k in ("checkpoint", "boundary", "events", "dialect", "people", "sources")}))  # fmt: skip
    (target / "SOURCES.md").write_text(sources_md(catalog))
    for name, t in catalog["tables"].items():
        (target / f"{name}.md").write_text(table_md(t))
