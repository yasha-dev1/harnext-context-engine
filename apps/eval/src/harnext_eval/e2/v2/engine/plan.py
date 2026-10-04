"""The typed query plan `find` fills (via Jev) and its deterministic compiler to ONE ClickHouse
SELECT. Identifiers come only from the catalog; every literal is typed by its column and quoted;
operators follow the column type (equality or `has` for arrays, case-insensitive substring for
text, instant ranges for times). An invalid plan raises PlanError with a message the agent can
act on, before anything runs.

Shapes: lookup / list / search (rows, newest or ordered), timeline (rows in time order), count
(one total, or one count per group value, with a distinct count), values (every distinct value
of one column among the matching rows), latest / earliest (the one newest / oldest row).
Filters: equality / `has`, exclusion (`!=`, NOT has), containment, ranges for times and numbers,
empty / set. Compares: two columns of the main table, or a main column against a column of the
linked row (`(key, a) [NOT] IN (SELECT key, b FROM link …)`). A link restricts the rows through
another table the catalog joins to (`main.target` matches `link.column`, scalar or array on
either side), in any source group. `times_sql` orders two events and measures the time between
them in SQL.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import asdict, dataclass, field

from harnext_eval.e2.v2.engine.prompts import SHAPES
from harnext_eval.e2.v2.stores.raw.backend import quote

OPS = ("eq", "ne", "contains", "before", "through", "from", "after", "on", "null", "notnull")
RANGES = ("before", "through", "from", "after")
# time or number ranges: before = strictly before the start of the named period (or <);
# through = up to and including its end (<=); from = from its start on (>=); after = strictly
# after its end (>). A named period is a year, month, day, minute, second or sub-second instant.
UNITS = {"milliseconds": 1, "seconds": 1000, "minutes": 60_000, "hours": 3_600_000, "days": 86_400_000, "weeks": 604_800_000}  # fmt: skip
ROUNDINGS = {"down": "floor", "nearest": "round", "up": "ceil", "exact": ""}
NO_VALUE = ("null", "notnull")
MAX_STATEMENT = 20_000  # the guard's cap on one statement
CONTROL = re.compile(r"[\x00-\x1f\x7f]")
# Every SQL function the compiler (and rawtools' value probe) may emit: the raw store's guard
# must allow each one (test_engine_integration checks this against the real export).
FUNCTIONS = frozenset({
    "positionCaseInsensitiveUTF8", "arrayExists", "has", "hasAny", "arrayJoin",
    "groupUniqArray", "groupUniqArrayArray", "assumeNotNull", "isNotNull", "count", "countIf",
    "uniqExact", "lengthUTF8", "substringUTF8", "concat", "if", "greatest", "isNull", "empty",
    "notEmpty", "arraySort", "coalesce", "dateDiff", "abs", "ifNull", "floor", "round", "ceil",
    "divide",
})  # fmt: skip
HIDDEN = {"raw"}  # never returned by find; read(event_id) gives the record
TIME = re.compile(r"(\d{4})(?:-(\d\d)(?:-(\d\d)(?:[T ](\d\d):(\d\d)(?::(\d\d)(?:\.(\d{1,9}))?)?)?)?)?\s*(?:Z|UTC|\+00:?00)?")  # fmt: skip
REF = re.compile(r"\b([a-z][a-z0-9_]*(?:/[a-z][a-z0-9_]*)*)\.([a-z][a-z0-9_]*)\b")
WORD = re.compile(r"[a-z][a-z0-9_]*")


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class Filter:
    table: str
    column: str
    op: str
    value: str
    also: tuple[str, ...] = ()  # sibling columns: the value may match any of them (OR)


@dataclass(frozen=True)
class Link:
    table: str  # the other table
    column: str  # its join column
    target: str  # the main table's join column


@dataclass(frozen=True)
class Compare:
    left: str  # a main-table column
    op: str  # eq | ne
    right: str  # a main-table column, or a column of the linked table when `linked`
    linked: bool = False


@dataclass
class Plan:
    table: str
    shape: str
    filters: list[Filter] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    group: str | None = None
    order: str | None = None
    desc: bool = False
    link: Link | None = None
    limit: int = 200
    compares: list[Compare] = field(default_factory=list)
    distinct: str | None = None  # count shape: count distinct values of this column
    offset: int = 0  # rows skipped (paging)
    all_values: bool = False  # several values for one column: all must match (else any)

    def to_dict(self) -> dict:
        return asdict(self)


# ---- column types ---------------------------------------------------------------------------


def inner(kind: str) -> str:
    """Element type without Nullable/LowCardinality/Array wrappers."""
    for wrapper in ("Array(", "Nullable(", "LowCardinality("):
        while kind.startswith(wrapper):
            kind = kind[len(wrapper) : -1]
    return kind


def is_array(kind: str) -> bool:
    return kind.startswith("Array(")


def is_time(kind: str) -> bool:
    return inner(kind).startswith("DateTime")


def is_text(kind: str) -> bool:
    return inner(kind) == "String"


def is_number(kind: str) -> bool:
    return inner(kind).startswith(("Int", "UInt", "Float"))


# ---- catalog --------------------------------------------------------------------------------


def shape_of(value: str) -> str | None:
    """Identifier shape of an example value ('ABC-123' -> 'ABC-9'), None for prose or numbers."""
    value = str(value)
    if " " in value or len(value) > 40 or not re.search(r"[A-Za-z]", value) or not re.search(r"\d", value):  # fmt: skip
        return None
    return re.sub(r"\d+", "9", value)


class Catalog:
    """Column types, meanings and join edges of one export (its catalog.json document)."""

    def __init__(self, doc: dict):
        self.doc = doc
        self.tables = doc["tables"]
        self.types = {t: {c["name"]: c["type"] for c in d["columns"]} for t, d in self.tables.items()}  # fmt: skip
        self.info = {t: {c["name"]: c for c in d["columns"]} for t, d in self.tables.items()}
        self.shapes = {}  # (table, column) -> value shape of a join column, e.g. 'ABC-<n>'
        for link in doc.get("links") or []:
            for end in ("from", "to"):
                if link.get("value_shape"):
                    self.shapes.setdefault((link[end]["table"], link[end]["column"]), link["value_shape"])  # fmt: skip
        self.edges = self.join_edges()

    def group(self, table: str) -> str:
        return table.split("_")[0]

    def groups(self) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for table in sorted(self.tables):
            found.setdefault(self.group(table), []).append(table)
        return found

    def key(self, table: str) -> str | None:
        """The first scalar key column (what a count counts distinctly)."""
        keys = self.tables[table].get("key_columns") or []
        return next((k for k in keys if k in self.types[table] and not is_array(self.types[table][k])), None)  # fmt: skip

    def join_edges(self) -> dict[tuple[str, str], tuple[str, str]]:
        """{(table, other): (column, other_column)} from the catalog's structured links, then its
        prose join lines, then columns of different tables whose example values share one
        identifier shape (e.g. 'ABC-9'): the links are pairwise, the fallbacks close the gaps."""
        edges: dict[tuple[str, str], tuple[str, str]] = {}

        def add(t, c, o, oc):
            if t != o and c in self.types.get(t, {}) and oc in self.types.get(o, {}):
                edges.setdefault((t, o), (c, oc))
                edges.setdefault((o, t), (oc, c))

        for link in self.doc.get("links") or []:  # the catalog's structured joins come first
            add(link["from"]["table"], link["from"]["column"], link["to"]["table"], link["to"]["column"])  # fmt: skip

        for table, doc in sorted(self.tables.items()):
            for line in doc.get("joins", []):
                for part in line.split(";"):
                    refs = REF.findall(part)
                    words = WORD.findall(REF.sub(" ", part))
                    local = next((w for w in words if w in self.types[table]), None)
                    for others, column in refs:
                        for other in others.split("/"):
                            add(table, local, other, column)
        shapes: dict[str, list[tuple[str, str]]] = {}
        for table, columns in sorted(self.info.items()):
            for name, column in columns.items():
                found = {shape_of(v) for v in column.get("examples", [])}
                if name not in ("event_id", "entities", "raw", "text") and len(column.get("examples", [])) >= 2 and len(found) == 1 and None not in found:  # fmt: skip
                    shapes.setdefault(found.pop(), []).append((table, name))
        for members in shapes.values():
            for t, c in members:
                for o, oc in members:
                    add(t, c, o, oc)
        return edges

    def siblings(self, table: str, column: str) -> list[str]:
        """Variants of a column in the same table (`x` and `<prefix>_x`, same type), e.g. ids
        taken from a subject and ids mentioned in a body."""
        types = self.types[table]
        return [c for c in types if c != column and types[c] == types[column] and (c.endswith("_" + column) or column.endswith("_" + c))]  # fmt: skip

    def notes(self, table: str) -> list[str]:
        entry = next((s for s in self.doc.get("sources", []) if s.get("table") == table), {})
        return list(entry.get("notes") or self.tables[table].get("notes") or [])

    def describe(self, table: str, column: str, examples: int = 1, width: int = 140) -> str:
        info = self.info[table][column]
        shown = "; ".join(str(e)[:40] for e in info.get("examples", [])[:examples])
        shape = f"; values look like {self.shapes[(table, column)]}" if (table, column) in self.shapes else ""  # fmt: skip
        return f"{info['description'][:width]} [{inner(info['type'])}{shape}]" + (f" e.g. {shown}" if shown else "")  # fmt: skip

    def summary(self, table: str) -> str:
        t = self.tables[table]
        return f"{t['about']} Use for: {t['use_for']}"


# ---- literals -------------------------------------------------------------------------------


def instant(value: str) -> tuple[str, str]:
    """(start, end) of the period a date/time literal names, as plain UTC string literals (a
    time column compares them in its own zone, UTC; no conversion function sees the host zone)."""
    found = TIME.fullmatch(value.strip())
    if not found:
        raise PlanError(f"{value!r} is not a date or time (use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)")
    y, mo, d, h, mi, s, frac = found.groups()
    try:
        start = dt.datetime(int(y), int(mo or 1), int(d or 1), int(h or 0), int(mi or 0), int(s or 0), int((frac or "0").ljust(6, "0")[:6]))  # fmt: skip
        if frac:
            end = start + dt.timedelta(microseconds=1)
        elif h:
            end = start + dt.timedelta(seconds=1 if s else 60)
        elif d:
            end = start + dt.timedelta(days=1)
        elif mo:
            end = start.replace(year=start.year + start.month // 12, month=start.month % 12 + 1)
        else:
            end = start.replace(year=start.year + 1)
    except (ValueError, OverflowError):
        raise PlanError(f"{value!r} is not a valid date or time") from None
    return tuple(lit(t.strftime("%Y-%m-%d %H:%M:%S.%f")) for t in (start, end))


def lit(value) -> str:
    """A string literal the guard accepts: control characters become spaces, quotes and
    backslashes are escaped (non-ASCII text stays as is)."""
    return quote(CONTROL.sub(" ", str(value)))


def literal(value: str, kind: str, column: str) -> str:
    base = inner(kind)
    text = str(value).strip()
    if base.startswith(("Int", "UInt")):
        found = re.fullmatch(r"#?\s*(-?[0-9]+)", text)
        if not found:
            raise PlanError(f"{value!r} is not a whole number, but column {column} holds numbers")
        return found.group(1)
    if base.startswith("Float"):
        try:
            return repr(float(text))
        except ValueError:
            raise PlanError(f"{value!r} is not a number, but column {column} holds numbers") from None  # fmt: skip
    if base == "Bool":
        if text.lower() not in ("true", "false"):
            raise PlanError(f"{value!r} is not true/false (column {column})")
        return text.lower()
    if base.startswith("DateTime"):
        return instant(text)[0]
    return lit(value)


def condition(f: Filter, kind: str) -> str:
    """One filter; with sibling columns, the same test on each of them joined by OR."""
    if f.also:
        parts = [condition(Filter(f.table, c, f.op, f.value), kind) for c in (f.column, *f.also)]
        return "(" + " OR ".join(f"({p})" for p in parts) + ")"
    col = f"`{f.table}`.`{f.column}`"
    if f.op in NO_VALUE:
        if is_array(kind):
            return f"{'empty' if f.op == 'null' else 'notEmpty'}({col})"
        return f"{'isNull' if f.op == 'null' else 'isNotNull'}({col})"
    if f.op == "ne":
        positive = condition(Filter(f.table, f.column, "eq", f.value), kind)
        return f"NOT ({positive})" if is_array(kind) else f"(isNull({col}) OR NOT ({positive}))"
    if f.op in RANGES and is_number(kind) and not is_array(kind):
        value = literal(f.value, kind, f.column)
        return f"{col} {dict(before='<', through='<=', after='>', **{'from': '>='})[f.op]} {value}"
    if f.op in (*RANGES, "on"):
        if not is_time(kind) or is_array(kind):
            raise PlanError(f"a range needs a time or number column; {f.table}.{f.column} is {kind}")  # fmt: skip
        low, high = instant(f.value)
        return {"before": f"{col} < {low}", "through": f"{col} < {high}", "from": f"{col} >= {low}",
                "after": f"{col} >= {high}", "on": f"{col} >= {low} AND {col} < {high}"}[f.op]  # fmt: skip
    if f.op == "contains" and is_text(kind):
        needle = lit(f.value)
        if is_array(kind):
            return f"arrayExists(x -> positionCaseInsensitiveUTF8(x, {needle}) > 0, {col})"
        return f"positionCaseInsensitiveUTF8({col}, {needle}) > 0"
    if is_time(kind):
        low, high = instant(f.value)
        return f"{col} >= {low} AND {col} < {high}"
    value = literal(f.value, kind, f.column)
    return f"has({col}, {value})" if is_array(kind) else f"{col} = {value}"


# ---- validation + compilation ---------------------------------------------------------------


def validate(plan: Plan, catalog: Catalog, max_rows: int) -> None:
    if plan.table not in catalog.types:
        raise PlanError(f"unknown table {plan.table!r}; tables: {', '.join(sorted(catalog.types))}")
    if plan.shape not in SHAPES:
        raise PlanError(f"unknown shape {plan.shape!r}; shapes: {', '.join(SHAPES)}")
    columns = catalog.types[plan.table]
    tables = {plan.table}
    if plan.link:
        link = plan.link
        if link.table not in catalog.types:
            raise PlanError(f"unknown table {link.table!r}")
        if catalog.edges.get((plan.table, link.table)) != (link.target, link.column):
            raise PlanError(f"{plan.table} and {link.table} are not joined on {link.target} = {link.column}")  # fmt: skip
        tables.add(link.table)
    for f in plan.filters:
        if f.table not in tables:
            raise PlanError(f"filter on {f.table}, which is not part of this query")
        if f.column not in catalog.types[f.table]:
            raise PlanError(f"unknown column {f.table}.{f.column}; columns: {', '.join(catalog.types[f.table])}")  # fmt: skip
        if f.op not in OPS:
            raise PlanError(f"unknown operator {f.op!r}")
        for other in f.also:
            if catalog.types[f.table].get(other) != catalog.types[f.table][f.column]:
                raise PlanError(f"{f.table}.{other} is not a sibling of {f.table}.{f.column}")
        condition(f, catalog.types[f.table][f.column])  # type checks
    for name in [*plan.columns, plan.group, plan.order]:
        if name is not None and (name not in columns or name in HIDDEN):
            raise PlanError(f"unknown column {plan.table}.{name}; columns: {', '.join(c for c in columns if c not in HIDDEN)}")  # fmt: skip
    for c in plan.compares:
        other = catalog.types[plan.link.table] if c.linked and plan.link else None if c.linked else columns  # fmt: skip
        if other is None:
            raise PlanError("a comparison with a linked column needs a link")
        if c.op not in ("eq", "ne") or c.left not in columns or c.right not in other:
            raise PlanError(f"invalid comparison {c.left} {c.op} {c.right}")
        if inner(columns[c.left]) != inner(other[c.right]) or is_array(columns[c.left]) or is_array(other[c.right]):  # fmt: skip
            raise PlanError(f"{c.left} and {c.right} are not comparable scalar columns of one type")
        if c.linked and (is_array(columns[plan.link.target]) or is_array(catalog.types[plan.link.table][plan.link.column])):  # fmt: skip
            raise PlanError("a linked comparison needs a scalar join key")
    if plan.distinct is not None and plan.distinct not in columns:
        raise PlanError(f"unknown column {plan.table}.{plan.distinct}")
    if plan.shape == "values" and not plan.group:
        raise PlanError("a list of values needs the column whose values are wanted")
    if plan.link and not any(f.table == plan.link.table for f in plan.filters):
        raise PlanError(f"the link to {plan.link.table} has no filter; pass values that identify those records")  # fmt: skip
    if plan.shape != "count" and not plan.filters and not plan.compares:
        raise PlanError("no value was used as a filter; pass the literal values that identify the records (keys, numbers, names, titles, dates)")  # fmt: skip
    if not 1 <= plan.limit <= max_rows:
        raise PlanError(f"limit must be 1..{max_rows}")
    if plan.offset < 0:
        raise PlanError("offset must not be negative")


def where(filters: list[Filter], catalog: Catalog, all_values: bool = False) -> str:
    """AND across columns; several values for one column are alternatives (OR) unless
    `all_values`; ranges, exclusions and empty checks always hold together (AND)."""
    groups: dict[tuple[str, str, str], list[str]] = {}
    for f in filters:
        key = (f.table, f.column, "value" if f.op in ("eq", "contains", "on") else "all")
        groups.setdefault(key, []).append(condition(f, catalog.types[f.table][f.column]))
    parts = []
    for (_, _, kind), conds in groups.items():
        joiner = " OR " if kind == "value" and not all_values else " AND "
        parts.append(conds[0] if len(conds) == 1 else "(" + joiner.join(f"({c})" for c in conds) + ")")  # fmt: skip
    return " AND ".join(parts) or "1"


def link_condition(plan: Plan, catalog: Catalog) -> str:
    link = plan.link
    main_kind, link_kind = catalog.types[plan.table][link.target], catalog.types[link.table][link.column]  # fmt: skip
    inner_where = where(
        [f for f in plan.filters if f.table == link.table], catalog, plan.all_values
    )
    target, column, table = f"`{plan.table}`.`{link.target}`", f"`{link.table}`.`{link.column}`", f"`{link.table}`"  # fmt: skip
    if is_array(main_kind):
        values = f"groupUniqArrayArray({column})" if is_array(link_kind) else f"groupUniqArray(assumeNotNull({column}))"  # fmt: skip
        return f"hasAny({target}, (SELECT {values} FROM {table} WHERE ({inner_where}) AND isNotNull({column})))"  # fmt: skip
    values = f"arrayJoin({column})" if is_array(link_kind) else column
    return f"{target} IN (SELECT {values} FROM {table} WHERE {inner_where})"


def compare_condition(plan: Plan, c: Compare) -> str:
    """Two columns of one row, or a main column against the linked row(s) with the same join key."""
    left = f"`{plan.table}`.`{c.left}`"
    if not c.linked:  # NULL is a value here: NULL vs 'x' differ, NULL vs NULL are equal
        right = f"`{plan.table}`.`{c.right}`"
        if c.op == "eq":
            return f"ifNull({left} = {right}, isNull({left}) AND isNull({right}))"
        return f"ifNull({left} != {right}, isNull({left}) != isNull({right}))"
    link = plan.link
    pairs = f"SELECT `{link.table}`.`{link.column}`, `{link.table}`.`{c.right}` FROM `{link.table}`"
    return (
        f"(`{plan.table}`.`{link.target}`, {left}) {'IN' if c.op == 'eq' else 'NOT IN'} ({pairs})"
    )


def cell(plan: Plan, name: str, kind: str, width: int) -> str:
    """A selected column; long text is cut to `width` characters (around a searched phrase)."""
    col = f"`{plan.table}`.`{name}`"
    if not is_text(kind) or is_array(kind) or name == "event_id":
        return f"{col} AS `{name}`"
    needle = next((f.value for f in plan.filters if f.table == plan.table and f.column == name and f.op == "contains"), None)  # fmt: skip
    if needle:
        start = f"greatest(1, positionCaseInsensitiveUTF8({col}, {lit(needle)}) - {width // 3})"
        cut = f"concat(if({start} > 1, '…', ''), substringUTF8({col}, {start}, {width}), '…')"
    else:
        cut = f"concat(substringUTF8({col}, 1, {width}), '…')"
    return f"if(lengthUTF8({col}) > {width}, {cut}, {col}) AS `{name}`"


def compile_plan(plan: Plan, catalog: Catalog, max_rows: int = 200, width: int = 300) -> str:
    sql = build(plan, catalog, max_rows, width)
    if len(sql) > MAX_STATEMENT:
        raise PlanError(f"the query would exceed {MAX_STATEMENT} characters; pass fewer or shorter values")  # fmt: skip
    return sql


def build(plan: Plan, catalog: Catalog, max_rows: int, width: int) -> str:
    validate(plan, catalog, max_rows)
    types = catalog.types[plan.table]
    table = f"`{plan.table}`"
    conds = [where([f for f in plan.filters if f.table == plan.table], catalog, plan.all_values)]
    if plan.link:
        conds.append(link_condition(plan, catalog))
    conds += [compare_condition(plan, c) for c in plan.compares]
    clause = " AND ".join(c for c in conds if c != "1") or "1"
    if plan.shape == "values":
        col = f"`{plan.table}`.`{plan.group}`"
        values = f"groupUniqArrayArray({col})" if is_array(types[plan.group]) else f"groupUniqArray({col})"  # fmt: skip
        return f"SELECT arraySort({values}) AS `values`, count() AS `rows` FROM {table} WHERE {clause}"  # fmt: skip
    if plan.shape == "count":
        key = plan.distinct or catalog.key(plan.table)
        distinct = f", uniqExact(`{plan.table}`.`{key}`) AS `distinct_{key}`" if key and key != plan.group else ""  # fmt: skip
        if not plan.group:
            return f"SELECT count() AS `n`{distinct} FROM {table} WHERE {clause}"
        kind = types[plan.group]
        group = f"arrayJoin(`{plan.table}`.`{plan.group}`)" if is_array(kind) else f"`{plan.table}`.`{plan.group}`"  # fmt: skip
        return f"SELECT {group} AS `{plan.group}`, count() AS `n`{distinct} FROM {table} WHERE {clause} GROUP BY `{plan.group}` ORDER BY `n` DESC, `{plan.group}` LIMIT {plan.limit}"  # fmt: skip
    names = ["event_id", "time", *[c for c in plan.columns if c not in ("event_id", "time")]]
    select = ", ".join(cell(plan, n, types[n], width) for n in names)
    order = plan.order or "time"
    keys = [order] + [k for k in ("time", "event_id") if k != order]
    order_by = ", ".join(f"`{plan.table}`.`{k}` {'DESC' if descending(plan) else 'ASC'}" for k in keys)  # fmt: skip
    limit = 1 if plan.shape in ("latest", "earliest") else plan.limit
    offset = f" OFFSET {plan.offset}" if plan.offset and plan.shape not in ("latest", "earliest") else ""  # fmt: skip
    return f"SELECT {select} FROM {table} WHERE {clause} ORDER BY {order_by} LIMIT {limit}{offset}"


def descending(plan: Plan) -> bool:
    return plan.shape == "latest" or (plan.desc and plan.shape != "earliest")


def precise_time(catalog: Catalog, table: str) -> str:
    """The most precise instant of a row: the first extra time column when it is set (e.g. a
    recorded instant with microseconds), else the event time."""
    extra = next((c for c, k in catalog.types[table].items() if c != "time" and is_time(k) and not is_array(k)), None)  # fmt: skip
    return f"coalesce(`{extra}`, `time`)" if extra else "`time`"


def times_sql(catalog: Catalog, a: tuple[str, str], b: tuple[str, str], unit: str = "seconds", rounding: str = "exact") -> str:  # fmt: skip
    """Order two events (table, event_id) and measure the time between them, all in SQL: the
    exact difference in milliseconds and the absolute difference in `unit`, rounded as asked."""
    if unit not in UNITS or rounding not in ROUNDINGS:
        raise PlanError(f"unit must be one of {', '.join(UNITS)}; rounding one of {', '.join(ROUNDINGS)}")  # fmt: skip

    def one(table: str, event_id: str) -> str:
        return f"(SELECT {precise_time(catalog, table)} AS `t` FROM `{table}` WHERE `event_id` = {lit(event_id)})"  # fmt: skip

    ms = "dateDiff('millisecond', `a`.`t`, `b`.`t`)"
    amount = f"divide(abs({ms}), {UNITS[unit]})"
    amount = f"{ROUNDINGS[rounding]}({amount})" if ROUNDINGS[rounding] else amount
    return (f"SELECT `a`.`t` AS `a_time`, `b`.`t` AS `b_time`, `a`.`t` < `b`.`t` AS `a_first`, "
            f"`a`.`t` = `b`.`t` AS `same_instant`, {ms} AS `milliseconds_a_to_b`, {amount} AS `amount` "
            f"FROM {one(*a)} AS `a` CROSS JOIN {one(*b)} AS `b`")  # fmt: skip


def count_plan(plan: Plan) -> Plan:
    """The plan counting every row the row plan matches (no LIMIT, no subquery in FROM)."""
    return Plan(**{**plan.__dict__, "shape": "count", "group": None, "distinct": None, "offset": 0})


def relax(plan: Plan, catalog: Catalog) -> Plan | None:
    """The same plan with exact text equality turned into case-insensitive containment, or None."""
    changed = [Filter(f.table, f.column, "contains", f.value, f.also) if f.op == "eq" and is_text(catalog.types[f.table][f.column]) else f for f in plan.filters]  # fmt: skip
    if changed == plan.filters:
        return None
    return Plan(**{**plan.__dict__, "filters": changed})
