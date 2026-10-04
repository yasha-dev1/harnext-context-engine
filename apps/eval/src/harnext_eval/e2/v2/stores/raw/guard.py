"""Statement guard of the read API: one read statement over the export's own tables, nothing else.

A small tokenizer (strings, quoted identifiers, numbers, words, operators) feeds three rules:

- shape: `SELECT …` / `WITH …` (one statement, optional trailing `;`), `DESCRIBE|DESC [TABLE]
  <export table>`, or `SHOW TABLES`. Comments (`--`, `#`, `/* */`), heredoc strings (`$$`) and
  query parameters (`{x:T}`) are rejected outright, so the text that runs is the text checked.
- tables: every FROM / JOIN target (outside a function's own parentheses) is a subquery, an
  export table or a CTE name — never a table function, a qualified name or a system table.
- functions, deny by default: every call (a bare or quoted name followed by `(`, unless the word
  is SQL syntax such as IN / OVER / a type in CAST) must be on `ALLOW`, or be an allowed
  aggregate plus the combinators If / Distinct / OrNull / OrDefault / Array. A function the list
  does not name — also one a chdb upgrade adds — is refused. Names match case-insensitively;
  the tests check against the engine's own registries that every other name, in any case or
  quoting, is refused and that the list names no table function and no missing function.

Plus: no DDL/DML/SETTINGS/FORMAT/INTO keyword and no `system` / `information_schema`. The engine
adds readonly = 1 and the limits per connection (`backend.ChdbBackend.select`); this module stops
what readonly = 1 alone lets through (scalar `file()`, `DESCRIBE file(…)`, `hostName()`,
`system.*`, `numbers()` — measured, see E2-ENGINE-DESIGN.md).
"""

from __future__ import annotations

import difflib
import re

# Literals: the boundaries are the engine's ('…' with '' and backslash escapes); inside, only
# escapes whose meaning is unambiguous pass (`ESCAPES`): the engine keeps unknown escapes as a
# backslash, turns \N into '', reads \xZZ as a byte and fails on \x4 — all refused here.
# Quoted identifiers take doubled quotes only, no backslash. ASCII only outside both.
TOKEN = re.compile(
    r"(?P<ws>[ \t\r\n]+)|(?P<comment>--|\#|/\*)|(?P<string>'(?:[^'\\]|\\.|'')*')"
    r"|(?P<quoted>`(?:[^`\\]|``)*`|\"(?:[^\"\\]|\"\")*\")"
    r"|(?P<number>\d[\w.]*(?:[eE][+-]?\d+)?)|(?P<word>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<bad>\$|\{|\})|(?P<op>::|->|<=|>=|!=|<>|\|\||==|[-+*/%=<>(),.;\[\]:?^!@&|~])",
    re.DOTALL | re.ASCII,
)
ESCAPES = re.compile(r"(?:[^'\\]|''|\\[\\'ntr0bfav]|\\x[0-7][0-9A-Fa-f])*", re.DOTALL)
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MAX_LENGTH = 20_000  # characters of one statement
KEYWORDS = {
    "INSERT", "INTO", "ALTER", "CREATE", "DROP", "ATTACH", "DETACH", "RENAME", "TRUNCATE",
    "OPTIMIZE", "SYSTEM", "KILL", "GRANT", "REVOKE", "SET", "SETTINGS", "FORMAT", "OUTFILE",
    "INFILE", "USE", "EXCHANGE", "DELETE", "UPDATE", "BACKUP", "RESTORE", "UNDROP", "MOVE",
    "WATCH", "EXPLAIN", "CHECK",
}  # fmt: skip
SCHEMAS = {"system", "information_schema"}
# SQL words that may stand before "(" without being a call; types appear in CAST(x AS T(…)).
SYNTAX = {
    "SELECT", "WHERE", "AND", "OR", "NOT", "ON", "USING", "AS", "BY", "THEN", "ELSE", "WHEN",
    "IN", "EXISTS", "OVER", "ALL", "FROM", "JOIN", "DISTINCT", "INTERVAL", "LIKE", "ILIKE",
    "BETWEEN", "HAVING", "PREWHERE", "FILTER", "WITH", "LIMIT", "OFFSET", "DIV", "IS", "CASE",
    "UNION", "EXCEPT", "INTERSECT", "QUALIFY", "GLOBAL", "RECURSIVE",
    "NULLABLE", "LOWCARDINALITY", "DATETIME64", "DECIMAL", "FIXEDSTRING", "ENUM8", "ENUM16",
    "DATETIME",
}  # fmt: skip
# ---- the allowlist, by group (engine spellings; `test_raw_store` checks each one exists) ------
ALLOW = {
    "comparison and logic": [
        "equals", "notEquals", "less", "greater", "lessOrEquals", "greaterOrEquals", "and", "or",
        "not", "xor", "in", "notIn", "like", "notLike", "ilike", "notILike",
    ],
    "arithmetic and rounding": [
        "plus", "minus", "multiply", "divide", "intDiv", "intDivOrZero", "modulo", "mod", "negate",
        "abs", "least", "greatest", "round", "roundBankers", "floor", "ceil", "ceiling",
        "truncate", "trunc", "sqrt", "pow", "power", "exp", "log", "ln", "log2", "log10", "sign",
    ],
    "strings and text search": [
        "length", "lengthUTF8", "CHAR_LENGTH", "lower", "upper", "lowerUTF8", "upperUTF8",
        "substring", "substr", "substringUTF8", "left", "right", "leftUTF8", "rightUTF8",
        "concat", "concatWithSeparator", "trim", "trimBoth", "trimLeft", "trimRight", "ltrim",
        "rtrim", "startsWith", "endsWith", "position", "positionCaseInsensitive", "positionUTF8",
        "positionCaseInsensitiveUTF8", "locate", "match", "extract", "replace", "replaceOne", "replaceAll", "replaceRegexpOne",
        "replaceRegexpAll", "regexpExtract", "hasToken", "hasTokenCaseInsensitive",
        "hasTokenOrNull", "hasTokenCaseInsensitiveOrNull", "multiSearchAny",
        "multiSearchAnyCaseInsensitive", "multiSearchAnyUTF8", "multiSearchAnyCaseInsensitiveUTF8",
        "multiSearchFirstIndex", "multiSearchFirstPosition", "splitByChar",
        "splitByString", "splitByRegexp", "splitByWhitespace", "splitByNonAlpha", "empty",
        "notEmpty", "countSubstrings", "countSubstringsCaseInsensitive",
        "ngramDistance", "ngramDistanceCaseInsensitive", "ngramDistanceUTF8",
        "ngramDistanceCaseInsensitiveUTF8", "ngramSearch", "ngramSearchCaseInsensitive",
        "ngramSearchUTF8", "ngramSearchCaseInsensitiveUTF8",
    ],
    "date and time": [
        "toDate", "toDate32", "toDateTime", "toDateTime64", "toStartOfDay", "toStartOfHour",
        "toStartOfMinute", "toStartOfWeek", "toStartOfMonth", "toStartOfQuarter",
        "toStartOfYear", "toMonday", "toYear", "toQuarter", "toMonth", "toDayOfMonth",
        "toDayOfWeek", "toDayOfYear", "toHour", "toMinute", "toSecond", "toYYYYMM", "toYYYYMMDD",
        "toISOWeek", "toUnixTimestamp", "dateDiff", "date_diff", "DATE_DIFF", "timestamp_diff",
        "TIMESTAMP_DIFF", "age", "DATE_TRUNC", "dateTrunc", "addSeconds", "addMinutes", "addHours",
        "addDays", "addWeeks", "addMonths", "addYears", "subtractSeconds", "subtractMinutes",
        "subtractHours", "subtractDays", "subtractWeeks", "subtractMonths", "subtractYears",
        "toIntervalSecond", "toIntervalMinute", "toIntervalHour", "toIntervalDay",
        "toIntervalWeek", "toIntervalMonth", "toIntervalYear", "parseDateTimeBestEffort",
        "parseDateTimeBestEffortOrNull", "parseDateTime64BestEffort",
        "parseDateTime64BestEffortOrNull", "formatDateTime",
    ],
    "arrays and higher-order": [
        "array", "has", "hasAny", "hasAll", "arrayJoin", "arrayMap", "arrayFilter",
        "arrayExists", "arrayAll", "arrayCount", "arrayFirst", "arrayLast", "arraySum",
        "arrayMin", "arrayMax", "arrayDistinct", "arraySort", "arrayReverseSort",
        "arrayStringConcat", "indexOf", "arrayElement", "arrayConcat", "arraySlice", "arrayUniq",
        "arrayIntersect", "arrayEnumerate", "arrayReverse", "arrayFlatten", "arrayZip",
        "countEqual",
    ],
    "JSON over raw": [
        "JSONExtract", "JSONExtractString", "JSONExtractInt", "JSONExtractUInt",
        "JSONExtractFloat", "JSONExtractBool", "JSONExtractRaw", "JSONExtractArrayRaw",
        "JSONExtractKeys", "JSONExtractKeysAndValues", "JSONHas", "JSONLength", "JSONType",
        "JSON_VALUE", "JSON_QUERY", "JSON_EXISTS", "isValidJSON", "simpleJSONExtractString",
        "simpleJSONExtractInt", "simpleJSONExtractRaw", "simpleJSONHas",
        "visitParamExtractString", "visitParamExtractInt", "visitParamExtractRaw",
        "visitParamHas",
    ],
    "type conversion": [
        "CAST", "accurateCastOrNull", "toString", "toTypeName", "toBool", "toInt8", "toInt16",
        "toInt32", "toInt64", "toUInt8", "toUInt16", "toUInt32", "toUInt64", "toFloat32",
        "toFloat64", "toInt32OrNull", "toInt32OrZero", "toInt64OrNull", "toInt64OrZero",
        "toUInt64OrNull", "toUInt64OrZero", "toFloat64OrNull", "toFloat64OrZero", "toDateOrNull",
        "toDateTimeOrNull", "toDateTime64OrNull",
    ],
    "conditionals and null handling": [
        "if", "multiIf", "coalesce", "ifNull", "nullIf", "isNull", "isNotNull", "assumeNotNull",
        "toNullable",
    ],
    "aggregates": [
        "count", "sum", "avg", "min", "max", "any", "anyLast", "argMin", "argMax", "uniq",
        "uniqExact", "uniqCombined", "groupArray", "groupUniqArray",
        "groupArraySorted", "groupConcat", "topK", "quantile", "quantiles", "quantileExact",
        "quantilesExact", "median", "medianExact", "stddevPop", "stddevSamp", "varPop", "varSamp",
        "first_value", "last_value", "singleValueOrNull",
    ],
    "window": [
        "row_number", "rank", "dense_rank", "percent_rank", "cume_dist", "ntile", "lagInFrame",
        "leadInFrame", "nth_value",
    ],
    "tuples and maps": [
        "tuple", "tupleElement", "map", "mapKeys", "mapValues", "mapContains",
    ],
}  # fmt: skip
COMBINATORS = ("If", "Distinct", "OrNull", "OrDefault", "Array")
NAMES = {n.lower(): n for group in ALLOW.values() for n in group}
AGGREGATES = {n.lower() for n in ALLOW["aggregates"]}


class QueryError(ValueError):
    pass


def allowed(function: str) -> bool:
    """On the list (any case), or an allowed aggregate plus allowed combinators."""
    if function.lower() in NAMES:
        return True
    base, stripped = function, False
    while suffix := next((c for c in COMBINATORS if base.endswith(c) and len(base) > len(c)), None):
        base, stripped = base[: -len(suffix)], True
        if base.lower() in AGGREGATES:
            return True
    return stripped and base.lower() in AGGREGATES


SYNONYMS = {
    "lag": "use lagInFrame(x) OVER (… ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)",
    "lead": "use leadInFrame(x) OVER (… ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)",
    "now": "the current time is not available (results are frozen at the checkpoint); compare with explicit dates",
    "now64": "the current time is not available (results are frozen at the checkpoint); compare with explicit dates",
    "today": "the current date is not available (results are frozen at the checkpoint); compare with explicit dates",
    "yesterday": "the current date is not available (results are frozen at the checkpoint); compare with explicit dates",
    "currentdate": "the current date is not available (results are frozen at the checkpoint); compare with explicit dates",
    "count_distinct": "use uniqExact(x) or count(DISTINCT x)",
    "string_agg": "use groupConcat(x) or arrayStringConcat(groupArray(x), ', ')",
    "listagg": "use groupConcat(x) or arrayStringConcat(groupArray(x), ', ')",
    "group_concat": "use groupConcat(x)", "date_part": "use toYear(t), toMonth(t), … or EXTRACT(YEAR FROM t)",
    "datediff": "use dateDiff('unit', a, b)", "strftime": "use formatDateTime(t, '%Y-%m-%d')",
}  # fmt: skip


def refusal(function: str) -> str:
    if function.lower() in SYNONYMS:
        return f"function {function} is not available: {SYNONYMS[function.lower()]}. See catalog SQL reference (RawDB.sql_reference())"  # fmt: skip
    close = difflib.get_close_matches(function.lower(), list(NAMES), n=3, cutoff=0.6)
    close += [n for n in NAMES if n.startswith(function.lower()[:4]) and n not in close][: 3 - len(close)]  # fmt: skip
    hint = f"; closest allowed: {', '.join(NAMES[n] for n in close)}" if close else ""
    return (f"function {function} is not available{hint}. Allowed functions: {', '.join(ALLOW)} "
            "(+ aggregate combinators If/Distinct/OrNull/OrDefault/Array); see the SQL reference (RawDB.sql_reference())")  # fmt: skip


def tokens(query: str) -> list[tuple[str, str]]:
    if len(query) > MAX_LENGTH:
        raise QueryError(f"the statement is longer than {MAX_LENGTH} characters")
    if found := CONTROL.search(query):
        raise QueryError(f"control character {found.group(0)!r} is not allowed")
    found, position = [], 0
    while position < len(query):
        match = TOKEN.match(query, position)
        if not match:
            rest = query[position : position + 20]
            if rest[:1] in ("'", "`", '"'):
                raise QueryError(f"unterminated or malformed literal/identifier at: {rest!r}")
            if not rest[:1].isascii():
                raise QueryError(f"non-ASCII character {rest[:1]!r} outside a string literal or quoted identifier")  # fmt: skip
            raise QueryError(f"cannot read the statement at: {rest!r}")
        kind, text = match.lastgroup, match.group(0)
        if kind == "string" and not ESCAPES.fullmatch(text[1:-1]):
            raise QueryError(r"string literal with an unsupported escape: allowed are \\ \' \n \t \r \0 \b \f \a \v \xHH (00-7F) and ''; write a regex backslash as \\ (e.g. '\\s+', '\\[')")  # fmt: skip
        if kind == "comment":
            raise QueryError("comments (--, #, /* */) are not allowed")
        if kind == "bad":
            raise QueryError(f"'{text}' is not allowed (no heredoc strings or query parameters)")
        if kind != "ws":
            found.append((kind, text))
        position = match.end()
    return found


def name(token: tuple[str, str]) -> str | None:
    kind, text = token
    if kind == "word":
        return text
    if kind == "quoted":  # `x` and "x" are both identifiers in ClickHouse
        return text[1:-1].replace(text[0] * 2, text[0])
    return None


def call(found, i) -> str | None:
    """The function name if token i starts a call (name followed by "(", not SQL syntax)."""
    word = name(found[i])
    if word is None or found[i + 1 : i + 2] != [("op", "(")]:
        return None
    if found[i][0] == "word" and word.upper() in SYNTAX:
        return None
    return word


def check(query: str, tables) -> str:
    """The statement to run (the input minus surrounding blanks and a trailing `;`), or
    QueryError saying what is not allowed."""
    found = tokens(query)
    while found and found[-1] == ("op", ";"):
        found.pop()
    if not found:
        raise QueryError("empty statement")
    if ("op", ";") in found:
        raise QueryError("only one statement per call")
    adjacent(found)
    known = set(tables)
    first = found[0][1].upper() if found[0][0] == "word" else ""
    if first in ("DESCRIBE", "DESC"):
        rest = found[2:] if len(found) > 1 and found[1][1].upper() == "TABLE" else found[1:]
        if len(rest) != 1 or name(rest[0]) not in known:
            raise QueryError(f"DESCRIBE takes one export table: {', '.join(sorted(known))}")
    elif first == "SHOW":
        if [t[1].upper() for t in found] != ["SHOW", "TABLES"]:
            raise QueryError("only SHOW TABLES is allowed")
    elif first in ("SELECT", "WITH"):
        body(found, known)
    else:
        raise QueryError("read-only: a statement must be SELECT, WITH, DESCRIBE <table> or SHOW TABLES")  # fmt: skip
    return query.strip().rstrip(";").strip()


OPERANDS = {"string", "number", "quoted"}
FOLLOW = {"END", "ASC", "DESC", "NULLS", "COLLATE", "ORDER", "GROUP", "TO", "ESCAPE", "ROWS",
          "RANGE", "MOD", "DIV", "NOT"}  # fmt: skip


def adjacent(found) -> None:
    """Two values in a row (`'a' 'b'`, `'a' x 'b'`, `'a'(`) are a syntax error for the engine;
    refusing them keeps the guard's reading of literals the engine's (odd stray quotes)."""
    words = KEYWORDS | SYNTAX | CLAUSES | FOLLOW
    for i, (kind, _) in enumerate(found[:-1]):
        if kind not in OPERANDS:
            continue
        after = found[i + 1]
        if after[0] == "word" and after[1].upper() not in words and i + 2 < len(found):
            after = found[i + 2]  # an alias: what follows it
        if after[0] in OPERANDS or (after == ("op", "(") and kind != "quoted"):
            raise QueryError(
                f"two values in a row near {found[i][1][:30]!r}; check quotes and commas"
            )


def body(found, known) -> None:
    ctes = {name(found[i - 1]) for i in range(1, len(found) - 1)
            if found[i][1].upper() == "AS" and found[i + 1] == ("op", "(") and name(found[i - 1])}  # fmt: skip
    allowed_tables = known | ctes
    stack: list[bool] = []  # per open parenthesis: is it a function call's argument list?
    for i, (kind, text) in enumerate(found):
        if kind == "op" and text == "(":
            stack.append(i > 0 and call(found, i - 1) is not None)
            continue
        if kind == "op" and text == ")":
            if stack:
                stack.pop()
            continue
        word = name((kind, text))
        if word is None:
            continue
        function = call(found, i)
        alias = i > 0 and found[i - 1][0] == "word" and found[i - 1][1].upper() == "AS" and function is None  # fmt: skip
        if alias:
            continue  # a name given with AS, even a reserved word (`x AS from`)
        if kind == "word" and text.upper() in KEYWORDS and function is None:
            raise QueryError(f"{text.upper()} is not allowed in a read query; to use it as a name, quote it with backticks (`{text}`)")  # fmt: skip
        if word.lower() in SCHEMAS:
            raise QueryError(f"{word} tables are not available; tables: {', '.join(sorted(known))}")
        if function is not None and not allowed(function):
            raise QueryError(refusal(function))
        if kind == "word" and text.upper() in ("FROM", "JOIN") and not (stack and stack[-1]):
            if text.upper() == "JOIN" and i > 0 and found[i - 1][1].upper() == "ARRAY":
                continue  # ARRAY JOIN takes an array expression
            target(found, i + 1, allowed_tables, known)


CLAUSES = {"WHERE", "GROUP", "ORDER", "HAVING", "LIMIT", "UNION", "EXCEPT", "INTERSECT", "LEFT",
           "RIGHT", "INNER", "OUTER", "FULL", "CROSS", "ARRAY", "ON", "USING", "PREWHERE",
           "FINAL", "SAMPLE", "WINDOW", "QUALIFY", "ANY", "ALL", "ASOF", "SEMI", "ANTI",
           "GLOBAL", "PASTE", "OFFSET"}  # fmt: skip


def target(found, i, tables, known) -> None:
    """FROM/JOIN targets: `(subquery)` or an export table / CTE, comma-separated."""
    while i < len(found):
        if found[i] == ("op", "("):
            return  # a subquery: its tokens are checked by `body` like everything else
        table = name(found[i])
        following = found[i + 1][1] if i + 1 < len(found) else ""
        if table is None or table not in tables or following in (".", "("):
            shown = found[i][1] + ("." + found[i + 2][1] if following == "." and i + 2 < len(found) else "(" if following == "(" else "")  # fmt: skip
            close = difflib.get_close_matches(table or "", sorted(known), n=2, cutoff=0.6)
            hint = f"; did you mean {' or '.join(close)}?" if close else ""
            reserved = f"; `{table}` is a reserved word — quote a name with backticks" if (table or "").upper() in KEYWORDS | CLAUSES | SYNTAX and (table or "").lower() not in SCHEMAS else ""  # fmt: skip
            raise QueryError(f"unknown table {shown!r}{hint}{reserved}; tables: {', '.join(sorted(known))}")  # fmt: skip
        i += 1
        if i < len(found) and found[i][1].upper() == "AS":
            i += 1
        if i < len(found) and name(found[i]) and found[i][1].upper() not in KEYWORDS | CLAUSES | {"FROM", "JOIN"}:  # fmt: skip
            i += 1  # alias
        if i < len(found) and found[i] == ("op", ","):
            i += 1
            continue
        return


def summary() -> str:
    """The allowed SQL, compact, for catalog/SQL.md."""
    lines = ["# SQL available to the `sql` tool", "",
             "One statement: `SELECT …` / `WITH …` over the tables of SOURCES.md (joins, CTEs, subqueries, "
             "window functions, ARRAY JOIN, LIMIT BY, CAST, CASE), `DESCRIBE <table>` or `SHOW TABLES`. "
             "No comments, SETTINGS, FORMAT, table functions or other databases. Functions not listed "
             "here are refused.", "",
             f"Aggregate combinators: {', '.join(COMBINATORS)} (e.g. countIf, argMaxIf, uniqExactIf, "
             "groupArrayDistinct, sumArray).", ""]  # fmt: skip
    for group, names in ALLOW.items():
        lines.append(f"- {group}: {', '.join(names)}")
    lines += ["", "Time (all columns are DateTime64(6, 'UTC'); the session time zone is UTC):",
              "- compare directly: `time >= '2026-03-01 00:00:00'` or `time >= '2026-03-01T00:00:00Z'`",
              "- inside IN and function arguments convert first: `at IN (parseDateTime64BestEffort('2026-03-01T10:00:00.123Z'))`, "
              "`dateDiff('second', at, parseDateTime64BestEffort('2026-03-01T00:00:00Z'))`",
              "- `dateDiff('day', a, b)` counts calendar-day boundaries; full 24-hour periods from a to b: "
              "`intDiv(dateDiff('second', a, b), 86400)`",
              "- print an exact instant: `formatDateTime(at, '%Y-%m-%dT%H:%i:%S.%fZ')`",
              "", "Strings: `'…'` with `''` or `\\'` for a quote; a regex backslash is written `\\\\` (`'\\\\s+'`). Values are "
              "case-sensitive (`'Reopened'`, `'approved'`): copy them from the catalog examples. A reserved word as a "
              "name needs backticks."]  # fmt: skip
    return "\n".join(lines) + "\n"
