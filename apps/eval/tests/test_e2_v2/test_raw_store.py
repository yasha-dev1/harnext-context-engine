"""Arm RAW on a synthetic stream that crosses a checkpoint boundary: dynamic schema (new field,
type conflicts, name collisions, unknown types, rebuild determinism, resume), time freeze of the
exports, the read-only query API (statement guard, engine-level readonly and limits, hostile
queries), scratch hygiene, the catalog; and answerability of w2026 from the real exports."""

import ast
import hashlib
import json
import os
import random
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from harnext_eval.e2.v2 import stream
from harnext_eval.e2.v2.stores import build as stores_build
from harnext_eval.e2.v2.stores.base import ROOT
from harnext_eval.e2.v2.stores.raw import catalog, guard, oracle, query, sources
from harnext_eval.e2.v2.stores.raw.store import RawStore

pytest.importorskip("chdb")

LATER = "SENTINEL-ONLY-AFTER-THE-BOUNDARY"
W2026 = ROOT / "out/kafka-500-v2/stores-w2026/RAW"
SMOKE = ROOT / "out/kafka-500-v2/stores-smoke/RAW"
BENCH = ROOT / "benchmarks/kafka-50-w2026"
PRIVATE = BENCH / "tasks.private.jsonl"
real = pytest.mark.skipif(not (W2026 / "export/2026-05/schema.json").exists(), reason="build RAW --scope w2026 first")  # fmt: skip


def event(id_, time_, type_, meta, text="", entities=(), actor="Rita", title="t"):
    return {"id": id_, "time": time_, "type": type_, "entities": list(entities), "actor": actor, "title": title, "text": text, "meta": meta}  # fmt: skip


def status(history, time_, at, before, after):
    facts = [("issue", "KAFKA-1 — Fix the loader"), ("event", "field change"), ("by", "Rita"), ("at", at), ("field", "status"), ("from", before), ("to", after)]  # fmt: skip
    title = f"KAFKA-1 status: {before} → {after}"
    return event(f"jira:100:change:{history}:0", time_, "jira.change", {"issue": "KAFKA-1", "field": "status", "from": before, "to": after}, stream.page(title, facts), ["issue:KAFKA-1"], title=title)  # fmt: skip


EVENTS = [
    event("ci:1", "2026-01-05T00:00:00Z", "ci.build", {"n": 1, "tags": [1, 2], "ok": True, "f": 1, "big": 1, "My Field": "a", "time": "not the time", "raw": "x", "件名": "subject"}, stream.page("Build 1", [("runner", "linux")])),  # fmt: skip
    status(5, "2026-01-06T00:00:00Z", "2026-01-06T00:00:00.123000Z", "Open", "In Progress"),
    status(6, "2026-01-06T00:00:00Z", "2026-01-06T00:00:00.123000Z", "In Progress", "Patch Available"),  # fmt: skip
    event("ci:2", "2026-01-20T00:00:00Z", "ci.build", {"n": "abc", "tags": ["x"], "config": {"a": 1}, "f": 1.5, "big": 123456789012345678901234567890, "my.field": "b"}),  # fmt: skip
    event("x:1", "2026-01-21T00:00:00Z", "jira.changes", {"issue": "KAFKA-9"}),
    # ---- boundary of checkpoint 2026-01 ----
    event("ci:3", "2026-02-02T00:00:00Z", "ci.build", {"n": 3, "later_only": LATER}),
    status(7, "2026-02-03T00:00:00Z", "2026-02-03T00:00:00.500000Z", "Patch Available", "Resolved"),  # fmt: skip
]


def hashes(export: Path) -> dict:
    return json.loads(export.with_name(export.name + ".manifest.json").read_text())["files"]


def scratches() -> set:
    return {p.name for p in Path(tempfile.gettempdir()).glob(f"{query.PREFIX}{os.getpid()}-*")}


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    out = tmp_path_factory.mktemp("raw") / "RAW"
    RawStore().build(iter(EVENTS), out)
    return out


@pytest.fixture
def db(store):
    with query.open(store / "export/2026-02") as found:
        yield found


# ---- schema -------------------------------------------------------------------------------------


def test_new_fields_add_columns_and_conflicts_widen_with_a_logged_cause(store):
    january = json.loads((store / "export/2026-01/schema.json").read_text())
    ci = dict(january["tables"]["other_ci_build"])
    assert ci["n"] == "Nullable(String)" and ci["ok"] == "Nullable(Bool)"
    assert ci["f"] == "Nullable(Float64)"  # int then float: numeric widening
    assert ci["big"] == "Nullable(String)"  # beyond Int64: exact digits as text
    assert ci["tags"] == "Array(String)" and ci["config"] == "Nullable(String)"
    assert ci["runner"] == "Nullable(String)" and "later_only" not in ci  # a fact line; nothing later
    jira = dict(january["tables"]["jira_changes"])
    assert jira["at"] == "Nullable(DateTime64(6, 'UTC'))" and jira["history_id"] == "Nullable(Int64)"
    log = [json.loads(line) for line in (store / "export/2026-01/schema-log.jsonl").open()]
    widen = {(e["column"], e["from_type"], e["type"], e["event_id"]) for e in log if e["change"] == "widen"}  # fmt: skip
    assert widen == {
        ("n", "Nullable(Int64)", "Nullable(String)", "ci:2"), ("tags", "Array(Int64)", "Array(String)", "ci:2"),
        ("f", "Nullable(Int64)", "Nullable(Float64)", "ci:2"), ("big", "Nullable(Int64)", "Nullable(String)", "ci:2"),
    }  # fmt: skip
    february = dict(json.loads((store / "export/2026-02/schema.json").read_text())["tables"]["other_ci_build"])  # fmt: skip
    assert february["later_only"] == "Nullable(String)"


def test_names_never_collide_and_keep_their_origin(store, db):
    schema = json.loads((store / "export/2026-02/schema.json").read_text())
    names = schema["names"]["other_ci_build"]
    hashed = "field_" + hashlib.sha256("件名".encode()).hexdigest()[:6]
    assert {k: names[k] for k in ("my_field", "meta_my_field", "meta_time", "meta_raw", hashed)} == {
        "my_field": "meta:My Field", "meta_my_field": "meta:my.field", "meta_time": "meta:time",
        "meta_raw": "meta:raw", hashed: "meta:件名"}  # fmt: skip
    rows = db.sql("SELECT event_id, my_field, meta_my_field, meta_time, n, f, big, tags, config, ok FROM other_ci_build ORDER BY time")["rows"]  # fmt: skip
    assert rows == [  # widened columns re-encoded old rows from `raw` exactly like new ones
        {"event_id": "ci:1", "my_field": "a", "meta_my_field": None, "meta_time": "not the time", "n": "1", "f": 1, "big": "1", "tags": ["1", "2"], "config": None, "ok": True},
        {"event_id": "ci:2", "my_field": None, "meta_my_field": "b", "meta_time": None, "n": "abc", "f": 1.5, "big": "123456789012345678901234567890", "tags": ["x"], "config": '{"a":1}', "ok": None},
        {"event_id": "ci:3", "my_field": None, "meta_my_field": None, "meta_time": None, "n": "3", "f": None, "big": None, "tags": [], "config": None, "ok": None},
    ]  # fmt: skip
    columns = {c["name"]: c["description"] for c in db.schema("other_ci_build")["columns"]}
    assert "original meta key 'my.field'" in columns["meta_my_field"]
    assert "original meta key 'time'" in columns["meta_time"]


def test_an_unknown_type_never_lands_in_a_known_table(db):
    assert db.sql("SELECT event_type, count() AS n FROM jira_changes GROUP BY event_type")["rows"] == [{"event_type": "jira.change", "n": 3}]  # fmt: skip
    assert db.sql("SELECT event_id FROM other_jira_changes")["rows"] == [{"event_id": "x:1"}]
    assert db.schema("other_jira_changes")["system"] == "unrecognised event type"


def test_rebuild_is_byte_identical_and_a_resumed_build_matches_a_straight_one(store, tmp_path):
    again = tmp_path / "again"
    RawStore().build(iter(EVENTS), again)
    killed = tmp_path / "killed"
    first = RawStore()
    first.kill_after = "2026-01"
    with pytest.raises(RuntimeError, match="killed after 2026-01"):
        first.build(iter(EVENTS), killed)
    RawStore().build(iter(EVENTS), killed)
    for ckpt in ("2026-01", "2026-02"):
        expected = hashes(store / "export" / ckpt)
        assert hashes(again / "export" / ckpt) == expected
        assert hashes(killed / "export" / ckpt) == expected
    RawStore().build(iter(EVENTS), killed)  # a finished store stays as it is
    assert json.loads((killed / "build-state.json").read_text())["checkpoints"] == ["2026-01", "2026-02"]  # fmt: skip


# ---- time freeze --------------------------------------------------------------------------------


def frozen(export: Path, checkpoint: str) -> None:
    """No value of any DateTime column and no `raw.time` at or after the boundary."""
    boundary = stream.boundary(checkpoint)
    schema = json.loads((export / "schema.json").read_text())["tables"]
    with query.open(export) as found:
        for table, columns in schema.items():
            times = [c for c, kind in columns if "DateTime" in kind]
            checks = ", ".join(f"countIf(`{c}` >= '{boundary}') AS `{c}`" for c in times)
            row = found.sql(f"SELECT {checks}, countIf(JSONExtractString(raw, 'time') >= '{boundary}') AS raw_time FROM {table}")["rows"][0]  # fmt: skip
            assert set(row.values()) == {0}, (checkpoint, table, row)
            assert "time" in row and len(row) == len(times) + 1


def test_an_export_holds_nothing_at_or_after_its_boundary(store):
    for ckpt in ("2026-01", "2026-02"):
        frozen(store / "export" / ckpt, ckpt)
    january = store / "export/2026-01"
    assert all(LATER not in p.read_text() for p in january.rglob("*") if p.suffix in (".json", ".jsonl", ".md"))  # fmt: skip
    with query.open(january) as db:
        assert db.sql(f"SELECT count() AS n FROM other_ci_build WHERE raw LIKE '%{LATER}%'")["rows"] == [{"n": 0}]  # fmt: skip
        assert db.sql("SELECT argMax(to_value, (at, history_id, item_index)) AS v FROM jira_changes WHERE issue_key = 'KAFKA-1' AND field = 'status'")["rows"] == [{"v": "Patch Available"}]  # fmt: skip


@real
def test_real_exports_are_frozen_on_every_time_column():
    for ckpt in ("2026-01", "2026-02", "2026-03", "2026-04", "2026-05"):
        frozen(W2026 / "export" / ckpt, ckpt)


# ---- the statement guard ------------------------------------------------------------------------

TABLES = ["jira_changes", "other_ci_build", "other_jira_changes"]
REJECTED = [
    # comments anywhere: they once hid the appended limits
    "SELECT getSetting('readonly') AS ro --", "SELECT 1 AS a -- x", "SELECT 1 #", "SELECT 1 # x",
    "SELECT 1 /* x */", "SELECT /* x */ 1", "SELECT 1 --\n", "SELECT * FROM file('/etc/hostname', LineAsString) --",
    f"SELECT * FROM file('{PRIVATE}', LineAsString) #", "SELECT 1 /*",
    "WITH 1 AS x INSERT INTO FUNCTION file('/tmp/raw-pwn.csv', 'CSV', 'a Int8') SELECT x --",
    # DESCRIBE / SHOW only of the export
    "DESCRIBE file('/etc/passwd', TSVWithNames)", f"DESCRIBE TABLE file('{PRIVATE}', TSVWithNames)",
    "DESCRIBE url('http://example.com/', CSV)", "DESC system.tables", "DESCRIBE jira_changes, other_ci_build",
    "SHOW DATABASES", "SHOW CREATE TABLE jira_changes", "SHOW TABLES FROM system",
    # table functions anywhere, also hidden in subqueries, CTEs, joins, IN
    "SELECT * FROM file('/etc/hostname', LineAsString)", "SELECT * FROM numbers(10)",
    "SELECT * FROM numbers_mt(1e12)", "SELECT * FROM generateRandom('a Int8')", "SELECT * FROM zeros(10)",
    "SELECT (SELECT count() FROM url('http://example.com', CSV))",
    "WITH q AS (SELECT * FROM file('/etc/hostname', LineAsString)) SELECT * FROM q",
    "SELECT * FROM jira_changes, file('/etc/x', CSV)", "SELECT * FROM jira_changes AS a JOIN remote('h', db.t) AS b ON 1",
    "SELECT 1 WHERE 1 IN (SELECT * FROM s3('http://x/y', CSV))", "SELECT * FROM merge('default', '.*')",
    "SELECT * FROM executable('ls', TSV, 'a String')", "SELECT * FROM input('a Int8')",
    "SELECT * FROM dictionary('d')", "SELECT * FROM cluster('c', system.one)", "SELECT * FROM view(SELECT 1)",
    # scalar functions that reach outside, however spelt
    "SELECT file('/etc/hostname')", "SELECT `file`('/etc/hostname')", 'SELECT "file"(\'/etc/hostname\')',
    "SELECT FILE('/etc/hostname')", "SELECT hostName()", "SELECT fqdn()", "SELECT getSetting('readonly')",
    "SELECT filesystemAvailable()", "SELECT dictGet('d', 'a', 1)", "SELECT sleepEachRow(1) FROM jira_changes",
    "SELECT arrayMap(x -> file(x), ['/etc/hostname'])",
    # other databases
    "SELECT * FROM system.tables", "SELECT * FROM `system`.`tables`", "SELECT * FROM information_schema.tables",
    "SELECT count() FROM default.jira_changes", "SELECT name FROM system.settings",
    # writes, settings, output
    "INSERT INTO jira_changes (event_id) VALUES ('x')", "ALTER TABLE jira_changes DELETE WHERE 1",
    "DROP TABLE jira_changes", "SELECT 1; DROP TABLE jira_changes", "SELECT 1;;SELECT 2",
    "CREATE TABLE x (a Int8) ENGINE = Memory", "SET readonly = 0", "USE system", "OPTIMIZE TABLE jira_changes",
    "SYSTEM DROP DNS CACHE", "EXPLAIN SELECT 1", "SELECT 1 SETTINGS readonly = 0",
    "SELECT 1 SETTINGS max_execution_time = 1000", "SELECT 1 INTO OUTFILE '/tmp/x'", "SELECT 1 FORMAT CSV",
    "SELECT $$abc$$", "SELECT {x:String}", "SELECT * FROM unknown_table", "",
]  # fmt: skip
ACCEPTED = [
    "SELECT 1", "SELECT '--;#/*' AS s", "SELECT count() FROM jira_changes;",
    "WITH x AS (SELECT issue_key FROM jira_changes) SELECT count() FROM x",
    "SELECT extract(DAY FROM time) AS d, trim(BOTH ' ' FROM title) AS t FROM jira_changes",
    "SELECT a.event_id FROM jira_changes AS a INNER JOIN jira_changes b ON a.event_id = b.event_id",
    "SELECT e FROM jira_changes ARRAY JOIN entities AS e", "SELECT * FROM (SELECT 1 AS a) AS q",
    "SELECT count() FROM jira_changes WHERE issue_key IN (SELECT issue_key FROM jira_changes)",
    "DESCRIBE TABLE jira_changes", "DESC `other_ci_build`", "SHOW TABLES",
]  # fmt: skip


@pytest.mark.parametrize("statement", REJECTED)
def test_the_guard_rejects(statement):
    with pytest.raises(guard.QueryError):
        guard.check(statement, TABLES)


@pytest.mark.parametrize("statement", ACCEPTED)
def test_the_guard_admits_ordinary_reads_and_runs_them_unchanged(statement):
    assert guard.check(statement, TABLES) == statement.strip().rstrip(";").strip()


def test_hostile_statements_fail_through_the_api_and_the_export_is_untouched(store, db):
    export = store / "export/2026-02"
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in export.rglob("*") if p.is_file()}  # fmt: skip
    target = Path("/tmp/raw-pwn.csv")
    target.unlink(missing_ok=True)
    for statement in REJECTED:
        found = db.sql(statement)
        assert found["error"] and found["rows"] == [], statement
    assert not target.exists()
    assert db.sql("SELECT ';' AS s")["rows"] == [{"s": ";"}]
    assert {r["name"] for r in db.sql("SHOW TABLES")["rows"]} == {"jira_changes", "other_ci_build", "other_jira_changes"}  # fmt: skip
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in export.rglob("*") if p.is_file()}  # fmt: skip


def test_readonly_and_limits_hold_in_the_engine_even_without_the_guard(db):
    """What the engine enforces by itself, statement text bypassing `guard.check`."""
    settings = query.limits(7, 1000, 3)
    names = ("readonly", "max_execution_time", "timeout_overflow_mode", "max_result_rows",
             "max_result_bytes", "result_overflow_mode", "max_memory_usage", "max_bytes_in_join",
             "join_overflow_mode", "max_threads", "cast_string_to_date_time_mode")  # fmt: skip
    found = db.backend.select("SELECT " + ", ".join(f"toString(getSetting('{n}')) AS {n}" for n in names), settings)  # fmt: skip
    assert found["rows"] == [{
        "readonly": "1", "max_execution_time": "3", "timeout_overflow_mode": "throw",
        "max_result_rows": "8", "max_result_bytes": "1002000", "result_overflow_mode": "break",
        "max_memory_usage": "1000000000", "max_bytes_in_join": "250000000",
        "join_overflow_mode": "throw", "max_threads": "4", "cast_string_to_date_time_mode": "best_effort",
    }]  # fmt: skip
    # `limit` is taken from the connection URL by chdb itself (getSetting shows 0), so it is read
    # back by its effect: 27 rows asked for, max_rows + 1 = 8 come out of the engine
    many = "SELECT a.event_id FROM jira_changes a CROSS JOIN jira_changes b CROSS JOIN other_ci_build c"
    assert len(db.backend.select(many, settings)["rows"]) == 8
    for statement in ("SELECT 1 SETTINGS readonly = 0", "SELECT 1 SETTINGS max_execution_time = 100",
                      "SELECT * FROM file('/etc/hostname', LineAsString)",
                      "INSERT INTO FUNCTION file('/tmp/raw-pwn.csv', 'CSV', 'a Int8') SELECT 1",
                      "CREATE TABLE x (a Int8) ENGINE = Memory", "SET readonly = 0"):  # fmt: skip
        with pytest.raises(Exception, match="READONLY|readonly"):
            db.backend.select(statement, settings)
    assert not Path("/tmp/raw-pwn.csv").exists()


# ---- lexer agreement: the guard reads literals and quoted identifiers as the engine does ---------

PIECES = ["a", "Z", "é", "ж", "日", "'", "''", "\\", "\\'", "\\\\", "\\n", "\\t", "\\0", "\\x41",
          "\\x7e", "\\xZZ", "\\x4", "\\x80", "\\A", "\\z", "\\N", "\\[", "\\b", "\\v", "\n", "\r",
          "\t", ";", "(", ")", "file(", "--", "#", "/*", " ", '"', '""', "`", "``", "AS s, "]  # fmt: skip
DOCUMENTED = {"\\\\": "\\", "\\'": "'", "\\n": "\n", "\\t": "\t", "\\r": "\r", "\\0": "\0",
              "\\b": "\b", "\\f": "\f", "\\a": "\a", "\\v": "\v"}  # fmt: skip


def cases(quote: str, count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return [quote + "".join(rng.choice(PIECES) for _ in range(rng.randint(1, 8))) + quote for _ in range(count)]  # fmt: skip


def one_token(text: str) -> bool:
    """By the engine's lexing rule (a backslash escapes the next character, a doubled quote is a
    quote): does the quoted token starting at text[0] end exactly at text[-1]?"""
    quote, i = text[0], 1
    while i < len(text):
        if text[i] == "\\":
            i += 2
        elif text[i] == quote and text[i + 1 : i + 2] == quote:
            i += 2
        elif text[i] == quote:
            return i == len(text) - 1
        else:
            i += 1
    return False


def decode(text: str) -> str | None:
    """The value by the documented rules (ClickHouse "Syntax", string literals): '' is ', a
    backslash before b f r n t 0 a v \\ ' gives that character, \\xHH the byte. The docs say any
    other escape gives the bare character, but the engine keeps its backslash: None there."""
    body, out, i = text[1:-1], [], 0
    while i < len(body):
        pair = body[i : i + 2]
        if pair == "''":
            out.append("'")
            i += 2
        elif pair in DOCUMENTED:
            out.append(DOCUMENTED[pair])
            i += 2
        elif re.fullmatch(r"\\x[0-7][0-9A-Fa-f]", body[i : i + 4]):
            out.append(chr(int(body[i + 2 : i + 4], 16)))
            i += 4
        elif body[i] in "\\'":
            return None
        else:
            out.append(body[i])
            i += 1
    return "".join(out)


def admitted(statement: str) -> bool:
    try:
        guard.check(statement, TABLES)
    except guard.QueryError:
        return False
    return True


def test_guard_and_engine_read_string_literals_alike(db):
    literals, accepted = cases("'", 2400, 2026), 0
    for literal in literals:
        statement = f"SELECT {literal} AS s"
        expected = decode(literal) if one_token(literal) else None
        if not admitted(statement):
            assert expected is None or guard.CONTROL.search(literal), literal  # not stricter than needed
            continue
        accepted += 1
        assert expected is not None, literal  # the guard took it as one documented literal …
        assert guard.tokens(statement) == [("word", "SELECT"), ("string", literal), ("word", "AS"), ("word", "s")]  # fmt: skip
        found = db.backend.select(statement, query.limits(5, 100_000, 5))  # … and so does the engine
        assert found["columns"] == [("s", "String")] and found["rows"] == [{"s": expected}], literal
    print(f"string literals: {len(literals)} generated, {accepted} accepted")
    assert len(literals) >= 2000 and accepted >= 200


@pytest.mark.parametrize("quote", ["`", '"'])
def test_guard_and_engine_read_quoted_identifiers_alike(db, quote):
    names, accepted = cases(quote, 1000, 2027), 0
    for name in names:
        statement = f"SELECT 1 AS {name}"
        if not admitted(statement):
            continue
        accepted += 1
        assert one_token(name) and "\\" not in name, name  # one identifier, no backslash escape
        assert guard.tokens(statement) == [("word", "SELECT"), ("number", "1"), ("word", "AS"), ("quoted", name)]  # fmt: skip
        decoded = name[1:-1].replace(quote * 2, quote)
        found = db.backend.select(statement, query.limits(5, 100_000, 5))
        assert found["columns"] == [(decoded, "UInt8")] and found["rows"] == [{decoded: 1}], name
    print(f"{quote}-identifiers: {len(names)} generated, {accepted} accepted")
    assert accepted >= 50


def test_input_hygiene():
    controls = [chr(c) for c in [*range(0, 9), 11, 12, *range(14, 32), 127]]
    for c in controls:
        for statement in (f"SELECT 1{c}", f"SELECT 'a{c}b' AS s", f"SELECT 1 AS `a{c}b`"):
            with pytest.raises(guard.QueryError, match="control character"):
                guard.check(statement, TABLES)
    for statement in ("SELECT 'ab", "SELECT 'ab\\'", "SELECT `ab", 'SELECT "ab', "SELECT 'a''",
                      "SELECT 'a' 'b'", "SELECT 'a'x'b'", "SELECT 'a'('b')", "SELECT 1 AS `a` b `c`",
                      "SELECT é", "SELECT 1 AS é", "SELECT ｓ", "SELECT 1", "SELECT 1 ",
                      "SELECT 1 AS `a\\`b`", "SELECT '\\x4'", "SELECT '\\z'", "SELECT '\\N'"):  # fmt: skip
        with pytest.raises(guard.QueryError):
            guard.check(statement, TABLES)
    for c in "\t\n\r":
        guard.check(f"SELECT{c}1", TABLES)
    guard.check("SELECT 'ünïcødé 日本' AS `ключ`", TABLES)
    guard.check("SELECT 1" + " " * (guard.MAX_LENGTH - 8), TABLES)
    with pytest.raises(guard.QueryError, match="longer than 20000"):
        guard.check("SELECT 1" + " " * (guard.MAX_LENGTH - 7), TABLES)


# ---- the function allowlist (deny by default) ----------------------------------------------------

DENIED_PARTS = ("file", "url", "remote", "s3", "hdfs", "dict", "sleep", "host", "env", "server",
                "setting", "user", "database", "rand", "now", "today", "uuid", "fuzz", "generate",
                "executable", "catboost", "jdbc", "odbc", "mysql", "postgres", "mongo", "redis",
                "sqlite", "azure", "gcs", "cluster", "merge", "input", "view", "joinget")  # fmt: skip
ALLOWED_NAMES = [n for group in guard.ALLOW.values() for n in group]


def registry(db) -> tuple[set, set, set]:
    """The engine's own lists (functions incl. aliases, table functions, combinators),
    read through the backend directly, bypassing the guard."""
    read = lambda sql: {r["name"] for r in db.backend.select(sql, {"readonly": 1})["rows"]}  # noqa: E731
    return (read("SELECT name FROM system.functions"), read("SELECT name FROM system.table_functions"),
            read("SELECT name FROM system.aggregate_function_combinators"))  # fmt: skip


def spellings(function: str) -> set:
    return {function, function.lower(), function.upper(), function[:1].upper() + function[1:].lower(),
            function.swapcase(), f"`{function}`", f'"{function}"'}  # fmt: skip


def test_every_engine_function_not_on_the_allowlist_is_refused(db):
    functions, table_functions, combinators = registry(db)
    assert len(functions) > 1500 and len(table_functions) > 50 and "If" in combinators
    checked = accepted = 0
    for function in sorted((functions | table_functions) - set(ALLOWED_NAMES)):
        for spelt in spellings(function):
            for statement in (f"SELECT {spelt}(1)", f"SELECT count() FROM jira_changes WHERE {spelt}('x', 1) = 1",
                              f"SELECT * FROM {spelt}(1)", f"WITH q AS (SELECT {spelt}(1)) SELECT * FROM q"):  # fmt: skip
                checked += 1
                try:
                    guard.check(statement, TABLES)
                    accepted += 1
                    print("ACCEPTED", statement)
                except guard.QueryError:
                    pass
    assert accepted == 0 and checked > 40_000
    assert [n for n in ALLOWED_NAMES if n not in functions] == []  # no dead entries
    assert [n for n in ALLOWED_NAMES if n.lower() in {t.lower() for t in table_functions}] == []
    for base in guard.ALLOW["aggregates"]:  # combinators: only the five allowed ones
        for combinator in sorted(combinators):
            statement = f"SELECT {base}{combinator}(1)"
            if combinator in guard.COMBINATORS:
                guard.check(statement, TABLES)
            else:
                with pytest.raises(guard.QueryError):
                    guard.check(statement, TABLES)
    for statement in ("SELECT lowerIf('a', 1)", "SELECT hasArray([1], 1)", "SELECT fileIf(1)", "SELECT fileArrayIf(1)"):  # fmt: skip
        with pytest.raises(guard.QueryError):
            guard.check(statement, TABLES)


JUSTIFIED = {  # allowlisted names that contain a denied part, each with its reason
    "toDayOfMonth": "to-Day-Of…, not today(): day number of its argument",
    "toDayOfWeek": "to-Day-Of…, not today(): weekday of its argument",
    "toDayOfYear": "to-Day-Of…, not today(): day of year of its argument",
}


def test_the_allowlist_has_no_function_of_a_denied_category():
    found = {n for n in ALLOWED_NAMES for part in DENIED_PARTS if part in n.lower()}
    assert found == set(JUSTIFIED)
    assert len(ALLOWED_NAMES) == len(set(ALLOWED_NAMES))


def test_a_refused_function_is_named_with_the_closest_allowed_ones(db):
    error = db.sql("SELECT uniqExactt(issue_key) FROM jira_changes")["error"]
    assert error.startswith("function uniqExactt is not available; closest allowed: uniqExact")
    assert "sql_reference()" in error and len(error) < 500
    assert db.sql("SELECT aiGenerate('x')")["error"].startswith("function aiGenerate is not available")  # fmt: skip
    assert db.sql("SELECT now()")["error"].startswith("function now is not available")


COMPATIBLE = [
    "SELECT count() FROM jira_changes",
    "SELECT field, count() AS n FROM jira_changes GROUP BY field ORDER BY n DESC LIMIT 5",
    "SELECT issue_key, countIf(field = 'status') AS s FROM jira_changes GROUP BY issue_key ORDER BY s DESC LIMIT 3",
    "SELECT argMaxIf(to_value, (at, history_id, item_index), field = 'status') AS v FROM jira_changes WHERE issue_key = 'KAFKA-20020'",
    "SELECT groupArrayDistinct(field) AS f FROM jira_changes WHERE issue_key = 'KAFKA-20020'",
    "SELECT uniqExactIf(pr_number, action = 'merged') AS n, uniqExactArray(issues) AS i, sumArray(prs) AS p FROM github_pull_requests, (SELECT [1, 2] AS prs) AS x",
    "SELECT p.pr_number, count() AS reviews FROM github_pull_requests AS p INNER JOIN github_reviews AS r ON p.pr_number = r.pr_number WHERE p.action = 'merged' GROUP BY p.pr_number ORDER BY reviews DESC LIMIT 3",
    "SELECT pr_number, r.reviewer FROM github_pull_requests LEFT JOIN github_reviews AS r USING pr_number WHERE action = 'opened' LIMIT 3",
    "WITH merged AS (SELECT pr_number, author FROM github_pull_requests WHERE action = 'merged') SELECT author, count() AS n FROM merged GROUP BY author ORDER BY n DESC LIMIT 3",
    "SELECT (SELECT max(time) FROM mail_messages) AS last, (SELECT count() FROM jira_issues) AS issues",
    "SELECT count() FROM github_reviews WHERE pr_number IN (SELECT pr_number FROM github_pull_requests WHERE action = 'merged')",
    "SELECT count() FROM github_reviews WHERE pr_number NOT IN (SELECT pr_number FROM github_pull_requests WHERE action = 'merged')",
    "SELECT count() FROM jira_issues WHERE EXISTS (SELECT 1 FROM jira_comments)",
    "SELECT issue_key, at, row_number() OVER (PARTITION BY issue_key ORDER BY at) AS n FROM jira_changes WHERE field = 'status' LIMIT 5",
    "SELECT author, count() AS n, rank() OVER (ORDER BY count() DESC) AS r, dense_rank() OVER (ORDER BY count() DESC) AS d FROM github_pull_requests GROUP BY author LIMIT 5",
    "SELECT issue_key, to_value, lagInFrame(to_value) OVER (PARTITION BY issue_key ORDER BY at ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS before FROM jira_changes WHERE field = 'status' LIMIT 5",
    "SELECT issue_key, at, to_value FROM jira_changes WHERE field = 'status' ORDER BY issue_key, at DESC LIMIT 1 BY issue_key LIMIT 5",
    "SELECT i, count() AS n FROM github_pull_requests ARRAY JOIN issues AS i GROUP BY i ORDER BY n DESC LIMIT 3",
    "SELECT pr_number, i FROM github_pull_requests LEFT ARRAY JOIN issues AS i LIMIT 3",
    "SELECT CAST(pr_number AS String) AS s, CAST(pr_title AS Nullable(String)) AS t, pr_number::Int32 AS n FROM github_pull_requests LIMIT 1",
    "SELECT count() AS n FROM mail_messages WHERE body LIKE '%file(%' OR body LIKE '%url(%' OR body LIKE '%system.%' OR body LIKE '%--%' OR body LIKE '%#%' OR body LIKE '%;%'",
    "SELECT count() AS file, max(time) AS now, min(time) AS sleep, any(actor) AS hostName FROM jira_changes",
    "SELECT issue_key AS url, summary AS `remote` FROM jira_issues LIMIT 1",
    "SELECT count() FROM jira_changes WHERE issue_key = 'KAFKA-20020'",
    "SELECT JSONExtractString(raw, 'id') AS id, JSONExtract(raw, 'meta', 'field', 'Nullable(String)') AS f, JSON_VALUE(raw, '$.type') AS t FROM jira_changes LIMIT 2",
    "SELECT formatDateTime(at, '%Y-%m-%dT%H:%i:%S.%fZ') AS iso FROM jira_changes WHERE at IS NOT NULL LIMIT 1",
    "SELECT toStartOfMonth(time) AS m, dateDiff('day', min(time), max(time)) AS days FROM github_pull_requests GROUP BY m ORDER BY m",
    "SELECT count() FROM github_pull_requests WHERE time >= parseDateTime64BestEffort('2026-03-01T00:00:00Z') AND time < '2026-04-01T00:00:00Z'",
    "SELECT CASE WHEN action = 'merged' THEN 'm' WHEN action = 'closed' THEN 'c' ELSE 'o' END AS a, count() AS n FROM github_pull_requests GROUP BY a",
    "SELECT multiIf(state = 'approved', 1, state = 'commented', 2, 3) AS k, coalesce(reviewer, '?') AS r, ifNull(body, '') AS b FROM github_reviews LIMIT 3",
    "SELECT count() FROM github_pull_requests WHERE has(issues, 'KAFKA-20020') OR hasAny(kips, ['KIP-1071']) OR arrayExists(x -> startsWith(x, 'KAFKA-2'), issues)",
    "SELECT arrayStringConcat(arraySort(groupUniqArray(author)), ', ') AS a, arrayMap(x -> lower(x), ['A']) AS m, arrayFilter(x -> x != '', ['a', '']) AS f FROM github_pull_requests",
    "SELECT EXTRACT(YEAR FROM time) AS y, extract(title, 'KAFKA-[0-9]+') AS k FROM jira_changes LIMIT 1",
    "SELECT count() FROM github_pull_requests WHERE time >= toDateTime64('2026-05-01 00:00:00', 6, 'UTC') - INTERVAL 7 DAY",
    "SELECT count() FROM mail_messages WHERE match(subject, '(?i)\\\\[discuss\\\\]')",
    "SELECT quantile(0.5)(dateDiff('hour', time, now_col)) AS q, topK(3)(author) AS t FROM (SELECT time, time AS now_col, author FROM github_pull_requests)",
    "SELECT author FROM github_pull_requests WHERE action = 'merged' UNION ALL SELECT reviewer FROM github_reviews LIMIT 3",
    "SELECT DISTINCT action FROM github_pull_requests ORDER BY action NULLS LAST",
    "SELECT author, count() AS n FROM github_pull_requests GROUP BY author HAVING n > 5 ORDER BY n DESC",
    "SELECT issue_key, ngramDistanceCaseInsensitiveUTF8(summary, 'share group') AS d FROM jira_issues ORDER BY d LIMIT 3",
    "SELECT count() FROM mail_messages WHERE hasTokenCaseInsensitive(body, 'tombstone') OR multiSearchAnyCaseInsensitive(body, ['metadata version', 'rebalance'])",
    "SELECT `event_id`, \"issue_key\" FROM `jira_changes` LIMIT 1",
    "SELECT splitByNonAlpha(lower(summary)) AS w, position(summary, 'a') AS p, replaceRegexpAll(summary, '\\\\s+', ' ') AS s, lengthUTF8(summary) AS l FROM jira_issues LIMIT 1",
    "SELECT toYYYYMM(time) AS m, avgOrNull(line) AS a, sumOrDefault(line) AS s FROM github_review_comments GROUP BY m",
    "SELECT tupleElement((1, 'a'), 2) AS t, [1, 2, 3][2] AS e, map('a', 1)['a'] AS m",
]  # fmt: skip


@real
def test_ordinary_analytical_statements_are_admitted_and_run():
    with query.open(W2026 / "export/2026-05") as db:
        for statement in COMPATIBLE:
            found = db.sql(statement, max_rows=5)
            assert found["error"] is None, (statement, found["error"])
    assert len(COMPATIBLE) >= 40


LONG = "x" * 4_000
CROSS = "FROM mail_messages a CROSS JOIN mail_messages b"
REMOVED = {  # taken off the allowlist by the audit
    "countMatches": "one call over a long body ran 9.4-10.1 s against a 3 s time cap",
    "extractAll": "ran 6.1-6.3 s against a 3 s cap", "extractAllGroups": "ran 4.0-4.5 s against a 3 s cap",
    "editDistance": "O(n*m) per pair of texts", "levenshteinDistance": "O(n*m) per pair of texts",
    "multiMatchAny": "hyperscan pattern compilation is not interrupted by the time cap",
}  # fmt: skip
AMPLIFIERS = {  # allowlisted functions whose output can outgrow their input or whose cost is super-linear
    "replaceAll": f"SELECT sum(length(replaceAll(a.body, 'e', '{LONG}'))) AS n {CROSS}",
    "replaceRegexpAll": f"SELECT sum(length(replaceRegexpAll(a.body, '', '{LONG}'))) AS n {CROSS}",
    "replaceOne": f"SELECT sum(length(replaceOne(a.body, 'e', '{LONG}'))) AS n {CROSS}",
    "concat": f"SELECT sum(length(concat(a.body, '{LONG}', b.body))) AS n {CROSS}",
    "concatWithSeparator": f"SELECT sum(length(concatWithSeparator('{LONG}', a.raw, b.raw))) AS n {CROSS}",
    "formatDateTime": f"SELECT sum(length(formatDateTime(a.time, '{'%Y-%m-%d %H:%i:%S ' * 900}'))) AS n {CROSS}",
    "groupArray": f"SELECT length(groupArray(a.raw)) AS n {CROSS}",
    "groupConcat": f"SELECT length(groupConcat(a.raw)) AS n {CROSS}",
    "groupUniqArray": f"SELECT length(groupUniqArray(concat(a.raw, b.raw))) AS n {CROSS}",
    "arraySort": f"SELECT length(arraySort(groupArray(concat(a.raw, b.raw)))) AS n {CROSS}",
    "window groupArray": f"SELECT length(groupArray(a.raw) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)) AS n {CROSS}",
    "topK": f"SELECT length(topK(1000000000)(a.raw)) AS n {CROSS}",
    "arrayJoin splitByChar": f"SELECT count() {CROSS} ARRAY JOIN splitByChar(' ', a.body) AS w",
    "arrayJoin literal": f"SELECT count() {CROSS} ARRAY JOIN [{', '.join(map(str, range(3000)))}] AS x",
    "splitByRegexp": f"SELECT sum(length(splitByRegexp('', a.body))) AS n {CROSS}",
    "splitByWhitespace, arrayFlatten": f"SELECT sum(length(arrayFlatten([splitByWhitespace(a.body), splitByNonAlpha(b.body)]))) AS n {CROSS}",
    "ngramDistance": f"SELECT sum(ngramDistance(a.body, b.body)) AS n {CROSS}",
    "match": f"SELECT countIf(match(a.body, '(a+)+$')) AS n {CROSS}",
    "extract": f"SELECT sum(length(extract(a.body, '(a|e)+$'))) AS n {CROSS}",
    "regexpExtract": f"SELECT sum(length(regexpExtract(a.body, '(.*)(.*)$', 1))) AS n {CROSS}",
    "multiSearchAny": f"SELECT sum(multiSearchAny(a.body, [{', '.join(f'{chr(39)}n{i}{chr(39)}' for i in range(1000))}])) AS n {CROSS}",
    "JSONExtractKeysAndValues": f"SELECT sum(length(JSONExtractKeysAndValues(a.raw, 'String'))) AS n {CROSS}",
    "arrayConcat": f"SELECT sum(length(arrayConcat(g, g, g, g))) AS n FROM (SELECT groupArray(a.raw) AS g {CROSS} GROUP BY b.event_id)",
    "nested arrayMap": "SELECT length(arrayMap(x -> arrayMap(y -> concat(x, y), g), g)) AS n FROM (SELECT groupArray(raw) AS g FROM github_review_comments)",
    "arrayCount arrayExists": "SELECT arrayCount(x -> arrayExists(y -> y = x, g), g) AS n FROM (SELECT groupArray(a.event_id) AS g FROM github_review_comments a CROSS JOIN github_reviews b)",
}  # fmt: skip
STOPPED = ("TIMEOUT_EXCEEDED", "MEMORY_LIMIT_EXCEEDED", "SET_SIZE_LIMIT_EXCEEDED", "TOO_MANY_ROWS_OR_BYTES",
           "TOO_LARGE_ARRAY_SIZE", "TOO_LARGE_STRING_SIZE", "ARGUMENT_OUT_OF_BOUND", "CANNOT_COMPILE_REGEXP",
           "TOO_MANY_ARGUMENTS_FOR_FUNCTION")  # fmt: skip


@real
@pytest.mark.parametrize("function", list(AMPLIFIERS))
def test_amplifying_functions_are_bounded_by_the_engine_caps(function):
    """Each allowlisted function that can outgrow its input or cost super-linearly, driven as
    hard as one statement can (cross joins, 4 kB literals, quadratic lambdas): the engine stops
    it (time, memory, size) or its result is small — within the time cap plus a margin and
    without the process growing past the memory cap."""
    statement = AMPLIFIERS[function]
    with query.open(W2026 / "export/2026-05") as db:
        assert guard.check(statement, db.tables)
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        started = time.monotonic()
        found = db.sql(statement, max_rows=10, max_bytes=100_000, timeout_s=3)
        spent = time.monotonic() - started
        grown = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before
    print(f"{function}: {spent:.1f} s, +{grown // 1024} MB, {(found['error'] or 'ok')[:60]}")
    assert spent < 5, (function, spent, found["error"])  # the 3 s time cap + 2 s
    assert grown < 1_300_000, (function, grown)  # KB: the 1 GB query cap plus slack
    assert found["error"] is None or found["error"].startswith(STOPPED), found["error"]
    if found["error"] is None:
        assert sum(len(json.dumps(r, default=str)) for r in found["rows"]) <= 100_000


def test_functions_removed_by_the_audit_are_refused():
    for function in REMOVED:
        assert function not in ALLOWED_NAMES
        with pytest.raises(guard.QueryError, match=f"function {function} is not available"):
            guard.check(f"SELECT {function}('a', 'b')", TABLES)


def test_harden_sets_os_limits_only_when_called(tmp_path, store):
    code = ("import resource; from harnext_eval.e2.v2.stores.raw import query; "
            "before = resource.getrlimit(resource.RLIMIT_CORE); caps = query.harden(cpu_seconds=600); "
            f"db = query.open({str(store / 'export/2026-02')!r}); rows = db.sql('SELECT count() AS n FROM jira_changes')['rows']; db.close(); "
            "print(rows == [{'n': 3}], resource.getrlimit(resource.RLIMIT_CORE)[0], resource.getrlimit(resource.RLIMIT_CPU)[0], resource.getrlimit(resource.RLIMIT_DATA)[0])")  # fmt: skip
    found = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path)
    assert found.returncode == 0, found.stderr[-2000:]
    assert found.stdout.split()[-4:] == ["True", "0", "600", str(8 << 30)]
    import resource

    assert resource.getrlimit(resource.RLIMIT_CPU)[0] != 600  # importing query applied nothing


def test_messages_help_a_model_recover(db):
    assert db.sql("SELECT from_value AS from, to_value AS to FROM jira_changes")["error"] is None
    assert "quote it with backticks" in db.sql("SELECT 1 AS x FROM jira_changes WHERE set = 1")["error"]
    lag = db.sql("SELECT lag(to_value) FROM jira_changes")["error"]
    assert lag.startswith("function lag is not available: use lagInFrame(x) OVER")
    assert "compare with explicit dates" in db.sql("SELECT today()")["error"]
    unknown = db.sql("SELECT * FROM jira_change LIMIT 1")["error"]
    assert unknown.startswith("unknown table 'jira_change'; did you mean jira_changes")
    iso_in = db.sql("SELECT count() FROM jira_changes WHERE dateDiff('second', at, '2026-01-06T00:00:00Z') > 0")["error"]  # fmt: skip
    assert iso_in and "parseDateTime64BestEffort" in iso_in


def test_percent_never_hides_a_small_share():
    assert [catalog.percent(v) for v in (None, 0, 1, 6 / 8382, 0.000001, 0.5, 0.99999999, 0.12345)] == [
        "-", "0", "100", "0.07", "<0.01", "50", ">99.99", "12.35"]  # fmt: skip


def test_a_title_spanning_lines_does_not_hide_the_facts():
    value = "Created PR:\r\nhttps://example.org\r\n\r\n - point one\r\n - point two"
    facts = [("issue", "KAFKA-7 — Fix it"), ("event", "field change"), ("by", "Rita"), ("at", "2026-01-06T00:00:00.500000Z"), ("field", "Comment"), ("from", value), ("to", "(none)")]  # fmt: skip
    title = f"KAFKA-7 Comment: {value} → (none)"
    found = sources.row(event("jira:7:change:9:0", "2026-01-06T00:00:00Z", "jira.change", {"issue": "KAFKA-7", "field": "Comment", "from": value, "to": None}, stream.page(title, facts), ["issue:KAFKA-7"], title=title))  # fmt: skip
    assert (found["at"], found["author"], found["issue_summary"], found["body"]) == ("2026-01-06T00:00:00.500000Z", "Rita", "Fix it", None)  # fmt: skip
    assert sources.mentioned("See kip-1071 and KAFKA-1, KIP-1071, kafka-2") == {"body_issues": ["KAFKA-1", "KAFKA-2"], "body_kips": ["KIP-1071"]}  # fmt: skip


def test_limits_and_actionable_errors(db):
    many = "SELECT a.event_id FROM jira_changes a CROSS JOIN jira_changes b CROSS JOIN other_ci_build c"
    capped = db.sql(many, max_rows=5)
    assert len(capped["rows"]) == 5 and capped["truncated"] and capped["error"] is None
    whole = db.sql("SELECT event_id FROM jira_changes", max_rows=3)
    assert len(whole["rows"]) == 3 and not whole["truncated"]
    small = db.sql("SELECT raw FROM other_ci_build", max_bytes=300)
    assert small["truncated"] and "note" in small and len(small["rows"]) < 3
    unknown = db.sql("SELECT nosuch FROM jira_changes")["error"]
    assert unknown.startswith("UNKNOWN_IDENTIFIER") and "Valid columns: jira_changes(" in unknown
    assert "to_value" in unknown and "other_ci_build(" not in unknown
    assert "tables: jira_changes, other_ci_build" in db.sql("SELECT * FROM nope")["error"]
    iso = "SELECT count() AS n FROM jira_changes WHERE time >= '2026-01-06T00:00:00Z' AND at < '2026-02-03T00:00:00.4Z'"  # fmt: skip
    assert db.sql(iso)["rows"] == [{"n": 2}]
    record = db.get("ci:2")
    assert record["source"] == "other_ci_build" and record["meta"]["n"] == "abc"
    assert db.get("missing") is None


@real
def test_explosive_and_huge_queries_are_stopped_by_the_engine():
    with query.open(W2026 / "export/2026-05") as db:
        started = time.monotonic()
        slow = db.sql("SELECT count() FROM github_review_comments a CROSS JOIN github_review_comments b CROSS JOIN github_reviews c", timeout_s=2)  # fmt: skip
        assert slow["error"].startswith("TIMEOUT_EXCEEDED") and time.monotonic() - started < 4
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        huge = db.sql("SELECT a.raw, a.text, b.raw FROM mail_messages a CROSS JOIN mail_messages b", max_rows=100_000, max_bytes=1_000_000, timeout_s=20)  # fmt: skip
        grown = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before
        assert huge["error"] is None and huge["truncated"] and grown < 400_000, grown  # KB
        assert sum(len(json.dumps(r, ensure_ascii=False, default=str)) for r in huge["rows"]) <= 1_000_000
        assert db.sql("SELECT count() AS n FROM github_pull_requests WHERE time >= '2026-03-01T00:00:00Z'")["rows"][0]["n"] > 0  # fmt: skip


# ---- scratch and imports -------------------------------------------------------------------------


def test_scratch_dirs_are_removed_on_failure_close_and_for_dead_processes(store):
    stale = Path(tempfile.gettempdir()) / f"{query.PREFIX}999999999-x"
    stale.mkdir(exist_ok=True)
    (stale / ".lock").write_text("")  # a lock nobody holds: its owner is gone
    before = scratches()
    first = query.open(store / "export/2026-01")
    assert not stale.exists() and len(scratches() - before) == 1
    with pytest.raises(RuntimeError, match="close it first"):
        query.open(store / "export/2026-02")
    assert len(scratches() - before) == 1
    first.close()
    assert scratches() == before
    with pytest.raises(FileNotFoundError):
        query.open(store / "export/no-such-checkpoint")
    assert scratches() == before


def test_the_sweep_never_removes_the_scratch_of_a_live_process(tmp_path):
    """A live owner holds the flock of its `.lock` (whatever its pid namespace): kept. A dir
    without a lock file is kept too; one whose lock is free is removed."""
    held, free, unlocked = (tmp_path / f"{query.PREFIX}{n}" for n in ("1-held", "2-free", "3-nolock"))
    for path in (held, free, unlocked):
        path.mkdir()
    (free / ".lock").write_text("")
    owner = subprocess.Popen([sys.executable, "-c", "import fcntl, os, sys, time; p = sys.argv[1]; fd = os.open(p + '/.lock.new', os.O_CREAT | os.O_RDWR); "
                              "fcntl.flock(fd, fcntl.LOCK_EX); os.replace(p + '/.lock.new', p + '/.lock'); print('locked', flush=True); time.sleep(60)", str(held)],
                             stdout=subprocess.PIPE, text=True)  # fmt: skip
    try:
        assert owner.stdout.readline().strip() == "locked"
        query.sweep(tmp_path)
        assert held.exists() and unlocked.exists() and not free.exists()
    finally:
        owner.kill()
        owner.wait()
    query.sweep(tmp_path)  # the owner is dead: the kernel released its lock
    assert not held.exists() and unlocked.exists()


def test_harden_lowers_hard_limits_and_cleans_up_on_sigterm(tmp_path, store):
    code = ("import os, resource, signal, sys; from pathlib import Path; from harnext_eval.e2.v2.stores.raw import query; "
            "query.harden(memory_bytes=6 << 30, cpu_seconds=900, file_bytes=2 << 30); "
            f"db = query.open({str(store / 'export/2026-02')!r}); print(db.scratch, flush=True); "
            "print(resource.getrlimit(resource.RLIMIT_DATA), resource.getrlimit(resource.RLIMIT_FSIZE), resource.getrlimit(resource.RLIMIT_CPU), flush=True); "
            "os.kill(os.getpid(), signal.SIGTERM)")  # fmt: skip
    found = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path)
    scratch, limits = found.stdout.strip().splitlines()[:2]
    assert found.returncode == 128 + 15, found.stderr[-1500:]
    assert limits == f"({6 << 30}, {6 << 30}) ({2 << 30}, {2 << 30}) (900, 910)"
    assert not Path(scratch).exists()  # removed by the SIGTERM handler harden() installed


def test_the_read_path_imports_only_its_own_modules(tmp_path, store):
    """As in the trial image: the v2 tree with empty harnext_eval/e2 package files."""
    root = tmp_path / "pkg/harnext_eval"
    shutil.copytree(ROOT / "src/harnext_eval/e2/v2", root / "e2/v2", ignore=shutil.ignore_patterns("__pycache__"))  # fmt: skip
    (root / "__init__.py").write_text("")
    (root / "e2/__init__.py").write_text("")
    code = (f"import sys; sys.path.insert(0, {str(tmp_path / 'pkg')!r}); from harnext_eval.e2.v2.stores.raw import query; "
            f"db = query.open({str(store / 'export/2026-02')!r}); assert db.sql('SELECT count() AS n FROM jira_changes')['rows'] == [{{'n': 3}}]; db.close(); "
            "print(sorted(m for m in sys.modules if m.startswith('harnext')))")  # fmt: skip
    found = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path, env={**os.environ, "PYTHONPATH": ""})  # fmt: skip
    assert found.returncode == 0, found.stderr[-2000:]
    modules = eval(found.stdout.strip().splitlines()[-1])  # noqa: S307 - our own printed list
    assert modules == ["harnext_eval", "harnext_eval.e2", "harnext_eval.e2.v2", "harnext_eval.e2.v2.stores",
                       "harnext_eval.e2.v2.stores.raw", "harnext_eval.e2.v2.stores.raw.backend",
                       "harnext_eval.e2.v2.stores.raw.guard", "harnext_eval.e2.v2.stores.raw.query",
                       "harnext_eval.e2.v2.stores.raw.schema"]  # fmt: skip


# ---- catalog ------------------------------------------------------------------------------------


def examples_run(export: Path) -> int:
    """Every worked query of every source runs and returns a non-empty first value."""
    count = 0
    with query.open(export) as db:
        for table in db.tables:
            for example in db.schema(table)["examples"]:
                found = db.sql(example["sql"], max_rows=5)
                assert found["error"] is None and found["rows"], (table, example)
                assert next(iter(found["rows"][0].values())) not in (None, "", 0, []), (table, example)
                count += 1
    return count


def test_catalog_is_small_describes_every_column_and_its_queries_run(store, db):
    root = store / "export/2026-02/catalog"
    assert {p.name for p in root.iterdir()} == {"SOURCES.md", "sources.json", "catalog.json", "jira_changes.md", "other_ci_build.md", "other_jira_changes.md"}  # fmt: skip
    assert "sql" not in json.loads((root / "catalog.json").read_text())  # rendered from the guard
    assert db.sql_reference() == guard.summary() and "intDiv(dateDiff('second', a, b), 86400)" in db.sql_reference()  # fmt: skip
    summary = db.catalog()
    assert [s["table"] for s in summary["sources"]] == ["jira_changes", "other_ci_build", "other_jira_changes"]  # fmt: skip
    assert summary["sources"][0]["rows"] == 3 and summary["sources"][0]["system"] == "JIRA (issues.apache.org)"  # fmt: skip
    columns = {c["name"]: c for c in db.schema("jira_changes")["columns"]}
    assert columns["to_value"]["examples"][0] in {"In Progress", "Patch Available", "Resolved"}
    assert "`actor` is set on every row" in columns["author"]["description"] and "UTC" in columns["time"]["description"]  # fmt: skip
    auto = {c["name"]: c for c in db.schema("other_ci_build")["columns"]}["later_only"]
    assert auto["description"].startswith("Auto-detected: original meta key 'later_only'") and "ci:3" in auto["description"]  # fmt: skip
    assert all(c["description"] for t in db.tables for c in db.schema(t)["columns"])
    assert "2026-03-01T00:00:00Z" in (root / "SOURCES.md").read_text()
    with pytest.raises(KeyError, match="sources: jira_changes, other_ci_build"):
        db.schema("nope")
    db.close()  # one export per process at a time
    assert examples_run(store / "export/2026-02") >= 1


def flat(value):
    if isinstance(value, dict):
        return [v for item in value.values() for v in flat(item)]
    if isinstance(value, list):
        return [v for item in value for v in flat(item)]
    return [value]


def identifying(literal: str) -> bool:
    """A literal that names one thing (a key, number, path, login-with-digits, a title); dates
    are checked against the boundary instead."""
    text = literal.strip("%")
    if re.fullmatch(r"\d{4}-\d\d-\d\d( \d\d:\d\d:\d\d)?", text):
        return False
    return any(c.isdigit() for c in text) or any(c in text for c in "/@.") or len(text.split()) >= 3


@real
def test_worked_queries_use_this_exports_data_never_a_benchmark_task():
    public, private = oracle.load(BENCH)
    prompts = " ".join(t["prompt"] for t in public).casefold()
    # People are exempt from the gold check (not from the prompt check): every active committer
    # is the gold answer of some approver / reviewer / assignee / starter task.
    people = {"pr_approvers", "top_reviewer_for_author", "kip_thread", "assignee_at_snapshot", "assignment_history"}  # fmt: skip
    gold = {str(v).casefold() for t in private.values() if t.get("family") not in people for v in flat(t.get("expected", {}).get("value")) if v is not None and len(str(v)) >= 4}  # fmt: skip
    for export in [W2026 / "export" / c for c in ("2026-01", "2026-02", "2026-03", "2026-04", "2026-05")] + [SMOKE / "export/2023-02"]:  # fmt: skip
        assert examples_run(export) >= 18, export
        boundary = json.loads((export / "schema.json").read_text())["boundary"]
        doc = json.loads((export / "catalog/catalog.json").read_text())
        for table in doc["tables"].values():
            for example in table["examples"]:
                sql = example["sql"]
                strings = [s[1:-1] for kind, s in guard.tokens(sql) if kind == "string"]
                numbers = [s for kind, s in guard.tokens(sql) if kind == "number" and len(s) >= 4]
                for literal in [s for s in strings if identifying(s)] + numbers:
                    assert literal.strip("%").casefold() not in prompts, (export, literal)
                    assert literal.strip("%").casefold() not in gold, (export, literal)
                for literal in strings:  # dates never beyond the boundary
                    if literal[:4].isdigit() and literal[4:5] == "-":
                        assert literal <= boundary.replace("T", " ").rstrip("Z"), (export, literal)


@real
def test_column_examples_are_formats_not_busy_entities():
    """Regression check for the example rule: enumerations show their most frequent values,
    every other column its 3 earliest distinct values (by event time) — recomputed here. An
    identifying example that also occurs in a benchmark prompt may only be one of those
    earliest values (chance of time, not popularity), and never a non-person gold value."""
    public, private = oracle.load(BENCH)
    prompts = " ".join(t["prompt"] for t in public).casefold()
    people = {"pr_approvers", "top_reviewer_for_author", "kip_thread", "assignee_at_snapshot", "assignment_history"}  # fmt: skip
    gold = {str(v).casefold() for t in private.values() if t.get("family") not in people for v in flat(t.get("expected", {}).get("value")) if v is not None and len(str(v)) >= 4}  # fmt: skip
    collisions = set()
    for ckpt in ("2026-02", "2026-05"):
        export = W2026 / "export" / ckpt
        doc = json.loads((export / "catalog/catalog.json").read_text())
        with query.open(export) as db:
            for table, entry in doc["tables"].items():
                for column in entry["columns"]:
                    if column["name"] in sources.ENUMERATIONS or not column["examples"] or column["name"] in ("raw", "text", "body", "description", "message"):  # fmt: skip
                        continue
                    kind, name = column["type"], column["name"]
                    value = f"arrayJoin(`{name}`)" if kind.startswith("Array(") else f"`{name}`"
                    earliest = db.sql(f"SELECT toString(v) AS v FROM (SELECT {value} AS v, time, event_id FROM {table} WHERE NOT ({catalog.empty(name, kind)})) GROUP BY v ORDER BY min((time, event_id)), v LIMIT 3")["rows"]  # fmt: skip
                    assert column["examples"] == [catalog.short(r["v"]) for r in earliest], (ckpt, table, name)
                    for example in column["examples"]:
                        text = example.rstrip("…").casefold()
                        if identifying(example) and len(text) >= 5 and text in prompts:
                            collisions.add((table, name, example))
                        assert not (identifying(example) and text in gold), (ckpt, table, name, example)
    print("examples also named in a prompt (earliest values, by chance):", sorted(collisions))
    assert len(collisions) <= 6


@real
def test_structured_links_run_and_return_rows():
    with query.open(W2026 / "export/2026-05") as db:
        links = db.catalog()["links"]
        assert len(links) >= 20
        for link in links:
            found = db.sql(link["sql"])
            assert found["error"] is None and found["rows"][0]["pairs"] > 0, link
            for end in ("from", "to"):
                assert link[end]["column"] in db.tables[link[end]["table"]]
                assert link in db.schema(link[end]["table"])["links"]
        shapes = {(x["from"]["table"], x["from"]["column"]): x["value_shape"] for x in links}
        assert shapes[("github_pull_requests", "issues")] == "KAFKA-<n>"
        assert shapes[("github_reviews", "pr_number")] == "<n>"
        assert shapes[("mail_messages", "thread_id")] == "<text>@<text>"
        assert shapes[("github_pull_requests", "kips")] == "KIP-<n>"
        assert "actor" in {s["table"]: s for s in db.catalog()["sources"]}["jira_changes"]["key_columns"]
        questions = [e["question"] for e in db.schema("github_pull_requests")["examples"]]
        assert any(q.startswith("Rows of one action vs all rows (1/2)") for q in questions)
        assert any(q.startswith("Rows of one action vs all rows (2/2)") for q in questions)
    assert catalog.shape(["KAFKA-1", "KAFKA-22", "KAFKA-333"]) == "KAFKA-<n>"
    assert catalog.shape(["a1b2@x.org", "zz@mail.example.com"]) == "<text>@<text>"


@real
def test_body_mentions_the_multiline_rows_and_window_notes():
    with query.open(W2026 / "export/2026-05") as db:
        mentions = db.sql("SELECT countIf(has(kips, 'KIP-1071')) AS subject, countIf(has(body_kips, 'KIP-1071')) AS body FROM mail_messages")["rows"][0]  # fmt: skip
        assert mentions["body"] > 10 * max(mentions["subject"], 1)
        for table in ("mail_messages", "jira_comments", "github_pr_comments", "github_review_comments", "github_reviews"):  # fmt: skip
            assert {"body_issues", "body_kips"} <= set(db.tables[table]), table
        repaired = db.sql("SELECT countIf(at IS NULL OR author IS NULL OR issue_summary IS NULL) AS broken, count() AS n FROM jira_changes WHERE field = 'Comment'")["rows"][0]  # fmt: skip
        assert repaired["n"] >= 6 and repaired["broken"] == 0
        notes = {s["table"]: s["notes"] for s in db.catalog()["sources"]}
    assert re.search(r"[\d,]+ threads, of which [\d,]+ begin in this export", notes["mail_messages"][0])
    assert re.search(r"^[\d,]+ pull requests; [\d,]+ of them lack an 'opened' row \(opened before the export began\)\.$", notes["github_pull_requests"][0])
    assert "no event pairs a login with a name" in db.catalog()["people"]


TZ_PROBE = """
import json, sys
from harnext_eval.e2.v2.stores.base import ROOT
from harnext_eval.e2.v2.stores.raw import oracle, query
statements = [
    "SELECT toUnixTimestamp(toDateTime('2026-03-01 00:00:00')) AS u, toString(parseDateTime64BestEffort('2026-03-01T00:00:00Z')) AS p, "
    "toString(toDateTime64('2026-05-31 00:00:00', 6)) AS d, toDateTime64('2026-05-31 00:00:00', 6) = parseDateTime64BestEffort('2026-05-31T00:00:00Z') AS same",
    "SELECT toString(min(time)) AS t, toString(min(at)) AS a, toString(toStartOfMonth(max(time))) AS m, toString(toStartOfDay(max(at))) AS d, "
    "dateDiff('hour', min(time), max(time)) AS h, dateDiff('day', min(at), max(at)) AS days, formatDateTime(max(at), '%Y-%m-%dT%H:%i:%S.%fZ') AS f FROM jira_changes",
    "SELECT count() AS n FROM github_pull_requests WHERE time >= toDateTime('2026-03-01 00:00:00') AND time < toDateTime64('2026-04-01 00:00:00', 6)",
    "SELECT toString(toDate(time)) AS day, count() AS n FROM mail_messages GROUP BY day ORDER BY day LIMIT 3",
]
with query.open(sys.argv[1]) as db:
    rows = [db.sql(s)["rows"] for s in statements]
public, private = oracle.load(ROOT / "benchmarks/kafka-50-w2026")
found = oracle.check(ROOT / "out/kafka-500-v2/stores-w2026/RAW", public, private)
print(json.dumps({"rows": rows, "oracle": oracle.summary(found)}, sort_keys=True, default=str))
"""


@real
def test_results_do_not_depend_on_the_host_time_zone():
    runs = {}
    for zone in ("UTC", "Europe/Bratislava", "America/Los_Angeles"):
        found = subprocess.run([sys.executable, "-c", TZ_PROBE, str(W2026 / "export/2026-05")], capture_output=True, text=True, env={**os.environ, "TZ": zone}, timeout=600)  # fmt: skip
        assert found.returncode == 0, found.stderr[-2000:]
        runs[zone] = json.loads(found.stdout.strip().splitlines()[-1])
    assert runs["UTC"] == runs["Europe/Bratislava"] == runs["America/Los_Angeles"]
    first = runs["UTC"]["rows"][0][0]
    assert first == {"u": 1772323200, "p": "2026-03-01 00:00:00.000", "d": "2026-05-31 00:00:00.000000", "same": 1}  # fmt: skip
    assert all(c["ok"] == c["tasks"] for c in runs["UTC"]["oracle"].values())


def test_a_build_does_not_depend_on_the_host_time_zone(tmp_path):
    events = tmp_path / "events.jsonl"
    events.write_text("".join(json.dumps(e, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for e in EVENTS))  # fmt: skip
    for zone in ("UTC", "America/Los_Angeles"):
        found = subprocess.run([sys.executable, "-m", "harnext_eval.e2.v2.stores.build", "--arm", "RAW", "--scope", "smoke",
                                "--root", str(tmp_path / zone.replace("/", "_")), "--events", str(events)],
                               capture_output=True, text=True, env={**os.environ, "TZ": zone}, timeout=600)  # fmt: skip
        assert found.returncode == 0, found.stderr[-2000:]
    for ckpt in ("2026-01", "2026-02"):
        utc, pacific = (tmp_path / z / "stores-smoke/RAW/export" / ckpt for z in ("UTC", "America_Los_Angeles"))
        assert hashes(utc) == hashes(pacific)


STOP = set("""a an the of in on at by for to and or is are was were be been it its as with from that this which who
what when how many each one any all not no only per has have had do does did their there then than into before after
if else via also both such these those them they we you your our would should could can may might will shall so up
out over under about between within without through during while same other another more most less least very just
whose where""".split())  # fmt: skip
IDS = re.compile(r"\b(?:KAFKA|KIP)-\d+\b|\b\d{4}-\d\d-\d\d(?:[T ][\d:.]+Z?)?|#?\d+", re.IGNORECASE)


def trigrams(text: str) -> set[str]:
    words = [w for w in re.findall(r"[a-z]+", IDS.sub(" ", text.lower())) if w not in STOP and len(w) > 1]
    return {" ".join(words[i : i + 3]) for i in range(len(words) - 2)}


def benchmark_phrases() -> set[str]:
    """Word 3-grams of the benchmark's question templates (families.py) and public prompts, stop
    words, entity ids and dates removed. Read by this test only; the store never reads them."""
    tree = ast.parse((ROOT / "src/harnext_eval/e2/v2/families.py").read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found |= trigrams(node.value)
    for task in oracle.load(BENCH)[0]:
        found |= trigrams(task["prompt"]) | trigrams(task["family"].replace("_", " "))
    return found


def catalog_prose(export: Path) -> list[str]:
    doc = json.loads((export / "catalog/catalog.json").read_text())
    prose = [doc["people"], (export / "catalog/SOURCES.md").read_text().split("\n\n## ")[0]]
    for table in doc["tables"].values():
        prose += [table["about"], table["use_for"], *table["notes"], *table["joins"]]
        prose += [e["question"] for e in table["examples"]] + [c["description"] for c in table["columns"]]
    return prose + [link["meaning"] for link in doc["links"]]


@real
def test_catalog_prose_does_not_echo_the_benchmark():
    phrases = benchmark_phrases()
    assert len(phrases) > 1000
    for export in (W2026 / "export/2026-05", W2026 / "export/2026-01", SMOKE / "export/2023-02"):
        shared = {g: text[:100] for text in catalog_prose(export) for g in trigrams(text) & phrases}
        for phrase, text in sorted(shared.items()):
            print(f"{export.name}: {phrase!r} <- {text}")
        assert len(shared) <= 3, sorted(shared)
    shown = [guard.summary(), *guard.SYNONYMS.values()]  # also shown to the model at query time
    assert not {g for text in shown for g in trigrams(text) & phrases}


@real
def test_real_export_catalog_budget():
    export = W2026 / "export/2026-05"
    small = (export / "catalog/SOURCES.md").read_text() + (export / "catalog/sources.json").read_text()  # fmt: skip
    assert len(small) < 24_000  # ~6k tokens at ~4 characters per token


# ---- answerability ------------------------------------------------------------------------------


@real
def test_w2026_tasks_are_answerable_from_the_raw_exports():
    public, private = oracle.load(BENCH)
    found = oracle.check(W2026, public, private)
    derived = [r for r in found.values() if r["mode"] == "derived"]
    assert len(derived) >= 20 and len({r["family"] for r in derived}) >= 15
    assert [(i, r["reason"]) for i, r in found.items() if not r["ok"]] == []
    with query.open(W2026 / "export/2026-05") as db:
        for record in derived:  # every oracle statement is admitted by the guard as written
            for sql in record["sql"]:
                assert guard.check(sql, db.tables) == sql.strip().rstrip(";").strip()
    assert len(derived) == 49
    prompts = {t["id"]: t["prompt"] for t in public}
    for task_id, record in found.items():  # the gold went into grading only, never into the SQL
        secret = [str(g) for g in flat(record["expected"]) if g is not None and len(str(g)) >= 4 and str(g) not in prompts[task_id]]  # fmt: skip
        assert not [g for g in secret for sql in record.get("sql", []) if g in sql], task_id


def test_build_dispatch_knows_the_arm():
    assert isinstance(stores_build.make("RAW"), RawStore)
