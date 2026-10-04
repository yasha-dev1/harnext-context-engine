"""Engine query plans: the compiler for every shape and column type, literal quoting, identifier
checks, join edges from the catalog, relaxation. No database needed (see the integration test
for the same SQL run on a real export)."""

import pytest
from harnext_eval.e2.v2.engine.plan import (
    Catalog,
    Compare,
    Filter,
    Link,
    Plan,
    PlanError,
    compile_plan,
    count_plan,
    instant,
    literal,
    relax,
    times_sql,
)

from .engine_fixtures import catalog_doc

CAT = Catalog(catalog_doc())
T = "tick_items"


def f(column, value, op="eq", table=T):
    return Filter(table, column, op, value)


def sql(plan):
    return compile_plan(plan, CAT, max_rows=200, width=300)


def test_every_shape_compiles_to_one_select():
    base = [f("key", "ABC-1")]
    lookup = sql(Plan(T, "lookup", base, ["state"]))
    assert lookup.startswith("SELECT `tick_items`.`event_id` AS `event_id`, `tick_items`.`time` AS `time`")  # fmt: skip
    assert "WHERE `tick_items`.`key` = 'ABC-1'" in lookup and lookup.endswith("LIMIT 200")
    assert "ORDER BY `tick_items`.`time` ASC" in lookup
    listed = sql(Plan(T, "list", base, ["state"], order="n", desc=True))
    assert "ORDER BY `tick_items`.`n` DESC, `tick_items`.`time` DESC, `tick_items`.`event_id` DESC" in listed  # fmt: skip
    timeline = sql(Plan(T, "timeline", base, ["state", "at"], order="at"))
    assert "ORDER BY `tick_items`.`at` ASC" in timeline
    newest = sql(Plan(T, "timeline", base, ["state"], order="at", desc=True, offset=25, limit=25))
    assert newest.endswith("ORDER BY `tick_items`.`at` DESC, `tick_items`.`time` DESC, `tick_items`.`event_id` DESC LIMIT 25 OFFSET 25")  # fmt: skip
    search = sql(Plan(T, "search", [f("body", "rebalance storm", "contains")], ["body"]))
    assert "positionCaseInsensitiveUTF8(`tick_items`.`body`, 'rebalance storm') > 0" in search
    assert "substringUTF8(`tick_items`.`body`, greatest(1, positionCaseInsensitiveUTF8" in search  # snippet around the match  # fmt: skip
    total = sql(Plan(T, "count", base))
    assert total == "SELECT count() AS `n`, uniqExact(`tick_items`.`key`) AS `distinct_key` FROM `tick_items` WHERE `tick_items`.`key` = 'ABC-1'"  # fmt: skip
    grouped = sql(Plan(T, "count", base, group="state"))
    assert "GROUP BY `state` ORDER BY `n` DESC, `state` LIMIT 200" in grouped
    assert "distinct_key" in grouped  # the key column is counted distinctly per group
    assert "distinct_" not in sql(Plan(T, "count", [], group="key"))
    tags = sql(Plan(T, "count", [], group="tags"))
    assert "arrayJoin(`tick_items`.`tags`) AS `tags`" in tags and "WHERE 1" in tags


CASES = [  # column, value, operator, expected condition
    ("key", "ABC-1", "eq", "`tick_items`.`key` = 'ABC-1'"),
    ("n", "#42", "eq", "`tick_items`.`n` = 42"),
    ("score", "1.5", "eq", "`tick_items`.`score` = 1.5"),
    ("ok", "True", "eq", "`tick_items`.`ok` = true"),
    ("tags", "blocker", "eq", "has(`tick_items`.`tags`, 'blocker')"),
    ("nums", "7", "eq", "has(`tick_items`.`nums`, 7)"),
    ("tags", "block", "contains", "arrayExists(x -> positionCaseInsensitiveUTF8(x, 'block') > 0, `tick_items`.`tags`)"),
    ("key", "abc", "contains", "positionCaseInsensitiveUTF8(`tick_items`.`key`, 'abc') > 0"),
    ("n", "42", "contains", "`tick_items`.`n` = 42"),  # containment is text-only
    ("at", "2030-01-05", "from", "`tick_items`.`at` >= '2030-01-05 00:00:00.000000'"),
    ("at", "2030-01-05", "after", "`tick_items`.`at` >= '2030-01-06 00:00:00.000000'"),
    ("at", "2030-01-05T10:00:00Z", "before", "`tick_items`.`at` < '2030-01-05 10:00:00.000000'"),
    ("at", "2030-02-28T23:59:59Z", "through", "`tick_items`.`at` < '2030-03-01 00:00:00.000000'"),
    ("at", "2030-02-28", "through", "`tick_items`.`at` < '2030-03-01 00:00:00.000000'"),
    ("at", "2030-02-28T23:59:59.25Z", "through", "`tick_items`.`at` < '2030-02-28 23:59:59.250001'"),
    ("at", "2030-12", "on", "`tick_items`.`at` >= '2030-12-01 00:00:00.000000' AND `tick_items`.`at` < '2031-01-01 00:00:00.000000'"),
    ("time", "2030-01-05", "eq", "`tick_items`.`time` >= '2030-01-05 00:00:00.000000' AND `tick_items`.`time` < '2030-01-06 00:00:00.000000'"),
]  # fmt: skip


def test_operators_follow_column_types():
    for column, value, op, expected in CASES:
        assert expected in sql(Plan(T, "list", [f(column, value, op)])), (column, value, op)


def test_times_in_any_agent_form_are_normalised():
    assert instant("2030-01-05T10:11:12.5Z") == ("'2030-01-05 10:11:12.500000'", "'2030-01-05 10:11:12.500001'")  # fmt: skip
    assert instant("2030-01-05T10:11:12Z") == ("'2030-01-05 10:11:12.000000'", "'2030-01-05 10:11:13.000000'")  # fmt: skip
    assert instant("2030-01-05 10:11")[1] == "'2030-01-05 10:12:00.000000'"
    assert instant("2030") == ("'2030-01-01 00:00:00.000000'", "'2031-01-01 00:00:00.000000'")
    with pytest.raises(PlanError, match="not a valid date"):
        instant("2030-02-30")
    with pytest.raises(PlanError, match="not a date"):
        instant("yesterday")


def test_hostile_literals_are_quoted():
    hostile = "'; DROP TABLE x; --"
    out = sql(Plan(T, "list", [f("key", hostile)]))
    assert "`tick_items`.`key` = '\\'; DROP TABLE x; --'" in out
    assert literal("a\\'b", "Nullable(String)", "key") == "'a\\\\\\'b'"
    assert "'x\\' OR 1=1 --'" in sql(Plan(T, "search", [f("body", "x' OR 1=1 --", "contains")], ["body"]))  # fmt: skip
    with pytest.raises(PlanError, match="whole number"):
        sql(Plan(T, "list", [f("n", "1 OR 1=1")]))
    with pytest.raises(PlanError, match="true/false"):
        sql(Plan(T, "list", [f("ok", "1; DROP")]))


def test_unknown_identifiers_and_bad_plans_are_rejected():
    rejects = {
        "unknown table": Plan("secrets", "list", [f("key", "a", table="secrets")]),
        "unknown shape": Plan(T, "pivot", [f("key", "a")]),
        "unknown column tick_items.nope": Plan(T, "list", [f("nope", "a")]),
        "unknown column tick_items.raw": Plan(T, "list", [f("key", "a")], ["raw"]),
        "unknown column tick_items.`x`": Plan(T, "list", [f("key", "a")], ["`x`"]),
        "unknown operator": Plan(T, "list", [f("key", "a", "regex")]),
        "needs a time or number column": Plan(T, "list", [f("key", "2030-01-01", "after")]),
        "not part of this query": Plan(T, "list", [f("author", "a", table="tick_notes")]),
        "no value was used": Plan(T, "list", []),
        "not joined": Plan(
            T,
            "list",
            [f("author", "a", table="tick_notes")],
            link=Link("tick_notes", "author", "key"),
        ),  # fmt: skip
        "has no filter": Plan(T, "list", [f("key", "a")], link=Link("tick_notes", "key", "key")),
        "limit must be": Plan(T, "list", [f("key", "a")], limit=10_000),
    }
    for message, plan in rejects.items():
        with pytest.raises(PlanError, match=message.split(" tick_items")[0]):
            sql(plan)


def test_join_edges_come_from_the_catalog_and_identifier_shapes():
    assert CAT.edges[("tick_notes", "tick_items")] == ("key", "key")
    assert CAT.edges[("tick_items", "tick_notes")] == ("key", "key")
    assert CAT.edges[("mail_msgs", "tick_items")] == ("refs", "key")  # both look like 'ABC-9'
    assert ("mail_msgs", "tick_notes") not in CAT.edges  # tick_notes.key has no examples
    assert CAT.groups() == {"mail": ["mail_msgs"], "tick": ["tick_items", "tick_notes"]}


def test_links_compile_for_scalar_and_array_sides():
    via_notes = sql(Plan(T, "list", [f("author", "rita", table="tick_notes")], link=Link("tick_notes", "key", "key")))  # fmt: skip
    assert "`tick_items`.`key` IN (SELECT `tick_notes`.`key` FROM `tick_notes` WHERE `tick_notes`.`author` = 'rita')" in via_notes  # fmt: skip
    from_mail = sql(Plan(T, "list", [f("sender", "Ann", table="mail_msgs")], link=Link("mail_msgs", "refs", "key")))  # fmt: skip
    assert "`tick_items`.`key` IN (SELECT arrayJoin(`mail_msgs`.`refs`) FROM `mail_msgs`" in from_mail  # fmt: skip
    to_mail = sql(Plan("mail_msgs", "list", [f("state", "Open")], link=Link(T, "key", "refs")))
    assert "hasAny(`mail_msgs`.`refs`, (SELECT groupUniqArray(assumeNotNull(`tick_items`.`key`)) FROM `tick_items` WHERE (`tick_items`.`state` = 'Open') AND isNotNull(`tick_items`.`key`)))" in to_mail  # fmt: skip


def test_several_values_for_one_column_are_alternatives():
    out = sql(Plan(T, "list", [f("key", "A-1"), f("key", "A-2"), f("state", "Open")]))
    assert "((`tick_items`.`key` = 'A-1') OR (`tick_items`.`key` = 'A-2')) AND `tick_items`.`state` = 'Open'" in out  # fmt: skip
    window = sql(Plan(T, "list", [f("at", "2030-01-01", "after"), f("at", "2030-01-10", "before")]))
    assert ") AND (`tick_items`.`at` <" in window  # a time window is both bounds


def test_relax_and_count_plan():
    plan = Plan(T, "list", [f("key", "abc"), f("n", "3")])
    loose = relax(plan, CAT)
    assert [x.op for x in loose.filters] == ["contains", "eq"]
    assert relax(loose, CAT) is None
    counted = sql(count_plan(Plan(T, "timeline", [f("key", "a")], ["state"], order="at")))
    assert counted.startswith("SELECT count() AS `n`") and "LIMIT" not in counted


def test_sibling_columns_are_alternatives():
    assert CAT.siblings("mail_msgs", "refs") == ["body_refs"] and CAT.siblings("tick_items", "key") == []  # fmt: skip
    out = sql(
        Plan("mail_msgs", "list", [Filter("mail_msgs", "refs", "eq", "ABC-1", ("body_refs",))])
    )
    assert "((has(`mail_msgs`.`refs`, 'ABC-1')) OR (has(`mail_msgs`.`body_refs`, 'ABC-1')))" in out
    with pytest.raises(PlanError, match="not a sibling"):
        sql(Plan("mail_msgs", "list", [Filter("mail_msgs", "refs", "eq", "x", ("sender",))]))


def test_control_characters_are_stripped_and_statements_capped():
    out = sql(Plan(T, "list", [f("key", "a\nb\x00c\\s+")]))
    assert "`tick_items`.`key` = 'a b c\\\\s+'" in out
    with pytest.raises(PlanError, match="exceed 20000 characters"):
        sql(Plan(T, "list", [f("key", "x" * 25_000)]))


def test_exclusion_empty_and_numeric_ranges():
    out = sql(Plan(T, "list", [f("state", "Closed", "ne"), f("state", "Resolved", "ne"), f("tags", "x", "ne")]))  # fmt: skip
    assert "((isNull(`tick_items`.`state`) OR NOT (`tick_items`.`state` = 'Closed'))) AND ((isNull(`tick_items`.`state`) OR NOT (`tick_items`.`state` = 'Resolved')))" in out  # fmt: skip
    assert "NOT (has(`tick_items`.`tags`, 'x'))" in out
    assert "isNull(`tick_items`.`body`)" in sql(
        Plan(T, "list", [f("key", "a"), f("body", "", "null")])
    )
    assert "notEmpty(`tick_items`.`tags`)" in sql(Plan(T, "list", [f("key", "a"), f("tags", "", "notnull")]))  # fmt: skip
    assert "((`tick_items`.`n` >= 3) AND (`tick_items`.`n` < 9))" in sql(Plan(T, "list", [f("n", "3", "from"), f("n", "9", "before")]))  # fmt: skip
    assert "((`tick_items`.`n` > 3) AND (`tick_items`.`n` <= 9))" in sql(Plan(T, "list", [f("n", "3", "after"), f("n", "9", "through")]))  # fmt: skip
    with pytest.raises(PlanError, match="whole number"):
        sql(Plan(T, "list", [f("n", "\u0663", "eq")]))  # a non-ASCII digit is not a number literal
    with pytest.raises(PlanError, match="whole number"):
        sql(Plan(T, "list", [f("n", "three", "after")]))


def test_column_comparisons():
    same = sql(Plan(T, "count", [], compares=[Compare("key", "ne", "state")]))
    assert "ifNull(`tick_items`.`key` != `tick_items`.`state`, isNull(`tick_items`.`key`) != isNull(`tick_items`.`state`))" in same  # fmt: skip
    linked = sql(Plan(T, "count", [f("author", "rita", table="tick_notes")], link=Link("tick_notes", "key", "key"), compares=[Compare("state", "ne", "author", True)]))  # fmt: skip
    assert "(`tick_items`.`key`, `tick_items`.`state`) NOT IN (SELECT `tick_notes`.`key`, `tick_notes`.`author` FROM `tick_notes`)" in linked  # fmt: skip
    for bad, message in [(Compare("key", "ne", "n"), "not comparable"), (Compare("key", "ne", "author", True), "needs a link"),  # fmt: skip
                         (Compare("key", "lt", "state"), "invalid comparison")]:  # fmt: skip
        with pytest.raises(PlanError, match=message):
            sql(Plan(T, "count", [], compares=[bad]))


def test_values_latest_earliest_and_distinct_counts():
    values = sql(Plan(T, "values", [f("key", "a")], group="tags"))
    assert values == "SELECT arraySort(groupUniqArrayArray(`tick_items`.`tags`)) AS `values`, count() AS `rows` FROM `tick_items` WHERE `tick_items`.`key` = 'a'"  # fmt: skip
    assert "groupUniqArray(`tick_items`.`state`)" in sql(Plan(T, "values", [f("key", "a")], group="state"))  # fmt: skip
    with pytest.raises(PlanError, match="needs the column"):
        sql(Plan(T, "values", [f("key", "a")]))
    latest = sql(Plan(T, "latest", [f("key", "a")], ["state"], order="at"))
    assert latest.endswith("ORDER BY `tick_items`.`at` DESC, `tick_items`.`time` DESC, `tick_items`.`event_id` DESC LIMIT 1")  # fmt: skip
    assert sql(Plan(T, "earliest", [f("key", "a")], ["state"], desc=True)).endswith("`event_id` ASC LIMIT 1")  # fmt: skip
    assert "uniqExact(`tick_items`.`state`) AS `distinct_state`" in sql(Plan(T, "count", [f("key", "a")], group="tags", distinct="state"))  # fmt: skip


def test_times_are_compared_in_sql():
    out = times_sql(CAT, ("tick_items", "tick:1"), ("tick_notes", "x'y"), "days", "down")
    assert "(SELECT coalesce(`at`, `time`) AS `t` FROM `tick_items` WHERE `event_id` = 'tick:1') AS `a`" in out  # fmt: skip
    assert "(SELECT `time` AS `t` FROM `tick_notes` WHERE `event_id` = 'x\\'y') AS `b`" in out
    assert (
        "floor(divide(abs(dateDiff('millisecond', `a`.`t`, `b`.`t`)), 86400000)) AS `amount`" in out
    )
    assert "AS `amount`" in times_sql(CAT, ("tick_items", "a"), ("tick_items", "b")) and "floor" not in times_sql(CAT, ("tick_items", "a"), ("tick_items", "b"))  # fmt: skip
    with pytest.raises(PlanError, match="unit must be"):
        times_sql(CAT, ("tick_items", "a"), ("tick_items", "b"), "fortnights")


def test_every_operator_compiles_and_is_explained():
    from harnext_eval.e2.v2.engine.plan import OPS
    from harnext_eval.e2.v2.engine.rawtools import explain

    for op in OPS:
        column, value = ("at", "2030-01-05") if op in ("before", "through", "from", "after", "on") else ("key", "" if op in ("null", "notnull") else "a")  # fmt: skip
        plan = Plan(T, "list", [f("key", "x"), f(column, value, op)])
        assert sql(plan) and explain(plan).startswith("[list over tick_items")
