"""The engine over the real smoke RAW export (chdb) with scripted fake models: compiled plans of
every shape run in the engine, hostile literals stay literals, and a full research call resolves
a title to a key and then replays the key's status history."""

import asyncio
import re

import pytest
from harnext_eval.e2.v2.engine.clients import FakeChat, FakeJev, choice, noul, text, tool_call
from harnext_eval.e2.v2.engine.config import EngineConfig
from harnext_eval.e2.v2.engine.engine import Engine
from harnext_eval.e2.v2.engine.plan import (
    FUNCTIONS,
    Catalog,
    Compare,
    Filter,
    Link,
    Plan,
    compile_plan,
    times_sql,
)
from harnext_eval.e2.v2.engine.rawtools import RawTools
from harnext_eval.e2.v2.engine.session import Session, SharedDB
from harnext_eval.e2.v2.stores.base import ROOT

pytest.importorskip("chdb")
EXPORT = ROOT / "out/kafka-500-v2/stores-smoke/RAW/export/2023-02"
pytestmark = pytest.mark.skipif(not EXPORT.exists(), reason="smoke RAW export not built")


@pytest.fixture
def raw():
    from harnext_eval.e2.v2.stores.raw.query import open as open_raw

    db = open_raw(EXPORT)
    yield db
    db.close()


def f(table, column, value, op="eq"):
    return Filter(table, column, op, value)


def test_every_compiled_shape_runs_on_the_real_export(raw):
    cat = Catalog(raw.catalog_doc)
    plans = [
        Plan(
            "jira_issues",
            "lookup",
            [f("jira_issues", "summary", "Move ConsoleProducer to tools")],
            ["issue_key", "summary"],
        ),  # fmt: skip
        Plan(
            "jira_changes",
            "timeline",
            [
                f("jira_changes", "issue_key", "KAFKA-14577"),
                f("jira_changes", "field", "status"),
                f("jira_changes", "at", "2023-02-28T23:59:59Z", "before"),
            ],
            ["from_value", "to_value", "at"],
            order="at",
        ),  # fmt: skip
        Plan(
            "mail_messages",
            "search",
            [f("mail_messages", "subject", "[DISCUSS] KIP", "contains")],
            ["subject", "sender", "body"],
            desc=True,
        ),  # fmt: skip
        Plan(
            "github_reviews",
            "count",
            [
                f("github_reviews", "state", "approved"),
                f("github_pull_requests", "action", "merged"),
            ],
            group="reviewer",
            link=Link("github_pull_requests", "pr_number", "pr_number"),
        ),  # fmt: skip
        Plan(
            "github_pull_requests",
            "list",
            [f("jira_issues", "summary", "ConsoleProducer", "contains")],
            ["pr_number", "action", "issues"],
            link=Link("jira_issues", "issue_key", "issues"),
        ),  # fmt: skip
        Plan(
            "jira_issues",
            "list",
            [f("github_pull_requests", "pr_number", "13095")],
            ["issue_key"],
            link=Link("github_pull_requests", "issues", "issue_key"),
        ),  # fmt: skip
        Plan(
            "github_pull_requests",
            "count",
            [
                f("github_pull_requests", "time", "2023-01", "on"),
                f("github_pull_requests", "changed_files", "build.gradle"),
            ],
        ),  # fmt: skip
        Plan(
            "jira_changes",
            "list",
            [f("jira_changes", "linked_pr", "#13095")],
            ["issue_key"],
            order="history_id",
            desc=True,
        ),  # fmt: skip
    ]
    for plan in plans:
        result = raw.sql(compile_plan(plan, cat, 200, 300), max_rows=200)
        assert result["error"] is None, (plan, result["error"])
        assert result["rows"], plan
    assert result["rows"][0]["issue_key"].startswith("KAFKA-")
    hostile = Plan("jira_issues", "list", [f("jira_issues", "summary", "'; DROP TABLE jira_issues; --"), f("jira_issues", "issue_key", "x\\' OR 1=1 --")])  # fmt: skip
    result = raw.sql(compile_plan(hostile, cat, 200, 300))
    assert result["error"] is None and result["rows"] == []
    assert raw.sql("SELECT count() AS n FROM jira_issues")["rows"][0]["n"] > 0


def plan_answers(state, qid, q):
    """Jev stand-in: sensible slots for the two needs of the scripted agent."""
    if "sources" in state:
        return noul(0.95 if qid == "jira" else 0.1)
    need = state.get("need", "")
    title = "titled" in need
    table = "jira_issues" if title else "jira_changes"
    slots = {"table": choice(table), "shape": choice("lookup" if title else "timeline"), "link": choice("none"),  # fmt: skip
             "bind_v1": choice("summary" if title else "issue_key"), "bind_v2": choice("field"),
             "order": choice("at"), "direction": choice("asc"),
             "out_issue_key": noul(0.9), "out_from_value": noul(0.9), "out_to_value": noul(0.9), "out_at": noul(0.9)}  # fmt: skip
    return slots.get(qid) or (noul(0.0) if qid.startswith("out_") else None)


def agent(messages, tools, purpose):
    """gpt-oss stand-in: title -> key -> status history -> done, citing the rows it saw."""
    if purpose == "report":
        finding = messages[-1]["content"]
        return text(f"Report. {finding[-300:]}")
    results = [m["content"] for m in messages if m["role"] == "tool"]
    if not results:
        return tool_call("find", {"need": "The issue key of the issue titled Move ConsoleProducer to tools", "values": ["Move ConsoleProducer to tools"]})  # fmt: skip
    if len(results) == 1:
        key = re.search(r"issue_key: (KAFKA-\d+)", results[0]).group(1)
        return tool_call("find", {"need": f"The status changes of {key} in time order", "values": [key, "status"]})  # fmt: skip
    ids = re.findall(r"event_id: (\S+)", results[1])
    last = re.findall(r"to_value: ([^|\n]+)", results[1])[-1].strip()
    return tool_call("done", {"finding": f"Latest status: {last} [{ids[-1]}]", "evidence": ids})


def test_research_over_the_real_export_with_scripted_models():
    made = Engine(EXPORT, config=EngineConfig(backoff_s=0), jev=FakeJev(plan_answers), chat=FakeChat(agent))  # fmt: skip

    async def go():
        try:
            return await made.research("What was the status of the issue titled 'Move ConsoleProducer to tools' at the snapshot?")  # fmt: skip
        finally:
            await made.aclose()

    result = asyncio.run(go())
    trace = result["trace"]
    assert trace["routing"]["chosen"] == ["jira"]
    branch = trace["branches"][0]
    first, second = branch["calls"]
    assert first["plan"]["table"] == "jira_issues" and first["rows"] == 1 and first["outcome"] == "rows"  # fmt: skip
    assert second["plan"]["shape"] == "timeline" and second["rows"] >= 2
    assert "`jira_changes`.`issue_key` = 'KAFKA-14577' AND `jira_changes`.`field` = 'status'" in second["sql"]  # fmt: skip
    assert branch["status"] == "done" and branch["evidence"] and not branch["unverified_evidence"]
    assert all(e.startswith("jira:") for e in result["evidence"])
    assert branch["finding"].startswith("Latest status: Patch Available [jira:13516666:change:")
    assert trace["stats"] == {"find": 2, "find_rows": 2, "find_empty": 0, "find_invalid": 0, "find_restate": 0,  # fmt: skip
                              "sql": 0, "sql_ok": 0, "read": 0, "search": 0, "compare_times": 0, "rejected": []}  # fmt: skip
    assert result["usage"]["jev"]["calls"] == 6  # route + 2 x (table, slots) + support


def called(sql: str) -> set[str]:
    from harnext_eval.e2.v2.stores.raw import guard

    found = guard.tokens(sql)
    return {c for i in range(len(found)) if (c := guard.call(found, i))}


def synthetic_plans():
    from .engine_fixtures import catalog_doc

    cat = Catalog(catalog_doc())
    t = "tick_items"
    shapes = [Plan(t, s, [f(t, "key", "a")], ["state", "body", "tags", "n"]) for s in ("lookup", "list", "timeline", "search")]  # fmt: skip
    types = [Plan(t, "list", [f(t, c, v, op)]) for c, v, op in [
        ("key", "a", "eq"), ("key", "a", "contains"), ("n", "1", "eq"), ("score", "1.5", "eq"),
        ("ok", "true", "eq"), ("tags", "x", "eq"), ("tags", "x", "contains"), ("nums", "3", "eq"),
        ("at", "2030-01-01", "after"), ("at", "2030-01-01T00:00:00Z", "before"), ("at", "2030-01", "on"),
        ("time", "2030-01-02", "eq"), ("body", "'; DROP TABLE x; --", "contains"), ("body", "a\\s+b\n\t'", "contains")]]  # fmt: skip
    counts = [Plan(t, "count", [f(t, "key", "a")]), Plan(t, "count", [], group="tags"), Plan(t, "count", [f(t, "key", "a")], group="state"),  # fmt: skip
              Plan(t, "count", [f(t, "key", "a")], group="state", distinct="n"), Plan(t, "values", [f(t, "key", "a")], group="tags"),  # fmt: skip
              Plan(t, "latest", [f(t, "key", "a")], ["state"], order="at"), Plan(t, "earliest", [f(t, "key", "a")], ["state"]),  # fmt: skip
              Plan(t, "list", [f(t, "state", "x", "ne"), f(t, "tags", "x", "ne"), f(t, "body", "", "null"), f(t, "tags", "", "notnull"), f(t, "n", "3", "after")]),  # fmt: skip
              Plan(t, "count", [], compares=[Compare("key", "ne", "state"), Compare("key", "eq", "state")]),  # fmt: skip
              Plan(t, "count", [f("tick_notes", "author", "r")], link=Link("tick_notes", "key", "key"), compares=[Compare("state", "ne", "author", True)])]  # fmt: skip
    links = [Plan(t, "list", [f("tick_notes", "author", "r")], link=Link("tick_notes", "key", "key")),
             Plan(t, "list", [f("mail_msgs", "sender", "s")], link=Link("mail_msgs", "refs", "key")),
             Plan("mail_msgs", "list", [f(t, "state", "Open")], link=Link(t, "key", "refs"))]  # fmt: skip
    return cat, shapes + types + counts + links


def test_compiled_sql_uses_only_listed_functions_and_passes_the_guard(raw):
    from harnext_eval.e2.v2.stores.raw import guard

    cat, plans = synthetic_plans()
    emitted = set()
    for plan in plans:
        sql = compile_plan(plan, cat, 200, 300)
        assert guard.check(sql, cat.types) == sql
    times = times_sql(cat, ("tick_items", "a"), ("tick_notes", "b"))
    assert guard.check(times, cat.types) == times
    emitted |= called(times)
    for plan in plans:
        sql = compile_plan(plan, cat, 200, 300)
        assert not re.search(r"\bAS [A-Za-z_]", sql), sql  # every alias is backticked
        emitted |= called(sql)
    real = Catalog(raw.catalog_doc)
    for plan in [
        Plan(
            "jira_changes",
            "timeline",
            [
                f("jira_changes", "issue_key", "KAFKA-14577"),
                f("jira_changes", "at", "2023-02-01T00:00:00Z", "before"),
            ],
            ["from_value", "to_value", "components"],
            order="at",
        ),  # fmt: skip
        Plan(
            "github_commits",
            "list",
            [
                f("github_commits", "prs", "13095"),
                f("github_commits", "issues", "KAFKA", "contains"),
            ],
            ["sha", "prs"],
        ),  # fmt: skip
        Plan(
            "github_reviews",
            "count",
            [f("github_pull_requests", "action", "merged")],
            group="reviewer",
            link=Link("github_pull_requests", "pr_number", "pr_number"),
        ),  # fmt: skip
    ]:
        sql = compile_plan(plan, real, 200, 300)
        assert guard.check(sql, real.types) == sql
        assert raw.sql(sql)["error"] is None
        emitted |= called(sql)

    async def probe():
        session = Session(EngineConfig(), FakeJev(), FakeChat([]), SharedDB(raw))
        tools = RawTools(session, ["jira_issues"])
        text, found = await tools.probe(["ConsoleProducer", "x' OR 1=1"])
        session.db.pool.shutdown()
        return text, found

    text, found = asyncio.run(probe())
    assert found and "'ConsoleProducer' in" in text
    emitted |= {"countIf", "positionCaseInsensitiveUTF8", "arrayExists"}
    assert emitted <= FUNCTIONS, emitted - FUNCTIONS
    assert all(guard.allowed(name) for name in FUNCTIONS)


def test_compare_times_runs_on_the_real_export(raw):
    async def go():
        session = Session(EngineConfig(), FakeJev(), FakeChat([]), SharedDB(raw))
        tools = RawTools(session, ["jira_changes"])
        rows = raw.sql("SELECT event_id FROM jira_changes WHERE issue_key = 'KAFKA-14628' AND field = 'resolution' ORDER BY at LIMIT 1")["rows"]  # fmt: skip
        merged = raw.sql("SELECT event_id FROM github_pull_requests WHERE pr_number = 13131 AND action = 'merged'")["rows"]  # fmt: skip
        out = await tools.compare_times(rows[0]["event_id"], merged[0]["event_id"], "days", "down")
        missing = await tools.compare_times("nope:1", merged[0]["event_id"], "days", "down")
        session.db.pool.shutdown()
        return out, missing, rows[0]["event_id"], tools

    out, missing, resolution, tools = asyncio.run(go())
    assert out.startswith(f"{resolution} at 2023-01-26T19:06:30.301000Z; github:api:apache/kafka:pr-13131-merged at 2023-01-26T19:06:09.000000Z")  # fmt: skip
    assert "github:api:apache/kafka:pr-13131-merged is earlier; difference 0 days (rounding: down); exact difference 21301 milliseconds" in out  # fmt: skip
    assert missing.startswith("error: no event with id 'nope:1'")
    assert resolution in tools.records and resolution in tools.seen
