"""The trial MCP server: tools per kind, the research reader (Harnext's product read interface),
server-side budgets, receipts hash chain, infra flagging."""

from __future__ import annotations

import ast
import asyncio
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastmcp")

from fastmcp import Client  # noqa: E402
from fastmcp.client.transports import StdioTransport  # noqa: E402
from harnext_builder.harness.base import ConversationTranscript, TranscriptTurn  # noqa: E402
from harnext_eval.e2.v2 import mcp_server as server  # noqa: E402

PRODUCT_MAIN = Path(__file__).resolve().parents[4] / "apps/mcp/src/harnext_mcp/main.py"


class FakeReader:
    meta = {"checkpoint": "2023-06", "sha256": "0" * 64}

    async def facts(self, query, k):
        return [{"uuid": f"f{i}", "text": f"{query} fact {i} " + "x" * 300} for i in range(k)]

    async def entities(self, query, k):
        return [{"uuid": "n1", "name": query, "text": "summary"}][:k]

    async def entity_edges(self, uuid):
        return [{"uuid": "f0", "source_entity": uuid, "text": "edge"}]

    async def episode(self, key):
        return [{"name": key, "text": "raw event"}]


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "store"
    (root / "entities/KAFKA-1").mkdir(parents=True)
    (root / "INDEX.md").write_text("# Index\n- KAFKA-1 flaky test\n")
    (root / "entities/KAFKA-1/OVERVIEW.md").write_text("\n".join(f"line {i}" for i in range(50)))
    (tmp_path / "secret.txt").write_text("outside")
    return root


def session(kind, backend, tmp_path, **budgets):
    return server.Session(kind, backend, tmp_path / "receipts.jsonl", **budgets)


def call(sess, tool, **arguments):
    return json.loads(asyncio.run(sess.invoke(tool, arguments)) or "{}")


def tools(sess):
    async def names():
        async with Client(server.create_server(sess)) as client:
            return sorted(tool.name for tool in await client.list_tools())

    return asyncio.run(names())


def research(tree, tmp_path, **kw):
    return session("research", server.ResearchBackend(tree, reader="fake", **kw), tmp_path)


def test_tools_registered_per_kind(tmp_path, tree):
    graph = session("graphiti", server.GraphitiBackend(tree, reader=FakeReader()), tmp_path)
    assert tools(graph) == ["get_entity_edges", "get_episode", "search_entities", "search_facts"]
    graph.close()
    (tmp_path / "receipts.jsonl").unlink()
    sess = research(tree, tmp_path)
    assert tools(sess) == ["context_research"]
    assert call(sess, "search_facts", query="x", k=1)["error"] == "tool not served by this checkpoint"
    assert server.KINDS["rag"] == ("search", "read_event")
    assert "files" not in server.KINDS and not hasattr(server, "FilesBackend")


def test_research_tool_is_the_production_tool():
    """Same name, same docstring as apps/mcp's context_research; identity recorded in receipts."""
    def docstring(path):
        tool = next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.AsyncFunctionDef) and n.name == "context_research")  # fmt: skip
        return [a.arg for a in tool.args.args], ast.get_docstring(tool)

    assert docstring(PRODUCT_MAIN) == docstring(Path(server.__file__))
    neutral = server.layout_neutral("x. Start from INDEX.md to navigate, then read the relevant entity files. y")
    assert neutral == "x. Start from INDEX.md if present, otherwise from the top-level listing, then read the relevant files. y"  # fmt: skip
    with pytest.raises(ValueError, match="production research prompt changed"):
        server.layout_neutral("something else")


def test_fake_reader_answers_and_receipts_record_the_call(tmp_path, tree):
    sess = research(tree, tmp_path)
    out = call(sess, "context_research", question="Who fixed KAFKA-1?")
    assert out["answer"] == "(fake reader) Who fixed KAFKA-1? [INDEX.md]" and out["files"] == ["INDEX.md"]  # fmt: skip
    assert {"turns", "usage", "seconds", "error", "truncated"} <= set(out) and out["error"] is None
    assert "items" not in out  # the product's shape, not the list envelope
    sess.close()
    rows = server.verify(tmp_path / "receipts.jsonl")
    assert rows[0]["store"]["reader"] == "fake" and rows[0]["budgets"]["calls"] == 100
    row = rows[1]
    assert row["tool"] == "context_research" and row["args"] == {"question": "Who fixed KAFKA-1?"}
    assert row["research"]["files"] == ["INDEX.md"] and row["research"]["answer_bytes"] == len(out["answer"])  # fmt: skip
    assert {"usage", "seconds", "turns", "stop_reason"} <= set(row["research"]) and "infra" not in row


def test_reader_infra_error_reaches_the_agent_and_flags_the_receipt(tmp_path, tree):
    sess = research(tree, tmp_path)
    out = call(sess, "context_research", question="!infra anything")
    assert out["answer"] is None and "usage limit" in out["error"]
    sess.close()
    rows = server.verify(tmp_path / "receipts.jsonl")
    assert rows[1]["infra"] is True and "usage limit" in rows[1]["error"]


def test_reader_model_failure_is_an_answer_not_infra(tmp_path, tree):
    class Stuck:
        name = "fake"

        async def run(self, req):
            return ConversationTranscript(
                harness="fake", model="m", stop_reason="max_turns", error="Reached max turns (40)",
                turns=[TranscriptTurn(role="assistant", content="still looking")],
            )  # fmt: skip

    sess = session("research", server.ResearchBackend(tree, reader="fake", harness=Stuck()), tmp_path)
    out = call(sess, "context_research", question="q")
    assert out["answer"].startswith("(the research agent failed: max_turns") and out["error"] is None
    sess.close()
    assert "infra" not in server.verify(tmp_path / "receipts.jsonl")[1]


def test_budgets_are_enforced_server_side(tmp_path, tree):
    backend = server.GraphitiBackend(tree, reader=FakeReader())
    sess = session("graphiti", backend, tmp_path, max_calls=3, max_bytes=1500, response_bytes=1000)
    first = asyncio.run(sess.invoke("search_facts", {"query": "q", "k": 10}))
    assert len(first.encode()) <= 1000 and json.loads(first)["truncated"]
    second = asyncio.run(sess.invoke("search_facts", {"query": "q", "k": 10}))
    assert len(first.encode()) + len(second.encode()) <= 1500  # the remaining bytes cap a reply
    assert call(sess, "search_facts", query="q", k=1).get("error", "byte").startswith("byte")
    assert asyncio.run(sess.invoke("get_episode", {"id": "e1"})) == ""  # 4th call: nothing left
    rows = server.verify(tmp_path / "receipts.jsonl")
    assert [row.get("error") for row in rows[1:]][-1] == "call budget exhausted"
    assert rows[-1]["cumulative_bytes"] <= 1500
    assert call(sess, "search_entities", query="q", k=500) == {}  # still logged, still refused
    assert server.verify(tmp_path / "receipts.jsonl")[-1]["error"] == "call budget exhausted"


def test_receipts_chain_detects_tampering_and_resumes(tmp_path, tree):
    backend = server.ResearchBackend(tree, reader="fake")
    sess = session("research", backend, tmp_path, max_calls=5)
    call(sess, "context_research", question="one")
    call(sess, "context_research", question="two")
    sess.close()
    log = tmp_path / "receipts.jsonl"
    rows = server.verify(log)
    assert [row["kind"] for row in rows] == ["start", "call", "call"]
    assert rows[1]["tool"] == "context_research" and rows[1]["args"] == {"question": "one"}
    assert {"bytes", "ms", "error", "output_sha256"} <= set(rows[2]) and rows[2][
        "previous"
    ] == rows[1]["hash"]
    again = session("research", backend, tmp_path, max_calls=5)  # a restarted server continues
    assert (again.calls, again.used) == (2, rows[1]["bytes"] + rows[2]["bytes"])
    again.close()
    with pytest.raises(ValueError, match="another store or budget"):
        session("research", backend, tmp_path, max_calls=50)
    log.write_text(log.read_text().replace('"question":"two"', '"question":"2"'))
    with pytest.raises(ValueError, match="hash chain"):
        session("research", backend, tmp_path, max_calls=5)


def test_stdio_server_answers_a_real_client(tmp_path, tree):
    log = tmp_path / "receipts.jsonl"
    arguments = ["-m", "harnext_eval.e2.v2.mcp_server", "--kind", "research", "--store", str(tree)]
    transport = StdioTransport(sys.executable, [*arguments, "--log", str(log), "--max-calls", "1", "--reader", "fake"])  # fmt: skip

    async def run():
        async with Client(transport) as client:
            listed = await client.list_tools()
            first = await client.call_tool("context_research", {"question": "who?"})
            second = await client.call_tool("context_research", {"question": "again?"})
            return listed, json.loads(first.content[0].text), second.content[0].text  # type: ignore[union-attr]

    listed, first, second = asyncio.run(run())
    assert [t.name for t in listed] == ["context_research"]
    assert listed[0].description.startswith("Search the organization's context")
    assert first["files"] == ["INDEX.md"] and "who?" in first["answer"]
    assert json.loads(second)["error"] == "call budget exhausted"
    assert len(server.verify(log)) == 3


def test_research_default_call_budget_is_25(tmp_path, tree):
    log = tmp_path / "receipts.jsonl"
    arguments = ["-m", "harnext_eval.e2.v2.mcp_server", "--kind", "research", "--store", str(tree)]
    transport = StdioTransport(sys.executable, [*arguments, "--log", str(log), "--reader", "fake"])

    async def run():
        async with Client(transport) as client:
            await client.call_tool("context_research", {"question": "q"})

    asyncio.run(run())
    assert server.verify(log)[0]["budgets"]["calls"] == 25


def test_research_search_serves_both_tools_with_production_docstrings(tmp_path, tree):
    def docstring(path, name):
        tool = next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.AsyncFunctionDef) and n.name == name)  # fmt: skip
        return [a.arg for a in tool.args.args], ast.get_docstring(tool)

    assert docstring(PRODUCT_MAIN, "search_context") == docstring(Path(server.__file__), "search_context")
    sess = session("research_search", server.ResearchSearchBackend(tree, reader="fake"), tmp_path)
    assert sorted(tools(sess)) == ["context_research", "search_context"]
    found = call(sess, "search_context", query="kafka-1 flaky", k=3)
    assert found["error"] is None and found["items"][0]["path"] == "INDEX.md"
    assert {"path", "lines", "score", "text"} <= set(found["items"][0])
    report = call(sess, "context_research", question="Who fixed KAFKA-1?")
    assert report["answer"].startswith("(fake reader)") and "items" not in report
    sess.close()
    rows = server.verify(tmp_path / "receipts.jsonl")
    assert rows[1]["tool"] == "search_context" and rows[1]["search"]["hits"] >= 1 and "research" not in rows[1]
    assert rows[2]["tool"] == "context_research" and "files" in rows[2]["research"]


def test_research_search_caps_context_research_but_not_search(tmp_path, tree):
    sess = session("research_search", server.ResearchSearchBackend(tree, reader="fake"), tmp_path)
    for _ in range(server.RESEARCH_CALLS):
        assert call(sess, "context_research", question="q")["error"] is None
    assert "use search_context" in call(sess, "context_research", question="q")["error"]
    assert call(sess, "search_context", query="flaky")["error"] is None
    sess.close()


def test_search_dir_ranks_identifiers_and_chunks_by_heading(tmp_path):
    from harnext_mcp.search import chunks, search_dir, tokens

    assert {"kafka-20592", "kafka", "20592"} <= set(tokens("See KAFKA-20592."))
    parts = chunks("a.md", "# One\nalpha\n# Two\nbeta\n")
    assert [(c.start, c.end) for c in parts] == [(1, 2), (3, 4)]
    (tmp_path / "x.md").write_text("# KAFKA-7\nassignee changed to Ann\n")
    (tmp_path / "y.md").write_text("# KAFKA-8\nnothing here\n")
    hits = search_dir(tmp_path, "KAFKA-7 assignee", 5)["hits"]
    assert hits[0]["path"] == "x.md" and hits[0]["lines"] == "1-2"


def test_reader_turn_cap_is_a_result_not_infra(tmp_path, tree):
    from harnext_builder.harness.base import ConversationTranscript, TranscriptTurn

    class Capped:
        name = "fake"

        async def run(self, request):
            return ConversationTranscript(
                harness="fake", model="m", turns=[TranscriptTurn(role="assistant", content="The network authentication path times out..."),
                       TranscriptTurn(role="tool_use", content="{}")],
                stop_reason="error", usage={},
                error="Claude Code returned an error result: Reached maximum number of turns (40) (exit code: 1)",
            )  # fmt: skip

    sess = session("research", server.ResearchBackend(tree, reader="fake", harness=Capped()), tmp_path)
    out = call(sess, "context_research", question="q")
    assert out["error"] is None and "research agent failed" in out["answer"]
    sess.close()
    assert not any(r.get("infra") for r in server.verify(tmp_path / "receipts.jsonl"))
