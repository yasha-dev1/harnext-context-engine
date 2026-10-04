"""Engine loop with fake models and a fake database: branch tool handling (malformed JSON, unknown
tools, repeats), budgets and partial results, SQL fallback gating per mode, the uncertain-slot
restate, row relevance, memory tools and path confinement, aggregation, infra propagation and
trace completeness."""

import asyncio
import json
import os

import pytest
from harnext_eval.e2.v2.engine.clients import (  # fmt: skip
    FakeChat,
    FakeJev,
    InfraError,
    choice,
    noul,
    text,
    tool_call,
)
from harnext_eval.e2.v2.engine.config import EngineConfig
from harnext_eval.e2.v2.engine.engine import Engine

from .engine_fixtures import FakeDB

ROW = {"event_id": "tick:1", "time": "2030-01-03 10:00:00.000000", "key": "ABC-1", "state": "Open"}


def jev_answers(plan=None, factual=None):
    plan, factual = plan or {}, factual or {}

    def answer(state, qid, q):
        if "sources" in state:  # routing
            return noul(0.9 if qid == "tick" else 0.2)
        if "findings" in state:  # judging: relevance and "states facts" per finding
            source, _, kind = qid.partition(":")
            return noul(0.9 if source == "tick" else 0.05) if kind == "relevant" else noul(factual.get(source, 0.9))  # fmt: skip
        if "rows" in state:
            return noul(0.9 if "keep" in state["rows"][qid] else 0.1)
        if "hits" in state:
            return noul(0.9 if "wanted" in state["hits"][qid] else 0.1)
        defaults = {"table": choice("tick_items"), "shape": choice("list"), "link": choice("none"),
                    "bind_v1": choice("key"), "direction": choice("asc")}  # fmt: skip
        found = {**defaults, **plan}.get(qid)
        return found or (noul(0.0) if qid.startswith("out_") else None)

    return answer


def last_tool(messages):
    return next(m["content"] for m in reversed(messages) if m["role"] == "tool")


def engine(script, *, db=None, plan=None, memory=None, **config):
    config = EngineConfig(**{"backoff_s": 0, **config})
    chat = FakeChat(script)
    jev = FakeJev(jev_answers(plan))
    made = Engine(db=db or FakeDB(lambda sql: [ROW]), memory=memory, config=config, jev=jev, chat=chat)  # fmt: skip
    return made, chat, jev


def research(made, question="What is the state of ABC-1?"):
    async def go():
        try:
            return await made.research(question)
        finally:
            await made.aclose()

    return asyncio.run(go())


def branch_trace(result, source="tick"):
    return next(b for b in result["trace"]["branches"] if b["source"] == source)


def test_a_full_research_call_and_its_trace():
    script = [tool_call("find", {"need": "the state of item ABC-1", "values": ["ABC-1"]}),
              tool_call("done", {"finding": "ABC-1 is Open [tick:1]", "evidence": ["tick:1", "made-up:9"]}),  # fmt: skip
              text("ABC-1 was Open [tick:1].")]  # fmt: skip
    made, chat, jev = engine(script)
    result = research(made)
    assert result["answer"] == "ABC-1 was Open [tick:1]." and result["partial"] is False
    assert result["evidence"] == ["tick:1"]
    trace = result["trace"]
    assert set(trace) >= {"question", "identity", "routing", "branches", "aggregation", "stats",
                          "answer", "evidence", "partial", "usage", "calls", "seconds"}  # fmt: skip
    assert trace["routing"] == {"probabilities": {"mail": 0.2, "tick": 0.9}, "chosen": ["tick"]}
    branch = branch_trace(result)
    assert branch["status"] == "done" and branch["unverified_evidence"] == ["made-up:9"]
    find = branch["calls"][0]
    assert find["outcome"] == "rows" and find["rows"] == 1 and find["plan"]["table"] == "tick_items"
    assert find["sql"].startswith("SELECT ") and "`tick_items`.`key` = 'ABC-1'" in find["sql"]
    assert "tick:1" in last_tool(chat.calls[1]["messages"]) and "2030-01-03T10:00:00.000000Z" in last_tool(chat.calls[1]["messages"])  # fmt: skip
    usage = result["usage"]
    assert usage["chat"]["calls"] == 3 and usage["jev"]["calls"] == 4  # route, 2 plan calls, verify
    assert usage["cost_usd"] == pytest.approx(usage["jev"]["cost_usd"] + usage["chat"]["cost_usd"])
    assert [c["purpose"] for c in trace["calls"]] == ["route", "branch:tick", "plan:table", "plan:slots", "branch:tick", "report", "verify"]  # fmt: skip
    assert trace["aggregation"]["support_by"] == {
        "tick": "contained in cited rows"
    }  # ABC-1 is in the cited row
    assert trace["verification"] == {"removed_citations": [], "unverified_values": [], "regenerated": [], "stripped": [], "verdict": "supported"}  # fmt: skip
    assert trace["stats"]["find"] == 1 and trace["stats"]["sql"] == 0
    assert trace["identity"]["config_sha256"] == made.hash
    json.dumps(trace)  # serialisable


def test_malformed_json_unknown_tools_and_repeats_go_back_to_the_model():
    script = [tool_call("find", '{"need": "the item", "values": ['),
              tool_call("grep", {"q": "x"}),
              tool_call("find", {"need": "the item ABC-1", "values": ["ABC-1"]}),
              tool_call("find", {"need": "the item  ABC-1", "values": ["ABC-1"]}),
              tool_call("read", {"event_id": "tick:404"}),
              tool_call("find", {"wrong": 1}),
              tool_call("done", {"finding": "Open", "evidence": ["tick:1"]}), text("Open")]  # fmt: skip
    made, chat, _ = engine(script)
    result = research(made)
    replies = [last_tool(c["messages"]) for c in chat.calls[1:7]]
    assert "not a valid JSON object" in replies[0]
    assert "unknown tool 'grep'" in replies[1]
    assert replies[2].startswith("1 matching rows")
    assert "exact find call before" in replies[3]
    assert "no event with id 'tick:404'" in replies[4]
    assert replies[5].startswith("error: bad arguments for find")
    branch = branch_trace(result)
    assert branch["malformed_calls"] == 1 and branch["status"] == "done"
    assert [c.get("outcome") for c in branch["calls"] if c["tool"] == "find"] == ["rows", "repeat"]  # fmt: skip


def test_turn_budget_ends_in_a_partial_finding():
    finds = [tool_call("find", {"need": f"item {i}", "values": [f"ABC-{i}"]}) for i in range(3)]
    made, chat, _ = engine([*finds, text("Found only ABC-0 so far [tick:1]."), text("Partial: ABC-0 open.")], branch_turns=3)  # fmt: skip
    result = research(made)
    branch = branch_trace(result)
    assert branch["status"] == "partial" and "turn budget" in branch["error"]
    assert branch["finding"] == "Found only ABC-0 so far [tick:1]." and branch["evidence"] == ["tick:1"]  # fmt: skip
    assert chat.calls[3]["tools"] == [] and result["partial"] is True
    assert result["answer"] == "Partial: ABC-0 open."


def test_time_budget_salvages_the_last_results():
    made, chat, _ = engine(lambda messages, tools, purpose: text("report") if purpose == "report" else tool_call("find", {"need": "x", "values": ["ABC-1"]}),  # fmt: skip
                           deadline_s=1.0, reserve_s=5.0)  # fmt: skip
    result = research(made)
    branch = branch_trace(result)
    assert branch["status"] == "partial" and "time budget" in branch["error"]
    assert branch["turns"] == 0 and result["partial"] is True


def test_a_text_reply_is_taken_as_the_finding():
    made, _, _ = engine([text(""), text("ABC-1 is Open"), text("Open")])
    result = research(made)
    assert branch_trace(result)["finding"] == "ABC-1 is Open"


@pytest.mark.parametrize("mode", ["auto", "always", "never"])
def test_sql_fallback_gating(mode):
    db = FakeDB(lambda sql: [] if "tick_items" in sql and "countIf" not in sql else [{"0:0": 0}])
    script = [tool_call("find", {"need": "the item ABC-1", "values": ["ABC-1"]}),
              tool_call("sql", {"query": "SELECT event_id FROM tick_items LIMIT 1"}),
              tool_call("done", {"finding": "nothing", "evidence": []}), text("nothing")]  # fmt: skip
    made, chat, _ = engine(script, db=db, sql_fallback=mode)
    result = research(made)
    offered = [c["tools"] for c in chat.calls if c["purpose"] == "branch:tick"]
    empty = last_tool(chat.calls[1]["messages"])
    assert empty.startswith("0 rows for [list over tick_items")
    assert "None of the values occurs anywhere" in empty  # the probe ran over every table
    sql_reply = last_tool(chat.calls[2]["messages"])
    branch = branch_trace(result)
    if mode == "never":
        assert all("sql" not in t for t in offered) and "unknown tool 'sql'" in sql_reply
        assert branch["sql_unlocked"] is False
    elif mode == "auto":
        assert "sql" not in offered[0] and "sql" in offered[1]
        assert "The sql tool is now available" in empty and "tick_items(event_id String" in empty
        assert result["trace"]["stats"]["sql"] == 1
    else:
        assert "sql" in offered[0] and "The sql tool is now available" not in empty
        described = next(t for t in chat.calls[0]["specs"] if t["function"]["name"] == "sql")
        assert "tick_items(event_id String" in described["function"]["description"]
    if mode != "never":
        assert branch["calls"][1] == {"tool": "sql", "query": "SELECT event_id FROM tick_items LIMIT 1", "rows": 0, "error": None}  # fmt: skip


def test_uncertain_slot_asks_to_restate_once():
    plan = {"table": {"type": "choice", "choice": "tick_items", "probabilities": {"tick_items": 0.4, "tick_notes": 0.35}}}  # fmt: skip
    script = [tool_call("find", {"need": "the thing ABC-1", "values": ["ABC-1"]}),
              tool_call("find", {"need": "the item record ABC-1", "values": ["ABC-1"]}),
              tool_call("done", {"finding": "Open", "evidence": ["tick:1"]}), text("Open")]  # fmt: skip
    made, chat, _ = engine(script, plan=plan)
    result = research(made)
    first, second = (last_tool(chat.calls[i]["messages"]) for i in (1, 2))
    assert first.startswith("The need is ambiguous (table: tick_items") and "Restate it once" in first  # fmt: skip
    assert second.startswith("1 matching rows")  # the second (restated) call runs despite the doubt
    assert made.db is None  # closed
    outcomes = [c["outcome"] for c in branch_trace(result)["calls"]]
    assert outcomes == ["restate", "rows"]


def test_many_rows_are_filtered_by_relevance():
    rows = [{**ROW, "event_id": f"tick:{i}", "state": "keep" if i % 10 == 0 else "noise"} for i in range(60)]  # fmt: skip
    script = [tool_call("find", {"need": "open items", "values": ["Open"]}),
              tool_call("done", {"finding": "x", "evidence": []}), text("x")]  # fmt: skip
    made, chat, jev = engine(script, db=FakeDB(lambda sql: rows))
    result = research(made)
    reply = last_tool(chat.calls[1]["messages"])
    assert "showing 6 (selected rows); a relevance model chose these 6 from the first 60 rows in this order" in reply  # fmt: skip
    assert reply.count("\n- ") == 6 and "page=1..3 (25 per page)" in reply
    assert "counts over all 60 fetched rows: state: noise (54), keep (6)" in reply
    assert [c["purpose"] for c in jev.calls].count("rows") == 3  # 60 rows in batches of 25
    assert branch_trace(result)["calls"][0]["relevance"] == {"scored": 60, "kept": 6}


def test_timelines_are_not_filtered_and_say_what_is_shown():
    rows = [{**ROW, "event_id": f"tick:{i}", "state": f"{i}" + "x" * 200} for i in range(80)]
    script = [tool_call("find", {"need": "history of ABC-1", "values": ["ABC-1"]}),
              tool_call("find", {"need": "history of ABC-1", "values": ["ABC-1"], "page": 2}),
              tool_call("done", {"finding": "x", "evidence": []}), text("x")]  # fmt: skip
    made, chat, jev = engine(script, db=FakeDB(lambda sql: rows[25:50] if "OFFSET 25" in sql else [{"n": 80}] if "count()" in sql else rows), plan={"shape": choice("timeline")})  # fmt: skip
    result = research(made)
    reply = last_tool(chat.calls[1]["messages"])
    assert reply.startswith("80 matching rows for [timeline over tick_items") and "oldest / lowest first" in reply  # fmt: skip
    shown = reply.count("\n- ")
    assert f"showing {shown} (rows 1-{shown} in this order)" in reply and "page=1..4 (25 per page)" in reply  # fmt: skip
    assert "tick:0 " in reply and "tick:79" not in reply  # never claims rows it did not show
    assert "rows" not in [c["purpose"] for c in jev.calls]  # timelines are not relevance-filtered
    page = last_tool(chat.calls[2]["messages"])
    assert "showing" in page and "(rows 26-" in page and "tick:25 " in page
    first = branch_trace(result)["calls"][0]
    assert first["shown"] == [f"tick:{i}" for i in range(shown)]


def test_findings_are_judged_and_low_ones_dropped(tmp_path):
    (tmp_path / "a.md").write_text("# ABC-1\nstate: wanted Open\n")
    looked = lambda messages: any(m["role"] == "tool" for m in messages)  # noqa: E731
    script = lambda messages, tools, purpose: {  # noqa: E731
        "branch:tick": tool_call("done", {"finding": "tick says Open", "evidence": []})
        if looked(messages)
        else tool_call("find", {"need": "ABC-1", "values": ["ABC-1"]}),  # fmt: skip
        "branch:memory": tool_call("done", {"finding": "memory says nothing", "evidence": []}),
        "report": text("Open"),
    }[purpose]
    made, chat, jev = engine(script, memory=tmp_path)
    made.jev.answer = lambda state, qid, q: noul(0.9 if qid in ("tick", "memory") else 0.1) if "sources" in state else jev_answers()(state, qid, q)  # fmt: skip
    result = research(made)
    assert set(result["trace"]["routing"]["chosen"]) == {"tick", "memory"}
    assert result["trace"]["aggregation"] == {"relevance": {"tick": 0.9, "memory": 0.05}, "factual": {"tick": 0.9, "memory": 0.9},  # fmt: skip
                                              "support": {"tick": 0.5, "memory": 0.0}, "support_by": {"tick": "latest results", "memory": "nothing retrieved"},  # fmt: skip
                                              "timing": 0.9, "kept": ["tick"], "flagged": []}  # fmt: skip
    report = next(c for c in chat.calls if c["purpose"] == "report")["messages"][1]["content"]
    assert "tick says Open" in report and "memory says nothing" not in report


def test_memory_tools_search_read_and_stay_inside_the_root(tmp_path):
    root = tmp_path / "notes"
    (root / "issues").mkdir(parents=True)
    (root / "issues" / "ABC-1.md").write_text("# ABC-1\n" + "\n".join(f"line {i}" for i in range(200)) + "\nwanted: decided to keep X\n")  # fmt: skip
    (root / "other.md").write_text("ABC-1 mentioned wanted\n")
    (tmp_path / "secret.md").write_text("ABC-1 secret\n")
    os.symlink(tmp_path / "secret.md", root / "escape.md")
    script = [tool_call("search", {"need": "decision on ABC-1", "keywords": ["ABC-1", "decided"]}),
              tool_call("read_file", {"path": "issues/ABC-1.md", "from_line": 190}),
              tool_call("read_file", {"path": "../secret.md"}),
              tool_call("read_file", {"path": "escape.md"}),
              tool_call("done", {"finding": "kept X", "evidence": ["issues/ABC-1.md"]}), text("kept X")]  # fmt: skip
    made, chat, _ = engine(script, memory=root)
    made.db = None  # memory only
    result = research(made)
    replies = [last_tool(chat.calls[i]["messages"]) for i in range(1, 5)]
    assert "issues/ABC-1.md:1: # ABC-1" in replies[0] and "secret" not in replies[0]
    assert replies[1].startswith("issues/ABC-1.md lines 190-202 of 202:") and "decided to keep X" in replies[1]  # fmt: skip
    assert "no notes file '../secret.md'" in replies[2] and "no notes file 'escape.md'" in replies[3]  # fmt: skip
    assert result["evidence"] == ["issues/ABC-1.md"]
    assert result["trace"]["routing"]["chosen"] == ["memory"]


def test_provider_failures_propagate_as_infra():
    made, _, _ = engine([])
    made.chat.fail = InfraError("HTTP 402: out of credits")
    with pytest.raises(InfraError, match="402"):
        research(made)
    made, _, _ = engine([tool_call("find", {"need": "x", "values": ["ABC-1"]})])
    made.jev.fail = InfraError("HTTP 429 after 4 attempts")
    with pytest.raises(InfraError, match="429"):
        research(made)


def test_jev_call_budget_degrades_gracefully():
    script = [tool_call("find", {"need": "the item ABC-1", "values": ["ABC-1"]}),
              tool_call("done", {"finding": "no data", "evidence": []}), text("unknown")]  # fmt: skip
    made, chat, _ = engine(script, max_jev_calls=1)
    result = research(made)
    assert "decision budget (1 Jev calls) is used up" in last_tool(chat.calls[1]["messages"])
    assert result["answer"] == "unknown"


def test_a_value_found_elsewhere_is_looked_up_there():
    def answer(sql):
        if "countIf" in sql:  # the probe
            return [{"c7": 2}] if "`tick_notes`" in sql else [{}]  # tick_notes.body
        if (
            "FROM `tick_notes`" in sql
            and "positionCaseInsensitiveUTF8(`tick_notes`.`body`, 'Old title')" in sql
        ):
            return [{"event_id": "note:7", "time": "2030-01-02 00:00:00.000000", "body": "Old title", "key": "ABC-7", "author": "rita"}]  # fmt: skip
        return []

    script = [tool_call("find", {"need": "the item titled 'Old title'", "values": []}),
              tool_call("done", {"finding": "ABC-7", "evidence": ["note:7"]}), text("ABC-7")]  # fmt: skip
    made, chat, _ = engine(script, db=FakeDB(answer), plan={"bind_v1": choice("state")})
    result = research(made)
    reply = last_tool(chat.calls[1]["messages"])
    assert "Looked up 'Old title' in tick_notes.body" in reply and "key: ABC-7" in reply
    assert "sql" not in chat.calls[2]["tools"]  # located: no fallback needed
    find = branch_trace(result)["calls"][0]
    assert (
        find["values"] == ["Old title"]
        and find["outcome"] == "located"
        and result["evidence"] == ["note:7"]
    )


def test_leaked_harmony_tool_calls_are_parsed():
    leak = '<|start|>assistant<|channel|>commentary to=functions.find<|constrain|>json<|message|>{"need": "item ABC-1", "values": ["ABC-1"]}<|call|>'  # fmt: skip
    made, chat, _ = engine([text(leak), tool_call("done", {"finding": "Open", "evidence": ["tick:1"]}), text("Open")])  # fmt: skip
    result = research(made)
    branch = branch_trace(result)
    assert branch["calls"][0]["outcome"] == "rows" and branch["malformed_calls"] == 1
    assert branch["finding"] == "Open"


def test_fallback_sql_times_are_normalised():
    made, chat, _ = engine([tool_call("sql", {"query": "SELECT event_id FROM tick_items WHERE time < '2030-01-05T10:00:00Z'"}),  # fmt: skip
                            tool_call("done", {"finding": "x", "evidence": []}), text("x")], sql_fallback="always")  # fmt: skip
    db = made.db.db
    research(made)
    assert db.queries[-1] == "SELECT event_id FROM tick_items WHERE time < '2030-01-05 10:00:00'"


def test_column_prefixes_are_stripped_from_values():
    script = [tool_call("find", {"need": "state changes of the item", "values": ["key: ABC-1", "state=Open", "KIP-9: x"]}),  # fmt: skip
              tool_call("done", {"finding": "x", "evidence": []}), text("x")]  # fmt: skip
    made, _, _ = engine(script)
    result = research(made)
    assert branch_trace(result)["calls"][0]["values"] == ["ABC-1", "Open", "KIP-9: x"]
    script = [tool_call("find", {"need": "merged items", "values": ["merged_at>=2030-01-01", "until < 2030-02-01T00:00:00Z", "a>b"]}),  # fmt: skip
              tool_call("done", {"finding": "x", "evidence": []}), text("x")]  # fmt: skip
    made, _, _ = engine(script)
    assert branch_trace(research(made))["calls"][0]["values"] == ["2030-01-01", "2030-02-01T00:00:00Z", "a>b"]  # fmt: skip


def test_unsupported_findings_are_flagged_and_marked_partial():
    script = [tool_call("find", {"need": "the item ABC-1", "values": ["ABC-1"]}),
              tool_call("done", {"finding": "ABC-1 was decided by a vote", "evidence": []}), text("vote")]  # fmt: skip
    made, chat, _ = engine(script)
    made.jev.answer = lambda state, qid, q: noul(0.1) if qid == "support" else jev_answers()(state, qid, q)  # fmt: skip
    result = research(made)
    assert result["trace"]["aggregation"]["flagged"] == ["tick"] and result["partial"] is True
    report = next(c for c in chat.calls if c["purpose"] == "report")["messages"][1]["content"]
    assert "(low confidence: NOT confirmed by the retrieved records)" in report


def test_old_tool_results_are_elided_to_fit_the_context():
    from harnext_eval.e2.v2.engine.branch import trim

    messages = [{"role": "system", "content": "s"}] + [{"role": "tool", "tool_call_id": str(i), "content": "x" * 3000} for i in range(6)]  # fmt: skip
    trim(messages, 2500)
    assert [m["content"][:5] for m in messages[1:]] == ["(olde"] * 4 + ["xxxxx"] * 2
    assert trim(messages, 10)[-2]["content"] == "x" * 3000  # the last two results always stay


def test_a_hung_model_call_is_cut_at_the_deadline(monkeypatch):
    from harnext_eval.e2.v2.engine import session as session_module

    monkeypatch.setattr(session_module, "GRACE", 0.1)

    class Hung(FakeChat):
        async def complete(self, messages, **kw):
            if kw.get("purpose", "").startswith("branch"):
                await asyncio.sleep(30)
            return await super().complete(messages, **kw)

    made, _, _ = engine([], deadline_s=0.5, reserve_s=0.0)
    made.chat = Hung([text("report")])
    result = research(made)
    branch = branch_trace(result)
    assert branch["status"] == "failed" and "time budget" in branch["error"]
    assert result["partial"] is True and result["seconds"] < 5


def test_nothing_found_findings_are_dropped_when_another_states_facts(tmp_path):
    (tmp_path / "a.md").write_text("# ABC-1 wanted\n")
    looked = lambda messages: any(m["role"] == "tool" for m in messages)  # noqa: E731
    script = lambda messages, tools, purpose: {  # noqa: E731
        "branch:tick": tool_call("done", {"finding": "ABC-1 is Open", "evidence": []})
        if looked(messages)
        else tool_call("find", {"need": "ABC-1", "values": ["ABC-1"]}),  # fmt: skip
        "branch:memory": tool_call("done", {"finding": "Nothing found.", "evidence": []})
        if looked(messages)
        else tool_call("search", {"need": "ABC-1", "keywords": ["ABC-1"]}),  # fmt: skip
        "report": text("Open"),
    }[purpose]
    made, chat, _ = engine(script, memory=tmp_path)
    base = jev_answers(factual={"tick": 0.95, "memory": 0.05})
    made.jev.answer = lambda state, qid, q: noul(0.9) if "sources" in state and qid in ("tick", "memory") else noul(0.9) if qid.endswith(":relevant") else base(state, qid, q)  # fmt: skip
    result = research(made)
    aggregation = result["trace"]["aggregation"]
    assert aggregation["factual"] == {"tick": 0.95, "memory": 0.05} and aggregation["kept"] == [
        "tick"
    ]
    report = next(c for c in chat.calls if c["purpose"] == "report")["messages"][1]["content"]
    assert "Nothing found." not in report and "ABC-1 is Open" in report


def test_tool_crashes_are_reported_to_the_model():
    made, chat, _ = engine([tool_call("find", {"need": "the item ABC-1", "values": ["ABC-1"]}),
                            tool_call("done", {"finding": "x", "evidence": []}), text("x")])  # fmt: skip

    async def broken(**_):
        raise RuntimeError("disk on fire")

    made.db.sql = lambda *a, **k: broken()
    research(made)
    assert last_tool(chat.calls[1]["messages"]) == "error: find failed: RuntimeError: disk on fire"


def test_provider_rejections_become_infra_with_usage():
    from harnext_eval.e2.v2.engine.clients import ProviderError

    made, _, _ = engine([tool_call("find", {"need": "x", "values": ["ABC-1"]})])
    base = made.jev.answer

    def answer(state, qid, q):
        if qid == "shape":
            raise ProviderError("HTTP 400: bad schema")
        return base(state, qid, q)

    made.jev.answer = answer
    with pytest.raises(InfraError, match="provider rejected a request: HTTP 400") as caught:
        research(made)
    assert caught.value.usage["chat"]["calls"] == 1 and caught.value.usage["jev"]["calls"] == 1  # spent before  # fmt: skip


def test_a_too_large_prompt_is_shrunk_once():
    from harnext_eval.e2.v2.engine.clients import PromptTooLarge

    class Picky(FakeChat):
        refused = False

        async def complete(self, messages, **kw):
            if not self.refused and sum(m["role"] == "tool" for m in messages) >= 2:
                self.refused = True
                raise PromptTooLarge("HTTP 413: too large")
            return await super().complete(messages, **kw)

    finds = [tool_call("find", {"need": f"item {i}", "values": [f"ABC-{i}"]}) for i in range(3)]
    made, _, _ = engine([], branch_turns=5)
    made.chat = Picky([*finds, tool_call("done", {"finding": "ok", "evidence": []}), text("ok")])
    result = research(made)
    assert branch_trace(result)["finding"] == "ok"
    retried = [c["messages"] for c in made.chat.calls if c["purpose"] == "branch:tick"]
    assert any("older result elided" in str(m) for m in retried[-1])


def test_cancelled_calls_are_metered(monkeypatch):
    from harnext_eval.e2.v2.engine import session as session_module

    monkeypatch.setattr(session_module, "GRACE", 0.1)

    class Hung(FakeChat):
        async def complete(self, messages, **kw):
            if kw.get("purpose", "").startswith("branch"):
                await asyncio.sleep(30)
            return await super().complete(messages, **kw)

    made, _, _ = engine([], deadline_s=0.5, reserve_s=0.0)
    made.chat = Hung([text("report")])
    result = research(made)
    cancelled = [c for c in result["trace"]["calls"] if c["cancelled"]]
    assert (
        cancelled
        and cancelled[0]["unmetered_usd"] > 0
        and result["usage"]["chat"]["cancelled"] == 1
    )


def test_several_sources_can_share_one_agent(tmp_path):
    script = [tool_call("find", {"need": "the item ABC-1", "values": ["ABC-1"]}),
              tool_call("done", {"finding": "ABC-1 is Open [tick:1]", "evidence": ["tick:1"]}), text("Open")]  # fmt: skip
    made, chat, _ = engine(script, cross_source="single")
    made.jev.answer = lambda state, qid, q: noul(0.9) if "sources" in state else jev_answers()(state, qid, q)  # fmt: skip
    result = research(made)
    assert [b["source"] for b in result["trace"]["branches"]] == ["mail+tick"]
    system = chat.calls[0]["messages"][0]["content"]
    assert "MAIL and TICK" in system and "mail_msgs" in system and "tick_items" in system
    table_question = next(c for c in made.jev.calls if c["purpose"] == "plan:table")["questions"][
        "table"
    ]
    assert set(table_question["criteria"]) == {"mail_msgs", "tick_items", "tick_notes"}


def test_read_thread_shows_relevant_records_in_full():
    thread = [{"event_id": f"note:{i}", "time": f"2030-01-0{i + 1} 00:00:00.000000", "actor": "rita", "body": ("decided to keep X " if i == 2 else "chatter ") * 30} for i in range(5)]  # fmt: skip

    def rows(sql):
        if "UNION ALL" in sql:
            return [{"source": "tick_notes", "event_id": "note:0"}]
        if "AS `k`" in sql:
            return [{"k": "ABC-1"}]
        return thread

    script = [tool_call("read_thread", {"event_id": "note:0", "need": "what was decided"}),
              tool_call("done", {"finding": "decided to keep X [note:2]", "evidence": ["note:2"]}), text("keep X")]  # fmt: skip
    made, chat, _ = engine(script, db=FakeDB(rows))
    made.jev.answer = lambda state, qid, q: noul(0.9 if "decided" in state["rows"][qid] else 0.1) if "rows" in state else jev_answers()(state, qid, q)  # fmt: skip
    result = research(made)
    reply = last_tool(chat.calls[1]["messages"])
    assert reply.startswith("5 records of tick_notes with key = 'ABC-1', oldest first; 1 relevant to the need shown in full")  # fmt: skip
    assert "- event_id: note:2 | time: 2030-01-03T00:00:00.000000Z | actor: rita\n  decided to keep X" in reply  # fmt: skip
    assert "- event_id: note:0 | time: 2030-01-01T00:00:00.000000Z | actor: rita | first words: chatter" in reply  # fmt: skip
    assert result["evidence"] == ["note:2"]


def test_a_restated_call_may_be_repeated_and_shape_doubt_does_not_block():
    plan = {"table": {"type": "choice", "choice": "tick_items", "probabilities": {"tick_items": 0.4, "tick_notes": 0.35}}}  # fmt: skip
    call = tool_call("find", {"need": "the thing ABC-1", "values": ["ABC-1"]})
    made, chat, _ = engine([call, call, tool_call("done", {"finding": "Open", "evidence": ["tick:1"]}), text("Open")], plan=plan)  # fmt: skip
    result = research(made)
    assert [c["outcome"] for c in branch_trace(result)["calls"]] == ["restate", "rows"]
    assert "or repeat the call to run it as it is" in last_tool(chat.calls[1]["messages"])
    shape = {"shape": {"type": "choice", "choice": "list", "probabilities": {"list": 0.5, "lookup": 0.45}}}  # fmt: skip
    made, chat, _ = engine([call, tool_call("done", {"finding": "Open", "evidence": ["tick:1"]}), text("Open")], plan=shape)  # fmt: skip
    result = research(made)
    reply = last_tool(chat.calls[1]["messages"])
    assert reply.startswith("1 matching rows") and "the kind of result was uncertain ({'list': 0.5, 'lookup': 0.45})" in reply  # fmt: skip


def test_json_shaped_findings_are_unwrapped_and_reports_see_cited_rows():
    finding = '{"evidence": ["tick:1"], "finding": "ABC-1 is Open"}'
    script = [tool_call("find", {"need": "the item ABC-1", "values": ["ABC-1"]}), tool_call("done", {"finding": finding, "evidence": []}), text("Open")]  # fmt: skip
    made, chat, _ = engine(script)
    result = research(made)
    branch = branch_trace(result)
    assert branch["finding"] == "ABC-1 is Open" and branch["evidence"] == ["tick:1"]
    report = next(c for c in chat.calls if c["purpose"] == "report")["messages"][1]["content"]
    assert "Records it cites:\n- event_id: tick:1 | time: 2030-01-03T10:00:00.000000Z | key: ABC-1 | state: Open" in report  # fmt: skip


def test_units_and_roundings_are_read_generously():
    from harnext_eval.e2.v2.engine.plan import ROUNDINGS, UNITS
    from harnext_eval.e2.v2.engine.rawtools import ROUNDING_WORDS, UNIT_WORDS, spelled

    assert [spelled(u, UNITS, UNIT_WORDS) for u in ("Day", "days", "h", "fortnight")] == ["days", "days", "hours", "fortnight"]  # fmt: skip
    assert [spelled(r, ROUNDINGS, ROUNDING_WORDS) for r in ("floor", "Round", "exact")] == ["down", "nearest", "exact"]  # fmt: skip


def test_model_sql_results_repeat_their_query():
    script = [tool_call("sql", {"query": "SELECT event_id FROM tick_items WHERE state = 'Open'"}), tool_call("done", {"finding": "x", "evidence": []}), text("x")]  # fmt: skip
    made, chat, _ = engine(script, sql_fallback="always")
    research(made)
    assert last_tool(chat.calls[1]["messages"]).startswith("1 rows for the query SELECT event_id FROM tick_items WHERE state = 'Open'")  # fmt: skip


def test_an_unreadable_reply_is_retried_once_then_taken_as_text():
    from harnext_eval.e2.v2.engine.clients import ModelOutputError

    class Garbled(FakeChat):
        async def complete(self, messages, **kw):
            if kw.get("purpose") == "branch:tick":
                error = ModelOutputError("HTTP 500: error parsing tool call")
                error.raw = "ABC-1 is Open, I think"
                self.calls.append({"messages": list(messages), "purpose": "branch:tick"})
                raise error
            return await super().complete(messages, **kw)

    made, _, _ = engine([])
    made.chat = Garbled([text("Open")])
    result = research(made)
    branch = branch_trace(result)
    assert branch["finding"] == "ABC-1 is Open, I think" and branch["malformed_calls"] == 1
    asked = [c["messages"] for c in made.chat.calls if c["purpose"] == "branch:tick"]
    assert asked[1][-1]["content"].startswith("Your last reply could not be read as a tool call")


def test_calls_in_flight_when_a_branch_is_cancelled_are_metered():
    class Slow(FakeChat):
        async def complete(self, messages, **kw):
            if kw.get("purpose", "").startswith("branch"):
                await asyncio.sleep(30)  # slow, not failing: no retry, no error
            return await super().complete(messages, **kw)

    made, _, _ = engine([], deadline_s=1.5, reserve_s=1.0)
    made.chat = Slow([text("report")])
    result = research(made)
    cancelled = [c for c in result["trace"]["calls"] if c["cancelled"]]
    assert len(cancelled) == 1 and cancelled[0]["purpose"] == "branch:tick" and cancelled[0]["unmetered_usd"] > 0  # fmt: skip
    assert result["usage"]["chat"]["cancelled"] == 1 and result["partial"] is True


def test_report_verification_can_be_switched_off():
    script = [tool_call("find", {"need": "the item ABC-1", "values": ["ABC-1"]}),
              tool_call("done", {"finding": "ABC-1 is Open [tick:1]", "evidence": ["tick:1"]}), text("ABC-1 uses foo.bar.class [tick:1]")]  # fmt: skip
    made, _, jev = engine(script, verify_report=False)
    result = research(made)
    assert result["answer"] == "ABC-1 uses foo.bar.class [tick:1]" and "verify" not in [c["purpose"] for c in jev.calls]  # fmt: skip
