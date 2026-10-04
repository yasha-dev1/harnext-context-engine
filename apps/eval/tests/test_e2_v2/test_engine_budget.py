"""Engine v2: tool-call budget feedback. Every tool result ends with "Tool call budget: m/n";
once the budget is used (or the deadline reserve is reached) calls are not run and the model is
told to answer; the branch ends with the model's own finding. With budget feedback off, the
prompts, the rendered branch prompts and the config hash inputs are byte-identical to code
1a21f39f (engine_v1_snapshot.json, captured from that code before this change)."""

import asyncio
import json
from pathlib import Path

from harnext_eval.e2.v2.engine import prompts
from harnext_eval.e2.v2.engine.clients import FakeChat, FakeJev, choice, noul, text, tool_call
from harnext_eval.e2.v2.engine.config import TRANSPORT, V2, canonical, config_hash, for_endpoint
from harnext_eval.e2.v2.engine.engine import Engine
from harnext_eval.e2.v2.engine.session import Session

from .engine_fixtures import FakeDB

SNAPSHOT = json.loads((Path(__file__).parent / "engine_v1_snapshot.json").read_text())
ROW = {"event_id": "tick:1", "time": "2030-01-03 10:00:00.000000", "key": "ABC-1", "state": "Open"}


def answers(state, qid, q):
    if "sources" in state:
        return noul(0.9 if qid == "tick" else 0.1)
    found = {"table": choice("tick_items"), "shape": choice("list"), "link": choice("none"),
             "bind_v1": choice("key"), "op_v1": choice("eq")}.get(qid)  # fmt: skip
    return found or (noul(0.0) if qid.startswith("out_") else None)


def engine(script, preset="openrouter-b8", **overrides):
    config = for_endpoint(preset, backoff_s=0, **overrides)
    return Engine(
        db=FakeDB(lambda sql: [ROW]), config=config, jev=FakeJev(answers), chat=FakeChat(script)
    )


def research(made):
    async def go():
        try:
            return await made.research("What is the state of ABC-1?")
        finally:
            await made.aclose()

    return asyncio.run(go())


def find(i):
    return tool_call("find", {"need": f"the item ABC-{i}", "values": [f"ABC-{i}"]})


def results(made):
    """The tool results the branch agent saw in its last turn (an answer-now turn included)."""
    last = [c for c in made.chat.calls if c["purpose"] in ("branch:tick", "branch:tick:answer")][-1]
    return [m["content"] for m in last["messages"] if m["role"] == "tool"]


def test_every_result_ends_with_the_budget_line_and_the_system_prompt_states_it():
    made = engine([find(1), find(2), tool_call("done", {"finding": "ABC-1 is Open [tick:1]", "evidence": ["tick:1"]}), text("Open")])  # fmt: skip
    result = research(made)
    branch_calls = [c for c in made.chat.calls if c["purpose"] == "branch:tick"]
    assert branch_calls[0]["messages"][0]["content"].endswith(
        prompts.BUDGET_SYSTEM.format(budget=8)
    )
    tools = [m["content"] for m in branch_calls[-1]["messages"] if m["role"] == "tool"]
    assert [t.rsplit("\n", 1)[1] for t in tools] == [
        "Tool call budget: 1/8",
        "Tool call budget: 2/8",
    ]
    branch = result["trace"]["branches"][0]
    assert (branch["ending"], branch["tool_calls"], branch["status"]) == ("normal", 2, "done")


def test_after_the_budget_calls_are_not_run_and_the_model_answers():
    script = [find(i) for i in range(9)] + [tool_call("done", {"finding": "ABC-1 is Open [tick:1]", "evidence": ["tick:1"]}), text("Open")]  # fmt: skip
    made = engine(script)
    sqls = made.db.db.queries
    result = research(made)
    branch = result["trace"]["branches"][0]
    assert branch["tool_calls"] == 8 and branch["ending"] == "budget" and branch["status"] == "done"
    assert branch["finding"] == "ABC-1 is Open [tick:1]" and result["partial"] is False
    assert results(made)[-1] == prompts.BUDGET_EXHAUSTED + "\n\nTool call budget: 8/8"
    assert results(made)[-2].endswith("Tool call budget: 8/8") and "ABC-7" in results(made)[-2]
    assert not any("ABC-8" in q for q in sqls)  # the ninth find (ABC-8) never ran


def test_rejected_calls_count_and_say_what_was_wrong():
    script = [tool_call("find", '{"need": "x", "values": ['), tool_call("grep", {"q": "x"}), find(1), find(1),
              tool_call("done", {"finding": "Open", "evidence": ["tick:1"]}), text("Open")]  # fmt: skip
    made = engine(script)
    result = research(made)
    tools = results(made)
    assert tools[0].startswith(
        "error: the arguments of find are not a valid JSON object"
    ) and tools[0].endswith("1/8")
    assert tools[1].startswith("error: unknown tool 'grep'") and tools[1].endswith("2/8")
    assert tools[3].startswith("You made this exact find call before") and tools[3].endswith("4/8")
    assert result["trace"]["branches"][0]["tool_calls"] == 4


def test_the_deadline_reserve_stops_tool_calls_and_asks_for_the_answer():
    made = engine([find(1), tool_call("done", {"finding": "nothing yet", "evidence": []}), text("x")], deadline_s=60, reserve_s=20, answer_reserve_s=45)  # fmt: skip
    result = research(made)
    branch = result["trace"]["branches"][0]
    assert results(made)[-1] == prompts.BUDGET_TIME + "\n\nTool call budget: 0/8"
    assert (branch["ending"], branch["tool_calls"], branch["status"]) == ("time", 0, "done")


def test_a_model_that_keeps_calling_tools_gets_one_answer_turn_without_tools():
    made = engine([find(i) for i in range(11)] + [text("Probably Open"), text("report")])
    result = research(made)
    branch = result["trace"]["branches"][0]
    assert (branch["ending"], branch["finding"], branch["status"]) == (
        "forced",
        "Probably Open",
        "done",
    )
    assert [c for c in made.chat.calls if c["purpose"] == "branch:tick:final"][0]["tools"] == []
    assert not branch["finding"].startswith("(unfinished")


def test_old_settings_are_byte_identical_to_code_1a21f39f():
    assert canonical(prompts.texts()) == SNAPSHOT["prompts"]
    for name in ("ollama", "openrouter"):
        config = for_endpoint(name)
        assert canonical(config.model_dump(exclude=TRANSPORT | set(V2))) == SNAPSHOT[f"config_{name}"]  # fmt: skip
        assert config_hash(config) == SNAPSHOT[f"hash_{name}"]
        assert not config.budget_feedback
    made = Engine(db=FakeDB(), config=for_endpoint("openrouter"), jev=FakeJev(), chat=FakeChat([]))
    session = Session(made.config, made.jev, made.chat, made.db)
    assert made.make(session, "tick")[1] == SNAPSHOT["system_tick"]
    assert made.make(session, "mail+tick")[1] == SNAPSHOT["system_mail_tick"]
    made.db.close()


def test_old_settings_keep_the_old_loop():
    made = engine([find(1), tool_call("done", {"finding": "Open", "evidence": ["tick:1"]}), text("Open")], preset="openrouter")  # fmt: skip
    result = research(made)
    assert "Tool call budget" not in results(made)[0] and prompts.BUDGET_SYSTEM[:20] not in made.chat.calls[0]["messages"][0]["content"]  # fmt: skip
    assert result["trace"]["branches"][0]["ending"] == "normal"


def test_presets():
    deepseek = for_endpoint("deepseek-b8")
    assert (deepseek.chat_model, deepseek.chat_price, deepseek.budget_feedback, deepseek.tool_budget) == ("deepseek/deepseek-v4.1-flash", (0.30, 1.20), True, 8)  # fmt: skip
    assert deepseek.chat_options == {"reasoning": {"effort": "low"}, "usage": {"include": True}}
    assert deepseek.temperature == 0.0 and deepseek.seed == 7
    b8 = for_endpoint("openrouter-b8")
    assert b8.chat_model == "openai/gpt-oss-20b" and b8.budget_feedback
    policy = (b8.deadline_s, b8.reserve_s, b8.answer_reserve_s, b8.conclude_s)
    assert policy == (240.0, 90.0, 30.0, 40.0)  # the deadline policy of every b8 preset
    old = for_endpoint("openrouter")
    assert (old.deadline_s, old.reserve_s, old.branch_turns) == (120.0, 20.0, 8)
    flash = for_endpoint("deepseek-flash-b8")
    assert (flash.chat_model, flash.chat_price, flash.chat_options) == ("deepseek/deepseek-v4-flash-0731", (0.018, 0.32), deepseek.chat_options)  # fmt: skip
    presets = ("openrouter", "openrouter-b8", "deepseek-b8", "deepseek-flash-b8")
    assert len({config_hash(for_endpoint(n)) for n in presets}) == 4
    # only the model differs, and the lowest reasoning each accepts for an answer-now call
    same = lambda c: c.model_dump(exclude={"chat_model", "chat_price", "answer_options"})  # noqa: E731
    assert same(b8) == same(deepseek) == same(flash)
    assert b8.answer_options == {"reasoning": {"effort": "low"}}  # gpt-oss: reasoning is mandatory
    assert deepseek.answer_options == flash.answer_options == {"reasoning": {"effort": "none"}}
    for c in (b8, deepseek, flash):
        assert (c.parallel_branches, c.answer_tokens, c.chat_concurrency) == (4, 1500, 1)
    assert (old.parallel_branches, old.answer_tokens, old.answer_options) == (0, 0, {})


def test_a_branch_cut_at_the_deadline_still_answers_itself():
    class Slow(FakeChat):
        async def complete(self, messages, **kw):
            turns = sum(m["role"] == "assistant" for m in messages)
            if kw.get("purpose") == "branch:tick" and turns >= 1:
                await asyncio.sleep(30)  # the second turn never returns in time
            return await super().complete(messages, **kw)

    made = engine([], deadline_s=3, reserve_s=1.5, answer_reserve_s=0.1, conclude_s=5)
    made.chat = Slow(lambda messages, tools, purpose: text("ABC-1 is Open [tick:1]") if purpose.endswith("conclude") or purpose == "report" else find(1))  # fmt: skip
    result = research(made)
    branch = result["trace"]["branches"][0]
    assert (branch["ending"], branch["status"], branch["finding"]) == ("time", "done", "ABC-1 is Open [tick:1]")  # fmt: skip
    conclude = next(c for c in made.chat.calls if c["purpose"] == "branch:tick:conclude")
    assert (
        conclude["tools"] == [] and conclude["messages"][-1]["content"] == prompts.BUDGET_TIME_FINAL
    )
    assert not branch["finding"].startswith("(unfinished")
    assert [
        c for c in result["trace"]["calls"] if c["purpose"] == "branch:tick:conclude"
    ]  # metered


def test_an_interrupted_turn_is_removed_before_concluding():
    from harnext_eval.e2.v2.engine.branch import complete_turns

    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]}, {"role": "tool", "tool_call_id": "a", "content": "r"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "b"}, {"id": "c"}]}, {"role": "tool", "tool_call_id": "b", "content": "r"}]  # fmt: skip
    assert complete_turns(messages) == messages[:4]


def test_a_conclusion_given_as_a_done_call_counts_and_a_timeout_is_metered():
    class Slow(FakeChat):
        async def complete(self, messages, **kw):
            turns = sum(m["role"] == "assistant" for m in messages)
            if kw.get("purpose") == "branch:tick" and turns >= 1:
                await asyncio.sleep(30)
            if kw.get("purpose", "").endswith("conclude") and self.hang:
                await asyncio.sleep(30)
            return await super().complete(messages, **kw)

    reply = lambda messages, tools, purpose: tool_call("done", {"finding": "ABC-1 is Open", "evidence": ["tick:1"]}) if purpose.endswith("conclude") else text("x") if purpose == "report" else find(1)  # noqa: E731  # fmt: skip
    for hang in (False, True):
        made = engine(
            [], deadline_s=3, reserve_s=1.5, answer_reserve_s=0.1, conclude_s=0.5 if hang else 5
        )
        made.chat = Slow(reply)
        made.chat.hang = hang
        result = research(made)
        branch = result["trace"]["branches"][0]
        if not hang:
            assert (branch["ending"], branch["finding"], branch["evidence"]) == (
                "time",
                "ABC-1 is Open",
                ["tick:1"],
            )
        else:
            assert branch["ending"] == "cutoff" and "timed out" in branch["error"]
            assert [
                c
                for c in result["trace"]["calls"]
                if c["purpose"] == "branch:tick:conclude" and c["cancelled"]
            ]


def test_slow_turns_bring_the_answer_now_point_forward():
    class Slowish(FakeChat):
        async def complete(self, messages, **kw):
            if kw.get("purpose") == "branch:tick":
                await asyncio.sleep(0.4)
            return await super().complete(messages, **kw)

    made = engine([], deadline_s=2.2, reserve_s=0.5, answer_reserve_s=0.05)
    made.chat = Slowish(lambda messages, tools, purpose: text("x") if purpose == "report" else tool_call("done", {"finding": "Open", "evidence": []}) if any(m["role"] == "tool" and "Not executed" in m["content"] for m in messages) else find(len(messages)))  # fmt: skip
    result = research(made)
    branch = result["trace"]["branches"][0]
    assert (
        branch["ending"] == "time" and branch["status"] == "done"
    )  # refused early, answered itself


def test_a_reply_spent_on_reasoning_is_asked_again_with_more_room():
    truncated = {"role": "assistant", "content": "", "finish_reason": "length"}
    for preset, retried in (("openrouter-b8", True), ("openrouter", False)):
        made = engine([find(1), tool_call("done", {"finding": "Open", "evidence": ["tick:1"]}), truncated, text("ABC-1 is Open [tick:1]")], preset=preset)  # fmt: skip
        result = research(made)
        purposes = [c["purpose"] for c in result["trace"]["calls"] if c["kind"] == "chat"]
        assert ("report:longer" in purposes) is retried
        if retried:
            assert result["answer"] == "ABC-1 is Open [tick:1]"
            longer = next(c for c in made.chat.calls if c["purpose"] == "report:longer")
            assert (
                longer["messages"]
                == next(c for c in made.chat.calls if c["purpose"] == "report")["messages"]
            )
        else:
            assert result["answer"].startswith("(report not written: time budget)")  # v1 unchanged


def test_deepseek_dsml_tool_calls_in_text_are_parsed():
    from harnext_eval.e2.v2.engine.branch import leaked, without_leaks

    done = '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="done">\n<｜DSML｜parameter name="finding" string="true">ABC-1 is Open</｜DSML｜parameter>\n<｜DSML｜parameter name="evidence" string="false">["tick:1"]</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>'  # fmt: skip
    call = leaked(done)[0]["function"]
    assert call["name"] == "done" and json.loads(call["arguments"]) == {"finding": "ABC-1 is Open", "evidence": ["tick:1"]}  # fmt: skip
    read = '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="read">\n<｜DSML｜parameter name="event_id" string="true">tick:1</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>'  # fmt: skip
    assert without_leaks("Answer: Open. " + read) == "Answer: Open." and without_leaks(read) == ""
    made = engine(
        [text(done), text("Open [tick:1]")]
    )  # a leaked done ends the branch like a real one
    result = research(made)
    branch = result["trace"]["branches"][0]
    assert (branch["finding"], branch["malformed_calls"]) == ("ABC-1 is Open", 1)
    assert branch["unverified_evidence"] == [
        "tick:1"
    ]  # never retrieved in this branch: not evidence


def test_one_stalled_turn_does_not_close_the_tool_window():
    class Stall(FakeChat):
        async def complete(self, messages, **kw):
            if kw.get("purpose") == "branch:tick" and not any(
                m["role"] == "tool" for m in messages
            ):
                await asyncio.sleep(1.0)  # the first turn stalls
            return await super().complete(messages, **kw)

    # deadline 6, reserve 1, answer 0.3, conclude 0.4: a doubled 1-s turn (2 s) is capped at 0.7 s
    made = engine([], deadline_s=6, reserve_s=1, answer_reserve_s=0.3, conclude_s=0.4)
    made.chat = Stall(
        [find(1), tool_call("done", {"finding": "Open", "evidence": ["tick:1"]}), text("Open")]
    )
    result = research(made)
    branch = result["trace"]["branches"][0]
    assert branch["tool_calls"] == 1 and branch["ending"] == "normal"


# ---- follow-up: branches side by side, deterministic order, answer-now calls ----------------


def two_sources(delays: dict, preset="openrouter-b8"):
    """A research call routed to tick (0.9) and mail (0.8): each branch answers with done after
    `delays[source]` seconds; records how many chat calls ran at once."""

    class Timed(FakeChat):
        running = peak = 0

        async def complete(self, messages, **kw):
            source = kw.get("purpose", "").split(":")[1] if kw.get("purpose", "").startswith("branch") else None  # fmt: skip
            Timed.running += 1
            Timed.peak = max(Timed.peak, Timed.running)
            try:
                if source:
                    await asyncio.sleep(delays[source])
            finally:
                Timed.running -= 1
            return await super().complete(messages, **kw)

    def reply(messages, tools, purpose):
        if purpose.startswith("branch:"):
            source = purpose.split(":")[1]
            return tool_call("done", {"finding": f"The {source} records say ABC-1 is Open.", "evidence": []})  # fmt: skip
        return text("report")

    def route(state, qid, q):
        return noul({"tick": 0.9, "mail": 0.8}[qid]) if "sources" in state else noul(0.9)

    config = for_endpoint(preset, backoff_s=0)
    made = Engine(db=FakeDB(lambda sql: [ROW]), config=config, jev=FakeJev(route), chat=Timed(reply))  # fmt: skip
    result = research(made)
    return result, made, Timed.peak


def test_budget_presets_run_branches_side_by_side_and_the_old_preset_one_call_at_a_time():
    _, _, peak = two_sources({"tick": 0.2, "mail": 0.2})
    assert peak == 2  # one chat call per branch at once (cap: parallel_branches)
    _, _, peak = two_sources({"tick": 0.2, "mail": 0.2}, preset="openrouter")
    assert peak == 1  # v1: chat_concurrency 1


def test_the_parallel_gate_is_capped():
    config = for_endpoint("openrouter-b8", parallel_branches=1, backoff_s=0)
    made = Engine(db=FakeDB(), config=config, jev=FakeJev(), chat=FakeChat([]))
    session = Session(made.config, made.jev, made.chat, made.db)
    session.parallel(min(3, config.parallel_branches))
    assert session.chat_gate._value == 1
    made.db.close()


def test_findings_are_ordered_by_source_name_whatever_finishes_first():
    runs = [two_sources({"tick": 0.3, "mail": 0.0}), two_sources({"tick": 0.0, "mail": 0.3})]
    finished = []
    for result, made, _ in runs:
        trace = result["trace"]
        assert [b["source"] for b in trace["branches"]] == [
            "mail",
            "tick",
        ]  # routing order: tick, mail
        assert trace["routing"]["chosen"] == ["tick", "mail"]
        judge = next(c for c in made.jev.calls if c["purpose"] == "judge")
        assert list(judge["state"]["findings"]) == ["mail", "tick"]
        finished.append(sorted(trace["branches"], key=lambda b: b["ended_s"])[0]["source"])
    assert finished == ["mail", "tick"]  # the completion order did differ
    reports = [[c["messages"] for c in made.chat.calls if c["purpose"] == "report"] for _, made, _ in runs]  # fmt: skip
    assert reports[0] == reports[1]  # the report prompt does not depend on timing
    old, _, _ = two_sources({"tick": 0.0, "mail": 0.0}, preset="openrouter")
    assert [b["source"] for b in old["trace"]["branches"]] == ["tick", "mail"]  # v1: routing order


def test_answer_now_calls_use_the_lowest_reasoning_and_a_fixed_allowance():
    made = engine([find(i) for i in range(11)] + [text("Probably Open"), text("report")], preset="deepseek-flash-b8")  # fmt: skip
    research(made)
    by = {c["purpose"]: c for c in made.chat.calls}
    assert "options" not in by["branch:tick"]  # a normal turn: the preset's own options
    for purpose in ("branch:tick:answer", "branch:tick:final"):
        assert (by[purpose]["options"], by[purpose]["max_tokens"]) == ({"reasoning": {"effort": "none"}}, 1500)  # fmt: skip
    assert by["branch:tick:answer"]["tools"] == ["done"] and by["branch:tick:final"]["tools"] == []
    answers = [c for c in made.chat.calls if c["purpose"] == "branch:tick:answer"]
    assert len(answers) == 2  # the turns left after the refused ninth call (budget + answer_turns)


def test_the_conclusion_at_the_cut_off_is_an_answer_now_call():
    class Slow(FakeChat):
        async def complete(self, messages, **kw):
            turns = sum(m["role"] == "assistant" for m in messages)
            if kw.get("purpose") == "branch:tick" and turns >= 1:
                await asyncio.sleep(30)
            return await super().complete(messages, **kw)

    made = engine([], preset="deepseek-flash-b8", deadline_s=3, reserve_s=1.5, answer_reserve_s=0.1, conclude_s=5)  # fmt: skip
    made.chat = Slow(lambda messages, tools, purpose: text("ABC-1 is Open [tick:1]") if purpose.endswith("conclude") or purpose == "report" else find(1))  # fmt: skip
    research(made)
    conclude = next(c for c in made.chat.calls if c["purpose"] == "branch:tick:conclude")
    assert (conclude["options"], conclude["max_tokens"]) == (
        {"reasoning": {"effort": "none"}},
        1500,
    )


def test_chat_client_merges_call_options_over_the_preset():
    import httpx
    from harnext_eval.e2.v2.engine.clients import ChatClient

    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}})  # fmt: skip

    config = for_endpoint("deepseek-flash-b8", backoff_s=0)
    client = ChatClient(config, key="k", http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))  # fmt: skip
    asyncio.run(client.complete([{"role": "user", "content": "q"}], options=config.answer_options, max_tokens=1500))  # fmt: skip
    asyncio.run(client.complete([{"role": "user", "content": "q"}]))
    assert seen[0]["reasoning"] == {"effort": "none"} and seen[0]["max_tokens"] == 1500
    assert seen[0]["usage"] == {"include": True}
    assert seen[1]["reasoning"] == {"effort": "low"} and seen[1]["max_tokens"] == config.max_tokens


def test_a_reply_spent_on_reasoning_is_asked_again_with_the_lowest_reasoning():
    truncated = {"role": "assistant", "content": "", "finish_reason": "length"}
    made = engine([find(1), truncated, tool_call("done", {"finding": "Open", "evidence": ["tick:1"]}), text("ABC-1 is Open [tick:1]")], preset="deepseek-flash-b8")  # fmt: skip
    result = research(made)
    longer = next(c for c in made.chat.calls if c["purpose"] == "branch:tick:longer")
    assert (longer["options"], longer["max_tokens"]) == ({"reasoning": {"effort": "none"}}, 2 * made.config.max_tokens)  # fmt: skip
    assert result["trace"]["branches"][0]["finding"] == "Open"
