"""Report verification and evidence-based support: values and citations are checked against what
the research call retrieved (deterministically), invented values trigger one regeneration and are
then removed, Jev's verdict on the report triggers one more, and a finding whose cited rows hold
its values is never dropped. Also the combine step (a SQL time comparison across branches)."""

import asyncio

from harnext_eval.e2.v2.engine import verify
from harnext_eval.e2.v2.engine.clients import FakeChat, FakeJev, choice, noul, text, tool_call
from harnext_eval.e2.v2.engine.config import EngineConfig
from harnext_eval.e2.v2.engine.engine import Engine

from .engine_fixtures import FakeDB

ROW = {"event_id": "tick:1", "time": "2030-01-03 10:00:00.000000", "key": "ABC-1", "state": "Open"}
FIND = tool_call("find", {"need": "the state of item ABC-1", "values": ["ABC-1"]})


def test_checkable_values_are_defined_generically():
    report = 'Status "Patch Available" of ABC\u20111 set 2030-01-03T10:00:00.123000Z in 3.4.0 by Rita Moreno; see src/a/B.java, PR 13095 and CamelCase, 2 times.'  # fmt: skip
    assert verify.values_in(report) == ["ABC-1", "2030-01-03T10:00:00.123000Z", "3.4.0", "src/a/B.java", "PR", "13095", "CamelCase", "2", "Patch Available", "Rita Moreno"]  # fmt: skip
    assert verify.values_in("plain words only, nothing special here.") == []
    corpus = "- event_id: tick:1 | key: ABC-1 | state: Patch Available | at: 2030-01-03T10:00:00.123000Z | files: src/a/B.java | pr: 13095 | by: rita moreno | camelcase 2"  # fmt: skip
    assert verify.unverified(report, corpus, known="version 3.4.0 PR") == []
    assert verify.unverified(report, corpus) == [
        "3.4.0"
    ]  # "PR" matches "pr" (letters compare case-insensitively)
    assert verify.unverified("at 2030-01-03T10:00:00Z", corpus) == [
        "2030-01-03T10:00:00Z"
    ]  # precision lost


def test_citations_keep_only_shown_ids_one_per_bracket():
    ids = {"jira:1:change:2:0", "jira:1:change:3:0"}
    report = 'Open [jira:1:change:2:0] and set by X [issue #12345]. JSON ["3.3.3", "3.4.0"] and [1, 2].\n```\n[kept: in code]\n```'  # fmt: skip
    cleaned, removed = verify.citations(report, ids)
    assert cleaned == 'Open [jira:1:change:2:0] and set by X . JSON ["3.3.3", "3.4.0"] and [1, 2].\n```\n[kept: in code]\n```'  # fmt: skip
    assert removed == ["issue", "#12345"]
    garbled, _ = verify.citations("x [event_id: jira:1:change:2:0, time: 2030-01-01, jira:1:change:3:0 | fake:9]", ids)  # fmt: skip
    assert garbled == "x [jira:1:change:2:0][jira:1:change:3:0]"
    assert verify.cited(garbled, ids) == ["jira:1:change:2:0", "jira:1:change:3:0"]
    nested, removed = verify.citations(
        "n [0 rows for [list over t; x = 1]] ok [jira:1:change:2:0]", ids
    )
    assert nested == "n  ok [jira:1:change:2:0]" and "rows" in removed


def test_strip_drops_lines_with_unverified_values():
    out = verify.strip("Answer: foo.bar.class\nThe issue is ABC-1.", ["foo.bar.class"])
    assert out.startswith("The issue is ABC-1.") and "Could not be determined" in out and "foo.bar.class (removed" in out  # fmt: skip


def engine(chat_script, jev=None, rows=(ROW,), **config):
    made = Engine(db=FakeDB(lambda sql: list(rows)), config=EngineConfig(**{"backoff_s": 0, **config}),
                  jev=FakeJev(jev or answers()), chat=FakeChat(chat_script))  # fmt: skip
    return made


def answers(verdicts=("supported",), support=0.9, extra=None):
    verdicts = list(verdicts)

    def answer(state, qid, q):
        if "sources" in state:
            return noul(0.9 if qid == "tick" else 0.1)
        if qid == "verdict":
            return choice(verdicts.pop(0) if len(verdicts) > 1 else verdicts[0])
        if qid == "support":
            return noul(support)
        found = {"table": choice("tick_items"), "shape": choice("list"), "link": choice("none"),
                 "bind_v1": choice("key"), "op_v1": choice("eq"), **(extra or {})}.get(qid)  # fmt: skip
        return found or (noul(0.0) if qid.startswith("out_") else None)

    return answer


def research(made, question="What is the state of ABC-1?"):
    async def go():
        try:
            return await made.research(question)
        finally:
            await made.aclose()

    return asyncio.run(go())


def reports(made):
    return [c for c in made.chat.calls if c["purpose"] == "report"]


def test_invented_values_are_regenerated_then_removed():
    done = tool_call("done", {"finding": "ABC-1 is Open [tick:1]", "evidence": ["tick:1"]})
    made = engine([FIND, done, text("ABC-1 uses property foo.bar.class [tick:1] [issue #12345]"),
                   text("ABC-1 uses property foo.bar.class [tick:1]")])  # fmt: skip
    result = research(made)
    check = result["trace"]["verification"]
    assert check["removed_citations"] == ["issue", "#12345"]
    assert check["unverified_values"] == ["foo.bar.class"] and check["regenerated"] == ["values"]
    assert check["stripped"] == ["foo.bar.class"] and result["partial"] is True
    assert "foo.bar.class" not in result["answer"].split("Could not be determined")[0]
    assert "occur neither in the records nor in the findings: foo.bar.class" in reports(made)[1]["messages"][-1]["content"]  # fmt: skip


def test_a_regeneration_that_fixes_the_values_is_kept():
    done = tool_call("done", {"finding": "ABC-1 is Open [tick:1]", "evidence": ["tick:1"]})
    made = engine([FIND, done, text("ABC-1 is Closed since 2031-05-05"), text("ABC-1 is Open [tick:1]")])  # fmt: skip
    result = research(made)
    assert result["answer"] == "ABC-1 is Open [tick:1]" and result["partial"] is False
    assert result["trace"]["verification"]["stripped"] == []


def test_unsupported_verdict_regenerates_once_then_marks_partial():
    done = tool_call("done", {"finding": "ABC-1 is Open [tick:1]", "evidence": ["tick:1"]})
    made = engine([FIND, done, text("ABC-1 is Open and was reopened by a vote"), text("ABC-1 is Open")],  # fmt: skip
                  jev=answers(verdicts=("unsupported", "supported")))  # fmt: skip
    result = research(made)
    check = result["trace"]["verification"]
    assert check["regenerated"] == ["unsupported"] and check["verdict"] == "supported"
    assert result["answer"] == "ABC-1 is Open" and result["partial"] is False
    made = engine([FIND, done, text("ABC-1 was reopened by a vote"), text("ABC-1 is not known")],  # fmt: skip
                  jev=answers(verdicts=("unsupported",)))  # fmt: skip
    result = research(made)
    check = result["trace"]["verification"]
    assert check["verdict"] == "unsupported" and result["partial"] is True
    assert result["answer"] == "ABC-1 was reopened by a vote"  # a rewrite that does not verify is not kept  # fmt: skip
    assert check["drafts"] == ["ABC-1 was reopened by a vote", "ABC-1 is not known"]


def test_nothing_found_is_reported_plainly():
    done = tool_call("done", {"finding": "Nothing relevant was found.", "evidence": []})
    made = engine([FIND, done, text("Not found in the records.")], jev=answers(verdicts=("declined",)))  # fmt: skip
    result = research(made)
    assert result["trace"]["verification"]["verdict"] == "declined"
    assert result["answer"] == "Not found in the records."


def test_support_uses_the_cited_rows():
    made = engine([FIND, tool_call("done", {"finding": "ABC-1 is Open", "evidence": ["tick:1"]}), text("Open [tick:1]")])  # fmt: skip
    result = research(made)
    assert result["trace"]["aggregation"]["support_by"] == {"tick": "contained in cited rows"}
    assert "support" not in [c["purpose"] for c in made.jev.calls]  # no Jev call needed
    made = engine([FIND, tool_call("done", {"finding": "ABC-2 is Closed", "evidence": ["tick:1"]}), text("x")], jev=answers(support=0.1))  # fmt: skip
    result = research(made)
    support_call = next(c for c in made.jev.calls if c["purpose"] == "support")
    assert support_call["state"]["records"].startswith("- event_id: tick:1") and "key: ABC-1" in support_call["state"]["records"]  # fmt: skip
    assert result["trace"]["aggregation"]["flagged"] == [
        "tick"
    ]  # the only finding: kept, low confidence


def test_combine_compares_the_times_of_events_from_two_branches(tmp_path):
    (tmp_path / "n.md").write_text("# ABC-1\nmerged 2030-01-04\n")
    times = [{"a_time": "2030-01-03 10:00:00.500000", "b_time": "2030-01-02 09:00:00.000000", "a_first": 0,
              "same_instant": 0, "milliseconds_a_to_b": -90000500, "amount": 1}]  # fmt: skip

    def rows(sql):
        if "UNION ALL" in sql:
            return [{"source": "tick_items", "event_id": "tick:1"}, {"source": "mail_msgs", "event_id": "note:1"}]  # fmt: skip
        if "CROSS JOIN" in sql:
            return times
        if "FROM `mail_msgs`" in sql:
            return [{"event_id": "note:1", "time": "2030-01-02 09:00:00.000000", "key": "ABC-1"}]
        return [ROW]

    def script(messages, tools, purpose):
        looked = any(m["role"] == "tool" for m in messages)
        if purpose == "branch:tick":
            return tool_call("done", {"finding": "ABC-1 set at tick:1", "evidence": ["tick:1"]}) if looked else FIND  # fmt: skip
        if purpose == "branch:mail":
            return tool_call("done", {"finding": "ABC-1 noted at note:1", "evidence": ["note:1"]}) if looked else tool_call("find", {"need": "notes of ABC-1", "values": ["ABC-1"]})  # fmt: skip
        return text("tick:1 came after note:1 [tick:1]")

    def jev(state, qid, q):
        if "sources" in state:
            return noul(0.9)
        if qid == "timing":
            return noul(0.9)
        if qid in ("first", "second"):
            return choice("e1" if qid == "first" else "e2")
        if qid in ("unit", "rounding"):
            return choice("days" if qid == "unit" else "down")
        return answers(extra={"table": choice("tick_items") if "item" in state.get("need", "ABC-1 item") else choice("tick_notes")})(state, qid, q)  # fmt: skip

    made = Engine(db=FakeDB(rows), config=EngineConfig(backoff_s=0), jev=FakeJev(jev), chat=FakeChat(script))  # fmt: skip
    result = research(made, "Which came first for ABC-1, the item change or the note, and how many days apart?")  # fmt: skip
    combined = next(b for b in result["trace"]["branches"] if b["source"] == "combined")
    assert combined["evidence"] == [
        "note:1",
        "tick:1",
    ]  # e1 = the first cited event in routing order
    assert combined["finding"] == "Computed from the records: note:1 at 2030-01-03T10:00:00.500000Z; tick:1 at 2030-01-02T09:00:00.000000Z; tick:1 is earlier; difference 1 days (rounding: down); exact difference 90000500 milliseconds"  # fmt: skip
    assert combined["calls"][0]["unit"] == "days" and combined["calls"][0]["rounding"] == "down"
    assert "CROSS JOIN" in combined["calls"][0]["sql"] and result["trace"]["stats"]["compare_times"] == 1  # fmt: skip
    report = next(c for c in made.chat.calls if c["purpose"] == "report")["messages"][1]["content"]
    assert "## Findings from the combined records" in report


def test_a_finding_whose_new_values_were_retrieved_is_kept_without_jev():
    rows = [ROW, {**ROW, "event_id": "tick:2", "state": "Merged", "key": "ABC-7"}]

    def run(finding):
        made = engine([FIND, tool_call("done", {"finding": finding, "evidence": ["tick:1"]}), text("Open [tick:1]")], rows=rows, jev=answers(support=0.0))  # fmt: skip
        return research(made)["trace"]["aggregation"]

    kept = run(
        "ABC-1 is Open [tick:1]; the related item ABC-7 is Merged"
    )  # ABC-7: not cited, but retrieved
    assert kept["support_by"] == {"tick": "contained in retrieved rows"} and kept["support"] == {"tick": 1.0}  # fmt: skip
    judged = run(
        "ABC-1 is Open [tick:1]; ABC-7 is Merged; no event gh.guess was seen"
    )  # half invented
    assert judged["support_by"] == {"tick": "cited rows"} and judged["flagged"] == ["tick"]
