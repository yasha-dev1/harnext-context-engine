"""Slot filling: canned Jev answers become a typed plan; the questions sent to Jev carry only
catalog identifiers and fixed options; low confidence is reported per slot."""

import asyncio

from harnext_eval.e2.v2.engine.clients import FakeChat, FakeJev, choice, noul
from harnext_eval.e2.v2.engine.config import EngineConfig
from harnext_eval.e2.v2.engine.plan import Compare, Filter, Link
from harnext_eval.e2.v2.engine.session import Session, SharedDB
from harnext_eval.e2.v2.engine.translate import translate

from .engine_fixtures import FakeDB

SCOPE = ["tick_items", "tick_notes"]


def run(answers: dict, need="history of the state of ABC-1 before 2030-01-20", values=("ABC-1", "2030-01-20"), scope=SCOPE):  # fmt: skip
    jev = FakeJev(
        lambda state, qid, q: answers.get(qid) or (noul(0.0) if qid.startswith("out_") else None)
    )
    db = SharedDB(FakeDB())
    session = Session(EngineConfig(), jev, FakeChat([]), db)
    try:
        plan, info = asyncio.run(translate(session, need, list(values), scope))
    finally:
        db.close()
    return plan, info, jev.calls


TIMELINE = {"table": choice("tick_items"), "shape": choice("timeline"), "link": choice("none"),
            "bind_v1": choice("key"), "bind_v2": choice("at"), "op_v2": choice("before"),
            "out_state": noul(0.9), "out_at": noul(0.8), "out_body": noul(0.1), "order": choice("at")}  # fmt: skip


def test_two_decision_calls_fill_a_timeline_plan():
    plan, info, calls = run(TIMELINE)
    assert len(calls) == 2 and [c["purpose"] for c in calls] == ["plan:table", "plan:slots"]
    assert (plan.table, plan.shape, plan.order, plan.link) == ("tick_items", "timeline", "at", None)
    assert plan.filters == [Filter("tick_items", "key", "eq", "ABC-1"), Filter("tick_items", "at", "before", "2030-01-20")]  # fmt: skip
    assert plan.columns == ["key", "at", "state"]  # bound columns first, then the chosen outputs
    assert info["uncertain"] == []
    first, second = calls
    assert set(first["questions"]) == {"table", "shape", "link", "compares", "empties"}
    assert set(first["questions"]["table"]["criteria"]) == set(SCOPE)
    assert set(first["questions"]["link"]["criteria"]) == {"none", "tick_items", "tick_notes", "mail_msgs"}  # fmt: skip
    assert first["state"] == {"need": "history of the state of ABC-1 before 2030-01-20", "values": {"v1": "ABC-1", "v2": "2030-01-20"}}  # fmt: skip
    # a date binds only to time columns; a timeline orders only by time; no direction asked
    assert set(second["questions"]["bind_v2"]["criteria"]) == {"time", "at", "none"}
    assert set(second["questions"]["order"]["criteria"]) == {"time", "at"}
    assert (
        "direction" in second["questions"] and "group" not in second["questions"]
    )  # newest or oldest first
    assert "values_mode" in second["questions"]  # two values: any or all
    assert "raw" not in second["questions"]["bind_v1"]["criteria"]
    assert not any(q.startswith(("out_raw", "out_text", "out_event_id")) for q in second["questions"])  # fmt: skip


def test_search_count_link_and_none_bindings():
    answers = {"table": choice("tick_items"), "shape": choice("search"), "link": choice("none"),
               "bind_v1": choice("body"), "bind_v2": choice("none"), "direction": choice("desc")}  # fmt: skip
    plan, _, _ = run(answers, "items that mention a storm", ["storm", "noise"])
    assert plan.filters == [Filter("tick_items", "body", "contains", "storm")] and plan.desc
    answers = {"table": choice("tick_items"), "shape": choice("count"), "link": choice("tick_notes"),
               "bind_v1": choice("tick_notes.author"), "group": choice("state")}  # fmt: skip
    plan, _, calls = run(answers, "how many items per state did rita comment on", ["rita"])
    assert plan.link == Link("tick_notes", "key", "key") and plan.group == "state"
    assert plan.filters == [Filter("tick_notes", "author", "eq", "rita")]
    assert "tick_notes.author" in calls[1]["questions"]["bind_v1"]["criteria"]


def test_a_link_without_a_filter_or_join_is_dropped():
    answers = {"table": choice("tick_items"), "shape": choice("list"), "link": choice("tick_notes"),
               "bind_v1": choice("key")}  # fmt: skip
    plan, _, _ = run(answers, "the item ABC-1", ["ABC-1"])
    assert plan.link is None
    answers = {"table": choice("tick_notes"), "shape": choice("list"), "link": choice("mail_msgs"),
               "bind_v1": choice("key")}  # fmt: skip
    plan, info, _ = run(answers, "notes of ABC-1", ["ABC-1"])
    assert plan.link is None and info["link"] is None  # tick_notes and mail_msgs are not joined


def test_low_confidence_slots_are_reported():
    answers = dict(TIMELINE, table={"type": "choice", "choice": "tick_items", "probabilities": {"tick_items": 0.45, "tick_notes": 0.4}},  # fmt: skip
                   bind_v1=choice("key", 0.3))  # fmt: skip
    _, info, _ = run(answers)
    slots = {u["slot"]: u for u in info["uncertain"]}
    assert set(slots) == {"table", "column for value 'ABC-1'"}
    assert slots["table"]["alternatives"] == {"tick_items": 0.45, "tick_notes": 0.4}
    assert slots["table"]["confidence"] == 0.45


def test_unknown_choices_fall_back_to_known_options():
    answers = dict(TIMELINE, table={"type": "choice", "choice": "DROP TABLE", "probabilities": {"DROP TABLE": 0.9, "tick_notes": 0.1}})  # fmt: skip
    plan, _, _ = run(answers)
    assert plan.table == "tick_notes"


def test_a_single_table_scope_asks_no_table_question():
    plan, _, calls = run({"shape": choice("list"), "bind_v1": choice("subject")}, "mail about x", ["x"], scope=["mail_msgs"])  # fmt: skip
    assert "table" not in calls[0]["questions"] and plan.table == "mail_msgs"


def test_values_quoted_in_the_need_are_added():
    from harnext_eval.e2.v2.engine.translate import clean

    assert clean(["ABC-1", " ABC-1 "], "changes of ABC-1 where field is 'fixVersion' or \"Fix Version\"") == ["ABC-1", "fixVersion", "Fix Version"]  # fmt: skip
    assert clean("x") == ["x"] and clean(None) == []


def test_the_runner_up_table_is_tried_when_a_value_stays_unused():
    def answers(state, qid, q):
        if qid == "table":
            return {"type": "choice", "choice": "tick_items", "probabilities": {"tick_items": 0.8, "tick_notes": 0.2}, "confidence": 0.8}  # fmt: skip
        if qid == "bind_v2":  # only notes carry an author
            return choice("author" if state["records"].startswith("tick_notes") else "none")
        found = {"shape": choice("list"), "link": choice("none"), "bind_v1": choice("key")}.get(qid)
        return found or (noul(0.0) if qid.startswith("out_") else None)

    jev = FakeJev(answers)
    db = SharedDB(FakeDB())
    session = Session(EngineConfig(), jev, FakeChat([]), db)
    try:
        plan, info = asyncio.run(translate(session, "notes of ABC-1 by rita", ["ABC-1", "rita"], SCOPE))  # fmt: skip
    finally:
        db.close()
    assert plan.table == "tick_notes" and len(jev.calls) == 3
    assert plan.filters == [Filter("tick_notes", "key", "eq", "ABC-1"), Filter("tick_notes", "author", "eq", "rita")]  # fmt: skip
    assert info["switched"] == {"from": "tick_items", "to": "tick_notes", "unused_before": ["rita"], "unused_after": []}  # fmt: skip
    assert info["unused_values"] == []


def test_a_value_can_bind_to_a_column_and_its_sibling():
    answers = {"shape": choice("search"), "bind_v1": choice("refs|body_refs")}
    plan, _, calls = run(answers, "messages discussing ABC-1", ["ABC-1"], scope=["mail_msgs"])
    assert "refs|body_refs" in calls[1]["questions"]["bind_v1"]["criteria"]
    assert plan.filters == [Filter("mail_msgs", "refs", "contains", "ABC-1", ("body_refs",))]


def test_operators_exclusion_ranges_compares_empty_and_distinct():
    answers = {"table": choice("tick_items"), "shape": choice("count"), "link": choice("tick_notes"),
               "compares": noul(0.95), "empties": noul(0.95),
               "bind_v1": choice("tick_notes.author"), "op_v1": choice("eq"),
               "bind_v2": choice("state"), "op_v2": choice("ne"),
               "bind_v3": choice("n"), "op_v3": choice("after"),
               "compare": choice("state!=tick_notes.author"), "empty": choice("body:null"),
               "group": choice("state"), "distinct": choice("key")}  # fmt: skip
    plan, _, calls = run(answers, "how many items by rita that are not closed, n at least 3", ["rita", "Closed", "3"])  # fmt: skip
    assert plan.filters == [Filter("tick_notes", "author", "eq", "rita"), Filter("tick_items", "state", "ne", "Closed"),  # fmt: skip
                            Filter("tick_items", "n", "after", "3"), Filter("tick_items", "body", "null", "")]  # fmt: skip
    assert plan.compares == [Compare("state", "ne", "author", True)]
    assert (plan.group, plan.distinct, plan.link.table) == ("state", "key", "tick_notes")
    second = calls[1]["questions"]
    assert set(second["op_v3"]["criteria"]) == {
        "eq",
        "from",
        "after",
        "through",
        "before",
        "ne",
    }  # a number
    assert set(second["op_v2"]["criteria"]) == {"eq", "ne"}  # text
    assert "key!=state" in second["compare"]["criteria"] and "state!=tick_notes.author" in second["compare"]["criteria"]  # fmt: skip
    assert not any(q.startswith("out_") for q in second)  # a count needs no output columns
    assert second["bind_v1"]["criteria"]["tick_notes.author"] == "The column described in `columns.tick_notes__author`."  # fmt: skip
    assert calls[1]["state"]["columns"]["tick_notes__author"].startswith("The author of a tick_notes row.")  # meanings once, in the state  # fmt: skip


def test_no_compare_or_empty_question_unless_call_one_says_so():
    _, _, calls = run(TIMELINE)
    assert "compare" not in calls[1]["questions"] and "empty" not in calls[1]["questions"]


def test_values_shape_asks_for_the_value_column():
    answers = {"table": choice("tick_items"), "shape": choice("values"), "link": choice("none"),
               "bind_v1": choice("key"), "group": choice("tags")}  # fmt: skip
    plan, _, calls = run(answers, "all tags of ABC-1", ["ABC-1"])
    assert (plan.shape, plan.group) == ("values", "tags") and "none" not in calls[1]["questions"]["group"]["criteria"]  # fmt: skip
