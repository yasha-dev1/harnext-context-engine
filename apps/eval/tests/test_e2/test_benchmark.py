from copy import deepcopy

import pytest
from harnext_eval.e2.benchmark import body_at, historical_task, timestamp
from harnext_eval.e2.benchmark_audit import score_answer, typed_equal


def issue():
    return {
        "key": "KAFKA-TEST",
        "fields": {
            "created": "2020-01-01T00:00:00Z",
            "summary": "Future title",
            "description": "Future fix instructions",
        },
        "changelog": {
            "startAt": 0,
            "total": 3,
            "histories": [
                {
                    "id": "1",
                    "created": "2020-01-02T00:00:00Z",
                    "items": [{"field": "status", "fromString": "Open", "toString": "In Progress"}],
                },
                {
                    "id": "2",
                    "created": "2020-01-03T00:00:00Z",
                    "items": [
                        {"field": "status", "fromString": "In Progress", "toString": "Resolved"}
                    ],
                },
                {
                    "id": "3",
                    "created": "2020-01-04T00:00:00Z",
                    "items": [
                        {
                            "field": "description",
                            "fromString": "Original bug report",
                            "toString": "Future fix instructions",
                        },
                        {
                            "field": "summary",
                            "fromString": "Original title",
                            "toString": "Future title",
                        },
                    ],
                },
            ],
        },
    }


def test_issue_text_rewinds_future_solution_edits():
    assert body_at(issue(), timestamp("2020-01-02T12:00:00Z")) == {
        "summary": "Original title",
        "description": "Original bug report",
    }
    bad = issue()
    bad["fields"]["description"] = "Unrecorded later edit"
    with pytest.raises(ValueError, match="chain inconsistent"):
        body_at(bad, timestamp("2020-01-02T12:00:00Z"))


def test_incomplete_history_rejected():
    bad = issue()
    bad["changelog"]["total"] = 4
    with pytest.raises(ValueError, match="incomplete"):
        body_at(bad, timestamp("2020-01-02T12:00:00Z"))


def test_historical_prompt_does_not_include_future_text_or_gold():
    cfg = {"seed": "fixed", "historical_year_min": 2020, "historical_year_max": 2020}
    task = historical_task(issue(), "status_at_snapshot", cfg, "In Progress")
    assert task["expected_answer"]["value"] == "In Progress"
    assert "Future fix instructions" not in str(task)
    assert task["tool_profile"] == "historical_mcp_only"
    assert len(task["evidence"]) == 1
    assert all(timestamp(x["at"]) <= timestamp(task["cutoff"]) for x in task["context_events"])
    assert task == historical_task(deepcopy(issue()), "status_at_snapshot", cfg, "In Progress")


def test_typed_nested_history_scoring_preserves_order_and_types():
    expected = [{"at": "t1", "from": None, "to": "A"}, {"at": "t2", "from": "A", "to": "B"}]
    assert typed_equal(deepcopy(expected), expected)
    assert not typed_equal(list(reversed(expected)), expected)
    assert not typed_equal(True, 1)
    task = {"kind": "historical", "expected_answer": {"value": expected}}
    assert score_answer(task, {"value": expected})["value_correct"]
    assert not score_answer(task, {"value": str(expected)})["value_correct"]
    with pytest.raises(ValueError, match="test execution"):
        score_answer({"kind": "coding"}, {})
