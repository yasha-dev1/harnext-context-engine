"""Offline Jira fixture checks for docs/evaluation-spec.md §3.1/§4.1."""

import json
from pathlib import Path
from typing import Any

from harnext_eval.corpus.jira import iter_search_pages, parse_search_pages

FIXTURES = Path(__file__).parent / "fixtures"


def _page(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text())


def test_search_pagination_advances_by_actual_page_length() -> None:
    pages = [_page("jira-page-1.json"), _page("jira-page-2.json")]
    starts: list[tuple[int, int]] = []

    def fetch_page(start_at: int, max_results: int) -> dict[str, Any]:
        starts.append((start_at, max_results))
        return pages[start_at]

    assert list(iter_search_pages(fetch_page, max_results=50)) == pages
    assert starts == [(0, 50), (1, 50)]


def test_parse_created_transition_items_and_comments() -> None:
    events = parse_search_pages(iter([_page("jira-page-1.json"), _page("jira-page-2.json")]))

    assert len(events) == 6
    assert {event.subject for event in events} == {
        "issue:KAFKA-19876",
        "issue:KAFKA-19900",
    }
    transitions = [event for event in events if event.type.endswith(".transition")]
    assert len(transitions) == 3
    assert {(event.data or {})["field"] for event in transitions} == {
        "status",
        "priority",
        "fixVersion",
    }
    priority = next(event for event in transitions if (event.data or {})["field"] == "priority")
    assert priority.data is not None
    assert priority.data["from"] == "Major"
    assert priority.data["to"] == "Critical"
    assert str(priority.data["actor"]).startswith("contributor:")
    assert "component:streams" in priority.baseline_keys
    created = next(event for event in events if event.id == "jira:19876:created")
    assert created.data is not None
    assert created.data["status"] == "Open"
    assert created.data["priority"] == "Major"
    comments = [event for event in events if event.type.endswith(".comment")]
    assert len(comments) == 1
    assert "fix is ready" in str((comments[0].data or {})["body"])
