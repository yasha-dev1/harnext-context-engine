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


def test_lossy_snapshot_is_diagnostic_and_changelog_remains_truth() -> None:
    # Minimal reproduction of the KAFKA-294 deleted/renamed-version export shape.
    page = _page("jira-page-2.json")
    issue = page["issues"][0]
    issue["fields"]["fixVersions"] = [{"name": "renamed-after-history"}]
    events = parse_search_pages([page])
    created = next(event for event in events if event.type.endswith(".created"))
    assert "fixVersion" in (created.data or {})["state_inconsistencies"]
    assert (created.data or {})["fix_versions"] != ["renamed-after-history"]
    transitions = [event for event in events if (event.data or {}).get("field") == "fixVersion"]
    assert transitions
    assert (transitions[-1].data or {})["to"] != ["renamed-after-history"]


def test_roster_stamps_jira_and_github_without_substring_matches() -> None:
    from harnext_eval.corpus.committers import CommitterRoster, stamp_events

    roster = CommitterRoster([{"name": "Jun Rao", "apache_ids": ["junrao"], "github": ["junrao"]}])
    assert roster.matches("jira:KAFKA", {"actor_name": " JUN  RAO "})
    assert roster.matches("jira:KAFKA", {"author_account_id": "junrao"})
    assert not roster.matches("jira:KAFKA", {"actor_name": "Jun Rao Jr"})
    assert roster.matches("github:apache/kafka", {"actor_login": "junrao"})
    assert roster.matches("github:apache/kafka", {"author_association": "MEMBER"})
    assert not roster.matches("github:apache/kafka", {"author_association": "CONTRIBUTOR"})
    page = _page("jira-page-1.json")
    page["issues"][0]["fields"]["creator"] = {"displayName": "Jun Rao"}
    events = parse_search_pages([page])
    created = next(e for e in events if e.type.endswith(".created"))
    assert (created.data or {})["is_committer"] is True
    stamped, counts = stamp_events([created], roster)
    assert (stamped[0].data or {})["is_committer"] is True
    assert counts == {"jira.total": 1, "jira.matched": 1}
