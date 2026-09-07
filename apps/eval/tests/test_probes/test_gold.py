"""Independent and catalogue-complete gold tests for evaluation-spec §4/§7 E2."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from harnext_eval.corpus.jira import parse_issue
from harnext_eval.probes.gold import (
    GoldAuditTrail,
    GoldRequest,
    PythonGold,
    cross_check_gold,
    field_value_python,
    field_value_sql,
)
from harnext_eval.types import EvalEvent


def _raw_issue(*, raw_final: str = "Closed") -> dict[str, object]:
    return {
        "id": "1",
        "key": "KAFKA-1",
        "fields": {
            "created": "2026-05-01T00:00:00Z",
            # Jira search returns export-time state, not issue-creation state.
            "status": {"name": raw_final},
            "assignee": {"displayName": "Alice", "accountId": "alice"},
            "priority": {"name": "Major"},
            "components": [{"name": "core"}],
            "fixVersions": [{"name": "4.0"}],
            "creator": {"displayName": "Alice", "accountId": "alice"},
            "reporter": {"displayName": "Alice", "accountId": "alice"},
            "comment": {"comments": []},
        },
        "changelog": {
            "histories": [
                {
                    "id": "h1",
                    "created": "2026-05-02T00:00:00Z",
                    "items": [
                        {"field": "status", "fromString": "Open", "toString": "In Progress"}
                    ],
                },
                {
                    "id": "h2",
                    "created": "2026-05-03T00:00:00Z",
                    "items": [
                        {
                            "field": "status",
                            "fromString": "In Progress",
                            "toString": raw_final,
                        }
                    ],
                },
            ]
        },
    }


def test_sql_reads_untouched_raw_jira_independently_from_python_replay() -> None:
    raw = _raw_issue()
    events = parse_issue(raw)
    at = datetime(2026, 5, 4, tzinfo=UTC)
    requests = [GoldRequest("KAFKA-1", "status", at)]

    assert field_value_python(events, "KAFKA-1", "status", at) == "Closed"
    assert field_value_sql(raw, "KAFKA-1", "status", at) == "Closed"
    assert cross_check_gold(events, requests, raw_jira=raw) == []
    assert field_value_sql(raw, "KAFKA-1", "components", at) == ["core"]
    assert field_value_sql(raw, "KAFKA-1", "fixVersion", at) == ["4.0"]
    assert field_value_sql(raw, "KAFKA-1", "assignee", at) == field_value_python(
        events, "KAFKA-1", "assignee", at
    )


def test_both_gold_derivations_ignore_post_window_search_snapshot() -> None:
    raw = _raw_issue()
    events = parse_issue(raw)
    before_first_change = datetime(2026, 5, 1, 12, tzinfo=UTC)

    created = next(event for event in events if event.type.endswith(".created"))
    fields = raw["fields"]
    assert isinstance(fields, dict)
    assert fields["status"] == {"name": "Closed"}
    assert created.data is not None
    assert created.data["status"] == "Open"
    assert field_value_python(events, "KAFKA-1", "status", before_first_change) == "Open"
    assert field_value_sql(raw, "KAFKA-1", "status", before_first_change) == "Open"


def test_correlated_normalisation_error_is_exposed_and_fails_98_percent_gate() -> None:
    raw = _raw_issue(raw_final="Closed")
    events = parse_issue(_raw_issue(raw_final="Done"))
    at = datetime(2026, 5, 4, tzinfo=UTC)
    disagreement = cross_check_gold(
        events,
        [GoldRequest("KAFKA-1", "status", at)],
        raw_jira=raw,
    )
    assert len(disagreement) == 1
    assert disagreement[0].python_value == "Done"
    assert disagreement[0].sql_value == "Closed"

    audit = GoldAuditTrail(source="raw-jira-export")
    audit.compare(disagreement[0].request, "Done", "Closed")
    assert audit.report()["disagreements"]
    with pytest.raises(ValueError, match="below 98%"):
        audit.require_valid(evidentiary=True)


def _event(
    event_id: str,
    minute: int,
    *,
    source: str,
    event_type: str,
    subject: str,
    data: dict[str, object],
) -> EvalEvent:
    return EvalEvent(
        id=event_id,
        source=source,
        type=event_type,
        subject=subject,
        time=datetime(2026, 5, 1, tzinfo=UTC) + timedelta(minutes=minute),
        mgtenant="test",
        data=data,
    )


def test_python_gold_covers_initial_pr_kip_thread_and_world_state_fields() -> None:
    raw = _raw_issue()
    events = parse_issue(raw)
    events.extend(
        [
            _event(
                "pr",
                1,
                source="github:apache/kafka",
                event_type="com.github.pull_request.merged",
                subject="pr:apache/kafka#1",
                data={"title": "KAFKA-1 fix", "merged_at": "2026-05-01T00:01:00Z"},
            ),
            _event(
                "vote",
                2,
                source="mail:dev@kafka.apache.org",
                event_type="org.apache.mail.message",
                subject="thread:vote",
                data={
                    "subject": "[RESULT] KIP-1 vote accepted",
                    "body": "accepted",
                    "author": "committer",
                    "in_reply_to": "root",
                },
            ),
            _event(
                "world",
                3,
                source="orgforge:test",
                event_type="orgforge.world_state.dump",
                subject="world:1",
                data={
                    "world_state": {
                        "entities": {
                            "account:7": {
                                "plan": "enterprise",
                                "status": "active",
                            }
                        }
                    }
                },
            ),
        ]
    )
    gold = PythonGold(sorted(events, key=lambda event: event.time))
    at = datetime(2026, 5, 5, tzinfo=UTC)

    assert str(gold.field_value("KAFKA-1", "assignee", at)).startswith("jira-user:")
    assert gold.field_value("KAFKA-1", "priority", at) == "Major"
    assert gold.field_value("KAFKA-1", "components", at) == ["core"]
    assert gold.field_value("KAFKA-1", "fixVersion", at) == ["4.0"]
    assert gold.field_value("pr:apache/kafka#1", "state", at) == "merged"
    assert gold.field_value("KIP-1", "vote_outcome", at) == "accepted"
    assert gold.field_value("thread:vote", "answered_by", at) == "committer"
    assert gold.field_value("account:7", "plan", at) == "enterprise"
