"""Offline Pony Mail fixture checks for docs/evaluation-spec.md §3.1/§4.1."""

import json
from pathlib import Path

from harnext_eval.corpus.pony_mail import parse_mbox, parse_stats_json, stats_message_count

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_monthly_mbox_threads_tags_and_baselines() -> None:
    events = parse_mbox(
        FIXTURES / "pony-dev-2026-01.mbox",
        list_name="dev",
        domain="kafka.apache.org",
        month="2026-01",
    )

    assert len(events) == 3
    assert all(event.type == "org.apache.mail.message" for event in events)
    assert events[0].subject == "thread:root.vote@kafka.apache.org"
    assert events[1].subject == events[0].subject
    assert events[1].data is not None
    assert events[1].data["in_reply_to"] == "root.vote@kafka.apache.org"
    assert events[0].data is not None
    assert events[0].data["subject_tags"] == ["[VOTE]", "KIP-1150", "KAFKA-19876"]
    assert events[2].data is not None
    assert events[2].data["subject_tags"] == ["[DISCUSS]", "KAFKA-19900"]
    assert all(any(key.startswith("contributor:") for key in event.baseline_keys) for event in events)
    assert all(any(key.startswith("thread:") for key in event.baseline_keys) for event in events)


def test_parse_stats_lua_json_count() -> None:
    payload = json.loads((FIXTURES / "pony-stats-2026-01.json").read_text())
    stats = parse_stats_json(payload)
    assert stats_message_count(stats, "2026-01") == 3



def test_mail_roster_matches_name_and_apache_local_part(tmp_path) -> None:
    from harnext_eval.corpus.committers import CommitterRoster

    roster = CommitterRoster([{"name": "Jun Rao", "apache_ids": ["junrao"], "github": []}])
    assert roster.matches("mail:dev@kafka.apache.org", {"author_email": "junrao@apache.org"})
    assert not roster.matches("mail:dev@kafka.apache.org", {"author_email": "junrao2@apache.org"})
    path = tmp_path / "dev-2026-01.mbox"
    path.write_text(
        'From junrao@example.org Thu Jan 01 10:00:00 2026\n'
        'From: Jun Rao <junrao@example.org>\n'
        'Date: Thu, 01 Jan 2026 10:00:00 +0000\n'
        'Message-ID: <test-roster@example.org>\nSubject: test\n\nbody\n'
    )
    events = parse_mbox(path, list_name="dev", domain="kafka.apache.org", month="2026-01")
    assert (events[0].data or {})["is_committer"] is True
