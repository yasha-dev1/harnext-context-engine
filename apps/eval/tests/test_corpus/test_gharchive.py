"""Offline GH Archive fixture checks for docs/evaluation-spec.md §3.1/§4.1."""

import gzip
import shutil
from datetime import UTC, datetime
from pathlib import Path

from harnext_eval.corpus.gharchive import iter_archive
from harnext_eval.probes.gen_code_location import generate_code_location_probes
from harnext_eval.probes.gen_multisource import JoinAuditTrail
from harnext_eval.types import EvalEvent

FIXTURES = Path(__file__).parent / "fixtures"


def test_streams_supported_repo_events_without_assuming_pr_payload_files(tmp_path: Path) -> None:
    archive = tmp_path / "2026-01-02-10.json.gz"
    with (FIXTURES / "gharchive-hour.jsonl").open("rb") as source, gzip.open(
        archive, "wb"
    ) as target:
        shutil.copyfileobj(source, target)

    events = list(iter_archive(archive, repo="apache/kafka"))

    assert len(events) == 7
    assert {event.type for event in events} == {
        "com.github.pull_request.opened",
        "com.github.pull_request.merged",
        "com.github.pull_request.closed",
        "com.github.review",
        "com.github.review_comment",
        "com.github.issue_comment",
        "com.github.push",
    }
    opened = next(event for event in events if event.type.endswith(".opened"))
    assert opened.subject == "pr:20412"
    assert opened.data is not None
    assert opened.data["changed_files"] == []
    pushed = next(event for event in events if event.type.endswith(".push"))
    assert pushed.subject == "issue:KAFKA-19876"
    assert pushed.data is not None
    assert pushed.data["changed_files"] == ["streams/src/main/java/Snapshot.java"]
    assert any(key.startswith("contributor:") for key in pushed.baseline_keys)


def test_real_extractor_path_joins_merged_pr_to_push_files_and_audits_keys(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "2026-01-02-10.json.gz"
    with (FIXTURES / "gharchive-hour.jsonl").open("rb") as source, gzip.open(
        archive, "wb"
    ) as target:
        shutil.copyfileobj(source, target)
    events = [
        EvalEvent(
            id="jira:19876:created",
            source="jira:KAFKA",
            type="org.apache.jira.issue.created",
            subject="issue:KAFKA-19876",
            time=datetime(2026, 1, 1, tzinfo=UTC),
            mgtenant="kafka",
            data={"issue_key": "KAFKA-19876", "status": "Open"},
        ),
        *iter_archive(archive, repo="apache/kafka"),
    ]
    audit = JoinAuditTrail(expected={"github:gh-5": ["KAFKA-19876"]})

    probes = generate_code_location_probes(
        events,
        count=1,
        seed=1,
        probe_start=datetime(2026, 1, 2, 10, 6, 1, tzinfo=UTC),
        probe_end=datetime(2026, 1, 3, tzinfo=UTC),
        join_audit=audit,
    )

    assert len(probes) == 1
    assert probes[0].family == "multisource"
    assert probes[0].gold_type == "files"
    assert probes[0].gold == {
        "files": ["streams/src/main/java/Snapshot.java"],
        "modules": ["streams/src"],
    }
    assert probes[0].source_event_ids == ["github:gh-5", "github:gh-6"]
    report = audit.report()
    assert report["join_precision"] == 1.0
    assert report["join_recall"] == 1.0
