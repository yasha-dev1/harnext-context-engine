"""Offline GH Archive fixture checks for docs/evaluation-spec.md §3.1/§4.1."""

import gzip
import shutil
from pathlib import Path

from harnext_eval.corpus.gharchive import iter_archive

FIXTURES = Path(__file__).parent / "fixtures"


def test_streams_supported_repo_events_and_changed_files(tmp_path: Path) -> None:
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
    assert opened.data["changed_files"] == [
        "streams/src/main/java/Snapshot.java",
        "streams/src/test/java/SnapshotTest.java",
    ]
    pushed = next(event for event in events if event.type.endswith(".push"))
    assert pushed.subject == "issue:KAFKA-19876"
    assert pushed.data is not None
    assert pushed.data["changed_files"] == ["streams/src/main/java/Snapshot.java"]
    assert any(key.startswith("contributor:") for key in pushed.baseline_keys)

