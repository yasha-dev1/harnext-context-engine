"""Deterministic snapshot-index tests for docs/evaluation-spec.md §3.3."""

from __future__ import annotations

import csv
import shutil
from datetime import UTC, datetime
from pathlib import Path

from harnext_builder.agentfs.git_backend import GitBackend
from harnext_eval.replay.snapshots import SnapshotIndex, materialise, snapshot


def test_snapshot_tie_uses_last_commit_and_materialises_via_git(tmp_path: Path) -> None:
    backend = GitBackend(tmp_path / "backend")
    backend.ensure_seeded("acme", {"state.txt": "zero\n"})
    backend.write_file("acme", "state.txt", "one\n")
    first_sha = backend.snapshot("acme", "first")
    backend.write_file("acme", "state.txt", "two\n")
    second_sha = backend.snapshot("acme", "second")
    at = datetime(2026, 1, 2, tzinfo=UTC)
    index_path = tmp_path / "snapshots.csv"
    with index_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=("T_last_event", "sha", "last_event_id", "lane")
        )
        writer.writeheader()
        writer.writerow(
            {
                "T_last_event": at.isoformat(),
                "sha": first_sha,
                "last_event_id": "first",
                "lane": "batch",
            }
        )
        writer.writerow(
            {
                "T_last_event": at.isoformat(),
                "sha": second_sha,
                "last_event_id": "second",
                "lane": "fast",
            }
        )

    ref = snapshot(at, index_path)
    assert ref.sha == second_sha
    assert SnapshotIndex(index_path, backend, "acme").snapshot(at) == ref

    checkout = materialise(ref, backend, "acme")
    try:
        assert (checkout / "state.txt").read_text() == "two\n"
    finally:
        shutil.rmtree(checkout)
