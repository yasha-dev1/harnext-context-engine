"""Deterministic snapshot-index tests for docs/evaluation-spec.md §3.3."""

from __future__ import annotations

import csv
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from harnext_builder.agentfs.git_backend import GitBackend
from harnext_eval.replay.snapshots import SnapshotIndex, materialise, snapshot
from harnext_eval.stores.base import SnapshotLedgerError


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


def test_snapshot_rejects_non_monotone_or_naive_watermarks(tmp_path: Path) -> None:
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
                "sha": "later",
                "last_event_id": "later",
                "lane": "fast",
            }
        )
        writer.writerow(
            {
                "T_last_event": (at - timedelta(seconds=1)).isoformat(),
                "sha": "older",
                "last_event_id": "older",
                "lane": "batch",
            }
        )
    with pytest.raises(SnapshotLedgerError, match="monotone"):
        snapshot(at, index_path)

    lines = index_path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace("+00:00", "")
    index_path.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    with pytest.raises(SnapshotLedgerError, match="timezone-aware"):
        snapshot(at, index_path)
