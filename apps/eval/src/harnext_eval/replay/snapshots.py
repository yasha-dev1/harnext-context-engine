"""Snapshot lookup and materialisation for docs/evaluation-spec.md §3.3 and §5."""

from __future__ import annotations

import csv
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from harnext_builder.agentfs.git_backend import GitBackend

from harnext_eval.types import SnapshotRef


def snapshot(
    T: datetime,  # noqa: N803 - notation fixed by spec
    snapshots_csv: str | Path,
) -> SnapshotRef:
    """Return the last CSV row with an event-time watermark at or before ``T``."""

    path = Path(snapshots_csv)
    if not path.exists():
        raise LookupError(f"snapshot index does not exist: {path}")
    eligible: list[tuple[int, SnapshotRef]] = []
    with path.open(newline="", encoding="utf-8") as source:
        for index, row in enumerate(csv.DictReader(source)):
            ref = SnapshotRef(
                sha=row["sha"],
                T_last_event=datetime.fromisoformat(row["T_last_event"]),
                last_event_id=row["last_event_id"],
                lane=row["lane"],
            )
            if ref.T_last_event <= T:
                eligible.append((index, ref))
    if not eligible:
        raise LookupError(f"no snapshot exists at or before {T.isoformat()}")
    return max(eligible, key=lambda pair: (pair[1].T_last_event, pair[0]))[1]


def materialise(ref: SnapshotRef, backend: GitBackend, org_id: str) -> Path:
    """Clone the backend's org repository into a detached temporary checkout."""

    repository = backend.root / "git" / org_id
    if not repository.exists():
        raise LookupError(f"git-backed store does not exist for org {org_id!r}")
    checkout = Path(tempfile.mkdtemp(prefix=f"harnext-eval-{org_id}-"))
    try:
        subprocess.run(
            ["git", "clone", "--quiet", "--no-checkout", str(repository), str(checkout)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "--quiet", "--detach", ref.sha],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        import shutil

        shutil.rmtree(checkout, ignore_errors=True)
        raise
    return checkout


@dataclass(frozen=True)
class SnapshotIndex:
    """Bound snapshot index for callers that perform repeated lookups."""

    snapshots_csv: Path
    backend: GitBackend
    org_id: str

    def snapshot(self, T: datetime) -> SnapshotRef:  # noqa: N803
        return snapshot(T, self.snapshots_csv)

    def materialise(self, ref: SnapshotRef) -> Path:
        return materialise(ref, self.backend, self.org_id)
