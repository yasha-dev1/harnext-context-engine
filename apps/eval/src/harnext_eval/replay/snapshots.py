"""Snapshot lookup and materialisation for docs/evaluation-spec.md §3.3 and §5."""

from __future__ import annotations

import csv
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from harnext_builder.agentfs.git_backend import GitBackend

from harnext_eval.stores.base import SnapshotLedgerError
from harnext_eval.types import SnapshotRef


def snapshot(
    T: datetime,  # noqa: N803 - notation fixed by spec
    snapshots_csv: str | Path,
    *,
    repository: str | Path | None = None,
) -> SnapshotRef:
    """Return the last monotone cumulative watermark at or before ``T``."""

    _require_aware(T, "T")
    path = Path(snapshots_csv)
    if not path.exists():
        raise LookupError(f"snapshot index does not exist: {path}")
    refs: list[SnapshotRef] = []
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            try:
                ref = SnapshotRef(
                    sha=row["sha"],
                    T_last_event=datetime.fromisoformat(
                        row["T_last_event"].replace("Z", "+00:00")
                    ),
                    last_event_id=row["last_event_id"],
                    lane=row["lane"],
                )
            except (KeyError, ValueError) as exc:
                raise SnapshotLedgerError(f"malformed snapshot row: {row!r}") from exc
            refs.append(ref)
    _validate_refs(refs, Path(repository) if repository is not None else None)
    eligible = [ref for ref in refs if ref.T_last_event <= T]
    if not eligible:
        raise LookupError(f"no snapshot exists at or before {T.isoformat()}")
    return eligible[-1]


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
        return snapshot(
            T,
            self.snapshots_csv,
            repository=self.backend.root / "git" / self.org_id,
        )

    def materialise(self, ref: SnapshotRef) -> Path:
        return materialise(ref, self.backend, self.org_id)


def _require_aware(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise SnapshotLedgerError(f"{field} must be timezone-aware")


def _validate_refs(refs: list[SnapshotRef], repository: Path | None) -> None:
    previous_time: datetime | None = None
    previous_sha: str | None = None
    seen: set[str] = set()
    for ref in refs:
        _require_aware(ref.T_last_event, "T_last_event")
        if not ref.sha or ref.sha in seen:
            raise SnapshotLedgerError("snapshot SHAs must be non-empty and unique")
        if previous_time is not None and ref.T_last_event < previous_time:
            raise SnapshotLedgerError("snapshot watermarks must be monotone")
        if repository is not None:
            exists = subprocess.run(
                ["git", "-C", str(repository), "cat-file", "-e", f"{ref.sha}^{{commit}}"],
                capture_output=True,
                text=True,
            )
            if exists.returncode != 0:
                raise SnapshotLedgerError(f"snapshot SHA does not exist: {ref.sha}")
            if previous_sha is not None:
                ancestor = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(repository),
                        "merge-base",
                        "--is-ancestor",
                        previous_sha,
                        ref.sha,
                    ],
                    capture_output=True,
                    text=True,
                )
                if ancestor.returncode != 0:
                    raise SnapshotLedgerError(
                        f"snapshot {ref.sha} is not a descendant of {previous_sha}"
                    )
        seen.add(ref.sha)
        previous_time = ref.T_last_event
        previous_sha = ref.sha
    if repository is not None and previous_sha is not None:
        head = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        ancestor = subprocess.run(
            ["git", "-C", str(repository), "merge-base", "--is-ancestor", previous_sha, head],
            capture_output=True,
            text=True,
        )
        if ancestor.returncode != 0:
            raise SnapshotLedgerError(f"latest indexed snapshot {previous_sha} is not in HEAD")
