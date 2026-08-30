"""Git-backed snapshot store contract from docs/evaluation-spec.md §3.3 and §5."""

from __future__ import annotations

import csv
import subprocess
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from harnext_builder.agentfs.git_backend import GitBackend
from harnext_builder.agentfs.seed import SEED_FILES

from harnext_eval.types import EvalEvent, SnapshotRef

type LayoutCallable = Callable[["StoreHandle", list[EvalEvent], str], None]
_LAYOUTS: dict[str, LayoutCallable] = {}
_SNAPSHOT_FIELDS = ("T_last_event", "sha", "last_event_id", "lane")


def register_layout(name: str, layout_callable: LayoutCallable) -> None:
    """Register T5's fold implementation for one S0–S5 layout."""

    _LAYOUTS[name.upper()] = layout_callable


class StoreHandle:
    """A live git worktree plus an immutable, event-time-indexed snapshot log."""

    def __init__(
        self,
        layout: str,
        org_id: str,
        root: str | Path,
        backend: GitBackend | str | None = None,
        snapshots_csv: str | Path | None = None,
    ) -> None:
        self.layout = layout.upper()
        self.org_id = org_id
        self.root = Path(root)
        if backend is None or backend == "git":
            self.backend = GitBackend(self.root)
        elif isinstance(backend, GitBackend):
            self.backend = backend
        else:
            raise ValueError("evaluation stores currently support only the git backend")
        self.snapshots_csv = (
            Path(snapshots_csv) if snapshots_csv is not None else self.root / "snapshots.csv"
        )
        self.snapshots_csv.parent.mkdir(parents=True, exist_ok=True)
        self.backend.ensure_seeded(self.org_id, SEED_FILES)

    @property
    def worktree(self) -> Path:
        return self.backend.root / "git" / self.org_id

    @classmethod
    def register_layout(cls, name: str, layout_callable: LayoutCallable) -> None:
        del cls
        register_layout(name, layout_callable)

    def write(self, relpath: str, content: str) -> None:
        self.backend.write_file(self.org_id, relpath, content)

    def fold(self, events: list[EvalEvent], lane: str) -> SnapshotRef:
        """Delegate layout mutation, commit it, and index the resulting snapshot."""

        if not events:
            raise ValueError("cannot fold an empty event list")
        try:
            layout_callable = _LAYOUTS[self.layout]
        except KeyError as exc:
            raise RuntimeError(
                f"no fold callable registered for layout {self.layout}; T5 must register one"
            ) from exc
        layout_callable(self, events, lane)
        last_event = max(events, key=lambda event: (event.time, event.id))
        sha = self.backend.snapshot(self.org_id, f"fold:{lane}:{last_event.id}")
        ref = SnapshotRef(
            sha=sha,
            T_last_event=last_event.time,
            last_event_id=last_event.id,
            lane=lane,
        )
        self._append_snapshot(ref)
        return ref

    def snapshot(self, T: datetime) -> SnapshotRef:  # noqa: N803 - name fixed by shared contract
        """Return the last recorded commit whose event time is at or before ``T``."""

        eligible = [ref for ref in self._snapshots() if ref.T_last_event <= T]
        if not eligible:
            raise LookupError(f"no snapshot exists at or before {T.isoformat()}")
        return max(enumerate(eligible), key=lambda pair: (pair[1].T_last_event, pair[0]))[1]

    def materialise(self, ref: SnapshotRef) -> Path:
        """Clone and detach a temporary checkout at ``ref``; the caller removes it."""

        checkout = Path(tempfile.mkdtemp(prefix=f"harnext-eval-{self.org_id}-"))
        repository = self.worktree
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
        return checkout

    def read(self, ref: SnapshotRef, relpath: str) -> str | None:
        return self.backend.read_file(self.org_id, relpath, ref.sha)

    def list_files(self, ref: SnapshotRef) -> list[str]:
        return sorted(self.backend.list_files(self.org_id, ref.sha))

    def _append_snapshot(self, ref: SnapshotRef) -> None:
        write_header = not self.snapshots_csv.exists() or self.snapshots_csv.stat().st_size == 0
        with self.snapshots_csv.open("a", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=_SNAPSHOT_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "T_last_event": ref.T_last_event.isoformat(),
                    "sha": ref.sha,
                    "last_event_id": ref.last_event_id,
                    "lane": ref.lane,
                }
            )

    def _snapshots(self) -> list[SnapshotRef]:
        if not self.snapshots_csv.exists():
            return []
        with self.snapshots_csv.open(newline="", encoding="utf-8") as source:
            return [
                SnapshotRef(
                    sha=row["sha"],
                    T_last_event=datetime.fromisoformat(row["T_last_event"]),
                    last_event_id=row["last_event_id"],
                    lane=row["lane"],
                )
                for row in csv.DictReader(source)
            ]
