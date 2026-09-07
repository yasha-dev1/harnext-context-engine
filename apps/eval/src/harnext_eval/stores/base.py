"""Git-backed snapshot store contract from docs/evaluation-spec.md §3.3 and §5."""

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from harnext_builder.agentfs.git_backend import GitBackend
from harnext_builder.agentfs.seed import SEED_FILES

from harnext_eval.types import EvalEvent, SnapshotRef

type LayoutCallable = Callable[["StoreHandle", list[EvalEvent], str], None]
_LAYOUTS: dict[str, LayoutCallable] = {}
_SNAPSHOT_STORES: dict[str, list[weakref.ReferenceType[StoreHandle]]] = {}
_SNAPSHOT_FIELDS = ("T_last_event", "sha", "last_event_id", "lane")
_DELIVERY_LEDGER_NAME = "delivered.jsonl"


class SnapshotLedgerError(ValueError):
    """Raised when snapshot provenance cannot be proved from the fold ledger."""


@dataclass(frozen=True, slots=True)
class DeliveryRecord:
    """One event delivered in one committed fold, in actual builder-delivery order."""

    sequence: int
    event_id: str
    event_time: datetime
    sha: str
    fold_index: int
    fold_event_index: int
    fold_max_event_time: datetime
    snapshot_T_last_event: datetime  # noqa: N815 - mirrors the public snapshot field
    lane: str

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> DeliveryRecord:
        try:
            return cls(
                sequence=int(row["sequence"]),
                event_id=str(row["event_id"]),
                event_time=_parse_aware_datetime(row["event_time"], field="event_time"),
                sha=str(row["sha"]),
                fold_index=int(row["fold_index"]),
                fold_event_index=int(row["fold_event_index"]),
                fold_max_event_time=_parse_aware_datetime(
                    row["fold_max_event_time"], field="fold_max_event_time"
                ),
                snapshot_T_last_event=_parse_aware_datetime(
                    row["snapshot_T_last_event"], field="snapshot_T_last_event"
                ),
                lane=str(row["lane"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SnapshotLedgerError(f"malformed delivery ledger row: {row!r}") from exc

    def as_dict(self) -> dict[str, str | int]:
        return {
            "sequence": self.sequence,
            "event_id": self.event_id,
            "event_time": self.event_time.isoformat(),
            "sha": self.sha,
            "fold_index": self.fold_index,
            "fold_event_index": self.fold_event_index,
            "fold_max_event_time": self.fold_max_event_time.isoformat(),
            "snapshot_T_last_event": self.snapshot_T_last_event.isoformat(),
            "lane": self.lane,
        }


def register_layout(name: str, layout_callable: LayoutCallable) -> None:
    """Register T5's fold implementation for one S0–S5 layout."""

    _LAYOUTS[name.upper()] = layout_callable


def store_for_snapshot(sha: str) -> StoreHandle | None:
    """Resolve a uniquely registered live store for a compatibility gate call."""

    live = {store for reference in _SNAPSHOT_STORES.get(sha, []) if (store := reference())}
    return next(iter(live)) if len(live) == 1 else None


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
        self.delivered_jsonl = self.snapshots_csv.with_name(_DELIVERY_LEDGER_NAME)
        self._provenance_signature: tuple[tuple[bool, int, int], ...] | None = None
        self._cached_snapshots: tuple[SnapshotRef, ...] = ()
        self._cached_deliveries: tuple[DeliveryRecord, ...] = ()
        self._cached_boundaries: dict[str, int] = {}
        self._cache_validated = False
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
        """Mutate, commit, and record exact cumulative snapshot provenance."""

        if not events:
            raise ValueError("cannot fold an empty event list")
        ordered_events = sorted(events, key=lambda event: (event.time, event.id))
        try:
            layout_callable = _LAYOUTS[self.layout]
        except KeyError as exc:
            raise RuntimeError(
                f"no fold callable registered for layout {self.layout}; T5 must register one"
            ) from exc
        snapshots = self._snapshots(validate_ledger=False)
        deliveries = self._delivery_records()
        if bool(snapshots) != bool(deliveries):
            raise SnapshotLedgerError(
                "snapshots.csv and delivered.jsonl must both describe every committed fold"
            )
        if snapshots:
            self._validate_ledger(snapshots, deliveries)
        incoming_ids = [event.id for event in ordered_events]
        if len(set(incoming_ids)) != len(incoming_ids):
            raise SnapshotLedgerError("one fold cannot deliver the same event ID twice")
        already_delivered = {row.event_id for row in deliveries}
        duplicate_ids = sorted(already_delivered.intersection(incoming_ids))
        if duplicate_ids:
            raise SnapshotLedgerError(
                "event IDs were already delivered: " + ",".join(duplicate_ids)
            )

        layout_callable(self, ordered_events, lane)
        fold_last_event = ordered_events[-1]
        previous_high_water = (
            (snapshots[-1].T_last_event, snapshots[-1].last_event_id) if snapshots else None
        )
        fold_high_water = (fold_last_event.time, fold_last_event.id)
        cumulative_high_water = max(
            value for value in (previous_high_water, fold_high_water) if value is not None
        )
        sha = self.backend.snapshot(self.org_id, f"fold:{lane}:{fold_last_event.id}")
        ref = SnapshotRef(
            sha=sha,
            T_last_event=cumulative_high_water[0],
            last_event_id=cumulative_high_water[1],
            lane=lane,
        )
        self._append_deliveries(
            ordered_events,
            ref,
            fold_index=len(snapshots),
            starting_sequence=len(deliveries),
            fold_max_event_time=fold_last_event.time,
        )
        self._append_snapshot(ref)
        self._register_snapshot(ref)
        return ref

    def snapshot(self, T: datetime) -> SnapshotRef:  # noqa: N803 - name fixed by shared contract
        """Return the last cumulative commit proved to contain no event after ``T``."""

        _require_aware(T, field="T")
        refs = self._snapshots()
        eligible = [ref for ref in refs if ref.T_last_event <= T]
        if not eligible:
            raise LookupError(f"no snapshot exists at or before {T.isoformat()}")
        ref = eligible[-1]
        self._register_snapshot(ref)
        return ref

    def delivery_records(self, ref: SnapshotRef | None = None) -> tuple[DeliveryRecord, ...]:
        """Return the validated cumulative delivery ledger, optionally through ``ref``."""

        self._snapshots()
        rows = self._delivery_records()
        if ref is None:
            return tuple(rows)
        try:
            boundary = self._cached_boundaries[ref.sha]
        except KeyError as exc:
            raise SnapshotLedgerError(
                f"snapshot SHA must resolve exactly once: {ref.sha!r}"
            ) from exc
        return tuple(rows[:boundary])

    def delivered_event_ids(self, ref: SnapshotRef | None = None) -> tuple[str, ...]:
        """Return complete ordered delivered IDs, optionally through one snapshot SHA."""

        return tuple(row.event_id for row in self.delivery_records(ref))

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
        self._invalidate_provenance_cache()

    def _snapshots(self, *, validate_ledger: bool = True) -> list[SnapshotRef]:
        self._refresh_provenance_cache(validate_ledger=validate_ledger)
        return list(self._cached_snapshots)

    def _read_snapshots(self) -> list[SnapshotRef]:
        if not self.snapshots_csv.exists():
            return []
        with self.snapshots_csv.open(newline="", encoding="utf-8") as source:
            refs = []
            for row in csv.DictReader(source):
                try:
                    refs.append(
                        SnapshotRef(
                            sha=row["sha"],
                            T_last_event=_parse_aware_datetime(
                                row["T_last_event"], field="T_last_event"
                            ),
                            last_event_id=row["last_event_id"],
                            lane=row["lane"],
                        )
                    )
                except (KeyError, ValueError) as exc:
                    raise SnapshotLedgerError(f"malformed snapshot row: {row!r}") from exc
        _validate_snapshot_rows(refs)
        return refs

    def _delivery_records(self) -> list[DeliveryRecord]:
        self._refresh_provenance_cache(validate_ledger=False)
        return list(self._cached_deliveries)

    def _read_delivery_records(self) -> list[DeliveryRecord]:
        if not self.delivered_jsonl.exists():
            return []
        records: list[DeliveryRecord] = []
        with self.delivered_jsonl.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SnapshotLedgerError(
                        f"invalid JSON in {self.delivered_jsonl}:{line_number}"
                    ) from exc
                if not isinstance(value, dict):
                    raise SnapshotLedgerError(
                        f"delivery row {line_number} must be a JSON object"
                    )
                records.append(DeliveryRecord.from_mapping(value))
        return records

    def _refresh_provenance_cache(self, *, validate_ledger: bool) -> None:
        signature = self._current_provenance_signature()
        if signature == self._provenance_signature and (
            self._cache_validated or not validate_ledger
        ):
            return
        refs = self._read_snapshots()
        records = self._read_delivery_records()
        if validate_ledger and refs:
            self._validate_ledger(refs, records)
        boundaries: dict[str, int] = {}
        for row in records:
            boundaries[row.sha] = row.sequence + 1
        self._provenance_signature = signature
        self._cached_snapshots = tuple(refs)
        self._cached_deliveries = tuple(records)
        self._cached_boundaries = boundaries
        self._cache_validated = validate_ledger or not refs

    def _current_provenance_signature(self) -> tuple[tuple[bool, int, int], ...]:
        values: list[tuple[bool, int, int]] = []
        for path in (self.snapshots_csv, self.delivered_jsonl):
            if path.exists():
                stat = path.stat()
                values.append((True, stat.st_size, stat.st_mtime_ns))
            else:
                values.append((False, 0, 0))
        return tuple(values)

    def _invalidate_provenance_cache(self) -> None:
        self._provenance_signature = None
        self._cache_validated = False

    def _register_snapshot(self, ref: SnapshotRef) -> None:
        live = [
            reference
            for reference in _SNAPSHOT_STORES.get(ref.sha, [])
            if reference() is not None
        ]
        if not any(reference() is self for reference in live):
            live.append(weakref.ref(self))
        _SNAPSHOT_STORES[ref.sha] = live

    def _append_deliveries(
        self,
        events: list[EvalEvent],
        ref: SnapshotRef,
        *,
        fold_index: int,
        starting_sequence: int,
        fold_max_event_time: datetime,
    ) -> None:
        with self.delivered_jsonl.open("a", encoding="utf-8") as destination:
            for event_index, event in enumerate(events):
                row = DeliveryRecord(
                    sequence=starting_sequence + event_index,
                    event_id=event.id,
                    event_time=event.time,
                    sha=ref.sha,
                    fold_index=fold_index,
                    fold_event_index=event_index,
                    fold_max_event_time=fold_max_event_time,
                    snapshot_T_last_event=ref.T_last_event,
                    lane=ref.lane,
                )
                destination.write(json.dumps(row.as_dict(), sort_keys=True) + "\n")
        self._invalidate_provenance_cache()

    def _validate_ledger(
        self, refs: list[SnapshotRef], records: list[DeliveryRecord]
    ) -> None:
        if not refs:
            if records:
                raise SnapshotLedgerError("delivery ledger exists without snapshot rows")
            return
        if not records:
            raise SnapshotLedgerError("snapshot index has no delivery ledger")
        if [row.sequence for row in records] != list(range(len(records))):
            raise SnapshotLedgerError("delivery sequence must be contiguous and start at zero")
        delivered_ids = [row.event_id for row in records]
        if len(set(delivered_ids)) != len(delivered_ids):
            raise SnapshotLedgerError("delivered event IDs must be unique")

        cumulative_max: tuple[datetime, str] | None = None
        record_offset = 0
        previous_sha: str | None = None
        rows_by_fold: dict[int, list[DeliveryRecord]] = {}
        for row in records:
            rows_by_fold.setdefault(row.fold_index, []).append(row)
        for fold_index, ref in enumerate(refs):
            fold_rows = rows_by_fold.get(fold_index, [])
            if not fold_rows:
                raise SnapshotLedgerError(f"snapshot {ref.sha} has no delivered events")
            if any(row.sha != ref.sha for row in fold_rows):
                raise SnapshotLedgerError(f"fold {fold_index} does not map exactly to {ref.sha}")
            if [row.sequence for row in fold_rows] != list(
                range(record_offset, record_offset + len(fold_rows))
            ):
                raise SnapshotLedgerError("fold rows must be contiguous in delivery order")
            if [row.fold_event_index for row in fold_rows] != list(range(len(fold_rows))):
                raise SnapshotLedgerError("fold event indexes must be contiguous")

            actual_fold_max = max(row.event_time for row in fold_rows)
            if any(row.fold_max_event_time != actual_fold_max for row in fold_rows):
                raise SnapshotLedgerError(f"fold {fold_index} has an invalid max event time")
            fold_max_key = max((row.event_time, row.event_id) for row in fold_rows)
            cumulative_max = max(
                value for value in (cumulative_max, fold_max_key) if value is not None
            )
            if (ref.T_last_event, ref.last_event_id) != cumulative_max:
                raise SnapshotLedgerError(
                    f"snapshot {ref.sha} watermark does not equal cumulative delivery high-water mark"
                )
            if any(row.snapshot_T_last_event != ref.T_last_event for row in fold_rows):
                raise SnapshotLedgerError(f"fold {fold_index} snapshot watermark disagrees with CSV")
            if any(row.lane != ref.lane for row in fold_rows):
                raise SnapshotLedgerError(f"fold {fold_index} lane disagrees with CSV")
            _verify_git_commit(self.worktree, ref.sha, previous_sha)
            previous_sha = ref.sha
            record_offset += len(fold_rows)

        if record_offset != len(records):
            raise SnapshotLedgerError("delivery ledger contains rows without a snapshot")
        head = subprocess.run(
            ["git", "-C", str(self.worktree), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        _verify_git_commit(self.worktree, head, refs[-1].sha)


def _parse_aware_datetime(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise SnapshotLedgerError(f"{field} must be a non-empty ISO datetime")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SnapshotLedgerError(f"{field} is not a valid ISO datetime: {value!r}") from exc
    _require_aware(parsed, field=field)
    return parsed


def _require_aware(value: datetime, *, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise SnapshotLedgerError(f"{field} must be timezone-aware")


def _validate_snapshot_rows(refs: list[SnapshotRef]) -> None:
    seen_shas: set[str] = set()
    previous: datetime | None = None
    for ref in refs:
        _require_aware(ref.T_last_event, field="T_last_event")
        if not ref.sha or ref.sha in seen_shas:
            raise SnapshotLedgerError("snapshot SHAs must be non-empty and unique")
        if previous is not None and ref.T_last_event < previous:
            raise SnapshotLedgerError("snapshot watermarks must be monotone")
        seen_shas.add(ref.sha)
        previous = ref.T_last_event


def _verify_git_commit(repository: Path, sha: str, parent: str | None) -> None:
    exists = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "-e", f"{sha}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    if exists.returncode != 0:
        raise SnapshotLedgerError(f"snapshot SHA does not exist in store repository: {sha}")
    if parent is None:
        return
    ancestor = subprocess.run(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", parent, sha],
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise SnapshotLedgerError(f"snapshot {sha} is not a descendant of {parent}")
