"""Deterministic replay builder for docs/evaluation-spec.md §3.3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

from harnext_eval.corpus.gharchive import iter_archive
from harnext_eval.corpus.jira import parse_search_page
from harnext_eval.corpus.keys import assign_event_keys
from harnext_eval.corpus.pony_mail import parse_mbox
from harnext_eval.types import EvalEvent


@dataclass(frozen=True)
class ReplayArtifact:
    path: Path
    sha256_path: Path
    sha256: str
    event_count: int


def merge_events(
    sources: Iterable[Iterable[EvalEvent]], *, mgtenant: str = "kafka"
) -> list[EvalEvent]:
    """Normalize, de-duplicate, and globally event-time sort source streams."""

    by_id: dict[str, EvalEvent] = {}
    canonical_json: dict[str, str] = {}
    for source in sources:
        for original in source:
            event = assign_event_keys(_aware_event(original), mgtenant=mgtenant)
            encoded = _canonical_line(event)
            previous = canonical_json.get(event.id)
            if previous is not None and previous != encoded:
                raise ValueError(f"conflicting events share id {event.id!r}")
            by_id[event.id] = event
            canonical_json[event.id] = encoded
    return sorted(by_id.values(), key=lambda event: (event.time, event.id, event.source, event.type))


def build_replay(
    sources: Iterable[Iterable[EvalEvent]],
    output: str | Path,
    *,
    mgtenant: str = "kafka",
) -> ReplayArtifact:
    """Merge source event iterables and write canonical JSONL plus SHA-256 sidecar."""

    events = merge_events(sources, mgtenant=mgtenant)
    return write_replay(events, output)


def write_replay(events: Iterable[EvalEvent], output: str | Path) -> ReplayArtifact:
    """Write an already-normalized replay atomically and hash its exact bytes."""

    path = Path(output)
    if path.suffix != ".jsonl":
        raise ValueError("replay output must have a .jsonl suffix")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
            for event in events:
                line = f"{_canonical_line(event)}\n".encode()
                temporary.write(line)
                digest.update(line)
                count += 1
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)

    checksum = digest.hexdigest()
    sidecar = path.with_suffix(f"{path.suffix}.sha256")
    sidecar.write_text(f"{checksum}  {path.name}\n", encoding="ascii")
    return ReplayArtifact(path, sidecar, checksum, count)


def read_replay(path: str | Path) -> Iterator[EvalEvent]:
    """Stream normalized events from an existing replay input."""

    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield EvalEvent.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"invalid EvalEvent on {path}:{line_number}") from exc


def _canonical_line(event: EvalEvent) -> str:
    payload = event.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _aware_event(event: EvalEvent) -> EvalEvent:
    if event.time.tzinfo is not None:
        return event.model_copy(update={"time": event.time.astimezone(UTC)})
    return event.model_copy(update={"time": event.time.replace(tzinfo=UTC)})


def _jira_fixture(path: str | Path, *, mgtenant: str) -> list[EvalEvent]:
    payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    pages = payload if isinstance(payload, list) else [payload]
    events: list[EvalEvent] = []
    for page in pages:
        if not isinstance(page, dict):
            raise ValueError(f"Jira fixture {path} must contain an object or list of objects")
        events.extend(parse_search_page(page, mgtenant=mgtenant))
    return events


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a sorted, hashed harnext evaluation replay from source extracts."
    )
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--tenant", default="kafka")
    parser.add_argument(
        "--input", action="append", default=[], type=Path, help="existing EvalEvent JSONL"
    )
    parser.add_argument(
        "--jira-json", action="append", default=[], type=Path, help="Jira search page JSON"
    )
    parser.add_argument(
        "--gharchive", action="append", default=[], type=Path, help="hourly .json.gz archive"
    )
    parser.add_argument("--repo", default="apache/kafka", help="GH Archive repository filter")
    parser.add_argument("--pony-mbox", action="append", default=[], type=Path)
    parser.add_argument("--pony-list", default="dev")
    parser.add_argument("--pony-domain", default="kafka.apache.org")
    parser.add_argument("--pony-month", help="YYYY-MM; required with --pony-mbox")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``python -m harnext_eval.corpus.build_replay``."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.pony_mbox and not args.pony_month:
        parser.error("--pony-month is required with --pony-mbox")
    if not (args.input or args.jira_json or args.gharchive or args.pony_mbox):
        parser.error("at least one source input is required")

    sources: list[Iterable[EvalEvent]] = []
    sources.extend(read_replay(path) for path in args.input)
    sources.extend(_jira_fixture(path, mgtenant=args.tenant) for path in args.jira_json)
    sources.extend(
        iter_archive(path, repo=args.repo, mgtenant=args.tenant) for path in args.gharchive
    )
    sources.extend(
        parse_mbox(
            path,
            list_name=args.pony_list,
            domain=args.pony_domain,
            month=args.pony_month,
            mgtenant=args.tenant,
        )
        for path in args.pony_mbox
    )
    artifact = build_replay(sources, args.output, mgtenant=args.tenant)
    print(f"{artifact.event_count} events -> {artifact.path} ({artifact.sha256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ReplayArtifact",
    "build_replay",
    "main",
    "merge_events",
    "read_replay",
    "write_replay",
]
