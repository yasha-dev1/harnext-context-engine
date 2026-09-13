"""Deterministic span exposure metrics, separate from citations and code success.

Annotations are evaluator-owned byte spans in an exact snapshot export. They
must be re-annotated for a different representation; event-ID presence is never
treated as proof that the supporting content was returned.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from harnext_eval.e2.context import Action, SnapshotTools


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit_id: str
    path: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str


def merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if start >= end:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def overlap(interval: tuple[int, int], intervals: list[tuple[int, int]]) -> int:
    start, end = interval
    return sum(max(0, min(end, right) - max(start, left)) for left, right in merge(intervals))


def returned_spans(files: dict[str, str], action: Action, result: str):
    """Map actual returned bodies to snapshot byte offsets, excluding wrappers."""
    if action.action == "read" and action.path in files:
        data = files[action.path].encode()
        fragment = data[action.offset : action.offset + len(result.encode())]
        # A mid-codepoint offset can generate replacement characters in the
        # legacy tool. Reject ambiguous alignment rather than award exposure.
        if fragment == result.encode():
            yield action.path, action.offset, action.offset + len(fragment)
    elif action.action == "search" and action.query:
        remaining = result.encode()
        first = True
        for path, text in sorted(files.items()):
            offset = 0
            for number, line in enumerate(text.splitlines(keepends=True), 1):
                body = line.rstrip("\r\n")
                if action.query.casefold() in body.casefold():
                    prefix = (("" if first else "\n") + f"{path}:{number}: ").encode()
                    first = False
                    if len(remaining) < len(prefix) or not remaining.startswith(prefix):
                        return
                    remaining = remaining[len(prefix) :]
                    size = min(len(body.encode()), len(remaining))
                    if remaining[:size] != body.encode()[:size]:
                        raise ValueError("search body cannot be aligned to snapshot")
                    yield path, offset, offset + size
                    remaining = remaining[size:]
                    if size < len(body.encode()):
                        return
                offset += len(line.encode())


def score_retrieval(
    files: dict[str, str],
    annotations: list[EvidenceSpan],
    required_groups: list[set[str]],
    trace: list[dict[str, Any]],
    *,
    budget: int,
    response_cap: int,
    annotations_exhaustive: bool,
) -> dict[str, Any]:
    """A group is satisfied by any one of its alternative supporting units.

    Recall concerns required groups; precision concerns annotated units returned.
    Precision is undefined if annotations are incomplete or no unit was exposed.
    Byte precision includes duplicate reads and protocol overhead in its denominator.
    """
    ids = {span.unit_id for span in annotations}
    for span in annotations:
        data = files.get(span.path, "").encode()
        if span.start >= span.end or data[span.start : span.end] != span.quote.encode():
            raise ValueError(f"stale or invalid evidence span: {span.unit_id}")
    if any(not group or not group <= ids for group in required_groups):
        raise ValueError("required evidence is not annotated")
    replay = SnapshotTools(files, budget=budget, response_cap=response_cap)
    coverage: dict[str, list[tuple[int, int]]] = {}
    exposed: set[str] = set()
    relevant = set().union(*required_groups) if required_groups else set()
    relevant_by_path: dict[str, list[tuple[int, int]]] = {}
    for span in annotations:
        if span.unit_id in relevant:
            relevant_by_path.setdefault(span.path, []).append((span.start, span.end))
    relevant_bytes = 0
    first_relevant = None
    steps = []
    for index, row in enumerate(trace, 1):
        action = Action.model_validate(row["action"])
        actual = replay.invoke(action)
        if actual != row["result"] or replay.log[-1] != row:
            raise ValueError("retrieval trace differs from immutable snapshot replay")
        for path, start, end in returned_spans(files, action, actual):
            coverage.setdefault(path, []).append((start, end))
            relevant_bytes += overlap((start, end), relevant_by_path.get(path, []))
        for span in annotations:
            if (
                overlap((span.start, span.end), coverage.get(span.path, []))
                == span.end - span.start
            ):
                exposed.add(span.unit_id)
        if first_relevant is None and exposed & relevant:
            first_relevant = index
        recall = (
            sum(bool(group & exposed) for group in required_groups) / len(required_groups)
            if required_groups
            else None
        )
        steps.append({"call": index, "bytes": replay.used, "required_group_recall": recall})
    relevant_exposed = exposed & relevant
    return {
        "context_calls": len(trace),
        "tool_mix": dict(Counter(row["action"]["action"] for row in trace)),
        "returned_bytes": replay.used,
        "exposed_units": sorted(exposed),
        "relevant_exposed_units": sorted(relevant_exposed),
        "required_group_recall": (
            sum(bool(group & exposed) for group in required_groups) / len(required_groups)
            if required_groups
            else None
        ),
        "unit_precision": (
            len(relevant_exposed) / len(exposed) if exposed and annotations_exhaustive else None
        ),
        "relevant_byte_precision": (
            relevant_bytes / replay.used if replay.used and annotations_exhaustive else None
        ),
        "first_relevant_context_call": first_relevant,
        "recall_curve": steps,
        "annotations_exhaustive": annotations_exhaustive,
        "tool_replay_passed": True,
    }
