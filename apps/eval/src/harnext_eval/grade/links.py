"""Link-set grading for docs/evaluation-spec.md §7 E2."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from harnext_eval.grade.exact import normalize_exact
from harnext_eval.types import GradeResult

_LINK_RE = re.compile(
    r"(?ix)(?<![\w:])(?:"
    r"pr:[a-z0-9_.-]+(?:/[a-z0-9_.-]+)*\#\d+|"
    r"pr:\d+|"
    r"thread:[a-z0-9_.@-]+/[a-z0-9_.@<>-]+|"
    r"thread:[a-z0-9_.@<>-]+|"
    r"commit:[a-z0-9_.-]+(?:/[a-z0-9_.-]+)*@[a-z0-9_.:-]+|"
    r"ticket:[a-z][a-z0-9]+\s*[-_]\s*\d+|"
    r"[a-z][a-z0-9]+\s*[-_]\s*\d+|"
    r"\#\d+"
    r")(?![\w])"
)


def _canonical_link(value: Any) -> str:
    text = str(value).strip().casefold().rstrip(".,;:!?")
    if text.startswith("ticket:"):
        return f"ticket:{normalize_exact(text.removeprefix('ticket:'))}"
    if re.fullmatch(r"[a-z][a-z0-9]+\s*[-_]\s*\d+", text):
        return normalize_exact(text)
    return text


def normalise_keys(values: str | Iterable[Any] | None) -> set[str]:
    """Extract every canonical PR/thread/commit/ticket identifier from text."""

    if values is None:
        return set()
    if isinstance(values, str):
        matches = _LINK_RE.findall(values)
        raw_values: Iterable[Any] = matches if matches else (values,)
    else:
        raw_values = values
    canonical: set[str] = set()
    for value in raw_values:
        if isinstance(value, str):
            matches = _LINK_RE.findall(value)
            pieces: Iterable[Any] = matches if matches else (value,)
        else:
            pieces = (value,)
        canonical.update(key for piece in pieces if (key := _canonical_link(piece)))
    return canonical


def grade_links(
    item_id: str,
    cited_keys: str | Iterable[Any] | None,
    gold_keys: str | Iterable[Any] | None,
) -> GradeResult:
    """Return link F1 with set precision and recall in the result details."""

    predicted = normalise_keys(cited_keys)
    gold = normalise_keys(gold_keys)
    overlap = predicted & gold
    precision = len(overlap) / len(predicted) if predicted else float(not gold)
    recall = len(overlap) / len(gold) if gold else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return GradeResult(
        item_id=item_id,
        metric="link_f1",
        value=f1,
        details={
            "precision": precision,
            "recall": recall,
            "predicted_keys": sorted(predicted),
            "gold_keys": sorted(gold),
            "matched_keys": sorted(overlap),
        },
    )
