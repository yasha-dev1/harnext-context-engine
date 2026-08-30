"""Link-set grading for docs/evaluation-spec.md §7 E2."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from harnext_eval.grade.exact import normalize_exact
from harnext_eval.types import GradeResult

_KEY_RE = re.compile(r"(?<![\w-])(?:[A-Za-z][A-Za-z0-9]+\s*[-_]\s*\d+|#\d+)(?![\w-])")


def normalise_keys(values: str | Iterable[Any] | None) -> set[str]:
    """Normalise a key collection, extracting ticket keys from free text."""

    if values is None:
        return set()
    if isinstance(values, str):
        matches = _KEY_RE.findall(values)
        raw_values: Iterable[Any] = matches if matches else (values,)
    else:
        raw_values = values
    return {key for value in raw_values if (key := normalize_exact(value))}


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
