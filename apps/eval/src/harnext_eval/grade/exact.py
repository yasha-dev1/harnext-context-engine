"""Normalised exact-match grading for docs/evaluation-spec.md §7 E2."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from harnext_eval.types import GradeResult

_TICKET_RE = re.compile(r"\b([a-z][a-z0-9]+)\s*[-_]\s*(\d+)\b", re.IGNORECASE)
_VERSION_RE = re.compile(
    r"^(?:version\s*|v\s*)?(\d+(?:\.\d+)+)([-+][0-9a-z.-]+)?$",
    re.IGNORECASE,
)
_UNKNOWN_RE = re.compile(r"^unknown[.!?]?$", re.IGNORECASE)


def normalize_exact(value: Any) -> str:
    """Return the canonical string used by deterministic exact matching.

    Normalisation is deliberately narrow: Unicode compatibility, case and
    whitespace are folded, ticket separators are canonicalised, and a leading
    ``v``/``version`` is removed from an otherwise complete version string.
    Empty strings and aliases such as ``N/A`` do not become ``UNKNOWN``.
    """

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"\s+", " ", text).casefold()
    if _UNKNOWN_RE.fullmatch(text):
        return "unknown"
    text = _TICKET_RE.sub(lambda match: f"{match.group(1).casefold()}-{match.group(2)}", text)
    version = _VERSION_RE.fullmatch(text)
    if version:
        numeric = ".".join(str(int(part)) for part in version.group(1).split("."))
        suffix = (version.group(2) or "").casefold()
        return numeric + suffix
    return text


def grade_exact(item_id: str, prediction: Any, gold: Any) -> GradeResult:
    """Grade one scalar prediction and retain both canonical strings."""

    normalised_prediction = normalize_exact(prediction)
    normalised_gold = normalize_exact(gold)
    return GradeResult(
        item_id=item_id,
        metric="exact",
        value=float(normalised_prediction == normalised_gold),
        details={
            "normalised_prediction": normalised_prediction,
            "normalised_gold": normalised_gold,
            "prediction_is_unknown": normalised_prediction == "unknown",
            "gold_is_unknown": normalised_gold == "unknown",
        },
    )
