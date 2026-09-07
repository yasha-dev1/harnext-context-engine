"""Code-localisation grading for docs/evaluation-spec.md §7 E2/E4."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import PurePosixPath

from harnext_eval.types import GradeResult


def normalize_path(path: str) -> str:
    """Canonicalise a repository-relative path without touching the filesystem."""

    cleaned = path.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return PurePosixPath(cleaned).as_posix().casefold().strip("/")


def module_for(path: str) -> str:
    """Return the first two path segments, or all segments when fewer exist."""

    parts = PurePosixPath(normalize_path(path)).parts
    return "/".join(parts[:2])


def _ordered_unique(paths: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(path for raw in paths if (path := normalize_path(raw))))


def localisation_scores(
    predicted_files: Iterable[str],
    gold_files: Iterable[str],
    *,
    k: int = 5,
    agentless_superset: bool = True,
) -> dict[str, float | list[str]]:
    """Compute localisation scores.

    Under the Agentless convention, hit@k is one only when the top-k predicted
    set is a superset of the complete gold file set.  Disabling the convention
    changes hit@k to the looser "any gold file" definition. Precision/recall
    remain ordinary set scores in either mode.
    """

    if k < 1:
        raise ValueError("k must be at least 1")
    predicted_ordered = _ordered_unique(predicted_files)
    predicted_top_k = set(predicted_ordered[:k])
    predicted = set(predicted_ordered)
    gold = set(_ordered_unique(gold_files))
    overlap = predicted & gold
    if gold:
        file_hit = float(gold <= predicted_top_k) if agentless_superset else float(bool(gold & predicted_top_k))
        recall = len(overlap) / len(gold)
    else:
        file_hit = float(not predicted_top_k)
        recall = float(not predicted)
    precision = len(overlap) / len(predicted) if predicted else float(not gold)
    predicted_modules = {module_for(path) for path in predicted_top_k}
    gold_modules = {module_for(path) for path in gold}
    if gold_modules:
        module_hit = (
            float(gold_modules <= predicted_modules)
            if agentless_superset
            else float(bool(gold_modules & predicted_modules))
        )
    else:
        module_hit = float(not predicted_modules)
    return {
        f"file_hit@{k}": file_hit,
        "file_recall": recall,
        "file_precision": precision,
        "module_hit": module_hit,
        "predicted_files": predicted_ordered,
        "gold_files": sorted(gold),
        "predicted_modules": sorted(predicted_modules),
        "gold_modules": sorted(gold_modules),
    }


def grade_localisation(
    item_id: str,
    predicted_files: Iterable[str],
    gold_files: Iterable[str],
    *,
    k: int = 5,
    agentless_superset: bool = True,
) -> GradeResult:
    """Return file hit@k as the value and every localisation score as details."""

    details = localisation_scores(
        predicted_files,
        gold_files,
        k=k,
        agentless_superset=agentless_superset,
    )
    metric = f"file_hit@{k}"
    value = details[metric]
    assert isinstance(value, float)
    return GradeResult(item_id=item_id, metric=metric, value=value, details=details)
