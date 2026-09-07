"""Materialised-store health checks for docs/evaluation-spec.md §5 and §7 E3."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from datasketch import MinHash

_MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
_WIKI_REF_RE = re.compile(r"\[\[([A-Za-z][A-Za-z0-9]+-\d+)\]\]")
_NAMED_ENTITY_REF_RE = re.compile(
    r"\b(?:entity|ticket|issue)(?:_ref)?\s*:\s*([A-Za-z][A-Za-z0-9]+-\d+)\b",
    re.IGNORECASE,
)
_ENTITY_KEY_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]+-\d+\b")
_GLOBAL_SUPERSEDED_RE = re.compile(
    r"^-\s*(?P<subject>.+):\s+[^=\s]+=(?P<value>.*?)\s+\[[^]]+\]\s+was superseded by\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[\w.-]+", re.UNICODE)
_IGNORED_TOP_LEVEL = {"_meta", "topics", "events", ".git"}


def _store_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    )


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def _entity_directories(root: Path) -> list[Path]:
    entities: dict[Path, None] = {}
    entities_root = root / "entities"
    if entities_root.is_dir():
        for path in entities_root.rglob("*"):
            if not path.is_dir() or path.name.startswith("."):
                continue
            if (path / "OVERVIEW.md").exists() or (path / "facts.md").exists():
                entities[path] = None
    for path in root.iterdir():
        if not path.is_dir() or path.name.startswith("."):
            continue
        if path.name.casefold() in _IGNORED_TOP_LEVEL or path.name == "entities":
            continue
        if (path / "OVERVIEW.md").exists() or (path / "facts.md").exists():
            entities[path] = None
    return sorted(entities)


def _path_within(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _link_target(source: Path, raw_target: str, root: Path) -> Path | None:
    target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or target.startswith(("#", "mailto:")):
        return None
    path_part = unquote(parsed.path)
    if not path_part:
        return None
    candidate = root / path_part.lstrip("/") if path_part.startswith("/") else source.parent / path_part
    return candidate if _path_within(root, candidate) else Path("/__outside_store__")


def _known_entity_keys(entity_dirs: Iterable[Path]) -> dict[str, Path]:
    return {path.name.casefold(): path for path in entity_dirs}


def _markdown_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.count("|") < 2:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append(cells)
    if rows:
        header = " ".join(rows[0]).casefold()
        if any(word in header for word in ("entity", "path", "file", "overview", "ticket")):
            rows = rows[1:]
    return rows


def _index_resolution(
    root: Path, files: Iterable[Path], entity_dirs: Iterable[Path]
) -> tuple[int, int, list[str]]:
    entries = 0
    resolved = 0
    unresolved_rows: list[str] = []
    known_entities = _known_entity_keys(entity_dirs)
    for index_path in (path for path in files if path.name.casefold() == "index.md"):
        text = _read_text(index_path)
        if text is None:
            continue
        for cells in _markdown_rows(text):
            entries += 1
            candidates = [
                target
                for raw_target in _MARKDOWN_LINK_RE.findall(" | ".join(cells))
                if (target := _link_target(index_path, raw_target, root)) is not None
            ]
            row_keys = {
                match.group(0).casefold()
                for match in _ENTITY_KEY_RE.finditer(" | ".join(cells))
            }
            row_resolves = any(candidate.exists() for candidate in candidates)
            row_resolves = row_resolves or any(key in known_entities for key in row_keys)
            if row_resolves:
                resolved += 1
            else:
                unresolved_rows.append(" | ".join(cells))
    return entries, resolved, unresolved_rows


def _cross_references(
    root: Path, files: Iterable[Path], entity_dirs: Iterable[Path]
) -> tuple[int, list[str]]:
    total = 0
    dangling: list[str] = []
    known_entities = _known_entity_keys(entity_dirs)
    for source in files:
        if source.suffix.casefold() != ".md":
            continue
        text = _read_text(source)
        if text is None:
            continue
        for raw_target in _MARKDOWN_LINK_RE.findall(text):
            target = _link_target(source, raw_target, root)
            if target is None:
                continue
            total += 1
            if not target.exists():
                dangling.append(f"{source.relative_to(root)} -> {raw_target}")
        entity_refs = set(_WIKI_REF_RE.findall(text)) | set(_NAMED_ENTITY_REF_RE.findall(text))
        for entity_ref in entity_refs:
            total += 1
            if entity_ref.casefold() not in known_entities:
                dangling.append(f"{source.relative_to(root)} -> entity:{entity_ref}")
    return total, dangling


def _fact_lines(files: Iterable[Path], root: Path) -> list[tuple[str, str]]:
    facts: list[tuple[str, str]] = []
    for path in files:
        if path.name.casefold() != "facts.md":
            continue
        text = _read_text(path)
        if text is None:
            continue
        for line in text.splitlines():
            cleaned = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", line).strip()
            if not cleaned or cleaned.startswith("#") or re.fullmatch(r"[-|: ]+", cleaned):
                continue
            facts.append((path.relative_to(root).as_posix(), cleaned))
    return facts


def _minhash(text: str, num_perm: int) -> MinHash:
    signature = MinHash(num_perm=num_perm, seed=1)
    tokens = {token.casefold() for token in _WORD_RE.findall(text)}
    for token in sorted(tokens):
        signature.update(token.encode("utf-8"))
    return signature


def _near_duplicates(
    facts: list[tuple[str, str]], num_perm: int, threshold: float
) -> tuple[set[int], list[dict[str, Any]]]:
    signatures = [_minhash(text, num_perm) for _, text in facts]
    duplicate_indexes: set[int] = set()
    pairs: list[dict[str, Any]] = []
    for left in range(len(facts)):
        for right in range(left + 1, len(facts)):
            similarity = signatures[left].jaccard(signatures[right])
            if similarity < threshold:
                continue
            duplicate_indexes.update((left, right))
            pairs.append(
                {
                    "left": {"file": facts[left][0], "fact": facts[left][1]},
                    "right": {"file": facts[right][0], "fact": facts[right][1]},
                    "jaccard": similarity,
                }
            )
    return duplicate_indexes, pairs


def _table_superseded_values(text: str) -> list[str]:
    rows = _markdown_rows(text)
    if not rows:
        return []
    raw_lines = [line for line in text.splitlines() if line.strip().startswith("|")]
    if not raw_lines:
        return []
    headers = [cell.strip().casefold() for cell in raw_lines[0].strip().strip("|").split("|")]
    value_index = next(
        (
            index
            for index, header in enumerate(headers)
            if header in {"value", "old", "previous", "superseded", "superseded value"}
        ),
        None,
    )
    if value_index is None:
        return []
    return [cells[value_index].strip(" `") for cells in rows if value_index < len(cells)]


def _superseded_values(text: str) -> list[str]:
    values = _table_superseded_values(text)
    table_lines = {line for line in text.splitlines() if line.strip().startswith("|")}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or line in table_lines:
            continue
        match = re.match(r"(?:[-*+]\s+)?(?:[^:]{1,40}:\s*)?(.+)$", stripped)
        if match:
            value = match.group(1).strip().strip("`").strip()
            if value and not re.fullmatch(r"[-|: ]+", value):
                values.append(value)
    return list(dict.fromkeys(values))


def _contains_value(text: str, value: str) -> bool:
    folded_text = re.sub(r"\s+", " ", text).casefold()
    folded_value = re.sub(r"\s+", " ", value).casefold().strip()
    if not folded_value:
        return False
    return re.search(rf"(?<!\w){re.escape(folded_value)}(?!\w)", folded_text) is not None


def _supersession_leaks(
    root: Path, entity_dirs: Iterable[Path]
) -> tuple[int, dict[str, list[str]]]:
    entity_list = list(entity_dirs)
    values_by_entity: dict[Path, list[str]] = defaultdict(list)
    subject_to_entity: dict[str, Path] = {}
    for entity_dir in entity_list:
        overview = _read_text(entity_dir / "OVERVIEW.md") or ""
        heading = next(
            (line.removeprefix("# ").strip() for line in overview.splitlines() if line.startswith("# ")),
            "",
        )
        if heading:
            subject_to_entity[heading.casefold()] = entity_dir
        subject_to_entity[entity_dir.name.casefold()] = entity_dir
        relative = entity_dir.relative_to(root).parts
        if len(relative) >= 3 and relative[0] == "entities":
            subject_to_entity[f"{relative[-2]}:{relative[-1]}".casefold()] = entity_dir

    global_superseded = root / "_meta" / "superseded.md"
    global_text = _read_text(global_superseded) if global_superseded.is_file() else None
    if global_text:
        for line in global_text.splitlines():
            match = _GLOBAL_SUPERSEDED_RE.match(line.strip())
            if match is None:
                continue
            entity_dir = subject_to_entity.get(match.group("subject").casefold())
            value = match.group("value").strip().strip("`")
            if entity_dir is not None and value:
                values_by_entity[entity_dir].append(value)

    for entity_dir in entity_list:
        superseded_path = entity_dir / "_meta" / "superseded.md"
        if superseded_path.is_file():
            values_by_entity[entity_dir].extend(
                _superseded_values(_read_text(superseded_path) or "")
            )

    leaks: dict[str, list[str]] = {}
    for entity_dir, raw_values in values_by_entity.items():
        values = list(dict.fromkeys(raw_values))
        overview_path = entity_dir / "OVERVIEW.md"
        if not values:
            continue
        overview = _read_text(overview_path) or ""
        found = [value for value in values if _contains_value(overview, value)]
        if found:
            leaks[entity_dir.relative_to(root).as_posix()] = found
    return len(values_by_entity), leaks


def compute_store_health(
    store_dir: str | Path,
    *,
    duplicate_threshold: float = 0.8,
    num_perm: int = 128,
) -> dict[str, Any]:
    """Compute all E3 health metrics for a materialised store directory."""

    root = Path(store_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if not 0.0 <= duplicate_threshold <= 1.0:
        raise ValueError("duplicate_threshold must be between 0 and 1")
    if num_perm < 1:
        raise ValueError("num_perm must be positive")

    files = _store_files(root)
    entity_dirs = _entity_directories(root)
    sizes = {path: path.stat().st_size for path in files}
    line_counts = {
        path: len(text.splitlines())
        for path in files
        if (text := _read_text(path)) is not None
    }
    over_cap = [path for path, count in line_counts.items() if count > 200]
    entity_file_counts = {
        entity.relative_to(root).as_posix(): sum(1 for path in files if entity in path.parents)
        for entity in entity_dirs
    }

    index_entries, index_resolved, unresolved_index = _index_resolution(
        root, files, entity_dirs
    )
    cross_ref_count, dangling = _cross_references(root, files, entity_dirs)
    facts = _fact_lines(files, root)
    duplicate_indexes, duplicate_pairs = _near_duplicates(
        facts, num_perm, duplicate_threshold
    )
    superseded_entities, supersession_leaks = _supersession_leaks(root, entity_dirs)
    file_count = len(files)
    duplicate_rate = len(duplicate_indexes) / len(facts) if facts else 0.0
    return {
        "store_dir": str(root),
        "files": file_count,
        "bytes": sum(sizes.values()),
        "entities": len(entity_dirs),
        "files_per_entity": (
            sum(entity_file_counts.values()) / len(entity_file_counts)
            if entity_file_counts
            else 0.0
        ),
        "entity_file_counts": entity_file_counts,
        "over_cap_files": len(over_cap),
        "over_cap_share": len(over_cap) / file_count if file_count else 0.0,
        "over_cap_paths": [path.relative_to(root).as_posix() for path in over_cap],
        "index_entries": index_entries,
        "index_resolved": index_resolved,
        "index_resolution_rate": index_resolved / index_entries if index_entries else 1.0,
        "unresolved_index_rows": unresolved_index,
        "cross_references": cross_ref_count,
        "dangling_cross_references": len(dangling),
        "dangling_cross_reference_rate": (
            len(dangling) / cross_ref_count if cross_ref_count else 0.0
        ),
        "dangling_references": dangling,
        "fact_lines": len(facts),
        "near_duplicate_fact_lines": len(duplicate_indexes),
        "near_duplicate_fact_rate": duplicate_rate,
        "near_duplicate_pairs": duplicate_pairs,
        "entities_with_supersession": superseded_entities,
        "entities_with_supersession_leakage": len(supersession_leaks),
        "supersession_leakage_rate": (
            len(supersession_leaks) / superseded_entities if superseded_entities else 0.0
        ),
        "supersession_leaks": supersession_leaks,
    }


def store_health_csv_row(metrics: Mapping[str, Any]) -> dict[str, str | int | float]:
    """Flatten the scalar health metrics into a CSV-writer-compatible row."""

    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, str | int | float) and not isinstance(value, bool)
    }


def analyse_store_health(
    store_dir: str | Path,
    *,
    duplicate_threshold: float = 0.8,
    num_perm: int = 128,
) -> tuple[dict[str, Any], dict[str, str | int | float]]:
    """Return the reviewable metric dictionary and its scalar CSV row."""

    metrics = compute_store_health(
        store_dir,
        duplicate_threshold=duplicate_threshold,
        num_perm=num_perm,
    )
    return metrics, store_health_csv_row(metrics)
