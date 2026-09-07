"""Corpus handle contract from docs/evaluation-spec.md §3 and §5."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from harnext_eval.types import EvalEvent


@dataclass(frozen=True)
class CorpusHandle:
    name: str
    replay_path: Path
    probes_path: Path | None
    tasks_path: Path | None
    window: str | tuple[datetime, datetime]
    meta: dict[str, Any] = field(default_factory=dict)

    def events(self) -> Iterator[EvalEvent]:
        with self.replay_path.open(encoding="utf-8") as replay:
            for line in replay:
                if line.strip():
                    yield EvalEvent.model_validate_json(line)


from harnext_eval.corpus.synthetic import (  # noqa: E402
    generate_synthetic_corpus,
    generate_synthetic_events,
)

__all__ = ["CorpusHandle", "generate_synthetic_corpus", "generate_synthetic_events"]
