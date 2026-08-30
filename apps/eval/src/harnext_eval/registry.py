"""Experiment registry contract from docs/evaluation-spec.md §5."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from harnext_eval.config import EngineConfig
from harnext_eval.corpus import CorpusHandle


@dataclass
class ExperimentResult:
    name: str
    metrics: dict[str, float] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    artifacts: list[Path] = field(default_factory=list)
    primary: dict[str, Any] = field(default_factory=dict)
    check_details: dict[str, dict[str, Any]] = field(default_factory=dict)


@runtime_checkable
class Experiment(Protocol):
    name: str

    def run(
        self, cfg: EngineConfig, corpus: CorpusHandle, out_dir: Path, seed: int
    ) -> ExperimentResult: ...

    def chart(self, result: ExperimentResult, out_dir: Path) -> list[Path]: ...


_EXPERIMENTS: dict[str, Experiment] = {}


def register_experiment(experiment: Experiment) -> Experiment:
    name = experiment.name.casefold()
    if name in _EXPERIMENTS:
        raise ValueError(f"experiment {name!r} is already registered")
    _EXPERIMENTS[name] = experiment
    return experiment


register = register_experiment


def get_experiment(name: str) -> Experiment:
    try:
        return _EXPERIMENTS[name.casefold()]
    except KeyError as exc:
        choices = ", ".join(list_experiments())
        raise KeyError(f"unknown experiment {name!r}; choose one of: {choices}") from exc


def list_experiments() -> list[str]:
    return sorted(_EXPERIMENTS)
