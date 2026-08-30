"""Strict YAML configuration for docs/evaluation-spec.md §5 and §12."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model that rejects misspelled or unsupported evaluation knobs."""

    model_config = ConfigDict(extra="forbid")


class RulesConfig(StrictModel):
    enabled: bool


class DeviationConfig(StrictModel):
    enabled: bool


class GuardsConfig(StrictModel):
    absolute_floor: float = Field(ge=0)
    multi_window: bool
    situation_dedup: bool


class RouterConfig(StrictModel):
    rules: RulesConfig
    deviation: DeviationConfig
    budget_pct: float = Field(gt=0, le=100)
    guards: GuardsConfig


class WindowConfig(StrictModel):
    gap_s: float = Field(gt=0)
    max_events: int = Field(gt=0)
    max_age_s: float = Field(gt=0)


class StoreConfig(StrictModel):
    layout: Literal["S0", "S1", "S2", "S3", "S4", "S5"]
    backend: Literal["git"]


class BuilderConfig(StrictModel):
    harness: Literal["fake", "claude_code"]
    model: str | None
    prompt_version: str

    @model_validator(mode="after")
    def real_harness_requires_model(self) -> BuilderConfig:
        if self.harness != "fake" and not self.model:
            raise ValueError("a non-fake builder harness requires a model")
        return self


class ReaderConfig(StrictModel):
    provider: Literal["fake", "anthropic"]
    budget_tokens: int = Field(gt=0)


class EmbeddingsConfig(StrictModel):
    provider: Literal["fake"]
    dim: int = Field(gt=0)


class EngineConfig(StrictModel):
    router: RouterConfig
    window: WindowConfig
    store: StoreConfig
    builder: BuilderConfig
    reader: ReaderConfig
    envelope: Literal["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]
    embeddings: EmbeddingsConfig
    lane_design: Literal["two-lane", "single"] = "two-lane"


class BudgetsConfig(StrictModel):
    read_tokens: list[int]

    @model_validator(mode="after")
    def positive_unique_budgets(self) -> BudgetsConfig:
        if not self.read_tokens or any(value <= 0 for value in self.read_tokens):
            raise ValueError("read_tokens must contain positive values")
        if len(set(self.read_tokens)) != len(self.read_tokens):
            raise ValueError("read_tokens must not contain duplicates")
        return self


class ExperimentConfig(StrictModel):
    offline: bool = True
    engine: EngineConfig
    budgets: BudgetsConfig
    seeds: list[int]

    @model_validator(mode="after")
    def at_least_one_seed(self) -> ExperimentConfig:
        if not self.seeds:
            raise ValueError("at least one seed is required")
        return self


def load_config(path: str | Path) -> ExperimentConfig:
    """Load one YAML profile and reject unknown or missing configuration keys."""

    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"configuration {config_path} must contain a YAML mapping")
    return ExperimentConfig.model_validate(raw)
