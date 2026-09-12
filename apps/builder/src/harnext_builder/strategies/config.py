"""One strict experiment contract, shared by builder, MCP and external trials."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileLayout(StrEnum):
    VERBATIM = "verbatim_dump"
    FOLDERED = "foldered_sessions"
    FLAT = "flat_facts"
    INDEXED = "indexed_entities"
    LEDGER = "indexed_temporal"
    TOPICS = "topic_hierarchy"
    CURATED = "agent_curated"


class WritingPrompt(StrEnum):
    PRESERVE = "preserve_supported"
    CONCISE = "concise_supported"
    DOMAIN = "domain_focused"


class Extraction(StrEnum):
    STRUCTURED = "structured"
    ATOMIC = "atomic_llm"


class Retention(StrEnum):
    ALL = "all"
    CURRENT = "current"
    WINDOW = "window"


class Summary(StrEnum):
    NONE = "none"
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"


class Update(StrEnum):
    APPEND = "append"
    REPLACE = "replace"
    SUPERSEDE = "supersede"


class RetrievalMethod(StrEnum):
    LITERAL = "literal"
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


class GraphSchema(StrEnum):
    GENERIC = "generic_attributed"
    TYPED = "typed_assertions"


class GraphExtraction(StrEnum):
    BROAD = "broad"
    DOMAIN = "domain_predicates"


class Resolution(StrEnum):
    CANONICAL = "canonical_only"
    ALIASES = "evidence_aliases"


class Traversal(StrEnum):
    NEIGHBORS = "neighbors"
    PATHS = "paths"
    PAGERANK = "personalized_pagerank"
    DUAL = "dual_level"


class AgentConfig(Strict):
    harness: Literal["none", "codex", "claude_code", "harnext"] = "none"
    model: str | None = None
    effort: Literal["low", "medium", "high", "xhigh"] | None = "medium"
    provider: str | None = None
    api_key_env: str = "HARNEXT_PROVIDER_API_KEY"
    timeout_s: int = Field(default=180, ge=1)
    max_calls: int = Field(default=40, ge=1)
    context_bytes: int = Field(default=131072, ge=1024)

    @model_validator(mode="after")
    def configured(self):
        if self.harness != "none" and not self.model:
            raise ValueError("learned builder requires an explicit model")
        if self.harness == "codex" and self.effort is None:
            raise ValueError("Codex requires explicit effort")
        if self.harness == "harnext" and (self.provider is None or self.effort is not None):
            raise ValueError(
                "Harnext adapter requires provider and effort=null; this SDK does not expose verified effort parity"
            )
        if self.harness == "claude_code" and self.effort == "xhigh":
            raise ValueError("Claude adapter supports low/medium/high or null, not xhigh")
        return self


class FileConfig(Strict):
    layout: FileLayout = FileLayout.INDEXED
    prompt: WritingPrompt = WritingPrompt.PRESERVE
    extraction: Extraction = Extraction.STRUCTURED
    retention: Retention = Retention.ALL
    retention_days: int = Field(default=90, ge=1)
    summarization: Summary = Summary.NONE
    update: Update = Update.SUPERSEDE
    summary_sentences: int = Field(default=3, ge=1)
    page_bytes: int = Field(default=8192, ge=256)
    functional_predicates: list[str] = Field(
        default_factory=lambda: ["assigned_to", "has_status", "targets_version"]
    )
    predicates: list[str] = Field(
        default_factory=lambda: [
            "assigned_to",
            "has_status",
            "targets_version",
            "references_issue",
            "changes_file",
            "discussed_in",
        ]
    )


class EncoderConfig(Strict):
    provider: Literal["fastembed", "openai_compatible"] = "fastembed"
    model: str = "BAAI/bge-small-en-v1.5"
    revision: str = "content-hash-at-build"
    cache_dir: str = ".harnext/models"
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_s: int = Field(default=60, ge=1)
    threads: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def remote(self):
        if self.provider == "openai_compatible" and (not self.base_url or not self.api_key_env):
            raise ValueError("remote encoder requires base_url and api_key_env")
        return self


class RetrievalConfig(Strict):
    method: RetrievalMethod = RetrievalMethod.BM25
    chunking: Literal["document", "fixed", "entity_bundle"] = "fixed"
    chunk_bytes: int = Field(default=2048, ge=64)
    overlap_bytes: int = Field(default=256, ge=0)
    candidates: int = Field(default=40, ge=1, le=1000)
    top_k: int = Field(default=10, ge=1, le=1000)
    rrf_k: int = Field(default=60, ge=1)
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    embeddings: EncoderConfig = Field(default_factory=EncoderConfig)
    reranker: Literal["none", "cross_encoder"] = "none"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    @model_validator(mode="after")
    def limits(self):
        if self.overlap_bytes >= self.chunk_bytes or self.top_k > self.candidates:
            raise ValueError("overlap must be smaller than chunk; top_k must not exceed candidates")
        return self


class GraphConfig(Strict):
    schema_name: GraphSchema = GraphSchema.TYPED
    extraction: GraphExtraction = GraphExtraction.DOMAIN
    resolution: Resolution = Resolution.CANONICAL
    traversal: Traversal = Traversal.NEIGHBORS
    hops: int = Field(default=1, ge=1, le=4)
    max_edges: int = Field(default=20, ge=1, le=500)
    max_expansions: int = Field(default=10000, ge=1, le=1000000)
    damping: float = Field(default=0.85, gt=0, lt=1)
    iterations: int = Field(default=30, ge=1, le=1000)
    functional_predicates: list[str] = Field(
        default_factory=lambda: ["assigned_to", "has_status", "targets_version"]
    )


class TrialConfig(Strict):
    run_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    task_id: str = Field(min_length=1)
    harness: str = "external"
    model: str = "externally-recorded"
    effort: str = "externally-recorded"
    max_calls: int = Field(default=100, ge=1)
    budget_bytes: int = Field(default=65536, ge=256)
    response_bytes: int = Field(default=12000, ge=128)
    timeout_s: int = Field(default=3600, ge=1)
    tools: list[
        Literal[
            "context_list",
            "context_read",
            "context_search",
            "find_entities",
            "get_assertions",
            "neighbors",
            "fetch_source",
        ]
    ] = Field(default_factory=lambda: ["context_list", "context_read", "context_search"])
    transport: Literal["stdio", "http"] = "stdio"
    host: str = "127.0.0.1"
    port: int = Field(default=8876, ge=1024, le=65535)
    token_env: str = "HARNEXT_EVAL_TOKEN"

    @model_validator(mode="after")
    def unique_tools(self):
        if not self.tools or len(self.tools) != len(set(self.tools)):
            raise ValueError("tools must be nonempty and unique")
        return self


class EvaluationConfig(Strict):
    grading: Literal["retrieval_only", "qa_exact", "coding_tests"] = "retrieval_only"
    task_file: Path | None = None
    gold_file: Path | None = None
    answer_file: Path | None = None
    candidate_files: Path | None = None
    workspace_dir: Path | None = None
    agent_trace: Path | None = None
    test_image: str = (
        "python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea"
    )

    @model_validator(mode="after")
    def complete(self):
        if self.grading != "retrieval_only" and (self.task_file is None or self.gold_file is None):
            raise ValueError("QA/coding grading requires public task_file and private gold_file")
        if self.grading == "qa_exact" and self.answer_file is None:
            raise ValueError("qa_exact requires answer_file")
        if self.grading == "coding_tests" and self.candidate_files is None:
            raise ValueError("coding_tests requires candidate_files")
        return self


class StrategyConfig(Strict):
    protocol: Literal["harnext-strategies-v1"] = "harnext-strategies-v1"
    representation: Literal["files", "graph", "hybrid"] = "files"
    source_jsonl: Path
    artifact_dir: Path
    observed_at: datetime
    valid_at: datetime
    builder: AgentConfig = Field(default_factory=AgentConfig)
    files: FileConfig = Field(default_factory=FileConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    expose_sources: bool = False
    trial: TrialConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def compatible(self):
        if any(value.utcoffset() is None for value in (self.observed_at, self.valid_at)):
            raise ValueError("cutoffs must be timezone-aware")
        learned = (
            self.files.extraction == Extraction.ATOMIC
            or self.files.summarization == Summary.ABSTRACTIVE
            or self.files.layout == FileLayout.CURATED
        )
        if learned and self.builder.harness == "none":
            raise ValueError("selected extraction/summary/layout requires a learned builder")
        if self.files.layout in {FileLayout.VERBATIM, FileLayout.FOLDERED} and (
            self.files.summarization != Summary.NONE
            or self.files.extraction != Extraction.STRUCTURED
            or self.files.retention == Retention.CURRENT
            or self.files.prompt != WritingPrompt.PRESERVE
        ):
            raise ValueError(
                "verbatim/foldered layouts require structured extraction, preserve_supported, no summary, and all/window retention"
            )
        graph_tools = {"find_entities", "get_assertions", "neighbors"}
        if self.representation == "files" and graph_tools.intersection(self.trial.tools):
            raise ValueError("graph tools require graph or hybrid representation")
        if "fetch_source" in self.trial.tools and not self.expose_sources:
            raise ValueError("fetch_source requires expose_sources")
        return self


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def digest(value) -> str:
    return hashlib.sha256(
        value if isinstance(value, bytes) else canonical(value).encode()
    ).hexdigest()


def load_config(path: Path) -> StrategyConfig:
    config = StrategyConfig.model_validate(yaml.safe_load(path.read_text()))
    for name in ("source_jsonl", "artifact_dir"):
        value = getattr(config, name)
        if not value.is_absolute():
            setattr(config, name, (path.resolve().parent / value).resolve())
    for name in (
        "task_file",
        "gold_file",
        "answer_file",
        "candidate_files",
        "workspace_dir",
        "agent_trace",
    ):
        value = getattr(config.evaluation, name)
        if value is not None and not value.is_absolute():
            setattr(config.evaluation, name, (path.resolve().parent / value).resolve())
    cache = Path(config.retrieval.embeddings.cache_dir)
    if not cache.is_absolute():
        config.retrieval.embeddings.cache_dir = str((path.resolve().parent / cache).resolve())
    return config
