"""Every threshold, budget and model setting of the engine, and a stable hash of config + prompts
(stage 3 pins it per arm). Transport fields (URLs, key variable, timeouts, retries) are not hashed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from harnext_eval.e2.v2.engine import prompts

TRANSPORT = {"jev_url", "chat_base_url", "key_env", "chat_key_env", "timeout_s", "jev_timeout_s", "retries", "backoff_s"}  # fmt: skip


class EngineConfig(BaseModel):
    # models
    jev_url: str = "https://openrouter.ai/api/alpha/decisions"
    jev_model: str = "typesafe/jev-1.13"
    jev_price: tuple[float, float] = (0.042, 0.0)  # US$ per million input / output tokens
    key_env: str = "OPENROUTER_API_KEY"
    # chat model: local Ollama by default (E2-ENGINE-OLLAMA.md); ENDPOINTS["openrouter"] for the API
    chat_base_url: str = "http://localhost:11434/v1"
    chat_model: str = "gpt-oss-20b-ctx32k"  # gpt-oss:20b with num_ctx 32768 pinned
    chat_key_env: str | None = None  # None: no Authorization header (Ollama)
    # list price per million input / output tokens, used when the endpoint sends no cost
    chat_price: tuple[float, float] = (0.018, 0.09)
    chat_options: dict = Field(default_factory=lambda: {"reasoning_effort": "low"})
    chat_context: int | None = (
        32768  # prompt_tokens >= this means the endpoint truncated the prompt
    )
    max_prompt_tokens: int = 28_000  # older tool results are elided above this estimate
    chat_response_format: bool = False  # Ollama: a format hides the reasoning tokens from usage
    chat_echo_reasoning: bool = True  # send the model's reasoning back with its tool calls
    seed: int | None = 7
    temperature: float | None = 0.0
    max_tokens: int = 2048  # always sent (an unbounded local request once ran > 10 min)
    timeout_s: float = 60.0  # per chat request
    jev_timeout_s: float = 12.0  # per decisions request (typically 0.3 s; a stuck one is retried)
    retries: int = 3
    backoff_s: float = 2.0
    chat_concurrency: int = 1  # one chat call at a time: no batching effects on results
    jev_concurrency: int = 4
    # routing
    route_threshold: float = 0.5
    max_branches: int = 3
    # branches
    cross_source: Literal["branches", "single"] = (
        "branches"  # several routed sources: one agent each, or one over all
    )
    branch_turns: int = 8
    confidence: float = 0.6  # a plan slot below this is "uncertain"
    restate: bool = True  # ask the agent to restate a need once when a slot is uncertain
    sql_fallback: Literal["auto", "always", "never"] = "always"
    fetch_rows: int = 200  # rows a plan fetches
    show_rows: int = 25  # rows returned to the agent after the relevance filter
    timeline_rows: int = 60
    show_chars: int = 7000  # characters of rendered rows per tool result
    cell_chars: int = 300  # a long text cell is cut to this (with the event id kept)
    relevance_batch: int = 25
    relevance_rows: int = 75  # rows scored at most per find
    relevance_min: float = 0.3
    read_chars: int = 6000
    memory_hits: int = 40
    memory_show: int = 10
    memory_lines: int = 80
    # aggregation
    finding_threshold: float = 0.3
    factual_threshold: float = 0.3  # a "nothing found" finding is dropped when another states facts
    support_threshold: float = 0.3  # finding vs the records its branch retrieved
    support_chars: int = 12_000  # of the cited rows (else the latest results) a support check sees
    support_share: float = 0.8  # share of a finding's values found in retrieved rows to keep it
    combine_events: int = 8  # cited events per finding offered to the combine step
    report_tokens: int = 2500
    verify_report: bool = True  # False (ablation): no value check, regeneration or Jev verdict
    finding_chars: int = 8000  # of each finding passed to the report
    grounds_chars: int = 3000  # of the rows a finding cites, passed with it to the report
    # budgets per research call
    deadline_s: float = 120.0
    reserve_s: float = 20.0  # kept back for aggregation and the report
    max_jev_calls: int = 60
    max_chat_tokens: int = 250_000
    # engine v2: tool-call budget feedback (off = the v1 behaviour: turn budget and cut-off)
    budget_feedback: bool = False
    tool_budget: int = 8  # tool calls per branch agent (done excluded; rejected calls count)
    answer_turns: int = 3  # model turns allowed beyond the budget to answer before a forced finish
    answer_reserve_s: float = 30.0  # before the branch cut-off: tool calls are refused, answer now
    conclude_s: float = 40.0  # at the cut-off, each unfinished branch answers in one call this long
    retry_truncated: bool = False  # a reply cut at the token limit with no text: ask once with 2x
    # 0: `chat_concurrency` holds; n > 0: as many chat calls at once as the research call has
    # branches, at most n (branches run side by side; findings are ordered by source name)
    parallel_branches: int = 0
    answer_options: dict = Field(default_factory=dict)  # chat options of an answer-now call
    answer_tokens: int = 0  # output allowance of an answer-now call (0: the usual limits)


V2 = (
    "budget_feedback",
    "tool_budget",
    "answer_turns",
    "answer_reserve_s",
    "conclude_s",
    "retry_truncated",
    "parallel_branches",
    "answer_options",
    "answer_tokens",
)  # hashed only when on
OPENROUTER = {"chat_base_url": "https://openrouter.ai/api/v1", "chat_key_env": "OPENROUTER_API_KEY",
              "chat_context": None, "chat_response_format": True}  # fmt: skip
# deadline policy of every budget-feedback preset, from smoke measurements (E2-ENGINE-DESIGN.md,
# engine v2): research deadline, time kept for aggregation, and the answer-now margin before it
V2_TIMES = {"deadline_s": 240.0, "reserve_s": 90.0, "answer_reserve_s": 30.0, "conclude_s": 40.0}
# every budget-feedback preset: branches side by side, and an answer-now call (after the budget
# or time message, the forced finish, the conclusion at the cut-off) with a fixed allowance
V2_RUN = {"budget_feedback": True, "tool_budget": 8, "retry_truncated": True, "parallel_branches": 4,
          "answer_tokens": 1500, **V2_TIMES}  # fmt: skip
USAGE = {"usage": {"include": True}}
LOW, NONE = (
    {"reasoning": {"effort": "low"}},
    {"reasoning": {"effort": "none"}},
)  # gpt-oss: low is lowest
ENDPOINTS = {
    "ollama": {},  # the defaults
    "openrouter": {"chat_base_url": "https://openrouter.ai/api/v1", "chat_model": "openai/gpt-oss-20b",
                   "chat_key_env": "OPENROUTER_API_KEY", "chat_context": None, "chat_response_format": True,
                   "chat_options": {"reasoning": {"effort": "low"}, "usage": {"include": True}}},
    "openrouter-b8": {**OPENROUTER, "chat_model": "openai/gpt-oss-20b", "chat_options": {**LOW, **USAGE},
                      "answer_options": LOW, **V2_RUN},
    "deepseek-b8": {**OPENROUTER, "chat_model": "deepseek/deepseek-v4.1-flash", "chat_price": (0.30, 1.20),
                    "chat_options": {**LOW, **USAGE}, "answer_options": NONE, **V2_RUN},
    "deepseek-flash-b8": {**OPENROUTER, "chat_model": "deepseek/deepseek-v4-flash-0731", "chat_price": (0.018, 0.32),
                          "chat_options": {**LOW, **USAGE}, "answer_options": NONE, **V2_RUN},
}  # fmt: skip


def for_endpoint(name: str, **overrides) -> EngineConfig:
    return EngineConfig(**{**ENDPOINTS[name], **overrides})


def source_files() -> dict[str, str]:
    """sha256 of every source file of this package, by relative path."""
    root = Path(__file__).resolve().parent
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*.py"))}  # fmt: skip


def code_hash() -> str:
    """sha256 over the package's sources (sorted paths and file bytes): pins the code."""
    root, digest = Path(__file__).resolve().parent, hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(
            path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0"
        )
    return digest.hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def config_hash(config: EngineConfig) -> str:
    """sha256 of the behaviour-relevant settings and every prompt text."""
    settings = config.model_dump(exclude=TRANSPORT | (set() if config.budget_feedback else set(V2)))
    texts = prompts.texts(budget=config.budget_feedback)
    return hashlib.sha256(canonical({"config": settings, "prompts": texts}).encode()).hexdigest()  # fmt: skip
