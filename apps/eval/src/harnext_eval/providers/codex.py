"""Codex reader completion provider for evaluation-spec §5 and merged E2.

Native tools are disabled. The merged reader exposes only host-budgeted tools
over an immutable snapshot. Generation length is observed, not hard-capped by
the CLI; max_tokens is a prompt instruction and must not be used as a cost cap.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from typing import Any

from harnext_builder.harness.codex import execute

from harnext_eval.providers.llm import LLMResult
from harnext_eval.providers.tokenizer import FakeTokenCounter


class CodexLLM:
    def __init__(self, *, model: str, reasoning_effort: str | None, timeout_s: int = 120):
        if not model or reasoning_effort is None:
            raise ValueError("Codex reader requires model and reasoning_effort")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_s = timeout_s
        # Compatibility for legacy smoke readers only; merged E2 uses an explicit
        # byte budget. This is never presented as the provider's exact tokenizer.
        self.tokenizer = FakeTokenCounter()
        self.calls: list[dict[str, Any]] = []

    def complete(self, system: str, user: str, *, json_schema: dict | None = None,
                 max_tokens: int) -> LLMResult:
        with tempfile.TemporaryDirectory(prefix="harnext-codex-reader-") as cwd:
            transcript = asyncio.run(execute(
                model=self.model, effort=self.reasoning_effort, cwd=cwd,
                prompt=f"{system}\n\n{user}\nKeep the response below {max_tokens} tokens.",
                tools_enabled=False, timeout_s=self.timeout_s, json_schema=json_schema,
            ))
        self.calls.append(transcript.model_dump(mode="json"))
        if not transcript.ok:
            raise RuntimeError(transcript.error or "Codex reader failed")
        messages = [turn.content for turn in transcript.turns if turn.role == "assistant"]
        if not messages:
            raise RuntimeError("Codex returned no final message")
        message = messages[-1]
        parsed = json.loads(message) if json_schema else None
        usage = {key: value for key, value in transcript.usage.items()
                 if isinstance(value, int) and not isinstance(value, bool)}
        return LLMResult(text=message, json=parsed, usage=usage)
