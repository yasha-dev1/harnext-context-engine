"""Model clients: Jev (OpenRouter decisions endpoint) and an endpoint-agnostic OpenAI-compatible
chat client (OpenRouter or a local Ollama), both recording tokens, cost and latency per call,
plus deterministic fakes for tests.

A provider or machine failure (401/402/403, 429 and 5xx after bounded retries, timeouts, an
unreachable endpoint) raises `InfraError` so the benchmark runner can pause instead of recording
a wrong answer; a request the provider rejects as malformed raises `ProviderError` (a bug).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from harnext_eval.e2.v2.engine.config import EngineConfig

FATAL = (401, 402, 403)


class InfraError(RuntimeError):
    """The provider or the machine failed; nothing about the question was learned."""


class ProviderError(ValueError):
    """The provider rejected the request itself (400/404/422), or answered unusably; `research`
    turns it into InfraError so the runner pauses instead of grading a failed call."""


class ModelOutputError(ValueError):
    """The endpoint could not parse the model's own reply (e.g. Ollama's HTTP 500 "error parsing
    tool call"): a model failure, not infrastructure. `.raw` holds the model's text."""


class PromptTooLarge(ProviderError):  # noqa: N818
    """HTTP 413 or a prompt the endpoint truncated: the caller may shrink the prompt once."""


@dataclass
class Call:
    kind: str  # jev | chat
    model: str
    purpose: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cost_reported: bool  # False: computed from tokens at list price
    seconds: float
    reasoning: bool = False  # the reply carried reasoning text (counted in output tokens)
    attempts: int = 1  # HTTP requests made for this call (retries included)
    failures: list = field(default_factory=list)  # why earlier attempts failed
    cancelled: bool = False  # cut at the research deadline before an answer arrived
    unmetered_usd: float = 0.0  # upper bound for requests that returned no usage
    reasoning_tokens: int = 0  # part of output_tokens, as the provider reports it
    cached_tokens: int = 0  # part of input_tokens served from the provider's prompt cache


def price(tokens_in: int, tokens_out: int, per_million: tuple[float, float]) -> float:
    return (tokens_in * per_million[0] + tokens_out * per_million[1]) / 1e6


class Meter:
    """Every model call of one research call."""

    def __init__(self):
        self.calls: list[Call] = []

    def add(self, call: Call) -> None:
        self.calls.append(call)

    def totals(self, kind: str) -> dict:
        calls = [c for c in self.calls if c.kind == kind]
        extra = {"with_reasoning": sum(c.reasoning for c in calls)} if kind == "chat" else {}
        extra |= {"reasoning_tokens": sum(c.reasoning_tokens for c in calls), "cached_tokens": sum(c.cached_tokens for c in calls)} if kind == "chat" else {}  # fmt: skip
        extra |= {"retries": sum(c.attempts - 1 for c in calls), "cancelled": sum(c.cancelled for c in calls),
                  "unmetered_upper_usd": round(sum(c.unmetered_usd for c in calls), 8)}  # fmt: skip
        return {"calls": len(calls), "input_tokens": sum(c.input_tokens for c in calls),
                "output_tokens": sum(c.output_tokens for c in calls),
                "cost_usd": round(sum(c.cost_usd for c in calls), 8),
                "cost_reported": all(c.cost_reported for c in calls),
                "seconds": round(sum(c.seconds for c in calls), 3), **extra}  # fmt: skip

    def summary(self) -> dict:
        jev, chat = self.totals("jev"), self.totals("chat")
        return {"jev": jev, "chat": chat, "cost_usd": round(jev["cost_usd"] + chat["cost_usd"], 8),
                "unmetered_upper_usd": round(jev["unmetered_upper_usd"] + chat["unmetered_upper_usd"], 8),
                "input_tokens": jev["input_tokens"] + chat["input_tokens"],
                "output_tokens": jev["output_tokens"] + chat["output_tokens"]}  # fmt: skip

    def records(self) -> list[dict]:
        return [asdict(c) for c in self.calls]


def load_key(name: str | None) -> str | None:
    """The key from the environment, else from the nearest `.env` above this file."""
    if not name:
        return None
    if os.environ.get(name):
        return os.environ[name]
    for folder in Path(__file__).resolve().parents:
        env = folder / ".env"
        if env.is_file():
            for line in env.read_text().splitlines():
                key, _, value = line.partition("=")
                if key.strip().removeprefix("export ").strip() == name:
                    return value.strip().strip("\"'")
    return None


def describe(status: int, body: str) -> str:
    return f"HTTP {status}: {' '.join(body.split())[:300]}"


RETRY = (408, 429)


async def post(http: httpx.AsyncClient, url: str, headers: dict, body: dict, config: EngineConfig, seconds: float | None = None) -> tuple[dict, list[str]]:  # fmt: skip
    """(data, failed attempts) after bounded retries on 408/429/5xx/timeouts/transport errors.
    Exhausted retries and auth/credit refusals raise InfraError, 413 PromptTooLarge, other 4xx
    ProviderError; each carries `.failures` (the failed attempts, for metering)."""
    failures, timeout = [], seconds or config.timeout_s

    def fail(kind, message):
        error = kind(message)
        error.failures = failures
        return error

    for attempt in range(config.retries + 1):
        if attempt:
            await asyncio.sleep(config.backoff_s * 2 ** (attempt - 1))
        try:
            # httpx's timeout bounds each read, not the request: a trickling reply is cut here
            response = await asyncio.wait_for(http.post(url, headers=headers, json=body, timeout=timeout), timeout)  # fmt: skip
        except (httpx.TimeoutException, TimeoutError) as error:
            failures.append(f"timeout after {timeout:g} s ({type(error).__name__})")
            continue
        except httpx.TransportError as error:
            failures.append(f"endpoint unreachable ({type(error).__name__}: {error})")
            continue
        status, text = response.status_code, response.text
        if status in FATAL:
            raise fail(InfraError, describe(status, text))
        if status == 413:
            raise fail(PromptTooLarge, describe(status, text))
        if status >= 500 and "error parsing tool call" in text:  # deterministic: not retried
            error = fail(ModelOutputError, describe(status, text))
            error.raw = unparsed(text)
            raise error
        if status in RETRY or status >= 500:
            failures.append(describe(status, text))
            continue
        if status >= 400:
            raise fail(ProviderError, describe(status, text))
        try:
            data = response.json()
        except ValueError:
            failures.append(f"invalid JSON from provider: {text[:200]}")
            continue
        error = data.get("error") if isinstance(data, dict) else None
        if error:  # OpenRouter reports some upstream failures inside a 200
            code = error.get("code") if isinstance(error, dict) else None
            message = f"provider error {code}: {json.dumps(error)[:300]}"
            if code in FATAL:
                raise fail(InfraError, message)
            if code == 413:
                raise fail(PromptTooLarge, message)
            if isinstance(code, int) and 400 <= code < 500 and code not in RETRY:
                raise fail(ProviderError, message)
            failures.append(message)
            continue
        return data, failures
    raise fail(InfraError, f"{url}: {failures[-1] if failures else 'no attempt'} (after {config.retries + 1} attempts)")  # fmt: skip


def unparsed(body: str) -> str:
    """The model text inside an endpoint's tool-call parse error ("… raw='<text>' …")."""
    try:
        message = json.loads(body).get("error", {}).get("message", body)
    except (ValueError, AttributeError):
        message = body
    return message.split("raw='", 1)[1].rsplit("'", 1)[0] if "raw='" in message else message


def bound(body: dict, per_million: tuple[float, float], out_tokens: int) -> float:
    """Upper bound of the cost of one request that returned no usage."""
    return price(len(json.dumps(body, ensure_ascii=False)) // 3, out_tokens, per_million)


class JevClient:
    def __init__(self, config: EngineConfig, key: str | None = None, http: httpx.AsyncClient | None = None):  # fmt: skip
        self.config, self.http = config, http or httpx.AsyncClient()
        self.key = key or load_key(config.key_env)
        if not self.key:
            raise InfraError(f"no API key for Jev (${config.key_env})")

    async def decide(self, state, questions: dict, purpose: str = "") -> tuple[dict, Call]:
        started = time.monotonic()
        body = {"model": self.config.jev_model, "state": state, "questions": questions}
        try:
            data, failures = await post(self.http, self.config.jev_url, {"Authorization": f"Bearer {self.key}"}, body, self.config, self.config.jev_timeout_s)  # fmt: skip
        except (InfraError, ProviderError) as error:
            error.call = self.unanswered(body, purpose, getattr(error, "failures", []), started)
            raise
        usage = data.get("usage") or {}
        tokens = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        cost = usage.get("cost")
        call = Call("jev", data.get("model", self.config.jev_model), purpose, *tokens,
                    cost if cost is not None else price(*tokens, self.config.jev_price),
                    cost is not None, round(time.monotonic() - started, 3), attempts=len(failures) + 1,
                    failures=failures, unmetered_usd=len(failures) * bound(body, self.config.jev_price, 0))  # fmt: skip
        answers = data.get("answers") or {}
        if missing := [q for q in questions if q not in answers]:
            error = ProviderError(f"decisions answer lacks {missing[:5]}")
            error.call = call  # the provider answered and billed: meter it
            raise error
        return answers, call

    def unanswered(self, body: dict, purpose: str, failures: list, started: float) -> Call:
        return Call("jev", self.config.jev_model, purpose, 0, 0, 0.0, False, round(time.monotonic() - started, 3),
                    attempts=max(1, len(failures)), failures=failures,
                    unmetered_usd=max(1, len(failures)) * bound(body, self.config.jev_price, 0))  # fmt: skip

    async def close(self) -> None:
        await self.http.aclose()


class ChatClient:
    """OpenAI-compatible /chat/completions: OpenRouter, Ollama or any other base URL."""

    def __init__(self, config: EngineConfig, key: str | None = None, http: httpx.AsyncClient | None = None):  # fmt: skip
        self.config, self.http = config, http or httpx.AsyncClient()
        self.key = key or load_key(config.chat_key_env)
        if config.chat_key_env and not self.key:
            raise InfraError(f"no API key for the chat model (${config.chat_key_env})")

    async def complete(self, messages: list[dict], *, tools: list[dict] | None = None,
                       response_format: dict | None = None, max_tokens: int | None = None,
                       purpose: str = "", options: dict | None = None) -> tuple[dict, Call]:  # fmt: skip
        """`options` override `chat_options` for this call (an answer-now call's reasoning)."""
        started, config = time.monotonic(), self.config
        body = {"model": config.chat_model, "messages": messages,
                "max_tokens": max_tokens or config.max_tokens, **config.chat_options, **(options or {})}  # fmt: skip
        if config.temperature is not None:
            body["temperature"] = config.temperature
        if config.seed is not None:
            body["seed"] = config.seed
        if tools:
            body["tools"] = tools
        if response_format and config.chat_response_format:
            body["response_format"] = response_format
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        worst = bound(body, config.chat_price, body["max_tokens"])
        try:
            data, failures = await post(self.http, config.chat_base_url.rstrip("/") + "/chat/completions", headers, body, config)  # fmt: skip
        except (InfraError, ProviderError, ModelOutputError) as error:
            fails = getattr(error, "failures", [])
            error.call = Call("chat", config.chat_model, purpose, 0, 0, 0.0, False, round(time.monotonic() - started, 3),
                              attempts=max(1, len(fails)), failures=fails, unmetered_usd=max(1, len(fails)) * worst)  # fmt: skip
            raise
        choices = data.get("choices") or []
        message = dict(choices[0].get("message") or {}) if choices else {}
        message["finish_reason"] = choices[0].get("finish_reason") if choices else None
        usage = data.get("usage") or {}
        tokens = usage.get("prompt_tokens", 0) or 0, usage.get("completion_tokens", 0) or 0
        cost = usage.get("cost")
        call = Call("chat", data.get("model", config.chat_model), purpose, *tokens,
                    cost if cost is not None else price(*tokens, config.chat_price),
                    cost is not None, round(time.monotonic() - started, 3), bool(message.get("reasoning")),
                    attempts=len(failures) + 1, failures=failures, unmetered_usd=len(failures) * worst,
                    reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
                    cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)  # fmt: skip
        if not choices:
            error = InfraError(f"chat endpoint returned no choices: {json.dumps(data)[:300]}")
            error.call = call
            raise error
        if config.chat_context and tokens[0] >= config.chat_context:
            error = PromptTooLarge(f"prompt truncated by the endpoint ({tokens[0]} >= context {config.chat_context} tokens)")  # fmt: skip
            error.call = call  # answered and counted: meter it
            raise error
        return message, call

    async def close(self) -> None:
        await self.http.aclose()


# ---- deterministic fakes (tests, offline smoke) ---------------------------------------------


def noul(p: float) -> dict:
    return {"type": "noul", "noul": p}


def choice(option: str, confidence: float = 1.0) -> dict:
    return {"type": "choice", "choice": option, "probabilities": {option: confidence}, "confidence": confidence}  # fmt: skip


def default_answer(question: dict) -> dict:
    if question["type"] == "noul":
        return noul(0.5)
    if question["type"] == "choice":
        return choice(next(iter(question["criteria"])))
    return {"type": "score", "score": (len(question["criteria"]) - 1) / 2, "confidence": 1.0}


class FakeJev:
    """`answer(state, qid, question)` returns an answer dict or None (default answer)."""

    def __init__(self, answer: Callable | None = None, fail: Exception | None = None):
        self.answer, self.fail, self.calls = answer or (lambda *_: None), fail, []

    async def decide(self, state, questions: dict, purpose: str = "") -> tuple[dict, Call]:
        self.calls.append({"state": state, "questions": questions, "purpose": purpose})
        if self.fail:
            raise self.fail
        answers = {q: self.answer(state, q, spec) or default_answer(spec) for q, spec in questions.items()}  # fmt: skip
        tokens = len(json.dumps([state, questions])) // 4
        return answers, Call("jev", "fake-jev", purpose, tokens, 10, price(tokens, 10, (0.042, 0)), False, 0.0)  # fmt: skip

    async def close(self) -> None:
        pass


def tool_call(name: str, arguments, id_: str | None = None) -> dict:
    """An assistant message calling one tool; `arguments` may be a raw (malformed) string."""
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"role": "assistant", "content": None, "finish_reason": "tool_calls",
            "tool_calls": [{"id": id_ or f"call_{name}", "type": "function",
                            "function": {"name": name, "arguments": raw}}]}  # fmt: skip


def text(content: str) -> dict:
    return {"role": "assistant", "content": content, "finish_reason": "stop"}


class FakeChat:
    """`script` is a list of replies (consumed in order) or `fn(messages, tools, purpose)`."""

    def __init__(self, script, fail: Exception | None = None):
        self.script, self.fail, self.calls = script, fail, []

    async def complete(self, messages, *, tools=None, response_format=None, max_tokens=None, purpose="", options=None):  # fmt: skip
        names = [t["function"]["name"] for t in tools or []]
        self.calls.append({"messages": [dict(m) for m in messages], "tools": names, "specs": tools or [], "purpose": purpose,
                           **({"options": options, "max_tokens": max_tokens} if options is not None else {})})  # fmt: skip
        if self.fail:
            raise self.fail
        if callable(self.script):
            reply = self.script(messages, tools, purpose)
        else:
            reply = self.script.pop(0) if self.script else text("(script exhausted)")
        tokens = len(json.dumps(messages)) // 4, len(json.dumps(reply)) // 4
        return dict(reply), Call("chat", "fake-chat", purpose, *tokens, price(*tokens, (0.018, 0.09)), False, 0.0)  # fmt: skip

    async def close(self) -> None:
        pass
