"""State of one research call: budgets, the usage meter, model concurrency, and the one shared
database handle (chdb binds one embedded server per process, so every query of every branch
runs on a single worker thread, one at a time)."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from harnext_eval.e2.v2.engine.clients import Call, Meter, bound
from harnext_eval.e2.v2.engine.config import EngineConfig
from harnext_eval.e2.v2.engine.plan import Catalog

GRACE = 2.0  # seconds a model call may still take once the deadline has passed


class BudgetExceeded(RuntimeError):  # noqa: N818
    pass


def truncated(message: dict) -> bool:
    """No text and no tool call, and the reply stopped at the token limit."""
    empty = not (message.get("content") or "").strip() and not message.get("tool_calls")
    return empty and message.get("finish_reason") == "length"


class SharedDB:
    """A RawDB behind one worker thread: the lock every branch shares."""

    def __init__(self, db):
        self.db = db
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="raw-db")
        self.catalog = Catalog(db.catalog_doc)

    async def run(self, fn, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(self.pool, partial(fn, *args, **kwargs))  # fmt: skip

    def reference(self) -> str:
        """What the fallback `sql` accepts (the store's `sql_reference`, else the guard's)."""
        from harnext_eval.e2.v2.stores.raw import guard

        return getattr(self.db, "sql_reference", guard.summary)()

    async def sql(self, query: str, **limits) -> dict:
        return await self.run(self.db.sql, query, **limits)

    async def sources(self, ids: list[str]) -> dict[str, str]:
        """{event id: table} for the ids that exist (one UNION query)."""
        from harnext_eval.e2.v2.engine.plan import lit

        tables = sorted(self.catalog.types)
        wanted = ", ".join(lit(i) for i in ids)
        union = " UNION ALL ".join(f"SELECT {lit(t)} AS `source`, `event_id` FROM `{t}` WHERE `event_id` IN ({wanted})" for t in tables)  # fmt: skip
        result = await self.sql(union, max_rows=len(ids) * 2 + 1)
        return {r["event_id"]: r["source"] for r in result["rows"]}

    async def get(self, event_id: str) -> dict | None:
        return await self.run(self.db.get, event_id)

    def close(self) -> None:
        self.pool.submit(self.db.close).result()
        self.pool.shutdown()


class Session:
    def __init__(self, config: EngineConfig, jev, chat, db: SharedDB | None, memory=None):
        self.config, self.jev, self.chat, self.db, self.memory = config, jev, chat, db, memory
        self.meter = Meter()
        self.started = time.monotonic()
        self.jev_gate = asyncio.Semaphore(config.jev_concurrency)
        self.chat_gate = asyncio.Semaphore(config.chat_concurrency)
        self.jev_calls = 0

    def parallel(self, calls: int) -> None:
        """Allow `calls` chat calls at once (engine v2: one per branch, set before they start)."""
        self.chat_gate = asyncio.Semaphore(max(1, calls))

    @property
    def catalog(self) -> Catalog:
        return self.db.catalog

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def left(self) -> float:
        return self.config.deadline_s - self.elapsed()

    def chat_tokens(self) -> int:
        return sum(c.input_tokens + c.output_tokens for c in self.meter.calls if c.kind == "chat")

    def exhausted(self) -> str | None:
        """Why the branches must stop now, or None."""
        if self.left() <= self.config.reserve_s:
            return f"time budget ({self.config.deadline_s:g} s)"
        if self.chat_tokens() >= self.config.max_chat_tokens:
            return f"token budget ({self.config.max_chat_tokens} chat tokens)"
        return None

    async def decide(self, state, questions: dict, purpose: str) -> dict:
        if self.jev_calls >= self.config.max_jev_calls:
            raise BudgetExceeded(f"decision budget ({self.config.max_jev_calls} Jev calls) is used up")  # fmt: skip
        self.jev_calls += 1
        estimate = bound({"state": state, "questions": questions}, self.config.jev_price, 0)
        async with self.jev_gate:
            answers, call = await self.bounded(self.jev.decide(state, questions, purpose=purpose), "jev", purpose, estimate)  # fmt: skip
        self.meter.add(call)
        return answers

    async def complete(self, messages: list[dict], purpose: str, **options) -> dict:
        """One chat call. With `retry_truncated`, a reply with no text and no tool call that
        stopped at the output-token limit (the model spent it all on reasoning) is asked once
        more with twice the limit and the answer-now options (`answer_options`: the lowest
        reasoning the model accepts; a second reasoning spiral would take as long again); both
        calls are metered."""
        message = await self.complete_once(messages, purpose, **options)
        if self.config.retry_truncated and truncated(message):
            limit = 2 * (options.get("max_tokens") or self.config.max_tokens)
            lower = {"options": self.config.answer_options} if self.config.answer_options else {}
            message = await self.complete_once(messages, f"{purpose}:longer", **{**options, **lower, "max_tokens": limit})  # fmt: skip
        return message

    async def complete_once(self, messages: list[dict], purpose: str, **options) -> dict:
        body = {"messages": messages, "tools": options.get("tools")}
        estimate = bound(body, self.config.chat_price, options.get("max_tokens") or self.config.max_tokens)  # fmt: skip
        async with self.chat_gate:
            message, call = await self.bounded(self.chat.complete(messages, purpose=purpose, **options), "chat", purpose, estimate)  # fmt: skip
        self.meter.add(call)
        return message

    async def bounded(self, call, kind: str, purpose: str, estimate: float):
        """A model call cut at the research deadline (at least GRACE seconds): BudgetExceeded.
        Every outcome is metered: failures carry their Call, a cancelled call is recorded with
        an upper bound of its cost."""
        started = time.monotonic()
        try:
            return await asyncio.wait_for(call, max(GRACE, self.left()))
        except TimeoutError:
            self.meter.add(Call(kind, "", purpose, 0, 0, 0.0, False, round(time.monotonic() - started, 3), cancelled=True, unmetered_usd=estimate))  # fmt: skip
            raise BudgetExceeded(f"time budget ({self.config.deadline_s:g} s) ran out during a model call") from None  # fmt: skip
        except asyncio.CancelledError:  # its branch was cancelled at the deadline while it ran
            self.meter.add(Call(kind, "", purpose, 0, 0, 0.0, False, round(time.monotonic() - started, 3), cancelled=True, unmetered_usd=estimate))  # fmt: skip
            raise
        except Exception as error:
            if isinstance(getattr(error, "call", None), Call):
                self.meter.add(error.call)
            raise
