"""One branch: a gpt-oss agent loop over one source's tools, ending with done(finding, evidence).

Malformed tool-call JSON, unknown tools and bad arguments come back to the model as tool
errors. On a spent turn/time/token budget the model is asked once, without tools, for what it
has found (status partial); if even that is impossible the branch's last tool results are the
partial finding. Only provider failures (InfraError) escape.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field

from harnext_eval.e2.v2.engine import prompts
from harnext_eval.e2.v2.engine.clients import (
    Call,
    InfraError,
    ModelOutputError,
    PromptTooLarge,
    ProviderError,
    bound,
)
from harnext_eval.e2.v2.engine.rawtools import function
from harnext_eval.e2.v2.engine.session import Session

DONE = function(
    "done",
    "Finish: `finding` is a compact statement, in full sentences naming their subject, of the "
    "facts that bear on the question (or that nothing relevant was found); `evidence` lists the "
    "record identifiers (event ids, notes file names) they come from.",
    {"finding": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "string"}}},
    ["finding", "evidence"],
)


@dataclass
class Branch:
    source: str
    kind: str  # raw | memory
    status: str = "running"  # done | partial | failed
    finding: str = ""
    evidence: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    turns: int = 0
    malformed: int = 0
    error: str | None = None
    results: list[str] = field(default_factory=list)  # tool results, for a partial finding
    tools: object = None
    tool_calls: int = 0  # tool calls counted against the budget (done excluded)
    ending: str = ""  # normal | budget | time | forced | turns | cutoff | failed
    started_s: float | None = None  # seconds into the research call
    ended_s: float | None = None
    messages: list = field(default_factory=list, repr=False)  # the agent's conversation (v2)
    slowest_turn_s: float = 0.0  # longest model turn of this branch so far (v2 answer-now timing)

    def salvage(self, reason: str) -> None:
        """Mark partial from the last tool results when the branch could not finish."""
        if self.status in ("done", "partial"):
            return
        if self.status == "running":
            self.status = "partial"
        self.error = self.error or reason
        if not self.finding:
            tail = "\n".join(r[:1500] for r in self.results[-2:])
            self.finding = f"(unfinished: {reason}) Last results:\n{tail}" if tail else f"(unfinished: {reason}) nothing found yet"  # fmt: skip

    def trace(self) -> dict:
        tools = self.tools
        return {"source": self.source, "kind": self.kind, "status": self.status, "finding": self.finding,
                "evidence": self.evidence, "unverified_evidence": self.unverified, "turns": self.turns,
                "malformed_calls": self.malformed, "error": self.error,
                "tool_calls": self.tool_calls, "ending": self.ending or {"done": "normal", "failed": "failed"}.get(self.status, "cutoff"),
                "started_s": self.started_s, "ended_s": self.ended_s,
                "calls": getattr(tools, "log", []), "results": self.results,
                "sql_unlocked": getattr(tools, "unlocked", None)}  # fmt: skip


def echo(message: dict, reasoning: bool) -> dict:
    kept = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("tool_calls"):
        kept["tool_calls"] = message["tool_calls"]
    if reasoning and message.get("reasoning"):
        kept["reasoning"] = message["reasoning"]
    return kept


LEAKED = re.compile(r"to=functions\.(\w+).*?<\|message\|>(.*?)(?:<\|call\|>|$)", re.DOTALL)


DSML_BLOCK = re.compile(r"<｜DSML｜tool_calls>.*?(?:</｜DSML｜tool_calls>|$)", re.DOTALL)
DSML_INVOKE = re.compile(r'<｜DSML｜invoke name="([^"]+)">(.*?)</｜DSML｜invoke>', re.DOTALL)
DSML_PARAM = re.compile(
    r'<｜DSML｜parameter name="([^"]+)" string="(true|false)">(.*?)</｜DSML｜parameter>', re.DOTALL
)


def dsml(content: str) -> list[tuple[str, str]]:
    """(name, JSON arguments) of the tool calls DeepSeek left in the text as DSML markup."""
    found = []
    for name, body in DSML_INVOKE.findall(content or ""):
        args = {}
        for key, is_string, value in DSML_PARAM.findall(body):
            try:
                args[key] = value if is_string == "true" else json.loads(value)
            except ValueError:
                args[key] = value
        found.append((name, json.dumps(args, ensure_ascii=False)))
    return found


def without_leaks(content: str) -> str:
    """The text with leaked DSML tool-call blocks removed."""
    return DSML_BLOCK.sub("", content or "").strip()


def leaked(content: str) -> list[dict]:
    """Tool calls a provider left in the text: gpt-oss harmony tokens or DeepSeek DSML markup."""
    found = [(name, body.strip()) for name, body in LEAKED.findall(content or "")] + dsml(content)
    return [{"id": f"leaked_{i}", "type": "function", "function": {"name": name, "arguments": args}} for i, (name, args) in enumerate(found)]  # fmt: skip


def arguments(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    value = json.loads(raw or "{}")
    if not isinstance(value, dict):
        raise ValueError("arguments must be a JSON object")
    return value


def cited(text: str, seen: set[str]) -> list[str]:
    return [s for s in sorted(seen) if s in text]


def trim(messages: list[dict], limit: int, specs: list | None = None, keep: int = 2) -> list[dict]:
    """Elide the oldest tool results (not the last `keep`) while the prompt estimate, tool
    schemas included, exceeds `limit` tokens (~3 characters per token, on the safe side)."""
    fixed = len(json.dumps(specs or [], ensure_ascii=False))
    size = lambda: (fixed + len(json.dumps(messages, ensure_ascii=False))) // 3  # noqa: E731
    tools = [i for i, m in enumerate(messages) if m["role"] == "tool"][: -keep or None]
    for i in tools:
        if size() <= limit:
            break
        messages[i] = {**messages[i], "content": "(older result elided to save context; repeat the call if needed)"}  # fmt: skip
    return messages


async def run(session: Session, branch: Branch, system: str, question: str) -> Branch:
    branch.started_s = round(session.elapsed(), 3)
    try:
        if session.config.budget_feedback:
            return await run_budgeted(session, branch, system, question)
        return await run_turns(session, branch, system, question)
    finally:
        branch.ended_s = round(session.elapsed(), 3)


async def execute(branch: Branch, handlers: dict, name: str, args: dict) -> str:
    """One tool call; errors of the call itself go back to the model as text."""
    if name not in handlers:
        return f"error: unknown tool {name!r}; tools: {', '.join([*handlers, 'done'])}"
    try:
        return await handlers[name](**args)
    except (InfraError, ProviderError):
        raise
    except TypeError as error:
        return f"error: bad arguments for {name}: {error}"
    except Exception as error:  # noqa: BLE001 - the model is told and may retry
        return f"error: {name} failed: {type(error).__name__}: {' '.join(str(error).split())[:300]}"


async def run_budgeted(session: Session, branch: Branch, system: str, question: str) -> Branch:
    """Engine v2: `tool_budget` tool calls per branch. Every tool result ends with the line
    "Tool call budget: m/n". Every call except done counts, also one rejected before running
    (malformed arguments, unknown tool, invalid plan, refused repeat): otherwise a confused
    model could loop without end. Once the budget is used, or the research deadline minus
    max(`answer_reserve_s`, twice the branch's slowest turn so far) is reached (slow branches wrap
    up earlier), a tool call is not run: its result tells the model to answer
    now with done, and from then on only done is offered (answer-now calls: `answer_now`). The
    branch ends with the model's own finding (done or a text reply); only if it keeps calling
    tools for `answer_turns` more turns is one tool-less answer requested."""
    config, tools = session.config, branch.tools
    budget = config.tool_budget
    system += "\n\n" + prompts.BUDGET_SYSTEM.format(budget=budget)
    messages = branch.messages = [{"role": "system", "content": system}, {"role": "user", "content": f"Question: {question}"}]  # fmt: skip
    stopped = ""  # budget | time: why tool calls are no longer run
    for _ in range(budget + config.answer_turns):
        handlers, specs = tools.handlers(), [*tools.specs(), DONE]
        started = session.elapsed()
        if stopped:  # told to answer now: only done is offered, with the answer-now call options
            message = await ask(session, messages, f"branch:{branch.source}:answer", [DONE], **answer_now(config))  # fmt: skip
        else:
            message = await ask(session, messages, f"branch:{branch.source}", specs)
        branch.slowest_turn_s = max(branch.slowest_turn_s, session.elapsed() - started)
        branch.malformed += bool(message.pop("unreadable", False))
        branch.turns += 1
        calls = message.get("tool_calls") or leaked(message.get("content") or "")
        if calls and not message.get("tool_calls"):
            branch.malformed += 1
            message = {**message, "content": "", "tool_calls": calls}
        messages.append(echo(message, config.chat_echo_reasoning))
        if not calls:
            content = (message.get("content") or "").strip()
            if content:
                branch.ending = stopped or "normal"
                return finish(branch, content, cited(content, tools.seen), tools.seen)
            messages.append({"role": "user", "content": prompts.TEXT_REPLY})
            continue
        for call in calls:
            name = (call.get("function") or {}).get("name", "")
            try:
                args = arguments((call.get("function") or {}).get("arguments"))
            except ValueError as error:
                args, parse_error = None, error
            if args is not None and name == "done":
                branch.ending = stopped or "normal"
                return finish(branch, str(args.get("finding", "")), args.get("evidence") or [], tools.seen)  # fmt: skip
            if branch.tool_calls >= budget:
                stopped = stopped or "budget"
                result = prompts.BUDGET_EXHAUSTED
            elif session.left() <= config.reserve_s + margin(config, branch):
                stopped = stopped or "time"
                result = prompts.BUDGET_TIME
            else:
                branch.tool_calls += 1
                if args is None:
                    branch.malformed += 1
                    result = f"error: the arguments of {name} are not a valid JSON object ({parse_error}); call it again"  # fmt: skip
                else:
                    result = await execute(branch, handlers, name, args)
            result += "\n\n" + prompts.BUDGET_LINE.format(used=branch.tool_calls, budget=budget)
            branch.results.append(result)
            messages.append({"role": "tool", "tool_call_id": call.get("id", name), "content": result})  # fmt: skip
    # still calling tools after being told to answer: one answer without tools
    messages.append({"role": "user", "content": prompts.FINISH_NOW})
    message = await ask(
        session, messages, f"branch:{branch.source}:final", None, **answer_now(config)
    )
    branch.turns += 1
    content = (message.get("content") or "").strip()
    if content:
        finish(branch, content, cited(content, tools.seen), tools.seen)
        branch.ending = "forced"
        return branch
    branch.salvage(f"no answer after {budget} tool calls")
    return branch


def answer_now(config) -> dict:
    """Options of an answer-now call (after the budget or time message, the forced finish, the
    conclusion at the cut-off): `answer_options` (the lowest reasoning the model accepts) and a
    fixed `answer_tokens` allowance, so the answer returns within `conclude_s`."""
    return ({"options": config.answer_options} if config.answer_options else {}) | ({"max_tokens": config.answer_tokens} if config.answer_tokens else {})  # fmt: skip


def margin(config, branch: Branch) -> float:
    """Time kept for the answer: `answer_reserve_s`, or twice the branch's slowest turn so far
    (slow branches wrap up earlier), capped so one stalled call cannot close the tool window."""
    return max(config.answer_reserve_s, min(2 * branch.slowest_turn_s, config.answer_reserve_s + config.conclude_s))  # fmt: skip


async def conclude(session: Session, branch: Branch) -> None:
    """A branch cut at the deadline answers from its conversation in one tool-less call of at most
    `conclude_s` (outside the one-call-at-a-time gate: every cut branch answers at once). An
    answer given as a done call counts too. A call that does not return in time is metered as
    cancelled. If no answer comes, the caller's salvage marks the branch unfinished."""
    if not branch.messages or branch.status != "running":
        return
    messages = complete_turns(branch.messages)
    messages.append({"role": "user", "content": prompts.BUDGET_TIME_FINAL})
    trim(messages, session.config.max_prompt_tokens)
    config = session.config
    purpose, options = f"branch:{branch.source}:conclude", answer_now(config)
    tokens = options.pop("max_tokens", 2 * config.report_tokens)
    started = session.elapsed()
    try:
        message, call = await asyncio.wait_for(session.chat.complete(messages, purpose=purpose, max_tokens=tokens, **options), config.conclude_s)  # fmt: skip
    except TimeoutError:
        estimate = bound({"messages": messages}, session.config.chat_price, tokens)
        session.meter.add(Call("chat", "", purpose, 0, 0, 0.0, False, round(session.elapsed() - started, 3), cancelled=True, unmetered_usd=estimate))  # fmt: skip
        branch.error = "no answer at the deadline: the final call timed out"
        return
    except (ModelOutputError, ProviderError, InfraError) as error:
        if isinstance(getattr(error, "call", None), Call):
            session.meter.add(error.call)
        branch.error = f"no answer at the deadline: {type(error).__name__}"
        return
    session.meter.add(call)
    content, evidence = (message.get("content") or "").strip(), None
    for tool in message.get("tool_calls") or leaked(content):
        if (tool.get("function") or {}).get("name") == "done":
            try:
                args = arguments(tool["function"].get("arguments"))
            except ValueError:
                continue
            content, evidence = str(args.get("finding", "")).strip(), args.get("evidence") or []
    content = without_leaks(content)  # a leaked call other than done is no answer
    if content:
        finish(branch, content, evidence if evidence is not None else cited(content, branch.tools.seen), branch.tools.seen)  # fmt: skip
        branch.ending, branch.turns = "time", branch.turns + 1
    else:
        branch.error = "no answer at the deadline: the final call returned no text"


def complete_turns(messages: list) -> list:
    """The conversation without a turn cut mid-way: an assistant message whose tool calls do not
    all have results is dropped with the results it has (the API rejects unanswered calls)."""
    answered = {m.get("tool_call_id") for m in messages if m["role"] == "tool"}
    dropped = {c.get("id") for m in messages if m["role"] == "assistant"
               for c in m.get("tool_calls") or [] if any(x.get("id") not in answered for x in m["tool_calls"])}  # fmt: skip
    return [m for m in messages if not (m["role"] == "assistant" and {c.get("id") for c in m.get("tool_calls") or []} & dropped)
            and not (m["role"] == "tool" and m.get("tool_call_id") in dropped)]  # fmt: skip


async def run_turns(session: Session, branch: Branch, system: str, question: str) -> Branch:
    """Engine v1 (budget feedback off): `branch_turns` model turns, then a cut-off."""
    config, tools = session.config, branch.tools
    messages = [{"role": "system", "content": system}, {"role": "user", "content": f"Question: {question}"}]  # fmt: skip
    for _ in range(config.branch_turns):
        if session.exhausted():
            break
        handlers, specs = tools.handlers(), [*tools.specs(), DONE]
        message = await ask(session, messages, f"branch:{branch.source}", specs)
        branch.malformed += bool(message.pop("unreadable", False))
        branch.turns += 1
        calls = message.get("tool_calls") or leaked(message.get("content") or "")
        if calls and not message.get("tool_calls"):
            branch.malformed += 1
            message = {**message, "content": "", "tool_calls": calls}
        messages.append(echo(message, config.chat_echo_reasoning))
        if not calls:
            content = (message.get("content") or "").strip()
            if content:
                return finish(branch, content, cited(content, tools.seen), tools.seen)
            messages.append({"role": "user", "content": prompts.TEXT_REPLY})
            continue
        for call in calls:
            name = (call.get("function") or {}).get("name", "")
            try:
                args = arguments((call.get("function") or {}).get("arguments"))
            except ValueError as error:
                branch.malformed += 1
                result = f"error: the arguments of {name} are not a valid JSON object ({error}); call it again"  # fmt: skip
            else:
                if name == "done":
                    return finish(branch, str(args.get("finding", "")), args.get("evidence") or [], tools.seen)  # fmt: skip
                if name not in handlers:
                    result = (
                        f"error: unknown tool {name!r}; tools: {', '.join([*handlers, 'done'])}"
                    )
                else:
                    try:
                        result = await handlers[name](**args)
                    except (InfraError, ProviderError):
                        raise
                    except TypeError as error:
                        result = f"error: bad arguments for {name}: {error}"
                    except Exception as error:  # noqa: BLE001 - the model is told and may retry
                        result = f"error: {name} failed: {type(error).__name__}: {' '.join(str(error).split())[:300]}"  # fmt: skip
            branch.results.append(result)
            messages.append({"role": "tool", "tool_call_id": call.get("id", name), "content": result})  # fmt: skip
    reason = session.exhausted() or f"turn budget ({config.branch_turns} turns)"
    if session.left() > config.reserve_s / 2:
        messages.append({"role": "user", "content": prompts.FINISH_NOW})
        message = await ask(session, messages, f"branch:{branch.source}:final", None)
        branch.turns += 1
        content = (message.get("content") or "").strip()
        if content:
            finish(branch, content, cited(content, tools.seen), tools.seen)
            branch.status, branch.error = "partial", reason
            branch.ending = "time" if reason.startswith("time") else "turns"
            return branch
    branch.salvage(reason)
    branch.ending = "cutoff"
    return branch


async def ask(session: Session, messages: list[dict], purpose: str, specs: list | None, **extra) -> dict:  # fmt: skip
    """One chat turn; a prompt the endpoint rejects as too large is shrunk once (every older
    tool result elided) and retried, then the failure is infra. `extra`: call options."""
    trim(messages, session.config.max_prompt_tokens, specs)
    options = ({"tools": specs} if specs else {}) | extra
    try:
        return await session.complete(messages, purpose, **options)
    except ModelOutputError as error:  # the model's reply was unreadable: say so and ask once more
        messages.append({"role": "user", "content": prompts.UNREADABLE})
        try:
            return await session.complete(messages, purpose, **options)
        except ModelOutputError:
            return {"role": "assistant", "content": getattr(error, "raw", ""), "unreadable": True}
    except PromptTooLarge:
        trim(messages, 0, specs, keep=1)
        try:
            return await session.complete(messages, purpose, **options)
        except PromptTooLarge as error:
            raise InfraError(f"prompt too large even after shrinking: {error}") from error


def unwrap(finding: str, evidence) -> tuple[str, list]:
    """A finding the model wrote as a JSON object {"finding": ..., "evidence": [...]}: its parts."""
    text = (
        str(finding).strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    )
    try:
        value = json.loads(text) if text.startswith("{") else None
    except ValueError:
        value = None
    if isinstance(value, dict) and isinstance(value.get("finding"), str):
        extra = value.get("evidence") if isinstance(value.get("evidence"), list) else []
        return value["finding"], [
            *([evidence] if isinstance(evidence, str) else evidence or []),
            *extra,
        ]
    return str(finding), evidence


def finish(branch: Branch, finding: str, evidence, seen: set[str]) -> Branch:
    finding, evidence = unwrap(finding, evidence)
    evidence = [str(e).strip() for e in ([evidence] if isinstance(evidence, str) else evidence) if str(e).strip()]  # fmt: skip
    evidence += [s for s in cited(finding, seen) if s not in evidence]
    branch.evidence = [e for e in evidence if e in seen or re.sub(r"^\./", "", e) in seen]
    branch.unverified = [e for e in evidence if e not in branch.evidence]
    branch.finding, branch.status = finding.strip() or "nothing relevant found", "done"
    return branch
