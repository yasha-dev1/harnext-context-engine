"""Deterministic agent for tests and dry plumbing runs: answers from a lookup, applies a
supplied patch on coding tasks, and can simulate an infra error or a timeout at the k-th call."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading

from harnext_eval.e2.v2 import grade
from harnext_eval.e2.v2.agents.base import AgentResult


class FakeAgent:
    name = "fake"
    needs_container = False

    def __init__(
        self, answers=None, patches=None, *, infra_at=None, timeout_at=None, reader_infra_at=None,
        engine_modes=None,
    ):  # fmt: skip
        """answers/patches: {"<task>|<arm>" or "<task>": value}; *_at: 1-based call index
        (reader_infra_at: the research reader refuses on that call's context_research).
        engine_modes: {call index or "<task>|<arm>": bug | slow | flaky | flaky503 | dead | failed | timeout}
        for engine arms (see `engine`; slow = every attempt degraded, flaky = the first one only;
        "<mode>*n" = that mode on the trial's first n runs only)."""
        self.answers, self.patches = answers or {}, patches or {}
        # keys: a call index (that call only) or "<task>|<arm>" (every call of that trial)
        self.engine_modes = {(int(k) if str(k).isdigit() else k): v for k, v in (engine_modes or {}).items()}
        self.runs = {}  # engine trial -> times run (modes like "slow*1")
        self.infra_at, self.timeout_at, self.reader_infra_at = infra_at, timeout_at, reader_infra_at
        self.calls, self.seen, self.lock = 0, [], threading.Lock()

    def version(self):
        return "fake-1"

    def pick(self, table, trial):
        return table.get(f"{trial.task_id}|{trial.arm}", table.get(trial.task_id))

    def run(self, trial):
        with self.lock:
            self.calls += 1
            call = self.calls
            self.seen.append((trial.task_id, trial.arm))
        trace = trial.out / "trace.jsonl"
        if call == self.infra_at:
            trace.write_text(json.dumps({"type": "error", "message": "usage limit reached"}) + "\n")
            return AgentResult("infra_error", trace=trace, detail="You've hit your usage limit")
        events = [{"type": "fake.turn", "task": trial.task_id, "arm": trial.arm}]
        trace.write_text("".join(json.dumps(e) + "\n" for e in events))
        usage = {
            "input_tokens": len(trial.prompt),
            "cached_input_tokens": 0,
            "output_tokens": 10,
        }
        if call == self.timeout_at:
            return AgentResult(
                "timeout", None, usage, {}, [], 0, trace, float(trial.timeout), "wall-clock", 1
            )
        patch = self.pick(self.patches, trial)
        if trial.kind == "coding" and patch:
            if grade.apply(patch, trial.workspace).returncode:
                raise RuntimeError("fake agent patch did not apply")
        value = self.pick(self.answers, trial)
        answer = (
            {"summary": "patched"} if trial.kind == "coding" else {"value": value, "evidence": []}
        )
        paths = ["/context/INDEX.md"] if trial.context else []
        tools = {"command_execution": len(paths)}
        if trial.mcp and trial.mcp.kind in ("research", "research_search") and trial.mcp.reader == "fake":
            self.research(trial, call)  # in-process, so receipts (and infra rows) are real
            tools["mcp_tool_call"] = 1
        if trial.mcp and trial.mcp.kind == "engine" and trial.mcp.reader == "fake":
            events = self.engine(trial, call)  # the real stdio server: one export per process (chdb)
            with trace.open("a") as stream:  # Claude Code's shape: init status, tool_use, tool_result
                stream.write("".join(json.dumps(e) + "\n" for e in events))
            tools["mcp_tool_call"] = 1
        return AgentResult(
            "completed", answer, usage, tools, paths, 64 * len(paths), trace, 1.0, None, 1
        )

    @staticmethod
    def question(trial, infra):
        return ("!infra " if infra else "") + trial.prompt.strip().splitlines()[0][:200]

    def research(self, trial, call):
        """One `context_research` call through the real Session/backend with the fake reader,
        writing the trial's receipts exactly as the containerised server would."""
        from harnext_eval.e2.v2 import mcp_server

        backend = mcp_server.ResearchBackend(trial.mcp.store, reader="fake")
        session = mcp_server.Session(
            "research", backend, trial.out / "mcp/receipts.jsonl",
            max_calls=mcp_server.DEFAULT_CALLS["research"],
        )  # fmt: skip
        try:
            question = self.question(trial, call == self.reader_infra_at)
            asyncio.run(session.invoke("context_research", {"question": question}))
        finally:
            session.close()

    def engine(self, trial, call):
        """One `context_research` call to the real engine MCP server (subprocess, stdio, fake
        models), which writes the trial's receipts and engine trace as in a container. Returns the
        Claude Code trace events of the call; `engine_modes` simulates failures: bug / slow (the
        engine raises / runs out of time after retried calls), dead (the server never receives
        the call), failed (the server is not connected at start), timeout (the agent's MCP client
        gives up on the call)."""
        from fastmcp import Client
        from fastmcp.client.transports import StdioTransport

        mode = self.engine_modes.get(call) or self.engine_modes.get(f"{trial.task_id}|{trial.arm}")
        if mode and "*" in mode:  # "slow*1": this mode for the trial's first n runs only
            mode, n = mode.split("*")
            with self.lock:
                self.runs[(trial.task_id, trial.arm)] = runs = self.runs.get((trial.task_id, trial.arm), 0) + 1
            mode = mode if runs <= int(n) else None
        server = trial.mcp.server
        init = {"type": "system", "subtype": "init",
                "mcp_servers": [{"name": server, "status": "failed" if mode == "failed" else "connected"}]}  # fmt: skip
        use = {"type": "tool_use", "id": f"toolu_fake{call}", "name": f"mcp__{server}__context_research"}
        text = "MCP error -32000: Connection closed"
        mcp = trial.out / "mcp"  # the server starts (writes its start receipt) in every mode
        mcp.mkdir(parents=True, exist_ok=True)
        args = [
            "-m", "harnext_eval.e2.v2.mcp_server", "--kind", "engine", "--store", str(trial.mcp.store),
            "--log", str(mcp / "receipts.jsonl"), "--clients", "fake", "--scratch", str(trial.out),
            *(["--memory", str(trial.mcp.memory)] if trial.mcp.memory else []),
        ]  # fmt: skip
        prefix = "!infra " if call == self.reader_infra_at else {"bug": "!bug ", "slow": "!slow ", "flaky": "!flaky ", "flaky503": "!flaky503 "}.get(mode, "")  # fmt: skip

        async def ask():
            env = {k: os.environ[k] for k in ("PYTHONPATH", "HOME", "PATH") if k in os.environ}  # the same code as ours
            async with Client(StdioTransport(sys.executable, args, env=env, keep_alive=False)) as client:
                if mode in ("dead", "failed"):  # the call never reaches the server
                    await client.list_tools()
                    return text
                question = prefix + self.question(trial, False)
                out = await client.call_tool("context_research", {"question": question}, raise_on_error=False)  # fmt: skip
                return out.content[0].text  # type: ignore[union-attr]

        text = asyncio.run(ask())
        if mode == "failed":
            return [init]
        if mode == "timeout":
            text = "MCP error -32001: Request timed out"
        result = {"type": "tool_result", "tool_use_id": use["id"], "content": text,
                  "is_error": mode in ("dead", "timeout")}  # fmt: skip
        return [init, {"type": "assistant", "message": {"id": f"msg_fake{call}", "content": [use]}},
                {"type": "user", "message": {"content": [result]}}]  # fmt: skip
