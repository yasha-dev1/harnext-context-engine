"""Codex CLI inside the trial container. Auth: the host's ~/.codex/auth.json, read-only."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from harnext_eval.e2.benchmark import save
from harnext_eval.e2.v2.agents import container
from harnext_eval.e2.v2.agents.base import (
    CONTEXT_PATH,
    AgentResult,
    classify,
    cli_calls,
    events_of,
    parse_answer,
)

AUTH = Path.home() / ".codex/auth.json"
MODEL_ITEMS = {
    "agent_message", "reasoning", "command_execution", "mcp_tool_call", "file_change",
    "todo_list", "web_search",
}  # fmt: skip
ACTIONS = {"agent_message", "command_execution", "mcp_tool_call", "file_change", "web_search"}
TOOLS = {"command_execution", "mcp_tool_call", "file_change", "web_search"}


def toml(value):
    return json.dumps(value)  # JSON scalars, strings and arrays are valid TOML


def command(trial):
    overrides = {
        "model_reasoning_effort": trial.effort,
        "web_search": "disabled",  # server-side tool: the proxy cannot see it, so it is off
        "features.multi_agent": False,
        "features.apps": False,
        "features.plugins": False,
        "features.browser_use": False,
        "features.in_app_browser": False,
        "features.image_generation": False,
    }
    if trial.mcp:
        program, args = container.mcp_command(trial.mcp)
        prefix = f"mcp_servers.{trial.mcp.server}."
        overrides |= {
            prefix + "command": program,
            prefix + "args": args,
            prefix + "env": container.mcp_env(trial.mcp),
            prefix + "required": True,
            prefix + "startup_timeout_sec": 120,
            prefix + "tool_timeout_sec": container.tool_timeout_ms(trial.mcp.kind, trial.mcp.engine) // 1000
            if trial.mcp.kind in ("research", "research_search", "engine")
            else 120,
            prefix + "default_tools_approval_mode": "approve",
        }
        if trial.mcp.tools:
            overrides[prefix + "enabled_tools"] = list(trial.mcp.tools)
        if trial.mcp.kind in ("research", "research_search") and os.environ.get("ANTHROPIC_API_KEY"):
            overrides[prefix + "env_vars"] = ["ANTHROPIC_API_KEY"]  # by name: never in argv
    argv = [
        "codex", "exec", "--json", "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check", "-C", "/workspace", "-m", trial.model,
    ]  # fmt: skip
    for key, value in overrides.items():
        if isinstance(value, dict):
            value = "{" + ", ".join(f"{k} = {toml(v)}" for k, v in value.items()) + "}"
            argv += ["-c", f"{key}={value}"]
        else:
            argv += ["-c", f"{key}={toml(value)}"]
    return [
        *argv, "--output-schema", "/home/agent/answer-schema.json",
        "--output-last-message", "/home/agent/answer.json", "-",
    ]  # fmt: skip


def digest(events):
    """(turns, errors, usage, tool_calls, context_paths, context_output_bytes, violations,
    cli_calls)."""
    items = [e["item"] for e in events if e.get("type") == "item.completed" and "item" in e]
    errors = [
        (e.get("error") or {}).get("message") or e.get("message") or json.dumps(e)
        for e in events
        if e.get("type") in {"turn.failed", "error"}
    ]
    usage = Counter()
    for event in events:
        if event.get("type") == "turn.completed":
            usage.update({k: v for k, v in (event.get("usage") or {}).items() if type(v) is int})
    tools = Counter(i["type"] for i in items if i.get("type") in TOOLS)
    paths, read_bytes = set(), 0
    for item in items:
        if item.get("type") == "command_execution":
            found = CONTEXT_PATH.findall(item.get("command") or "")
            paths.update(found)
            if found:
                read_bytes += len((item.get("aggregated_output") or "").encode())
    return (
        sum(i.get("type") in ACTIONS for i in items),  # turns = model responses, no reasoning
        errors,
        {k: usage.get(k, 0) for k in ("input_tokens", "cached_input_tokens", "output_tokens")}
        | {k: v for k, v in usage.items() if k.startswith("reasoning")},
        dict(tools),
        sorted(paths),
        read_bytes,
        ["web_search item in trace"] if tools.get("web_search") else [],
        cli_calls(i.get("command") for i in items if i.get("type") == "command_execution"),
    )


def turn_counter():
    return is_turn


def is_turn(event) -> bool:
    """One model response: a message or a tool action. Codex's `reasoning` and `todo_list`
    items are the model's private bookkeeping and do not count."""
    return (
        event.get("type") == "item.completed" and (event.get("item") or {}).get("type") in ACTIONS
    )


class CodexAgent:
    name = "codex"

    def version(self):
        done = container.docker("run", "--rm", "--network", "none", container.IMAGE, "codex", "--version")  # fmt: skip
        return done.stdout.strip()

    def run(self, trial):
        if not AUTH.exists():
            return AgentResult("infra_error", detail=f"{AUTH} missing: run `codex login`")
        auth = [(AUTH, "/home/agent/.codex/auth.json"), *container.reader_auth(trial)]
        home = container.prepare_home(trial, auth)
        save(home / "answer-schema.json", trial.answer_schema)
        name = f"harnext-e2-{os.getpid()}-{abs(hash((trial.task_id, trial.arm))) % 10**8}"
        key = os.environ.get("ANTHROPIC_API_KEY") and len(auth) == 1  # the reader's key, by name
        argv = container.run_argv(
            trial, name, command(trial), auth=auth, home=home,
            passthrough=["ANTHROPIC_API_KEY"] if key and trial.mcp else [],
        )  # fmt: skip
        trace, stderr = trial.out / "trace.jsonl", trial.out / "agent.stderr"
        save(
            trial.out / "launch.json", {"argv": argv, "model": trial.model, "effort": trial.effort,
             "image_id": container.image_ref()}
        )
        code, seconds, timed_out = container.execute(
            argv,
            name,
            stdin=trial.prompt,
            stdout=trace,
            stderr=stderr,
            timeout=trial.timeout,
            **({"turn_limit": (trial.max_turns, turn_counter())} if trial.max_turns else {}),
        )
        turns, errors, usage, tools, paths, read_bytes, violations, clis = digest(events_of(trace))
        last = home / "answer.json"
        answer = parse_answer(last.read_text(errors="replace")) if last.exists() else None
        err = stderr.read_text(errors="replace")
        if code in {125, 126, 127}:  # docker itself failed, the CLI never ran
            status, detail = "infra_error", f"docker exit {code}: {err[-500:]}"
        else:
            status, detail = classify(
                returncode=code, timed_out=timed_out, turns=turns, errors=errors, stderr=err,
                answer=answer, needs_answer=trial.kind == "historical",
            )  # fmt: skip
        if timed_out == "turns" and status == "timeout":
            status, detail = "turn_limit", f"turn limit reached ({trial.max_turns})"
        return AgentResult(
            status, answer, usage, tools, paths, read_bytes, trace, seconds, detail, turns,
            violations, clis,
        )  # fmt: skip
