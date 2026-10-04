"""Claude Code CLI inside the trial container. Auth: the host's ~/.claude/.credentials.json
(where the Linux CLI keeps its OAuth login), read-only; ANTHROPIC_API_KEY is passed if set."""

from __future__ import annotations

import json
import os
from collections import Counter

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

AUTH = container.CLAUDE_AUTH  # also the research reader's login (container.reader_auth)
NETWORK_TOOLS = {"WebSearch", "WebFetch"}


def command(trial):
    argv = [
        "claude", "-p", "--output-format", "stream-json", "--verbose",
        "--dangerously-skip-permissions", "--model", trial.model, "--effort", trial.effort,
        "--disallowedTools", ",".join(sorted(NETWORK_TOOLS)), "--strict-mcp-config",
        "--json-schema", json.dumps(trial.answer_schema),
    ]  # fmt: skip
    if trial.mcp:
        program, args = container.mcp_command(trial.mcp)
        server = {"command": program, "args": args, "env": container.mcp_env(trial.mcp)}
        argv += ["--mcp-config", json.dumps({"mcpServers": {trial.mcp.server: server}})]
    return argv


def turn_counter():
    """One turn = one model response. Claude Code streams one `assistant` event per content
    BLOCK (thinking, text, each tool_use), all carrying the response's message id, so turns are
    distinct message ids, not events."""
    seen = set()

    def is_turn(event) -> bool:
        if event.get("type") != "assistant":
            return False
        key = (event.get("message") or {}).get("id") or len(seen)
        fresh = key not in seen
        seen.add(key)
        return fresh

    return is_turn


def digest(events):
    """(turns, errors, usage, tool_calls, context_paths, context_output_bytes, violations,
    answer, cli_calls). Usage is the result event's cumulative total, mapped onto the shared three keys."""
    tools, paths, read_bytes, turns, errors = Counter(), set(), 0, 0, []
    counted = turn_counter()
    watched, result, shell = set(), None, []
    for event in events:
        content = (event.get("message") or {}).get("content")
        content = content if isinstance(content, list) else []
        if event.get("type") == "assistant":
            turns += counted(event)  # distinct model responses, not content blocks
            if event.get("error"):
                errors.append(str(event["error"]))
            for block in content:
                if block.get("type") == "text" and block.get("text", "").startswith("API Error"):
                    errors.append(block["text"])
                if block.get("type") != "tool_use":
                    continue
                tools[block.get("name", "?")] += 1
                if block.get("name") == "Bash":
                    shell.append((block.get("input") or {}).get("command"))
                found = CONTEXT_PATH.findall(json.dumps(block.get("input") or {}))
                paths.update(found)
                if found:
                    watched.add(block.get("id"))
        elif event.get("type") == "user":
            for block in content:
                if block.get("type") == "tool_result" and block.get("tool_use_id") in watched:
                    read_bytes += len(json.dumps(block.get("content")).encode())
        elif event.get("type") == "result":
            result = event
    answer = None
    usage = {}
    if result:
        raw = result.get("usage") or {}
        cached = raw.get("cache_read_input_tokens", 0)
        usage = {
            "input_tokens": raw.get("input_tokens", 0)
            + cached
            + raw.get("cache_creation_input_tokens", 0),
            "cached_input_tokens": cached,
            "output_tokens": raw.get("output_tokens", 0),
            "cost_usd": result.get("total_cost_usd"),
        }
        structured = result.get("structured_output")
        answer = structured if isinstance(structured, dict) else parse_answer(result.get("result"))
        if result.get("is_error"):
            errors.append(
                f"{result.get('subtype')} {result.get('terminal_reason')} "
                f"{result.get('api_error_status')}: {str(result.get('result'))[:1000]}"
            )
    violations = [f"{name} call in trace" for name in sorted(NETWORK_TOOLS & set(tools))]
    return (
        turns, errors, usage, dict(tools), sorted(paths), read_bytes, violations, answer,
        cli_calls(shell),
    )  # fmt: skip


class ClaudeCodeAgent:
    name = "claude-code"

    def version(self):
        done = container.docker("run", "--rm", "--network", "none", container.IMAGE, "claude", "--version")  # fmt: skip
        return done.stdout.strip()

    def run(self, trial):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key and not AUTH.exists():
            return AgentResult("infra_error", detail=f"{AUTH} missing: run `claude` and log in")
        auth = [] if key else [(AUTH, container.CLAUDE_AUTH_IN)]
        home = container.prepare_home(trial, auth)
        name = f"harnext-e2-{os.getpid()}-{abs(hash((trial.task_id, trial.arm))) % 10**8}"
        argv = container.run_argv(
            trial, name, command(trial), auth=auth, home=home,
            passthrough=["ANTHROPIC_API_KEY"] if key else [],
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
        turns, errors, usage, tools, paths, read_bytes, violations, answer, clis = digest(
            events_of(trace)
        )
        err = stderr.read_text(errors="replace")
        if code in {125, 126, 127}:
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
