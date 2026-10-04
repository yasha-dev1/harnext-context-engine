import json
from typing import Any

import pytest
from harnext_eval.e2.v2.agents import claude_code, codex, container
from harnext_eval.e2.v2.agents.base import McpSpec, TrialSpec, classify, parse_answer
from harnext_eval.e2.v2.errors import PauseRun, is_infra


def spec(tmp_path, **extra):
    (tmp_path / "ws").mkdir(exist_ok=True)
    (tmp_path / "out").mkdir(exist_ok=True)
    base: dict[str, Any] = dict(task_id="H-1", arm="F1", kind="historical", prompt="Q?", answer_schema={"type": "object"},
                workspace=tmp_path / "ws", out=tmp_path / "out", model="gpt-x", effort="medium", timeout=5)  # fmt: skip
    return TrialSpec(**(base | extra))


REAL_MESSAGES = [
    "You've hit your usage limit. Upgrade to Pro or try again in 3 hours.",
    "Credit balance is too low",
    '{"type":"error","error":{"type":"rate_limit_error","message":"..."}}',
    '{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}',
    "API Error: 529 Overloaded",
    "exceeded retry limit, last status: 429 Too Many Requests",
    "stream disconnected before completion",
    "Your access token could not be refreshed. Please log out and sign in again.",
    "Invalid API key · Please run /login",
    "unexpected status 401 Unauthorized",
]


@pytest.mark.parametrize("message", REAL_MESSAGES)
def test_provider_messages_are_infrastructure(message):
    assert is_infra(message)
    status, _ = classify(returncode=1, timed_out=False, turns=3, errors=[message], stderr="", answer=None, needs_answer=True)  # fmt: skip
    assert status == "infra_error"


def run(**kw):
    base: dict[str, Any] = dict(returncode=0, timed_out=False, turns=2, errors=[], stderr="", answer={"value": 1}, needs_answer=True)  # fmt: skip
    return classify(**(base | kw))[0]


def test_classification_table():
    assert run() == "completed"
    assert run(answer=None) == "model_failure"
    assert run(answer=None, needs_answer=False) == "completed"
    assert run(timed_out=True, returncode=None) == "timeout"
    assert run(timed_out=True, returncode=None, turns=0) == "infra_error"  # the model never spoke
    assert run(returncode=1, turns=0, stderr="panicked at main.rs") == "infra_error"  # CLI crash
    assert run(returncode=1, errors=["something nobody has seen before"]) == "infra_error"
    assert (
        run(returncode=1, errors=["Your input exceeds the context window of this model"])
        == "model_failure"
    )
    assert run(errors=["error_max_turns: reached"]) == "model_failure"


def test_parse_answer():
    assert parse_answer('{"value": null, "evidence": []}') == {"value": None, "evidence": []}
    assert parse_answer('Here:\n```json\n{"value": "x"}\n```') == {"value": "x"}
    assert (
        parse_answer("no json") is None
        and parse_answer("[1]") is None
        and parse_answer(None) is None
    )


def test_codex_command_and_container_boundary(tmp_path):
    trial = spec(tmp_path, context=tmp_path / "ctx")
    command = codex.command(trial)
    assert command[:3] == ["codex", "exec", "--json"]
    assert (
        "--dangerously-bypass-approvals-and-sandbox" in command
        and "--skip-git-repo-check" in command
    )
    assert 'model_reasoning_effort="medium"' in command and 'web_search="disabled"' in command
    assert (
        command[-1] == "-" and "--output-schema" in command and "--output-last-message" in command
    )
    auth = [(tmp_path / "auth.json", "/home/agent/.codex/auth.json")]
    home = container.prepare_home(trial, auth)
    assert (
        home / ".codex/auth.json"
    ).exists()  # a mountpoint we own, not one Docker creates as root
    argv = container.run_argv(trial, "n", command, auth=auth, home=home)
    joined = " ".join(argv)
    assert (
        "--network harnext-e2-net" in joined and "--memory 10g --cpus 4 --pids-limit 4096" in joined
    )
    assert (
        f"{tmp_path / 'ctx'}:/context:ro" in argv
        and f"{tmp_path / 'auth.json'}:/home/agent/.codex/auth.json:ro" in argv
    )
    assert "HTTPS_PROXY=http://harnext-e2-proxy:8888" in argv and "/store" not in joined
    assert not any("kafka.git" in a or "benchmarks" in a or "private" in a for a in argv)


def test_mcp_wiring_for_both_clis(tmp_path):
    trial = spec(tmp_path, mcp=McpSpec("rag", tmp_path / "store", "history", ("search", "read")))
    command = codex.command(trial)
    assert 'mcp_servers.history.command="/opt/venv/bin/python"' in command
    args = next(c for c in command if c.startswith("mcp_servers.history.args="))
    assert json.loads(args.split("=", 1)[1])[:4] == [
        "-m",
        "harnext_eval.e2.v2.mcp_server",
        "--kind",
        "rag",
    ]
    assert 'mcp_servers.history.enabled_tools=["search", "read"]' in command
    claude = claude_code.command(trial)
    servers = json.loads(claude[claude.index("--mcp-config") + 1])["mcpServers"]
    assert servers["history"]["args"][-4:] == [
        "--store",
        "/store",
        "--log",
        "/mcp-log/receipts.jsonl",
    ]
    assert "--dangerously-skip-permissions" in claude and "--strict-mcp-config" in claude
    assert claude[claude.index("--disallowedTools") + 1] == "WebFetch,WebSearch"
    argv = container.run_argv(trial, "n", claude, auth=[], home=tmp_path / "out/home", passthrough=["ANTHROPIC_API_KEY"])  # fmt: skip
    assert f"{tmp_path / 'store'}:/store:ro" in argv and "/context" not in " ".join(argv)
    assert argv[argv.index("ANTHROPIC_API_KEY") - 1] == "-e"  # by name: the value never enters argv


def test_codex_digest_extracts_usage_tools_and_context_paths():
    events = [
        {"type": "thread.started"},
        {"type": "item.completed", "item": {"type": "reasoning", "text": "…"}},
        {"type": "item.completed", "item": {"type": "command_execution", "command": "/bin/bash -lc \"rg -n KAFKA-1 /context/entities | head; cat /context/INDEX.md\"", "aggregated_output": "x" * 100, "exit_code": 0}},
        {"type": "item.completed", "item": {"type": "command_execution", "command": "ls /workspace", "aggregated_output": "y" * 50}},
        {"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "history", "tool": "search"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "{}"}},
        {"type": "turn.completed", "usage": {"input_tokens": 1000, "cached_input_tokens": 600, "output_tokens": 40, "reasoning_output_tokens": 7}},
    ]  # fmt: skip
    turns, errors, usage, tools, paths, read_bytes, violations, _ = codex.digest(events)
    assert turns == 4 and not errors and not violations  # the reasoning item is not a turn
    assert usage == {
        "input_tokens": 1000,
        "cached_input_tokens": 600,
        "output_tokens": 40,
        "reasoning_output_tokens": 7,
    }
    assert tools == {"command_execution": 2, "mcp_tool_call": 1}
    assert paths == ["/context/INDEX.md", "/context/entities"] and read_bytes == 100
    failed = codex.digest(
        [{"type": "turn.failed", "error": {"message": "You've hit your usage limit."}}]
    )
    assert failed[0] == 0 and failed[1] == ["You've hit your usage limit."]
    assert codex.digest([{"type": "item.completed", "item": {"type": "web_search"}}])[6]


def test_claude_digest_extracts_usage_tools_answer_and_api_errors():
    events = [
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/context/topics/a.md"}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "abc"}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}},
        {"type": "result", "subtype": "success", "is_error": False, "result": "{\"value\": \"x\", \"evidence\": []}",
         "usage": {"input_tokens": 10, "cache_read_input_tokens": 90, "cache_creation_input_tokens": 5, "output_tokens": 3}, "total_cost_usd": 0.01},
    ]  # fmt: skip
    turns, errors, usage, tools, paths, read_bytes, violations, answer, _ = claude_code.digest(
        events
    )
    assert turns == 2 and not errors and tools == {"Read": 1} and paths == ["/context/topics/a.md"]
    assert (
        usage["input_tokens"] == 105
        and usage["cached_input_tokens"] == 90
        and usage["output_tokens"] == 3
    )
    assert answer == {"value": "x", "evidence": []} and read_bytes == 5 and not violations
    broke = [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "API Error: Credit balance is too low"}]}},
        {"type": "result", "subtype": "success", "is_error": True, "result": "Credit balance is too low"},
    ]  # fmt: skip
    errors = claude_code.digest(broke)[1]
    assert classify(returncode=1, timed_out=False, turns=1, errors=errors, stderr="", answer=None, needs_answer=True)[0] == "infra_error"  # fmt: skip


def test_adapter_end_to_end_with_mocked_docker(tmp_path, monkeypatch):
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    monkeypatch.setattr(codex, "AUTH", auth)
    trial = spec(tmp_path)

    def execute(argv, name, *, stdin, stdout, stderr, timeout):
        assert stdin == "Q?" and timeout == 5
        stdout.write_text(
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "{}"}}) + "\n"
            + json.dumps({"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 1}}) + "\n"
        )  # fmt: skip
        stderr.write_text("")
        (trial.out / "home/answer.json").write_text('{"value": null, "evidence": []}')
        return 0, 1.5, False

    monkeypatch.setattr(container, "execute", execute)
    result = codex.CodexAgent().run(trial)
    assert result.status == "completed" and result.answer == {"value": None, "evidence": []}
    assert result.usage["input_tokens"] == 5 and result.elapsed_s == 1.5
    launch = json.loads((trial.out / "launch.json").read_text())
    assert launch["model"] == "gpt-x" and "docker" == launch["argv"][0]

    def dead(argv, name, *, stdin, stdout, stderr, timeout):
        stdout.write_text("")
        stderr.write_text("docker: Error response from daemon: network harnext-e2-net not found")
        return 125, 0.2, False

    monkeypatch.setattr(container, "execute", dead)
    assert codex.CodexAgent().run(trial).status == "infra_error"
    monkeypatch.setattr(codex, "AUTH", tmp_path / "absent.json")
    assert codex.CodexAgent().run(trial).status == "infra_error"


def test_research_kind_wiring(tmp_path, monkeypatch):
    """Every agent's container gets the reader's login, a long MCP tool timeout and the server's
    --reader flag when the arm is served through context_research."""
    creds = tmp_path / "creds.json"
    creds.write_text("{}")
    monkeypatch.setattr(container, "CLAUDE_AUTH", creds)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    trial = spec(
        tmp_path, mcp=McpSpec("research", tmp_path / "store", "history", ("context_research",))
    )
    assert container.reader_auth(trial) == [(creds, "/home/agent/.claude/.credentials.json")]
    assert container.reader_auth(spec(tmp_path, mcp=McpSpec("rag", tmp_path / "store"))) == []
    assert (
        container.reader_auth(
            spec(tmp_path, mcp=McpSpec("research", tmp_path / "store", reader="fake"))
        )
        == []
    )
    program, args = container.mcp_command(trial.mcp)
    assert args[-2:] == ["--reader", "claude"] and "--kind" in args and "research" in args
    command = codex.command(trial)
    assert "mcp_servers.history.tool_timeout_sec=240" in command
    env = next(c for c in command if c.startswith("mcp_servers.history.env="))
    assert 'HTTPS_PROXY = "http://harnext-e2-proxy:8888"' in env and 'HOME = "/home/agent"' in env
    assert container.mcp_env(McpSpec("rag", tmp_path)) == {"PYTHONPATH": "/opt/harnext/src"}
    argv = container.run_argv(
        trial, "n", command, auth=container.reader_auth(trial), home=tmp_path / "out/home"
    )
    assert f"{creds}:/home/agent/.claude/.credentials.json:ro" in argv
    assert "MCP_TOOL_TIMEOUT=240000" in argv
    creds.unlink()
    with pytest.raises(PauseRun, match="Claude Code login"):
        container.reader_auth(trial)


def test_standing_instructions_reach_both_cli_instruction_files(tmp_path):
    from harnext_eval.e2.v2 import runner
    from harnext_eval.e2.v2.agents import container
    from harnext_eval.e2.v2.agents.base import TrialSpec

    server = {"name": "history"}
    research = runner.standing_instructions({"access": "mcp", "mcp_kind": "research"}, server)
    assert "DO NOT START UNTIL YOU GATHER CONTEXT FROM context_research mcp tool" in research
    assert "`history`" in research
    assert "`search`" in runner.standing_instructions({"access": "mcp", "mcp_kind": "rag"}, server)
    assert "`jira`" in runner.standing_instructions({"access": "sources"}, server)
    assert runner.standing_instructions({"access": "none"}, server) is None  # closed book
    trial = TrialSpec(
        task_id="t", arm="F5", kind="historical", prompt="p", answer_schema={},
        workspace=tmp_path / "w", out=tmp_path / "out", model="m", effort="medium", timeout=1,
        instructions=research,
    )  # fmt: skip
    home = container.prepare_home(trial, [])
    assert (home / ".claude/CLAUDE.md").read_text() == research
    assert (home / ".codex/AGENTS.md").read_text() == research
    assert (tmp_path / "out/instructions.md").read_text() == research
    from dataclasses import replace

    plain = replace(trial, instructions=None, out=tmp_path / "plain")
    assert not (container.prepare_home(plain, []) / ".claude/CLAUDE.md").exists()
