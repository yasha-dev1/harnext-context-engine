"""Offline Codex protocol fixtures; never invoke a model in pytest."""

import pytest
from harnext_builder.harness.base import ConversationTranscript, HarnessRequest
from harnext_builder.harness.codex import CodexHarness, command, decode_events
from harnext_builder.harness.registry import get_harness


def test_pinned_command_and_reader_native_tools_disabled():
    argv = command("gpt-5.6-luna", "medium", "/tmp/store", tools_enabled=False)
    assert argv[argv.index("--model") + 1] == "gpt-5.6-luna"
    assert 'model_reasoning_effort="medium"' in argv
    assert "features.shell_tool=false" in argv
    assert "--ignore-user-config" in argv
    assert "--ephemeral" in argv
    assert isinstance(get_harness("codex"), CodexHarness)


@pytest.mark.parametrize("reason", ["error", "timeout", "max_turns", "unknown"])
def test_incomplete_build_is_not_committable(reason):
    assert not ConversationTranscript(harness="codex", stop_reason=reason).ok


def test_success_requires_complete_turn_and_zero_exit():
    events = [{"type": "item.completed", "item": {"type": "agent_message", "text": "done"}},
              {"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 80,
                                                    "output_tokens": 10}}]
    result = decode_events(events, model="gpt-5.6-luna", effort="medium", returncode=0)
    assert result.ok
    assert result.usage["input_tokens"] == 100  # cache hits are already included
    assert result.usage["total_cost_usd"] is None
    assert not decode_events(events[:-1], model="m", effort="medium", returncode=0).ok
    assert not decode_events(events, model="m", effort="medium", returncode=1).ok
    assert not decode_events(events + [{"type": "turn.failed", "error": "quota"}],
                             model="m", effort="medium", returncode=0).ok
    item_error = {"type": "item.completed", "item": {"type": "error", "message": "bridge disabled"}}
    assert not decode_events([item_error, *events], model="m", effort="medium", returncode=0).ok


@pytest.mark.asyncio
async def test_process_success_and_deadline_without_network(monkeypatch, tmp_path):
    import sys

    from harnext_builder.harness import codex

    program = ("import sys; sys.stdin.read(); "
               "print('{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":3}}')")
    monkeypatch.setattr(codex, "command", lambda *a, **k: [sys.executable, "-c", program])
    result = await codex.execute(model="fixture", effort="medium", cwd=str(tmp_path),
                                 prompt="test", tools_enabled=False, timeout_s=2)
    assert result.ok and result.usage["input_tokens"] == 3
    program = "import time; time.sleep(30)"
    result = await codex.execute(model="fixture", effort="medium", cwd=str(tmp_path),
                                 prompt="test", tools_enabled=False, timeout_s=1)
    assert not result.ok and result.stop_reason == "timeout"


@pytest.mark.asyncio
async def test_missing_effort_or_unsupported_seed_fails_without_network():
    request = HarnessRequest(harness="codex", model="gpt-5.6-luna", working_dir=".",
                             instruction="test", system_prompt="test")
    assert not (await CodexHarness().run(request)).ok
    request.reasoning_effort = "medium"
    request.seed = 42
    assert "sampling-seed" in (await CodexHarness().run(request)).error
