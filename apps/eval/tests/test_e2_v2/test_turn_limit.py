"""The agent turn cap: enforced from the live trace, recorded as a model result."""

import json
import sys
import time

from harnext_eval.e2.v2.agents import claude_code, codex, container


def emitter(tmp_path, lines, pause=0.05):
    """A stand-in for the agent CLI: prints one trace event per line, slowly."""
    script = tmp_path / "emit.py"
    script.write_text(
        "import sys,time\nsys.stdin.read()\n"
        f"for line in {lines!r}:\n    print(line, flush=True); time.sleep({pause})\n"
    )
    return [sys.executable, str(script)]


def test_execute_stops_after_the_cap_and_not_before(tmp_path):
    turns = [json.dumps({"type": "assistant", "message": {"id": f"m{i}"}}) for i in range(40)]
    other = json.dumps({"type": "user"})
    argv = emitter(tmp_path, [x for t in turns for x in (t, t, other)])
    out, err = tmp_path / "trace.jsonl", tmp_path / "err"
    started = time.monotonic()
    code, _, stopped = container.execute(
        argv, "no-such-container", stdin="go", stdout=out, stderr=err, timeout=60,
        turn_limit=(5, claude_code.turn_counter()),
    )  # fmt: skip
    assert stopped == "turns" and code is None and time.monotonic() - started < 30
    code, _, stopped = container.execute(
        emitter(tmp_path, turns[:3] * 4, pause=0), "x", stdin="go", stdout=out, stderr=err, timeout=60,
        turn_limit=(5, claude_code.turn_counter()),
    )  # fmt: skip
    assert stopped is False and code == 0


def test_turn_definitions_ignore_private_reasoning():
    count = claude_code.turn_counter()
    same = {"type": "assistant", "message": {"id": "m1"}}
    assert count(same) and not count(same) and not count({"type": "user"})
    assert count({"type": "assistant", "message": {"id": "m2"}})
    done = {"type": "item.completed", "item": {"type": "command_execution"}}
    assert codex.is_turn(done) and codex.is_turn({**done, "item": {"type": "agent_message"}})
    assert not codex.is_turn({**done, "item": {"type": "reasoning"}})
    assert not codex.is_turn({"type": "item.started", "item": {"type": "command_execution"}})
