"""File-tool adapter fixtures: no network and no model calls."""

import json

import pytest
from harnext_builder.harness.base import ConversationTranscript, HarnessRequest, TranscriptTurn
from harnext_builder.harness.codex_files import checked_path, run_file_tools


@pytest.mark.parametrize("name", ["../gold.json", "/tmp/escape.md", ".git/config",
                                 "_meta/input.json", "CLAUDE.md", "_event/source.py"])
def test_write_allowlist(tmp_path, name):
    with pytest.raises(ValueError):
        checked_path(tmp_path, name, write=True)


def test_rejects_symlinks_even_when_target_is_inside_store(tmp_path):
    (tmp_path / "entities").mkdir()
    (tmp_path / "entities" / "real.md").write_text("data")
    (tmp_path / "entities" / "alias.md").symlink_to(tmp_path / "entities" / "real.md")
    with pytest.raises(ValueError, match="symlink"):
        checked_path(tmp_path, "entities/alias.md")


@pytest.mark.asyncio
async def test_file_agent_reads_then_writes_using_only_host_tools(tmp_path, monkeypatch):
    from harnext_builder.harness import codex_files

    (tmp_path / "CLAUDE.md").write_text("operating manual")
    (tmp_path / "_meta").mkdir()
    (tmp_path / "_meta" / "input.json").write_text("evaluator private metadata")
    actions = iter([
        {"reads": ["CLAUDE.md"], "writes": [], "done": False},
        {"reads": [], "writes": [{"path": "entities/issue/CTX-41/facts.md",
                                  "content": "[ev-001] assignee=Mira"}], "done": True},
    ])

    async def complete(**kwargs):
        assert kwargs["tools_enabled"] is False
        assert kwargs["model"] == "gpt-5.6-luna"
        assert kwargs["effort"] == "medium"
        assert "evaluator private metadata" not in kwargs["prompt"]
        assert "_meta/input.json" not in json.loads(kwargs["prompt"])["available_paths"]
        return ConversationTranscript(harness="codex", model=kwargs["model"],
            turns=[TranscriptTurn(role="assistant", content=json.dumps(next(actions)))],
            usage={"input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 20})

    monkeypatch.setattr(codex_files, "execute", complete)
    request = HarnessRequest(harness="codex", model="gpt-5.6-luna", reasoning_effort="medium",
        tool_policy="files", working_dir=str(tmp_path), instruction="event", system_prompt="maintain")
    result = await run_file_tools(request)
    assert result.ok
    assert result.usage["input_tokens"] == 200
    assert result.usage["cached_input_tokens"] == 160
    assert result.usage["file_tools"][0]["read_results"] == {"CLAUDE.md": "operating manual"}
    assert (tmp_path / "entities/issue/CTX-41/facts.md").read_text() == "[ev-001] assignee=Mira"
