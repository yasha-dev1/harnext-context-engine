import asyncio
import json

import pytest
from harnext_eval.e2.benchmark_pilot import (
    PilotConfig,
    answer_schema,
    bind_evidence,
    compare_receipts,
    run_codex,
    trial_prompt,
    verify_native_item,
)


def test_evidence_binding_supports_raw_and_escaped_full_record():
    quote = '{"to":"Žofia","from":null}'
    task = {"required_evidence_ids": ["event-1"], "evidence": [{"id": "event-1", "text": quote}]}
    artifact = {
        "sha256": "frozen",
        "payload": {
            "documents": {
                "raw.md": "é " + quote,
                "graph.json": json.dumps({"quote": quote}, ensure_ascii=False),
                "id-only.md": "event-1",
            }
        },
    }
    gold = bind_evidence(task, artifact)
    assert {s["path"] for s in gold["spans"]} == {"raw.md", "graph.json"}
    assert next(s for s in gold["spans"] if s["path"] == "raw.md")["start"] == 3
    for s in gold["spans"]:
        assert (
            artifact["payload"]["documents"][s["path"]].encode()[s["start"] : s["end"]]
            == s["quote"].encode()
        )
    with pytest.raises(ValueError, match="missing"):
        bind_evidence(task, {"sha256": "other", "payload": {"documents": {"id.md": "event-1"}}})


def test_native_tool_monitor_rejects_shell_files_and_other_servers():
    for item in [
        {"type": "command_execution"},
        {"type": "file_change"},
        {"type": "web_search"},
        {"type": "mcp_tool_call", "server": "other", "tool": "context_read"},
        {"type": "mcp_tool_call", "server": "harnext", "tool": "write_context"},
    ]:
        with pytest.raises(ValueError):
            verify_native_item(item, ["context_read"])
    verify_native_item(
        {"type": "mcp_tool_call", "server": "harnext", "tool": "context_read"}, ["context_read"]
    )


def test_native_receipts_require_actual_matching_returned_bytes(tmp_path):
    log = tmp_path / "receipts.jsonl"
    row = {
        "tool": "context_read",
        "arguments": {"path": "x", "offset": 0},
        "output": "exact MCP output",
    }
    log.write_text(json.dumps({"kind": "start"}) + "\n" + json.dumps(row) + "\n")
    item = {
        "type": "mcp_tool_call",
        "tool": "context_read",
        "arguments": {"path": "x"},
        "result": {"content": [{"type": "text", "text": row["output"]}]},
    }
    events = [{"type": "item.completed", "item": item}]
    assert compare_receipts(events, log)["matched_calls"] == 1
    item["result"]["content"][0]["text"] = "different output"
    with pytest.raises(ValueError, match="result differs"):
        compare_receipts(events, log)


def test_answer_schemas_do_not_encode_gold_values_or_history_length():
    schema = answer_schema("assignment_history")
    assert schema["properties"]["value"]["items"]["required"] == ["at", "from", "to"]
    assert "minItems" not in schema["properties"]["value"]
    assert answer_schema("closed_at_snapshot")["properties"]["value"] == {"type": "boolean"}
    assert answer_schema("assignee_at_snapshot")["properties"]["value"] == {
        "type": ["string", "null"]
    }


def test_no_mcp_prompt_preserves_question_cutoff_and_answer_contract():
    prompt = (
        "Cutoff: 2020-01-01. Who owned KAFKA-1?\n"
        "Use only the Harnext context-engine MCP tools. "
        "You have no shell, native filesystem, browser or other data source. "
        'Return JSON {"value": <answer>, "evidence_ids": [<source IDs>]}.'
    )
    task = {"agent_prompt": prompt}
    assert trial_prompt(task, "mcp") == prompt
    baseline = trial_prompt(task, "none")
    assert baseline.split("\n")[0] == prompt.split("\n")[0]
    assert baseline.endswith(prompt.split("You have no shell")[1])
    assert "No context-engine MCP tools" in baseline
    with pytest.raises(ValueError, match="exactly one"):
        trial_prompt({"agent_prompt": "unknown prompt format"}, "none")


def test_no_mcp_monitor_rejects_even_normally_allowed_harnext_call():
    with pytest.raises(ValueError, match="unapproved MCP"):
        verify_native_item(
            {"type": "mcp_tool_call", "server": "harnext", "tool": "context_search"}, []
        )


def test_no_mcp_launch_never_loads_or_registers_context_server(tmp_path, monkeypatch):
    from harnext_eval.e2 import benchmark_pilot

    pilot = PilotConfig(
        protocol="harnext-benchmark-pilot-v1",
        benchmark=tmp_path / "absent.json",
        engine_profile=tmp_path / "absent.yaml",
        selection_seed="test",
        families=["status_at_snapshot"],
        model="gpt-5.6-luna",
        effort="medium",
        context_access="none",
    )
    captured = []

    async def stop_before_process(*argv, **kwargs):
        captured.extend(argv)
        raise RuntimeError("captured launch")

    monkeypatch.setattr(benchmark_pilot.asyncio, "create_subprocess_exec", stop_before_process)
    monkeypatch.setattr(benchmark_pilot.subprocess, "check_output", lambda *a, **k: "test-version")
    monkeypatch.setattr(
        benchmark_pilot, "load_config", lambda *a: pytest.fail("context config accessed")
    )
    with pytest.raises(RuntimeError, match="captured launch"):
        asyncio.run(
            run_codex(
                None,
                pilot,
                "public prompt",
                answer_schema("status_at_snapshot"),
                tmp_path / "reader",
                workspace=tmp_path,
            )
        )
    assert "--ignore-user-config" in captured
    assert "features.shell_tool=false" in captured
    assert 'web_search="disabled"' in captured
    assert not any("mcp_servers" in value for value in captured)
    metadata = json.loads((tmp_path / "reader" / "launch.json").read_text())
    assert metadata["allowed_mcp_tools"] == []
    assert metadata["context_access"] == "none"
