"""Gold isolation, real span exposure, and coding-task execution contracts."""

import json

import pytest
from harnext_eval.e2.context import Action, SnapshotTools
from harnext_eval.e2.development import coding_trial, export
from harnext_eval.e2.development_dataset import dataset
from harnext_eval.e2.retrieval_metrics import EvidenceSpan, score_retrieval
from harnext_eval.providers.llm import LLMResult


def action(kind="read", **kwargs):
    return Action.model_validate(
        dict(
            action=kind,
            path="facts.md",
            query="",
            offset=0,
            value=None,
            unknown=False,
            evidence_ids=[],
        )
        | kwargs
    )


def fixture():
    files = {"facts.md": "ev-1\nUse tenant plus key.\nUnrelated text.\n"}
    text = "Use tenant plus key."
    start = files["facts.md"].encode().index(text.encode())
    spans = [
        EvidenceSpan(
            unit_id="scope",
            path="facts.md",
            start=start,
            end=start + len(text.encode()),
            quote=text,
        )
    ]
    return files, spans


def score(files, spans, trace, **kwargs):
    return score_retrieval(
        files,
        spans,
        [{"scope"}],
        trace,
        budget=1000,
        response_cap=kwargs.pop("cap", 1000),
        annotations_exhaustive=kwargs.pop("exhaustive", True),
        **kwargs,
    )


def test_event_id_or_truncated_support_does_not_earn_recall():
    files, spans = fixture()
    tools = SnapshotTools(files, budget=1000, response_cap=5)
    tools.invoke(action())
    result = score(files, spans, tools.log, cap=5)
    assert result["required_group_recall"] == 0
    assert result["unit_precision"] is None
    assert result["returned_bytes"] == 5


def test_partial_reads_combine_and_repeats_are_charged():
    files, spans = fixture()
    tools = SnapshotTools(files, budget=1000, response_cap=10)
    for offset in (5, 15, 5):
        tools.invoke(action(offset=offset))
    result = score(files, spans, tools.log, cap=10)
    assert result["required_group_recall"] == 1
    assert result["first_relevant_context_call"] == 2
    assert result["context_calls"] == 3 and result["returned_bytes"] == 30
    assert result["relevant_byte_precision"] == 1


def test_search_maps_content_not_paths_and_charges_wrapper_bytes():
    files, spans = fixture()
    tools = SnapshotTools(files, budget=1000, response_cap=1000)
    tools.invoke(action("search", query="tenant"))
    result = score(files, spans, tools.log)
    assert result["required_group_recall"] == result["unit_precision"] == 1
    assert 0 < result["relevant_byte_precision"] < 1
    assert score(files, spans, tools.log, exhaustive=False)["unit_precision"] is None
    tools.log[0]["result"] = "fabricated"
    with pytest.raises(ValueError, match="replay"):
        score(files, spans, tools.log)


def test_stale_annotations_and_absent_gold_rejected():
    files, spans = fixture()
    with pytest.raises(ValueError, match="stale"):
        score({"facts.md": "changed"}, spans, [])
    with pytest.raises(ValueError, match="not annotated"):
        score_retrieval(
            files,
            spans,
            [{"missing"}],
            [],
            budget=100,
            response_cap=100,
            annotations_exhaustive=True,
        )


def test_alternative_support_and_empty_gold_are_explicit():
    files, spans = fixture()
    tools = SnapshotTools(files, budget=1000, response_cap=1000)
    tools.invoke(action())
    spans.append(EvidenceSpan(unit_id="alternate", path="facts.md", start=0, end=4, quote="ev-1"))
    result = score_retrieval(
        files,
        spans,
        [{"scope", "alternate"}],
        tools.log,
        budget=1000,
        response_cap=1000,
        annotations_exhaustive=True,
    )
    assert result["required_group_recall"] == 1
    empty = score_retrieval(
        files, spans, [], [], budget=1000, response_cap=1000, annotations_exhaustive=True
    )
    assert empty["required_group_recall"] is None and empty["unit_precision"] is None


def test_export_separates_reference_tests_gold_and_public_inputs(tmp_path):
    out = tmp_path / "dataset"
    result = export(out)
    assert result["qa_count"] == 32 and result["coding_count"] == 5
    public = json.loads((out / "public/dev-code-retry.json").read_text())
    assert set(public) == {
        "task_id",
        "prompt",
        "base_files",
        "visible_tests",
        "base_content_sha256",
    }
    assert "hidden_tests" not in public and "reference" not in public
    questions = json.loads((out / "public/questions.json").read_text())
    assert all("gold" not in row and "source_event_ids" not in row for row in questions)
    with pytest.raises(ValueError, match="new or empty"):
        export(out)


def test_coding_loop_no_gold_and_hidden_grading_after_finish(monkeypatch):
    _, _, _, _, tasks = dataset()
    task = tasks[2]
    test_calls = []
    provider_prompts = []

    def run(files, visible, hidden, **kwargs):
        test_calls.append(hidden)
        return {"status": "completed", "passed": True, "tests": {"test": "passed"}}

    monkeypatch.setattr("harnext_eval.e2.development.run_tests", run)
    actions = [
        ("context_read", "events/retry-delay.md", ""),
        ("repo_write", "test_visible.py", "try to replace tests"),
        ("run_tests", "", ""),
        ("repo_write", "service.py", task.reference["service.py"]),
        ("finish", "", ""),
    ]

    class ScriptedProvider:
        def complete(self, system, user, **kwargs):
            provider_prompts.append(user)
            kind, path, content = actions.pop(0)
            raw = dict(
                action=kind,
                path=path,
                content=content,
                query="",
                offset=0,
                summary="local test fixture patch",
            )
            return LLMResult(text=json.dumps(raw), json=raw, usage={"input_tokens": 10})

    result = coding_trial(task, ScriptedProvider(), arm="raw", image="sha256:test")
    assert result["resolved"] and result["provider_calls"] == 5
    assert result["first_edit_step"] == 4
    assert result["retrieval"]["context_calls"] == 1
    assert result["retrieval"]["required_group_recall"] == pytest.approx(1 / 3)
    assert test_calls == [None, task.hidden_tests]
    assert "ERROR: only existing" in result["trace"][1]["result"]
    assert all("test_acceptance_header" not in prompt for prompt in provider_prompts)
    assert result["usage"]["input_tokens"] == 50
    assert "+++ b/service.py" in result["patch"]
