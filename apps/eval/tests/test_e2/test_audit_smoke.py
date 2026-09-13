"""Saved-evidence replay and narrow citation-format diagnostics for merged E2."""

import pytest
from harnext_eval.e2.audit_smoke import audit, citation_diagnostic
from harnext_eval.e2.context import Action, Profile, run, smoke_dataset


@pytest.mark.parametrize("citation,exposed,correct", [
    ("smoke/jira#ev-001", "[smoke/jira#ev-001] Mira", True),
    ("invented#ev-001", "[smoke/jira#ev-001] Mira", False),
    ("smoke/jira#ev-999", "[smoke/jira#ev-999] Mira", False),
])
def test_citation_alias_requires_exact_exposure_and_delivered_id(citation, exposed, correct):
    _, probes = smoke_dataset()
    answer = Action(action="answer", path="", query="", offset=0, value="Mira",
                    unknown=False, evidence_ids=[citation])
    result = {"answer": answer.model_dump(), "retrieval": [{"result": exposed}]}
    diagnostic = citation_diagnostic(probes[0], result, {"ev-001"})
    assert diagnostic["grade"]["evidence_correct"] is correct
    assert result["answer"]["evidence_ids"] == [citation]  # original evidence is immutable


def test_saved_tool_replay_detects_tampering(tmp_path):
    import json

    run_dir = tmp_path / "run"
    run(Profile(layouts=["S1"]), run_dir)
    report = audit(run_dir, tmp_path / "audit")
    assert report["audit_passed"] and report["probe_count"] == 10
    path = run_dir / "S1-repeat-1/answers/smoke-01.json"
    answer = json.loads(path.read_text())
    answer["retrieval"][0]["result"] += " forged evidence"
    path.write_text(json.dumps(answer))
    with pytest.raises(AssertionError):
        audit(run_dir, tmp_path / "bad-audit")
