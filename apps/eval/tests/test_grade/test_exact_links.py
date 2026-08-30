"""Hand-built exact and link grader cases from evaluation-spec §7 E2."""

from harnext_eval.grade.exact import grade_exact, normalize_exact
from harnext_eval.grade.links import grade_links


def test_exact_normalises_case_whitespace_ticket_keys_and_versions() -> None:
    assert grade_exact("p1", "  kafka - 123  ", "KAFKA-123").value == 1.0
    version = grade_exact("p2", " Version 03.07.0 ", "v3.7.0")
    assert version.value == 1.0
    assert version.details["normalised_prediction"] == "3.7.0"
    assert normalize_exact("Fix KAFKA_123 now") == "fix kafka-123 now"


def test_exact_unknown_is_explicit_not_an_empty_or_na_alias() -> None:
    result = grade_exact("p", " unknown. ", "UNKNOWN")
    assert result.value == 1.0
    assert result.details["prediction_is_unknown"] is True
    assert grade_exact("empty", "", "UNKNOWN").value == 0.0
    assert grade_exact("na", "N/A", "UNKNOWN").value == 0.0


def test_link_set_precision_recall_and_empty_conventions() -> None:
    result = grade_links("p", "KAFKA-1, kafka_2, KIP-9", {"KAFKA-1", "KAFKA-2"})
    assert result.value == 0.8
    assert result.details["precision"] == 2 / 3
    assert result.details["recall"] == 1.0
    assert grade_links("empty", [], []).value == 1.0
    assert grade_links("miss", [], ["KAFKA-1"]).value == 0.0
