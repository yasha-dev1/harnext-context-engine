"""Store health tests with the defects required by evaluation-spec §7 E3."""

from pathlib import Path

from harnext_eval.health.store_health import analyse_store_health, compute_store_health


def _write_clean_store(root: Path) -> Path:
    entity = root / "entities" / "issue" / "KAFKA-1"
    entity.mkdir(parents=True)
    (root / "_meta").mkdir()
    (root / "INDEX.md").write_text(
        "| Entity | Path |\n"
        "| --- | --- |\n"
        "| KAFKA-1 | [overview](entities/issue/KAFKA-1/OVERVIEW.md) |\n",
        encoding="utf-8",
    )
    (entity / "OVERVIEW.md").write_text(
        "# KAFKA-1\nCurrent assignee: Alice.\nSee [facts](facts.md).\nentity: KAFKA-1\n",
        encoding="utf-8",
    )
    (entity / "facts.md").write_text(
        "# Facts\n- Replication uses quorum acknowledgements.\n"
        "- Consumers checkpoint offsets after processing.\n",
        encoding="utf-8",
    )
    (root / "_meta" / "superseded.md").write_text(
        "# Superseded\n"
        "- issue:KAFKA-1: assignee=Bob [jira#old] was superseded by "
        "assignee=Alice on 2026-01-01 [jira#new]\n",
        encoding="utf-8",
    )
    return entity


def test_clean_store_scores_clean(tmp_path: Path) -> None:
    _write_clean_store(tmp_path)
    metrics, csv_row = analyse_store_health(tmp_path)
    assert metrics["files"] == 4
    assert metrics["entities"] == 1
    assert metrics["files_per_entity"] == 2.0
    assert metrics["over_cap_share"] == 0.0
    assert metrics["index_resolution_rate"] == 1.0
    assert metrics["dangling_cross_references"] == 0
    assert metrics["near_duplicate_fact_rate"] == 0.0
    assert metrics["supersession_leakage_rate"] == 0.0
    assert "near_duplicate_pairs" not in csv_row
    assert csv_row["files"] == 4


def test_planted_defects_are_each_detected(tmp_path: Path) -> None:
    entity = _write_clean_store(tmp_path)
    overview = entity / "OVERVIEW.md"
    overview.write_text(
        overview.read_text(encoding="utf-8")
        + "Old assignee Bob remains. See [missing](missing.md).\n",
        encoding="utf-8",
    )
    facts = entity / "facts.md"
    facts.write_text(
        facts.read_text(encoding="utf-8")
        + "- Replication uses quorum acknowledgements.\n",
        encoding="utf-8",
    )
    over_cap = entity / "timeline.md"
    over_cap.write_text("\n".join(f"event {index}" for index in range(201)), encoding="utf-8")

    metrics = compute_store_health(tmp_path)
    assert metrics["dangling_cross_references"] >= 1
    assert any("missing.md" in ref for ref in metrics["dangling_references"])
    assert metrics["near_duplicate_fact_lines"] >= 2
    assert metrics["near_duplicate_fact_rate"] > 0.0
    assert metrics["entities_with_supersession_leakage"] == 1
    assert metrics["supersession_leakage_rate"] == 1.0
    assert metrics["over_cap_files"] == 1
    assert "entities/issue/KAFKA-1/timeline.md" in metrics["over_cap_paths"]


def test_unresolved_index_row_and_explicit_missing_entity_are_detected(tmp_path: Path) -> None:
    entity = _write_clean_store(tmp_path)
    with (tmp_path / "INDEX.md").open("a", encoding="utf-8") as index:
        index.write("| KAFKA-404 | [overview](entities/issue/KAFKA-404/OVERVIEW.md) |\n")
    with (entity / "OVERVIEW.md").open("a", encoding="utf-8") as overview:
        overview.write("Related entity: KAFKA-404\n")
    metrics = compute_store_health(tmp_path)
    assert metrics["index_entries"] == 2
    assert metrics["index_resolution_rate"] == 0.5
    assert metrics["dangling_cross_references"] >= 2
