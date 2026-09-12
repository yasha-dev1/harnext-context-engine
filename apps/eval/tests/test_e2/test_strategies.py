"""Behavioral checks of YAML treatments, time boundaries and real MCP transport."""

import json
from pathlib import Path

import pytest
from fastmcp import Client
from harnext_builder.strategies.config import FileLayout, StrategyConfig, load_config
from harnext_builder.strategies.engine import build, load_artifact
from harnext_builder.strategies.graph import eligible, traverse
from harnext_builder.strategies.retrieval import chunks
from harnext_eval.e2.mcp_experiment import audit_receipts
from harnext_eval.e2.strategy_matrix import expand
from harnext_mcp.experiment import Trial, create_server

PROFILE = Path(__file__).resolve().parents[2] / "configs/strategies/indexed.yaml"


def test_learned_output_schema_requires_explicit_optional_fields():
    from harnext_builder.strategies.records import ExtractionResult, extraction_schema

    schema = extraction_schema()
    fact = schema["$defs"]["Fact"]
    assert set(fact["required"]) == set(fact["properties"])
    assert fact["additionalProperties"] is False
    assert "default" not in fact["properties"]["topic"]
    assert {"type": "null"} in fact["properties"]["path"]["anyOf"]
    # Transport constraints must not mutate the domain model's defaults.
    assert (
        ExtractionResult.model_json_schema()["$defs"]["Fact"]["properties"]["topic"]["default"]
        == "general"
    )


def config(tmp_path, **overrides):
    base = load_config(PROFILE).model_dump(mode="json")
    base["artifact_dir"] = str(tmp_path / "snapshot")
    for section, values in overrides.items():
        if isinstance(values, dict):
            base[section].update(values)
        else:
            base[section] = values
    return StrategyConfig.model_validate(base)


@pytest.mark.parametrize("layout", [item for item in FileLayout if item != FileLayout.CURATED])
async def test_layouts_change_materialized_paths_and_preserve_cutoff(tmp_path, layout):
    cfg = config(
        tmp_path,
        files={"layout": layout},
        representation="files",
        trial={"tools": ["context_list", "context_read", "context_search"]},
        expose_sources=False,
    )
    artifact = await build(cfg)
    docs = artifact["payload"]["documents"]
    assert "FUTURE_SENTINEL" not in json.dumps(docs)
    assert artifact["payload"]["source_ids"] == ["event-1", "event-2", "event-3"]
    expected = {
        FileLayout.VERBATIM: "events/",
        FileLayout.FOLDERED: "entities/",
        FileLayout.FLAT: "facts/",
        FileLayout.INDEXED: "entities/",
        FileLayout.LEDGER: "ledger/",
        FileLayout.TOPICS: "topics/",
    }
    assert any(path.startswith(expected[layout]) for path in docs)
    assert ("INDEX.md" in docs) == (
        layout in {FileLayout.INDEXED, FileLayout.LEDGER, FileLayout.TOPICS}
    )
    with pytest.raises(ValueError, match="new or empty"):
        await build(cfg)


async def test_policy_effects_current_retention_and_concise_rendering(tmp_path):
    cfg = config(
        tmp_path,
        files={"layout": "indexed_entities", "retention": "current", "prompt": "concise_supported"},
        expose_sources=False,
        trial={"tools": ["context_list", "context_read", "context_search"]},
    )
    artifact = await build(cfg)
    contents = "\n".join(artifact["payload"]["documents"].values())
    assert "person:Mira" not in contents and "assigned to Mira" not in contents
    assert "Noah" in contents and "Quote:" not in contents
    assert "pr:71" in contents


async def test_temporal_graph_latest_valid_known_alias_and_two_hops(tmp_path):
    cfg = config(tmp_path)
    artifact = await build(cfg)
    rows = artifact["payload"]["graph"]
    old = eligible(
        rows,
        cfg.valid_at.replace(day=1),
        cfg.observed_at.replace(day=1),
        ["assigned_to", "has_status"],
    )
    assert any(row["object"] == "person:Mira" for row in old)
    assert not any(row["object"] == "person:Noah" for row in old)
    trial = Trial(cfg)
    try:
        assert "person:Noah" in trial.invoke(
            "get_assertions", {"entity": "Cache Repair", "predicate": "assigned_to"}
        )
        result = trial.invoke("neighbors", {"entity": "issue:CTX-41"})
        assert "file:src/cache.py" in result
        assert "person:Mira" not in result and "Uma" not in result
    finally:
        trial.close()


@pytest.mark.parametrize("method", ["neighbors", "paths", "personalized_pagerank", "dual_level"])
async def test_graph_traversal_modes_return_source_backed_edges(tmp_path, method):
    cfg = config(tmp_path, graph={"traversal": method})
    artifact = await build(cfg)
    rows = eligible(
        artifact["payload"]["graph"], cfg.valid_at, cfg.observed_at, cfg.graph.functional_predicates
    )
    found = traverse(rows, ["issue:CTX-41"], cfg, "file:src/cache.py")
    assert any(row["object"] == "file:src/cache.py" for row in found)
    assert all(row["source_id"] in artifact["payload"]["source_ids"] for row in found)


async def test_budget_restart_integrity_and_invalid_reads(tmp_path):
    cfg = config(tmp_path, trial={"budget_bytes": 512, "response_bytes": 256})
    await build(cfg)
    trial = Trial(cfg)
    result = trial.invoke("context_read", {"path": "../../private/gold.json"})
    assert "unavailable" in result
    trial.invoke("context_search", {"query": "assigned Noah"})
    used = trial.used
    trial.close()
    trial = Trial(cfg)
    try:
        assert trial.used == used
        for _ in range(5):
            trial.invoke("context_search", {"query": "cache"})
        assert trial.used <= 512
        log_path = trial.log_path
    finally:
        trial.close()
    lines = log_path.read_text().splitlines()
    tampered = json.loads(lines[-1])
    tampered["bytes"] += 1
    lines[-1] = json.dumps(tampered)
    log_path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="hash chain"):
        Trial(cfg)


async def test_real_mcp_client_exposes_only_configured_tools(tmp_path):
    cfg = config(tmp_path, trial={"tools": ["context_search", "context_read"]})
    await build(cfg)
    trial = Trial(cfg)
    try:
        async with Client(create_server(trial)) as client:
            assert {tool.name for tool in await client.list_tools()} == {
                "context_search",
                "context_read",
            }
            response = await client.call_tool("context_search", {"query": "assigned Noah"})
            assert "Noah" in response.content[0].text
            assert trial.calls == 1 and trial.used == len(response.content[0].text.encode())
    finally:
        trial.close()


def test_unknown_options_and_incompatible_methods_fail_early(tmp_path):
    with pytest.raises(ValueError):
        config(tmp_path, files={"layout": "imaginary"})
    with pytest.raises(ValueError, match="learned"):
        config(tmp_path, files={"layout": "agent_curated"})
    with pytest.raises(ValueError, match="verbatim"):
        config(tmp_path, files={"layout": "verbatim_dump", "summarization": "extractive"})
    with pytest.raises(ValueError, match="overlap"):
        config(tmp_path, retrieval={"chunk_bytes": 256, "overlap_bytes": 256})


def test_chunk_offsets_and_overlap_are_utf8_exact(tmp_path):
    cfg = config(tmp_path, retrieval={"chunk_bytes": 64, "overlap_bytes": 10})
    docs = {"unicode.md": "á🙂abcdefghij" * 20}
    records = chunks(docs, cfg.retrieval)
    assert len(records) > 1
    for row in records:
        assert docs[row["path"]].encode()[row["start"] : row["end"]] == row["text"].encode()


async def test_artifact_mutation_is_rejected(tmp_path):
    cfg = config(tmp_path)
    artifact = await build(cfg)
    artifact["payload"]["documents"]["INDEX.md"] = "changed"
    path = cfg.artifact_dir / "snapshot.json"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="hash"):
        load_artifact(path)


async def test_receipts_measure_support_and_duplicates_without_citation_shortcuts(tmp_path):
    cfg = config(tmp_path)
    artifact = await build(cfg)
    trial = Trial(cfg)
    try:
        for _ in range(2):
            trial.invoke("get_assertions", {"entity": "issue:CTX-41", "predicate": "assigned_to"})
        row = json.loads(trial.log_path.read_text().splitlines()[1])
        item = json.loads(row["output"])["items"][0]
        quote = "CTX-41 is now assigned to Noah."
        start = item["text"].encode().index(quote.encode())
        gold = {
            "snapshot_sha256": artifact["sha256"],
            "required_groups": [["owner"]],
            "spans": [
                {
                    "unit_id": "owner",
                    "path": item["path"],
                    "start": start,
                    "end": start + len(quote.encode()),
                    "quote": quote,
                }
            ],
            "annotations_exhaustive": False,
        }
        report = audit_receipts(artifact, trial.log_path, gold)
        assert report["required_group_recall"] == 1
        assert report["context_calls"] == 2 and report["first_relevant_call"] == 1
        assert 0 < report["repeated_byte_fraction"] < 1
        assert report["unit_precision"] is None
        gold["snapshot_sha256"] = "wrong"
        with pytest.raises(ValueError, match="exact snapshot"):
            audit_receipts(artifact, trial.log_path, gold)
    finally:
        trial.close()


async def test_aliases_never_merge_conflicting_explicit_keys(tmp_path):
    from harnext_builder.strategies.graph import build_graph

    cfg = config(tmp_path)
    artifact = await build(cfg)
    rows = artifact["payload"]["graph"]
    rows[0]["aliases"] = ["issue:OTHER"]
    rows[0]["quote"] += " issue:OTHER"
    _, decisions = build_graph(rows, cfg)
    assert any(item["decision"] == "canonical_conflict" for item in decisions)


async def test_entity_bundles_are_real_bounded_index_documents(tmp_path):
    cfg = config(
        tmp_path, retrieval={"chunking": "entity_bundle", "chunk_bytes": 512, "overlap_bytes": 64}
    )
    artifact = await build(cfg)
    records = artifact["payload"]["index"]["records"]
    assert records and all(row["path"].startswith("bundles/") for row in records)
    assert all(len(row["text"].encode()) <= 512 for row in records)
    assert any("[document:" in row["text"] for row in records)


async def test_window_retention_changes_observed_source_prefix(tmp_path):
    cfg = config(tmp_path, files={"retention": "window", "retention_days": 1})
    artifact = await build(cfg)
    assert artifact["payload"]["source_ids"] == ["event-3"]


def test_matrix_validates_all_cells_before_writing(tmp_path):
    import yaml

    path = tmp_path / "matrix.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "protocol": "harnext-matrix-v1",
                "base": str(PROFILE),
                "output_dir": "matrix",
                "factors": {"files.prompt": ["preserve_supported", "concise_supported"]},
            }
        )
    )
    report = expand(path)
    assert report["conditions"] == 2
    configs = [load_config(p) for p in (tmp_path / "matrix").glob("*.yaml")]
    assert len({c.artifact_dir for c in configs}) == 2
    path.write_text(
        yaml.safe_dump(
            {
                "protocol": "harnext-matrix-v1",
                "base": str(PROFILE),
                "output_dir": "invalid",
                "factors": {"files.layout": ["indexed_entities", "bogus"]},
            }
        )
    )
    with pytest.raises(ValueError):
        expand(path)
    assert not (tmp_path / "invalid").exists()
