"""Offline provider and subprocess isolation tests for audit recommendations 1–6."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml
from harnext_builder.agentfs.git_backend import GitBackend
from harnext_builder.harness.base import HarnessRequest
from harnext_eval.config import ExperimentConfig, load_config
from harnext_eval.manifest import build_manifest
from harnext_eval.providers.factory import (
    OfflineViolation,
    assert_offline_ok,
    make_embeddings,
    make_harness_name,
    make_llm,
    provider_summary,
)
from pydantic import ValidationError

CONFIGS = Path(__file__).parents[2] / "configs"
POISONED_IMPORTS = ("anthropic", "openai", "claude_agent_sdk")


@pytest.mark.parametrize("profile", ["baseline-minimal.yaml", "s1-templated.yaml"])
def test_offline_profiles_do_not_import_real_providers(
    monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    for module_name in POISONED_IMPORTS:
        monkeypatch.setitem(sys.modules, module_name, None)

    cfg = load_config(CONFIGS / profile)
    assert_offline_ok(cfg)
    assert type(make_llm(cfg)).__name__ == "FakeLLM"
    assert type(make_embeddings(cfg)).__name__ == "FakeEmbeddings"
    assert make_harness_name(cfg) == "fake"
    assert provider_summary(cfg) == {
        "reader": "FakeLLM",
        "embeddings": "FakeEmbeddings",
        "builder": "FakeHarness",
        "offline_enforced": True,
    }


def test_offline_rejects_anthropic_reader_before_construction() -> None:
    raw = yaml.safe_load((CONFIGS / "baseline-minimal.yaml").read_text(encoding="utf-8"))
    raw["engine"]["reader"]["provider"] = "anthropic"
    cfg = ExperimentConfig.model_validate(raw)

    with pytest.raises(OfflineViolation, match="reader.provider=anthropic"):
        make_llm(cfg)


def test_config_rejects_embeddings_without_an_adapter() -> None:
    raw = yaml.safe_load((CONFIGS / "baseline-minimal.yaml").read_text(encoding="utf-8"))
    raw["engine"]["embeddings"]["provider"] = "openai"

    with pytest.raises(ValidationError, match="embeddings.provider"):
        ExperimentConfig.model_validate(raw)


def test_offline_guard_covers_kafka_and_corpus_fetch() -> None:
    cfg = load_config(CONFIGS / "baseline-minimal.yaml")

    with pytest.raises(OfflineViolation, match="transport=kafka"):
        assert_offline_ok(cfg, transport="kafka")
    with pytest.raises(OfflineViolation, match="corpus fetch=pony-mail"):
        assert_offline_ok(cfg, corpus_fetch="pony-mail")


def test_manifest_stores_resolved_provider_summary(tmp_path: Path) -> None:
    replay = tmp_path / "replay.jsonl"
    replay.write_text("{}\n", encoding="utf-8")
    cfg = load_config(CONFIGS / "baseline-minimal.yaml")
    make_llm(cfg)
    make_embeddings(cfg)
    make_harness_name(cfg)

    manifest = build_manifest(
        run_id="offline-test",
        config=cfg,
        replay_path=replay,
        provider_summary=provider_summary(cfg),
    )

    assert manifest.provider_summary["reader"] == "FakeLLM"
    assert manifest.provider_summary["embeddings"] == "FakeEmbeddings"
    assert manifest.provider_summary["builder"] == "FakeHarness"
    assert manifest.provider_summary["offline_enforced"] is True


def test_fake_harness_child_environment_has_no_api_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in (
        "ANTHROPIC_API_KEY",
        "HARNEXT_API_KEY",
        "NVIDIA_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.setenv(name, "must-not-reach-child")
    backend = GitBackend(tmp_path / "backend")
    backend.ensure_seeded("offline", {"INDEX.md": "# offline\n"})
    request = HarnessRequest(
        harness="fake",
        working_dir=".",
        instruction="test",
        system_prompt="test",
    )
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    request_path.write_text(request.model_dump_json(), encoding="utf-8")

    result = backend.run_build(
        "offline",
        [
            sys.executable,
            "-c",
            "import json, os; print(json.dumps(dict(os.environ)))",
        ],
        {"REQUEST_PATH": str(request_path), "RESULT_PATH": str(result_path)},
        30,
    )

    assert result.ok, result.stderr
    child_env = json.loads(result.stdout)
    assert not [name for name in child_env if name.endswith("_API_KEY")]
    assert child_env["REQUEST_PATH"] == str(request_path)
    assert child_env["RESULT_PATH"] == str(result_path)
