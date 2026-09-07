"""Run-manifest hashing and writing for docs/evaluation-spec.md §11."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from harnext_eval.types import RunManifest


class ProviderRunManifest(RunManifest):
    """Run manifest extended with resolved, non-secret provider metadata."""

    provider_summary: dict[str, str | bool]


def _embedding_model_id(config: BaseModel | dict[str, Any]) -> str:
    """Return the configured embedding model and revision for the manifest."""

    raw: Any = config.model_dump(mode="python") if isinstance(config, BaseModel) else config
    engine = raw.get("engine", raw) if isinstance(raw, dict) else {}
    embeddings = engine.get("embeddings", {}) if isinstance(engine, dict) else {}
    if not isinstance(embeddings, dict):
        return ""
    if embeddings.get("provider") == "fake":
        return "fake-feature-hash-blake2b-v1@1"
    model = embeddings.get("model")
    revision = embeddings.get("revision")
    if isinstance(model, str) and isinstance(revision, str) and model and revision:
        return f"{model}@{revision}"
    return ""


def sha256_file(path: str | Path | None) -> str:
    """Return a file's SHA-256, or the empty-input hash when no file is supplied."""

    digest = hashlib.sha256()
    if path is not None:
        with Path(path).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def current_git_sha(root: str | Path | None = None) -> str:
    """Resolve the current commit without changing repository state."""

    command = ["git"]
    if root is not None:
        command.extend(["-C", str(Path(root))])
    command.extend(["rev-parse", "HEAD"])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def build_manifest(
    *,
    run_id: str,
    config: BaseModel | dict[str, Any],
    replay_path: str | Path,
    probe_path: str | Path | None = None,
    model_ids: dict[str, str] | None = None,
    prices: dict[str, float] | None = None,
    seeds: list[int] | None = None,
    prereg_ref: str | None = None,
    repo_root: str | Path | None = None,
    provider_summary: dict[str, str | bool] | None = None,
) -> ProviderRunManifest:
    resolved_model_ids = dict(model_ids or {})
    embedding_model_id = _embedding_model_id(config)
    if embedding_model_id:
        resolved_model_ids["embeddings"] = embedding_model_id
    return ProviderRunManifest(
        run_id=run_id,
        created_at=datetime.now(UTC),
        config_hash=sha256_json(config),
        replay_hash=sha256_file(replay_path),
        probe_hash=sha256_file(probe_path),
        git_sha=current_git_sha(repo_root),
        model_ids=resolved_model_ids,
        prices=prices or {},
        seeds=seeds or [],
        prereg_ref=prereg_ref,
        provider_summary=provider_summary or {},
    )


def write_manifest(manifest: RunManifest, destination: str | Path) -> Path:
    """Write ``manifest.json`` atomically enough for local evaluation runs."""

    path = Path(destination)
    if path.suffix != ".json":
        path = path / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
