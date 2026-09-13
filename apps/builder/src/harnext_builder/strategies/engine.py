"""Build and load immutable strategy artifacts using the production harness seam."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import defaultdict
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any

from .config import FileLayout, StrategyConfig, canonical, digest
from .graph import build_graph, eligible
from .prompts import render_prompt
from .records import ExtractionResult, Source, extraction_schema
from .retrieval import SearchIndex, chunks


def filename(value: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80]
    return f"{readable}-{digest(value)[:10]}"


async def learned_extract(source: Source, config: StrategyConfig, previous: list[dict]):
    if config.builder.model is None:
        raise ValueError("learned extraction requires a model")
    from harnext_builder.harness.base import HarnessRequest
    from harnext_builder.harness.registry import get_harness

    instruction = canonical(
        {
            "source": source.model_dump(mode="json"),
            "existing_assertions": previous,
            "schema": extraction_schema(),
        }
    )
    if len((render_prompt(config) + instruction).encode()) > config.builder.context_bytes:
        raise ValueError("builder context byte allowance exceeded; no silent history truncation")
    with tempfile.TemporaryDirectory(prefix="harnext-extract-") as cwd:
        if config.builder.harness == "codex":
            if config.builder.effort is None:
                raise ValueError("Codex requires explicit effort")
            from harnext_builder.harness.codex import execute

            transcript = await execute(
                model=config.builder.model,
                effort=config.builder.effort,
                cwd=cwd,
                prompt=render_prompt(config) + "\n" + instruction,
                tools_enabled=False,
                timeout_s=config.builder.timeout_s,
                json_schema=extraction_schema(),
            )
        else:
            request = HarnessRequest(
                harness=config.builder.harness,
                working_dir=cwd,
                instruction=instruction,
                system_prompt=render_prompt(config),
                allowed_tools=[],
                disallowed_tools=["Bash", "Read", "Write", "Edit", "Task", "WebFetch", "WebSearch"],
                model=config.builder.model,
                reasoning_effort=config.builder.effort,
                max_turns=1,
                timeout_s=config.builder.timeout_s,
            )
            if config.builder.harness == "harnext":
                from harnext_builder.harness.harnext import HarnextHarness
                from harnext_builder.settings import BuilderSettings

                settings_data: dict[str, Any] = {
                    "_env_file": None,
                    "harnext_model": config.builder.model,
                    "harnext_provider": config.builder.provider,
                    "harnext_api_key": os.environ.get(config.builder.api_key_env),
                    "harnext_api_key_env": config.builder.api_key_env,
                }
                settings = BuilderSettings(**settings_data)
                harness = HarnextHarness(settings)
            else:
                harness = get_harness(config.builder.harness)
            transcript = await harness.run(request)
    if not transcript.ok:
        raise RuntimeError(transcript.error or transcript.stop_reason)
    candidates = [turn.content for turn in transcript.turns if turn.role in {"assistant", "result"}]
    parsed = None
    for candidate in reversed(candidates):
        try:
            parsed = ExtractionResult.model_validate_json(candidate)
            break
        except ValueError:
            continue
    if parsed is None:
        raise ValueError("learned extraction did not return a complete valid JSON fact set")
    validated = Source.model_validate({**source.model_dump(), "facts": parsed.facts})
    return validated, transcript.model_dump(mode="json")


def render_files(sources: list[Source], records: list[dict], config: StrategyConfig):
    groups = defaultdict(list)
    entities = {}
    if config.files.layout in {FileLayout.VERBATIM, FileLayout.FOLDERED}:
        for source in sources:
            parent = (
                "events"
                if config.files.layout == FileLayout.VERBATIM
                else f"entities/{filename(source.entity)}"
            )
            path = f"{parent}/{filename(source.id)}.md"
            groups[path].append(source.text)
            entities[path] = source.entity
    else:
        rows = records
        if config.files.update == "replace" or config.files.retention == "current":
            rows = eligible(
                records,
                config.valid_at,
                config.observed_at,
                config.files.functional_predicates,
            )
        for row in rows:
            if config.files.layout == FileLayout.FLAT:
                path = f"facts/{filename(row['id'])}.md"
            elif config.files.layout == FileLayout.TOPICS:
                path = f"topics/{filename(row['topic'])}/{filename(row['subject'])}.md"
            elif config.files.layout == FileLayout.CURATED:
                candidate = PurePosixPath(row["path"] or "")
                if (
                    not row["path"]
                    or candidate.is_absolute()
                    or ".." in candidate.parts
                    or candidate.suffix != ".md"
                    or candidate.parts[0] in {"sources", "graph", "_meta"}
                    or str(candidate) == "INDEX.md"
                ):
                    raise ValueError(
                        "agent_curated requires a safe, nonreserved Markdown path for every fact"
                    )
                path = str(candidate)
            else:
                path = f"entities/{filename(row['subject'])}/facts.md"
            text = row["text"]
            if config.files.summarization == "extractive":
                text = " ".join(re.split(r"(?<=[.!?])\s+", text)[: config.files.summary_sentences])
            line = f"- {text} [source:{row['source_id']}; assertion:{row['id']}]"
            if config.files.prompt == "preserve_supported":
                line += f"\n  Quote: {row['quote']}"
            if config.files.update == "supersede":
                line += f"\n  valid_from={row['valid_from']}; known_from={row['known_from']}; predicate={row['predicate']}"
            groups[path].append(line)
            entities[path] = (
                row["subject"] if entities.get(path, row["subject"]) == row["subject"] else path
            )
        if config.files.layout == FileLayout.LEDGER:
            for row in records:
                path = f"ledger/{filename(row['source_id'])}.md"
                groups[path].append(canonical(row))
                entities[path] = row["subject"]
    docs, mapping = {}, {}
    for path, entries in sorted(groups.items()):
        # Verbatim variants are never clipped or rewritten by a page cap.
        if config.files.layout in {FileLayout.VERBATIM, FileLayout.FOLDERED}:
            docs[path] = "".join(entries)
            mapping[path] = entities[path]
            continue
        pages, current = [], ""
        for entry in entries:
            block = entry + "\n"
            if len(block.encode()) > config.files.page_bytes:
                raise ValueError(
                    "one fact exceeds page_bytes; increase cap rather than silently lose content"
                )
            if current and len((current + block).encode()) > config.files.page_bytes:
                pages.append(current)
                current = ""
            current += block
        if current:
            pages.append(current)
        for index, page in enumerate(pages):
            target = path if index == 0 else path.removesuffix(".md") + f"-{index + 1}.md"
            docs[target], mapping[target] = page, entities[path]
    if config.files.layout in {
        FileLayout.INDEXED,
        FileLayout.LEDGER,
        FileLayout.TOPICS,
        FileLayout.CURATED,
    }:
        docs["INDEX.md"] = (
            "# Context index\n" + "\n".join(f"- {path}" for path in sorted(docs)) + "\n"
        )
    return docs, mapping


async def build(config: StrategyConfig) -> dict:
    out = config.artifact_dir
    if out.exists() and any(out.iterdir()):
        raise ValueError("build artifact directory must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    raw = config.source_jsonl.read_bytes()
    all_sources = [
        Source.model_validate_json(line) for line in raw.decode().splitlines() if line.strip()
    ]
    if len({s.id for s in all_sources}) != len(all_sources):
        raise ValueError("duplicate source IDs; use immutable version IDs")
    sources = sorted(
        (s for s in all_sources if s.observed_at <= config.observed_at),
        key=lambda s: (s.observed_at, s.id),
    )
    if config.files.retention == "window":
        boundary = config.observed_at - timedelta(days=config.files.retention_days)
        sources = [s for s in sources if s.observed_at >= boundary]
    calls, records = [], []
    learned = (
        config.files.extraction == "atomic_llm"
        or config.files.summarization == "abstractive"
        or config.files.layout == FileLayout.CURATED
    )
    for source in sources:
        if learned:
            if len(calls) >= config.builder.max_calls:
                raise ValueError("builder call budget exhausted")
            source, transcript = await learned_extract(source, config, records)
            calls.append(transcript)
            (out / "builder-transcripts.json").write_text(canonical(calls))
        elif not source.facts and (
            config.representation != "files"
            or config.files.layout not in {FileLayout.VERBATIM, FileLayout.FOLDERED}
        ):
            raise ValueError(
                "structured extraction requires explicit source-backed facts; select atomic_llm for unstructured sources"
            )
        for fact in source.facts:
            if (
                config.files.prompt == "domain_focused"
                and fact.predicate not in config.files.predicates
            ):
                continue
            records.append(
                {
                    **fact.model_dump(mode="json"),
                    "id": f"{source.id}/{fact.id}",
                    "source_id": source.id,
                    "valid_from": (fact.valid_from or source.valid_at).isoformat(),
                    "known_from": source.observed_at.isoformat(),
                }
            )
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate assertion IDs")
    if config.files.retention == "current":
        records = eligible(
            records, config.valid_at, config.observed_at, config.files.functional_predicates
        )
    graph, resolutions = (
        build_graph(records, config) if config.representation in {"graph", "hybrid"} else ([], [])
    )
    documents, entity_map = (
        render_files(sources, records, config) if config.representation != "graph" else ({}, {})
    )
    for edge in graph:
        path = f"graph/{filename(edge['id'])}.json"
        edge["document_path"] = path
        documents[path] = canonical({k: v for k, v in edge.items() if k != "document_path"})
        entity_map[path] = edge["subject"]
    if config.expose_sources:
        for source in sources:
            path = f"sources/{filename(source.id)}.md"
            documents[path], entity_map[path] = source.text, source.entity
    index_docs = documents
    if config.retrieval.chunking == "entity_bundle":
        grouped = defaultdict(list)
        for path, text in sorted(documents.items()):
            if path != "INDEX.md":
                grouped[entity_map[path]].append(f"[document:{path}]\n{text}")
        index_docs = {
            f"bundles/{filename(entity)}.md": "\n".join(parts) for entity, parts in grouped.items()
        }
        documents.update(index_docs)
    index = SearchIndex(config.retrieval, chunks(index_docs, config.retrieval))
    payload = {
        "config": config.model_dump(mode="json"),
        "source_hash": digest(raw),
        "source_ids": [s.id for s in sources],
        "documents": documents,
        "graph": graph,
        "alias_decisions": resolutions,
        "index": index.serializable(),
        "prompt": render_prompt(config),
        "prompt_hash": digest(render_prompt(config)),
        "implementation_hash": digest(
            {p.name: digest(p.read_bytes()) for p in Path(__file__).parent.glob("*.py")}
        ),
        "versions": {name: version(name) for name in ["harnext-builder", "numpy", "rank-bm25"]},
    }
    artifact = {"sha256": digest(payload), "payload": payload}
    (out / "snapshot.json").write_text(canonical(artifact))
    (out / "resolved.yaml").write_text(
        __import__("yaml").safe_dump(config.model_dump(mode="json"), sort_keys=False)
    )
    (out / "builder-prompt.txt").write_text(render_prompt(config))
    return artifact


def load_artifact(path: Path) -> dict:
    artifact = json.loads(path.read_text())
    if artifact["sha256"] != digest(artifact["payload"]):
        raise ValueError("snapshot content hash mismatch")
    return artifact
