"""Arm G-graphiti: the official Zep/Graphiti package (arXiv 2501.13956), pinned, unmodified.

Extraction, entity/edge resolution, temporal invalidation and hybrid search are all
graphiti-core's own; this module only feeds it the stream (one episode per event, in
order), keeps a resume cursor, measures cost, and freezes a physical snapshot of the
embedded Kuzu database at every checkpoint boundary. See
benchmarks/kafka-500-v2/fidelity/G-graphiti.md for why snapshots, not as-of filters.

    python -m harnext_eval.e2.v2.stores.graphiti --probe 20
    python -m harnext_eval.e2.v2.stores.graphiti --until 2023-06
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
import time
import warnings
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from harnext_eval.e2.v2.errors import PauseRun, is_infra, reset_delay

os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")  # no PostHog events from builds

ROOT = Path(__file__).resolve().parents[5]
ARM = "G-graphiti"
BENCH = ROOT / "benchmarks/kafka-500-v2"
OUT = ROOT / "out/kafka-500-v2"
CONFIG = BENCH / "graphiti.yaml"
MODELS = ROOT.parents[1] / ".harnext/models"
DB = "graph.kuzu"
BACKUP_EVERY = 500  # events between rolling database backups (a killed kuzu can corrupt its file)
BACKUPS_KEPT = 2
FIRST_CHECKPOINT = "2023-01"  # it already holds everything from the corpus start
SMOKE_LIMIT = 30
MAX_FAILURES = 3
DEFAULTS: dict[str, Any] = {
    "version": 1,
    "group_id": "kafka-500-v2",
    "llm": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
    "embedder": {"provider": "fastembed", "model": "BAAI/bge-small-en-v1.5", "dim": 384},
    "reranker": {"provider": "fastembed", "model": "BAAI/bge-reranker-base"},
    "reader": {"facts": "EDGE_HYBRID_SEARCH_RRF", "entities": "NODE_HYBRID_SEARCH_RRF"},
    "max_coroutines": 8,
    "kuzu_buffer_mb": 4096,
    "smoke": False,
}
KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
PROVIDER_ERRORS = ("RateLimit", "APIConnection", "APITimeout", "Authentication", "Permission")


class ConfigError(RuntimeError):
    """The configured providers cannot be honoured; never fall back to another one."""


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical(value) + "\n")
    os.replace(temporary, path)


def load_config(path: Path | None = None) -> dict:
    path = path or CONFIG
    given = yaml.safe_load(path.read_text()) if path.exists() else {}
    config = {
        key: {**default, **(given.get(key) or {})}
        if isinstance(default, dict)
        else given.get(key, default)
        for key, default in DEFAULTS.items()
    }
    if unknown := set(given) - set(DEFAULTS):
        raise ConfigError(f"unknown graphiti.yaml keys: {sorted(unknown)}")
    return config


def parse_time(value: str) -> datetime:
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def boundary(checkpoint: str) -> datetime:
    """Exclusive upper bound of ckpt-YYYY-MM: the first instant of the next month."""
    year, month = map(int, checkpoint.split("-"))
    return datetime(year + month // 12, month % 12 + 1, 1, tzinfo=UTC)


def following(checkpoint: str) -> str:
    return boundary(checkpoint).strftime("%Y-%m")


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


# --- clients: explicit providers only, nothing defaults to OpenAI silently ---


def key_for(provider: str, role: str) -> str:
    name = KEYS[provider]
    if not os.environ.get(name):
        raise ConfigError(
            f"{role} provider is {provider!r} but {name} is not set; refusing to build "
            "G-graphiti with a different provider. Export the key or edit graphiti.yaml."
        )
    return os.environ[name]


def make_embedder(spec: dict):
    from graphiti_core.embedder.client import EmbedderClient

    if spec["provider"] == "openai":
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

        return OpenAIEmbedder(
            OpenAIEmbedderConfig(
                api_key=key_for("openai", "embedder"),
                embedding_model=spec["model"],
                embedding_dim=spec["dim"],
            )
        )
    if spec["provider"] != "fastembed":
        raise ConfigError(f"unsupported embedder provider {spec['provider']!r}")

    class FastEmbedder(EmbedderClient):
        """Local ONNX embedder behind graphiti-core's EmbedderClient interface."""

        def __init__(self):
            from fastembed import TextEmbedding

            self.model = TextEmbedding(model_name=spec["model"], cache_dir=str(MODELS))

        def encode(self, texts: list[str]) -> list[list[float]]:
            return [row.tolist()[: spec["dim"]] for row in self.model.embed(texts)]

        async def create(self, input_data) -> list[float]:
            texts = [input_data] if isinstance(input_data, str) else [str(x) for x in input_data]
            return (await asyncio.to_thread(self.encode, texts[:1]))[0]

        async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
            return await asyncio.to_thread(self.encode, list(input_data_list))

    return FastEmbedder()


def make_reranker(spec: dict):
    from graphiti_core.cross_encoder.client import CrossEncoderClient

    if spec["provider"] == "openai":
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
        from graphiti_core.llm_client.config import LLMConfig

        return OpenAIRerankerClient(
            LLMConfig(api_key=key_for("openai", "reranker"), model=spec["model"])
        )
    if spec["provider"] == "bge":  # graphiti-core's own sentence-transformers client (needs torch)
        from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient

        return BGERerankerClient()
    if spec["provider"] != "fastembed":
        raise ConfigError(f"unsupported reranker provider {spec['provider']!r}")

    class FastReranker(CrossEncoderClient):
        """Local ONNX cross-encoder; loaded on first use (RRF recipes never call it)."""

        model = None

        def score(self, query: str, passages: list[str]) -> list[float]:
            if self.model is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                self.model = TextCrossEncoder(spec["model"], cache_dir=str(MODELS))
            return [float(score) for score in self.model.rerank(query, passages)]

        async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
            if not passages:
                return []
            scores = await asyncio.to_thread(self.score, query, passages)
            return sorted(zip(passages, scores, strict=True), key=lambda pair: -pair[1])

    return FastReranker()


def make_llm(config: dict):
    from graphiti_core.llm_client.config import LLMConfig

    spec = config["llm"]
    settings = LLMConfig(
        api_key=key_for(spec["provider"], "llm") if spec["provider"] in KEYS else None,
        model=spec["model"],
        small_model=spec.get("small_model", spec["model"]),
        **{k: spec[k] for k in ("temperature", "max_tokens") if k in spec},
    )
    if spec["provider"] == "anthropic":
        from graphiti_core.llm_client.anthropic_client import AnthropicClient

        return AnthropicClient(settings)
    if spec["provider"] == "openai":
        if not config["smoke"]:
            raise ConfigError(
                "llm provider 'openai' is a smoke configuration only: the decided builder is "
                f"Claude Haiku 4.5. Set smoke: true (caps the build at {SMOKE_LIMIT} episodes)."
            )
        from graphiti_core.llm_client.openai_client import OpenAIClient

        return OpenAIClient(settings)
    if spec["provider"] == "ollama":
        # Local open-weights model through Ollama's NATIVE chat API: the OpenAI-compatible
        # endpoint cannot switch Qwen3's thinking off, so the whole token budget goes to a
        # reasoning trace and the JSON body comes back empty. Deviation from the paper's
        # GPT-4o-class builder: see fidelity/G-graphiti.md.
        import httpx
        from graphiti_core.llm_client.client import LLMClient
        from graphiti_core.llm_client.errors import EmptyResponseError

        base = spec.get("base_url", "http://127.0.0.1:11434")
        max_tokens_default = spec.get("max_tokens", 4096)

        class OllamaClient(LLMClient):
            def __init__(self):
                super().__init__(settings, cache=False)
                self.http = httpx.AsyncClient(base_url=base, timeout=spec.get("timeout_s", 600))
                self.calls_total = self.input_total = self.output_total = 0

            async def _generate_response(
                self, messages, response_model=None, max_tokens=None, model_size=None
            ):
                body = {
                    "model": settings.model,
                    "messages": [{"role": m.role, "content": m.content} for m in messages],
                    "stream": False,
                    "think": not spec.get("no_think", True),
                    "format": response_model.model_json_schema() if response_model else "json",
                    "options": {
                        "num_predict": max_tokens or max_tokens_default,
                        "temperature": settings.temperature,
                        "num_ctx": spec.get("num_ctx", 32768),
                    },
                }
                reply = await self.http.post("/api/chat", json=body)
                reply.raise_for_status()
                data = reply.json()
                self.calls_total += 1
                self.input_total += data.get("prompt_eval_count", 0)
                self.output_total += data.get("eval_count", 0)
                text = (data.get("message") or {}).get("content") or ""
                if not text.strip():
                    raise EmptyResponseError("LLM returned an empty response")
                return json.loads(text)

        return OllamaClient()
    raise ConfigError(f"unsupported llm provider {spec['provider']!r}")


def no_llm():
    """Readers never call a model: search_ is embeddings + BM25 + graph traversal only."""
    from graphiti_core.llm_client.client import LLMClient
    from graphiti_core.llm_client.config import LLMConfig

    class NoLLM(LLMClient):
        async def _generate_response(self, *_, **__):
            raise RuntimeError("the read-only Graphiti checkpoint has no LLM client")

    return NoLLM(LLMConfig(api_key="none", model="none"), cache=False)


@contextmanager
def kuzu_mode(read_only: bool, buffer_mb: int):
    """KuzuDriver exposes neither read-only mode nor the buffer pool (kuzu's default pool is
    80 % of RAM per open database), so its kuzu.Database call is parameterised here."""
    import kuzu  # the same module object graphiti_core.driver.kuzu_driver calls into

    original = kuzu.Database
    kuzu.Database = lambda path: original(  # type: ignore[assignment,misc]
        path, read_only=read_only, buffer_pool_size=buffer_mb << 20
    )
    try:
        yield
    finally:
        kuzu.Database = original  # type: ignore[misc]


def open_graph(db: Path, config: dict, *, read_only: bool = False, clients=None):
    """The package's Graphiti over an embedded Kuzu file. `clients` = (llm, embedder, reranker)
    overrides the configured ones (tests inject fakes)."""
    from graphiti_core import Graphiti
    from graphiti_core.driver.kuzu_driver import KuzuDriver

    class ReadOnlyKuzu(KuzuDriver):
        def setup_schema(self):  # the schema is already in the snapshot
            pass

    llm, embedder, reranker = clients or (
        no_llm() if read_only else make_llm(config),
        make_embedder(config["embedder"]),
        make_reranker(config["reranker"]),
    )
    with (
        warnings.catch_warnings(),
        kuzu_mode(read_only, 512 if read_only else config["kuzu_buffer_mb"]),
    ):
        warnings.simplefilter("ignore", DeprecationWarning)
        driver = (ReadOnlyKuzu if read_only else KuzuDriver)(str(db))
    # graphiti-core 0.30.2 reads GraphDriver._database in add_episode's request scoping but its
    # KuzuDriver never sets it (AttributeError). Naming it after our only group keeps the
    # package on its shared-driver path; nothing else in the pipeline is touched.
    driver._database = config["group_id"]
    return Graphiti(
        graph_driver=driver,
        llm_client=llm,
        embedder=embedder,
        cross_encoder=reranker,
        max_coroutines=config["max_coroutines"],
    )


async def build_indices(graph) -> None:
    """Graphiti.build_indices_and_constraints() reaches KuzuDriver's no-op in 0.30.2, so the
    full-text indices every search needs are never made; run the package's own Kuzu
    maintenance operation (the four CREATE_FTS_INDEX statements) once per database."""
    from graphiti_core.graph_queries import get_fulltext_indices

    rows, _, _ = await graph.driver.execute_query("CALL SHOW_INDEXES() RETURN *")
    present = {str(value) for row in rows or [] for value in row.values()}
    for query in get_fulltext_indices(graph.driver.provider):
        if not any(f"'{name}'" in query for name in present):
            await graph.driver.execute_query(query)


async def flush(graph) -> None:
    """Merge the write-ahead log into the database file so the file alone is the whole graph.
    The handles are never closed explicitly: KuzuDriver.close() is a no-op upstream, and kuzu
    0.11.3 segfaults when a QueryResult is finalized after its database was closed."""
    if not getattr(graph.driver.db, "read_only", False):
        await graph.driver.execute_query("CHECKPOINT;")


# --- build ---


def usage_of(llm) -> dict:
    if hasattr(llm, "calls_total"):  # the native Ollama client counts for itself
        return {
            "calls": llm.calls_total,
            "input_tokens": llm.input_total,
            "output_tokens": llm.output_total,
        }
    tracker = getattr(llm, "token_tracker", None)
    if tracker is None:
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0}
    rows = tracker.get_usage().values()
    return {
        "calls": sum(row.call_count for row in rows),
        "input_tokens": sum(row.total_input_tokens for row in rows),
        "output_tokens": sum(row.total_output_tokens for row in rows),
    }


def provider_failure(exc: BaseException) -> bool:
    chain, seen = [], exc
    while seen is not None and seen not in chain:
        chain.append(seen)
        seen = seen.__cause__ or seen.__context__
    return any(
        any(mark in type(item).__name__ for mark in PROVIDER_ERRORS) or is_infra(str(item))
        for item in chain
    )


class GraphitiStore:
    arm = ARM

    def __init__(self, config: dict | None = None, *, clients=None):
        self.config = config or load_config()
        self.clients = clients
        self.failures = 0

    def identity(self) -> str:
        from importlib.metadata import version

        pinned = {name: version(name) for name in ("graphiti-core", "kuzu")}
        return hashlib.sha256(canonical([self.config, pinned]).encode()).hexdigest()

    def build(
        self,
        events: Iterable[dict],
        out: Path,
        *,
        until: str | None = None,
        limit: int | None = None,
    ) -> dict:
        """Resumable. `until` is a checkpoint (YYYY-MM, inclusive) or an ISO instant (exclusive).
        `limit` caps the episodes ingested by this call (probe/smoke). Returns the state."""
        out.mkdir(parents=True, exist_ok=True)
        return asyncio.run(self._build(events, out, until, limit))

    async def _build(self, events, out: Path, until, limit) -> dict:
        path = out / "build-state.json"
        state = (
            json.loads(path.read_text())
            if path.exists()
            else {
                "arm": ARM,
                "identity": self.identity(),
                "cursor": 0,
                "last_id": None,
                "checkpoints": [],
                "skipped": 0,
                "totals": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0},
            }
        )
        if state["identity"] != self.identity():
            raise ConfigError("graphiti.yaml or the pinned packages changed since this build began")
        if self.config["smoke"]:
            limit = min(limit or SMOKE_LIMIT, SMOKE_LIMIT - state["cursor"])
        stop = None if until is None else boundary(until) if len(until) == 7 else parse_time(until)
        pending = following(state["checkpoints"][-1]) if state["checkpoints"] else FIRST_CHECKPOINT
        graph = open_graph(out / DB, self.config, clients=self.clients)
        await build_indices(graph)
        done, previous, complete, latest, resume = 0, None, True, None, state["cursor"]
        try:
            for index, event in enumerate(events):
                moment = parse_time(event["time"])
                if index < resume:
                    previous, latest = event, moment
                    continue
                if index == resume and previous and previous["id"] != state["last_id"]:
                    raise ConfigError("the stream differs from the one this build was resumed on")
                if stop and moment >= stop:
                    break
                if limit is not None and done >= limit:
                    complete = False
                    break
                while moment >= boundary(pending):
                    graph = await self.snapshot(graph, out, pending, state)
                    pending = following(pending)
                await self.ingest(graph, event, index, state, out)
                done, latest = done + 1, moment
                if state["cursor"] % BACKUP_EVERY == 0:
                    await self.backup(graph, out, state)
            # Everything below `stop` (else the whole stream) is in: close the months it covers.
            end = stop or (boundary(latest.strftime("%Y-%m")) if latest else None)
            while complete and end and boundary(pending) <= end:
                graph = await self.snapshot(graph, out, pending, state)
                pending = following(pending)
        finally:
            save(path, state)
            await flush(graph)
        return state

    async def backup(self, graph, out: Path, state: dict) -> None:
        """Rolling copy of the flushed database with the cursor it belongs to. A process killed
        mid-write (OOM killer, power) can leave kuzu unopenable; `restore_latest` rewinds to
        the newest copy instead of losing weeks of ingestion."""
        await flush(graph)
        target = out / "backups" / f"{state['cursor']:09d}"
        staging = target.with_name(target.name + ".staging")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        shutil.copy2(out / DB, staging / DB)
        save(staging / "build-state.json", state)
        staging.replace(target)
        for old in sorted(p for p in target.parent.iterdir() if p.name.isdigit())[:-BACKUPS_KEPT]:
            shutil.rmtree(old, ignore_errors=True)

    async def ingest(self, graph, event: dict, index: int, state: dict, out: Path) -> None:
        from graphiti_core.nodes import EpisodeType

        before, start, error = usage_of(graph.llm_client), time.monotonic(), None
        for _attempt in range(2):  # one retry: extraction by a small local model is flaky
            try:
                await self.forget(graph, event["id"])  # a crash mid-episode leaves a partial one
                await graph.add_episode(
                    name=event["id"],
                    episode_body=event["text"],
                    source_description=event["type"],
                    reference_time=parse_time(event["time"]),
                    source=EpisodeType.text,
                    group_id=self.config["group_id"],
                )
                error = None
                break
            except Exception as exc:
                if provider_failure(exc):
                    save(out / "build-state.json", state)
                    raise PauseRun(f"{ARM} paused at event {index} ({event['id']}): {exc}") from exc
                error = f"{type(exc).__name__}: {exc}"[:500]
        if error:
            # Never drop an event silently: keep it verbatim like the file arms' _unprocessed/.
            with (out / "unprocessed.jsonl").open("a") as sink:
                sink.write(canonical({**event, "error": error}) + "\n")
            state["skipped"] += 1
            self.failures += 1
            if self.failures >= MAX_FAILURES:  # a systemic fault, not one odd episode
                save(out / "build-state.json", state)
                raise RuntimeError(f"{ARM}: {MAX_FAILURES} consecutive failures: {error}")
        else:
            self.failures = 0
        after = usage_of(graph.llm_client)
        row = {
            "index": index,
            "id": event["id"],
            "time": event["time"],
            "characters": len(event["text"]),
            "seconds": round(time.monotonic() - start, 3),
            **{key: after[key] - before[key] for key in after},
            "error": error,
        }
        with (out / "build-log.jsonl").open("a") as stream:
            stream.write(canonical(row) + "\n")
        for key in ("calls", "input_tokens", "output_tokens", "seconds"):
            state["totals"][key] = round(state["totals"][key] + row[key], 3)
        state["cursor"], state["last_id"] = index + 1, event["id"]
        save(out / "build-state.json", state)

    async def forget(self, graph, name: str) -> None:
        rows, _, _ = await graph.driver.execute_query(
            "MATCH (e:Episodic) WHERE e.name = $name RETURN e.uuid AS uuid", name=name
        )
        for row in rows or []:
            await graph.remove_episode(row["uuid"])

    async def snapshot(self, graph, out: Path, checkpoint: str, state: dict):
        """Freeze the database as it is now — every episode below the boundary, none above."""
        rows, _, _ = await graph.driver.execute_query(
            "MATCH (e:Episodic) RETURN count(e) AS episodes, max(e.valid_at) AS latest"
        )
        await flush(graph)  # sequential build: nothing writes while the file is copied
        target = out / "export" / checkpoint
        staging = target.with_name(f".{checkpoint}.partial")
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(target, ignore_errors=True)
        staging.mkdir(parents=True)
        shutil.copy2(out / DB, staging / DB)
        latest = rows[0]["latest"] if rows else None
        save(
            staging / "meta.json",
            {
                "arm": ARM,
                "checkpoint": checkpoint,
                "boundary": boundary(checkpoint).isoformat(),
                "events": state["cursor"],
                "episodes": rows[0]["episodes"] if rows else 0,
                "latest_episode": latest.isoformat() if isinstance(latest, datetime) else latest,
                "sha256": file_sha(staging / DB),
                "identity": state["identity"],
                "config": self.config,
            },
        )
        os.replace(staging, target)
        state["checkpoints"].append(checkpoint)
        save(out / "build-state.json", state)
        return graph

    def export(self, out: Path, checkpoint: str) -> Path:
        """Snapshots are cut during the build (a later graph cannot be rewound); verify one."""
        target = out / "export" / checkpoint
        if not (target / "meta.json").exists():
            raise FileNotFoundError(f"{ARM} has no snapshot for {checkpoint}; build up to it first")
        meta = json.loads((target / "meta.json").read_text())
        if file_sha(target / DB) != meta["sha256"]:
            raise ValueError(f"{ARM} snapshot {checkpoint} does not match its recorded hash")
        if meta["latest_episode"] and parse_time(meta["latest_episode"]) >= boundary(checkpoint):
            raise ValueError(f"{ARM} snapshot {checkpoint} holds an episode past its boundary")
        return target


# --- reads (used by mcp_server) ---


class Reader:
    """The package's own search over one frozen snapshot. No LLM, no writes."""

    def __init__(self, export: Path, *, clients=None):
        self.meta = json.loads((export / "meta.json").read_text())
        self.config = self.meta["config"]
        self.group = [self.config["group_id"]]
        self.graph = open_graph(export / DB, self.config, read_only=True, clients=clients)

    async def search(self, query: str, k: int, recipe: str):
        from graphiti_core.search import search_config_recipes as recipes

        if not recipe.isupper() or not hasattr(recipes, recipe):
            raise ValueError(f"unknown graphiti search recipe {recipe}")
        config = getattr(recipes, recipe).model_copy(deep=True)
        config.limit = k
        return await self.graph.search_(query, config=config, group_ids=self.group)

    async def facts(self, query: str, k: int) -> list[dict]:
        found = await self.search(query, k, self.config["reader"]["facts"])
        return [edge_view(edge) for edge in found.edges]

    async def entities(self, query: str, k: int) -> list[dict]:
        found = await self.search(query, k, self.config["reader"]["entities"])
        return [
            {"uuid": n.uuid, "name": n.name, "labels": n.labels, "text": n.summary}
            for n in found.nodes
        ]

    async def entity_edges(self, uuid: str) -> list[dict]:
        from graphiti_core.edges import EntityEdge

        edges = await EntityEdge.get_by_node_uuid(self.graph.driver, uuid)
        return [edge_view(edge) for edge in sorted(edges, key=lambda e: str(e.valid_at or ""))]

    async def episode(self, key: str) -> list[dict]:
        rows, _, _ = await self.graph.driver.execute_query(
            "MATCH (e:Episodic) WHERE e.name = $key OR e.uuid = $key "
            "RETURN e.uuid AS uuid, e.name AS name, e.source_description AS type, "
            "e.valid_at AS time, e.content AS text",
            key=key,
        )
        return [
            {**row, "time": row["time"].isoformat() if row.get("time") else None}
            for row in rows or []
        ]


def edge_view(edge) -> dict:
    stamp = lambda value: value.isoformat() if value else None  # noqa: E731
    return {
        "uuid": edge.uuid,
        "relation": edge.name,
        "source_entity": edge.source_node_uuid,
        "target_entity": edge.target_node_uuid,
        "valid_at": stamp(edge.valid_at),
        "invalid_at": stamp(edge.invalid_at),
        "superseded": edge.expired_at is not None,
        "episodes": edge.episodes,
        "text": edge.fact,
    }


# --- CLI ---


def stream(path: Path):
    with path.open() as lines:
        for line in lines:
            if line.strip():
                yield json.loads(line)


def report(rows: list[dict]) -> dict:
    count = max(len(rows), 1)
    mean = lambda key: round(sum(row[key] for row in rows) / count, 2)  # noqa: E731
    return {
        "episodes": len(rows),
        "errors": sum(bool(row["error"]) for row in rows),
        "llm_calls_per_episode": mean("calls"),
        "input_tokens_per_episode": mean("input_tokens"),
        "output_tokens_per_episode": mean("output_tokens"),
        "seconds_per_episode": mean("seconds"),
        "characters_per_episode": mean("characters"),
    }


def restore_latest(out: Path) -> int | None:
    """Put the newest rolling backup back (database, cursor, build log) and return its cursor;
    with no backup, clear the store so the build starts from the first event. Used by the
    launcher after a crash on open (kuzu segfaults on a file torn by a kill)."""
    backups = (
        sorted(p for p in (out / "backups").glob("*") if p.name.isdigit() and (p / DB).exists())
        if (out / "backups").exists()
        else []
    )
    for stale in out.glob(DB + "*"):
        stale.unlink()
    log = out / "build-log.jsonl"
    if not backups:
        for name in ("build-state.json", "build-log.jsonl"):
            (out / name).unlink(missing_ok=True)
        return None
    latest = backups[-1]
    shutil.copy2(latest / DB, out / DB)
    shutil.copy2(latest / "build-state.json", out / "build-state.json")
    cursor = json.loads((out / "build-state.json").read_text())["cursor"]
    if log.exists():
        rows = [r for r in log.read_text().splitlines() if json.loads(r)["index"] < cursor]
        log.write_text("".join(r + "\n" for r in rows))
    return cursor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=OUT / "stream/events.jsonl")
    parser.add_argument("--out", type=Path, default=OUT / "stores" / ARM)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--until", help="checkpoint YYYY-MM (inclusive) or ISO instant (exclusive)")
    parser.add_argument("--probe", type=int, help="ingest N more episodes and print unit costs")
    parser.add_argument(
        "--no-wait", action="store_true", help="exit on a provider pause instead of waiting"
    )
    parser.add_argument(
        "--restore-latest",
        action="store_true",
        help="rewind the store to its newest rolling backup (or to empty) and exit",
    )
    args = parser.parse_args()
    if args.restore_latest:
        print(canonical({"restored_cursor": restore_latest(args.out)}))
        return
    store = GraphitiStore(load_config(args.config))
    log = args.out / "build-log.jsonl"
    seen = len(log.read_text().splitlines()) if log.exists() else 0
    while True:
        try:
            state = store.build(stream(args.events), args.out, until=args.until, limit=args.probe)
            break
        except ConfigError as exc:
            raise SystemExit(f"ConfigError: {exc}") from exc
        except PauseRun as exc:
            if args.no_wait:
                raise SystemExit(f"PauseRun: {exc}") from exc
            delay = reset_delay(str(exc))
            print(f"PAUSED: {exc}\nwaiting {delay // 60} min, then resuming from the cursor",
                  file=sys.stderr, flush=True)  # fmt: skip
            time.sleep(delay)
    rows = (
        [json.loads(line) for line in log.read_text().splitlines()[seen:]] if log.exists() else []
    )
    print(canonical({"state": state, "this_run": report(rows), "config": store.config["llm"]}))


if __name__ == "__main__":
    main()
