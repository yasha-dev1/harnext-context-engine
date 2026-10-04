"""G-graphiti without a network: the real graphiti-core pipeline on embedded Kuzu, driven by a
scripted LLM and a hashing embedder."""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

pytest.importorskip("graphiti_core")
pytest.importorskip("kuzu")

from graphiti_core.cross_encoder.client import CrossEncoderClient  # noqa: E402
from graphiti_core.embedder.client import EmbedderClient  # noqa: E402
from graphiti_core.llm_client.client import LLMClient  # noqa: E402
from graphiti_core.llm_client.config import LLMConfig  # noqa: E402
from harnext_eval.e2.v2.errors import PauseRun  # noqa: E402
from harnext_eval.e2.v2.stores import graphiti as store  # noqa: E402

CONFIG = {**store.DEFAULTS, "llm": {"provider": "fake", "model": "fake"}, "kuzu_buffer_mb": 256}
store.FIRST_CHECKPOINT = "2023-06"  # keep the fixture short; production starts at 2023-01


class FakeEmbedder(EmbedderClient):
    def vector(self, text: str) -> list[float]:
        return [byte / 255 for byte in hashlib.sha256(text.encode()).digest()[:16]]

    async def create(self, input_data) -> list[float]:
        return self.vector(input_data if isinstance(input_data, str) else str(input_data))

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return [self.vector(text) for text in input_data_list]


class FakeReranker(CrossEncoderClient):
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        return [(passage, 1.0) for passage in passages]


class FakeLLM(LLMClient):
    """Fills whatever response model graphiti asks for: one entity named after the episode's
    marker word, no edges. `fail_at` raises a provider-style 429 on the n-th episode."""

    def __init__(self, fail_at: int | None = None):
        super().__init__(LLMConfig(api_key="none", model="fake"), cache=False)
        self.fail_at, self.episodes = fail_at, 0

    async def _generate_response(self, messages, response_model=None, *_, **__):
        text = "\n".join(message.content for message in messages)
        fields = response_model.model_fields if response_model else {}
        if "extracted_entities" in fields:
            self.episodes += 1
            if self.fail_at is not None and self.episodes == self.fail_at:
                raise RuntimeError("Error code: 429 - rate limit exceeded for this organization")
            marker = next((word for word in text.split() if word.startswith("MARK")), "MARK-none")
            self.token_tracker.record("extract_nodes", 100, 10)
            return {"extracted_entities": [{"name": marker.strip(".,"), "entity_type_id": 0}]}
        return {
            name: [] if "list" in str(field.annotation).lower() else None
            for name, field in fields.items()
        }


def events():
    days = ["2023-05-10", "2023-06-10", "2023-06-20", "2023-07-05", "2023-08-02"]
    return [
        {
            "id": f"e{i}",
            "time": f"{day}T10:00:00Z",
            "type": "jira.comment",
            "entities": [],
            "actor": None,
            "title": f"event {i}",
            "text": f"Event number {i} mentions MARK{i} only.",
            "meta": {},
        }
        for i, day in enumerate(days)
    ]


def build(tmp_path, llm, **kwargs):
    builder = store.GraphitiStore(CONFIG, clients=(llm, FakeEmbedder(), FakeReranker()))
    return builder.build(events(), tmp_path, **kwargs)


def episodes(export):
    reader = store.Reader(export, clients=(store.no_llm(), FakeEmbedder(), FakeReranker()))

    async def names():
        rows, _, _ = await reader.graph.driver.execute_query(
            "MATCH (e:Episodic) RETURN e.name AS name ORDER BY name"
        )
        return [row["name"] for row in rows], reader

    return asyncio.run(names())


def test_boundary():
    assert store.boundary("2023-06").isoformat() == "2023-07-01T00:00:00+00:00"
    assert store.following("2023-12") == "2024-01"


def test_missing_anthropic_key_refuses(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "present")
    with pytest.raises(store.ConfigError, match="ANTHROPIC_API_KEY is not set"):
        store.make_llm(store.DEFAULTS)


def test_openai_llm_only_as_smoke(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "present")
    config = {**store.DEFAULTS, "llm": {"provider": "openai", "model": "gpt-4.1-mini"}}
    with pytest.raises(store.ConfigError, match="smoke configuration only"):
        store.make_llm(config)


def test_probe_then_resume_has_no_duplicates(tmp_path):
    llm = FakeLLM()
    state = build(tmp_path, llm, limit=2)
    assert (state["cursor"], state["last_id"], state["checkpoints"]) == (2, "e1", [])
    state = build(tmp_path, llm)
    assert state["cursor"] == 5 and state["skipped"] == 0
    assert state["checkpoints"] == ["2023-06", "2023-07", "2023-08"]
    log = [json.loads(line) for line in (tmp_path / "build-log.jsonl").read_text().splitlines()]
    assert [row["id"] for row in log] == ["e0", "e1", "e2", "e3", "e4"]
    assert all(row["calls"] >= 1 and row["input_tokens"] >= 100 for row in log)


def test_rate_limit_pauses_with_cursor_saved(tmp_path):
    with pytest.raises(PauseRun, match="e2"):
        build(tmp_path, FakeLLM(fail_at=3))
    state = json.loads((tmp_path / "build-state.json").read_text())
    assert (state["cursor"], state["last_id"], state["skipped"]) == (2, "e1", 0)
    state = build(tmp_path, FakeLLM())  # resume retries e2, which the pause never recorded
    assert state["cursor"] == 5
    names, _ = episodes(store.GraphitiStore(CONFIG).export(tmp_path, "2023-08"))
    assert names == ["e0", "e1", "e2", "e3", "e4"]


def test_changed_stream_is_refused(tmp_path):
    build(tmp_path, FakeLLM(), limit=2)
    changed = events()
    changed[1]["id"] = "other"
    builder = store.GraphitiStore(CONFIG, clients=(FakeLLM(), FakeEmbedder(), FakeReranker()))
    with pytest.raises(store.ConfigError, match="stream differs"):
        builder.build(changed, tmp_path)


def test_checkpoint_holds_nothing_past_its_boundary(tmp_path):
    build(tmp_path, FakeLLM())
    builder = store.GraphitiStore(CONFIG)
    export = builder.export(tmp_path, "2023-06")
    meta = json.loads((export / "meta.json").read_text())
    assert meta["events"] == 3 and meta["latest_episode"] < "2023-07-01"
    names, reader = episodes(export)
    assert names == ["e0", "e1", "e2"]

    async def probe():
        late = await reader.entities("MARK3", 10) + await reader.entities("MARK4", 10)
        return (
            [item["name"] for item in late],
            await reader.episode("e3"),
            await reader.episode("e1"),
            await reader.facts("MARK4", 5),
        )

    late, missing, present, facts = asyncio.run(probe())
    assert not {"MARK3", "MARK4"} & set(late)
    assert missing == [] and facts == []
    assert present[0]["text"].startswith("Event number 1") and present[0]["type"] == "jira.comment"


def test_until_stops_and_export_validates(tmp_path):
    state = build(tmp_path, FakeLLM(), until="2023-06")
    assert state["cursor"] == 3 and state["checkpoints"] == ["2023-06"]
    builder = store.GraphitiStore(CONFIG)
    with pytest.raises(FileNotFoundError):
        builder.export(tmp_path, "2023-07")
    with (tmp_path / "export/2023-06" / store.DB).open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="recorded hash"):
        builder.export(tmp_path, "2023-06")
