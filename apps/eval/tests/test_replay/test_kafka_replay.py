"""Broker-free Kafka replay tests for docs/evaluation-spec.md §3.3."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from harnext_eval.corpus.synthetic import generate_synthetic_corpus
from harnext_eval.replay.kafka_replay import RAW_EVENTS_TOPIC, replay_jsonl
from harnext_eval.types import EvalEvent


class FakeProducer:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.sent: list[tuple[str, bytes]] = []

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def send_and_wait(self, topic: str, value: bytes) -> object:
        self.sent.append((topic, value))
        return object()


def test_fixed_rate_replay_stamps_intended_send_time_without_broker(tmp_path: Path) -> None:
    corpus = generate_synthetic_corpus(tmp_path, seed=3, event_count=3, days=1, entity_count=2)
    producer = FakeProducer()
    clock = [0.0]

    async def advance(seconds: float) -> None:
        clock[0] += seconds

    start = datetime(2026, 8, 30, tzinfo=UTC)
    stats = asyncio.run(
        replay_jsonl(
            corpus.replay_path,
            fixed_rate=2,
            producer=producer,
            now=lambda: start,
            monotonic=lambda: clock[0],
            sleep=advance,
        )
    )

    stamped = [EvalEvent.model_validate_json(value) for _, value in producer.sent]
    assert producer.started and producer.stopped
    assert [topic for topic, _ in producer.sent] == [RAW_EVENTS_TOPIC] * 3
    assert all(event.intended_send_ts is not None for event in stamped)
    assert [
        (event.intended_send_ts - start).total_seconds()
        for event in stamped
        if event.intended_send_ts is not None
    ] == [0, 0.5, 1]
    assert stats.events_sent == 3
