"""Timed JSONL-to-Kafka replay producer for docs/evaluation-spec.md §3.3 and §5."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from harnext_eval.types import EvalEvent

RAW_EVENTS_TOPIC = "cms.events.raw.v1"


class Producer(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def send_and_wait(self, topic: str, value: bytes) -> Awaitable[object]: ...


@dataclass(frozen=True)
class ReplayStats:
    events_sent: int
    first_intended_send_ts: datetime | None
    last_intended_send_ts: datetime | None


async def replay_jsonl(
    path: str | Path,
    *,
    bootstrap_servers: str = "localhost:9092",
    topic: str = RAW_EVENTS_TOPIC,
    speedup: float | None = 60.0,
    fixed_rate: float | None = None,
    producer: Producer | None = None,
    cutoff: datetime | None = None,
    now: Callable[[], datetime] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> ReplayStats:
    """Publish replay events at event-time speed-up or an open-loop fixed rate."""

    if fixed_rate is not None:
        if fixed_rate <= 0:
            raise ValueError("fixed_rate must be positive")
        speedup = None
    elif speedup is None or speedup <= 0:
        raise ValueError("speedup must be positive when fixed_rate is not set")

    events = _read_events(path, cutoff=cutoff)
    if producer is None:
        try:
            from aiokafka import AIOKafkaProducer
        except ImportError as exc:  # pragma: no cover - dependency is optional at runtime
            raise RuntimeError("aiokafka is required for broker-backed replay") from exc
        producer = AIOKafkaProducer(bootstrap_servers=bootstrap_servers)

    wall_now = now or (lambda: datetime.now(UTC))
    wall_start = wall_now()
    mono_start = monotonic()
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    await producer.start()
    try:
        event_origin = events[0].time if events else None
        for index, event in enumerate(events):
            if fixed_rate is not None:
                offset_s = index / fixed_rate
            else:
                assert speedup is not None and event_origin is not None
                offset_s = (event.time - event_origin).total_seconds() / speedup
            intended = wall_start + timedelta(seconds=offset_s)
            delay = mono_start + offset_s - monotonic()
            if delay > 0:
                await sleep(delay)
            stamped = event.model_copy(update={"intended_send_ts": intended})
            await producer.send_and_wait(topic, stamped.model_dump_json().encode("utf-8"))
            first_ts = intended if first_ts is None else first_ts
            last_ts = intended
    finally:
        await producer.stop()

    return ReplayStats(len(events), first_ts, last_ts)


replay_to_kafka = replay_jsonl


def _read_events(path: str | Path, *, cutoff: datetime | None) -> list[EvalEvent]:
    events: list[EvalEvent] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                event = EvalEvent.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"invalid EvalEvent on line {line_number} of {path}") from exc
            if cutoff is not None and event.time > cutoff:
                break
            events.append(event)
    if events != sorted(events, key=lambda event: (event.time, event.id)):
        raise ValueError("replay JSONL must be sorted by event time and id")
    return events
