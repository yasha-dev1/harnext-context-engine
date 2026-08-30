"""Deterministic fake-fold accounting for docs/evaluation-spec.md §7 E5."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from harnext_eval.types import EvalEvent

FAKE_INPUT_PER_MILLION = 1.0
FAKE_OUTPUT_PER_MILLION = 4.0
_INSTRUCTION_OVERHEAD_BYTES = 1_024


@dataclass(frozen=True, slots=True)
class FakeFoldUsage:
    """Frozen token and price estimate derived only from observable byte counts."""

    input_tokens: int
    output_tokens: int
    cost_usd: float
    instruction_bytes: int
    files_read_bytes: int
    bytes_written: int


def estimate_fake_fold_usage(
    events: list[EvalEvent],
    *,
    files_read_bytes: int = 0,
    bytes_written: int | None = None,
) -> FakeFoldUsage:
    """Price one fold from instruction/read bytes and deterministic output bytes.

    The protocol/schema prefix is paid once per fold. Event JSON is the variable
    instruction body. In the offline projection, written bytes are conservatively
    modelled as the canonical event JSON bytes unless the caller has an exact
    byte count. The fixed fake prices are deliberately not vendor prices.
    """

    if files_read_bytes < 0 or (bytes_written is not None and bytes_written < 0):
        raise ValueError("byte counts cannot be negative")
    event_bytes = sum(len(event.model_dump_json().encode()) for event in events)
    instruction_bytes = _INSTRUCTION_OVERHEAD_BYTES + event_bytes
    output_bytes = event_bytes if bytes_written is None else bytes_written
    input_tokens = max(1, math.ceil((instruction_bytes + files_read_bytes) / 4))
    output_tokens = max(1, math.ceil(output_bytes / 4))
    cost_usd = (
        input_tokens * FAKE_INPUT_PER_MILLION
        + output_tokens * FAKE_OUTPUT_PER_MILLION
    ) / 1_000_000
    return FakeFoldUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        instruction_bytes=instruction_bytes,
        files_read_bytes=files_read_bytes,
        bytes_written=output_bytes,
    )


def fake_fold_token_counts(events: list[EvalEvent]) -> tuple[int, int]:
    """Compatibility-sized tuple for the E5 usage JSONL writer."""

    usage = estimate_fake_fold_usage(events)
    return usage.input_tokens, usage.output_tokens


def ensure_fake_fold_usage(
    path: Path,
    prior_records: int,
    events: list[EvalEvent],
    lane: str,
    layout: str,
) -> bool:
    """Append one fake-provider row only when the layout emitted none itself."""

    current_records = 0
    if path.exists():
        current_records = sum(bool(line.strip()) for line in path.read_text().splitlines())
    if current_records > prior_records:
        return False
    usage = estimate_fake_fold_usage(events)
    row = {
        "cost_usd": usage.cost_usd,
        "event_count": len(events),
        "event_ids": [event.id for event in events],
        "harness": "fake",
        "input_tokens": usage.input_tokens,
        "lane": lane,
        "layout": layout,
        "model": "fake",
        "output_tokens": usage.output_tokens,
        "status": "success",
        "tokens_in": usage.input_tokens,
        "tokens_out": usage.output_tokens,
        "total_cost_usd": usage.cost_usd,
        "usage": {
            "instruction_bytes": usage.instruction_bytes,
            "files_read_bytes": usage.files_read_bytes,
            "bytes_written": usage.bytes_written,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(row, sort_keys=True) + "\n")
    return True
