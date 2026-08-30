"""Probe-set CLI for docs/evaluation-spec.md §5 and §7 E2."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from harnext_eval.probes.common import load_replay, parse_time, validate_period
from harnext_eval.probes.gen_abstention import generate_abstention_probes
from harnext_eval.probes.gen_code_location import generate_code_location_probes
from harnext_eval.probes.gen_extraction import generate_extraction_probes
from harnext_eval.probes.gen_multisource import generate_multisource_probes
from harnext_eval.probes.gen_temporal import generate_temporal_probes
from harnext_eval.probes.gen_update import generate_update_probes
from harnext_eval.types import EvalEvent, Probe


def generate_probe_set(
    events: list[EvalEvent],
    *,
    per_family: int,
    seed: int,
    probe_start: datetime,
    probe_end: datetime,
) -> list[Probe]:
    """Generate the six frozen families, each stratified to ``per_family``."""

    validate_period(probe_start, probe_end)
    if per_family < 0:
        raise ValueError("per-family must be non-negative")
    common = {
        "count": per_family,
        "seed": seed,
        "probe_start": probe_start,
        "probe_end": probe_end,
    }
    probes: list[Probe] = []
    probes.extend(generate_extraction_probes(events, **common))
    probes.extend(generate_temporal_probes(events, **common))
    probes.extend(generate_update_probes(events, **common))
    probes.extend(generate_multisource_probes(events, **common))
    probes.extend(generate_code_location_probes(events, **common))
    probes.extend(generate_abstention_probes(events, **common))
    if len({probe.probe_id for probe in probes}) != len(probes):
        raise ValueError("generated duplicate probe ids")
    return probes


def write_probe_set(probes: Sequence[Probe], output: str | Path) -> tuple[Path, Path, str]:
    """Write canonical JSONL and a conventional SHA-256 sidecar."""

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(probe.model_dump_json() + "\n" for probe in probes).encode()
    output_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = Path(f"{output_path}.sha256")
    sidecar.write_text(f"{digest}  {output_path.name}\n", encoding="utf-8")
    return output_path, sidecar, digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate frozen harnext evaluation probes")
    parser.add_argument("--replay", required=True, type=Path, help="EvalEvent JSONL replay")
    parser.add_argument("--out", required=True, type=Path, help="output probes JSONL")
    parser.add_argument("--per-family", default=60, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--probe-start", required=True)
    parser.add_argument("--probe-end", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        start = parse_time(args.probe_start)
        end = parse_time(args.probe_end)
        events = load_replay(args.replay)
        probes = generate_probe_set(
            events,
            per_family=args.per_family,
            seed=args.seed,
            probe_start=start,
            probe_end=end,
        )
        output, sidecar, digest = write_probe_set(probes, args.out)
    except ValueError as exc:
        _parser().error(str(exc))
    print(f"wrote {len(probes)} probes to {output}")
    print(f"sha256 {digest} ({sidecar})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
