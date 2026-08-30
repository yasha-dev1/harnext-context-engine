"""Dedicated deployment CLI for evaluation spec §7 E6."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from harnext_eval.config import load_config
from harnext_eval.e6.loadgen import fit_workload, situations_from_meta
from harnext_eval.e6.run import BenchmarkConfig, run_benchmark
from harnext_eval.types import EvalEvent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the paired E6 systems benchmark")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "research"), default="smoke")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--situations-json", type=Path)
    parser.add_argument("--steady-duration-s", type=float)
    parser.add_argument("--burst-duration-s", type=float)
    parser.add_argument("--burst-window-s", type=float)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--bootstrap-resamples", type=int)
    parser.add_argument("--kafka-bootstrap-servers")
    parser.add_argument("--kafka-output-topic")
    parser.add_argument("--load-generator-host")
    parser.add_argument("--kafka-telemetry-path", type=Path)
    return parser


def _events(path: Path) -> list[EvalEvent]:
    with path.open(encoding="utf-8") as source:
        return [EvalEvent.model_validate_json(line) for line in source if line.strip()]


def _research_settings(args: argparse.Namespace, fit: Any) -> BenchmarkConfig:
    required = {
        "--kafka-bootstrap-servers": args.kafka_bootstrap_servers,
        "--kafka-output-topic": args.kafka_output_topic,
        "--load-generator-host": args.load_generator_host,
        "--kafka-telemetry-path": args.kafka_telemetry_path,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(f"research profile requires: {', '.join(missing)}")
    return BenchmarkConfig.research(
        fit,
        kafka_bootstrap_servers=args.kafka_bootstrap_servers,
        kafka_output_topic=args.kafka_output_topic,
        load_generator_host=args.load_generator_host,
        kafka_telemetry_path=args.kafka_telemetry_path,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    experiment = load_config(args.config)
    fit = fit_workload(_events(args.replay))
    settings = (
        BenchmarkConfig.smoke()
        if args.profile == "smoke"
        else _research_settings(args, fit)
    )
    overrides = {
        name: value
        for name, value in {
            "steady_duration_s": args.steady_duration_s,
            "burst_duration_s": args.burst_duration_s,
            "burst_window_s": args.burst_window_s,
            "repetitions": args.repetitions,
            "bootstrap_resamples": args.bootstrap_resamples,
        }.items()
        if value is not None
    }
    if overrides:
        settings = replace(settings, **overrides)
    meta: dict[str, Any] | None = None
    if args.situations_json is not None:
        meta = json.loads(args.situations_json.read_text(encoding="utf-8"))
    catalogue = situations_from_meta(fit, meta)
    asyncio.run(
        run_benchmark(
            fit,
            experiment.engine,
            args.out,
            seed=args.seed,
            benchmark=settings,
            situations=catalogue,
        )
    )


if __name__ == "__main__":
    main()
