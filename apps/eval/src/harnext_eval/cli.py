"""Command-line entry point for docs/evaluation-spec.md §11."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
import yaml

from harnext_eval.config import ExperimentConfig, load_config
from harnext_eval.corpus import CorpusHandle
from harnext_eval.corpus.synthetic import generate_synthetic_corpus
from harnext_eval.manifest import build_manifest, write_manifest
from harnext_eval.probes.common import load_replay, parse_time
from harnext_eval.probes.gen import generate_probe_set, write_probe_set
from harnext_eval.probes.gold import GoldAuditTrail, write_gold_report
from harnext_eval.providers.factory import (
    assert_offline_ok,
    make_embeddings,
    make_harness_name,
    make_llm,
    provider_summary,
)
from harnext_eval.registry import ExperimentResult, get_experiment, list_experiments
from harnext_eval.replay.driver import DriverStats, run_pipeline
from harnext_eval.report import build_report
from harnext_eval.stores.base import StoreHandle
from harnext_eval.stores.layouts import configure_store
from harnext_eval.types import EvalEvent

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


def _discover_experiments() -> None:
    """Import each real registry adapter exactly once."""

    from harnext_eval.e1 import run as _e1  # noqa: F401
    from harnext_eval.e2 import run as _e2  # noqa: F401
    from harnext_eval.e3 import run as _e3  # noqa: F401
    from harnext_eval.e4 import run as _e4  # noqa: F401
    from harnext_eval.e5 import run as _e5  # noqa: F401
    from harnext_eval.e6 import run as _e6  # noqa: F401


def _check_results(metrics: dict[str, float]) -> dict[str, bool | dict[str, object]]:
    count_metrics = {
        "checks.leakage_gate_passed",
        "checks.leakage_gate_failed",
        "checks.tasks_accepted",
    }
    checks: dict[str, bool | dict[str, object]] = {}
    for name, value in metrics.items():
        if not name.startswith(("check.", "checks.")) or name in count_metrics:
            continue
        check_name = name.split(".", 1)[1]
        checks[check_name] = (
            {"passed": None, "value": "not-applicable", "reason": "metric is undefined"}
            if isinstance(value, float) and not math.isfinite(value)
            else bool(value)
        )
    return checks


def _write_result(result: ExperimentResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    checks = _check_results(result.metrics)
    checks.update(result.check_details)
    payload = {
        "name": result.name,
        "metrics": result.metrics,
        "checks": checks,
        "tables": {name: table.to_dict(orient="records") for name, table in result.tables.items()},
        "artifacts": [str(path) for path in result.artifacts],
        "primary": result.primary,
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _handle_for_replay(path: Path, name: str) -> CorpusHandle:
    events = load_replay(path)
    if not events:
        raise typer.BadParameter("the replay is empty", param_hint="--replay")
    return CorpusHandle(
        name=name,
        replay_path=path,
        probes_path=None,
        tasks_path=None,
        window=(events[0].time, events[-1].time),
        meta={"event_count": len(events), "entity_count": len({event.subject for event in events})},
    )


def _resolve_corpus(
    *,
    corpus: str,
    replay: Path | None,
    output: Path,
    seed: int,
    event_count: int,
    entity_count: int,
) -> CorpusHandle:
    if replay is not None:
        return _handle_for_replay(replay, corpus if corpus != "synthetic" else replay.stem)
    candidate = Path(corpus)
    if corpus != "synthetic" and candidate.is_file():
        return _handle_for_replay(candidate, candidate.stem)
    if corpus != "synthetic":
        raise typer.BadParameter(
            "use --corpus synthetic or pass a JSONL path/--replay",
            param_hint="--corpus",
        )
    return generate_synthetic_corpus(
        output,
        seed,
        event_count=event_count,
        entity_count=entity_count,
    )


def _probe_window(handle: CorpusHandle) -> tuple[datetime, datetime]:
    if isinstance(handle.window, tuple):
        return handle.window
    events = list(handle.events())
    if not events:
        raise ValueError("cannot generate probes from an empty corpus")
    return events[0].time, events[-1].time


def _generate_probes(
    handle: CorpusHandle,
    output: Path,
    *,
    per_family: int,
    seed: int,
    probe_start: datetime | None = None,
    probe_end: datetime | None = None,
) -> CorpusHandle:
    start, end = _probe_window(handle)
    audit = GoldAuditTrail(source="normalised-smoke-adapter")
    gold_report = Path(f"{output}.gold.json")
    try:
        probes = generate_probe_set(
            list(handle.events()),
            per_family=per_family,
            seed=seed,
            probe_start=probe_start or start,
            probe_end=probe_end or end,
            gold_audit=audit,
        )
    finally:
        write_gold_report(audit, gold_report)
    probe_path, _, digest = write_probe_set(probes, output)
    return replace(
        handle,
        probes_path=probe_path,
        meta={
            **handle.meta,
            "probe_count": len(probes),
            "probe_hash": digest,
            "dual_gold_agreement": audit.agreement_rate,
            "dual_gold_report": str(gold_report),
        },
    )


def _build_store(
    *,
    layout: str,
    events: list[EvalEvent],
    cfg: ExperimentConfig,
    root: Path,
    seed: int,
    model: str | None = None,
) -> tuple[StoreHandle, DriverStats]:
    store = StoreHandle(layout, f"eval-{layout.casefold()}-{seed}", root)
    configure_store(
        store,
        harness=make_harness_name(cfg),
        model=model if model is not None else cfg.engine.builder.model,
        embeddings=make_embeddings(cfg),
    )
    return store, run_pipeline(events, cfg.engine, store, cutoff=None, on_decision=None)


def _build_run_stores(
    *,
    selected: list[str],
    cfg: ExperimentConfig,
    events: list[EvalEvent],
    root: Path,
    seed: int,
    smoke: bool,
) -> dict[str, StoreHandle]:
    consumers = {"e2", "e4", "e5"}.intersection(selected)
    if not consumers:
        return {}
    requested = {"S0"}
    if cfg.engine.store.layout in {"S1", "S2", "S3", "S4", "S5"}:
        requested.add(cfg.engine.store.layout)
    if "e4" in selected and (smoke or cfg.engine.store.layout == "S3"):
        requested.add("S3")

    stores: dict[str, StoreHandle] = {}
    registry_rows: list[dict[str, object]] = []
    for layout in sorted(requested):
        typer.echo(f"building {layout} store for seed {seed} ({len(events)} events)")
        store, stats = _build_store(
            layout=layout,
            events=events,
            cfg=cfg,
            root=root / layout.casefold() / f"seed-{seed}",
            seed=seed,
        )
        stores[layout] = store
        registry_rows.append(
            {
                "layout": layout,
                "status": "built",
                "events": stats.events,
                "folds": stats.folds_per_lane,
                "root": str(store.root),
            }
        )
    for layout in ("S2", "S3", "S5"):
        if layout not in stores:
            registry_rows.append(
                {
                    "layout": layout,
                    "status": "registered-not-built",
                    "reason": "not required by this run; fake-harness ablation may duplicate S0/S1",
                }
            )
    root.mkdir(parents=True, exist_ok=True)
    (root / f"registry-seed-{seed}.json").write_text(
        json.dumps(registry_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return stores


def _write_e3_condition_metadata(condition: object) -> None:
    from harnext_eval.e3.run import StoreCondition

    if not isinstance(condition, StoreCondition):
        raise TypeError("expected an E3 StoreCondition")
    payload = {
        "label": condition.stable_label,
        "layout": condition.layout,
        "seed": condition.seed,
        "tier": condition.tier,
        "model": condition.model,
        "replay_hash": condition.replay_hash,
    }
    path = Path(condition.store.root) / "e3-condition.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_e3_conditions(
    *,
    cfg: ExperimentConfig,
    events: list[EvalEvent],
    root: Path,
    replay_hash: str,
    smoke: bool,
    optional_stores: set[str],
    opus_model: str | None,
) -> list[object]:
    """Build one complete E3 matrix, aggregating all configured seeds."""

    from harnext_eval.e3.run import StoreCondition

    invalid_optional = optional_stores - {"S2", "S5"}
    if invalid_optional:
        joined = ", ".join(sorted(invalid_optional))
        raise typer.BadParameter(
            f"E3 optional stores may only contain S2/S5, got {joined}",
            param_hint="--e3-optional-stores",
        )
    if cfg.engine.store.layout in {"S2", "S5"}:
        optional_stores.add(cfg.engine.store.layout)

    configured_model = cfg.engine.builder.model
    configured_opus = (
        configured_model
        if configured_model is not None and "opus" in configured_model.casefold()
        else None
    )
    resolved_opus = opus_model or configured_opus
    sonnet_model = "claude-sonnet-5" if configured_opus else configured_model
    if resolved_opus and not smoke and cfg.engine.builder.harness != "claude_code":
        raise typer.BadParameter(
            "an Opus-tier E3 build requires engine.builder.harness=claude_code",
            param_hint="--e3-opus-model",
        )

    conditions: list[StoreCondition] = []

    def build(
        layout: str,
        *,
        seed: int | None,
        tier: str,
        model: str | None,
        label: str | None = None,
    ) -> None:
        build_seed = cfg.seeds[0] if seed is None else seed
        stable_label = label or (
            layout if layout in {"S0", "S1", "S4"} else f"{layout}-{tier}-seed-{build_seed}"
        )
        typer.echo(f"building {stable_label} from the shared E3 replay ({len(events)} events)")
        store, _ = _build_store(
            layout=layout,
            events=events,
            cfg=cfg,
            root=root / stable_label.casefold(),
            seed=build_seed,
            model=model,
        )
        condition = StoreCondition(
            store=store,
            seed=seed,
            tier=tier,
            replay_hash=replay_hash,
            model=model or cfg.engine.builder.harness,
            label=stable_label,
        )
        _write_e3_condition_metadata(condition)
        conditions.append(condition)

    for layout in ("S0", "S1", "S4"):
        build(layout, seed=None, tier="baseline", model=None)
    for configured_seed in cfg.seeds:
        build("S3", seed=configured_seed, tier="sonnet", model=sonnet_model)
    for layout in sorted(optional_stores):
        for configured_seed in cfg.seeds:
            build(layout, seed=configured_seed, tier="sonnet", model=sonnet_model)
    if resolved_opus and not smoke:
        build("S3", seed=1, tier="opus", model=resolved_opus)

    registry = [
        {
            "label": condition.stable_label,
            "layout": condition.layout,
            "seed": condition.seed,
            "tier": condition.tier,
            "model": condition.model,
            "status": "built",
        }
        for condition in conditions
    ]
    if resolved_opus and smoke:
        registry.append(
            {
                "label": "S3-opus-seed-1",
                "layout": "S3",
                "seed": 1,
                "tier": "opus",
                "model": resolved_opus,
                "status": "supported-not-run",
                "reason": "the smoke profile intentionally omits the optional Opus tier",
            }
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / "e3-registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return list(conditions)


@app.command("corpus")
def corpus_command(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "apps/eval/out/corpus/synthetic.jsonl"
    ),
    seed: Annotated[int, typer.Option("--seed")] = 1,
    replay: Annotated[Path | None, typer.Option("--replay", exists=True, dir_okay=False)] = None,
    event_count: Annotated[int, typer.Option("--event-count", min=1)] = 2_000,
    entity_count: Annotated[int, typer.Option("--entity-count", min=1)] = 40,
    fetch: Annotated[str | None, typer.Option("--fetch")] = None,
    config: Annotated[Path | None, typer.Option("--config", exists=True, dir_okay=False)] = None,
) -> None:
    """Generate the synthetic corpus or validate and load a real JSONL replay."""

    if fetch is not None:
        if config is None:
            raise typer.BadParameter("--fetch requires --config", param_hint="--config")
        assert_offline_ok(load_config(config), corpus_fetch=fetch)
        raise typer.BadParameter(
            "network corpus extractors are not wired into this command",
            param_hint="--fetch",
        )

    handle = _resolve_corpus(
        corpus="synthetic" if replay is None else replay.stem,
        replay=replay,
        output=output,
        seed=seed,
        event_count=event_count,
        entity_count=entity_count,
    )
    action = "loaded" if replay is not None else "wrote"
    typer.echo(f"{action} {handle.meta['event_count']} events from {handle.replay_path}")


@app.command("probes")
def probes_command(
    replay: Annotated[Path, typer.Option("--replay", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--out", "-o")],
    per_family: Annotated[int, typer.Option("--per-family", min=0)] = 60,
    seed: Annotated[int, typer.Option("--seed")] = 1,
    probe_start: Annotated[str | None, typer.Option("--probe-start")] = None,
    probe_end: Annotated[str | None, typer.Option("--probe-end")] = None,
) -> None:
    """Generate all six frozen probe families from a replay."""

    handle = _handle_for_replay(replay, replay.stem)
    handle = _generate_probes(
        handle,
        output,
        per_family=per_family,
        seed=seed,
        probe_start=parse_time(probe_start) if probe_start else None,
        probe_end=parse_time(probe_end) if probe_end else None,
    )
    typer.echo(f"wrote {handle.meta['probe_count']} probes to {handle.probes_path}")


@app.command("stores")
def stores_command(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    replay: Annotated[Path, typer.Option("--replay", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--out", "-o")] = Path("apps/eval/out/stores"),
    layouts: Annotated[str | None, typer.Option("--layouts")] = None,
    seed: Annotated[int, typer.Option("--seed")] = 1,
) -> None:
    """Build configured store layouts through the shared replay driver."""

    cfg = load_config(config)
    assert_offline_ok(cfg)
    events = load_replay(replay)
    requested = (
        [item.strip().upper() for item in layouts.split(",")]
        if layouts
        else [cfg.engine.store.layout]
    )
    for layout in requested:
        if layout not in {"S0", "S1", "S2", "S3", "S4", "S5"}:
            raise typer.BadParameter(f"unknown layout {layout}", param_hint="--layouts")
        store, stats = _build_store(
            layout=layout,
            events=events,
            cfg=cfg,
            root=output / layout.casefold() / f"seed-{seed}",
            seed=seed,
        )
        typer.echo(f"built {layout}: {stats.events} events, {len(stats.snapshots)} snapshots at {store.root}")


@app.command("run")
def run_command(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    corpus: Annotated[str, typer.Option("--corpus")] = "synthetic",
    all_experiments: Annotated[bool, typer.Option("--all")] = False,
    experiment: Annotated[list[str] | None, typer.Option("--experiment", "-e")] = None,
    experiments: Annotated[str | None, typer.Option("--experiments")] = None,
    out: Annotated[Path, typer.Option("--out")] = Path("apps/eval/out"),
    replay: Annotated[Path | None, typer.Option("--replay", exists=True, dir_okay=False)] = None,
    per_family: Annotated[int, typer.Option("--per-family", min=0)] = 10,
    event_count: Annotated[int, typer.Option("--event-count", min=1)] = 120,
    entity_count: Annotated[int, typer.Option("--entity-count", min=1)] = 12,
    smoke: Annotated[bool, typer.Option("--smoke")] = False,
    e3_optional_stores: Annotated[
        str | None, typer.Option("--e3-optional-stores")
    ] = None,
    e3_opus_model: Annotated[str | None, typer.Option("--e3-opus-model")] = None,
) -> None:
    """Run registered experiments and write a reproducible run directory."""

    _discover_experiments()
    requested = list(experiment or [])
    if experiments:
        requested.extend(item.strip() for item in experiments.split(",") if item.strip())
    selected = list_experiments() if all_experiments else list(dict.fromkeys(requested))
    if not selected:
        raise typer.BadParameter("pass --all or at least one --experiment")
    unknown = sorted(set(selected).difference(list_experiments()))
    if unknown:
        raise typer.BadParameter(f"unknown experiments: {', '.join(unknown)}")
    cfg: ExperimentConfig = load_config(config)
    assert_offline_ok(cfg)
    make_llm(cfg)
    make_embeddings(cfg)
    make_harness_name(cfg)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{config.stem}"
    run_dir = out / run_id
    handle = _resolve_corpus(
        corpus=corpus,
        replay=replay,
        output=run_dir / "replay" / "synthetic.jsonl",
        seed=cfg.seeds[0],
        event_count=event_count,
        entity_count=entity_count,
    )
    handle = _generate_probes(
        handle,
        run_dir / "probes" / f"{handle.name}.jsonl",
        per_family=per_family,
        seed=cfg.seeds[0],
    )
    handle = replace(handle, meta={**handle.meta, "smoke": smoke})
    resolved = cfg.model_dump(mode="json")
    (run_dir / "config.yaml").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    manifest = build_manifest(
        run_id=run_id,
        config=cfg,
        replay_path=handle.replay_path,
        probe_path=handle.probes_path,
        model_ids={
            "builder": cfg.engine.builder.model or cfg.engine.builder.harness,
            "reader": cfg.engine.reader.provider,
        },
        prices={key: float(value) for key, value in (cfg.engine.prices or {}).items() if isinstance(value, (int, float))},
        seeds=cfg.seeds,
        provider_summary=provider_summary(cfg),
    )
    write_manifest(manifest, run_dir)
    events = list(handle.events())
    if "e3" in selected:
        optional_stores = {
            item.strip().upper()
            for item in (e3_optional_stores or "").split(",")
            if item.strip()
        }
        conditions = _build_e3_conditions(
            cfg=cfg,
            events=events,
            root=run_dir / "stores" / "e3",
            replay_hash=manifest.replay_hash,
            smoke=smoke,
            optional_stores=optional_stores,
            opus_model=e3_opus_model,
        )
        e3_handle = replace(
            handle,
            meta={
                **handle.meta,
                "stores": conditions,
                "read_budgets": cfg.budgets.read_tokens,
                "smoke": smoke,
            },
        )
        runner = get_experiment("e3")
        experiment_dir = run_dir / "e3"
        typer.echo(f"running e3 across configured seeds {cfg.seeds}")
        result = runner.run(cfg.engine, e3_handle, experiment_dir, cfg.seeds[0])
        result.artifacts.extend(runner.chart(result, experiment_dir))
        _write_result(result, experiment_dir)

    per_seed_selected = [name for name in selected if name != "e3"]
    for seed in cfg.seeds:
        stores = _build_run_stores(
            selected=per_seed_selected,
            cfg=cfg,
            events=events,
            root=run_dir / "stores",
            seed=seed,
            smoke=smoke,
        )
        run_handle = replace(
            handle,
            meta={
                **handle.meta,
                "store": stores.get(cfg.engine.store.layout, stores.get("S0")),
                "store_handle": stores.get("S3", stores.get(cfg.engine.store.layout)),
                "stores": [stores[name] for name in sorted(stores)],
            },
        )
        for name in per_seed_selected:
            runner = get_experiment(name)
            experiment_dir = run_dir / name / f"seed-{seed}"
            typer.echo(f"running {name} seed {seed}")
            result = runner.run(cfg.engine, run_handle, experiment_dir, seed)
            result.artifacts.extend(runner.chart(result, experiment_dir / "charts"))
            _write_result(result, experiment_dir)
    report = build_report(run_dir)
    typer.echo(f"report: {report}")
    typer.echo(str(run_dir))


@app.command("report")
def report_command(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Build or rebuild the self-contained report for a run directory."""

    typer.echo(str(build_report(run_dir)))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
