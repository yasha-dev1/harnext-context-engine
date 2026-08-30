"""Command-line entry point for docs/evaluation-spec.md §11."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
import yaml

from harnext_eval.config import EngineConfig, ExperimentConfig, load_config
from harnext_eval.corpus import CorpusHandle
from harnext_eval.corpus.synthetic import generate_synthetic_corpus
from harnext_eval.manifest import build_manifest, write_manifest
from harnext_eval.probes.common import load_replay, parse_time
from harnext_eval.probes.gen import generate_probe_set, write_probe_set
from harnext_eval.providers.embeddings import FakeEmbeddings
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


def _check_results(metrics: dict[str, float]) -> dict[str, bool]:
    count_metrics = {
        "checks.leakage_gate_passed",
        "checks.leakage_gate_failed",
        "checks.tasks_accepted",
    }
    return {
        name.split(".", 1)[1]: bool(value)
        for name, value in metrics.items()
        if name.startswith(("check.", "checks."))
        and name not in count_metrics
    }


def _write_result(result: ExperimentResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": result.name,
        "metrics": result.metrics,
        "checks": _check_results(result.metrics),
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
    probes = generate_probe_set(
        list(handle.events()),
        per_family=per_family,
        seed=seed,
        probe_start=probe_start or start,
        probe_end=probe_end or end,
    )
    probe_path, _, digest = write_probe_set(probes, output)
    return replace(
        handle,
        probes_path=probe_path,
        meta={**handle.meta, "probe_count": len(probes), "probe_hash": digest},
    )


def _build_store(
    *,
    layout: str,
    events: list[EvalEvent],
    cfg: EngineConfig,
    root: Path,
    seed: int,
) -> tuple[StoreHandle, DriverStats]:
    store = StoreHandle(layout, f"eval-{layout.casefold()}-{seed}", root)
    configure_store(
        store,
        harness=cfg.builder.harness,
        model=cfg.builder.model,
        embeddings=FakeEmbeddings(dim=cfg.embeddings.dim),
    )
    return store, run_pipeline(events, cfg, store, cutoff=None, on_decision=None)


def _build_run_stores(
    *,
    selected: list[str],
    cfg: EngineConfig,
    events: list[EvalEvent],
    root: Path,
    seed: int,
    smoke: bool,
) -> dict[str, StoreHandle]:
    consumers = {"e2", "e3", "e4", "e5"}.intersection(selected)
    if not consumers:
        return {}
    requested = {"S0"}
    if "e3" in selected:
        requested.update(("S1", "S4"))
    if cfg.store.layout in {"S1", "S2", "S3", "S4", "S5"}:
        requested.add(cfg.store.layout)
    if "e4" in selected and (smoke or cfg.store.layout == "S3"):
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


@app.command("corpus")
def corpus_command(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "apps/eval/out/corpus/synthetic.jsonl"
    ),
    seed: Annotated[int, typer.Option("--seed")] = 1,
    replay: Annotated[Path | None, typer.Option("--replay", exists=True, dir_okay=False)] = None,
    event_count: Annotated[int, typer.Option("--event-count", min=1)] = 2_000,
    entity_count: Annotated[int, typer.Option("--entity-count", min=1)] = 40,
) -> None:
    """Generate the synthetic corpus or validate and load a real JSONL replay."""

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

    cfg = load_config(config).engine
    events = load_replay(replay)
    requested = [item.strip().upper() for item in layouts.split(",")] if layouts else [cfg.store.layout]
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
        seeds=cfg.seeds,
    )
    write_manifest(manifest, run_dir)
    events = list(handle.events())
    for seed in cfg.seeds:
        stores = _build_run_stores(
            selected=selected,
            cfg=cfg.engine,
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
        for name in selected:
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
