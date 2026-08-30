"""Command-line entry point for docs/evaluation-spec.md §11."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
import yaml

from harnext_eval.config import load_config
from harnext_eval.corpus.synthetic import generate_synthetic_corpus
from harnext_eval.manifest import build_manifest, write_manifest
from harnext_eval.registry import ExperimentResult, get_experiment, list_experiments

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


def _write_result(result: ExperimentResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": result.name,
        "metrics": result.metrics,
        "tables": {name: table.to_dict(orient="records") for name, table in result.tables.items()},
        "artifacts": [str(path) for path in result.artifacts],
        "primary": result.primary,
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


@app.command("corpus")
def corpus_command(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "apps/eval/out/corpus/synthetic.jsonl"
    ),
    seed: Annotated[int, typer.Option("--seed")] = 1,
) -> None:
    """Generate the deterministic offline corpus."""

    handle = generate_synthetic_corpus(output, seed)
    typer.echo(f"wrote {handle.meta['event_count']} events to {handle.replay_path}")


@app.command("probes")
def probes_command() -> None:
    """Describe the probe entry point reserved for T3 integration."""

    typer.echo("probe generators are registered by T3")


@app.command("stores")
def stores_command() -> None:
    """Describe the store entry point reserved for T5 integration."""

    typer.echo("store layouts are registered by T5")


@app.command("run")
def run_command(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    corpus: Annotated[str, typer.Option("--corpus")] = "synthetic",
    all_experiments: Annotated[bool, typer.Option("--all")] = False,
    experiment: Annotated[list[str] | None, typer.Option("--experiment", "-e")] = None,
    out: Annotated[Path, typer.Option("--out")] = Path("apps/eval/out"),
) -> None:
    """Run registered experiments and write a reproducible run directory."""

    if corpus != "synthetic":
        raise typer.BadParameter("T0 supports only the synthetic corpus", param_hint="--corpus")
    selected = list_experiments() if all_experiments else (experiment or [])
    if not selected:
        raise typer.BadParameter("pass --all or at least one --experiment")
    cfg = load_config(config)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{config.stem}"
    run_dir = out / run_id
    handle = generate_synthetic_corpus(run_dir / "replay" / "synthetic.jsonl", cfg.seeds[0])
    resolved = cfg.model_dump(mode="json")
    (run_dir / "config.yaml").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    manifest = build_manifest(
        run_id=run_id,
        config=cfg,
        replay_path=handle.replay_path,
        model_ids={
            "builder": cfg.engine.builder.model or cfg.engine.builder.harness,
            "reader": cfg.engine.reader.provider,
        },
        seeds=cfg.seeds,
    )
    write_manifest(manifest, run_dir)
    for seed in cfg.seeds:
        for name in selected:
            runner = get_experiment(name)
            experiment_dir = run_dir / name / f"seed-{seed}"
            result = runner.run(cfg.engine, handle, experiment_dir, seed)
            result.artifacts.extend(runner.chart(result, experiment_dir / "charts"))
            _write_result(result, experiment_dir)
    typer.echo(str(run_dir))


@app.command("report")
def report_command(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Describe the report entry point reserved for T10 integration."""

    typer.echo(f"report generation is registered by T10 for {run_dir}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
