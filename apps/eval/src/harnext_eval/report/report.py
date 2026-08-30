"""Self-contained run report for docs/evaluation-spec.md §2, §7, §8, and §9."""

from __future__ import annotations

import argparse
import base64
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _display(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _rows(mapping: Mapping[str, object]) -> list[dict[str, str]]:
    return [{"name": str(name), "value": _display(value)} for name, value in mapping.items()]


def _nudge_rows(config: Mapping[str, Any]) -> list[dict[str, str]]:
    engine = config.get("engine", {})
    if not isinstance(engine, Mapping):
        return []
    router = engine.get("router", {})
    window = engine.get("window", {})
    store = engine.get("store", {})
    builder = engine.get("builder", {})
    reader = engine.get("reader", {})

    def get(source: object, key: str, default: object = "—") -> object:
        return source.get(key, default) if isinstance(source, Mapping) else default

    rules = get(router, "rules", {})
    deviation = get(router, "deviation", {})
    guards = get(router, "guards", {})
    summaries = {
        "router": (
            f"budget={get(router, 'budget_pct')}%; rules={get(rules, 'enabled')}; "
            f"deviation={get(deviation, 'enabled')}; guards={_display(guards)}"
        ),
        "window": (
            f"gap={get(window, 'gap_s')}s; max_events={get(window, 'max_events')}; "
            f"max_age={get(window, 'max_age_s')}s"
        ),
        "store": f"layout={get(store, 'layout')}; backend={get(store, 'backend')}",
        "builder": (
            f"harness={get(builder, 'harness')}; model={get(builder, 'model')}; "
            f"prompt={get(builder, 'prompt_version')}"
        ),
        "reader": (
            f"provider={get(reader, 'provider')}; budget={get(reader, 'budget_tokens')} tokens"
        ),
        "envelope": _display(engine.get("envelope", "—")),
    }
    return [{"component": component, "setting": setting} for component, setting in summaries.items()]


def _normalise_records(frame: pd.DataFrame) -> tuple[list[str], list[dict[str, str]]]:
    frame = frame.where(pd.notna(frame), None)
    columns = [str(column) for column in frame.columns]
    records = [
        {str(column): _display(value) for column, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]
    return columns, records


def _experiment_name(relative: Path) -> str | None:
    for part in relative.parts:
        folded = part.casefold()
        if len(folded) == 2 and folded[0] == "e" and folded[1].isdigit():
            return folded
    return None


def _extract_checks(
    results: list[tuple[Path, dict[str, Any]]], check_csvs: list[Path], run_dir: Path
) -> list[dict[str, str | bool | None]]:
    checks: list[dict[str, str | bool | None]] = []
    for path, payload in results:
        experiment = _experiment_name(path.relative_to(run_dir)) or payload.get("name", "run")
        candidates: list[tuple[str, object]] = []
        direct = payload.get("checks")
        if isinstance(direct, Mapping):
            candidates.extend((str(name), value) for name, value in direct.items())
        metrics = payload.get("metrics", {})
        if isinstance(metrics, Mapping):
            nested = metrics.get("checks")
            if isinstance(nested, Mapping):
                candidates.extend((str(name), value) for name, value in nested.items())
            candidates.extend(
                (str(name).split(".", 1)[1], value)
                for name, value in metrics.items()
                if str(name).startswith(("checks.", "check."))
            )
        for name, value in candidates:
            passed: bool | None = value if isinstance(value, bool) else None
            display_value: object = value
            if isinstance(value, Mapping):
                candidate = value.get("passed")
                passed = candidate if isinstance(candidate, bool) else None
                display_value = value.get("value", candidate)
            checks.append(
                {
                    "experiment": str(experiment).upper(),
                    "check": name,
                    "value": _display(display_value),
                    "passed": passed,
                }
            )

    for path in check_csvs:
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            continue
        relative = path.relative_to(run_dir)
        experiment = (_experiment_name(relative) or relative.parent.name or "run").upper()
        for row_index, (_, row) in enumerate(frame.iterrows(), start=1):
            name = next(
                (row[column] for column in ("check", "name", "metric") if column in row),
                f"row-{row_index}",
            )
            raw_passed = next(
                (row[column] for column in ("passed", "pass", "status") if column in row),
                None,
            )
            if isinstance(raw_passed, str):
                passed: bool | None = raw_passed.strip().casefold() in {"pass", "passed", "true", "1"}
                if raw_passed.strip().casefold() not in {
                    "pass",
                    "passed",
                    "true",
                    "1",
                    "fail",
                    "failed",
                    "false",
                    "0",
                }:
                    passed = None
            elif raw_passed is None or (
                isinstance(raw_passed, float) and math.isnan(raw_passed)
            ):
                passed = None
            else:
                passed = bool(raw_passed) if raw_passed is not None else None
            value = next(
                (row[column] for column in ("value", "observed", "count") if column in row),
                raw_passed,
            )
            checks.append(
                {
                    "experiment": experiment,
                    "check": str(name),
                    "value": _display(value),
                    "passed": passed,
                }
            )
    return checks


def _chart(path: Path, run_dir: Path) -> dict[str, str]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "name": path.stem.replace("_", " ").title(),
        "path": path.relative_to(run_dir).as_posix(),
        "data_uri": f"data:image/png;base64,{encoded}",
    }


def build_report(run_dir: str | Path) -> Path:
    """Render ``report.html`` for one completed evaluation run.

    The run directory may contain root ``manifest.json`` and ``config.yaml``,
    experiment ``results.json`` files at any depth, ``contrasts.csv`` files,
    validity/check CSVs, arbitrary CSV outputs, and PNG charts. PNG bytes and CSS
    are inlined; CSV links remain relative links to reproducible run artifacts.
    """

    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    manifest = _load_json(root / "manifest.json")
    config_path = root / "config.yaml"
    config_text = config_path.read_text(encoding="utf-8") if config_path.exists() else "{}\n"
    loaded_config = yaml.safe_load(config_text) or {}
    if not isinstance(loaded_config, Mapping):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    pretty_config = yaml.safe_dump(dict(loaded_config), sort_keys=False, allow_unicode=True)

    result_files = sorted(root.rglob("results.json"))
    loaded_results = [(path, _load_json(path)) for path in result_files]
    csv_paths = sorted(root.rglob("*.csv"))
    png_paths = sorted(root.rglob("*.png"))
    check_csvs = [
        path
        for path in csv_paths
        if "check" in path.stem.casefold() or path.stem.casefold() == "gate"
    ]

    experiment_names = {
        name
        for path in [*result_files, *csv_paths, *png_paths]
        if (name := _experiment_name(path.relative_to(root))) is not None
    }
    experiments: list[dict[str, object]] = []
    for name in sorted(experiment_names):
        experiment_results: list[dict[str, object]] = []
        for path, payload in loaded_results:
            if _experiment_name(path.relative_to(root)) != name:
                continue
            primary = payload.get("primary", {})
            metrics = payload.get("metrics", {})
            primary_mapping = primary if isinstance(primary, Mapping) and primary else metrics
            if not isinstance(primary_mapping, Mapping):
                primary_mapping = {}
            experiment_results.append(
                {
                    "run": path.parent.relative_to(root).as_posix(),
                    "primary": _rows(primary_mapping),
                }
            )

        contrasts: list[dict[str, object]] = []
        for path in csv_paths:
            if _experiment_name(path.relative_to(root)) != name or path.stem.casefold() != "contrasts":
                continue
            try:
                columns, records = _normalise_records(pd.read_csv(path))
            except (OSError, pd.errors.ParserError, UnicodeDecodeError):
                columns, records = ["error"], [{"error": "Could not read CSV"}]
            contrasts.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "columns": columns,
                    "records": records,
                }
            )

        experiments.append(
            {
                "name": name.upper(),
                "results": experiment_results,
                "contrasts": contrasts,
                "charts": [
                    _chart(path, root)
                    for path in png_paths
                    if _experiment_name(path.relative_to(root)) == name
                ],
                "csvs": [
                    path.relative_to(root).as_posix()
                    for path in csv_paths
                    if _experiment_name(path.relative_to(root)) == name
                ],
            }
        )

    unassigned_charts = [
        _chart(path, root)
        for path in png_paths
        if _experiment_name(path.relative_to(root)) is None
    ]
    unassigned_csvs = [
        path.relative_to(root).as_posix()
        for path in csv_paths
        if _experiment_name(path.relative_to(root)) is None
    ]
    context = {
        "title": f"Evaluation run {manifest.get('run_id', root.name)}",
        "manifest": _rows(manifest),
        "config_yaml": pretty_config,
        "nudges": _nudge_rows(loaded_config),
        "experiments": experiments,
        "checks": _extract_checks(loaded_results, check_csvs, root),
        "unassigned_charts": unassigned_charts,
        "unassigned_csvs": unassigned_csvs,
        "all_csvs": [path.relative_to(root).as_posix() for path in csv_paths],
    }

    environment = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=select_autoescape(("html", "xml")),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    rendered = environment.get_template("report.html.j2").render(**context)
    destination = root / "report.html"
    temporary = root / "report.html.tmp"
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    """CLI for ``python -m harnext_eval.report.report <run_dir>``."""

    parser = argparse.ArgumentParser(description="Build a self-contained evaluation report")
    parser.add_argument("run_dir", type=Path)
    arguments = parser.parse_args(argv)
    print(build_report(arguments.run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
