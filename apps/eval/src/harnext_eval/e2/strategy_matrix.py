"""Expand a YAML grid into validated, separately named engine configurations."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from harnext_builder.strategies.config import StrategyConfig, Strict, digest, load_config


class Matrix(Strict):
    protocol: Literal["harnext-matrix-v1"]
    base: Path
    output_dir: Path
    factors: dict[str, list[Any]]


def expand(path: Path):
    matrix = Matrix.model_validate(yaml.safe_load(path.read_text()))
    base_path = (path.parent / matrix.base).resolve()
    output = (path.parent / matrix.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("matrix output must be new or empty")
    base = load_config(base_path).model_dump(mode="json")
    if not matrix.factors or any(not values for values in matrix.factors.values()):
        raise ValueError("matrix must have nonempty factor levels")
    cases = []
    for index, levels in enumerate(itertools.product(*matrix.factors.values()), 1):
        case = copy.deepcopy(base)
        selected = dict(zip(matrix.factors, levels, strict=True))
        for field, value in selected.items():
            parts = field.split(".")
            target = case
            for part in parts[:-1]:
                target = target[part]
            if parts[-1] not in target:
                raise ValueError(f"unknown matrix field {field}")
            target[parts[-1]] = value
        name = f"condition-{index:03}"
        case["artifact_dir"] = str(output / "artifacts" / name)
        case["trial"]["run_id"] = name + "-read-001"
        resolved = StrategyConfig.model_validate(case).model_dump(mode="json")
        cases.append(
            {"id": name, "factors": selected, "config": resolved, "sha256": digest(resolved)}
        )
    # Validate the entire grid before writing any condition: no silent cell skips.
    output.mkdir(parents=True, exist_ok=True)
    for case in cases:
        (output / f"{case['id']}.yaml").write_text(yaml.safe_dump(case["config"], sort_keys=False))
    (output / "matrix.json").write_text(json.dumps(cases, indent=2))
    return {"conditions": len(cases), "output": str(output), "matrix_sha256": digest(cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(expand(args.config), indent=2))


if __name__ == "__main__":
    main()
