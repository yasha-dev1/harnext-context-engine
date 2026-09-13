"""Build, inspect and enumerate supported strategy YAML configurations."""

import argparse
import asyncio
import json
from pathlib import Path

from .config import StrategyConfig, load_config
from .engine import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "validate", "schema"])
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.command == "schema":
        print(json.dumps(StrategyConfig.model_json_schema(), indent=2))
        return
    if args.config is None:
        parser.error("--config is required")
    config = load_config(args.config)
    if args.command == "build":
        artifact = asyncio.run(build(config))
        print(
            json.dumps(
                {"snapshot_sha256": artifact["sha256"], "artifact": str(config.artifact_dir)}
            )
        )
    else:
        print(config.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
