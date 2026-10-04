"""What pins kind `engine` to a run: sha256 of the code that serves it and of the engine config per
chat preset. The runner pipes this file into `python -` inside the trial image (so an image built
before this module existed is measured the same way) and runs it on the host; both are compared
with the run manifest.

    python -m harnext_eval.e2.v2.engine_pin [openrouter,ollama]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import harnext_eval.e2.v2 as v2  # noqa: E402 - the installed package, also when this runs from stdin

HERE = Path(v2.__file__).resolve().parent


def tree_hash(root: Path) -> str:
    """sha256 over a package's sources (sorted relative paths and bytes), as engine.config.code_hash."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(
            path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0"
        )
    return digest.hexdigest()


def identity(presets=("openrouter", "ollama")) -> dict:
    from harnext_eval.e2.v2.engine.config import code_hash, config_hash, for_endpoint

    return {
        "engine_code_sha256": code_hash(),
        "raw_store_code_sha256": tree_hash(HERE / "stores/raw"),
        "mcp_server_sha256": hashlib.sha256((HERE / "mcp_server.py").read_bytes()).hexdigest(),
        "config_sha256": {p: config_hash(for_endpoint(p)) for p in presets},
    }


if __name__ == "__main__":
    presets = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("openrouter", "ollama")
    print(json.dumps(identity(presets), sort_keys=True))
