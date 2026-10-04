"""Ask the engine one question from the command line.

    python -m harnext_eval.e2.v2.engine.ask --store <RAW export dir> [--memory <notes dir>] \\
        --question "..." [--endpoint ollama|openrouter] [--chat-base-url URL --chat-model NAME] \\
        [--sql-fallback auto|always|never] [--config overrides.json] [--trace out.json]

Prints the report, then usage and cost; --trace writes the full structured trace.
"""

from __future__ import annotations

import os
import time

os.environ["TZ"] = "UTC"  # before chdb loads: SQL conversion functions use the process zone
time.tzset()

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from harnext_eval.e2.v2.engine.config import ENDPOINTS, EngineConfig  # noqa: E402
from harnext_eval.e2.v2.engine.engine import Engine  # noqa: E402


def config_from(args) -> EngineConfig:
    values = {**ENDPOINTS[args.endpoint], **(json.loads(Path(args.config).read_text()) if args.config else {})}  # fmt: skip
    if args.chat_base_url:
        values["chat_base_url"] = args.chat_base_url
    if args.chat_model:
        values["chat_model"] = args.chat_model
    if args.sql_fallback:
        values["sql_fallback"] = args.sql_fallback
    return EngineConfig(**values)


async def main_async(args) -> dict:
    scratch = Path(tempfile.mkdtemp(prefix="engine-ask-"))  # private scratch for the export copy
    engine = Engine(args.store, args.memory, config_from(args), scratch=scratch)
    try:
        return await engine.research(args.question)
    finally:
        await engine.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("--store", type=Path)
    parser.add_argument("--memory", type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--chat-base-url")
    parser.add_argument("--chat-model")
    parser.add_argument("--endpoint", choices=sorted(ENDPOINTS), default="ollama", help="chat model endpoint preset (default: local Ollama)")  # fmt: skip
    parser.add_argument("--sql-fallback", choices=("auto", "always", "never"))
    parser.add_argument("--config", type=Path, help="JSON object of EngineConfig overrides")
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    result = asyncio.run(main_async(args))
    print(result["answer"])
    usage = result["usage"]
    print(f"\n-- {result['seconds']:.1f} s, {result['turns']} turns, partial={result['partial']}, "
          f"cost ${usage['cost_usd']:.6f} (jev ${usage['jev']['cost_usd']:.6f} / {usage['jev']['calls']} calls, "
          f"chat ${usage['chat']['cost_usd']:.6f} / {usage['chat']['calls']} calls)")  # fmt: skip
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        args.trace.write_text(json.dumps(result["trace"], indent=1, ensure_ascii=False, default=str) + "\n")  # fmt: skip


if __name__ == "__main__":
    main()
