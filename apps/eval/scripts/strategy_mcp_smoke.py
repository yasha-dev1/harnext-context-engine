"""Real neural retrieval and separate-process MCP protocol verification.

This client follows fixed test instructions; its results are not agent accuracy.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import yaml
from fastmcp import Client
from harnext_builder.strategies.config import StrategyConfig, load_config
from harnext_builder.strategies.engine import build
from harnext_eval.e2.mcp_experiment import audit_receipts


def prepare_output(output: Path):
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)


async def run(base: Path, output: Path):
    prepare_output(output)
    reports = []
    for method in ["bm25", "dense", "hybrid"]:
        for reranker in ["none", "cross_encoder"]:
            name = f"{method}-{reranker}"
            cfg = load_config(base).model_dump(mode="json")
            cfg["artifact_dir"] = str((output / name).resolve())
            cfg["retrieval"].update(method=method, reranker=reranker)
            cfg["trial"].update(
                run_id=name,
                harness="external-python-mcp-client",
                model="none",
                effort="none",
                transport="stdio",
            )
            config = StrategyConfig.model_validate(cfg)
            config_path = (output / f"{name}.yaml").resolve()
            config_path.write_text(yaml.safe_dump(cfg))
            started = time.monotonic()
            artifact = await build(config)
            connection = {
                "mcpServers": {
                    "harnext": {
                        "command": sys.executable,
                        "args": ["-m", "harnext_mcp.experiment", "--config", str(config_path)],
                    }
                }
            }
            async with Client(connection, timeout=90) as client:
                tools = [tool.name for tool in await client.list_tools()]
                retrieved = await client.call_tool(
                    "context_search", {"query": "Who owns the cache repair issue now?"}
                )
                assert not retrieved.is_error
                assert "Noah" in retrieved.content[0].text
                response = await client.call_tool(
                    "get_assertions", {"entity": "Cache Repair", "predicate": "assigned_to"}
                )
                assert "person:Noah" in response.content[0].text
                assert "person:Mira" not in response.content[0].text
            quote = "CTX-41 is now assigned to Noah."
            spans = []
            for path, text in artifact["payload"]["documents"].items():
                encoded = text.encode()
                start = encoded.find(quote.encode())
                if start >= 0:
                    spans.append(
                        {
                            "unit_id": "current-owner",
                            "path": path,
                            "start": start,
                            "end": start + len(quote.encode()),
                            "quote": quote,
                        }
                    )
            gold = {
                "snapshot_sha256": artifact["sha256"],
                "spans": spans,
                "required_groups": [["current-owner"]],
                "annotations_exhaustive": False,
            }
            report = audit_receipts(
                artifact, config.artifact_dir / "trials" / f"{name}.jsonl", gold
            )
            assert report["required_group_recall"] == 1
            report.update(
                condition=name,
                tools=tools,
                elapsed_s=time.monotonic() - started,
                pins=artifact["payload"]["index"]["pins"],
                purpose="protocol verification; not model performance",
            )
            reports.append(report)
            (output / "report.json").write_text(json.dumps(reports, indent=2))
            print(
                f"PASS {name}: separate-process MCP, snapshot bytes, evidence coverage", flush=True
            )
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.config, args.out))
