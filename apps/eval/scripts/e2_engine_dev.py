"""Dev evaluation of the E2 query-time engine, without the outer coding agent: for each historical
task, research the task's question at its checkpoint, extract the answer value with one extra
gpt-oss call under the benchmark's reply contract, grade with `grade_historical`.

    .venv/bin/python apps/eval/scripts/e2_engine_dev.py \\
        --benchmark apps/eval/benchmarks/kafka-500-v2-smoke \\
        --stores apps/eval/out/kafka-500-v2/stores-smoke/RAW/export \\
        --mode auto --endpoint ollama|openrouter --out apps/eval/out/kafka-500-v2/engine-dev/smoke-auto \\
        [--tasks ID,ID] \\
        [--memory-root apps/eval/out/kafka-500-v2/checkouts-smoke/F5] [--config overrides.json] \\
        [--max-usd 1.0]
        [--questions invented.jsonl]   # {"id", "checkpoint", "question"} rows: engine alone,
                                       # timing and endings only (no extraction, no grading)

Writes results.jsonl (one row per task, resumable), traces/<task>.json and summary.json
(accuracy; mean cost per question split Jev / chat, extraction separate; seconds; turns; share
of find plans vs fallback SQL; routing decisions). Develop on the smoke benchmark only.
"""

from __future__ import annotations

import os
import time

os.environ["TZ"] = "UTC"  # before chdb loads: SQL conversion functions use the process zone
time.tzset()

import argparse  # noqa: E402
import asyncio  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from harnext_eval.e2.v2.engine.clients import InfraError, Meter  # noqa: E402
from harnext_eval.e2.v2.engine.config import ENDPOINTS, EngineConfig, source_files  # noqa: E402
from harnext_eval.e2.v2.engine.engine import Engine  # noqa: E402
from harnext_eval.e2.v2.grade import grade_historical  # noqa: E402
from harnext_eval.e2.v2.runner import CONTRACT  # noqa: E402

EXTRACT = (
    "You turn a research report into the answer of a task. Use only the report. Copy exact "
    "values (identifiers, names, timestamps) from the report; convert them only when the task "
    "demands a format. " + CONTRACT["historical"] + " Reply with the JSON object only."
)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def question_of(task: dict) -> str:
    """The task prompt without its reply-format tail (the engine is asked the question); an
    invented question (--questions) as it is."""
    if "question" in task:
        return task["question"]
    return task["prompt"].split("\n\nReturn JSON", 1)[0].strip()


def parse(content: str):
    content = (content or "").strip()
    start, end = content.find("{"), content.rfind("}")
    try:
        return json.loads(content[start : end + 1]) if start >= 0 else None
    except ValueError:
        return None


def coerce(answer, schema: dict | None):
    """A value the model JSON-encoded as a string although the schema wants another type."""
    if not isinstance(answer, dict) or not schema or not isinstance(answer.get("value"), str):
        return answer
    wanted = schema.get("properties", {}).get("value", {}).get("type")
    wanted = set(wanted) if isinstance(wanted, list) else {wanted}
    if "string" in wanted:
        return answer
    try:
        return {**answer, "value": json.loads(answer["value"])}
    except ValueError:
        return answer


async def extract(engine: Engine, task: dict, report: str, meter: Meter) -> dict | None:
    messages = [{"role": "system", "content": EXTRACT},
                {"role": "user", "content": f"Task:\n{task['prompt']}\n\nResearch report:\n{report}"}]  # fmt: skip
    schema = task.get("answer_schema")
    fmt = {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": schema}} if schema else None  # fmt: skip
    message, call = await engine.chat.complete(messages, response_format=fmt, max_tokens=1500, purpose="extract")  # fmt: skip
    meter.add(call)
    found = parse(message.get("content"))
    if (not isinstance(found, dict) or "value" not in found) and fmt:  # schema ignored: plain JSON
        message, call = await engine.chat.complete(messages, max_tokens=1500, purpose="extract")
        meter.add(call)
        found = parse(message.get("content"))
    return coerce(found, schema)


def log_attempt(out: Path, task: dict, error: Exception) -> float:
    """A failed research attempt: logged with its metered usage (counted in the spend)."""
    usage = getattr(error, "usage", None) or {}
    with (out / "attempts.jsonl").open("a") as stream:
        stream.write(json.dumps({"id": task["id"], "error": f"{type(error).__name__}: {error}"[:500], "usage": usage}) + "\n")  # fmt: skip
    return usage.get("cost_usd", 0.0)


def grounded(answer, result: dict):
    """Evidence of the extracted answer restricted to ids the engine retrieved and showed."""
    if not isinstance(answer, dict) or not isinstance(answer.get("evidence"), list):
        return answer
    shown = "\n".join(r for b in result["trace"]["branches"] for r in b.get("results", []))
    keep = [e for e in answer["evidence"] if isinstance(e, str) and (e in result["evidence"] or (len(e) > 3 and e in shown))]  # fmt: skip
    return {**answer, "evidence": keep}


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    mean = lambda key: round(statistics.mean(r[key] for r in rows), 6)  # noqa: E731
    finds = sum(r["stats"]["find"] for r in rows)
    sql = sum(r["stats"]["sql"] for r in rows)
    jev = [s for r in rows for s in r["jev_seconds"]]
    return {
        "tasks": len(rows), "correct": sum(r["correct"] for r in rows),
        "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4),
        "mean_cost_usd": mean("cost_usd"), "mean_cost_jev_usd": mean("cost_jev_usd"),
        "mean_cost_chat_usd": mean("cost_chat_usd"), "mean_cost_extract_usd": mean("cost_extract_usd"),
        "total_cost_usd": round(sum(r["cost_usd"] + r["cost_extract_usd"] for r in rows), 6),
        "mean_seconds": mean("seconds"), "max_seconds": max(r["seconds"] for r in rows),
        "p50_seconds": round(statistics.median(r["seconds"] for r in rows), 2),
        "p95_seconds": round(sorted(r["seconds"] for r in rows)[int(0.95 * (len(rows) - 1))], 2),
        "verification_changed": sum(bool(v and (v["removed_citations"] or v["regenerated"] or v["stripped"])) for r in rows if (v := r.get("verification"))),  # fmt: skip
        "verification_regenerated": sum(bool(v and v["regenerated"]) for r in rows if (v := r.get("verification"))),  # fmt: skip
        "not_found": sum((r.get("verification") or {}).get("verdict") == "declined" for r in rows),
        "stores": sorted({json.dumps(r.get("store"), sort_keys=True) for r in rows}),
        "code_sha256": sorted({r.get("code_sha256") for r in rows if r.get("code_sha256")}),
        "config_sha256": sorted({r.get("config_sha256") for r in rows if r.get("config_sha256")}),
        "unmetered_upper_usd": round(sum(r.get("unmetered_upper_usd", 0) for r in rows), 6),
        "retries": sum(r.get("retries", 0) for r in rows),
        "mean_turns": mean("turns"), "partial": sum(r["partial"] for r in rows),
        "mean_chat_input_tokens": mean("chat_input_tokens"), "mean_chat_output_tokens": mean("chat_output_tokens"),
        "mean_chat_seconds": mean("chat_seconds"), "mean_chat_calls": mean("chat_calls"),
        "rejected_sql": [x for r in rows for x in r["stats"].get("rejected", [])],
        "find_calls": finds, "find_with_rows": sum(r["stats"]["find_rows"] for r in rows),
        "sql_calls": sql, "sql_ok": sum(r["stats"]["sql_ok"] for r in rows),
        "plan_share": round(finds / (finds + sql), 4) if finds + sql else None,
        "jev_calls": sum(len(r["jev_seconds"]) for r in rows),
        "jev_p50_s": round(statistics.median(jev), 3) if jev else None,
        "jev_p95_s": round(sorted(jev)[int(0.95 * (len(jev) - 1))], 3) if jev else None,
        "jev_tokens_per_call": round(statistics.mean(r["jev_tokens"] for r in rows if r["jev_seconds"]) / max(1, statistics.mean(len(r["jev_seconds"]) for r in rows)), 1),
        "routes": {r["id"]: r["routes"] for r in rows},
        "families": {r["family"]: r["correct"] for r in rows},
    }  # fmt: skip


async def run(args) -> None:
    ungraded = args.questions is not None  # invented questions: timing and endings only
    if ungraded:
        public = {t["id"]: {**t, "kind": "historical", "family": "invented"} for t in jsonl(args.questions)}  # fmt: skip
        private = {}
    else:
        public = {t["id"]: t for t in jsonl(args.benchmark / "tasks.public.jsonl")}
        private = {t["id"]: t for t in jsonl(args.benchmark / "tasks.private.jsonl")}
    wanted = set(args.tasks.split(",")) if args.tasks else None
    tasks = [t for t in public.values() if t["kind"] == "historical" and (not wanted or t["id"] in wanted)]  # fmt: skip
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "source.json").write_text(json.dumps(source_files(), indent=1, sort_keys=True) + "\n")  # fmt: skip
    (args.out / "traces").mkdir(exist_ok=True)
    results = args.out / "results.jsonl"
    done = {r["id"] for r in jsonl(results)} if results.exists() else set()
    overrides = json.loads(args.config.read_text()) if args.config else {}
    config = EngineConfig(**{**ENDPOINTS[args.endpoint], **overrides, "sql_fallback": args.mode})
    spent = sum(r["cost_usd"] + r["cost_extract_usd"] for r in jsonl(results)) if results.exists() else 0.0  # fmt: skip
    for checkpoint in sorted({t["checkpoint"] for t in tasks}):
        todo = [t for t in tasks if t["checkpoint"] == checkpoint and t["id"] not in done]
        if not todo:
            continue
        memory = args.memory_root / checkpoint if args.memory_root else None
        scratch = Path(
            tempfile.mkdtemp(prefix="engine-dev-")
        )  # private: no other process sweeps it
        opened = functools.partial(Engine, args.stores / checkpoint, memory, config, scratch=scratch)  # fmt: skip
        manifest = json.loads((args.stores / f"{checkpoint}.manifest.json").read_text())
        store = {"build_version": manifest.get("build", {}).get("version"), "content_sha256": manifest.get("content_sha256")}  # fmt: skip
        engine = opened()  # opened fresh per checkpoint
        try:
            for task in todo:
                if spent >= args.max_usd:
                    print(f"stop: spend cap ${args.max_usd} reached (${spent:.4f})")
                    return
                meter = Meter()
                try:
                    try:
                        result = await engine.research(question_of(task))
                    except InfraError:
                        raise
                    except Exception as error:  # noqa: BLE001 - e.g. the export swapped mid-run
                        spent += log_attempt(args.out, task, error)
                        print(
                            f"{task['id']}: {type(error).__name__}: {error}; reopening once (logged)"
                        )
                        await engine.aclose()
                        engine = opened()
                        result = await engine.research(question_of(task))
                    answer = None if ungraded else grounded(await extract(engine, task, result["answer"], meter), result)  # fmt: skip
                except InfraError as error:
                    spent += log_attempt(args.out, task, error)
                    print(f"{task['id']}: INFRA {error}; stopping (rerun resumes; attempt logged)")
                    return
                graded = {"correct": None, "expected": None, "actual": None} if ungraded else grade_historical(private[task["id"]], answer)  # fmt: skip
                usage, trace = result["usage"], result["trace"]
                jev_calls = [c for c in trace["calls"] if c["kind"] == "jev"]
                row = {"id": task["id"], "family": task["family"], "checkpoint": checkpoint,
                       "mode": args.mode, "endpoint": args.endpoint, "correct": graded["correct"], "expected": graded["expected"],
                       "actual": graded["actual"], "answer": answer, "partial": result["partial"],
                       "seconds": result["seconds"], "turns": result["turns"],
                       "cost_usd": usage["cost_usd"], "cost_jev_usd": usage["jev"]["cost_usd"],
                       "cost_chat_usd": usage["chat"]["cost_usd"],
                       "cost_extract_usd": meter.summary()["cost_usd"],
                       "unmetered_upper_usd": usage.get("unmetered_upper_usd", 0.0),
                       "retries": usage["jev"].get("retries", 0) + usage["chat"].get("retries", 0),
                       "chat_tokens": usage["chat"]["input_tokens"] + usage["chat"]["output_tokens"],
                       "jev_tokens": usage["jev"]["input_tokens"] + usage["jev"]["output_tokens"],
                       "chat_input_tokens": usage["chat"]["input_tokens"], "chat_output_tokens": usage["chat"]["output_tokens"],
                       "chat_calls": usage["chat"]["calls"], "chat_with_reasoning": usage["chat"].get("with_reasoning"),
                       "chat_seconds": usage["chat"]["seconds"], "jev_calls": usage["jev"]["calls"],
                       "jev_seconds": [c["seconds"] for c in jev_calls],
                       "stats": trace["stats"], "routes": trace["routing"],
                       "config_sha256": engine.hash, "code_sha256": engine.identity["code_sha256"], "store": store,  # fmt: skip
                       "verification": trace.get("verification")}  # fmt: skip
                spent += row["cost_usd"] + row["cost_extract_usd"]
                (args.out / "traces" / f"{task['id']}.json").write_text(json.dumps({**trace, "extracted": answer, "grade": graded}, indent=1, ensure_ascii=False, default=str) + "\n")  # fmt: skip
                with results.open("a") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                mark = "-- " if ungraded else "OK " if graded["correct"] else "BAD"
                print(f"{mark} {task['id']:45s} {result['seconds']:6.1f}s ${row['cost_usd']:.5f} routes={trace['routing']['chosen']} got={json.dumps(graded['actual'])[:80]} want={json.dumps(graded['expected'])[:80]}", flush=True)  # fmt: skip
        finally:
            await engine.aclose()
    rows = [r for r in jsonl(results) if not wanted or r["id"] in wanted]
    if ungraded:  # e2_engine_numbers.py reads the traces
        return
    summary = summarize(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("routes", "families")}, indent=1))  # fmt: skip


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--stores", type=Path, required=True, help="RAW export root (<ckpt>/ below)")  # fmt: skip
    parser.add_argument("--memory-root", type=Path, help="notes checkouts root (<ckpt>/ below)")
    parser.add_argument("--mode", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--endpoint", choices=sorted(ENDPOINTS), default="ollama")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", help="comma-separated task ids")
    parser.add_argument("--config", type=Path, help="JSON object of EngineConfig overrides")
    parser.add_argument("--max-usd", type=float, default=1.0)
    parser.add_argument("--questions", type=Path, help="invented questions (JSONL): no grading")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
