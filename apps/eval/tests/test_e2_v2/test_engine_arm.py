"""Stage 3 of the engine (E2-ENGINE-DESIGN.md): MCP kind `engine`, the H-engine arms, their pins and
their container arguments. The engine itself is tested in test_engine_*.py."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml

pytest.importorskip("fastmcp")

from fastmcp import Client  # noqa: E402
from fastmcp.client.transports import StdioTransport  # noqa: E402
from harnext_eval.e2.benchmark import canonical, sha  # noqa: E402
from harnext_eval.e2.v2 import mcp_server as server  # noqa: E402
from harnext_eval.e2.v2 import runner  # noqa: E402
from harnext_eval.e2.v2.agents import claude_code, codex, container  # noqa: E402
from harnext_eval.e2.v2.agents.base import McpSpec, TrialSpec  # noqa: E402
from harnext_eval.e2.v2.engine.clients import (  # noqa: E402
    ChatClient,
    FakeChat,
    FakeJev,
    InfraError,
    JevClient,
    text,
)
from harnext_eval.e2.v2.engine.config import for_endpoint  # noqa: E402
from harnext_eval.e2.v2.engine.engine import Engine  # noqa: E402
from harnext_eval.e2.v2.stores.base import ROOT  # noqa: E402

from .engine_fixtures import FakeDB  # noqa: E402

ARMS = ROOT / "benchmarks/kafka-500-v2/arms.yaml"
EXPORT = ROOT / "out/kafka-500-v2/stores-smoke/RAW/export/2023-02"
KEY = "sk-or-v1-TESTSENTINEL0123456789abcdef"
# Pins of every arm that existed before kind engine (arm_pin, 2026-09-28): adding a kind must not move them.
PINS = {  # arm_pin of every arm before kind engine (2026-09-28): adding a kind must not move one
    "F0": "64acba5d7347cc26f95cf6ecf0e536514a6c0e1f4e477a2137a5ac899c386c09",
    "F0-search": "44ea82bf3f1d20b3cf63be6b7ce412fedc78517dd17c01b73769176a88db42b0",
    "F1": "3f5b8dc1c7cc07b9b58a4f75e545459d9378532c8e8e024e1e00f33ef98b0283",
    "F1-search": "0f6cb3867a93001b7bf9d1cea5b392daee77bb66a2598024d6da9d70ccefb3a1",
    "F2": "faf20f01dcc5ca604c52cd29f8de93992bf86838a7d48fc9297b823388112461",
    "F2-search": "b3332d3b55c23281517c7522d414de33c1aa5279ae7c6fe74946d72f5edfc234",
    "F3": "48d50cbf3f8342f728386a616dfeda43c0f3f915c560156c60d8fa43d0ae9b66",
    "F3-search": "10b718003fad75fe05f804da8e9c2b32b9049003394aaea73113f59eb30c2425",
    "F4-entity": "df1439c26f6ac4f8a08fc2fa5746af982c2582e61de7c2a95d4f9ee3b892af9f",
    "F4-entity-search": "1cdb0b910f04fc860df0cd9e6c6641035bfb89960db252ebab994d38c84a6ba1",
    "F4-fact": "483d08ae0b90f6e18803424d7bfd7453b92232d2894562b9ec25f5dbe6ba6f8c",
    "F4-fact-search": "41ac41ba05dbb8ad36ef6196d0d6b5d74d710dfdc0030a4ae757c1d4740b5826",
    "F4-topic": "a4131becb4bfadd93321bdb3210a3561e427bced5e85715e362f3abd7c7ec653",
    "F4-topic-search": "59cfbb9d3f353b92f4e355175e9294f8edc1f6ff9cffd3f040e3d6fafcc01275",
    "F5": "da218f27a03608f7b56f56631cb66efab54862b3681b09a29182a00844359130",
    "F5-search": "7bfe8d7f58a86e34d5e038edf0570425b8a733616bcb1475ab77938d4d6e0d9b",
    "F6": "2abc8cf0ac0953db6ef2301f993bde31c078f3145310db233d7a84e68d156fbf",
    "F6-search": "4e37ad46c9157ad795a25c7ca7c166560ec51a59db46fb11e6ba7f997756b677",
    "N": "66816f4528f39cf69864fd7e529f6670ed2868b26e664ccee7a82aa499f9bd53",
    "R-hyb": "c15b0f4605f8045cc764b443e0c8b276db695aed27c7c33e985aea6f761bfdae",
    "S-tools": "03f6b364c37a793b3739c66781eb86f84c8a451cbebc795d0ebedff9904b78f7",
}
TEMPLATE = "5012cc3cc5d76cbb631b5ad340b865071406c5af4fe329818e2ce2d1231f856b"  # prompt_template_sha256 of every current run
MAIN_RUN = (
    ROOT.parents[1] / ".harnext/evaluations/kafka-500-v2/w2026-claude-code-haiku-4-5/manifest.json"
)


def fake_engine(fail=None):
    def agent(messages, tools, purpose):
        return (
            text("Report: it was Alice [x:1].")
            if purpose == "report"
            else text("Alice did it [x:1].")
        )

    return Engine(db=FakeDB(), config=for_endpoint("openrouter", backoff_s=0), jev=FakeJev(fail=fail), chat=FakeChat(agent))  # fmt: skip


def engine_session(tmp_path, engine=None, **budgets):
    backend = server.EngineBackend(EXPORT, clients="fake", trace=tmp_path / server.ENGINE_TRACE, engine=engine or fake_engine())  # fmt: skip
    return server.Session("engine", backend, tmp_path / "receipts.jsonl", **budgets)


def call(sess, **arguments):
    return json.loads(asyncio.run(sess.invoke("context_research", arguments)))


def listing(sess):
    async def names():
        async with Client(server.create_server(sess)) as client:
            return [(t.name, t.description) for t in await client.list_tools()]

    return asyncio.run(names())


def test_engine_serves_the_research_tool_unchanged(tmp_path):
    sess = engine_session(tmp_path)
    engine_tools = listing(sess)
    sess.close()
    (tmp_path / "research").mkdir()
    research = server.Session("research", server.ResearchBackend(tmp_path, reader="fake"), tmp_path / "research/r.jsonl")  # fmt: skip
    assert engine_tools == listing(research) and engine_tools[0][0] == "context_research"
    assert (
        server.KINDS["engine"] == server.KINDS["research"] and server.DEFAULT_CALLS["engine"] == 25
    )


def test_call_answers_like_research_and_receipts_carry_engine_accounting(tmp_path):
    sess = engine_session(tmp_path)
    out = call(sess, question="Who fixed ABC-1?")
    assert out["answer"].startswith("Report: it was Alice") and out["error"] is None
    assert set(out) == {"answer", "files", "seconds", "turns", "usage", "error", "truncated"}
    assert set(out["usage"]) == {"input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "cost_usd"}  # fmt: skip
    sess.close()
    start, row = server.verify(tmp_path / "receipts.jsonl")
    assert start["store"]["preset"] == "openrouter" and start["store"]["export"] == "2023-02"
    info = row["research"]
    assert {"files", "usage", "seconds", "turns", "partial", "answer_bytes", "config_sha256", "identity", "trace"} <= set(info)  # fmt: skip
    for model in ("jev", "chat"):
        assert {"calls", "input_tokens", "output_tokens", "cost_usd"} <= set(info["usage"][model])
        assert info["usage"][model]["calls"] >= 1
    assert info["usage"]["cost_usd"] == pytest.approx(info["usage"]["jev"]["cost_usd"] + info["usage"]["chat"]["cost_usd"])  # fmt: skip
    assert info["answer_bytes"] == len(out["answer"].encode()) and info["identity"]["config_sha256"] == info["config_sha256"]  # fmt: skip
    trace = [json.loads(line) for line in (tmp_path / server.ENGINE_TRACE).read_text().splitlines()]
    assert info["trace"] == {"file": server.ENGINE_TRACE, "line": 1, "sha256": server.digest(trace[0]["trace"])}  # fmt: skip
    assert trace[0]["question"] == "Who fixed ABC-1?" and "routing" in trace[0]["trace"]
    totals = server.engine_usage([start, row])
    assert totals["calls"] == 1 and totals["cost_usd"] == info["usage"]["cost_usd"]


def test_budget_is_25_calls_and_restart_continues_the_trace(tmp_path):
    sess = engine_session(tmp_path, max_calls=server.DEFAULT_CALLS["engine"])
    for _ in range(25):
        assert call(sess, question="q")["error"] is None
    assert call(sess, question="q")["error"] == "call budget exhausted"
    sess.close()
    again = engine_session(tmp_path, max_calls=25)  # a CLI restarting its MCP server mid-trial
    assert (
        again.backend.lines == 25 and call(again, question="q")["error"] == "call budget exhausted"
    )
    again.close()


def test_provider_failure_is_infra_and_pauses_the_runner(tmp_path):
    sess = engine_session(tmp_path, engine=fake_engine(fail=InfraError("HTTP 402: out of credits")))
    out = call(sess, question="q")
    assert out["answer"] is None and "context engine provider failure: HTTP 402" in out["error"]
    sess.close()
    rows = server.verify(tmp_path / "receipts.jsonl")
    assert rows[1]["infra"] is True
    assert "HTTP 402" in runner.reader_infra(tmp_path / "receipts.jsonl")


def test_unreachable_local_endpoint_is_an_infra_row_at_start(tmp_path):
    backend = server.EngineBackend(
        EXPORT, preset="ollama", chat_base_url="http://127.0.0.1:9/v1", key=KEY
    )
    assert "local chat endpoint http://127.0.0.1:9/v1 is unreachable" in backend.failed
    sess = server.Session("engine", backend, tmp_path / "receipts.jsonl")
    server.announce(sess)
    assert "unreachable" in call(sess, question="q")["error"]
    sess.close()
    rows = server.verify(tmp_path / "receipts.jsonl")
    assert rows[1]["kind"] == "infra" and rows[1]["infra"] is True and rows[2]["infra"] is True
    assert "unreachable" in runner.reader_infra(tmp_path / "receipts.jsonl")


def test_key_never_reaches_receipts_or_trace_even_when_a_provider_echoes_it(tmp_path):
    def echo(request):
        return httpx.Response(401, text=f"bad credentials {request.headers['authorization']}")

    config = for_endpoint("openrouter", backoff_s=0)
    http = lambda: httpx.AsyncClient(transport=httpx.MockTransport(echo))  # noqa: E731
    engine = Engine(db=FakeDB(), config=config, jev=JevClient(config, key=KEY, http=http()), chat=ChatClient(config, key=KEY, http=http()))  # fmt: skip
    backend = server.EngineBackend(
        EXPORT, key=KEY, trace=tmp_path / server.ENGINE_TRACE, engine=engine
    )
    sess = server.Session("engine", backend, tmp_path / "receipts.jsonl")
    out = call(sess, question="q")
    sess.close()
    assert "HTTP 401" in out["error"] and KEY not in json.dumps(out)
    assert KEY not in (tmp_path / "receipts.jsonl").read_text()
    assert not (tmp_path / server.ENGINE_TRACE).exists()


@pytest.mark.skipif(not EXPORT.exists(), reason="smoke RAW export not built")
def test_stdio_server_reads_and_deletes_the_key_file_and_answers(tmp_path):
    pytest.importorskip("chdb")
    secret = tmp_path / "key"
    secret.write_text(KEY)
    log = tmp_path / "mcp/receipts.jsonl"
    args = ["-m", "harnext_eval.e2.v2.mcp_server", "--kind", "engine", "--store", str(EXPORT), "--log", str(log),
            "--clients", "fake", "--key-file", str(secret), "--scratch", str(tmp_path)]  # fmt: skip

    async def run():
        async with Client(StdioTransport(sys.executable, args, keep_alive=False)) as client:
            listed = await client.list_tools()
            answer = await client.call_tool("context_research", {"question": "Who is assigned?"})
            return listed, json.loads(answer.content[0].text)  # type: ignore[union-attr]

    listed, answer = asyncio.run(run())
    assert [t.name for t in listed] == ["context_research"] and answer["answer"].startswith(
        "(fake engine)"
    )
    assert not secret.exists()  # read once at start, then gone
    rows = server.verify(log)
    assert rows[0]["budgets"]["calls"] == 25 and rows[1]["research"]["trace"]["line"] == 1
    assert KEY not in log.read_text() and KEY not in (log.parent / server.ENGINE_TRACE).read_text()
    for _ in range(50):  # the client does not wait for the server to exit
        if not list(tmp_path.glob("harnext-raw-*")):
            break
        time.sleep(0.1)
    assert not list(tmp_path.glob("harnext-raw-*"))  # the export's scratch database is removed


def test_engine_arms_load_with_the_research_texts():
    arms, srv = runner.load_arms(ARMS)
    by = {a["id"]: a for a in arms}
    assert by["H-engine"] == {"id": "H-engine", "access": "mcp", "mcp_kind": "engine", "store": "RAW", "engine": {"preset": "openrouter"}}  # fmt: skip
    assert by["H-engine-mem"]["memory_store"] == "F5" and by["H-engine-mem"]["store"] == "RAW"
    for task in ({"prompt": "Q?", "kind": "historical"}, {"prompt": "Q?", "kind": "coding"}):
        assert runner.build_prompt(task, by["H-engine"], srv) == runner.build_prompt(
            task, by["F5"], srv
        )
    assert runner.standing_instructions(by["H-engine-mem"], srv) == runner.standing_instructions(
        by["F5"], srv
    )
    pinned = runner.arm_pin(by["H-engine"], srv)
    assert pinned == sha(canonical([by["H-engine"], srv, runner.ACCESS["mcp:research"], runner.GATHER_FIRST["research"]]))  # fmt: skip
    assert runner.arm_pin(runner.with_preset(by["H-engine"], "ollama"), srv) != pinned
    assert runner.with_preset(by["F5"], "ollama") is by["F5"]


def test_existing_pins_and_prompt_template_do_not_move():
    arms, srv = runner.load_arms(ARMS)
    assert {a["id"]: runner.arm_pin(a, srv) for a in arms if a["mcp_kind" if "mcp_kind" in a else "id"] != "engine"} == PINS  # fmt: skip
    assert sha(canonical([runner.ACCESS, runner.CONTRACT, runner.GATHER_FIRST])) == TEMPLATE
    assert "mcp:engine" not in runner.ACCESS and "engine" not in runner.GATHER_FIRST
    if MAIN_RUN.exists():  # the recorded pins of the main run still match
        recorded = json.loads(MAIN_RUN.read_text())
        assert recorded["prompt_template_sha256"] == TEMPLATE
        assert all(PINS[a] == pin for a, pin in recorded["arms"].items())


@pytest.mark.parametrize("bad", [
    {"id": "X", "access": "mcp", "mcp_kind": "research", "store": "F5", "memory_store": "F5"},
    {"id": "X", "access": "mcp", "mcp_kind": "research", "store": "F5", "engine": {"preset": "openrouter"}},
    {"id": "X", "access": "mcp", "mcp_kind": "engine", "store": "RAW", "engine": {"preset": "somewhere"}},
])  # fmt: skip
def test_engine_fields_are_validated(tmp_path, bad):
    path = tmp_path / "arms.yaml"
    path.write_text(yaml.safe_dump({"arms": [bad]}))
    with pytest.raises(ValueError, match="engine"):
        runner.load_arms(path)


def trial(tmp_path, **mcp):
    spec = McpSpec("engine", tmp_path / "raw", "history", ("context_research",), **mcp)
    return TrialSpec(task_id="t", arm="H-engine", kind="historical", prompt="p", answer_schema={},
                     workspace=tmp_path / "w", out=tmp_path / "out", model="m", effort="medium", timeout=1, mcp=spec)  # fmt: skip


def test_container_arguments_for_the_engine_kind(tmp_path, monkeypatch):
    monkeypatch.setattr(container, "gateway", lambda: "172.19.0.1")
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    spec = trial(tmp_path, memory=tmp_path / "notes", engine="openrouter")
    home = container.prepare_home(spec, [])
    command = claude_code.command(spec)
    argv = container.run_argv(spec, "n", command, auth=container.reader_auth(spec), home=home)
    joined = " ".join(argv)
    assert f"{tmp_path / 'raw'}:/store:ro" in argv and f"{tmp_path / 'notes'}:/memory:ro" in argv
    assert f"{tmp_path / 'out/engine-key'}:/run/harnext-engine" in argv
    assert "--tmpfs /scratch:rw,nosuid,nodev,size=512m,mode=1777" in joined
    assert (
        "--memory 10g --cpus 4 --pids-limit 4096" in joined
        and f"--user {os.getuid()}:{os.getgid()}" in joined
    )
    assert "--cap-drop ALL" in joined and "MCP_TOOL_TIMEOUT=678000" in argv
    assert container.reader_auth(spec) == []  # no Claude login for the engine
    key = tmp_path / "out/engine-key/key"
    assert key.read_text() == KEY and oct(key.stat().st_mode & 0o777) == "0o600"
    assert KEY not in joined and not any("benchmarks" in a or "private" in a for a in argv)
    mcp = json.loads(command[command.index("--mcp-config") + 1])["mcpServers"]["history"]
    assert mcp["args"][-8:] == ["--preset", "openrouter", "--scratch", "/scratch", "--key-file", "/run/harnext-engine/key", "--memory", "/memory"]  # fmt: skip
    assert mcp["env"]["NO_PROXY"] == mcp["env"]["no_proxy"] == "localhost,127.0.0.1,172.19.0.1"
    assert mcp["env"]["HTTPS_PROXY"] == container.PROXY_URL and KEY not in json.dumps(mcp)
    assert "mcp_servers.history.tool_timeout_sec=678" in codex.command(spec)
    local = trial(tmp_path, engine="ollama")
    assert container.mcp_command(local.mcp)[1][-2:] == [
        "--chat-base-url",
        "http://172.19.0.1:11434/v1",
    ]
    offline = trial(tmp_path / "f", reader="fake")
    args = container.mcp_command(offline.mcp)[1]
    assert "--clients" in args and "--key-file" not in args
    assert "/run/harnext-engine" not in " ".join(container.run_argv(offline, "n", ["x"], auth=[], home=tmp_path / "f/home"))  # fmt: skip
    assert "https://openrouter.ai/" in container.ALLOWED


def test_missing_key_pauses_instead_of_starting(tmp_path, monkeypatch):
    monkeypatch.setattr(container, "gateway", lambda: "172.19.0.1")
    monkeypatch.setattr("harnext_eval.e2.v2.engine.clients.load_key", lambda name: None)
    with pytest.raises(container.PauseRun, match="OPENROUTER_API_KEY missing"):
        container.run_argv(trial(tmp_path), "n", ["x"], auth=[], home=tmp_path / "home")


# ---- review fixes (2026-09-28): pinning, failures that must pause, drift subset, key grace ----


@pytest.fixture
def engine_bench(tmp_path):
    """Three historical tasks at 2023-02 under N and H-engine, the real smoke RAW export as store."""
    pytest.importorskip("chdb")
    if not EXPORT.exists():
        pytest.skip("smoke RAW export not built")
    directory = tmp_path / "bench"
    directory.mkdir()
    public = [{"id": f"H-{i}", "kind": "historical", "family": "status_at_snapshot", "checkpoint": "2023-02",
               "prompt": f"Q{i}?", "answer_schema": {"type": "object"}} for i in range(3)]  # fmt: skip
    private = [
        {"id": f"H-{i}", "expected": {"value": f"v{i}"}, "evidence_ids": []} for i in range(3)
    ]
    for name, rows in (("tasks.public.jsonl", public), ("tasks.private.jsonl", private)):
        (directory / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    files = {
        n: sha((directory / n).read_bytes()) for n in ("tasks.public.jsonl", "tasks.private.jsonl")
    }
    (directory / "manifest.json").write_text(
        json.dumps({"files": files, "run_order": [t["id"] for t in public]})
    )
    arms = [{"id": "N", "access": "none"},
            {"id": "H-engine", "access": "mcp", "mcp_kind": "engine", "store": "RAW", "engine": {"preset": "openrouter"}}]  # fmt: skip
    (directory / "arms.yaml").write_text(
        yaml.safe_dump({"mcp_server": {"name": "history"}, "arms": arms})
    )
    (tmp_path / "stores").mkdir()
    (tmp_path / "stores/RAW").symlink_to(EXPORT.parents[1])

    def config(run_id, **extra):
        return runner.Config(
            protocol="harnext-e2-v2", run_id=run_id, benchmark=directory, arms_file=directory / "arms.yaml", agent="fake", model="m",
            reader=extra.pop("reader", "fake"), stores=tmp_path / "stores", results=tmp_path / "results" / run_id,
            scratch=tmp_path / "scratch" / run_id, workspaces=tmp_path / "ws", checkouts=tmp_path / "co",
            workers=extra.pop("workers", 1), min_free_gib=0, **extra,
        )  # fmt: skip

    return config


def summary(config):
    return json.loads((config.results / "summary.json").read_text())


def test_engine_run_pins_code_and_config_and_records(engine_bench):
    from harnext_eval.e2.v2 import engine_pin
    from harnext_eval.e2.v2.agents.fake import FakeAgent

    config = engine_bench("full")
    assert runner.Runner(config, FakeAgent()).live() == 0
    pin = json.loads((config.results / "manifest.json").read_text())["engine"]
    host = engine_pin.identity(["openrouter"])
    assert pin["image_id"] == "host (no container)" and pin["image_overrides"] == []
    assert pin["code"] == {k: v for k, v in host.items() if k != "config_sha256"}
    assert pin["config_sha256"] == host["config_sha256"]
    record = json.loads((config.results / "trials/H-0/H-engine/result.json").read_text())
    assert record["engine"]["calls"] == 1 and "unmetered_upper_usd" in record["engine"]
    assert runner.Runner(config, FakeAgent()).live() == 0  # resume: same pin


@pytest.mark.parametrize(("mode", "reason"), [
    ("bug", "context engine error: RuntimeError"),
    ("failed", "not connected at start"),
])  # fmt: skip
def test_run_wide_engine_failures_pause_the_run(engine_bench, mode, reason):
    from harnext_eval.e2.v2.agents.fake import FakeAgent

    config = engine_bench(f"fail-{mode}")
    agent = FakeAgent(engine_modes={2: mode})  # call 2 = H-engine on the first task
    assert runner.Runner(config, agent).live() == runner.EXIT_PAUSED
    paused = summary(config)
    assert paused["status"] == "paused" and reason in paused["reason"], paused["reason"]
    assert (
        paused["not_recorded"] == ["H-0/H-engine"]
        and not (config.results / "trials/H-0/H-engine").exists()
    )
    kept = list((config.results / "paused").glob("H-0-H-engine-*"))  # evidence kept for diagnosis
    assert len(kept) == 1 and (kept[0] / "trace.jsonl").exists()
    assert runner.Runner(config, FakeAgent()).live() == 0 and summary(config)["recorded"] == 6


@pytest.mark.parametrize(("mode", "reason"), [
    ("slow", "ran out of time with"),  # every server attempt degraded: provider trouble outlived the redos
    ("dead", "went away during context_research"),
    ("timeout", "timed out on the agent side"),
])  # fmt: skip
def test_trial_scoped_engine_failures_redo_that_trial_only(engine_bench, mode, reason):
    from harnext_eval.e2.v2.agents.fake import FakeAgent

    config = engine_bench(f"local-{mode}")
    agent = FakeAgent(engine_modes={2: mode})
    assert runner.Runner(config, agent).live() == 0  # the run went on and redid the trial
    done = summary(config)
    assert done["status"] == "complete" and done["recorded"] == 6
    session = json.loads(max((config.results / "sessions").glob("*.json")).read_text())
    assert [p["trial"] for p in session["trial_pauses"]] == ["H-0/H-engine"] and session[
        "deferred"
    ] == []
    assert reason in session["trial_pauses"][0]["reason"]
    assert len(list((config.results / "paused").glob("H-0-H-engine-*"))) == 1


def test_engine_bug_receipt_carries_the_traceback(tmp_path):
    sess = engine_session(tmp_path, engine=fake_engine(fail=RuntimeError("boom")))
    out = call(sess, question="q")
    sess.close()
    row = server.verify(tmp_path / "receipts.jsonl")[1]
    assert out["error"] == "context engine error: RuntimeError: boom" and row["infra"] is True
    assert (
        "Traceback" in row["research"]["traceback"]
        and "RuntimeError: boom" in row["research"]["traceback"]
    )


def trouble(partial=True, errors=(), retries=0, calls=(), answer="a"):
    branches = [
        {"source": "jira", "status": "failed" if e.startswith(("RuntimeError", "BudgetExceeded")) else "partial", "error": e}
        for e in errors
    ]  # fmt: skip
    usage = {"jev": {"retries": retries}, "chat": {"retries": 0}}
    trace = {"branches": branches, "calls": list(calls)}
    return server.engine_trouble(
        {"partial": partial, "answer": answer, "usage": usage, "trace": trace}
    )


def test_partial_answers_are_infra_only_with_time_and_provider_trouble():
    cut = {"kind": "chat", "cancelled": True, "seconds": 8.0}  # in flight at the deadline: normal
    hung = {"kind": "chat", "cancelled": True, "seconds": 61.0}  # ran past its own request timeout
    assert trouble(errors=["turn budget (8 turns)"], retries=3) is None  # the engine's own budget
    assert trouble(errors=["time budget (120 s)"], calls=[cut]) is None  # slow, no provider fault
    assert (
        trouble(
            errors=["BudgetExceeded: time budget (120 s) ran out during a model call"], calls=[cut]
        )
        is None
    )
    assert "1 failed, retried or timed-out" in trouble(errors=["time budget (120 s)"], calls=[hung])
    assert "2 failed" in trouble(errors=["time budget (120 s)"], retries=1, calls=[hung, cut])
    assert trouble(
        errors=["time budget (120 s)"], calls=[{"kind": "jev", "cancelled": True, "seconds": 12.5}]
    )
    assert trouble(
        answer="(report not written: time budget) x",
        calls=[{"kind": "jev", "failures": ["HTTP 503"]}],
    )
    assert trouble(partial=False, retries=2) is None
    assert "branch jira failed: RuntimeError" in trouble(partial=False, errors=["RuntimeError: x"])


def test_mcp_trouble_reads_codex_traces_too(tmp_path):
    trace, receipts = tmp_path / "trace.jsonl", tmp_path / "receipts.jsonl"
    item = {"id": "i1", "type": "mcp_tool_call", "server": "history", "tool": "context_research", "status": "completed", "error": None}  # fmt: skip
    trace.write_text(json.dumps({"type": "item.completed", "item": item}) + "\n")
    assert "1 context_research calls in the agent trace, 0 in the receipts" in runner.mcp_trouble(trace, receipts, "history", "completed")  # fmt: skip
    assert (
        runner.mcp_trouble(trace, receipts, "history", "timeout") is None
    )  # killed mid-call: a model result
    receipts.write_text(json.dumps({"kind": "call", "tool": "context_research"}) + "\n")
    assert runner.mcp_trouble(trace, receipts, "history", "completed") is None
    failed = item | {"status": "failed", "error": {"message": "tool call timed out after 240s"}}
    trace.write_text(json.dumps({"type": "item.completed", "item": failed}) + "\n")
    assert "timed out" in runner.mcp_trouble(trace, receipts, "history", "completed")


def test_engine_pin_refuses_changed_code_config_or_image(engine_bench, monkeypatch):
    run = runner.Runner(engine_bench("pin"))
    code = {"engine_code_sha256": "a", "raw_store_code_sha256": "b", "mcp_server_sha256": "c"}
    pinned = {
        "image_id": "sha256:1",
        "code": code,
        "config_sha256": {"openrouter": "x"},
        "image_overrides": [],
    }
    same = code | {"config_sha256": {"openrouter": "x"}}
    monkeypatch.setattr(run, "engine_identity", lambda: ("sha256:1", same, same))
    assert run.engine_pin(pinned)["image_id"] == "sha256:1"
    monkeypatch.setattr(
        run, "engine_identity", lambda: ("sha256:1", same | {"engine_code_sha256": "z"}, same)
    )
    with pytest.raises(ValueError, match="engine code in the image differs"):
        run.engine_pin(pinned)
    monkeypatch.setattr(
        run,
        "engine_identity",
        lambda: ("sha256:1", code | {"config_sha256": {"openrouter": "y"}}, same),
    )
    with pytest.raises(ValueError, match="engine config differs"):
        run.engine_pin(pinned)
    monkeypatch.setattr(run, "engine_identity", lambda: ("sha256:2", same, same))
    with pytest.raises(ValueError, match="--allow-image-change"):
        run.engine_pin(pinned)
    run.allow_image_change = True
    moved = run.engine_pin(pinned)
    assert moved["image_id"] == "sha256:2" and moved["image_overrides"][0]["from"] == "sha256:1"
    server_only = same | {"mcp_server_sha256": "c2"}
    monkeypatch.setattr(run, "engine_identity", lambda: ("sha256:3", server_only, server_only))
    wrapped = run.engine_pin(pinned)  # the MCP wrapper alone may change, with the flag, recorded
    assert (
        wrapped["code"]["mcp_server_sha256"] == "c2"
        and wrapped["image_overrides"][-1]["code_from"] == code
    )
    run.allow_image_change = False
    with pytest.raises(ValueError, match="only a change of mcp_server.py alone"):
        run.engine_pin(pinned)
    run.allow_image_change = True
    both = same | {"mcp_server_sha256": "c2", "raw_store_code_sha256": "b2"}
    monkeypatch.setattr(run, "engine_identity", lambda: ("sha256:3", both, both))
    with pytest.raises(ValueError, match="engine code in the image differs"):
        run.engine_pin(pinned)
    monkeypatch.setattr(
        run, "engine_identity", lambda: ("sha256:1", same, same | {"engine_code_sha256": "tree"})
    )
    with pytest.raises(ValueError, match="rebuild the image"):
        run.engine_pin(None)  # first use: a stale image is refused


def test_every_nth_keeps_a_fixed_subset_per_category():
    public = {f"C{i}": {"kind": "coding"} for i in range(4)} | {f"H{i}": {"kind": "historical", "family": "x"} for i in range(5)}  # fmt: skip
    public |= {f"D{i}": {"kind": "historical", "family": "design_decision"} for i in range(3)}
    order = ["C0", "H0", "D0", "H1", "C1", "H2", "D1", "H3", "C2", "H4", "D2", "C3"]
    rows = runner.thin(
        runner.plan(order, [{"id": "S-tools"}, {"id": "H-engine"}]), public, {"S-tools": 3}
    )
    kept = [t for _, t, arms in rows if "S-tools" in arms]
    assert kept == ["C0", "H0", "D0", "H3", "C3"]  # positions 0 and 3 of each category
    assert all("H-engine" in arms for _, _, arms in rows)


def test_key_file_is_removed_after_the_grace_period(tmp_path, monkeypatch):
    monkeypatch.setattr(container, "ENGINE_KEY_GRACE_S", 0.2)
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    folder = container.engine_key(trial(tmp_path))
    assert (folder / "key").read_text() == KEY
    for _ in range(50):
        if not (folder / "key").exists():
            break
        time.sleep(0.05)
    assert not (folder / "key").exists()


def test_engine_pin_measures_the_same_when_piped_as_the_image_runs_it():
    import subprocess

    from harnext_eval.e2.v2 import engine_pin

    script = (Path(server.__file__).parent / "engine_pin.py").read_text()
    done = subprocess.run([sys.executable, "-", "openrouter"], input=script, capture_output=True, text=True, check=True)  # fmt: skip
    assert json.loads(done.stdout) == engine_pin.identity(["openrouter"])


def test_a_completed_run_refuses_new_arms_unless_extend_run(engine_bench, tmp_path):
    from harnext_eval.e2.v2.agents.fake import FakeAgent

    config = engine_bench("finished", arms=["N"])
    assert (
        runner.Runner(config, FakeAgent()).live() == 0 and summary(config)["status"] == "complete"
    )
    grown = config.model_copy(update={"arms": ["N", "H-engine"]})
    with pytest.raises(
        ValueError, match="is complete; the config adds arms it never had: \\['H-engine'\\]"
    ):
        runner.Runner(grown, FakeAgent()).live()
    assert runner.Runner(config, FakeAgent()).live() == 0  # same arms: resumes (nothing to do)
    extended = runner.Runner(grown, FakeAgent())
    extended.extend_run = True
    assert extended.live() == 0 and summary(grown)["recorded"] == 6
    unfinished = engine_bench("unfinished", arms=["N"])
    assert runner.Runner(unfinished, FakeAgent(infra_at=2)).live() == runner.EXIT_PAUSED
    assert (
        runner.Runner(unfinished.model_copy(update={"arms": ["N", "H-engine"]}), FakeAgent()).live()
        == 0
    )


def test_jev_health_probe_flags_stalls_and_errors(monkeypatch):
    import harnext_eval.e2.v2.engine.config as engine_config

    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    fast = engine_config.for_endpoint("openrouter", jev_timeout_s=0.2)  # the limit, scaled down
    monkeypatch.setattr(engine_config, "for_endpoint", lambda name, **kw: fast)
    real = httpx.AsyncClient

    def client_with(handler):
        return lambda *a, **kw: real(transport=httpx.MockTransport(handler))

    async def ok(request):
        return httpx.Response(200, json={"answers": {}})

    async def stall(request):
        await asyncio.sleep(0.3)  # past the 0.2 s request timeout
        return httpx.Response(200, json={"answers": {}})

    monkeypatch.setattr(httpx, "AsyncClient", client_with(ok))
    assert container.jev_unhealthy(calls=3, spacing_s=0) is None
    monkeypatch.setattr(
        httpx, "AsyncClient", client_with(lambda r: httpx.Response(503, text="upstream"))
    )
    assert "failed" in container.jev_unhealthy(calls=3, spacing_s=0)
    monkeypatch.setattr(httpx, "AsyncClient", client_with(stall))
    why = container.jev_unhealthy(calls=3, spacing_s=0)
    assert "Jev call 1/3 took" in why and "(request timeout 0.2 s)" in why and KEY not in why


def test_a_stalling_provider_pauses_before_any_trial(engine_bench, monkeypatch):
    from harnext_eval.e2.v2.agents.fake import FakeAgent

    class Live(FakeAgent):
        needs_container = True

    config = engine_bench("gated", reader="claude")
    run = runner.Runner(config, Live())
    monkeypatch.setattr(run, "preflight", lambda live: {})
    monkeypatch.setattr(
        container, "jev_unhealthy", lambda: "provider unhealthy: Jev call 2/8 took 25.1 s"
    )
    assert run.live() == runner.EXIT_PAUSED and "took 25.1 s" in run.reason
    assert not run.agent.seen and not config.results.exists()  # no trial started, nothing written
    fake = runner.Runner(engine_bench("gated-fake"), Live())
    assert fake.provider_gate() is None  # fake models: no probe


# ---- provider trouble: redo inside the MCP server, trial-scoped pauses, deferred trials ----


def fake_session(tmp_path, **kw):
    backend = server.EngineBackend(
        EXPORT, clients="fake", scratch=tmp_path, trace=tmp_path / server.ENGINE_TRACE, **kw
    )
    return server.Session("engine", backend, tmp_path / "receipts.jsonl", max_calls=25), backend


@pytest.mark.skipif(not EXPORT.exists(), reason="smoke RAW export not built")
def test_a_degraded_attempt_is_redone_and_only_the_clean_answer_is_shown(tmp_path):
    pytest.importorskip("chdb")
    sess, backend = fake_session(tmp_path)
    out = call(sess, question="!flaky who?")
    assert out["error"] is None and out["answer"].startswith("(fake engine)")
    again = call(sess, question="!flaky503 who?")  # a transient InfraError (HTTP 503) is redone too
    assert again["error"] is None
    sess.close()
    asyncio.run(backend.engine.aclose())
    rows = server.verify(tmp_path / "receipts.jsonl")
    first, second = rows[1]["research"], rows[2]["research"]
    assert "infra" not in rows[1] and "infra" not in rows[2]
    (redo,) = first["attempts"]
    assert (
        redo["attempt"] == 1
        and "ran out of time" in redo["reason"]
        and redo["wait_after"]["healthy"]
    )
    assert (
        redo["usage"]["cost_usd"] > 0 and first["provider_retry_usd"] == redo["usage"]["cost_usd"]
    )
    assert (
        first["usage"]["cost_usd"] > 0 and first["trace"]["line"] == 2
    )  # the answer's own trace line
    assert "HTTP 503" in second["attempts"][0]["reason"]
    lines = [json.loads(line) for line in (tmp_path / server.ENGINE_TRACE).read_text().splitlines()]
    assert [x.get("attempt", 1) for x in lines] == [1, 2, 2] and "ran out of time" in lines[0][
        "redo"
    ]
    totals = server.engine_usage(rows)
    assert totals["discarded_attempts"] == 2 and totals["provider_retry_usd"] == round(
        redo["usage"]["cost_usd"], 8
    )


@pytest.mark.skipif(not EXPORT.exists(), reason="smoke RAW export not built")
def test_trouble_in_every_attempt_pauses_this_trial_only(tmp_path):
    pytest.importorskip("chdb")
    sess, backend = fake_session(tmp_path)
    out = call(sess, question="!slow who?")
    sess.close()
    asyncio.run(backend.engine.aclose())
    assert out["answer"] is None and "(in all 3 attempts)" in out["error"]
    row = server.verify(tmp_path / "receipts.jsonl")[1]
    assert row["infra"] is True and row["infra_scope"] == "trial"
    assert [a["attempt"] for a in row["research"]["attempts"]] == [1, 2, 3]
    assert runner.reader_infra_scope(tmp_path / "receipts.jsonl") == "trial"
    assert server.engine_usage([row])["provider_retry_usd"] > 0  # counted though no answer came


def test_non_transient_failures_pause_the_run_at_once(tmp_path):
    for fail, scope in (
        (InfraError("HTTP 402: insufficient credits"), "run"),
        (RuntimeError("boom"), "run"),
    ):
        (tmp_path / "receipts.jsonl").unlink(missing_ok=True)
        sess = engine_session(tmp_path, engine=fake_engine(fail=fail))
        out = call(sess, question="q")
        sess.close()
        row = server.verify(tmp_path / "receipts.jsonl")[1]
        assert (
            out["answer"] is None
            and row["infra_scope"] == scope
            and row["research"]["attempts"] == []
        )
        assert runner.reader_infra_scope(tmp_path / "receipts.jsonl") == "run"


def test_transient_and_fatal_provider_texts():
    for message in ("https://openrouter.ai/api/alpha/decisions: timeout after 12 s (TimeoutError) (after 4 attempts)",
                 "HTTP 429: rate limited", "HTTP 503: upstream", "HTTP 520: web server returned an unknown error",
                 "endpoint unreachable (ConnectError: ...)", "provider error 502: {}"):  # fmt: skip
        assert server.transient(message), message
    for message in ("HTTP 401: no auth", "HTTP 402: insufficient credits", "HTTP 403: forbidden", "HTTP 404: none",
                 "provider rejected a request: HTTP 400: bad", "no API key for Jev ($OPENROUTER_API_KEY)",
                 "something unexpected"):  # fmt: skip
        assert not server.transient(message), message


def test_health_wait_backs_off_and_gives_up(tmp_path):
    backend = server.EngineBackend(
        EXPORT, trace=tmp_path / server.ENGINE_TRACE, engine=fake_engine()
    )
    backend.retry = {**server.ENGINE_RETRY, "backoff_s": (0.01, 0.02, 0.03), "wait_max_s": 1.0}
    answers = iter([False, False, True])

    async def probe():
        return {"healthy": next(answers), "seconds": [0.1]}

    backend.probe = probe
    wait = asyncio.run(backend.wait_healthy())
    assert wait["healthy"] and len(wait["probes"]) == 3

    async def sick():
        return {"healthy": False, "seconds": [4.0]}

    backend.probe = sick
    wait = asyncio.run(backend.wait_healthy())
    assert (
        not wait["healthy"] and len(wait["probes"]) == 4
    )  # first probe + three waits, then give up
    assert server.ENGINE_RETRY["attempts"] == 3 and server.ENGINE_RETRY["wait_max_s"] == 90


def test_tool_timeout_covers_every_redo_per_kind(tmp_path, monkeypatch):
    from harnext_eval.e2.v2.engine.config import EngineConfig

    assert server.ENGINE_DEADLINE_S == EngineConfig().deadline_s
    assert server.ENGINE_TOOL_TIMEOUT_S == 3 * (120 + 10) + 2 * (90 + 3 * 8) + 60 == 678
    w2026 = runner.load(ROOT / "configs/e2-w2026-engine-claude-haiku.yaml")
    assert (
        server.ENGINE_TOOL_TIMEOUT_S < w2026.timeouts.historical == 900
    )  # 222 s left for the agent
    assert (
        container.tool_timeout_ms("engine") == 678_000
        and container.tool_timeout_ms("research") == 240_000
    )
    assert container.tool_timeout_ms("research_search") == 240_000
    monkeypatch.setattr(container, "gateway", lambda: "172.19.0.1")
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    engine = trial(tmp_path / "e")
    assert "MCP_TOOL_TIMEOUT=678000" in container.run_argv(
        engine, "n", ["x"], auth=[], home=tmp_path / "e/home"
    )
    assert "mcp_servers.history.tool_timeout_sec=678" in codex.command(engine)
    research = TrialSpec(task_id="t", arm="F5", kind="historical", prompt="p", answer_schema={}, workspace=tmp_path / "w",
                         out=tmp_path / "r", model="m", effort="medium", timeout=1,
                         mcp=McpSpec("research", tmp_path / "s", "history", ("context_research",), "fake"))  # fmt: skip
    assert "MCP_TOOL_TIMEOUT=240000" in container.run_argv(
        research, "n", ["x"], auth=[], home=tmp_path / "r/home"
    )
    assert "mcp_servers.history.tool_timeout_sec=240" in codex.command(research)


def test_granular_pause_redoes_one_trial_after_the_gate_while_the_others_finish(
    engine_bench, monkeypatch
):
    from harnext_eval.e2.v2.agents.fake import FakeAgent

    config = engine_bench("granular", workers=2)
    run = runner.Runner(config, FakeAgent(engine_modes={"H-0|H-engine": "slow*1"}))
    gates = []
    monkeypatch.setattr(run, "provider_gate", lambda: gates.append(1))
    assert run.live() == 0
    done = summary(config)
    assert (
        done["status"] == "complete"
        and done["recorded"] == 6
        and len(gates) == 2  # at start + once after the pause
    )  # one gate, no sleep
    session = json.loads(max((config.results / "sessions").glob("*.json")).read_text())
    assert [(p["trial"], p["pause"], p["deferred"]) for p in session["trial_pauses"]] == [
        ("H-0/H-engine", 1, False)
    ]
    assert (
        run.agent.seen.count(("H-0", "H-engine")) == 2 and run.agent.seen.count(("H-0", "N")) == 1
    )


def test_a_trial_paused_three_times_is_deferred_to_the_end(engine_bench, monkeypatch):
    from harnext_eval.e2.v2.agents.fake import FakeAgent

    config = engine_bench("deferred", workers=2)
    run = runner.Runner(config, FakeAgent(engine_modes={"H-0|H-engine": "slow"}))
    monkeypatch.setattr(run, "provider_gate", lambda: None)
    assert run.live() == runner.EXIT_PAUSED and "deferred" in run.reason
    paused = summary(config)
    assert paused["recorded"] == 5 and paused["not_recorded"] == ["H-0/H-engine"]
    session = json.loads(max((config.results / "sessions").glob("*.json")).read_text())
    assert [p["pause"] for p in session["trial_pauses"]] == [
        1,
        2,
        3,
        4,
    ]  # the 4th: the deferred last round
    assert session["deferred"] == ["H-0/H-engine"]
    assert [p["deferred"] for p in session["trial_pauses"]] == [False, False, True, True]
    order = [t for t in run.agent.seen if t != ("H-0", "H-engine")]
    assert (
        len(order) == 5
    )  # every other trial ran (and was recorded) before the deferred one's last try


def test_an_unhealthy_gate_after_a_trial_pause_pauses_the_run(engine_bench, monkeypatch):
    from harnext_eval.e2.v2.agents.fake import FakeAgent

    config = engine_bench("gate-down")
    run = runner.Runner(config, FakeAgent(engine_modes={2: "slow"}))
    answers = iter(
        [None, "provider unhealthy: Jev call 1/8 took 25 s"]
    )  # healthy at start, then not
    monkeypatch.setattr(run, "provider_gate", lambda: next(answers))
    assert run.live() == runner.EXIT_PAUSED
    assert (
        "after a trial pause: provider unhealthy" in run.reason and summary(config)["recorded"] == 1
    )


# ---- engine v2 arms (29 Sep): -b8 presets, deadline-derived redo policy, chat probe, budget endings ----


def test_redo_policy_and_tool_timeout_follow_the_preset_deadline(tmp_path, monkeypatch):
    from harnext_eval.e2.v2.engine.config import ENDPOINTS, for_endpoint

    assert set(server.ENGINE_PRESETS) == set(ENDPOINTS)
    v1, v2 = server.engine_policy(120.0), server.engine_policy(240.0)
    assert (v1["attempts"], v1["tool_timeout_s"]) == (3, 678)  # H-engine: exactly as before
    assert (v2["attempts"], v2["tool_timeout_s"]) == (2, 674)  # 3 attempts would need 1,038 s
    assert 900 - v2["tool_timeout_s"] >= server.AGENT_ROOM_S == 120
    assert {p: for_endpoint(p).deadline_s for p in ("openrouter", "openrouter-b8", "deepseek-flash-b8")} == {
        "openrouter": 120.0, "openrouter-b8": 240.0, "deepseek-flash-b8": 240.0}  # fmt: skip
    assert (
        container.tool_timeout_ms("engine")
        == container.tool_timeout_ms("engine", "openrouter")
        == 678_000
    )
    assert (
        container.tool_timeout_ms("engine", "openrouter-b8")
        == container.tool_timeout_ms("engine", "deepseek-flash-b8")
        == 674_000
    )
    assert container.tool_timeout_ms("research") == 240_000
    monkeypatch.setattr(container, "gateway", lambda: "172.19.0.1")
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    b8 = trial(tmp_path / "b8", engine="deepseek-flash-b8")
    assert "MCP_TOOL_TIMEOUT=674000" in container.run_argv(
        b8, "n", ["x"], auth=[], home=tmp_path / "b8/home"
    )
    assert "mcp_servers.history.tool_timeout_sec=674" in codex.command(b8)
    assert container.mcp_command(b8.mcp)[1][-6:-4] == ["--preset", "deepseek-flash-b8"]
    backend = server.EngineBackend(
        EXPORT, preset="deepseek-flash-b8", trace=tmp_path / "t", engine=fake_engine()
    )
    assert backend.retry["attempts"] == 2 and backend.retry["tool_timeout_s"] == 674
    assert (
        server.EngineBackend(EXPORT, trace=tmp_path / "u", engine=fake_engine()).retry["attempts"]
        == 3
    )


def v2_trouble(ending, error="", retries=0, budget=True):
    branches = [{"source": "jira", "status": "done", "ending": ending, "error": error}]
    usage = {"jev": {"retries": retries}, "chat": {"retries": 0}}
    trace = {"identity": {"budget_feedback": budget}, "branches": branches, "calls": []}
    return server.engine_trouble({"partial": True, "answer": "a", "usage": usage, "trace": trace})


def test_budget_feedback_endings_are_classified_by_time_pressure_only():
    assert v2_trouble("budget", retries=2) is None  # used its 8 tool calls and answered: normal
    assert v2_trouble("forced", retries=2) is None  # made to answer after its answer turns: normal
    assert (
        v2_trouble("time") is None and v2_trouble("cutoff") is None
    )  # slow, but no provider fault
    assert "ran out of time" in v2_trouble("time", retries=1)
    assert "ran out of time" in v2_trouble("cutoff", retries=1)
    assert "ran out of time" in v2_trouble(
        "normal", error="no answer at the deadline: the final call timed out", retries=1
    )
    assert (
        v2_trouble("cutoff", retries=1, budget=False) is None
    )  # engine v1: only "time budget" counts, as before


@pytest.mark.skipif(not EXPORT.exists(), reason="smoke RAW export not built")
def test_a_b8_preset_redoes_once_and_receipts_carry_the_new_usage_fields(tmp_path):
    pytest.importorskip("chdb")
    backend = server.EngineBackend(EXPORT, preset="openrouter-b8", clients="fake", scratch=tmp_path, trace=tmp_path / server.ENGINE_TRACE)  # fmt: skip
    sess = server.Session("engine", backend, tmp_path / "receipts.jsonl", max_calls=25)
    clean = call(sess, question="plain who?")
    degraded = call(sess, question="!slow who?")
    sess.close()
    asyncio.run(backend.engine.aclose())
    assert clean["error"] is None and backend.retry["attempts"] == 2
    assert degraded["answer"] is None and "(in all 2 attempts)" in degraded["error"]
    rows = server.verify(tmp_path / "receipts.jsonl")
    assert rows[2]["infra_scope"] == "trial" and [
        a["attempt"] for a in rows[2]["research"]["attempts"]
    ] == [1, 2]
    chat = rows[1]["research"]["usage"]["chat"]
    assert {"reasoning_tokens", "cached_tokens", "retries", "cancelled"} <= set(chat)
    assert rows[1]["research"]["identity"]["budget_feedback"] is True


def test_chat_probe_per_preset(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    real, seen = httpx.AsyncClient, []

    def client_with(handler):
        return lambda *a, **kw: real(transport=httpx.MockTransport(handler))

    async def ok(request):
        seen.append(json.loads(request.content)["model"])
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

    async def slow(request):
        await asyncio.sleep(0.3)
        return httpx.Response(200, json={})

    monkeypatch.setattr(httpx, "AsyncClient", client_with(ok))
    assert (
        container.chat_unhealthy("deepseek-flash-b8") is None
        and seen == ["deepseek/deepseek-v4-flash-0731"] * 2
    )
    monkeypatch.setattr(
        httpx, "AsyncClient", client_with(lambda r: httpx.Response(502, text="bad gateway"))
    )
    assert "call 1/2 failed" in container.chat_unhealthy("openrouter-b8")
    monkeypatch.setattr(httpx, "AsyncClient", client_with(slow))
    why = container.chat_unhealthy("openrouter-b8", limit_s=0.2)
    assert "failed after" in why and "TimeoutError" in why and KEY not in why
    assert container.chat_unhealthy("ollama") is None  # local: the preflight checks its endpoint


def test_the_gate_probes_jev_and_the_chat_model_of_every_preset_to_run(engine_bench, monkeypatch):
    from harnext_eval.e2.v2.agents.fake import FakeAgent

    class Live(FakeAgent):
        needs_container = True

    run = runner.Runner(engine_bench("gate-presets", reader="claude"), Live())
    run.by_id["H-engine"] = {**run.by_id["H-engine"], "engine": {"preset": "deepseek-flash-b8"}}
    probed = []
    monkeypatch.setattr(container, "jev_unhealthy", lambda: probed.append("jev"))
    monkeypatch.setattr(container, "chat_unhealthy", lambda preset: probed.append(preset))
    assert run.provider_gate() is None and probed == ["jev", "deepseek-flash-b8"]
    monkeypatch.setattr(
        container, "chat_unhealthy", lambda preset: f"chat model unhealthy: {preset}"
    )
    assert run.provider_gate() == "chat model unhealthy: deepseek-flash-b8"


def test_raising_a_cap_extends_a_run_that_finished_its_first_stage(engine_bench):
    from harnext_eval.e2.v2.agents.fake import FakeAgent

    stage1 = engine_bench("staged", max_historical=2)
    assert runner.Runner(stage1, FakeAgent()).live() == 0
    assert summary(stage1)["status"] == "complete" and summary(stage1)["recorded"] == 4
    stage2 = stage1.model_copy(update={"max_historical": 3})  # same run, same arms, a higher cap
    assert runner.Runner(stage2, FakeAgent()).live() == 0  # the finished-run guard lets it through
    assert summary(stage2)["status"] == "complete" and summary(stage2)["recorded"] == 6
