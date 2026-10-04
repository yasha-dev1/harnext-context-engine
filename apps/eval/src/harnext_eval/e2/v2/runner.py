"""E2 v2 runner: every task under every arm, atomic per-trial archive, pause-and-resume.

    python -m harnext_eval.e2.v2.runner --config <yaml> --live | --check-resume | --dry-run

A trial is recorded only when agent and grader both finished on working infrastructure.
Quota, credit, auth, rate-limit, docker or Gradle-environment trouble raises PauseRun: the
unit in flight leaves no trace, summary.json says `paused`, the process exits 75 (EX_TEMPFAIL)
and the same command resumes. Trials with a result.json are never rerun.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from harnext_eval.e2 import kafka_coding as kc
from harnext_eval.e2.benchmark import ROOT, canonical, read, sha
from harnext_eval.e2.v2 import checkouts as store_checkouts
from harnext_eval.e2.v2 import grade
from harnext_eval.e2.v2 import workspace as workspaces
from harnext_eval.e2.v2.agents.base import McpSpec, TrialSpec, events_of
from harnext_eval.e2.v2.errors import PauseRun, reset_delay
from harnext_eval.e2.v2.mcp_server import ENGINE_PRESETS, KINDS, engine_usage

EXIT_PAUSED = 75
SOURCES_DB = (
    "sources.sqlite"  # sources/index.py DB; not imported here: keeps sqlite out of the runner
)
GZIP_OVER = 64 * 1024
ARCHIVED = (
    "prompt.txt", "launch.json", "trace.jsonl", "agent.stderr", "answer.json", "patch.diff",
    "gradle.log.gz", "mcp/receipts.jsonl",
    "mcp/engine-trace.jsonl",  # kind engine (2026-09-28): the full engine trace of every call
)  # fmt: skip
ACCESS = {
    "none": "No project-history context is provided for this task. Work from the task "
    "statement{workspace} and your own knowledge; there is no network access.",
    "files": "A read-only directory /context holds the project's recorded history up to the "
    "cutoff. Use your normal file tools (ls, grep/rg, cat) to consult it as you see fit; "
    "nothing later than the cutoff is in it and there is no network access.",
    "mcp": "The MCP server `{server}` holds the project's recorded history up to the cutoff. "
    "Use its tools ({tools}) to consult it as you see fit; nothing later than the cutoff is "
    "in it and there is no network access.",
    # kind research (every F arm): Harnext's product read interface, one tool.
    "mcp:research": "The MCP server `{server}` holds the project's recorded history up to the "
    "cutoff. Its tool `context_research(question)` asks Harnext's research agent, which reads "
    "the stored context and returns a cited answer; ask as many focused questions as you need. "
    "Nothing later than the cutoff is in it and there is no network access.",
    "sources": "No pre-built project memory is provided. You can query time-frozen offline "
    "replicas of the project's systems of record as of the cutoff with the usual command-line "
    "tools: `gh` (pull requests, reviews, commits), `jira` (issues, comments, change history), "
    "`mail` (mailing lists). Run `<tool> --help` for the supported commands. The live internet "
    "is not reachable.",
}
GIT_HISTORY = (  # every coding prompt, every arm, identical wording
    "The workspace is a git repository whose history ends at the base commit; `git log`, "
    "`git blame` and `git show` work offline."
)
CONTRACT = {
    "historical": 'Reply with one JSON object {"value": ..., "evidence": [...]}: `value` is '
    "the answer in the requested type (null if it cannot be determined), `evidence` lists the "
    "sources you relied on.",
    "coding": f"The repository is checked out at /workspace. {GIT_HISTORY} Make the change "
    "by editing source files there; the tests named above, restored to their original content, decide whether the issue is resolved. "
    "Gradle works offline only (`./gradlew :module:test --tests X --offline`). "
    'Finish with one JSON object {"summary": "..."}.',
}
# Standing instructions per access type (the CLAUDE.md / AGENTS.md of the trial). Decided
# 2026-09-21: an agent that has a context source must consult it BEFORE it starts, otherwise
# half the trials never touch the store and the arms cannot differ. The same rule is given to
# the baseline agent for its own sources, so the instruction itself is not a confound; the
# closed-book arm has nothing to consult and gets no file.
GATHER_FIRST = {
    "research": (
        "# Project context: read this first\n\n"
        "This environment is connected to the Harnext context engine through the MCP server "
        "`{server}`. It holds this project's recorded history (issues, discussions, code reviews, "
        "pull requests, commits, mailing list) up to the task's cutoff.\n\n"
        "DO NOT START UNTIL YOU GATHER CONTEXT FROM context_research mcp tool.\n\n"
        "Before you read code, edit files or answer, call `context_research` with focused "
        "questions: what the issue is about, which components and earlier changes are involved, "
        "what was discussed and decided, and which conventions the project follows in this area. "
        "Ask follow-up questions until you understand the background, then do the task. The "
        "context engine knows nothing later than the cutoff, so it cannot tell you the solution; "
        "ask it about the project, not for the answer.\n"
    ),
    "rag": (
        "# Project context: read this first\n\n"
        "This environment is connected to a retrieval service over the project's recorded history "
        "(issues, discussions, code reviews, pull requests, commits, mailing list) up to the "
        "task's cutoff, through the MCP server `{server}`.\n\n"
        "DO NOT START UNTIL YOU GATHER CONTEXT FROM the `search` mcp tool.\n\n"
        "Before you read code, edit files or answer, call `search` (and `read_event` for a full "
        "record) with focused queries: what the issue is about, which components and earlier "
        "changes are involved, what was discussed and decided, and which conventions the project "
        "follows in this area. Search again until you understand the background, then do the "
        "task. Nothing later than the cutoff is indexed, so the solution is not in it.\n"
    ),
    "sources": (
        "# Project context: read this first\n\n"
        "This environment has offline replicas of the project's systems of record as of the "
        "task's cutoff: `gh` (pull requests, reviews, commits), `jira` (issues, comments, change "
        "history) and `mail` (mailing lists).\n\n"
        "DO NOT START UNTIL YOU GATHER CONTEXT FROM the `gh`, `jira` and `mail` tools.\n\n"
        "Before you read code, edit files or answer, query them: what the issue is about, which "
        "components and earlier changes are involved, what was discussed and decided, and which "
        "conventions the project follows in this area. Query again until you understand the "
        "background, then do the task. Nothing later than the cutoff is in them, so the solution "
        "is not there.\n"
    ),
}


# Kinds added after the first run began (2026-09-27). Kept out of ACCESS / GATHER_FIRST, whose
# hash the running run recorded; every arm of these kinds pins its own texts instead (arm_pin).
ACCESS_EXTRA = {
    "mcp:research_search": "The MCP server `{server}` holds the project's recorded history up to "
    "the cutoff. It has two tools: `context_research(question)` asks Harnext's research agent, "
    "which reads the stored context and returns a cited report; `search_context(query, k)` "
    "returns the best-matching stored passages with their file paths, for you to read yourself. "
    "Nothing later than the cutoff is in it and there is no network access.",
}
GATHER_FIRST_EXTRA = {
    "research_search": (
        "# Project context: read this first\n\n"
        "This environment is connected to the Harnext context engine through the MCP server "
        "`{server}`. It holds this project's recorded history (issues, discussions, code reviews, "
        "pull requests, commits, mailing list) up to the task's cutoff.\n\n"
        "DO NOT START UNTIL YOU GATHER CONTEXT FROM the Harnext MCP tools.\n\n"
        "Use context_research for full reports about the codebase and use search_context for "
        "lightweight context retrievals.\n\n"
        "`context_research(question)` sends a research agent through the whole context store and "
        "returns a synthesized, cited report: use it when you need the full background of an "
        "area (what an issue is about, which earlier changes and discussions are involved, what "
        "was decided, which conventions apply). `search_context(query)` is a fast keyword "
        "lookup that returns matching passages with their file paths: use it for specific facts "
        "(an issue key, a PR number, a class or config name, who changed or reviewed what) and "
        "to check details. Before you read code, edit files or answer, gather the background "
        "this way, then do the task. The context engine knows nothing later than the cutoff, so "
        "it cannot tell you the solution; ask it about the project, not for the answer.\n"
    ),
}
# Kind engine (2026-09-28, E2-ENGINE-DESIGN.md stage 3): the same tool as kind research, so the
# access paragraph and the standing instructions are the research texts, verbatim (the instruction
# is not a confound); pinned per arm like every kind added after the first run began.
ACCESS_EXTRA["mcp:engine"] = ACCESS["mcp:research"]
GATHER_FIRST_EXTRA["engine"] = GATHER_FIRST["research"]


def arm_pin(arm, server):
    """What pins an arm on first use: its definition and the server block, plus, for kinds added
    later, the exact texts its trials receive (older arms keep their original pin)."""
    extra = []
    key = f"{arm['access']}:{arm.get('mcp_kind')}"
    if key in ACCESS_EXTRA:
        extra = [ACCESS_EXTRA[key], GATHER_FIRST_EXTRA.get(arm.get("mcp_kind"))]
    return sha(canonical([arm, server, *extra]))


def standing_instructions(arm, server):
    key = arm.get("mcp_kind") if arm["access"] == "mcp" else arm["access"]
    text = GATHER_FIRST.get(key) or GATHER_FIRST_EXTRA.get(key)
    return text.format(server=server["name"]) if text else None


CODING_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}


class Timeouts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    historical: int = 600
    coding: int = 3600
    grading: int = 3000


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol: Literal["harnext-e2-v2"]
    run_id: str = Field(pattern=r"^[\w.-]+$")
    benchmark: Path
    arms_file: Path | None = None  # default: <benchmark>/arms.yaml
    arms: list[str] | None = None  # optional subset, in arms.yaml order
    agent: Literal["codex", "claude-code", "fake"]
    model: str
    effort: str = "medium"
    reader: Literal["claude", "fake"] = "claude"  # research kind: the store's read agent; engine: fake models
    # Kind engine: chat endpoint for every engine arm of this run (else the arm's `engine.preset`).
    # Applied to the arm definition, so the arm pin records it; not in the manifest's config.
    engine_preset: Literal["openrouter", "ollama"] | None = None
    stores: Path = ROOT / "out/kafka-500-v2/stores"
    checkouts: Path = store_checkouts.DEFAULT_ROOT  # on-demand read-only checkpoint extracts
    keep_per_arm: int = Field(default=3, ge=1)  # checkouts kept per git arm (LRU beyond)
    results: Path | None = None  # default: <repo>/.harnext/evaluations/kafka-500-v2/<run_id>
    scratch: Path | None = None  # default: apps/eval/out/kafka-500-v2/runs/<run_id>/scratch
    workspaces: Path = workspaces.WORKSPACES  # per-task git repositories (pruned history)
    keep_task_repos: bool = False  # ~150-250 MB each: dropped once every arm of the task is in
    workers: int = Field(default=2, ge=1, le=8)
    coding_workers: int = Field(default=2, ge=1, le=8)  # RAM: measured ~3.5 GB agent, then ~5 GB grading
    timeouts: Timeouts = Timeouts()
    commit_every: int = Field(default=0, ge=0)  # trials between git checkpoint commits; 0 = off
    min_free_gib: int = Field(default=15, ge=0)
    store_hash: Literal["manifest", "tree"] = "manifest"
    limit_tasks: int | None = None  # smoke runs: first n of run_order
    # Shrinking (or growing) a run in the middle: the first n tasks of each kind in the frozen
    # run order. Neither is pinned in the manifest, so they may change between resumes; trials
    # already recorded stay recorded and count in every report.
    max_coding: int | None = None
    max_historical: int | None = None
    # Agent turn cap per trial (None = unlimited). Not pinned: it was introduced after the first
    # chunk of the first run; every result.json records the cap it ran under.
    max_turns: int | None = None
    # Arms that run on a fixed subset only (e.g. an S-tools drift control): {arm: n} keeps every
    # n-th task of each category (coding, design decisions, implicit links, factual history) in run
    # order, positions 0, n, 2n, ... Not outcome-dependent; not pinned (like max_*).
    every_nth: dict[str, int] = {}
    fake: dict = {}  # FakeAgent arguments


def load(path):
    config = Config.model_validate(yaml.safe_load(path.read_text()))
    base = path.resolve().parent
    for name in ("benchmark", "arms_file", "stores", "checkouts", "results", "scratch", "workspaces"):  # fmt: skip
        value = getattr(config, name)
        if value is not None and not value.is_absolute():
            setattr(config, name, (base / value).resolve())
    config.arms_file = config.arms_file or config.benchmark / "arms.yaml"
    config.results = config.results or (
        ROOT.parents[1] / ".harnext/evaluations/kafka-500-v2" / config.run_id
    )
    config.scratch = config.scratch or ROOT / "out/kafka-500-v2/runs" / config.run_id / "scratch"
    return config


def make_agent(config):
    if config.agent == "fake":
        from harnext_eval.e2.v2.agents.fake import FakeAgent

        return FakeAgent(**config.fake)
    if config.agent == "codex":
        from harnext_eval.e2.v2.agents.codex import CodexAgent

        return CodexAgent()
    from harnext_eval.e2.v2.agents.claude_code import ClaudeCodeAgent

    return ClaudeCodeAgent()


def fsync_path(path):
    """Flush a file or a directory entry to disk: a power cut after this cannot tear it."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    fsync_path(temporary)
    temporary.replace(path)
    fsync_path(path.parent)


def intact(trial):
    """The trial's result record if every archived file matches archive-manifest.json, else None."""
    try:
        files = read(trial / "archive-manifest.json")["files_sha256"]
        for name, digest in files.items():
            if sha((trial / name).read_bytes()) != digest:
                return None
        return read(trial / "result.json")
    except (OSError, ValueError, KeyError, TypeError):
        return None


def quarantine(results, trial):
    target = results / "quarantine" / f"{trial.parent.name}-{trial.name}-{time.time_ns()}"
    target.parent.mkdir(parents=True, exist_ok=True)
    trial.replace(target)
    print(f"quarantined a damaged trial archive: {trial} -> {target}", file=sys.stderr, flush=True)


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def manifest_hash(manifest, name):
    """The recorded sha256 of a benchmark file, under any of the manifest's spellings."""
    stem = re.sub(r"\W", "_", name.rsplit(".", 1)[0])
    for table in ("files", "sha256", "hashes"):
        entry = (
            (manifest.get(table) or {}).get(name) if isinstance(manifest.get(table), dict) else None
        )
        if entry:
            return entry["sha256"] if isinstance(entry, dict) else entry
    short = name.split(".")[1] if name.count(".") > 1 else stem  # tasks.public.jsonl -> public
    for key in (f"{short}_sha256", f"{stem}_sha256", f"{name}_sha256"):
        if key in manifest:
            return manifest[key]
    raise ValueError(f"benchmark manifest records no sha256 for {name}")


def load_benchmark(directory):
    """(public tasks by id, private by id, run_order); refuses files that differ from the freeze."""
    manifest = read(directory / "manifest.json")
    for name in ("tasks.public.jsonl", "tasks.private.jsonl"):
        if sha((directory / name).read_bytes()) != manifest_hash(manifest, name):
            raise ValueError(f"{name} differs from the frozen manifest")
    public = {t["id"]: t for t in jsonl(directory / "tasks.public.jsonl")}
    private = {t["id"]: t for t in jsonl(directory / "tasks.private.jsonl")}
    order = manifest["run_order"]
    if set(order) != set(public) or set(public) != set(private) or len(order) != len(public):
        raise ValueError("run_order, public and private tasks do not describe the same set")
    return public, private, order


def load_arms(path, subset=None):
    document = yaml.safe_load(path.read_text())
    arms = document["arms"]
    for arm in arms:
        if arm["access"] not in ACCESS or (arm["access"] != "none") != bool(arm.get("store")):
            raise ValueError(f"arm {arm['id']}: bad access/store")
        if (arm["access"] == "mcp") != bool(arm.get("mcp_kind")):
            raise ValueError(f"arm {arm['id']}: mcp_kind goes with access mcp, only")
        if arm.get("mcp_kind") and arm["mcp_kind"] not in KINDS:
            raise ValueError(f"arm {arm['id']}: unknown mcp_kind {arm['mcp_kind']!r}")
        if (arm.get("memory_store") or arm.get("engine")) and arm.get("mcp_kind") != "engine":
            raise ValueError(f"arm {arm['id']}: memory_store / engine go with mcp_kind engine, only")
        if (arm.get("engine") or {}).get("preset", "openrouter") not in ENGINE_PRESETS:
            raise ValueError(f"arm {arm['id']}: engine preset must be one of {ENGINE_PRESETS}")
    if subset is not None:
        if unknown := set(subset) - {a["id"] for a in arms}:
            raise ValueError(f"unknown arms: {sorted(unknown)}")
        arms = [a for a in arms if a["id"] in subset]
    return arms, document.get("mcp_server") or {"name": "history"}


def with_preset(arm, preset):
    """An engine arm under the run's chat endpoint (config `engine_preset`), else unchanged."""
    if not preset or arm.get("mcp_kind") != "engine":
        return arm
    return {**arm, "engine": {**(arm.get("engine") or {}), "preset": preset}}


def engine_preset(arm):
    return (arm.get("engine") or {}).get("preset", "openrouter")


def plan(order, arms):
    """[(task index, task id, [arm ids in run order])]: control arms on every fifth task,
    arm order rotated by task index so no arm always runs first (or on a warm cache)."""
    rows = []
    for index, task_id in enumerate(order):
        applicable = [a["id"] for a in arms if a.get("subset") != "control" or index % 5 == 0]
        shift = index % len(applicable)
        rows.append((index, task_id, applicable[shift:] + applicable[:shift]))
    return rows


def category(task):
    family = task.get("family")
    if task["kind"] == "coding":
        return "coding"
    return family if family in ("design_decision", "implicit_link") else "factual"


def thin(rows, public, every_nth):
    """plan rows with the `every_nth` arms kept on every n-th task of each category only."""
    seen, out = defaultdict(int), []
    for index, task_id, arms in rows:
        position = seen[category(public[task_id])]
        seen[category(public[task_id])] += 1
        out.append((index, task_id, [a for a in arms if position % every_nth.get(a, 1) == 0]))
    return out


def mcp_trouble(trace, receipts, server, status):
    """Why an engine trial's tool traffic cannot be trusted, or None: the MCP server was not
    connected at start, a `context_research` call timed out or lost its server on the agent side,
    or the agent's trace and the receipts disagree on the number of calls (the server died)."""
    name, uses, errors, states = f"mcp__{server}__context_research", set(), [], []
    for event in events_of(trace):
        if event.get("type") == "system" and event.get("subtype") == "init":
            states += [m.get("status") for m in event.get("mcp_servers") or [] if m.get("name") == server]
        item = event.get("item") or {}
        if event.get("type") == "item.completed" and item.get("type") == "mcp_tool_call" and item.get("tool") == "context_research":  # fmt: skip
            uses.add(item.get("id"))
            if item.get("status") == "failed" or item.get("error"):
                errors.append(json.dumps(item.get("error")))
        content = (event.get("message") or {}).get("content")
        for block in content if isinstance(content, list) else []:
            if block.get("type") == "tool_use" and block.get("name") == name:
                uses.add(block.get("id"))
            if block.get("type") == "tool_result" and block.get("tool_use_id") in uses and block.get("is_error"):
                errors.append(json.dumps(block.get("content")))
    if bad := [s for s in states if s != "connected"]:
        return f"MCP server {server} not connected at start ({bad[0]})"
    for text in errors:
        if re.search(r"timed? ?out|timeout|-32001", text, re.I):
            return f"context_research timed out on the agent side: {text[:200]}"
        if re.search(r"connection closed|not connected|-32000|server.*(exit|crash)", text, re.I):
            return f"the MCP server went away during context_research: {text[:200]}"
    calls = sum(r.get("tool") == "context_research" for r in jsonl(receipts)) if receipts.exists() else 0
    if len(uses) != calls and not (status in ("timeout", "turn_limit") and len(uses) == calls + 1):
        return f"{len(uses)} context_research calls in the agent trace, {calls} in the receipts"
    return None


def build_prompt(task, arm, server):
    key = f"{arm['access']}:{arm.get('mcp_kind')}"
    template = ACCESS.get(key) or ACCESS_EXTRA.get(key) or ACCESS[arm["access"]]
    access = template.format(
        workspace=", the repository" if task["kind"] == "coding" else "",
        server=server["name"],
        tools=", ".join(KINDS.get(arm.get("mcp_kind"), ())),
    )
    return "\n\n".join([task["prompt"].strip(), access, CONTRACT[task["kind"]]]) + "\n"


def server_started(receipts):
    """True once the trial's MCP server wrote its `start` receipt."""
    return receipts.exists() and any(row.get("kind") == "start" for row in jsonl(receipts))


class TrialPause(PauseRun):  # noqa: N818
    """Kind engine: trouble of this trial only (provider trouble that outlived the server's redos,
    an agent-side tool timeout, a dead MCP server mid-call): the trial is left unrecorded and
    redone later in the same session; the rest of the run goes on (after the health gate)."""


TRIAL_PAUSES_BEFORE_DEFER = 3  # a trial paused this often in one session waits until the end


def reader_infra_scope(receipts):
    """Kind engine: "trial" when every infra row of the receipts is scoped to this trial
    (mcp_server.ProviderTrouble), else "run". Rows without a scope (older servers, other kinds) are "run"."""
    scopes = {row.get("infra_scope") for row in jsonl(receipts) if row.get("infra")} if receipts.exists() else set()
    return "trial" if scopes == {"trial"} else "run"


def reader_infra(receipts):
    """The provider text of the first `context_research` call the reader's provider refused
    (mcp_server writes `"infra": true`), or None. Checked after every MCP trial: an agent that
    was told "usage limit" by its tool is not a model result."""
    if not receipts.exists():
        return None
    for row in jsonl(receipts):
        if row.get("infra"):
            return row.get("error") or "reader refused"
    return None


def summarize(results, arms, status, reason=None, not_recorded=()):
    """arm x kind x family cells + paired N-vs-arm counts. Pure function of recorded results."""

    def cell(rows):
        return {
            "n": len(rows),
            "success": sum(r["success"] for r in rows),
            "status": {
                s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})
            },
            "input_tokens": sum(r["usage"].get("input_tokens", 0) for r in rows),
            "cached_input_tokens": sum(r["usage"].get("cached_input_tokens", 0) for r in rows),
            "output_tokens": sum(r["usage"].get("output_tokens", 0) for r in rows),
            "agent_seconds": round(sum(r["elapsed_s"] for r in rows), 1),
            "tool_calls": sum(sum(r["tool_calls"].values()) for r in rows),
            "context_bytes": sum(r["context_output_bytes"] for r in rows),
        }

    table = {}
    for arm in arms:
        rows = [r for r in results.values() if r["arm"] == arm]
        by_kind = defaultdict(lambda: defaultdict(list))
        for row in rows:
            by_kind[row["kind"]][row["family"]].append(row)
        table[arm] = cell(rows) | {
            "by_kind": {
                kind: cell([r for f in families.values() for r in f])
                | {"by_family": {f: cell(v) for f, v in sorted(families.items())}}
                for kind, families in sorted(by_kind.items())
            }
        }
    paired = {}
    for arm in arms:
        if arm == "N":
            continue
        pairs = [
            (results[(t, "N")]["success"], r["success"])
            for (t, a), r in results.items()
            if a == arm and (t, "N") in results
        ]
        paired[arm] = {
            "pairs": len(pairs),
            "helped": sum(not n and a for n, a in pairs),
            "harmed": sum(n and not a for n, a in pairs),
            "both": sum(n and a for n, a in pairs),
            "neither": sum(not n and not a for n, a in pairs),
        }
    return {
        "status": status,
        "reason": reason,
        "not_recorded": sorted(not_recorded),
        "recorded": len(results),
        "arms": table,
        "paired_vs_N": paired,
    }


def commit_checkpoint(results, message):
    """v1 rule: only the results directory is committed; unrelated staged work is never swept in."""

    def git(*args, **kwargs):
        return subprocess.run(["git", *args], cwd=results, text=True, **kwargs)

    staged = git("diff", "--cached", "--name-only", capture_output=True, check=True).stdout
    if staged.strip():
        raise ValueError("checkpoint refused: unrelated staged changes need committing separately")
    git("add", "--", str(results), check=True)
    if git("diff", "--cached", "--quiet").returncode:
        git("commit", "-q", "-m", message, check=True)


class Runner:
    def __init__(self, config, agent=None):
        self.config = config
        self.agent = agent or make_agent(config)
        self.public, self.private, order = load_benchmark(config.benchmark)
        order = order[: config.limit_tasks] if config.limit_tasks else order
        caps = {"coding": config.max_coding, "historical": config.max_historical}
        seen: dict[str, int] = {}
        self.order = []
        for task_id in order:
            kind = self.public[task_id]["kind"]
            seen[kind] = seen.get(kind, 0) + 1
            if caps.get(kind) is None or seen[kind] <= caps[kind]:
                self.order.append(task_id)
        arms, self.server = load_arms(config.arms_file, config.arms)
        self.arms = [with_preset(a, config.engine_preset) for a in arms]
        self.by_id = {a["id"]: a for a in self.arms}
        self.plan = plan(self.order, self.arms)
        if config.every_nth:
            self.plan = thin(self.plan, self.public, config.every_nth)
        self.results = {}
        for _, task_id, arms in self.plan:
            for arm in arms:
                saved = self.trial_dir(task_id, arm) / "result.json"
                if saved.exists():
                    record = intact(saved.parent)
                    if record is None:  # torn by a power cut: set aside, the trial reruns
                        quarantine(config.results, saved.parent)
                    else:
                        self.results[(task_id, arm)] = record
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.interrupted = False
        self.draining = False
        self.task_arms = {}
        self.reason = None
        self.in_flight = set()
        self.coding_slots = threading.Semaphore(config.coding_workers)
        self.since_commit = 0
        self.checkouts = store_checkouts.Checkouts(config.checkouts, config.keep_per_arm)
        self.pins = {}  # "<store>/<checkpoint>" -> identity, from preflight
        self.image_id = None  # the agent image's ID, resolved in live preflight (containers only)
        self.allow_image_change = False  # --allow-image-change: recorded in the engine pin
        # Kind engine, trial-scoped pauses (TrialPause): pauses per trial this session, the trials to
        # redo, those set aside until the end, and the health gate owed before the next trial starts.
        self.trial_pauses, self.pause_log = defaultdict(int), []
        self.requeue, self.deferred = [], []
        self.gate_needed, self.gate_lock, self.session_path = False, threading.Lock(), None
        self.extend_run = False  # --extend-run: add arms to a completed run

    # ---- identity -------------------------------------------------------------------------
    def trial_dir(self, task_id, arm):
        return self.config.results / "trials" / task_id / arm

    def remaining(self):
        return [(i, t, a) for i, t, arms in self.plan for a in arms if (t, a) not in self.results]

    def containerised(self):
        return getattr(self.agent, "needs_container", True)

    def store_hashes(self, units=None):
        """Identity per (store, checkpoint) the remaining trials need: git arms pin the tag's
        tree hash (no export needs to exist), the others their export manifest. Fails once,
        listing every missing tag / export."""
        needed = {
            (self.by_id[a][field], self.public[t]["checkpoint"])
            for _, t, a in (self.remaining() if units is None else units)
            for field in ("store", "memory_store")  # memory_store: kind engine's notes (a file store)
            if self.by_id[a].get(field)
        }
        hashes, missing = {}, []
        for store, ckpt in sorted(needed):
            try:
                hashes[f"{store}/{ckpt}"] = store_checkouts.identity(
                    self.config.stores / store, ckpt, self.config.store_hash
                )
            except LookupError as exc:
                missing.append(str(exc))
        if missing:
            raise ValueError("stores lack checkpoints the run needs:\n  " + "\n  ".join(missing))
        return hashes

    def manifest(self):
        config = self.config.model_dump(mode="json", exclude={
            "workers", "coding_workers", "fake", "commit_every", "min_free_gib", "limit_tasks",
            "max_coding", "max_historical", "max_turns",
            "engine_preset",  # pinned through the engine arms' definitions (with_preset)
            "every_nth",
            "arms",  # arms join a run as their stores are frozen; each store is pinned on first use
            "benchmark", "arms_file", "stores", "checkouts", "results", "scratch",  # pinned by hash
            "workspaces", "keep_task_repos",
        })  # fmt: skip
        directory = self.config.benchmark
        return {
            "run_id": self.config.run_id,
            "config": config,
            "agent": self.agent.name,
            "benchmark": directory.name,
            "prompt_template_sha256": sha(canonical([ACCESS, CONTRACT, GATHER_FIRST])),
        }

    def preflight(self, *, live):
        """Everything that can be checked without a model call. Returns the run manifest."""
        manifest = self.manifest()
        path = self.config.results / "manifest.json"
        previous = read(path) if path.exists() else None
        self.refuse_extension(previous)
        if previous:
            for key, value in manifest.items():
                if key not in {"tasks", "arms", "stores", "engine"} and previous.get(key) != value:
                    raise ValueError(f"cannot resume: {key} differs from the recorded run manifest")
        # Tasks are pinned one by one, on first use: the coding half grows (append-only) while
        # admission continues, and a task a run has touched may never change afterwards.
        tasks = {t: sha(canonical([self.public[t], self.private[t]])) for t in self.order}
        known = (previous or {}).get("tasks", {})
        if moved := sorted(t for t, v in tasks.items() if known.get(t, v) != v):
            raise ValueError(f"cannot resume: task records changed since the run began: {moved}")
        manifest["tasks"] = known | tasks
        # Arms too are pinned one by one on first use (their definition and the server block),
        # so the arms file may gain or lose OTHER arms while a run is in progress.
        arms = {a["id"]: arm_pin(a, self.server) for a in self.arms}
        pinned = (previous or {}).get("arms", {})
        if moved := sorted(a for a, v in arms.items() if pinned.get(a, v) != v):
            raise ValueError(f"cannot resume: arm definitions changed since the run began: {moved}")
        manifest["arms"] = pinned | arms
        stores = self.store_hashes()
        recorded = (previous or {}).get("stores", {})
        if changed := sorted(k for k, v in stores.items() if recorded.get(k, v) != v):
            raise ValueError(f"cannot resume: stores changed since the run began: {changed}")
        manifest["stores"] = self.pins = recorded | stores
        if any(self.public[t]["kind"] == "coding" for _, t, _ in self.remaining()):
            if self.containerised() and not kc.REPO.exists():
                raise ValueError(f"Kafka bare clone missing: {kc.REPO}")
        version = (previous or {}).get("agent_version")
        if live or previous:
            if self.containerised():
                from harnext_eval.e2.v2.agents import container

                if not container.image_present() or not container.image_present(
                    container.PROXY_IMAGE
                ):
                    raise ValueError("docker images harnext-e2-agent / harnext-e2-proxy not built")
                if live:
                    ok, detail = container.egress_ok()
                    if not ok:
                        raise ValueError("egress check failed:\n" + detail)
                    if self.config.reader != "fake" and any(engine_preset(self.by_id[a]) == "ollama" for _, _, a in self.remaining() if self.by_id[a].get("mcp_kind") == "engine"):  # fmt: skip
                        if why := container.local_model_unreachable():
                            raise ValueError(why)
            current = self.agent.version()
            if version and current != version:
                raise ValueError(f"agent CLI version {current!r} differs from recorded {version!r}")
            version = current
        manifest["agent_version"] = version
        if live and self.containerised():
            from harnext_eval.e2.v2.agents import container

            self.image_id = container.pin_image()
        if (live or previous) and any(a.get("mcp_kind") == "engine" for a in self.arms):
            manifest["engine"] = self.engine_pin((previous or {}).get("engine"))
        return manifest

    def refuse_extension(self, previous):
        """A finished run (summary `complete`, every pinned arm has recorded trials) is not extended
        by new arms unless --extend-run: a config rewritten by accident (28 Sep: a watcher listed
        every arm of arms.yaml) must not append trials to a finished run. Unfinished runs: unchanged."""
        summary = self.config.results / "summary.json"
        if not previous or self.extend_run or not summary.exists() or read(summary).get("status") != "complete":
            return
        pinned = set(previous.get("arms") or {})
        if not pinned <= {arm for _, arm in self.results}:
            return
        if new := sorted({a["id"] for a in self.arms} - pinned):
            raise ValueError(f"run {self.config.run_id} is complete; the config adds arms it never had: "
                             f"{new}. Use a new run id, or pass --extend-run to append them to this run")  # fmt: skip

    def provider_gate(self):
        """Engine arms with live models: probe Jev, and the chat model of every preset still to run,
        before starting trials (every start, every automatic resume, after a trial pause); a stalling
        provider pauses the run here instead of in a paid trial."""
        presets = sorted({engine_preset(self.by_id[a]) for _, _, a in self.remaining() if self.by_id[a].get("mcp_kind") == "engine"})  # fmt: skip
        if not presets or self.config.reader == "fake" or not self.containerised():
            return None
        from harnext_eval.e2.v2.agents import container

        why = container.jev_unhealthy()
        for preset in presets:
            why = why or container.chat_unhealthy(preset)
        return why

    def engine_identity(self):
        """(image ID, code + config hashes the engine arms would run with, host tree hashes)."""
        from harnext_eval.e2.v2 import engine_pin

        presets = sorted({engine_preset(a) for a in self.arms if a.get("mcp_kind") == "engine"})
        host = engine_pin.identity(presets)
        if not self.containerised():
            return "host (no container)", host, host
        from harnext_eval.e2.v2.agents import container

        image = self.image_id or container.image_id()
        return image, container.image_identity(image, presets), host

    def engine_pin(self, pinned):
        """Pin, on first use, the image ID, the engine / raw store / server code and the engine
        config of the engine arms, measured INSIDE the image that runs the trials; refuse a resume
        whose code or config differs, and a changed image unless --allow-image-change (recorded)."""
        from harnext_eval.e2.v2.agents import container

        image, inside, host = self.engine_identity()
        code = {k: v for k, v in inside.items() if k != "config_sha256"}
        if pinned is None:
            if {k: host[k] for k in code} != code:
                raise ValueError("the agent image's engine code differs from the working tree: rebuild "
                                 f"the image first\n  image {code}\n  tree  {({k: host[k] for k in code})}")  # fmt: skip
            return {"image_id": image, "code": code, "config_sha256": inside["config_sha256"],
                    "proxy_allowlist": container.proxy_allowlist() if self.containerised() else None,
                    "pinned_at_unix": time.time(), "image_overrides": []}  # fmt: skip
        moved_code = sorted(k for k in code if code[k] != pinned["code"].get(k))
        if moved_code and (moved_code != ["mcp_server_sha256"] or not self.allow_image_change):
            raise ValueError(f"cannot resume: engine code in the image differs from the pin ({moved_code}): {code} != {pinned['code']}"
                             + ("; only a change of mcp_server.py alone can be accepted, with --allow-image-change" if moved_code == ["mcp_server_sha256"] else ""))  # fmt: skip
        if moved := sorted(p for p, h in inside["config_sha256"].items() if pinned["config_sha256"].get(p, h) != h):  # fmt: skip
            raise ValueError(f"cannot resume: engine config differs from the pin for presets {moved}")
        pin = {**pinned, "config_sha256": pinned["config_sha256"] | inside["config_sha256"]}
        if image != pinned["image_id"]:
            if not self.allow_image_change:
                raise ValueError(f"cannot resume: agent image {image} differs from the pinned "
                                 f"{pinned['image_id']} (same engine code); pass --allow-image-change to accept")  # fmt: skip
            change = {"from": pinned["image_id"], "to": image, "at_unix": time.time(), "argv": sys.argv}
            if moved_code:  # the MCP server wrapper changed; engine and raw store code did not
                change["code_from"], change["code_to"] = pinned["code"], code
                pin["code"] = code
            pin["image_overrides"] = [*pinned.get("image_overrides", []), change]
            pin["image_id"] = image
        return pin

    def identity_report(self):
        """Everything that pins this config's engine arms, for quoting: image ID, code and config
        hashes (inside the image and in the tree), store manifest hashes, the recorded pin."""
        from harnext_eval.e2.v2.agents import container

        image, inside, host = self.engine_identity()
        path = self.config.results / "manifest.json"
        units = [(i, t, a) for i, t, arms in self.plan for a in arms]
        return {
            "run_id": self.config.run_id, "image_tag": container.IMAGE, "image_id": image,
            "in_image": inside, "working_tree": host, "tree_matches_image": inside == host,
            "engine_arms": {a["id"]: {"preset": engine_preset(a), "memory_store": a.get("memory_store")}
                            for a in self.arms if a.get("mcp_kind") == "engine"},
            "stores": self.store_hashes(units),
            "proxy_allowlist": container.proxy_allowlist() if self.containerised() else None,
            "pinned": read(path).get("engine") if path.exists() else None,
        }  # fmt: skip

    # ---- one trial ------------------------------------------------------------------------
    def trial(self, index, task_id, arm_id):
        """Leases the store checkout for the trial's (arm, checkpoint) around staging, agent,
        grading and archive, so a concurrent eviction can never pull /context from under it."""
        config, task, arm = self.config, self.public[task_id], self.by_id[arm_id]
        if shutil.disk_usage(config.scratch).free < config.min_free_gib * 2**30:
            raise PauseRun(f"less than {config.min_free_gib} GiB free under {config.scratch}")
        leased = {}
        try:
            for field in ("store", "memory_store"):  # memory_store: kind engine's notes
                if store := arm.get(field):
                    ckpt = task["checkpoint"]
                    leased[field] = self.checkouts.acquire(
                        store, config.stores / store, ckpt,
                        expect=self.pins.get(f"{store}/{ckpt}"), mode=config.store_hash,
                    )  # fmt: skip
            export, memory = (leased[f].path if f in leased else None for f in ("store", "memory_store"))  # fmt: skip
            return self.staged_trial(index, task_id, arm_id, export, memory=memory)
        finally:
            for checkout in leased.values():
                self.checkouts.release(checkout)

    def staged_trial(self, index, task_id, arm_id, export, *, memory=None):
        config, task, arm = self.config, self.public[task_id], self.by_id[arm_id]
        work = config.scratch / task_id / arm_id
        shutil.rmtree(work, ignore_errors=True)  # a crashed attempt was never recorded
        out = work / "out"
        out.mkdir(parents=True)
        workspace = work / "workspace"
        workspace.mkdir()
        dist = None
        if task["kind"] == "coding":
            self.prepare_workspace(task, workspace)
            wrapper = workspace / "gradle/wrapper/gradle-wrapper.properties"
            found = (
                re.search(r"(gradle-[\d.]+)-", wrapper.read_text()) if wrapper.exists() else None
            )
            dist = found.group(1) if found else None
        prompt = build_prompt(task, arm, self.server)
        (out / "prompt.txt").write_text(prompt)
        spec = TrialSpec(
            task_id=task_id, arm=arm_id, kind=task["kind"], prompt=prompt,
            answer_schema=task.get("answer_schema") or CODING_SCHEMA, workspace=workspace, out=out,
            model=config.model, effort=config.effort,
            timeout=getattr(config.timeouts, task["kind"]),
            context=export if arm["access"] == "files" else None,
            sources=export / SOURCES_DB if arm["access"] == "sources" and export else None,
            mcp=McpSpec(
                arm["mcp_kind"], export, self.server["name"], KINDS[arm["mcp_kind"]], config.reader,
                memory=memory, engine=engine_preset(arm) if arm["mcp_kind"] == "engine" else None,
            )
            if arm["access"] == "mcp" and export else None,
            gradle_dist=dist, max_turns=config.max_turns,
            instructions=standing_instructions(arm, self.server),
        )  # fmt: skip
        print(canonical({"trial": [task_id, arm_id], "phase": "agent"}), flush=True)
        try:
            result = self.agent.run(spec)
        finally:  # the engine key file never outlives the container, whatever happened
            shutil.rmtree(out / "engine-key", ignore_errors=True)
        if self.interrupted:
            raise PauseRun("interrupted")
        if result.status == "infra_error":
            raise PauseRun(f"{task_id}/{arm_id}: {result.detail}")
        if spec.mcp and not server_started(out / "mcp/receipts.jsonl"):
            # The MCP server never came up (e.g. a module missing from the image): the agent
            # ran without its tool, which is not a model result of this arm.
            raise PauseRun(f"{task_id}/{arm_id}: the {arm['mcp_kind']} MCP server never started")
        engine = arm.get("mcp_kind") == "engine"
        if spec.mcp and (refusal := reader_infra(out / "mcp/receipts.jsonl")):
            local = engine and reader_infra_scope(out / "mcp/receipts.jsonl") == "trial"
            raise (TrialPause if local else PauseRun)(f"{task_id}/{arm_id}: research reader infrastructure error: {refusal}")
        if engine and (why := mcp_trouble(out / "trace.jsonl", out / "mcp/receipts.jsonl", self.server["name"], result.status)):  # fmt: skip
            raise (PauseRun if "not connected at start" in why else TrialPause)(f"{task_id}/{arm_id}: {why}")
        if engine and result.status in ("timeout", "turn_limit") and (waits := engine_usage(jsonl(out / "mcp/receipts.jsonl"))).get("discarded_attempts"):  # fmt: skip
            # the agent ran out of time while the engine was redoing research the provider degraded
            raise TrialPause(f"{task_id}/{arm_id}: {result.status} after {waits['discarded_attempts']} provider redos ({waits['waited_s']} s of health waits)")  # fmt: skip
        if result.answer is not None:
            atomic_json(out / "answer.json", result.answer)
        if task["kind"] == "historical":
            graded = grade.grade_historical(self.private[task_id], result.answer)
            success = graded["correct"]
        else:
            graded = grade.grade_coding(
                task, self.private[task_id], workspace, out, scratch=work,
                timeout=config.timeouts.grading,
            )  # fmt: skip
            success = graded["resolved"]
        if self.interrupted:
            raise PauseRun("interrupted")
        record = result.record() | {
            "task_id": task_id,
            "arm": arm_id,
            "kind": task["kind"],
            "family": task.get("family") or task["kind"],
            "checkpoint": task.get("checkpoint"),
            "run_index": index,
            "agent": self.agent.name,
            "model": config.model,
            "effort": config.effort,
            "max_turns": config.max_turns,
            "instructions_sha256": sha(spec.instructions) if spec.instructions else None,
            "prompt_sha256": sha(prompt),
            "success": bool(success and result.status == "completed"),
            "grade": graded,
        }
        if arm.get("mcp_kind") == "engine":  # the read side's cost, per model, from the receipts
            record["engine"] = engine_usage(jsonl(out / "mcp/receipts.jsonl"))
        if self.image_id:
            record["image_id"] = self.image_id
        self.archive(out, self.trial_dir(task_id, arm_id), record)
        self.discard(task_id, arm_id)
        return record

    def keep_evidence(self, task_id, arm_id):
        """Engine arms: a paused trial's receipts, engine trace and agent trace go to
        <results>/paused/ for diagnosis (never a result; the trial reruns on resume)."""
        if self.by_id[arm_id].get("mcp_kind") != "engine":
            return
        out = self.config.scratch / task_id / arm_id / "out"
        target = self.config.results / "paused" / f"{task_id}-{arm_id}-{time.time_ns()}"
        for name in ("mcp/receipts.jsonl", "mcp/engine-trace.jsonl", "trace.jsonl", "agent.stderr"):
            if (out / name).is_file():
                target.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(out / name, target / name.replace("/", "-"))

    def discard(self, task_id, arm_id):
        """Scratch is only ever removed by the runner that created it, after publication or
        because the unit is not being recorded."""
        work = self.config.scratch / task_id / arm_id
        shutil.rmtree(work, ignore_errors=True)
        try:
            work.parent.rmdir()
        except OSError:
            pass  # a sibling arm of the same task is still running

    def prepare_workspace(self, task, workspace):
        """Real git repository pruned to the base commit's ancestors (v2/workspace.py)."""
        reference = self.private[task["id"]].get("reference_commit")
        workspaces.stage_workspace(task, workspace, reference, root=self.config.workspaces)
        grade.provision_wrapper(workspace)
        # Visible tests: the golden PR's test changes are part of what the agent is given.
        if (done := grade.apply(self.private[task["id"]]["test_patch"], workspace)).returncode:
            raise PauseRun(
                f"test patch does not apply to the staged workspace: {done.stderr[:300]}"
            )
        if self.containerised():  # the agent's and the grader's offline Gradle must resolve
            grade.ensure_offline_ready(task, workspace, workspace.parent / "prep")

    def archive(self, out, destination, record):
        """Staging directory + rename: a trial directory either holds a complete, hashed
        archive with result.json, or does not exist."""
        stage = destination.with_name(destination.name + ".staging")
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True)
        for name in ARCHIVED:
            source = out / name
            if not source.is_file():
                continue
            target = stage / name.replace("/", "-")
            if source.suffix != ".gz" and source.stat().st_size > GZIP_OVER:
                target = target.with_name(target.name + ".gz")
                target.write_bytes(gzip.compress(source.read_bytes(), mtime=0))
            else:
                shutil.copyfile(source, target)
        atomic_json(stage / "result.json", record)
        files = {p.name: sha(p.read_bytes()) for p in sorted(stage.iterdir())}
        atomic_json(stage / "archive-manifest.json", {"files_sha256": files})
        for staged in stage.iterdir():
            fsync_path(staged)
        stage.replace(destination)
        fsync_path(destination.parent)

    # ---- the run --------------------------------------------------------------------------
    def write_summary(self, status, not_recorded=()):
        summary = summarize(
            self.results, [a["id"] for a in self.arms], status, self.reason, not_recorded
        ) | {"target": sum(len(arms) for _, _, arms in self.plan), "updated_at_unix": time.time()}
        atomic_json(self.config.results / "summary.json", summary)
        return summary

    def task_unit(self, index, task_id, arms):
        coding = self.public[task_id]["kind"] == "coding"
        try:
            self.run_arms(index, task_id, arms, coding)
        finally:
            # The task's repo goes only when EVERY arm of the task is recorded (a coding unit
            # may hold a single arm, the others running in parallel units).
            every = self.task_arms.get(task_id, arms)
            done = all((task_id, arm) in self.results for arm in every)
            if coding and done and not self.config.keep_task_repos:
                workspaces.discard(task_id, root=self.config.workspaces)

    def pass_gate(self):
        """After a trial-scoped pause: the provider health gate, once, before the next trial starts.
        Healthy -> go on at once; unhealthy -> the whole run pauses as before (EXIT_PAUSED, wait)."""
        with self.gate_lock:
            if not self.gate_needed:
                return True
            if why := self.provider_gate():
                with self.lock:
                    self.reason = self.reason or f"after a trial pause: {why}"
                self.stop.set()
                return False
            self.gate_needed = False
            return True

    def run_arms(self, index, task_id, arms, coding):
        for arm in arms:
            if self.stop.is_set() or not self.pass_gate():
                return
            if (task_id, arm) in self.results:
                continue
            with self.lock:
                self.in_flight.add(f"{task_id}/{arm}")
            try:
                if coding:
                    with self.coding_slots:
                        if self.stop.is_set():  # stopped (drain) while waiting for a slot
                            with self.lock:
                                self.in_flight.discard(f"{task_id}/{arm}")
                            return
                        record = self.trial(index, task_id, arm)
                else:
                    record = self.trial(index, task_id, arm)
            except TrialPause as exc:  # this trial only: redo it later, the others go on
                with self.lock:
                    self.in_flight.discard(f"{task_id}/{arm}")
                    self.trial_pauses[(task_id, arm)] += 1
                    count = self.trial_pauses[(task_id, arm)]
                    deferred = count >= TRIAL_PAUSES_BEFORE_DEFER
                    (self.deferred if deferred else self.requeue).append((index, task_id, [arm]))
                    self.pause_log.append({"trial": f"{task_id}/{arm}", "pause": count, "deferred": deferred,
                                           "reason": str(exc)[:500], "at_unix": time.time()})  # fmt: skip
                    self.gate_needed = True
                self.keep_evidence(task_id, arm)
                self.discard(task_id, arm)
                print(canonical({"trial": [task_id, arm], "trial_pause": count, "deferred": deferred, "reason": str(exc)[:300]}),
                      file=sys.stderr, flush=True)  # fmt: skip
                continue
            except PauseRun as exc:
                with self.lock:
                    self.reason = self.reason or str(exc)
                self.stop.set()
                self.keep_evidence(task_id, arm)
                self.discard(task_id, arm)
                return
            except Exception as exc:  # a harness bug is not a model result either
                with self.lock:
                    self.reason = self.reason or f"harness error in {task_id}/{arm}: {exc!r}"
                self.stop.set()
                self.discard(task_id, arm)
                return
            with self.lock:
                self.in_flight.discard(f"{task_id}/{arm}")
                self.results[(task_id, arm)] = record
                self.since_commit += 1
                print(
                    canonical(
                        {
                            "trial": [task_id, arm],
                            "status": record["status"],
                            "success": record["success"],
                            "recorded": len(self.results),
                        }
                    ),
                    flush=True,
                )
                self.write_summary("running")
                if self.config.commit_every and self.since_commit >= self.config.commit_every:
                    self.since_commit = 0
                    try:
                        commit_checkpoint(
                            self.config.results,
                            f"eval: checkpoint {self.config.run_id} ({len(self.results)} trials)",
                        )
                    except (ValueError, subprocess.CalledProcessError) as exc:
                        self.reason = self.reason or f"git checkpoint failed: {exc}"
                        self.stop.set()

    def drain(self, signum=None, _frame=None):
        """SIGUSR1: start no new trial, let the ones in flight finish and be recorded, then
        return, so `main` reloads the config (arms, workers, caps) for the next trials only."""
        self.draining = True
        with self.lock:
            self.reason = self.reason or "draining for a config reload"
        self.stop.set()
        print("DRAIN: no new trials; waiting for the ones in flight", file=sys.stderr, flush=True)

    def interrupt(self, signum, _frame):
        self.interrupted = True
        self.reason = self.reason or f"signal {signal.Signals(signum).name}"
        self.stop.set()
        names = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True
        ).stdout.split()
        mine = [n for n in names if re.match(rf"harnext-(e2|admit)-(.*-)?{os.getpid()}-", n)]
        if mine:
            subprocess.run(["docker", "kill", *mine], capture_output=True)

    def live(self):
        config = self.config
        manifest = self.preflight(live=True)
        if (why := self.provider_gate()):
            self.reason = why
            print(f"PAUSED before any trial: {why}", file=sys.stderr, flush=True)
            return EXIT_PAUSED
        config.results.mkdir(parents=True, exist_ok=True)
        (config.results / ".gitignore").write_text("*.staging/\n*.tmp\n")
        for stale in config.results.glob("trials/*/*.staging"):
            shutil.rmtree(stale)
        for _ in range(5):  # a killed trial's last writes can race the first delete
            shutil.rmtree(config.scratch, ignore_errors=True)  # never recorded, by construction
            if not config.scratch.exists():
                break
            time.sleep(2)
        config.scratch.mkdir(parents=True, exist_ok=True)
        lock = (config.scratch.parent / "runner.lock").open("w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError("another runner holds this run_id") from None
        atomic_json(config.results / "manifest.json", manifest)
        self.session_path = config.results / "sessions" / f"{time.time_ns()}.json"
        self.session = {"argv": sys.argv, "recorded_at_start": len(self.results), "agent_version": manifest["agent_version"],
                        **({"image_id": self.image_id} if self.image_id else {}),
                        **({"allow_image_change": True} if self.allow_image_change else {})}  # fmt: skip
        atomic_json(self.session_path, self.session)
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                signal.signal(signum, self.interrupt)
            signal.signal(signal.SIGUSR1, self.drain)
        self.write_summary("running")
        todo = [
            (i, t, arms)
            for i, t, arms in self.plan
            if any((t, a) not in self.results for a in arms)
        ]
        self.task_arms = {t: arms for _, t, arms in self.plan}
        self.run_units(todo)
        # Kind engine: trials paused on their own (TrialPause) are redone while the run goes on;
        # a trial paused TRIAL_PAUSES_BEFORE_DEFER times waits for one last round at the end.
        while not self.stop.is_set() and self.requeue:
            units, self.requeue = self.requeue, []
            self.run_units(units)
        if not self.stop.is_set() and self.deferred:
            units, self.deferred = self.deferred, []
            print(canonical({"deferred_round": [f"{t}/{a[0]}" for _, t, a in units]}), file=sys.stderr, flush=True)
            self.run_units(units)
        left = sorted({f"{t}/{a[0]}" for _, t, a in self.requeue + self.deferred})
        if left and not self.stop.is_set():
            self.reason = self.reason or f"trials paused {TRIAL_PAUSES_BEFORE_DEFER}+ times (deferred): {left}"
            self.stop.set()
        if self.pause_log:  # the session file records every trial pause and what was deferred
            atomic_json(self.session_path, self.session | {"trial_pauses": self.pause_log, "deferred": left})
            if left:
                print(f"DEFERRED (unrecorded, retried on resume): {left}", file=sys.stderr, flush=True)
        paused = self.stop.is_set()
        summary = self.write_summary(
            "paused" if paused else "complete", sorted(set(self.in_flight) | set(left)) if paused else ()
        )
        if config.commit_every:
            try:
                commit_checkpoint(config.results, f"eval: {summary['status']} {config.run_id} ({len(self.results)} trials)")  # fmt: skip
            except (ValueError, subprocess.CalledProcessError) as exc:
                print(f"results are saved but NOT committed: {exc}", file=sys.stderr)
        lock.close()
        if paused:
            print(
                f"PAUSED: {self.reason}\nNothing was recorded for: {sorted(self.in_flight) or 'n/a'}\n"
                f"{len(self.results)}/{summary['target']} trials recorded. Fix the cause, then "
                f"resume with the same command:\n  python -m harnext_eval.e2.v2.runner "
                f"--config <same yaml> --live",
                file=sys.stderr, flush=True,
            )  # fmt: skip
            return EXIT_PAUSED
        print(canonical({"status": "complete", "recorded": len(self.results)}), flush=True)
        return 0

    def run_units(self, todo):
        """Run task units: coding units in their own slots when there are workers beyond them."""
        config = self.config
        # Two queues: coding units hold the coding slots, historical units get the remaining
        # workers, so neither kind waits in line behind the other (with one shared queue, spare
        # workers sat blocked on the coding semaphore while historical tasks were pending).
        # Only when there are workers beyond the coding slots; otherwise one queue, as before.
        if config.workers > config.coding_workers:
            # One coding unit per (task, arm): the arms of one task run in parallel slots, so
            # the last tasks do not run their 12 arms one after another (the run's long tail).
            coding = [(i, t, [a]) for i, t, arms in todo if self.public[t]["kind"] == "coding"
                      for a in arms if (t, a) not in self.results]  # fmt: skip
            other = [u for u in todo if self.public[u[1]]["kind"] != "coding"]
            with (
                ThreadPoolExecutor(config.coding_workers) as code_pool,
                ThreadPoolExecutor(config.workers - config.coding_workers) as other_pool,
            ):
                pending = [code_pool.map(lambda unit: self.task_unit(*unit), coding),
                           other_pool.map(lambda unit: self.task_unit(*unit), other)]  # fmt: skip
                for results in pending:
                    list(results)
        else:
            with ThreadPoolExecutor(config.workers) as pool:
                list(pool.map(lambda unit: self.task_unit(*unit), todo))

    def check_resume(self, *, dry=False):
        if not dry:
            self.preflight(live=False)
        left = self.remaining()
        report = {
            "status": "dry_run" if dry else "resume_validated",
            "model_calls": 0,
            "recorded": len(self.results),
            "remaining": len(left),
            "per_arm": {
                a["id"]: {
                    "recorded": sum(arm == a["id"] for _, arm in self.results),
                    "remaining": sum(arm == a["id"] for _, _, arm in left),
                }
                for a in self.arms
            },
        }
        if dry:
            report["first_units"] = [[t, arms] for _, t, arms in self.plan[:5]]
            report["access_paragraphs"] = {
                a["access"]: build_prompt({"prompt": "", "kind": "historical"}, a, self.server)
                for a in self.arms
            }
        print(json.dumps(report, indent=2), flush=True)
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--check-resume", action="store_true", help="validate; no model calls")
    mode.add_argument("--dry-run", action="store_true", help="print the plan; no checks on docker")
    mode.add_argument("--identity", action="store_true",
                      help="print image ID, engine / raw store code hashes, engine config and store hashes")  # fmt: skip
    parser.add_argument("--extend-run", action="store_true",
                        help="add arms to a run that is complete (refused otherwise)")  # fmt: skip
    parser.add_argument("--allow-image-change", action="store_true",
                        help="resume engine arms on a different image ID with the same code (recorded)")  # fmt: skip
    parser.add_argument("--no-wait", action="store_true",
                        help="exit 75 on a usage-limit pause instead of sleeping until the reset")  # fmt: skip
    args = parser.parse_args(argv)
    if args.identity:
        print(json.dumps(Runner(load(args.config)).identity_report(), indent=2, sort_keys=True))
        return 0
    if not args.live:
        runner = Runner(load(args.config))
        runner.extend_run = args.extend_run
        return runner.check_resume(dry=args.dry_run)
    while True:
        runner = Runner(load(args.config))
        runner.allow_image_change = args.allow_image_change
        runner.extend_run = args.extend_run
        code = runner.live()
        if runner.draining and not runner.interrupted:
            print("DRAIN done: reloading the config", file=sys.stderr, flush=True)
            continue
        if code != EXIT_PAUSED or args.no_wait or runner.interrupted:
            return code
        # Usage limits: every recorded trial is on disk; a fresh Runner resumes past them.
        delay = reset_delay(runner.reason)
        print(f"waiting {delay // 60} min, then resuming", file=sys.stderr, flush=True)
        time.sleep(delay)


if __name__ == "__main__":
    sys.exit(main())
