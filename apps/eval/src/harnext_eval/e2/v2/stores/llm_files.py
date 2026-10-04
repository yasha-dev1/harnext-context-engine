"""LLM-built file stores F2–F6 (Zhou et al. 2607.26637 variants over a company event stream;
F5 is the exception: the Harnext product's own batch lane, see `product.py`).

Each window is handed to a builder agent with file tools over the store tree, through the
builder's harness layer (Claude Code harness, Haiku 4.5; or the offline fake). The paper's
seven tools map to: view→Read/Glob, create→Write, str_replace+insert→Edit, grep→Grep,
delete/rename→the in-process tools of `fs_tools.py` (the harness's sandboxed Bash cannot
start on the build host, so no builder has a shell); file tools are confined to the tree by
permission rules. F2 is move-only through a validated move plan (`foldered.py`).
What the sandbox does not enforce is verified after every attempt: git HEAD
untouched, no tool call addressed a path outside the tree, and per-arm invariants.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path, PurePosixPath

from harnext_builder.harness.base import ConversationTranscript, EventFile, HarnessRequest
from harnext_builder.harness.runner import _materialize

from harnext_eval.e2.v2.errors import PauseRun, is_infra
from harnext_eval.e2.v2.stores import foldered, fs_tools, product, prompts
from harnext_eval.e2.v2.stores.base import (
    Repo,
    Store,
    event_path,
    render,
    safe_id,
    tree_files,
    write,
)
from harnext_eval.e2.v2.stores.fake import SECTIONS, FakeBuilder

WRITE_TOOLS = fs_tools.ALLOWED_TOOLS  # the paper's seven tools, no shell (`fs_tools.py`)
DENIED = fs_tools.DENIED_TOOLS
INLINE_LIMIT = 120_000  # characters of event text inlined into one instruction
CONSECUTIVE_FAILURES = 3
ABSOLUTE = re.compile(r"(?<![\w.~-])/(?:[\w.+-]+/)*[\w.+-]+")
REFUSED = re.compile(
    r"Permission to use \w+ has been denied|(delete|rename) refused \(|"
    r"(<tool_use_error>)?(Path|File) does not exist"
)
HARMLESS = ("/dev/null", "/bin/", "/usr/bin/")
TOOL_RESULTS = fs_tools.TOOL_RESULTS
FILE_TOOLS = {"Read", "Write", "Edit", "Glob", "Grep", "MultiEdit", "NotebookEdit"}
TRUNCATED_ARG = re.compile(r'"(file_path|path|notebook_path|old_path|new_path|glob)"\s*:\s*"((?:[^"\\]|\\.)*)"')
PATH_KEYS = {"file_path", "path", "notebook_path", "command", "cwd", "pattern", "glob",
             "old_path", "new_path"}  # fmt: skip


def get_builder(builder: str, arm: str):
    if builder == "fake":
        return FakeBuilder(arm)
    if builder == "claude":
        if arm == "F2":  # move plan only: no delete/rename tools on offer
            from harnext_builder.harness.registry import get_harness

            return get_harness("claude_code")
        return fs_tools.StoreHarness()
    raise ValueError(f"unknown builder: {builder!r}")


def event_files(events: list[dict]) -> list[EventFile]:
    names = [f"_event/{n:04d}-{safe_id(e['id'])}.md" for n, e in enumerate(events, 1)]
    listing = "\n".join(
        f"- `{name}` — {e['time']} {e['type']}: {e.get('title') or ''}"
        for name, e in zip(names, events, strict=True)
    )
    manifest = f"# Events of this window, in time order\n\n{listing}\n"
    return [
        EventFile(path="_event/MANIFEST.md", content=manifest),
        *(EventFile(path=n, content=render(e)) for n, e in zip(names, events, strict=True)),
    ]


def inline(events: list[dict]) -> str:
    return "\n".join(
        f"===== event {n} of {len(events)} =====\n{render(e)}" for n, e in enumerate(events, 1)
    )


def usage_of(transcript: ConversationTranscript, seconds: float) -> dict:
    usage = transcript.usage.get("usage") or {}
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cache_read_tokens": usage.get("cache_read_input_tokens"),
        "cache_write_tokens": usage.get("cache_creation_input_tokens"),
        "cost_usd": transcript.usage.get("total_cost_usd"),
        "turns": transcript.usage.get("num_turns"),
        "seconds": round(seconds, 2),
        "stop_reason": transcript.stop_reason,
    }


def outside(tree: Path, transcript: ConversationTranscript) -> list[str]:
    """Absolute or parent-relative paths in tool calls that leave the tree."""
    root, found = str(tree.resolve()), []
    calls = [t for t in transcript.turns if t.role == "tool_use"]
    results = [t.content for t in transcript.turns if t.role == "tool_result"]
    for n, turn in enumerate(calls):
        # A call the harness refused had no effect and showed the builder nothing: the
        # permission rules and `fs_tools` deny every path outside the tree. Haiku often
        # writes the store root as `/` (`/issues/x.md`), is refused, and retries relatively.
        if n < len(results) and len(calls) == len(results) and REFUSED.match(results[n]):
            continue
        name = getattr(turn, "tool_name", None)
        try:
            # Only path-bearing arguments: file *contents* legitimately hold root-relative
            # links such as "/INDEX.md", which are not tool calls outside the tree.
            call = json.loads(turn.content)
            if not isinstance(call, dict):
                raise ValueError
        except (ValueError, AttributeError):
            # Transcripts truncate long inputs (a big Write/Edit): recover the path arguments,
            # which come before the content; never scan the content itself.
            call = dict(TRUNCATED_ARG.findall(turn.content or ""))
            if not call and name not in FILE_TOOLS:
                call = {"command": turn.content}
        for key, value in call.items():
            value = str(value)
            if key not in PATH_KEYS or (key == "pattern" and name == "Grep"):
                continue  # a Grep pattern is a regex over contents (URLs, "/tmp"), not a path
            if name == "Read" and key == "file_path" and TOOL_RESULTS.match(value):
                continue  # Claude Code's own spill file for an oversized tool output
            if key == "command":  # free text: scan it
                paths = ABSOLUTE.findall(value) + re.findall(r"(?:^|[\s'\"=])(\.\./\S*)", value)
            else:  # the argument IS one path or glob ("entities/*/OVERVIEW.md" is relative)
                paths = [value] if value.startswith(("/", "~")) or ".." in PurePosixPath(value).parts else []  # fmt: skip
            found += [
                p
                for p in paths
                if not (p == root or p.startswith(root + "/") or p.startswith(HARMLESS))
            ]
    return sorted(set(found))


def verify_moves(pre: dict[str, str], post: dict[str, str]) -> str | None:
    """F2 invariant: the multiset of file contents is unchanged — every file still exists
    byte-identical exactly once somewhere, and no other content file was created."""
    before, after = Counter(pre.values()), Counter(post.values())
    if before == after:
        return None
    lost, extra = sum((before - after).values()), sum((after - before).values())
    return f"move-only violated: {lost} file(s) lost or edited, {extra} foreign file(s)"


class LLMStore(Store):
    llm = True
    tools = WRITE_TOOLS
    denied = DENIED

    window_default = 25
    bytes_default = 60_000  # event text per window; the cost lever is window size, never dropping

    def __init__(self, *, builder: str = "claude", harness=None, max_turns: int = 200,
                 timeout_s: int = 1800, window_bytes: int | None = None, **kwargs):  # fmt: skip
        super().__init__(window_bytes=window_bytes or self.bytes_default, **kwargs)
        self.cache: dict = {}  # stat-keyed hashes and line counts: no full re-read per window
        self.builder, self.max_turns, self.timeout_s = builder, max_turns, timeout_s
        self.harness = harness or get_builder(builder, self.arm)

    def config(self) -> dict:
        return super().config() | {
            "builder": self.builder,
            "model": prompts.MODEL if self.builder == "claude" else self.builder,
            "max_turns": self.max_turns,
            "timeout_s": self.timeout_s,
            "tools": self.tools,
        }

    # -- per-arm hooks ---------------------------------------------------------------
    def system(self) -> str:
        return prompts.system(self.arm)

    def prepare(self, tree: Path, events: list[dict]) -> None:
        """Code-written files of this window; redone after every rollback."""

    def request(self, tree: Path, events: list[dict], index: int) -> tuple[str, list[EventFile]]:
        body = inline(events)
        template = (
            prompts.WINDOW_INSTRUCTION if len(body) <= INLINE_LIMIT else prompts.WINDOW_INSTRUCTION_FILES
        )  # fmt: skip
        instruction = template.format(
            index=index, count=len(events), first=events[0]["time"], last=events[-1]["time"], events=body
        )  # fmt: skip
        return instruction, event_files(events)

    def verify(self, tree: Path, pre: dict[str, str], post: dict[str, str]) -> str | None:
        return None

    def fallback(self, tree: Path, events: list[dict]) -> None:
        """A failed window loses nothing: its events are kept verbatim."""
        for event in events:
            write(tree / event_path(event, "_unprocessed"), render(event))

    def kept_verbatim(self, tree: Path, ids: list[str]) -> list[str]:
        """Events of failed windows must still sit verbatim under `_unprocessed/`."""
        names = {p.name for p in (tree / "_unprocessed").rglob("*.md")} if ids else set()
        return [i for i in ids if f"{safe_id(i)}.md" not in names]

    def extras(self, tree: Path) -> dict:
        return {}

    # -- one window -------------------------------------------------------------------
    def run(self, tree: Path, instruction: str, files: list[EventFile]):
        req = HarnessRequest(
            harness=self.harness.name,
            working_dir=str(tree),
            instruction=instruction,
            system_prompt=self.system(),
            allowed_tools=self.tools,
            disallowed_tools=self.denied,
            event_files=files,
            model=prompts.MODEL,
            max_turns=self.max_turns,
            timeout_s=self.timeout_s,
        )
        _materialize(tree, files)
        started = time.monotonic()
        try:
            transcript = asyncio.run(self.harness.run(req))
        finally:
            shutil.rmtree(tree / "_event", ignore_errors=True)
        return transcript, usage_of(transcript, time.monotonic() - started)

    def problem(self, tree: Path, transcript: ConversationTranscript, head: str, pre, post):
        if transcript.stop_reason == "timeout":
            return "timeout"
        if transcript.error:
            spoke = any(t.role in ("assistant", "tool_use") for t in transcript.turns)
            # A provider refusal (usage limit, 429) arrives as a "result" whose text the SDK
            # drops from its exception, and the model never consumed a token.
            # Only the CLI's final result can carry a provider refusal; the builder's own words
            # about Kafka ("network", "authentication") are never evidence of one.
            said = " ".join(t.content for t in transcript.turns if t.role == "result")
            spent = [usage_of(transcript, 0.0).get(k) for k in ("input_tokens", "output_tokens")]
            unused = all(v == 0 for v in spent)  # reported as zero, not merely unreported
            if is_infra(transcript.error) or is_infra(said[-2000:]) or unused or not spoke:
                raise PauseRun(f"{self.arm}: builder infrastructure error: {transcript.error}")
            return f"{transcript.stop_reason}: {transcript.error}"[:500]
        if transcript.stop_reason not in ("completed", "end_turn"):
            return transcript.stop_reason
        if Repo(tree).head() != head:
            return "builder touched git history"
        if paths := outside(tree, transcript):
            return f"tool call outside the store: {paths[:5]}"
        if pre == post:
            return "no change to the store"
        kept = {p: h for p, h in pre.items() if p.startswith("_unprocessed/")}
        if any(post.get(p) != h for p, h in kept.items()):
            return "builder changed files under _unprocessed/"
        return self.verify(tree, pre, post)

    def save_transcript(self, transcript: ConversationTranscript, index: int, attempt: int) -> str:
        """Every builder turn is kept so any file in the store can be traced to the turn
        that wrote it (E2-V2-DESIGN.md: saved transcripts)."""
        folder = self.out / "transcripts"
        folder.mkdir(exist_ok=True)
        name = f"{index:06d}-{attempt}.json"
        (folder / name).write_text(transcript.model_dump_json())
        return f"transcripts/{name}"

    def attempt(self, tree: Path, events: list[dict], index: int) -> dict:
        repo, attempts = Repo(tree), []
        head = repo.head()
        for _ in range(2):
            repo.rollback()
            self.prepare(tree, events)
            pre = tree_files(tree, self.cache)
            instruction, files = self.request(tree, events, index)
            transcript, record = self.run(tree, instruction, files)
            attempts.append(record)
            record["transcript"] = self.save_transcript(transcript, index, len(attempts))
            post = tree_files(tree, self.cache)
            record["problem"] = self.problem(tree, transcript, head, pre, post)
            if record["problem"] is None:
                return {"attempts": attempts, **self.extras(tree)}
        recent = [f["window"] for f in self.state["failed_windows"][-(CONSECUTIVE_FAILURES - 1) :]]
        if recent == list(range(index - CONSECUTIVE_FAILURES + 1, index)):
            raise PauseRun(
                f"{self.arm}: {CONSECUTIVE_FAILURES} consecutive failed windows, last: "
                f"{attempts[-1]['problem']}"
            )
        repo.rollback(head)
        self.prepare(tree, events)
        self.fallback(tree, events)
        return {"attempts": attempts, "failed": attempts[-1]["problem"], **self.extras(tree)}

    def apply(self, tree: Path, events: list[dict], index: int, last_of_month: bool) -> dict:
        return self.attempt(tree, events, index)


class F2(LLMStore):
    """Foldered (Zhou §3): F0 files written by code, then moved intact by the model. The
    model has no shell and no rename tool: it writes the move plan `_moves.json`, which
    `foldered.execute` validates and performs (`stores/foldered.py`, own prompt digest)."""

    arm = "F2"
    tools = foldered.ALLOWED_TOOLS
    denied = foldered.DENIED_TOOLS

    def system(self) -> str:
        return foldered.system()

    def manifest(self) -> dict:
        return {**self.config(), "prompts_sha256": foldered.digest()}

    def prepare(self, tree: Path, events: list[dict]) -> None:
        for event in events:
            write(tree / event_path(event), render(event))

    def request(self, tree: Path, events: list[dict], index: int):
        listing = "\n".join(
            f"- `{event_path(e)}` — {e['type']} {' '.join(e.get('entities') or [])}: {e.get('title') or ''}"
            for e in events
        )
        return foldered.INSTRUCTION.format(
            index=index, count=len(events), listing=listing, folders=foldered.folders(self.cache)
        ), []

    def verify(self, tree: Path, pre, post) -> str | None:
        """`post` is the tree as the builder left it: identical to `pre` plus the plan."""
        written = {p: h for p, h in post.items() if not foldered.PLANS.fullmatch(p)}
        if len(written) == len(post):
            return "no change to the store" if pre == post else verify_moves(pre, post)
        if written != pre:
            return verify_moves(pre, written) or "builder moved or wrote files itself"
        if problem := foldered.execute(tree, pre):
            return problem
        return verify_moves(pre, tree_files(tree, self.cache))

    def fallback(self, tree: Path, events: list[dict]) -> None:
        """`prepare` already placed the window's files at their default F0 paths."""

    def kept_verbatim(self, tree: Path, ids: list[str]) -> list[str]:
        return []  # enforced for every window by `verify_moves`, and by the count below

    def audit(self, out: Path, state: dict, seen: list) -> dict:
        result = super().audit(out, state, seen)
        files = len(tree_files(out / "tree", self.cache))
        if files != result["events"]:
            raise RuntimeError(f"F2: {files} files in the tree for {result['events']} events")
        return result


class F3(LLMStore):
    """Reorganized (preserve): verbatim intake, restructured every `reorg_every` windows
    and at every checkpoint boundary under the keep-every-fact rule."""

    arm = "F3"

    def __init__(self, *, reorg_every: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.reorg_every = reorg_every

    def config(self) -> dict:
        return super().config() | {"reorg_every": self.reorg_every}

    def prepare(self, tree: Path, events: list[dict]) -> None:
        for event in events:
            write(tree / event_path(event, "_intake"), render(event))

    def intake(self, tree: Path) -> list[Path]:
        return sorted((tree / "_intake").rglob("*.md"))

    def request(self, tree: Path, events: list[dict], index: int):
        count = len(self.intake(tree))
        return prompts.REORG_INSTRUCTION.format(index=index, count=count), []

    def fallback(self, tree: Path, events: list[dict]) -> None:
        """Nothing to do: unrestructured events stay in `_intake/` for the next pass."""

    def kept_verbatim(self, tree: Path, ids: list[str]) -> list[str]:
        return []  # a failed pass is rolled back to the full intake; a later pass takes it

    def extras(self, tree: Path) -> dict:
        return {"intake_left": len(self.intake(tree))}

    def apply(self, tree: Path, events: list[dict], index: int, last_of_month: bool) -> dict:
        if (index + 1) % self.reorg_every and not last_of_month:
            self.prepare(tree, events)
            return self.extras(tree)
        before = sum(p.stat().st_size for p in tree.rglob("*.md") if ".git" not in p.parts)
        record = self.attempt(tree, events, index)
        after = sum(p.stat().st_size for p in tree.rglob("*.md") if ".git" not in p.parts)
        return record | {"bytes_before": before + sum(len(render(e).encode()) for e in events), "bytes_after": after}  # fmt: skip


class Curated(LLMStore):
    """Agent-curated stores with a prescribed granularity (F4-*) or none (F6)."""


class F4Fact(Curated):
    arm = "F4-fact"


class F4Entity(Curated):
    arm = "F4-entity"


class F4Topic(Curated):
    arm = "F4-topic"


class F5(LLMStore):
    """Harnext as shipped: the product's batch lane (`stores/product.py`) — its seed, its
    system prompt, its batch instruction per entity, its tool policy. Only the cadence is
    the eval's: one builder run per slice of the stream, not one per 30 s entity window."""

    arm = "F5"
    tools = product.TOOLS
    denied = product.DENIED

    def seed(self) -> dict[str, str]:
        return dict(product.SEED_FILES)

    def system(self) -> str:
        return product.SYSTEM_PROMPT

    def manifest(self) -> dict:
        return {**self.config(), "prompts_sha256": product.digest()}

    def request(self, tree: Path, events: list[dict], index: int):
        return product.request(events)

    def verify(self, tree: Path, pre, post) -> str | None:
        missing = [path for path in product.KEEP if path not in post]
        return f"layout file removed: {missing}" if missing else None


class F6(Curated):
    arm = "F6"


ARMS = {cls.arm: cls for cls in (F2, F3, F4Fact, F4Entity, F4Topic, F5, F6)}
__all__ = ["ARMS", "SECTIONS", "LLMStore", "outside", "verify_moves"]
