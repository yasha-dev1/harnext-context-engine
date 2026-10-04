"""Common mechanics of every file store: git tree, windows, checkpoints, resume, export.

A store lives in `out/kafka-500-v2/stores/<arm>/`: `tree/` is a git repository with
one commit per build window and a tag `ckpt-YYYY-MM` per checkpoint; `build-state.json`
is the resume cursor, written atomically after each commit; `build-log.jsonl` has one
line per window attempt; `export/<YYYY-MM>/` is the read-only checkpoint for trials.
Identity and dates of every commit derive from the events, so an uninterrupted build
and a killed-and-resumed one produce the same commit hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Protocol

import yaml

from harnext_eval.e2.v2.stores import prompts

ROOT = Path(__file__).resolve().parents[5]  # apps/eval
OUT = ROOT / "out/kafka-500-v2"
WINDOW_EVENTS = 25
EPOCH = "2023-01-01T00:00:00Z"
IDENTITY = {
    "GIT_AUTHOR_NAME": "harnext-store-builder",
    "GIT_AUTHOR_EMAIL": "builder@harnext.invalid",
    "GIT_COMMITTER_NAME": "harnext-store-builder",
    "GIT_COMMITTER_EMAIL": "builder@harnext.invalid",
}
USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")


class StoreBuilder(Protocol):
    arm: str

    def build(self, events: Iterable[dict], out: Path, *, until: str | None = None) -> None: ...

    def export(self, out: Path, checkpoint: str) -> Path: ...


class InjectedKill(RuntimeError):  # noqa: N818
    """Raised by tests through `Store.kill_after` to simulate a killed build."""


def safe_id(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.") or "event"
    return (
        slug if slug == value else f"{slug[:80]}-{hashlib.sha256(value.encode()).hexdigest()[:8]}"
    )


def month(event: dict) -> str:
    return event["time"][:7]


def next_month(ym: str) -> str:
    year, mon = int(ym[:4]), int(ym[5:7])
    return f"{year + mon // 12:04d}-{mon % 12 + 1:02d}"


def frontmatter(event: dict) -> str:
    fields = {k: event.get(k) for k in ("id", "time", "type", "entities", "actor", "title")}
    return yaml.safe_dump(fields, sort_keys=False, allow_unicode=True, width=10**6)


def render(event: dict) -> str:
    """The F0 file of an event: YAML frontmatter, then `text` verbatim."""
    return f"---\n{frontmatter(event)}---\n\n{event.get('text') or ''}\n"


def event_path(event: dict, root: str = "events") -> str:
    year, mon, day = event["time"][:10].split("-")
    return f"{root}/{year}/{mon}/{day}/{safe_id(event['id'])}.md"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def save_json(path: Path, value) -> None:
    """Atomic: a killed process leaves the previous file or the new one, never a torn one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_events(path: Path) -> Iterator[dict]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def windows(
    events: Iterable[dict], size: int, max_bytes: int | None = None
) -> Iterator[tuple[str, list[dict], bool]]:
    """(month, events, last-of-month). A window closes at `size` events or before the event
    whose text would take it past `max_bytes` (an oversized event is a window of its own),
    whichever hits first, and never crosses a checkpoint boundary. The partition restarts
    each month and depends only on the events, so a cursor at a window end reproduces it."""
    batch: list[dict] = []
    used = 0
    for event in events:
        weight = len((event.get("text") or "").encode())
        full = len(batch) == size or (max_bytes is not None and used + weight > max_bytes)
        if batch and (full or month(event) != month(batch[0])):
            yield month(batch[0]), batch, month(event) != month(batch[0])
            batch, used = [], 0
        batch.append(event)
        used += weight
    if batch:
        yield month(batch[0]), batch, True


def tree_files(tree: Path, cache: dict | None = None) -> dict[str, str]:
    """relpath → sha256 of every file in the tree outside `.git`. With a `cache`
    (relpath → (mtime_ns, size, sha256, lines)) only files whose stat changed are read, so
    scanning a large store every window costs a walk, not a re-hash of the whole store."""
    result, fresh = {}, {}
    for directory, names, files in os.walk(tree):
        names[:] = sorted(n for n in names if n != ".git")
        for name in sorted(files):
            path = Path(directory) / name
            rel, stat = str(path.relative_to(tree)), path.stat()
            known = cache.get(rel) if cache is not None else None
            if not known or known[:2] != (stat.st_mtime_ns, stat.st_size):
                data = path.read_bytes()
                lines = data.count(b"\n") + (not data.endswith(b"\n") and bool(data))
                known = (stat.st_mtime_ns, stat.st_size, hashlib.sha256(data).hexdigest(), lines)
            fresh[rel] = known
            result[rel] = known[2]
    if cache is not None:
        cache.clear()
        cache.update(fresh)
    return result


def content_hash(files: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(sorted(files.items())).encode()).hexdigest()


class Repo:
    def __init__(self, path: Path):
        self.path = path

    def git(self, *args: str, date: str | None = None) -> str:
        env = {
            **os.environ,
            **IDENTITY,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
        }
        if date:
            env |= {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
        return subprocess.run(
            ["git", "-C", str(self.path), *args],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        ).stdout.strip()

    def init(self) -> None:
        if (self.path / ".git").exists():
            return
        self.path.mkdir(parents=True, exist_ok=True)
        self.git("init", "-q", "-b", "main")
        self.git("commit", "-q", "--allow-empty", "-m", "store: empty", date=EPOCH)

    def head(self) -> str:
        return self.git("rev-parse", "HEAD")

    def commit(self, message: str, date: str) -> str:
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", message, date=date)
        return self.head()

    def rollback(self, to: str = "HEAD") -> None:
        self.git("reset", "-q", "--hard", to)
        self.git("clean", "-fdq")

    def tag(self, name: str) -> None:
        self.git("tag", "-f", name)


def _is_ancestor(repo: Repo, ref: str, head: str) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(repo.path), "merge-base", "--is-ancestor", ref, head],
            capture_output=True,
        ).returncode
        == 0
    )


class Store:
    """Base of every git-tree store. Subclasses implement `apply` (write one window into
    the tree, return a usage record) and may override `seed` and `config`."""

    arm = ""
    llm = False

    window_default = 1000  # events per commit for code-built stores; LLM stores use 25

    def __init__(self, *, window_events: int | None = None, window_bytes: int | None = None):
        self.window_events = window_events or self.window_default
        self.window_bytes = window_bytes
        self.kill_after: int | None = None  # tests: raise InjectedKill at that window
        self.kill_point = "mid_window"  # or "after_commit" (committed, cursor not yet saved)

    # -- hooks ---------------------------------------------------------------------
    def seed(self) -> dict[str, str]:
        return {}

    def config(self) -> dict:
        return {
            "arm": self.arm,
            "window_events": self.window_events,
            "window_bytes": self.window_bytes,
        }

    def apply(self, tree: Path, events: list[dict], index: int, last_of_month: bool) -> dict:
        raise NotImplementedError

    # -- mechanics -----------------------------------------------------------------
    def manifest(self) -> dict:
        return {**self.config(), "prompts_sha256": prompts.digest() if self.llm else None}

    def load(self, out: Path) -> dict:
        path = out / "build-state.json"
        state = json.loads(path.read_text()) if path.exists() else None
        if state and state["config"] != self.manifest():
            raise ValueError(
                f"{self.arm}: store was built with {state['config']}, now {self.manifest()}; "
                "rebuild from scratch or restore the configuration"
            )
        return state or {
            "config": self.manifest(),
            "last_event_id": None,
            "last_event_time": None,
            "window": 0,
            "head": None,
            "checkpoints": [],
            "failed_windows": [],
            "usage": dict.fromkeys(USAGE_KEYS, 0) | {"cost_usd": 0.0, "seconds": 0.0, "turns": 0},
            "done": False,
        }

    def log(self, out: Path, record: dict) -> None:
        with (out / "build-log.jsonl").open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def build(self, events: Iterable[dict], out: Path, *, until: str | None = None) -> None:
        out.mkdir(parents=True, exist_ok=True)
        repo, state = Repo(out / "tree"), self.load(out)
        self.out, self.state = out, state
        repo.init()
        if state["head"] is None:
            for path, text in self.seed().items():
                write(repo.path / path, text)
            state["head"] = repo.commit("store: seed", EPOCH)
            save_json(out / "build-state.json", state)
        # Anything past the cursor (a kill between commit and cursor write, or mid-window
        # debris) is discarded: the window is redone from the recorded head.
        repo.rollback(state["head"])
        for name in repo.git("tag", "--list", "ckpt-*").split():
            if not _is_ancestor(repo, name, state["head"]):
                repo.git("tag", "-d", name)

        seen: list[tuple[str, str]] = []
        pending = self.pending(events, state, seen)
        for ym, batch, last_of_month in windows(pending, self.window_events, self.window_bytes):
            if until and ym > until:
                break
            self.gap_tags(repo, state, ym)
            index = state["window"]
            if self.kill_after == index and self.kill_point == "mid_window":
                write(repo.path / "_debris.md", "half-written window\n")
                raise InjectedKill(f"window {index}")
            try:
                record = self.apply(repo.path, batch, index, last_of_month)
            except BaseException:
                repo.rollback()  # PauseRun, Ctrl-C or a bug: leave a clean tree at the cursor
                raise
            first, last = batch[0], batch[-1]
            state["head"] = repo.commit(
                f"window {index}: {first['id']} .. {last['id']} ({len(batch)} events)", last["time"]
            )
            if self.kill_after == index and self.kill_point == "after_commit":
                raise InjectedKill(f"window {index} committed, cursor not saved")
            state |= {
                "window": index + 1,
                "last_event_id": last["id"],
                "last_event_time": last["time"],
            }
            self.account(state, record, index)
            if last_of_month:
                repo.tag(f"ckpt-{ym}")
                state["checkpoints"].append(ym)
            # Log first, cursor second: a kill in between redoes the window and logs it again;
            # readers of the log take the last record of a window index.
            self.log(out, {"window": index, "month": ym, "first": first["id"], "last": last["id"],
                           "events": len(batch), "ids": [e["id"] for e in batch],
                           "commit": state["head"], **record})  # fmt: skip
            save_json(out / "build-state.json", state)
        else:
            state["done"] = until is None
        if until:
            self.gap_tags(repo, state, next_month(until))
        state["audit"] = self.audit(out, state, seen)
        save_json(out / "build-state.json", state)

    def audit(self, out: Path, state: dict, seen: list[tuple[str, str]]) -> dict:
        """Build-end invariant: the ids the build log accounts for (processed windows plus
        failed windows kept verbatim) are exactly the stream's ids up to the cursor —
        nothing skipped, sampled, duplicated or invented."""
        cursor = (state["last_event_time"] or "", state["last_event_id"] or "")
        stream = [i for t, i in seen if (t, i) <= cursor]
        records: dict[int, dict] = {}
        path = out / "build-log.jsonl"
        for line in path.open() if path.exists() else []:
            record = json.loads(line)
            if record["window"] < state["window"]:
                records[record["window"]] = record
        logged = [i for index in sorted(records) for i in records[index]["ids"]]
        failed = {f["window"] for f in state["failed_windows"]}
        kept = self.kept_verbatim(out / "tree", [i for w in failed for i in records[w]["ids"]])
        problems = []
        if logged != stream:
            missing, extra = set(stream) - set(logged), set(logged) - set(stream)
            problems.append(
                f"log/stream mismatch: {len(missing)} missing, {len(extra)} extra, "
                f"{len(logged) - len(set(logged))} duplicated, order differs: {not missing and not extra}"
            )
        if kept:
            problems.append(
                f"{len(kept)} events of failed windows are not kept verbatim: {kept[:5]}"
            )
        if problems:
            raise RuntimeError(
                f"{self.arm}: event accounting invariant violated — " + "; ".join(problems)
            )
        return {"events": len(stream), "processed": len(stream) - sum(records[w]["events"] for w in failed),
                "unprocessed": sum(records[w]["events"] for w in failed), "through": cursor[1] or None}  # fmt: skip

    def kept_verbatim(self, tree: Path, ids: list[str]) -> list[str]:
        """Ids of failed-window events that are NOT preserved verbatim in the tree."""
        return []

    def pending(self, events: Iterable[dict], state: dict, seen: list) -> Iterator[dict]:
        """Events after the cursor; every event read is noted in `seen` for the audit. The
        stream must be strictly sorted by (time, id): resume and checkpoints rely on it."""
        cursor = (state["last_event_time"], state["last_event_id"])
        previous = ("", "")
        for event in events:
            key = (event["time"], event["id"])
            if key <= previous:
                raise ValueError(f"stream is not strictly sorted by (time, id) at {event['id']}")
            previous = key
            seen.append(key)
            if cursor[1] is None or key > cursor:
                yield event

    def gap_tags(self, repo: Repo, state: dict, ym: str) -> None:
        """Months without events still get a checkpoint: the state at their boundary."""
        done = state["checkpoints"]
        gap = next_month(done[-1]) if done else ym
        while gap < ym:
            repo.tag(f"ckpt-{gap}")
            done.append(gap)
            gap = next_month(gap)

    def account(self, state: dict, record: dict, index: int) -> None:
        usage = state["usage"]
        for attempt in record.get("attempts", []):
            for key in USAGE_KEYS:
                usage[key] += attempt.get(key) or 0
            usage["cost_usd"] += attempt.get("cost_usd") or 0.0
            usage["seconds"] += attempt.get("seconds") or 0.0
            usage["turns"] += attempt.get("turns") or 0
        if record.get("failed"):
            state["failed_windows"].append({"window": index, "reason": record["failed"]})

    def export(self, out: Path, checkpoint: str) -> Path:
        repo, target = Repo(out / "tree"), out / "export" / checkpoint
        tag = f"ckpt-{checkpoint}"
        commit = repo.git("rev-parse", f"{tag}^{{commit}}")
        if target.exists():
            unlock(target)
            shutil.rmtree(target)
        target.mkdir(parents=True)
        archive = subprocess.Popen(
            ["git", "-C", str(repo.path), "archive", tag], stdout=subprocess.PIPE
        )
        subprocess.run(["tar", "-x", "-C", str(target)], stdin=archive.stdout, check=True)
        if archive.wait():
            raise RuntimeError(f"git archive {tag} failed")
        files = tree_files(target)
        save_json(
            out / "export" / f"{checkpoint}.manifest.json",
            {
                "arm": self.arm,
                "checkpoint": checkpoint,
                "commit": commit,
                "content_sha256": content_hash(files),
                "files": len(files),
                "bytes": sum((target / path).stat().st_size for path in files),
                "build": self.manifest(),
            },
        )
        lock(target)
        return target


def lock(root: Path) -> None:
    for directory, _, files in os.walk(root, topdown=False):
        for name in files:
            os.chmod(Path(directory) / name, 0o444)
        os.chmod(directory, 0o555)


def unlock(root: Path) -> None:
    for directory, _, files in os.walk(root):
        os.chmod(directory, 0o755)
        for name in files:
            os.chmod(Path(directory) / name, 0o644)
