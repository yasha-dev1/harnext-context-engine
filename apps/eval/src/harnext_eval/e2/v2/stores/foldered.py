"""Arm F2's move-only tool surface: the builder writes a move plan, code executes it.

The paper's Foldered pass "runs view, grep, and rename alone". The Claude Code harness has
no rename tool, and its sandboxed Bash does not start on the build host (`bwrap: loopback:
Failed RTM_NEWADDR`), so the builder gets view tools confined to the tree and may write only the plan,
`_moves.json` ({"from": "to"}). The framework checks that nothing else changed, validates
the plan (every source exists, every target is a new `.md` path inside the tree, no
overwrite), executes it with `os.rename`, removes the plan, and `verify_moves` then proves
byte identity. F2 owns its prompts and digest so the digest other arms pin stays frozen.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path, PurePosixPath

from harnext_eval.e2.v2.stores import prompts

PLAN = "_moves.json"
PLANS = re.compile(r"_moves(-\d+)?\.json")  # a long plan may continue in _moves-2.json, …
# Claude Code permission rules under the harness's default-deny mode: view tools (Read, Glob,
# Grep) only inside the working directory, file writes only to the plan files. Verified live:
# a Write of `notes.md`, `../x`, `/tmp/x` and a Read/Grep/Glob of `/etc` are all refused.
ALLOWED_TOOLS = ["Read(./**)", "Edit(./_moves*.json)"]
DENIED_TOOLS = ["Bash", "WebFetch", "WebSearch", "Task", "NotebookEdit", "TodoWrite"]
FOLDER_LINES = 250

ROLE = """You are a Memory Filesystem Organizer. The store under the current working directory holds one verbatim markdown file per project event of Apache Kafka (JIRA, GitHub, mailing list). New event files arrive under `events/YYYY/MM/DD/`.
## Your Role
Design and maintain a folder taxonomy and move the event files into it intact. You may ONLY move/rename files; you look at the store with Read, Glob and Grep. Zero byte edits: you must not create, edit, truncate, copy, concatenate or delete any store file; every file must remain byte-identical, exactly once, somewhere in the tree. File and folder names are the only signal you add, so give a moved file a name that tells a reader what it is (keep the `.md` extension and keep the event's identifier in the name). Move every new file out of `events/`; you may also re-home earlier files when the taxonomy is better served.
## How you move files
You do not move files yourself. As your last action, use the Write tool to write the move plan with `file_path` exactly `_moves.json` (relative to the working directory, no leading slash): one JSON object mapping each current path to its new path, both relative to the store root, e.g. `{"events/2011/07/19/jira-12514638-created-1ffa8312.md": "jira/KAFKA-1/2011-07-19-created-jira-12514638-created-1ffa8312.md"}`. A new path that ends in `/` names a folder and keeps the file's current name: `{"events/2011/07/19/jira-12514638-created-1ffa8312.md": "jira/KAFKA-1/"}`. A plan of more than about 100 moves should be continued in `_moves-2.json`, `_moves-3.json`, … (same format, each source listed once overall). The pipeline then performs the moves, creating folders as needed, and deletes the plan files. The plan is rejected as a whole, and your work discarded, if a source does not exist, a target already exists or is named twice, a target is not a `.md` path inside the store, or any file other than the plan was written or changed. To rename a folder, list every file in it. There is no shell.
## Paths
Every path you give a tool or write in the plan is relative to the working directory (`github/pr-1/x.md`), never starting with `/` or `..`: a tool call that addresses a path outside the store discards your work."""

INSTRUCTION = """Window {index}: {count} new verbatim event files were added under `events/`. Decide where each belongs in the folder taxonomy and write the move plan `_moves.json` covering all of them (move only, zero byte edits).

Folders of the store now (files beneath each):

{folders}

New files, in time order:

{listing}"""


def system() -> str:
    return "\n".join([ROLE, prompts.READER, prompts.TAXONOMY, prompts.NAMING, prompts.OUTPUT])


def digest() -> str:
    payload = {"model": prompts.MODEL, "system": system(), "instruction": INSTRUCTION,
               "plan": PLANS.pattern, "tools": [ALLOWED_TOOLS, DENIED_TOOLS]}  # fmt: skip
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def folders(paths) -> str:
    """The taxonomy so far, as folder → number of files beneath it (`events/` left out).
    Deep levels are dropped until the listing fits, so a large store stays promptable."""
    homed = [PurePosixPath(p).parent.parts for p in paths if not p.startswith("events/")]
    for depth in range(max((len(p) for p in homed), default=0), 0, -1):
        counts = Counter(parts[:n] for parts in homed for n in range(1, min(len(parts), depth) + 1))
        if len(counts) <= FOLDER_LINES or depth == 1:
            note = "" if depth == max(len(p) for p in homed) else "\n(deeper folders not shown: Glob them)"  # fmt: skip
            return "\n".join(f"- `{'/'.join(k)}/` — {v}" for k, v in sorted(counts.items())) + note
    return "(none yet: this window founds the taxonomy)"


def _bad(path) -> str | None:
    if not isinstance(path, str) or not path:
        return "not a path string"
    parts = PurePosixPath(path).parts
    if path.startswith(("/", "~")) or "\\" in path or "\0" in path or ".." in parts:
        return "outside the store"
    if parts[0] in (".git", "_event", "_unprocessed") or PLANS.fullmatch(path):
        return "reserved location"
    return None


def execute(tree: Path, pre: dict[str, str]) -> str | None:
    """Validate and run the builder's move plan; returns the problem, or None when every
    move was done. Nothing is moved unless the whole plan is valid."""
    names, plan = sorted(n for n in os.listdir(tree) if PLANS.fullmatch(n)), {}
    for name in names:
        try:
            part = json.loads((tree / name).read_text())
        except (OSError, ValueError) as error:
            return f"move plan unreadable: {name}: {error}"[:300]
        if not isinstance(part, dict):
            return "move plan is not a JSON object {from: to}"
        if twice := sorted(set(part) & set(plan)):
            return f"move rejected (source named twice): {twice[:3]}"[:300]
        plan.update(part)
    plan = {
        a: b + PurePosixPath(a).name
        if isinstance(a, str) and isinstance(b, str) and b.endswith("/")
        else b
        for a, b in plan.items()
    }  # "folder/" keeps the file name
    plan = {a: b for a, b in plan.items() if a != b}
    # A mistyped source (long hashed event names) moves nothing: that file stays where it
    # is, still in the store, and the rest of the plan applies.
    plan = {a: b for a, b in plan.items() if not isinstance(a, str) or a in pre}
    if not plan:
        return "no change to the store"
    for source, target in plan.items():
        if reason := _bad(source) or _bad(target):
            return f"move rejected ({reason}): {source!r} -> {target!r}"[:300]
        if os.path.normpath(source) != source or os.path.normpath(target) != target:
            return f"move rejected (path not normalized): {source!r} -> {target!r}"[:300]
        if not target.endswith(".md"):
            return f"move rejected (target is not .md): {target!r}"[:300]
    targets = list(plan.values())
    if len(set(targets)) != len(targets):
        return f"move rejected (target named twice): {[t for t, n in Counter(targets).items() if n > 1][:3]}"
    final = (set(pre) - set(plan)) | set(targets)
    for target in targets:
        if target in pre:  # even if that file moves away too: no overwrite, no chains
            return f"move rejected (target exists): {target!r}"[:300]
        if (tree / target).exists() or any(str(p) in final for p in PurePosixPath(target).parents):
            return f"move rejected (target collides with a file or folder): {target!r}"[:300]
    root = tree.resolve()
    for source, target in plan.items():
        destination = tree / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (
            root not in destination.parent.resolve().parents
            and destination.parent.resolve() != root
        ):
            return f"move rejected (outside the store): {target!r}"[:300]
        os.rename(tree / source, destination)
    for name in names:
        (tree / name).unlink()
    for directory, _, _ in os.walk(tree / "events", topdown=False):
        if not os.listdir(directory):
            os.rmdir(directory)
    return None
