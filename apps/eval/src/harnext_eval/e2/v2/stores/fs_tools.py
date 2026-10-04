"""The two tools of the paper's seven that Claude Code lacks: `delete` and `rename`.

Zhou et al.'s management agent has view, create, str_replace, insert, delete, rename, grep.
Claude Code covers five (Read/Glob, Write, Edit, Grep); delete and rename were `rm`/`mv` in
sandboxed Bash, which cannot start on the build host (`bwrap: loopback: Failed RTM_NEWADDR`,
every Bash call of every builder failed). They are now real tools, served in-process to the
Claude Code harness (SDK MCP server `fs`), executed immediately by this module: relative,
normalized paths inside the store only, never `.git/`, `_event/` or `_unprocessed/`, no
overwrite. With them the builders have no shell at all, and Read/Write/Edit are confined to
the working directory by permission rules.
"""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath

from harnext_builder.harness.base import HarnessRequest
from harnext_builder.harness.claude_code import ClaudeCodeHarness

DELETE, RENAME = "mcp__fs__delete", "mcp__fs__rename"
# Permission rules under default-deny: Read(./**) covers Read, Glob and Grep; Edit(./**)
# covers Write and Edit; both only inside the working directory (verified live).
ALLOWED_TOOLS = ["Read(./**)", "Edit(./**)", DELETE, RENAME]
DENIED_TOOLS = ["Bash", "WebFetch", "WebSearch", "Task", "Skill", "NotebookEdit", "TodoWrite"]
RESERVED = (".git", "_event", "_unprocessed")
# Where Claude Code spills an oversized tool output and tells the model to Read it back.
TOOL_RESULTS = re.compile(
    re.escape(str(Path.home())) + r"/\.claude/projects/[\w.-]+/[\w-]+/tool-results/[\w.-]+\Z"
)


def bad(path) -> str | None:
    if not isinstance(path, str) or not path:
        return "not a path string"
    parts = PurePosixPath(path).parts
    if path.startswith(("/", "~")) or "\\" in path or "\0" in path or ".." in parts:
        return "outside the store: give a path relative to the store root"
    if os.path.normpath(path) != path:
        return "path not normalized"
    if parts[0] in RESERVED:
        return f"`{parts[0]}/` is reserved"
    return None


def inside(tree: Path, path: Path) -> bool:
    root, real = tree.resolve(), path.resolve()
    return real == root or root in real.parents


def prune(tree: Path, folder: Path) -> None:
    while folder != tree and folder.is_dir() and not os.listdir(folder):
        folder.rmdir()
        folder = folder.parent


def delete(tree: Path, path) -> str:
    """Delete one file; raises ValueError with the reason when refused."""
    if reason := bad(path):
        raise ValueError(f"delete refused ({reason}): {path!r}")
    target = tree / path
    if target.is_symlink() or not target.is_file() or not inside(tree, target):
        raise ValueError(f"delete refused (no such file; folders vanish when empty): {path!r}")
    target.unlink()
    prune(tree, target.parent)
    return f"deleted {path}"


def rename(tree: Path, old, new) -> str:
    """Move/rename one file or folder, creating parents; never overwrites."""
    if reason := bad(old) or bad(new):
        raise ValueError(f"rename refused ({reason}): {old!r} -> {new!r}")
    source, target = tree / old, tree / new
    if source.is_symlink() or not source.exists() or not inside(tree, source):
        raise ValueError(f"rename refused (no such file or folder): {old!r}")
    if target.exists() or target.is_symlink():
        raise ValueError(f"rename refused (target exists): {new!r}")
    if new.startswith(old + "/"):
        raise ValueError(f"rename refused (folder into itself): {old!r} -> {new!r}")
    for parent in target.relative_to(tree).parents:
        if (tree / parent).is_file():
            raise ValueError(f"rename refused (`{parent}` is a file): {new!r}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not inside(tree, target.parent):
        raise ValueError(f"rename refused (outside the store): {new!r}")
    os.rename(source, target)
    prune(tree, source.parent)
    return f"renamed {old} -> {new}"


def server(tree: Path):
    from claude_agent_sdk import create_sdk_mcp_server, tool

    def answer(call, *args) -> dict:
        try:
            return {"content": [{"type": "text", "text": call(tree, *args)}]}
        except (ValueError, OSError) as error:
            return {"content": [{"type": "text", "text": str(error)}], "is_error": True}

    @tool("delete", "Delete one file of the store. `path` is relative to the store root. "
          "A folder disappears when its last file is gone.", {"path": str})  # fmt: skip
    async def _delete(args):
        return answer(delete, args.get("path"))

    @tool("rename", "Move or rename one file or folder of the store. Both paths are relative "
          "to the store root; missing parent folders are created; an existing target is never "
          "overwritten.", {"old_path": str, "new_path": str})  # fmt: skip
    async def _rename(args):
        return answer(rename, args.get("old_path"), args.get("new_path"))

    return create_sdk_mcp_server("fs", tools=[_delete, _rename])


def path_hint(tree: Path):
    """PreToolUse hook: a file tool addressed outside the store is denied with a reason the
    model can act on. (The permission rules deny it anyway, but with a generic message after
    which Haiku, which likes to write the store root as `/`, tends to give up.)"""
    root = str(tree.resolve())

    async def hook(data, tool_use_id, context):
        values = [
            str(v) for k, v in (data.get("tool_input") or {}).items() if k in ("file_path", "path")
        ]
        if data.get("tool_name") == "Read" and all(TOOL_RESULTS.match(v) for v in values):
            return {}  # Claude Code tells the model to Read its spilled oversized tool output
        wrong = [v for v in values if (v.startswith(("/", "~")) and v != root and not v.startswith(root + "/")) or ".." in PurePosixPath(v).parts]  # fmt: skip
        if not wrong:
            return {}
        hint = wrong[0].lstrip("/~") or "."
        reason = (f"Permission to use {data.get('tool_name')} has been denied: `{wrong[0]}` is outside the store. "
                  f"Paths are relative to the working directory, which is the store root: use `{hint}`.")  # fmt: skip
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                       "permissionDecision": "deny",
                                       "permissionDecisionReason": reason}}  # fmt: skip

    return hook


class StoreHarness(ClaudeCodeHarness):
    """The builder's Claude Code harness plus the in-process `delete`/`rename` tools."""

    def options(self, req: HarnessRequest):
        options = super().options(req)
        options.mcp_servers = {"fs": server(Path(req.working_dir))}
        # Transport only: one oversized tool result (a big Read/Grep) overran the SDK's 1 MiB
        # message buffer and stalled a whole window as an "infrastructure error".
        options.max_buffer_size = 32 * 1024 * 1024
        from claude_agent_sdk import HookMatcher

        options.hooks = {"PreToolUse": [HookMatcher(matcher="Read|Write|Edit|Glob|Grep",
                                                    hooks=[path_hint(Path(req.working_dir))])]}  # fmt: skip
        return options
