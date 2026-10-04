"""context_research — an internal read agent over the latest snapshot.

Materializes the org's latest (immutable) snapshot to a temp dir and runs a
read-only harness over it, returning a synthesized, cited answer. Reading a
snapshot (not the live FS) gives a consistent, never-torn view while the builder
keeps writing.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

from harnext_builder.agentfs.store import OrgFsStore
from harnext_builder.harness.base import ConversationTranscript, HarnessRequest
from harnext_builder.harness.registry import get_harness
from harnext_builder.settings import BuilderSettings

_READ_TOOLS = ["Read", "Glob", "Grep", "LS"]
_NO_TOOLS = ["Write", "Edit", "MultiEdit", "Bash", "WebFetch", "WebSearch", "Task"]

_RESEARCH_SYSTEM = """\
You are a research assistant over an organization's context filesystem, mounted
read-only at your working directory. Answer the question using ONLY these files.
Start from INDEX.md to navigate, then read the relevant entity files. Be concise
and cite the file paths you used. If the answer is not in the context, say so
plainly — do not speculate."""

_RESEARCH_TMPL = """\
Question: {question}

Search the context filesystem and answer it, citing the files (by path) you drew
the answer from."""


def _extract_answer(t: ConversationTranscript) -> str:
    for turn in reversed(t.turns):
        if turn.role in ("result", "assistant") and turn.content.strip():
            return turn.content
    return t.error or "(the research agent produced no answer)"


def _files_read(t: ConversationTranscript, directory: Path) -> list[str]:
    """Paths the agent read (Read tool_use turns), relative to `directory`, in first-read order."""
    root, seen = str(Path(directory).resolve()), []
    for turn in t.turns:
        if turn.role != "tool_use" or turn.tool_name != "Read":
            continue
        try:
            path = str((json.loads(turn.content) or {}).get("file_path") or "")
        except ValueError:
            continue
        if not path:
            continue
        rel = path[len(root) + 1 :] if path.startswith(root + "/") else path.lstrip("./") or path
        if rel not in seen:
            seen.append(rel)
    return seen


async def research_dir(
    directory: Path | str,
    settings: BuilderSettings,
    question: str,
    *,
    system_prompt: str = _RESEARCH_SYSTEM,
    harness=None,
) -> dict[str, Any]:
    """Run the read agent over a directory that already holds a context snapshot.

    Shared by the product (over a materialized AgentFS snapshot) and the E2 evaluation (over a
    frozen store checkpoint). Returns the answer, the files the agent read, usage, turn count,
    wall seconds and the transcript; never raises for a model failure (see `transcript.error`)."""
    req = HarnessRequest(
        harness=settings.harness,
        working_dir=str(directory),
        instruction=_RESEARCH_TMPL.format(question=question),
        system_prompt=system_prompt,
        allowed_tools=_READ_TOOLS,
        disallowed_tools=_NO_TOOLS,
        model=settings.builder_model,
        max_turns=settings.builder_max_turns,
        timeout_s=settings.builder_timeout_s,
    )
    started = time.monotonic()
    transcript = await (harness or get_harness(settings.harness)).run(req)
    return {
        "answer": _extract_answer(transcript),
        "files": _files_read(transcript, Path(directory)),
        "usage": transcript.usage,
        "turns": sum(t.role == "assistant" for t in transcript.turns),
        "seconds": round(time.monotonic() - started, 2),
        "transcript": transcript,
    }


async def research(
    store: OrgFsStore, settings: BuilderSettings, org_id: str, question: str
) -> dict[str, Any]:
    snap = await store.latest_snapshot(org_id)
    if snap is None:
        return {
            "answer": "No context has been built for this organization yet.",
            "snapshot_id": None,
            "files": [],
        }

    files = await store.list_files(org_id, snapshot_id=snap.id)
    with tempfile.TemporaryDirectory(prefix=f"research-{org_id}-") as tmp:
        for rel in files:
            content = await store.read_file(org_id, rel, snapshot_id=snap.id)
            if content is None:
                continue
            dst = Path(tmp) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(content)
        out = await research_dir(tmp, settings, question)

    return {
        "answer": out["answer"],
        "snapshot_id": snap.id,
        "files": files,
        "usage": out["usage"],
    }
