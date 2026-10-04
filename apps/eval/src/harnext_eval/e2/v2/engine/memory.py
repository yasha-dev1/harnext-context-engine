"""Tools of the memory branch over a markdown notes tree (a file-arm checkout): keyword search
(Jev picks the hits worth reading) and paged file reads. Paths are confined to the memory root:
every path is resolved (symlinks included) and must stay inside it; nothing else is read."""

from __future__ import annotations

import re
from pathlib import Path

from harnext_eval.e2.v2.engine import prompts
from harnext_eval.e2.v2.engine.rawtools import as_int, function
from harnext_eval.e2.v2.engine.session import BudgetExceeded, Session

SEARCH = function(
    "search",
    "Keyword search over every notes file. `need`: one sentence saying what you look for. "
    "`keywords`: words, identifiers or short phrases (case-insensitive; a line matches any).",
    {"need": {"type": "string"}, "keywords": {"type": "array", "items": {"type": "string"}}},
    ["need", "keywords"],
)
READ_FILE = function(
    "read_file",
    "Read a notes file (path as returned by search) from a line on.",
    {"path": {"type": "string"}, "from_line": {"type": "integer", "minimum": 1}},
    ["path"],
)
MAX_FILE = 2_000_000


def load(root: Path) -> dict[str, list[str]]:
    """{relative path: lines} of every markdown file whose real path is inside `root`."""
    root = Path(root).resolve()
    found = {}
    for path in sorted(root.rglob("*.md")):
        real = path.resolve()
        if real.is_relative_to(root) and real.is_file() and real.stat().st_size <= MAX_FILE:
            found[path.relative_to(root).as_posix()] = real.read_text(errors="replace").splitlines()  # fmt: skip
    return found


class MemoryTools:
    def __init__(self, session: Session, files: dict[str, list[str]]):
        self.session, self.files, self.config = session, files, session.config
        self.seen: set[str] = set()
        self.log: list[dict] = []
        self.records: dict[str, str] = {}  # path -> the lines retrieved from it

    def specs(self) -> list[dict]:
        return [SEARCH, READ_FILE]

    def handlers(self) -> dict:
        return {"search": self.search, "read_file": self.read_file}

    async def search(self, need: str = "", keywords=None) -> str:
        record = {"tool": "search", "need": need, "keywords": keywords}
        self.log.append(record)
        words = [str(w).strip() for w in (keywords if isinstance(keywords, list) else [keywords] if keywords else []) if str(w).strip()][:10]  # fmt: skip
        if not words:
            record["error"] = "no keywords"
            return "error: pass at least one keyword"
        pattern = re.compile("|".join(re.escape(str(w)) for w in words), re.IGNORECASE)
        hits, per_file = [], {}
        for path, text in self.files.items():
            matched = [(n, line) for n, line in enumerate(text, 1) if pattern.search(line)]
            if matched:
                distinct = {m.group(0).lower() for _, line in matched for m in pattern.finditer(line)}  # fmt: skip
                per_file[path] = len(distinct)
                hits += [(path, n, line) for n, line in matched[:3]]
        hits.sort(key=lambda h: (-per_file[h[0]], h[0], h[1]))
        hits = hits[: self.config.memory_hits]
        record["hits"] = len(hits)
        if not hits:
            return "No notes line matches these keywords; try other words or identifiers."
        keep = list(range(len(hits)))
        if len(hits) > self.config.memory_show:
            state = {"need": str(need), "hits": {f"h{i}": f"{p}:{n}: {line.strip()[:240]}" for i, (p, n, line) in enumerate(hits)}}  # fmt: skip
            questions = {f"h{i}": {"type": "noul", "instructions": prompts.HIT.format(name=f"h{i}")} for i in range(len(hits))}  # fmt: skip
            try:
                answers = await self.session.decide(state, questions, "memory-hits")
                scores = {i: float(answers[f"h{i}"].get("noul", 0)) for i in range(len(hits))}
                keep = sorted(sorted(scores, key=lambda i: -scores[i])[: self.config.memory_show])
            except BudgetExceeded:
                keep = keep[: self.config.memory_show]
        shown = [hits[i] for i in keep]
        self.seen |= {p for p, _, _ in shown}
        for p, n, line in shown:
            self.records[p] = self.records.get(p, "") + f"\n{n}: {line.strip()}"
        return f"{len(hits)} matching lines; {len(shown)} shown:\n" + "\n".join(f"- {p}:{n}: {line.strip()[:300]}" for p, n, line in shown)  # fmt: skip

    def resolve(self, path: str) -> str | None:
        key = str(path).strip().lstrip("./")
        return key if key in self.files else None

    async def read_file(self, path: str = "", from_line: int = 1) -> str:
        record = {"tool": "read_file", "path": path, "from_line": from_line}
        self.log.append(record)
        key = self.resolve(path)
        if key is None:
            record["error"] = "unknown path"
            return f"error: no notes file {path!r} (use a path exactly as returned by search)"
        self.seen.add(key)
        text = self.files[key]
        start = min(max(1, as_int(from_line)), max(1, len(text)))
        part = text[start - 1 : start - 1 + self.config.memory_lines]
        end = start + len(part) - 1
        more = f" (more from line {end + 1})" if end < len(text) else ""
        self.records[key] = self.records.get(key, "") + "\n" + "\n".join(part)
        return f"{key} lines {start}-{end} of {len(text)}{more}:\n" + "\n".join(f"{start + i}: {line}" for i, line in enumerate(part))  # fmt: skip
