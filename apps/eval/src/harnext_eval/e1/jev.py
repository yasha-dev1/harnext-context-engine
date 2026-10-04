"""Jev routing arms for E1 (R14, R15): one software-engineering triage decision per event.

Jev (`typesafe/jev-1.13`, OpenRouter decisions endpoint) answers typed questions with
probabilities only. Each event gets ONE call with two questions:

- ``category`` (choice): which kind of engineering work the event belongs to, over a fixed
  software-engineering taxonomy. Four categories are fast-lane by definition
  (``FAST_CATEGORIES``); the mapping is fixed before any scoring.
- ``urgency`` (noul): probability that the event needs a maintainer within hours (fast lane)
  rather than in the next daily digest (batch lane).

R14 (category router) ranks by the probability mass on the fast-lane categories, ties broken
by ``urgency``; R15 (direct urgency) ranks by ``urgency`` alone. Both share the R0-R6 monthly
capacity like R2. Scores are computed once, before the E1 run, by ``scripts/e1_jev.py score``
into an append-only JSONL cache keyed by event id and ``prompt_sha256()``; the run only reads
that cache.

What Jev sees (``event_state``) is the gold-stripped, policy-visible event with three
guards added in review (2026-10-04):

- JIRA summary, description and labels are rebuilt AS OF the event time from the changelog
  (``AsOfText``); the replay carries export-time snapshots of them.
- Issue keys, pull-request numbers, commit hashes and links are removed, so the model cannot
  look up a known issue by its identifier (the model build postdates the window).
- Author roles are only ``person`` / ``automation``; the committer roster is a current snapshot.

Text that the sources never versioned (comment bodies, PR titles and bodies) stays as exported;
that limitation is shared with the rule floor and is stated in the registration.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import sys
import time
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harnext_eval.types import EvalEvent

PROMPT_VERSION = "se-triage-v2"
JEV_MODEL = "typesafe/jev-1.13"

CONTEXT = (
    "You triage the event stream of a large open-source software project: issue-tracker events, "
    "pull-request activity and the developer mailing list. Judge only from what the event in "
    "`event` says at the moment it happens."
)

CATEGORY_INSTRUCTIONS = (
    CONTEXT + " Which kind of engineering work does this event belong to? Use the work the issue, "
    "pull request or thread is about; choose code_review or bookkeeping only when the event itself "
    "adds nothing beyond routine review or an administrative or automated update."
)

CATEGORIES: dict[str, str] = {
    "regression_or_outage": (
        "A defect that breaks existing behaviour of a released or development version: crash, hang, "
        "data loss or corruption, wrong results, failed upgrade, broken compatibility, severe "
        "performance regression."
    ),
    "security": (
        "A security vulnerability, a credential or permission exposure, or a fix or disclosure for one."
    ),
    "build_break": (
        "The main build, continuous integration or test infrastructure is broken: compilation "
        "failures, tests failing for everyone, changes that stop others from merging."
    ),
    "release_risk": (
        "Something that endangers a release: problems with a release candidate, issues that must be "
        "fixed before a version ships, release steps that need action now, fixes that released "
        "versions need quickly."
    ),
    "minor_bug": (
        "A defect with limited impact or a workaround: edge cases, misleading logs or error "
        "messages, cosmetic problems, test-only bugs, occasionally failing tests."
    ),
    "feature_work": (
        "New functionality, an enhancement proposal or its implementation, refactoring or a "
        "performance improvement that does not fix a regression."
    ),
    "design_discussion": (
        "A proposal, design question or decision under discussion that is not time-critical."
    ),
    "code_review": (
        "Routine review activity: review comments, approvals, requested changes, rebases, requests "
        "for review."
    ),
    "docs_or_chore": (
        "Documentation, typos, cleanup, dependency updates, build-tooling upkeep, small test additions."
    ),
    "user_question": "A usage question, support request or configuration help.",
    "bookkeeping": (
        "Administrative or automated updates without new technical content: assignment, links, "
        "status or field changes, bot notifications, mirrors of activity happening elsewhere."
    ),
}

FAST_CATEGORIES = ("regression_or_outage", "security", "build_break", "release_risk")

URGENCY_INSTRUCTIONS = (
    CONTEXT + " The project routes every event either to the FAST lane (an on-duty maintainer "
    "looks at it within the next few hours) or to the BATCH lane (it waits for the next daily "
    "digest). How likely is it that this event belongs in the fast lane? Likely for new "
    "breakages, regressions, security problems, threats to a release and messages that ask the "
    "maintainers for a time-critical response; unlikely for routine review, feature work, "
    "discussion, chores and automated bookkeeping."
)


def questions() -> dict[str, dict[str, Any]]:
    return {
        "category": {"type": "choice", "instructions": CATEGORY_INSTRUCTIONS, "criteria": dict(CATEGORIES)},
        "urgency": {"type": "noul", "instructions": URGENCY_INSTRUCTIONS},
    }


# ---- as-of JIRA text -------------------------------------------------------------------------


class AsOfText:
    """JIRA summary / description / labels as they were at a given time, from the changelog.

    Every changelog event in the replay carries the old (``from``) and new (``to``) text. The
    value at ``t`` is the ``to`` of the last change at or before ``t``; before the first change it
    is that change's ``from`` (the creation value); an issue never edited keeps its snapshot.
    """

    FIELDS = ("summary", "description", "labels")

    def __init__(self) -> None:
        self._changes: dict[tuple[str, str], list[tuple[datetime, int, str, str]]] = defaultdict(list)
        self._times: dict[tuple[str, str], list[datetime]] = {}

    @classmethod
    def from_events(cls, events: Iterable[EvalEvent]) -> AsOfText:
        timeline = cls()
        for order, event in enumerate(events):
            if event.type != "org.apache.jira.issue.transition":
                continue
            data = event.data or {}
            name = str(data.get("field") or "")
            if name in cls.FIELDS and data.get("issue_key"):
                timeline._changes[(str(data["issue_key"]), name)].append(
                    (event.time, order, "" if data.get("from") is None else str(data["from"]),
                     "" if data.get("to") is None else str(data["to"]))
                )
        for key, changes in timeline._changes.items():
            changes.sort(key=lambda item: (item[0], item[1]))
            timeline._times[key] = [item[0] for item in changes]
        return timeline

    def value(self, issue: Any, name: str, at: datetime, snapshot: Any) -> Any:
        changes = self._changes.get((str(issue), name))
        if not changes:
            return snapshot
        index = bisect_right(self._times[(str(issue), name)], at)
        text = changes[index - 1][3] if index else changes[0][2]
        if name == "labels":
            return [label for label in text.split() if label]
        return text


# ---- event view ----------------------------------------------------------------------------

_BODY_CHARS = 1500
_CONTEXT_CHARS = 500
_WS = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")
_CHECKLIST = re.compile(r"#+\s*Committer Checklist.*", re.IGNORECASE | re.DOTALL)
_JIRA_FOOTER = re.compile(r"\n--\s*\nThis message was sent by Atlassian Jira.*", re.DOTALL)
_ON_WROTE = re.compile(r"^On .{0,200}wrote:\s*$", re.MULTILINE)
_AUTOMATION = re.compile(r"\[bot\]|\bbot\b|jira@apache\.org|\(jira\)|noreply@github\.com|git@apache\.org")
# Identifiers that would let a model recall a specific issue or change rather than read it.
_IDENTIFIERS = (
    (re.compile(r"https?://\S+"), "<link>"),
    (re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b"), "<issue>"),
    (re.compile(r"(?<![\w&])#\d{2,7}\b"), "#<pr>"),
    (re.compile(r"\b[0-9a-f]{12,40}\b"), "<sha>"),
)


def _scrub(text: str) -> str:
    for pattern, replacement in _IDENTIFIERS:
        text = pattern.sub(replacement, text)
    return text


def _clean(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    text = _CHECKLIST.sub("", text)
    text = _scrub(text)
    text = _WS.sub(" ", text.replace(" ", " "))
    text = _BLANK_LINES.sub("\n", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _mail_body(value: Any) -> str:
    text = _JIRA_FOOTER.sub("", "" if value is None else str(value))
    match = _ON_WROTE.search(text)
    if match:  # everything below "On … wrote:" is the quoted earlier message
        text = text[: match.start()]
    lines = [line for line in text.splitlines() if not line.lstrip().startswith(">")]
    return _clean("\n".join(lines), _BODY_CHARS)


def _role(data: dict[str, Any], *names: str) -> str:
    who = " ".join(str(data.get(name) or "") for name in names).casefold()
    return "automation" if _AUTOMATION.search(who) else "person"


_GITHUB_KINDS = {
    "com.github.pull_request.opened": "pull request opened",
    "com.github.pull_request.merged": "pull request merged",
    "com.github.pull_request.closed": "pull request closed without merge",
    "com.github.review": "pull request review submitted",
    "com.github.review_comment": "review comment on a code line",
    "com.github.issue_comment": "pull request conversation comment",
    "com.github.push": "commits pushed",
}


def event_state(event: EvalEvent, as_of: AsOfText | None = None) -> dict[str, Any]:
    """The event as Jev sees it: what happened, on which kind of work item, and its text, truncated.

    Only fields the event itself carries (gold already stripped by the caller), JIRA text as of
    the event time when ``as_of`` is given, and no identifiers, hashes, timestamps or links.
    """

    data = event.data or {}
    kind = event.type
    state: dict[str, Any]
    if kind.startswith("org.apache.jira."):
        issue = data.get("issue_key")

        def text(name: str) -> Any:
            return as_of.value(issue, name, event.time, data.get(name)) if as_of is not None else data.get(name)

        item = {"summary": _clean(text("summary"), 300), "components": list(data.get("components") or [])[:6]}
        if kind.endswith(".created"):
            state = {
                "source": "issue tracker",
                "event": "issue created",
                **item,
                "priority": data.get("priority"),
                "labels": list(text("labels") or [])[:8],
                "reporter": _role(data, "creator_name"),
                "description": _clean(text("description"), _BODY_CHARS),
            }
        elif kind.endswith(".comment"):
            state = {
                "source": "issue tracker",
                "event": "comment on issue",
                **item,
                "author": _role(data, "author_name"),
                "comment": _clean(data.get("body"), _BODY_CHARS),
                "issue_description": _clean(text("description"), _CONTEXT_CHARS),
            }
        else:
            state = {
                "source": "issue tracker",
                "event": "issue field changed",
                **item,
                "change": {
                    "field": data.get("field"),
                    "from": _clean(data.get("from"), 200),
                    "to": _clean(data.get("to"), 200),
                },
                "author": _role(data, "actor_name", "actor_account_id"),
                "issue_description": _clean(text("description"), _CONTEXT_CHARS),
            }
    elif kind.startswith("com.github."):
        state = {
            "source": "pull requests",
            "event": _GITHUB_KINDS.get(kind, kind.removeprefix("com.github.").replace("_", " ")),
            "title": _clean(data.get("title"), 300),
            "author": _role(data, "author_login", "actor_login"),
        }
        if kind == "com.github.push":
            state["branch"] = data.get("branch")
            state["commit_messages"] = [
                _clean(commit.get("message"), 300) for commit in list(data.get("commits") or [])[:5]
            ] or [_clean(data.get("message"), 300)]
        else:
            if kind == "com.github.review":
                state["review_state"] = data.get("state")
            if kind == "com.github.review_comment":
                state["file"] = data.get("path")
            if data.get("labels"):
                state["labels"] = list(data.get("labels") or [])[:8]
            files = list(data.get("changed_files") or [])
            if files:
                state["changed_files"] = files[:8] + ([f"… {len(files) - 8} more"] if len(files) > 8 else [])
            limit = _CONTEXT_CHARS if kind in {"com.github.pull_request.merged", "com.github.pull_request.closed"} else _BODY_CHARS
            state["text"] = _clean(data.get("body"), limit)
    elif kind.startswith("org.apache.mail."):
        state = {
            "source": "developer mailing list",
            "event": "reply in thread" if data.get("in_reply_to") else "new thread",
            "subject": _clean(data.get("subject"), 300),
            "author": _role(data, "author_name", "author_email"),
            "body": _mail_body(data.get("body")),
        }
    else:
        state = {"source": "other", "event": kind, "data": _clean(json.dumps(data, default=str), _BODY_CHARS)}
    return {"event": {key: value for key, value in state.items() if value not in (None, "", [], {})}}


def prompt_sha256() -> str:
    """Identity of everything that shapes the decision except the event: this module's source
    (prompt, taxonomy, view, scrubbing, as-of rebuild, scoring) and the model id."""

    payload = {"version": PROMPT_VERSION, "model": JEV_MODEL, "source": inspect.getsource(sys.modules[__name__])}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ---- scores ----------------------------------------------------------------------------------


@dataclass
class JevDecision:
    event_id: str
    prompt_sha256: str
    category: str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    urgency: float | None = None
    model: str | None = None
    input_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    attempts: int = 0
    error: str | None = None
    decided_at: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.category is not None and self.urgency is not None

    @property
    def fast_category_p(self) -> float:
        return float(sum(self.probabilities.get(name, 0.0) for name in FAST_CATEGORIES))

    def score(self, policy: str) -> float:
        """R14: fast-lane category mass, ties broken by urgency; R15: urgency. Failed calls rank last."""

        if not self.ok:
            return -1.0
        if policy == "R14":
            return self.fast_category_p + 1e-3 * float(self.urgency or 0.0)
        if policy == "R15":
            return float(self.urgency or 0.0)
        raise ValueError(f"not a Jev policy: {policy}")


@dataclass
class CacheSummary:
    records: int = 0
    bad_lines: int = 0
    cost_usd: float = 0.0
    """Cost of every record under the prompt, including failed and superseded attempts."""


def load_decisions(
    path: str | Path, prompt_sha: str | None = None, summary: CacheSummary | None = None
) -> dict[str, JevDecision]:
    """Latest successful decision per event id (else the latest failure); other prompts are ignored.

    A truncated or corrupt line (an interrupted write) is skipped and counted in ``summary``.
    """

    result: dict[str, JevDecision] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = JevDecision(**json.loads(line))
            except (ValueError, TypeError):
                if summary is not None:
                    summary.bad_lines += 1
                continue
            if prompt_sha is not None and record.prompt_sha256 != prompt_sha:
                continue
            if summary is not None:
                summary.records += 1
                summary.cost_usd += record.cost_usd
            previous = result.get(record.event_id)
            if previous is None or record.ok or not previous.ok:
                result[record.event_id] = record
    return result


def _parse(event_id: str, sha: str, answers: dict[str, Any], call: Any) -> JevDecision:
    category = answers.get("category") or {}
    probabilities = {
        name: float(value)
        for name, value in (category.get("probabilities") or {}).items()
        if name in CATEGORIES and value is not None
    }
    choice = category.get("choice")
    if choice in CATEGORIES and not probabilities:
        probabilities = {choice: float(category.get("confidence") or 1.0)}
    total = sum(probabilities.values())
    if total > 0:
        probabilities = {name: value / total for name, value in probabilities.items()}
    urgency = (answers.get("urgency") or {}).get("noul")
    return JevDecision(
        event_id=event_id,
        prompt_sha256=sha,
        category=choice if choice in CATEGORIES else (max(probabilities, key=probabilities.__getitem__) if probabilities else None),
        probabilities=probabilities,
        urgency=None if urgency is None else float(urgency),
        model=call.model,
        input_tokens=int(call.input_tokens),
        cost_usd=float(call.cost_usd) + float(getattr(call, "unmetered_usd", 0.0) or 0.0),
        seconds=float(call.seconds),
        attempts=int(getattr(call, "attempts", 1) or 1),
        error=None if choice in CATEGORIES or probabilities else "no category in answer",
        decided_at=datetime.now(UTC).isoformat(),
    )


async def score_events(
    events: Sequence[EvalEvent],
    out: Path,
    *,
    client: Any,
    as_of: AsOfText | None = None,
    concurrency: int = 16,
    max_usd: float | None = None,
    progress_every: int = 1000,
    max_consecutive_failures: int = 200,
) -> dict[str, Any]:
    """Decide every event not yet decided under the current prompt; append to ``out`` (JSONL).

    Resumable: successful decisions already in ``out`` are skipped, failures are retried.
    ``max_usd`` caps the total cost recorded in the cache under this prompt (across sessions).
    Stops early on refused credentials/credit (raises), on ``max_consecutive_failures`` failures
    in a row (provider outage), or at ``max_usd``; ``stopped_early`` reports it.
    """

    from harnext_eval.e2.v2.engine.clients import InfraError

    sha = prompt_sha256()
    before = CacheSummary()
    # One synchronous read of the cache before any request is made.
    exists = out.exists()  # noqa: ASYNC240
    done = {key for key, value in load_decisions(out, sha, before).items() if value.ok} if exists else set()
    todo = [event for event in events if event.id not in done]
    spec = questions()
    stats: dict[str, Any] = {"todo": len(todo), "skipped": len(done), "ok": 0, "failed": 0, "cost_usd": 0.0,
                             "input_tokens": 0, "cost_usd_cache_before": round(before.cost_usd, 6),
                             "bad_lines_before": before.bad_lines}
    out.parent.mkdir(parents=True, exist_ok=True)
    queue: asyncio.Queue[EvalEvent] = asyncio.Queue()
    for event in todo:
        queue.put_nowait(event)
    stop = asyncio.Event()
    started = time.monotonic()
    fatal: list[BaseException] = []
    consecutive = 0

    with out.open("a", encoding="utf-8") as sink:

        def record(decision: JevDecision) -> None:
            nonlocal consecutive
            sink.write(json.dumps(asdict(decision), sort_keys=True) + "\n")
            stats["ok" if decision.ok else "failed"] += 1
            stats["cost_usd"] += decision.cost_usd
            stats["input_tokens"] += decision.input_tokens
            consecutive = 0 if decision.ok else consecutive + 1
            if consecutive >= max_consecutive_failures:
                stop.set()
            if max_usd is not None and before.cost_usd + stats["cost_usd"] >= max_usd:
                stop.set()
            finished = stats["ok"] + stats["failed"]
            if progress_every and finished % progress_every == 0:
                sink.flush()
                rate = finished / max(time.monotonic() - started, 1e-6)
                print(
                    f"jev {finished}/{len(todo)} ok={stats['ok']} failed={stats['failed']} "
                    f"${stats['cost_usd']:.4f} {rate:.1f}/s eta {(len(todo) - finished) / max(rate, 1e-6) / 60:.0f} min",
                    flush=True,
                )

        async def worker() -> None:
            while not stop.is_set():
                try:
                    event = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    answers, call = await client.decide(event_state(event, as_of), spec, "e1-route")
                    decision = _parse(event.id, sha, answers, call)
                except InfraError as error:
                    # Exhausted retries end in "(after N attempts)"; anything else is a refusal
                    # of the key or credit (401/402/403), which no retry fixes.
                    if "attempts)" not in str(error):
                        fatal.append(error)
                        stop.set()
                    call = getattr(error, "call", None)
                    decision = JevDecision(event.id, sha, error=f"{type(error).__name__}: {error}"[:500],
                                           cost_usd=float(getattr(call, "unmetered_usd", 0.0) or 0.0) if call else 0.0,
                                           decided_at=datetime.now(UTC).isoformat())
                except Exception as error:  # noqa: BLE001 - every failure is recorded and retried later
                    call = getattr(error, "call", None)
                    decision = JevDecision(event.id, sha, error=f"{type(error).__name__}: {error}"[:500],
                                           cost_usd=float(getattr(call, "cost_usd", 0.0) or 0.0) if call else 0.0,
                                           decided_at=datetime.now(UTC).isoformat())
                record(decision)

        await asyncio.gather(*(worker() for _ in range(max(1, concurrency))))
        sink.flush()
    stats["seconds"] = round(time.monotonic() - started, 1)
    stats["stopped_early"] = stop.is_set() and not queue.empty()
    stats["remaining"] = queue.qsize()
    stats["cost_usd_cache_total"] = round(before.cost_usd + stats["cost_usd"], 6)
    if fatal:
        raise fatal[0]
    return stats


def decision_frame(decisions: Iterable[JevDecision]):
    import pandas as pd

    rows = []
    for decision in decisions:
        rows.append(
            {
                "event_id": decision.event_id,
                "ok": decision.ok,
                "category": decision.category,
                "fast_category_p": decision.fast_category_p if decision.ok else float("nan"),
                "urgency": decision.urgency,
                "R14": decision.score("R14"),
                "R15": decision.score("R15"),
                "input_tokens": decision.input_tokens,
                "cost_usd": decision.cost_usd,
                "seconds": decision.seconds,
            }
        )
    return pd.DataFrame(rows)
