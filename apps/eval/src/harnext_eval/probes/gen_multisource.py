"""Regex-key cross-source joins for docs/evaluation-spec.md §4 and §7 E2."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from harnext_eval.probes.common import ISSUE_KEY_RE, uniform_time, unique, validate_period
from harnext_eval.probes.schema import ProbeCandidate, stratified_sample
from harnext_eval.types import EvalEvent, Probe


@dataclass
class JoinAuditTrail:
    """Compare regex edges with a frozen human/synthetic edge audit."""

    expected: Mapping[str, list[str]] = field(default_factory=dict)
    predicted: dict[str, list[str]] = field(default_factory=dict)

    def record(self, event_id: str, keys: list[str]) -> None:
        self.predicted[event_id] = sorted(set(keys))

    def report(self) -> dict[str, Any]:
        audited_ids = sorted(self.expected)
        predicted_edges = {
            (event_id, key.upper())
            for event_id in audited_ids
            for key in self.predicted.get(event_id, [])
        }
        expected_edges = {
            (event_id, key.upper())
            for event_id in audited_ids
            for key in self.expected[event_id]
        }
        overlap = predicted_edges & expected_edges
        precision = (
            len(overlap) / len(predicted_edges) if predicted_edges else float(not expected_edges)
        )
        recall = len(overlap) / len(expected_edges) if expected_edges else float(not predicted_edges)
        return {
            "status": "audited" if audited_ids else "supported-not-run",
            "audited_events": len(audited_ids),
            "predicted_edges": len(predicted_edges),
            "expected_edges": len(expected_edges),
            "matched_edges": len(overlap),
            "join_precision": precision if audited_ids else None,
            "join_recall": recall if audited_ids else None,
            "predictions": {key: self.predicted[key] for key in sorted(self.predicted)},
        }

    def require_valid(self, *, evidentiary: bool) -> None:
        if evidentiary and not self.expected:
            raise ValueError("evidentiary multi-source generation requires --join-audit")


def regex_join_keys(event: EvalEvent) -> list[str]:
    """Extract only keys in preregistered commit/thread/PR-title locations."""

    data = event.data or {}
    event_type = event.type.casefold()
    if "pull_request" in event_type:
        surface = f"{data.get('title', '')}\n{data.get('head_ref', '')}"
    elif "mail" in event_type:
        surface = str(data.get("subject", ""))
    elif "commit" in event_type or "push" in event_type:
        messages: list[str] = [str(data.get("commit_message", ""))]
        commits = data.get("commits", [])
        if isinstance(commits, list):
            messages.extend(
                str(item.get("message", ""))
                for item in commits
                if isinstance(item, Mapping)
            )
        surface = "\n".join(messages)
    else:
        return []
    return unique(key.upper() for key in ISSUE_KEY_RE.findall(surface))


def canonical_source_link(event: EvalEvent) -> str | None:
    data = event.data or {}
    event_type = event.type.casefold()
    if "pull_request" in event_type:
        number = data.get("number") or data.get("pr_number") or data.get("pull_request_number")
        return f"pr:{_repository(event)}#{number}" if number is not None else None
    if "mail" in event_type:
        root = data.get("thread_root") or data.get("thread_id") or data.get("thread_key")
        if not isinstance(root, str) or not root:
            return None
        return f"thread:{_namespace(event)}/{root.removeprefix('thread:')}"
    if "commit" in event_type or "push" in event_type:
        sha = data.get("sha") or data.get("commit_sha") or data.get("head") or event.id
        return f"commit:{_repository(event)}@{sha}"
    return None


def _repository(event: EvalEvent) -> str:
    data = event.data or {}
    raw = data.get("repository") or data.get("repo_name") or data.get("repo")
    if isinstance(raw, Mapping):
        raw = raw.get("full_name") or raw.get("name")
    if isinstance(raw, str) and raw:
        return raw.casefold()
    return event.source.split(":", 1)[-1].casefold()


def _namespace(event: EvalEvent) -> str:
    return event.source.split(":", 1)[-1].casefold()


def _links_as_of(
    events: list[EvalEvent],
    entity: str,
    at: datetime,
    *,
    audit: JoinAuditTrail | None = None,
) -> tuple[list[str], list[str]]:
    links: list[str] = []
    source_ids: list[str] = []
    for event in events:
        if event.time > at:
            break
        keys = regex_join_keys(event)
        if audit is not None:
            audit.record(event.id, keys)
        if entity.casefold() not in {key.casefold() for key in keys}:
            continue
        source_link = canonical_source_link(event)
        event_links = [source_link] if source_link else []
        event_links.extend(f"ticket:{key}" for key in keys if key.casefold() != entity.casefold())
        for link in event_links:
            if link not in links:
                links.append(link)
                source_ids.append(event.id)
    return sorted(links), unique(source_ids)


def generate_multisource_probes(
    events: list[EvalEvent],
    *,
    count: int,
    seed: int,
    probe_start: datetime,
    probe_end: datetime,
    join_audit: JoinAuditTrail | None = None,
) -> list[Probe]:
    """Ask for regex-key joined PR, commit, thread, and ticket identifiers."""

    start, end = validate_period(probe_start, probe_end)
    rng = random.Random(f"multisource:{seed}")
    candidates: list[ProbeCandidate] = []
    for event in events:
        source_link = canonical_source_link(event)
        keys = regex_join_keys(event)
        if join_audit is not None:
            join_audit.record(event.id, keys)
        if source_link is None or not keys or event.time > end:
            continue
        lower = max(start, event.time)
        if lower > end:
            continue
        snapshot_time = uniform_time(rng, lower, end)
        for entity in keys:
            links, source_ids = _links_as_of(
                events, entity, snapshot_time, audit=join_audit
            )
            if not links:
                continue
            candidates.append(
                ProbeCandidate(
                    family="multisource",
                    entity=entity,
                    T=snapshot_time,
                    question=(
                        "Which pull requests, commits, mail threads, or tickets are "
                        f"regex-key joined to {entity}?"
                    ),
                    gold=links,
                    gold_type="links",
                    source_event_ids=tuple(source_ids),
                    stratum=f"links:{entity.casefold()}",
                )
            )
    return stratified_sample(candidates, count, seed=seed)


def write_join_report(audit: JoinAuditTrail, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit.report(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


generate = generate_multisource_probes
