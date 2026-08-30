"""Cross-source link probe generator for docs/evaluation-spec.md §4 and §7 E2."""

from __future__ import annotations

import random
from datetime import datetime

from harnext_eval.probes.common import ISSUE_KEY_RE, uniform_time, unique, validate_period
from harnext_eval.probes.schema import ProbeCandidate, stratified_sample
from harnext_eval.types import EvalEvent, Probe


def _links_for_event(event: EvalEvent) -> list[str]:
    data = event.data or {}
    event_type = event.type.casefold()
    if "pull_request" in event_type:
        number = data.get("number") or data.get("pr_number")
        return [f"pr:{number}"] if number is not None else []
    if "mail" in event_type:
        thread = data.get("thread_id") or data.get("thread_root")
        if isinstance(thread, str) and thread:
            return [thread if thread.startswith("thread:") else f"thread:{thread}"]
    raw_links = data.get("linked_issues") or data.get("issue_links")
    if isinstance(raw_links, list):
        text = " ".join(str(item) for item in raw_links)
        return unique(ISSUE_KEY_RE.findall(text))
    return []


def _referenced_issue_keys(event: EvalEvent) -> list[str]:
    data = event.data or {}
    values = [
        event.subject,
        data.get("issue_key", ""),
        data.get("title", ""),
        data.get("subject", ""),
        data.get("body", ""),
        data.get("subject_tags", ""),
    ]
    return unique(ISSUE_KEY_RE.findall(" ".join(str(value) for value in values)))


def _links_as_of(
    events: list[EvalEvent], entity: str, at: datetime
) -> tuple[list[str], list[str]]:
    links: list[str] = []
    source_ids: list[str] = []
    for event in events:
        if event.time > at:
            break
        if entity.casefold() not in {
            issue_key.casefold() for issue_key in _referenced_issue_keys(event)
        }:
            continue
        for link in _links_for_event(event):
            if link not in links:
                links.append(link)
                source_ids.append(event.id)
    return sorted(links), source_ids


def generate_multisource_probes(
    events: list[EvalEvent],
    *,
    count: int,
    seed: int,
    probe_start: datetime,
    probe_end: datetime,
) -> list[Probe]:
    """Ask for PR and thread identifiers regex-joined to an issue entity."""

    start, end = validate_period(probe_start, probe_end)
    rng = random.Random(f"multisource:{seed}")
    candidates: list[ProbeCandidate] = []
    for event in events:
        links_for_event = _links_for_event(event)
        issue_keys = _referenced_issue_keys(event)
        if not links_for_event or not issue_keys or event.time > end:
            continue
        lower = max(start, event.time)
        if lower > end:
            continue
        snapshot_time = uniform_time(rng, lower, end)
        for entity in issue_keys:
            links, source_ids = _links_as_of(events, entity, snapshot_time)
            if not links:
                continue
            candidates.append(
                ProbeCandidate(
                    family="multisource",
                    entity=entity,
                    T=snapshot_time,
                    question=f"Which pull requests or mail threads are related to {entity}?",
                    gold=links,
                    gold_type="links",
                    source_event_ids=tuple(unique(source_ids)),
                    stratum=entity.casefold(),
                )
            )
    return stratified_sample(candidates, count, seed=seed)


generate = generate_multisource_probes
