"""Task population and derived gold for docs/evaluation-spec.md §7 E4 and D14."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

from harnext_eval.config import WindowConfig
from harnext_eval.types import EvalEvent, Probe, Task

GOLD_GROUPS = ("people", "category", "place", "text")
DEFAULT_BOT_ACCOUNTS = frozenset(
    {
        "asfbot",
        "dependabot",
        "dependabot[bot]",
        "github-actions",
        "github-actions[bot]",
        "jenkins",
        "jira",
        "kafka-merge-bot",
        "renovate",
        "renovate[bot]",
    }
)
_RULE_RE = re.compile(r"(?:\[vote\]|\bcve(?:-\d{4}-\d+)?\b)", re.IGNORECASE)
_KEY_RE = re.compile(r"\b(?:[A-Z][A-Z0-9]+-\d+|KIP-\d+|CVE-\d{4}-\d+)\b", re.IGNORECASE)
_FORMAT_ONLY_RE = re.compile(r"\b(?:format(?:ting)?|whitespace|style-only)\b", re.IGNORECASE)


def _data(event: EvalEvent) -> dict[str, Any]:
    return event.data if isinstance(event.data, dict) else {}


def _json_text(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _normalise_account(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if "<" in text:
        text = text.split("<", 1)[0].strip()
    if "@" in text and not text.endswith("[bot]"):
        text = text.split("@", 1)[0]
    return text


def _is_bot(value: Any, bots: frozenset[str]) -> bool:
    account = _normalise_account(value)
    return bool(account) and (
        account in bots or account.endswith("[bot]") or account.endswith("-bot")
    )


def _actor(event: EvalEvent) -> str | None:
    data = _data(event)
    for key in ("actor", "author", "user", "login", "reviewer", "from"):
        value = data.get(key)
        if value:
            if isinstance(value, Mapping):
                value = value.get("login") or value.get("name") or value.get("email")
            return str(value) if value else None
    return None


def _priority_values(data: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("priority", "declared_priority"):
        value = data.get(key)
        if value is not None:
            values.append(str(value))
    state = data.get("state")
    if isinstance(state, Mapping) and state.get("priority") is not None:
        values.append(str(state["priority"]))
    if str(data.get("field", "")).casefold() == "priority":
        values.append(str(data.get("to", "")))
    return values


def is_rule_promoted(event: EvalEvent) -> bool:
    """Return whether the rules floor promotes an E4 trigger event."""

    if any(value.casefold() in {"blocker", "critical"} for value in _priority_values(_data(event))):
        return True
    return bool(_RULE_RE.search(_json_text({"type": event.type, "data": event.data})))


def _issue_keys(event: EvalEvent) -> set[str]:
    keys = {match.group(0).upper() for match in _KEY_RE.finditer(_json_text(event.data))}
    keys.update(match.group(0).upper() for match in _KEY_RE.finditer(event.subject or ""))
    return keys


def _related(trigger: EvalEvent, candidate: EvalEvent) -> bool:
    if trigger.subject == candidate.subject:
        return True
    trigger_keys = _issue_keys(trigger)
    return bool(trigger_keys and trigger_keys.intersection(_issue_keys(candidate)))


def _field_change(data: Mapping[str, Any], names: set[str]) -> Any | None:
    field = str(data.get("field", "")).casefold().replace("_", "")
    normalised = {name.casefold().replace("_", "") for name in names}
    if field in normalised:
        return data.get("to") if "to" in data else data.get("value")
    changelog = data.get("changelog")
    if isinstance(changelog, Mapping):
        items = changelog.get("items")
        if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
            for item in items:
                if isinstance(item, Mapping):
                    value = _field_change(item, names)
                    if value is not None:
                        return value
    return None


def _as_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        value = value.get("name") or value.get("value") or value.get("login")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result: list[str] = []
        for item in value:
            result.extend(_as_strings(item))
        return result
    text = str(value).strip()
    return [text] if text else []


def _append_unique(target: list[str], values: Iterable[str]) -> None:
    seen = {item.casefold() for item in target}
    for value in values:
        if value.casefold() not in seen:
            target.append(value)
            seen.add(value.casefold())


def _module(path: str) -> str:
    parts = [part for part in path.replace("\\", "/").split("/") if part]
    return "/".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "")


def _formatting_only(event: EvalEvent, files: list[str]) -> bool:
    data = _data(event)
    if data.get("formatting_only") is True:
        return True
    title = str(data.get("title", ""))
    if _FORMAT_ONLY_RE.search(title):
        return True
    non_code_suffixes = {".md", ".txt", ".rst"}
    return bool(files) and all(
        any(path.casefold().endswith(suffix) for suffix in non_code_suffixes) for path in files
    )


def _is_committer_reply(event: EvalEvent) -> bool:
    data = _data(event)
    if not any(token in event.type.casefold() for token in ("comment", "message", "reply")):
        return False
    explicit = data.get("is_committer")
    if explicit is not None:
        return bool(explicit)
    role = str(data.get("author_role") or data.get("role") or "").casefold()
    return not role or role in {"committer", "maintainer", "member", "developer"}


def _reply_body(event: EvalEvent) -> str | None:
    data = _data(event)
    for key in ("body", "comment", "text", "message"):
        if data.get(key):
            return str(data[key]).strip()
    return None


def _empty_gold(trigger: EvalEvent) -> dict[str, Any]:
    required_ids = sorted(_issue_keys(trigger))
    if not required_ids:
        required_ids = [trigger.subject]
    return {
        "people": {"assignees": [], "reviewers": [], "decision_times": [], "event_ids": []},
        "category": {
            "components": [],
            "duplicate_of": [],
            "priority_changes": [],
            "required_ids": required_ids,
            "decision_times": [],
            "event_ids": [],
        },
        "place": {"files": [], "modules": [], "decision_times": [], "event_ids": []},
        "text": {"replies": [], "decision_times": [], "event_ids": []},
        "_trigger_event": trigger.model_dump(mode="json"),
    }


def derive_gold(
    trigger: EvalEvent,
    later_events: Iterable[EvalEvent],
    *,
    bot_accounts: Iterable[str] = DEFAULT_BOT_ACCOUNTS,
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Derive the four unordered post-T gold groups for one fast task."""

    bots = frozenset(_normalise_account(bot) for bot in bot_accounts)
    gold = _empty_gold(trigger)
    first_reply_found = False
    for event in sorted(later_events, key=lambda item: (item.time, item.id)):
        if event.time <= trigger.time or not _related(trigger, event):
            continue
        age = event.time - trigger.time
        if age > timedelta(days=14):
            break
        actor = _actor(event)
        if _is_bot(actor, bots):
            continue
        data = _data(event)
        when = event.time.isoformat()

        if age <= timedelta(hours=24):
            assignee = _field_change(data, {"assignee"})
            assignees = [value for value in _as_strings(assignee) if not _is_bot(value, bots)]
            if assignees:
                _append_unique(gold["people"]["assignees"], assignees)
                gold["people"]["decision_times"].append(when)
                gold["people"]["event_ids"].append(event.id)

            component = _field_change(data, {"component", "components"})
            if component is not None:
                _append_unique(gold["category"]["components"], _as_strings(component))
                gold["category"]["decision_times"].append(when)
                gold["category"]["event_ids"].append(event.id)

            priority = _field_change(data, {"priority"})
            if priority is not None:
                _append_unique(gold["category"]["priority_changes"], _as_strings(priority))
                gold["category"]["decision_times"].append(when)
                gold["category"]["event_ids"].append(event.id)

        reviewer_values: list[str] = []
        if "review" in event.type.casefold():
            reviewer_values.extend(_as_strings(data.get("reviewer") or data.get("user") or actor))
        reviewer_values = [value for value in reviewer_values if not _is_bot(value, bots)]
        if reviewer_values:
            _append_unique(gold["people"]["reviewers"], reviewer_values)
            gold["people"]["decision_times"].append(when)
            gold["people"]["event_ids"].append(event.id)

        if age <= timedelta(days=7):
            duplicate = _field_change(data, {"duplicate", "duplicateof", "duplicate_of"})
            duplicate = duplicate if duplicate is not None else data.get("duplicate_of")
            if duplicate is not None:
                _append_unique(gold["category"]["duplicate_of"], _as_strings(duplicate))
                gold["category"]["decision_times"].append(when)
                gold["category"]["event_ids"].append(event.id)

        merged = (
            "merged" in event.type.casefold() or str(data.get("state", "")).casefold() == "merged"
        )
        files = _as_strings(data.get("changed_files") or data.get("files"))
        if merged and files and not _formatting_only(event, files):
            _append_unique(gold["place"]["files"], files)
            _append_unique(
                gold["place"]["modules"], (_module(path) for path in files if _module(path))
            )
            gold["place"]["decision_times"].append(when)
            gold["place"]["event_ids"].append(event.id)
        if not first_reply_found and _is_committer_reply(event):
            reply = _reply_body(event)
            if reply:
                gold["text"]["replies"].append(reply)
                gold["text"]["decision_times"].append(when)
                gold["text"]["event_ids"].append(event.id)
                first_reply_found = True

    coverage = {
        "people": bool(gold["people"]["assignees"] or gold["people"]["reviewers"]),
        "category": bool(
            gold["category"]["components"]
            or gold["category"]["duplicate_of"]
            or gold["category"]["priority_changes"]
        ),
        "place": bool(gold["place"]["files"]),
        "text": bool(gold["text"]["replies"]),
    }
    return gold, coverage


def select_fast_tasks(
    events: Iterable[EvalEvent],
    *,
    corpus: str,
    bot_accounts: Iterable[str] = DEFAULT_BOT_ACCOUNTS,
    limit: int | None = None,
) -> list[Task]:
    """Select rule-promoted events having at least one derived post-T decision."""

    ordered = sorted(events, key=lambda item: (item.time, item.id))
    tasks: list[Task] = []
    for index, trigger in enumerate(ordered):
        if not is_rule_promoted(trigger):
            continue
        gold, coverage = derive_gold(trigger, ordered[index + 1 :], bot_accounts=bot_accounts)
        if not any(coverage.values()):
            continue
        tasks.append(
            Task(
                task_id=f"{corpus}:fast:{trigger.id}",
                corpus=corpus,
                T=trigger.time,
                trigger_event_id=trigger.id,
                entity=trigger.subject,
                kind="fast",
                gold=gold,
                gold_coverage=coverage,
            )
        )
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def _window_closes(
    events: Iterable[EvalEvent], window: WindowConfig
) -> dict[tuple[str, datetime], str]:
    by_entity: dict[str, list[EvalEvent]] = defaultdict(list)
    for event in sorted(events, key=lambda item: (item.time, item.id)):
        by_entity[event.subject].append(event)
    closes: dict[tuple[str, datetime], str] = {}
    for entity, entity_events in by_entity.items():
        current: list[EvalEvent] = []
        opened: datetime | None = None
        for event in entity_events:
            if current and (
                (event.time - current[-1].time).total_seconds() > window.gap_s
                or len(current) >= window.max_events
                or (opened is not None and (event.time - opened).total_seconds() > window.max_age_s)
            ):
                closes[(entity, current[-1].time)] = current[-1].id
                current = []
                opened = None
            if not current:
                opened = event.time
            current.append(event)
            if len(current) >= window.max_events:
                closes[(entity, current[-1].time)] = current[-1].id
                current = []
                opened = None
        if current:
            closes[(entity, current[-1].time)] = current[-1].id
    return closes


def build_batch_tasks(
    events: Iterable[EvalEvent],
    probes: Iterable[Probe],
    *,
    corpus: str,
    window: WindowConfig,
    limit: int | None = None,
) -> list[Task]:
    """Pair E2 probes with their entity's batch window at its close time."""

    closes = _window_closes(events, window)
    grouped: dict[tuple[str, datetime], list[Probe]] = defaultdict(list)
    for probe in probes:
        grouped[(probe.entity, probe.T)].append(probe)
    tasks: list[Task] = []
    for (entity, close_time), trigger_id in sorted(
        closes.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        paired = grouped.get((entity, close_time), [])
        if not paired:
            continue
        tasks.append(
            Task(
                task_id=f"{corpus}:batch:{trigger_id}",
                corpus=corpus,
                T=close_time,
                trigger_event_id=trigger_id,
                entity=entity,
                kind="batch",
                gold={"probes": [probe.model_dump(mode="json") for probe in paired]},
                gold_coverage={group: False for group in GOLD_GROUPS},
            )
        )
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def build_tasks(
    events: Iterable[EvalEvent],
    *,
    corpus: str,
    probes: Iterable[Probe] = (),
    window: WindowConfig | None = None,
    bot_accounts: Iterable[str] = DEFAULT_BOT_ACCOUNTS,
    fast_limit: int | None = 150,
    batch_limit: int | None = 150,
) -> list[Task]:
    """Build the E4 fast and (when probes/window are supplied) batch populations."""

    event_list = list(events)
    tasks = select_fast_tasks(
        event_list, corpus=corpus, bot_accounts=bot_accounts, limit=fast_limit
    )
    if window is not None:
        tasks.extend(
            build_batch_tasks(
                event_list,
                probes,
                corpus=corpus,
                window=window,
                limit=batch_limit,
            )
        )
    return tasks
