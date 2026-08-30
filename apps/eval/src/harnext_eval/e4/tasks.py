"""Task populations and corpus-specific gold for docs/evaluation-spec.md §7 E4."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from harnext_eval.config import WindowConfig
from harnext_eval.types import EvalEvent, Probe, Task

GOLD_GROUPS = ("people", "category", "place", "text")
DEFAULT_BOT_ACCOUNTS = frozenset(
    {
        "asfbot", "dependabot", "dependabot[bot]", "github-actions",
        "github-actions[bot]", "jenkins", "jira", "kafka-merge-bot",
        "renovate", "renovate[bot]",
    }
)
_KEY_RE = re.compile(r"\b(?:[A-Z][A-Z0-9]+-\d+|KIP-\d+|CVE-\d{4}-\d+)\b", re.I)
_CVE_RE = re.compile(r"\bCVE-\d{4}-\d+\b", re.I)
_FORMAT_ONLY_RE = re.compile(r"\b(?:format(?:ting)?|whitespace|style-only)\b", re.I)


def _data(event: EvalEvent) -> dict[str, Any]:
    return event.data if isinstance(event.data, dict) else {}


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
        if isinstance(value, Mapping):
            value = value.get("login") or value.get("name") or value.get("email")
        if value:
            return str(value)
    return None


def _event_text(event: EvalEvent) -> str:
    return json.dumps(
        {"source": event.source, "type": event.type, "subject": event.subject, "data": event.data},
        sort_keys=True,
        default=str,
    )


def is_rule_promoted(event: EvalEvent) -> bool:
    """Match only the three preregistered fast-trigger event shapes."""

    kind = event.type.casefold()
    source = event.source.casefold()
    data = _data(event)
    title = str(data.get("title") or data.get("subject") or "")
    body = str(data.get("body") or data.get("text") or data.get("message") or "")
    issue_creation = kind == "org.apache.jira.issue.created" and source.startswith("jira:")
    declared_priority = str(data.get("priority") or data.get("declared_priority") or "")
    if issue_creation and declared_priority.casefold() in {"blocker", "critical"}:
        return True
    mail_shape = kind == "org.apache.mail.message" and (
        source.startswith("mail:dev@") or source.startswith("dev@")
    )
    if mail_shape and "[vote]" in f"{title} {body}".casefold():
        return True
    # The configured rules floor promotes a CVE mention in any event payload;
    # unlike the Jira/VOTE cases it is deliberately not source-shape limited.
    return bool(_CVE_RE.search(json.dumps(data, sort_keys=True, default=str)))


def _issue_keys(event: EvalEvent) -> set[str]:
    values = [event.subject, str(_data(event).get("issue_key") or "")]
    return {match.group(0).upper() for value in values for match in _KEY_RE.finditer(value)}


def _title_keys(event: EvalEvent) -> set[str]:
    title = str(_data(event).get("title") or "")
    return {match.group(0).upper() for match in _KEY_RE.finditer(title)}


def _related_entity(trigger: EvalEvent, candidate: EvalEvent) -> bool:
    return trigger.subject == candidate.subject


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
                if isinstance(item, Mapping) and (value := _field_change(item, names)) is not None:
                    return value
    return None


def _as_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        value = value.get("name") or value.get("value") or value.get("login")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [child for item in value for child in _as_strings(item)]
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


def _formatting_only(event: EvalEvent) -> bool:
    data = _data(event)
    classification = str(data.get("change_classification") or "").casefold()
    return data.get("formatting_only") is True or classification == "formatting-only" or bool(
        _FORMAT_ONLY_RE.search(str(data.get("title") or ""))
    )


def _is_committer_reply(event: EvalEvent, committers: frozenset[str]) -> bool:
    data = _data(event)
    if not any(token in event.type.casefold() for token in ("comment", "message", "reply")):
        return False
    if data.get("is_committer") is True:
        return True
    role = str(data.get("author_role") or data.get("role") or "").casefold()
    if role in {"committer", "maintainer"}:
        return True
    return _normalise_account(_actor(event)) in committers


def _reply_body(event: EvalEvent) -> str | None:
    data = _data(event)
    for key in ("body", "comment", "text", "message"):
        if data.get(key):
            return str(data[key]).strip()
    return None


def _empty_gold(trigger: EvalEvent) -> dict[str, Any]:
    required_ids = sorted(_issue_keys(trigger))
    return {
        "people": {"assignees": [], "reviewers": [], "decision_times": [], "event_ids": []},
        "category": {
            "components": [], "duplicate_of": [], "priority_changes": [],
            "required_ids": required_ids, "decision_times": [], "event_ids": [],
        },
        "place": {"files": [], "modules": [], "decision_times": [], "event_ids": []},
        "text": {"replies": [], "decision_times": [], "event_ids": []},
        "_trigger_event": trigger.model_dump(mode="json"),
        "_gold_source": "derived-corpus-r",
    }


def derive_gold(
    trigger: EvalEvent,
    later_events: Iterable[EvalEvent],
    *,
    bot_accounts: Iterable[str] = DEFAULT_BOT_ACCOUNTS,
    committer_accounts: Iterable[str] = (),
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Derive Corpus-R post-T gold with exact horizons and title-only PR joins."""

    bots = frozenset(_normalise_account(bot) for bot in bot_accounts)
    committers = frozenset(_normalise_account(value) for value in committer_accounts)
    gold = _empty_gold(trigger)
    first_reply_found = False
    trigger_keys = _issue_keys(trigger)
    for event in sorted(later_events, key=lambda item: (item.time, item.id)):
        if event.time <= trigger.time:
            continue
        age = event.time - trigger.time
        if age > timedelta(days=14):
            break
        data = _data(event)
        is_pr = "pull_request" in event.type.casefold() or "github" in event.source.casefold()
        related = _related_entity(trigger, event) or (is_pr and bool(trigger_keys & _title_keys(event)))
        if not related:
            continue
        actor = _actor(event)
        if _is_bot(actor, bots):
            continue
        when = event.time.isoformat()

        if age <= timedelta(hours=24) and _related_entity(trigger, event):
            assignees = [
                value for value in _as_strings(_field_change(data, {"assignee"}))
                if not _is_bot(value, bots)
            ]
            if assignees:
                _append_unique(gold["people"]["assignees"], assignees)
                gold["people"]["decision_times"].append(when)
                gold["people"]["event_ids"].append(event.id)
            for names, key in (
                ({"component", "components"}, "components"),
                ({"priority"}, "priority_changes"),
            ):
                value = _field_change(data, names)
                if value is not None:
                    _append_unique(gold["category"][key], _as_strings(value))
                    gold["category"]["decision_times"].append(when)
                    gold["category"]["event_ids"].append(event.id)

        if "review" in event.type.casefold() and is_pr:
            reviewers = [
                value for value in _as_strings(data.get("reviewer") or data.get("user") or actor)
                if not _is_bot(value, bots)
            ]
            if reviewers:
                _append_unique(gold["people"]["reviewers"], reviewers)
                gold["people"]["decision_times"].append(when)
                gold["people"]["event_ids"].append(event.id)

        if age <= timedelta(days=7) and _related_entity(trigger, event):
            duplicate = _field_change(data, {"duplicate", "duplicateof", "duplicate_of"})
            duplicate = duplicate if duplicate is not None else data.get("duplicate_of")
            if duplicate is not None:
                _append_unique(gold["category"]["duplicate_of"], _as_strings(duplicate))
                gold["category"]["decision_times"].append(when)
                gold["category"]["event_ids"].append(event.id)

        merged = "merged" in event.type.casefold() or str(data.get("state", "")).casefold() == "merged"
        files = _as_strings(data.get("changed_files") or data.get("files"))
        if is_pr and merged and files and trigger_keys & _title_keys(event) and not _formatting_only(event):
            _append_unique(gold["place"]["files"], files)
            _append_unique(gold["place"]["modules"], (_module(path) for path in files if _module(path)))
            gold["place"]["decision_times"].append(when)
            gold["place"]["event_ids"].append(event.id)
            number = data.get("number") or data.get("pr_number")
            if number is not None:
                _append_unique(gold["category"]["required_ids"], [f"PR-{str(number).removeprefix('#')}"])
            _append_unique(
                gold["category"]["required_ids"],
                (key for key in _title_keys(event) if key.startswith("KIP-")),
            )

        if not first_reply_found and _related_entity(trigger, event) and _is_committer_reply(event, committers):
            if reply := _reply_body(event):
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


def _archetype(event: EvalEvent) -> str:
    text = _event_text(event).casefold()
    if "[vote]" in text:
        return "vote"
    if _CVE_RE.search(text):
        return "cve"
    return "declared_priority"


def _stratified_sample(tasks: list[Task], limit: int | None, seed: int) -> list[Task]:
    if limit is None or len(tasks) <= limit:
        return sorted(tasks, key=lambda task: (task.T, task.task_id))
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[Task]] = defaultdict(list)
    for task in tasks:
        trigger = task.gold.get("_trigger_event", {})
        source = str(trigger.get("source", "unknown")) if isinstance(trigger, Mapping) else "unknown"
        strata[(str(task.gold.get("_archetype", task.kind)), source)].append(task)
    queues: dict[tuple[str, str], deque[Task]] = {}
    for key, values in sorted(strata.items()):
        rng.shuffle(values)
        queues[key] = deque(values)
    selected: list[Task] = []
    while len(selected) < limit and any(queues.values()):
        for key in sorted(queues):
            if queues[key] and len(selected) < limit:
                selected.append(queues[key].popleft())
    return sorted(selected, key=lambda task: task.task_id)


def select_fast_tasks(
    events: Iterable[EvalEvent],
    *,
    corpus: str,
    bot_accounts: Iterable[str] = DEFAULT_BOT_ACCOUNTS,
    committer_accounts: Iterable[str] = (),
    limit: int | None = None,
    seed: int = 0,
) -> list[Task]:
    """Select seeded, stratified Corpus-R tasks with any human-decision gold."""

    ordered = sorted(events, key=lambda item: (item.time, item.id))
    tasks: list[Task] = []
    for index, trigger in enumerate(ordered):
        if not is_rule_promoted(trigger):
            continue
        gold, coverage = derive_gold(
            trigger,
            ordered[index + 1 :],
            bot_accounts=bot_accounts,
            committer_accounts=committer_accounts,
        )
        if not any(coverage.values()):
            continue
        gold["_archetype"] = _archetype(trigger)
        tasks.append(
            Task(
                task_id=f"{corpus}:fast:{trigger.id}", corpus=corpus, T=trigger.time,
                trigger_event_id=trigger.id, entity=trigger.subject, kind="fast",
                gold=gold, gold_coverage=coverage,
            )
        )
    return _stratified_sample(tasks, limit, seed)


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=parsed.tzinfo or UTC)


def _state_for_situation(
    meta: Mapping[str, Any], entity: str, onset: datetime
) -> tuple[Mapping[str, Any], str]:
    """Resolve the latest versioned simulator snapshot at or before onset."""

    raw = meta.get("world_state_snapshots")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("Corpus S E4 requires versioned world_state_snapshots metadata")
    candidates: list[tuple[datetime, int, str, Mapping[str, Any]]] = []
    for row in raw:
        if not isinstance(row, Mapping) or row.get("schema_version") != 1:
            continue
        recorded_at = _parse_time(row.get("time"))
        snapshot_id = str(row.get("snapshot_id") or "")
        entities = row.get("entities")
        if recorded_at is None or recorded_at > onset or not snapshot_id or not isinstance(entities, Mapping):
            continue
        state = entities.get(entity, entities.get(entity.split(":", 1)[-1]))
        if isinstance(state, Mapping):
            specificity = int(row.get("cadence") == "situation-onset")
            candidates.append((recorded_at, specificity, snapshot_id, state))
    if not candidates:
        raise ValueError(f"Corpus S E4 has no versioned world-state snapshot for {entity} at onset")
    _, _, snapshot_id, state = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    return state, snapshot_id


def _state_facts(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate stable fact identities and normalise the state fact table."""

    raw = state.get("facts")
    facts: list[dict[str, Any]] = []
    if isinstance(raw, Mapping):
        for fact_id, payload in raw.items():
            if isinstance(payload, Mapping):
                facts.append({"id": str(fact_id), **dict(payload)})
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        facts = [dict(payload) for payload in raw if isinstance(payload, Mapping)]
    ids = [str(fact.get("id") or "").strip() for fact in facts]
    if not facts or any(not fact_id for fact_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Corpus S world-state facts require unique, stable non-empty IDs")
    return facts


def _fact_for(facts: Sequence[Mapping[str, Any]], *fields: str) -> Mapping[str, Any] | None:
    wanted = {field.casefold() for field in fields}
    return next(
        (fact for fact in facts if str(fact.get("field") or "").casefold() in wanted),
        None,
    )


def _refund_action(value: Any) -> str:
    normalised = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    if normalised in {"approve", "approved", "eligible", "refund", "refunded", "true", "yes"}:
        return "refund"
    if normalised in {"deny", "denied", "ineligible", "do_not_refund", "false", "no"}:
        return "do_not_refund"
    if normalised in {"not_applicable", "none", "n/a", "na"}:
        return "route_and_reply"
    raise ValueError(f"unsupported Corpus S refund decision {value!r}")


def _action_event(
    situation_id: str,
    owner: Any,
    events: Iterable[EvalEvent],
    onset: datetime,
) -> EvalEvent:
    candidates = [
        event
        for event in events
        if event.time > onset
        and str(_data(event).get("outcome_for") or "") == situation_id
        and str(_data(event).get("field") or "").casefold() == "assignee"
        and _normalise_account(_data(event).get("to")) == _normalise_account(owner)
    ]
    if not candidates:
        raise ValueError("Corpus S handling has no matching post-onset action event")
    return min(candidates, key=lambda event: (event.time, event.id))


def build_constructed_tasks(
    events: Iterable[EvalEvent],
    *,
    corpus: str,
    corpus_meta: Mapping[str, Any],
    limit: int | None = 150,
    seed: int = 0,
) -> list[Task]:
    """Derive Corpus-S handling exclusively from versioned simulator state."""

    by_id = {event.id: event for event in events}
    raw_situations = corpus_meta.get("injected_situations") or corpus_meta.get("situations")
    if not isinstance(raw_situations, Sequence) or isinstance(raw_situations, (str, bytes)):
        raise ValueError("Corpus S E4 requires injected_situations metadata")
    tasks: list[Task] = []
    for raw in raw_situations:
        if not isinstance(raw, Mapping):
            continue
        event_id = str(raw.get("event_id") or raw.get("trigger_event_id") or "")
        trigger = by_id.get(event_id)
        if trigger is None:
            continue
        onset = _parse_time(raw.get("onset")) or trigger.time
        entity = str(raw.get("entity") or trigger.subject)
        state, snapshot_id = _state_for_situation(corpus_meta, entity, onset)
        facts = _state_facts(state)
        owner_fact = _fact_for(facts, "handling_owner", "owner")
        refund_fact = _fact_for(facts, "refund_decision")
        if owner_fact is None or refund_fact is None:
            raise ValueError("Corpus S handling state requires owner and refund_decision facts")
        owner = owner_fact.get("value")
        if not _as_strings(owner):
            raise ValueError("Corpus S handling owner fact has no value")
        situation_id = str(raw.get("situation_id") or "")
        if not situation_id:
            raise ValueError("Corpus S injected situation requires a stable situation_id")
        action_event = _action_event(situation_id, owner, by_id.values(), onset)
        action_time = action_event.time
        required = [
            str(fact["id"])
            for fact in facts
            if fact.get("required_for_handling") is True
        ]
        for fact in (owner_fact, refund_fact):
            if str(fact["id"]) not in required:
                required.append(str(fact["id"]))
        action = _refund_action(refund_fact.get("value"))
        category = {
            "components": [],
            "duplicate_of": [],
            "priority_changes": [],
            "refund_decision": str(refund_fact.get("value")),
            "required_ids": required,
            "decision_times": [],
            "event_ids": [],
        }
        gold = {
            "people": {
                "assignees": _as_strings(owner), "reviewers": [],
                "decision_times": [action_time.isoformat()], "event_ids": [action_event.id],
            },
            "category": category,
            "place": {"files": [], "modules": [], "decision_times": [], "event_ids": []},
            "text": {"replies": [], "decision_times": [], "event_ids": []},
            "refund_decision": str(refund_fact.get("value")),
            "scripted_action": action,
            "_trigger_event": trigger.model_dump(mode="json"),
            "_gold_source": "constructed-corpus-s-world-state-v1",
            "_world_state_snapshot_id": snapshot_id,
            "_archetype": str(raw.get("archetype") or raw.get("situation_archetype") or "situation"),
        }
        coverage = {
            "people": True,
            "category": bool(category["components"] or category["duplicate_of"] or category["priority_changes"]),
            "place": False,
            "text": False,
        }
        tasks.append(
            Task(
                task_id=f"{corpus}:fast:{event_id}", corpus=corpus, T=onset,
                trigger_event_id=event_id, entity=entity, kind="fast", gold=gold,
                gold_coverage=coverage,
            )
        )
    if not tasks:
        raise ValueError("Corpus S injected situations lack versioned world-state handling gold")
    return _stratified_sample(tasks, limit, seed)


def _window_closes(
    events: Iterable[EvalEvent], window: WindowConfig
) -> dict[tuple[str, datetime], tuple[str, ...]]:
    by_entity: dict[str, list[EvalEvent]] = defaultdict(list)
    for event in sorted(events, key=lambda item: (item.time, item.id)):
        by_entity[event.subject].append(event)
    closes: dict[tuple[str, datetime], tuple[str, ...]] = {}
    for entity, entity_events in by_entity.items():
        current: list[EvalEvent] = []
        opened: datetime | None = None
        for event in entity_events:
            if current and (
                (event.time - current[-1].time).total_seconds() > window.gap_s
                or len(current) >= window.max_events
                or (opened is not None and (event.time - opened).total_seconds() > window.max_age_s)
            ):
                closes[(entity, current[-1].time)] = tuple(item.id for item in current)
                current, opened = [], None
            if not current:
                opened = event.time
            current.append(event)
            if len(current) >= window.max_events:
                closes[(entity, current[-1].time)] = tuple(item.id for item in current)
                current, opened = [], None
        if current:
            closes[(entity, current[-1].time)] = tuple(item.id for item in current)
    return closes


def build_batch_tasks(
    events: Iterable[EvalEvent],
    probes: Iterable[Probe],
    *,
    corpus: str,
    window: WindowConfig,
    limit: int | None = None,
    seed: int = 0,
) -> list[Task]:
    """Seed-sample batch closes having frozen E2 probes at exactly that close."""

    closes = _window_closes(events, window)
    grouped: dict[tuple[str, datetime], list[Probe]] = defaultdict(list)
    for probe in probes:
        grouped[(probe.entity, probe.T)].append(probe)
    candidates: list[Task] = []
    for (entity, close_time), window_event_ids in sorted(
        closes.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        paired = sorted(grouped.get((entity, close_time), []), key=lambda probe: probe.probe_id)
        if paired:
            trigger_id = window_event_ids[-1]
            candidates.append(
                Task(
                    task_id=f"{corpus}:batch:{trigger_id}", corpus=corpus, T=close_time,
                    trigger_event_id=trigger_id, entity=entity, kind="batch",
                    gold={
                        "probes": [probe.model_dump(mode="json") for probe in paired],
                        "window_event_ids": list(window_event_ids),
                    },
                    gold_coverage={group: False for group in GOLD_GROUPS},
                )
            )
    if limit is None or len(candidates) <= limit:
        return candidates
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return sorted(candidates[:limit], key=lambda task: task.task_id)


def build_tasks(
    events: Iterable[EvalEvent],
    *,
    corpus: str,
    probes: Iterable[Probe] = (),
    window: WindowConfig | None = None,
    bot_accounts: Iterable[str] = DEFAULT_BOT_ACCOUNTS,
    committer_accounts: Iterable[str] = (),
    corpus_meta: Mapping[str, Any] | None = None,
    fast_limit: int | None = 150,
    batch_limit: int | None = 150,
    seed: int = 0,
) -> list[Task]:
    """Build distinct Corpus-R/Corpus-S fast gold and batch populations."""

    event_list = list(events)
    meta = corpus_meta or {}
    situations = meta.get("injected_situations") or meta.get("situations")
    if situations:
        tasks = build_constructed_tasks(
            event_list, corpus=corpus, corpus_meta=meta, limit=fast_limit, seed=seed
        )
    else:
        tasks = select_fast_tasks(
            event_list, corpus=corpus, bot_accounts=bot_accounts,
            committer_accounts=committer_accounts, limit=fast_limit, seed=seed,
        )
    if window is not None:
        tasks.extend(
            build_batch_tasks(
                event_list, probes, corpus=corpus, window=window,
                limit=batch_limit, seed=seed,
            )
        )
    return tasks


__all__ = [
    "DEFAULT_BOT_ACCOUNTS", "GOLD_GROUPS", "build_batch_tasks",
    "build_constructed_tasks", "build_tasks", "derive_gold",
    "is_rule_promoted", "select_fast_tasks",
]
