"""Jira REST corpus extractor for docs/evaluation-spec.md §3.1/§4.1."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, BinaryIO, cast
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from harnext_eval.corpus.keys import contributor_key, derive_baseline_keys, issue_subject
from harnext_eval.types import EvalEvent

JsonObject = dict[str, Any]
PageFetcher = Callable[[int, int], Mapping[str, Any]]


def parse_search_page(payload: bytes | str | Mapping[str, Any], *, mgtenant: str = "kafka") -> list[EvalEvent]:
    """Parse one Jira search page expanded with changelogs and comments."""

    page = _json_object(payload)
    issues = page.get("issues", [])
    if not isinstance(issues, list):
        raise ValueError("Jira search payload 'issues' must be a list")
    events: list[EvalEvent] = []
    for issue in issues:
        if not isinstance(issue, Mapping):
            raise ValueError("each Jira issue must be an object")
        events.extend(parse_issue(issue, mgtenant=mgtenant))
    return sorted(events, key=lambda event: (event.time, event.id))


def parse_issue(issue: Mapping[str, Any], *, mgtenant: str = "kafka") -> list[EvalEvent]:
    """Expand one Jira issue into created, transition-item, and comment events."""

    issue_key = _required_string(issue, "key")
    issue_id = str(issue.get("id") or issue_key)
    fields = issue.get("fields")
    if not isinstance(fields, Mapping):
        raise ValueError(f"Jira issue {issue_key} has no fields object")

    components = _named_values(fields.get("components"))
    creator = _identity(fields.get("creator") or fields.get("reporter"))
    baseline_components = components
    common = {
        "issue_key": issue_key,
        "summary": fields.get("summary"),
        "description": fields.get("description"),
        "components": components,
    }
    created_data = {
        **common,
        "status": _named_value(fields.get("status")),
        "priority": _named_value(fields.get("priority")),
        "assignee": _identity_value(fields.get("assignee")),
        "reporter": _identity_value(fields.get("reporter")),
        "creator": creator.value,
        "fix_versions": _named_values(fields.get("fixVersions")),
        "labels": fields.get("labels") if isinstance(fields.get("labels"), list) else [],
    }
    events = [
        EvalEvent(
            id=f"jira:{issue_id}:created",
            source=f"jira:{issue_key.split('-', 1)[0]}",
            type="org.apache.jira.issue.created",
            subject=issue_subject(issue_key),
            time=_parse_time(_required_string(fields, "created")),
            mgtenant=mgtenant,
            baseline_keys=derive_baseline_keys(
                author_emails=[creator.email] if creator.email else [],
                components=baseline_components,
            ),
            data=created_data,
        )
    ]

    changelog = issue.get("changelog")
    histories = changelog.get("histories", []) if isinstance(changelog, Mapping) else []
    if not isinstance(histories, list):
        raise ValueError(f"Jira issue {issue_key} changelog histories must be a list")
    for history_index, history in enumerate(histories):
        if not isinstance(history, Mapping):
            continue
        changed_at = _parse_time(_required_string(history, "created"))
        actor = _identity(history.get("author"))
        items = history.get("items", [])
        if not isinstance(items, list):
            continue
        history_id = str(history.get("id") or history_index)
        for item_index, item in enumerate(items):
            if not isinstance(item, Mapping):
                continue
            field = str(item.get("field") or item.get("fieldId") or "unknown")
            data = {
                **common,
                "changelog_id": history_id,
                "field": field,
                "from": item.get("fromString", item.get("from")),
                "to": item.get("toString", item.get("to")),
                "actor": actor.value,
                "actor_name": actor.display_name,
                "actor_account_id": actor.account_id,
            }
            events.append(
                EvalEvent(
                    id=f"jira:{issue_id}:change:{history_id}:{item_index}",
                    source=f"jira:{issue_key.split('-', 1)[0]}",
                    type="org.apache.jira.issue.transition",
                    subject=issue_subject(issue_key),
                    time=changed_at,
                    mgtenant=mgtenant,
                    baseline_keys=derive_baseline_keys(
                        author_emails=[actor.email] if actor.email else [],
                        components=baseline_components,
                    ),
                    data=data,
                )
            )

    for comment in _comments(fields.get("comment")):
        comment_id = str(comment.get("id") or _stable_fragment(comment))
        author = _identity(comment.get("author"))
        data = {
            **common,
            "comment_id": comment_id,
            "body": comment.get("body"),
            "author": author.value,
            "author_name": author.display_name,
            "author_account_id": author.account_id,
            "author_active": author.active,
            "updated": comment.get("updated"),
        }
        events.append(
            EvalEvent(
                id=f"jira:{issue_id}:comment:{comment_id}",
                source=f"jira:{issue_key.split('-', 1)[0]}",
                type="org.apache.jira.issue.comment",
                subject=issue_subject(issue_key),
                time=_parse_time(_required_string(comment, "created")),
                mgtenant=mgtenant,
                baseline_keys=derive_baseline_keys(
                    author_emails=[author.email] if author.email else [],
                    components=baseline_components,
                ),
                data=data,
            )
        )
    return sorted(events, key=lambda event: (event.time, event.id))


def iter_search_pages(
    fetch_page: PageFetcher,
    *,
    start_at: int = 0,
    max_results: int = 100,
) -> Iterator[Mapping[str, Any]]:
    """Yield Jira search pages without assuming the server honours page size."""

    if start_at < 0 or max_results <= 0:
        raise ValueError("start_at must be non-negative and max_results positive")
    cursor = start_at
    while True:
        page = fetch_page(cursor, max_results)
        issues = page.get("issues", [])
        if not isinstance(issues, list):
            raise ValueError("Jira search payload 'issues' must be a list")
        yield page
        count = len(issues)
        if count == 0 or page.get("isLast") is True or page.get("isLastPage") is True:
            return
        page_start = page.get("startAt", cursor)
        if not isinstance(page_start, int):
            page_start = cursor
        next_cursor = page_start + count
        total = page.get("total")
        if isinstance(total, int) and next_cursor >= total:
            return
        if next_cursor <= cursor:
            raise ValueError("Jira pagination did not advance")
        cursor = next_cursor


def parse_search_pages(
    pages: Iterator[Mapping[str, Any]] | list[Mapping[str, Any]], *, mgtenant: str = "kafka"
) -> list[EvalEvent]:
    """Parse and globally order a sequence of already-fetched search pages."""

    events = [event for page in pages for event in parse_search_page(page, mgtenant=mgtenant)]
    return sorted(events, key=lambda event: (event.time, event.id))


def fetch(
    *,
    base_url: str,
    jql: str,
    mgtenant: str = "kafka",
    max_results: int = 100,
    timeout_s: float = 60,
    opener: Callable[..., BinaryIO] = urlopen,
) -> list[EvalEvent]:
    """Explicitly fetch all pages from Jira's REST search endpoint."""

    endpoint = f"{base_url.rstrip('/')}/rest/api/2/search"

    def fetch_page(start_at: int, page_size: int) -> Mapping[str, Any]:
        query = urlencode(
            {
                "jql": jql,
                "startAt": start_at,
                "maxResults": page_size,
                "expand": "changelog",
                "fields": ",".join(
                    (
                        "summary",
                        "description",
                        "created",
                        "creator",
                        "reporter",
                        "assignee",
                        "status",
                        "priority",
                        "fixVersions",
                        "components",
                        "labels",
                        "comment",
                    )
                ),
            }
        )
        request = Request(f"{endpoint}?{query}", headers={"Accept": "application/json"})
        with opener(request, timeout=timeout_s) as response:
            return _json_object(response.read())

    return parse_search_pages(
        iter_search_pages(fetch_page, max_results=max_results), mgtenant=mgtenant
    )


@dataclass(frozen=True)
class _Identity:
    value: str | None
    email: str | None
    display_name: str | None
    account_id: str | None
    active: bool | None


def _identity(value: Any) -> _Identity:
    if not isinstance(value, Mapping):
        return _Identity(None, None, None, None, None)
    raw_email = value.get("emailAddress")
    email = raw_email if isinstance(raw_email, str) else None
    account = value.get("accountId") or value.get("key") or value.get("name")
    account_id = str(account) if account else None
    display = value.get("displayName")
    display_name = str(display) if display else None
    identity: str | None = None
    if email:
        try:
            identity = contributor_key(email)
        except ValueError:
            identity = None
    if identity is None and account_id:
        digest = hashlib.sha256(account_id.encode()).hexdigest()[:12]
        identity = f"jira-user:{digest}"
    active = value.get("active") if isinstance(value.get("active"), bool) else None
    return _Identity(identity, email, display_name, account_id, active)


def _identity_value(value: Any) -> str | None:
    return _identity(value).value


def _comments(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        comments = value.get("comments", [])
    else:
        comments = value
    if not isinstance(comments, list):
        return []
    return [comment for comment in comments if isinstance(comment, Mapping)]


def _named_value(value: Any) -> str | None:
    if isinstance(value, Mapping):
        name = value.get("name") or value.get("value")
        return str(name) if name is not None else None
    return str(value) if value is not None else None


def _named_values(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [name for item in value if (name := _named_value(item)) is not None]


def _parse_time(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"Jira object is missing string field {key!r}")
    return item


def _stable_fragment(value: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _json_object(payload: bytes | str | Mapping[str, Any]) -> JsonObject:
    if isinstance(payload, Mapping):
        return dict(payload)
    raw = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Jira response must be a JSON object")
    return cast(JsonObject, value)


__all__ = [
    "fetch",
    "iter_search_pages",
    "parse_issue",
    "parse_search_page",
    "parse_search_pages",
]
