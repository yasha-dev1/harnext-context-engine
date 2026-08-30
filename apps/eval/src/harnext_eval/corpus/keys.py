"""Entity and routing-baseline keys from docs/evaluation-spec.md §3.1."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from email.utils import parseaddr
from typing import Any

from harnext_eval.types import EvalEvent

_ISSUE_RE = re.compile(r"(?<![A-Z0-9])KAFKA-(\d+)\b", re.IGNORECASE)
_KIP_RE = re.compile(r"(?<![A-Z0-9])KIP-(\d+)\b", re.IGNORECASE)
_TEXT_KEY_RE = re.compile(
    r"(?<![A-Z0-9])(?:(?P<issue>KAFKA)-(?P<issue_number>\d+)"
    r"|(?P<kip>KIP)-(?P<kip_number>\d+))\b",
    re.IGNORECASE,
)
_SPACE_RE = re.compile(r"\s+")


def extract_issue_keys(text: str | None) -> list[str]:
    """Return canonical ``KAFKA-N`` references in first-occurrence order."""

    if not text:
        return []
    return _dedupe(f"KAFKA-{match.group(1)}" for match in _ISSUE_RE.finditer(text))


def extract_kip_keys(text: str | None) -> list[str]:
    """Return canonical ``KIP-N`` references in first-occurrence order."""

    if not text:
        return []
    return _dedupe(f"KIP-{match.group(1)}" for match in _KIP_RE.finditer(text))


def extract_text_keys(text: str | None) -> list[str]:
    """Return subject-form issue/KIP keys in their order in arbitrary text."""

    if not text:
        return []
    keys: list[str] = []
    for match in _TEXT_KEY_RE.finditer(text):
        if match.group("issue"):
            keys.append(f"issue:KAFKA-{match.group('issue_number')}")
        else:
            keys.append(f"kip:{match.group('kip_number')}")
    return _dedupe(keys)


def issue_subject(issue_key: str) -> str:
    """Canonicalise an Apache Kafka Jira key as an event subject."""

    matches = extract_issue_keys(issue_key)
    if not matches:
        raise ValueError(f"not a KAFKA issue key: {issue_key!r}")
    return f"issue:{matches[0]}"


def kip_subject(kip: str | int) -> str:
    """Canonicalise a KIP number or ``KIP-N`` string as an event subject."""

    value = str(kip).strip()
    if value.isdigit():
        return f"kip:{int(value)}"
    matches = extract_kip_keys(value)
    if not matches:
        raise ValueError(f"not a KIP key: {kip!r}")
    return f"kip:{int(matches[0].split('-', 1)[1])}"


def pr_subject(number: str | int) -> str:
    """Canonicalise a GitHub pull-request number as an event subject."""

    value = str(number).strip().removeprefix("#")
    if not value.isdigit():
        raise ValueError(f"not a pull-request number: {number!r}")
    return f"pr:{int(value)}"


def normalize_message_id(message_id: str) -> str:
    """Remove RFC 5322 angle brackets while retaining the globally unique id."""

    value = message_id.strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1].strip()
    if not value:
        raise ValueError("message id cannot be empty")
    return value


def thread_subject(root_message_id: str) -> str:
    """Build the thread entity key used by mail state and routing baselines."""

    return f"thread:{normalize_message_id(root_message_id)}"


def contributor_key(email: str) -> str:
    """Hash a canonical email address without retaining it in the entity key."""

    if email.startswith("contributor:"):
        return email
    address = parseaddr(email)[1].strip().casefold()
    if not address or "@" not in address:
        raise ValueError(f"not an email address: {email!r}")
    digest = hashlib.sha256(address.encode("utf-8")).hexdigest()[:12]
    return f"contributor:{digest}"


def component_key(component: str) -> str:
    """Canonicalise a Jira component for a stable, long-lived baseline key."""

    if component.startswith("component:"):
        component = component.split(":", 1)[1]
    value = _SPACE_RE.sub("-", component.strip().casefold())
    if not value:
        raise ValueError("component cannot be empty")
    return f"component:{value}"


def derive_baseline_keys(
    *,
    author_emails: Iterable[str] = (),
    components: Iterable[str] = (),
    thread_root: str | None = None,
) -> list[str]:
    """Derive long-lived anomaly keys from author, component, and mail thread."""

    keys: list[str] = []
    for email in author_emails:
        try:
            keys.append(contributor_key(email))
        except ValueError:
            continue
    for component in components:
        try:
            keys.append(component_key(component))
        except ValueError:
            continue
    if thread_root:
        try:
            keys.append(thread_subject(thread_root))
        except ValueError:
            pass
    return _dedupe(keys)


def derive_subject(event_type: str, data: Mapping[str, Any], fallback: str | None = None) -> str:
    """Derive the state entity for a normalized source event.

    Source-specific identifiers win over cross-source references: a pull request
    mentioning KAFKA-123 remains ``pr:N``, while a push with no native state
    entity attaches to its first KAFKA/KIP reference.
    """

    lowered_type = event_type.casefold()
    if ".jira.issue." in lowered_type or lowered_type.startswith("org.apache.jira"):
        issue_key = data.get("issue_key") or data.get("key")
        if issue_key:
            return issue_subject(str(issue_key))

    if lowered_type == "org.apache.mail.message" or ".mail.message" in lowered_type:
        root = data.get("thread_root") or data.get("message_id")
        if root:
            return thread_subject(str(root))

    if "pull_request" in lowered_type or lowered_type.endswith(
        (".review", ".review_comment")
    ):
        number = data.get("pull_request_number") or data.get("number")
        if number is not None:
            return pr_subject(str(number))

    if lowered_type.endswith(".issue_comment"):
        number = data.get("pull_request_number")
        if number is not None:
            return pr_subject(str(number))

    text = "\n".join(_text_candidates(data))
    text_keys = extract_text_keys(text)
    if text_keys:
        return text_keys[0]

    if fallback:
        return fallback
    raise ValueError(f"cannot derive subject for {event_type!r}")


def derive_event_baseline_keys(event: EvalEvent) -> list[str]:
    """Derive baseline keys from a parsed event, preserving explicit valid keys."""

    data = event.data or {}
    emails: list[str] = []
    for field in ("author_email", "actor_email", "reporter_email", "creator_email"):
        value = data.get(field)
        if isinstance(value, str):
            emails.append(value)

    raw_components = data.get("components", [])
    components: list[str]
    if isinstance(raw_components, str):
        components = [raw_components]
    elif isinstance(raw_components, list):
        components = [str(item) for item in raw_components if item]
    else:
        components = []

    root = data.get("thread_root")
    derived = derive_baseline_keys(
        author_emails=emails,
        components=components,
        thread_root=str(root) if root else None,
    )
    valid_explicit = [
        key
        for key in event.baseline_keys
        if key.startswith(("contributor:", "component:", "thread:"))
    ]
    return _dedupe([*valid_explicit, *derived])


def assign_event_keys(event: EvalEvent, *, mgtenant: str | None = None) -> EvalEvent:
    """Return an event with canonical subject, baselines, and optional tenant."""

    return event.model_copy(
        update={
            "subject": derive_subject(event.type, event.data or {}, event.subject),
            "baseline_keys": derive_event_baseline_keys(event),
            "mgtenant": mgtenant or event.mgtenant,
        }
    )


def _text_candidates(data: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for field in ("title", "subject", "body", "message", "ref"):
        value = data.get(field)
        if isinstance(value, str):
            values.append(value)
    commits = data.get("commits")
    if isinstance(commits, list):
        for commit in commits:
            if isinstance(commit, Mapping) and isinstance(commit.get("message"), str):
                values.append(commit["message"])
    return values


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


__all__ = [
    "assign_event_keys",
    "component_key",
    "contributor_key",
    "derive_baseline_keys",
    "derive_event_baseline_keys",
    "derive_subject",
    "extract_issue_keys",
    "extract_kip_keys",
    "extract_text_keys",
    "issue_subject",
    "kip_subject",
    "normalize_message_id",
    "pr_subject",
    "thread_subject",
]
