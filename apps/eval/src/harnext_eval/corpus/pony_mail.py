"""Pony Mail monthly archive extractor for docs/evaluation-spec.md §3.1/§4.1."""

from __future__ import annotations

import hashlib
import json
import mailbox
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.header import decode_header, make_header
from email.message import Message
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path
from typing import Any, BinaryIO, cast
from urllib.parse import urlencode
from urllib.request import urlopen

from harnext_eval.corpus.keys import (
    contributor_key,
    derive_baseline_keys,
    extract_issue_keys,
    extract_kip_keys,
    normalize_message_id,
    thread_subject,
)
from harnext_eval.types import EvalEvent

_MESSAGE_ID_RE = re.compile(r"<([^<>\s]+)>")
_BRACKET_TAG_RE = re.compile(r"\[(VOTE|DISCUSS)\]", re.IGNORECASE)
_MONTH_RE = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])$")


@dataclass(frozen=True)
class PonyMonth:
    """One fetched list-month and the independent Pony Mail statistics payload."""

    list_name: str
    domain: str
    month: str
    events: tuple[EvalEvent, ...]
    stats: dict[str, Any]

    @property
    def stats_count(self) -> int | None:
        return stats_message_count(self.stats, self.month)


@dataclass(frozen=True)
class _MailRecord:
    message_id: str
    in_reply_to: str | None
    references: tuple[str, ...]
    message: Message
    index: int


def parse_stats_json(payload: bytes | str | Mapping[str, Any]) -> dict[str, Any]:
    """Parse the JSON returned by Pony Mail's ``stats.lua`` endpoint."""

    if isinstance(payload, Mapping):
        return dict(payload)
    raw = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Pony Mail stats payload must be a JSON object")
    return cast(dict[str, Any], parsed)


def stats_message_count(stats: Mapping[str, Any], month: str) -> int | None:
    """Read a monthly count from current or older ``stats.lua`` response shapes."""

    monthly = stats.get("monthly_emails")
    if isinstance(monthly, Mapping):
        candidates = (month, f"{month}-01", month.replace("-", "/"))
        for candidate in candidates:
            value = monthly.get(candidate)
            if isinstance(value, int):
                return value
    hits = stats.get("hits")
    return hits if isinstance(hits, int) else None


def parse_mbox(
    path: str | Path,
    *,
    list_name: str,
    domain: str,
    month: str,
    mgtenant: str = "kafka",
) -> list[EvalEvent]:
    """Parse a monthly Unix mbox into ordered Apache mail ``EvalEvent`` records."""

    _validate_source(list_name, domain, month)
    archive = mailbox.mbox(Path(path), create=False)
    try:
        records = [_record(message, index) for index, message in enumerate(archive)]
    finally:
        archive.close()

    roots: dict[str, str] = {}
    by_id = {record.message_id: record for record in records}

    def resolve_root(record: _MailRecord, visiting: set[str] | None = None) -> str:
        known = roots.get(record.message_id)
        if known:
            return known
        if record.references:
            root = record.references[0]
        elif record.in_reply_to and record.in_reply_to in by_id:
            seen = set() if visiting is None else visiting
            if record.message_id in seen:
                root = record.message_id
            else:
                root = resolve_root(by_id[record.in_reply_to], seen | {record.message_id})
        elif record.in_reply_to:
            root = record.in_reply_to
        else:
            root = record.message_id
        roots[record.message_id] = root
        return root

    events = [
        _to_event(
            record,
            thread_root=resolve_root(record),
            list_name=list_name,
            domain=domain,
            month=month,
            mgtenant=mgtenant,
        )
        for record in records
    ]
    return sorted(events, key=lambda event: (event.time, event.id))


def fetch(
    *,
    list_name: str,
    domain: str,
    month: str,
    destination: str | Path,
    mgtenant: str = "kafka",
    base_url: str = "https://lists.apache.org/api",
    timeout_s: float = 60,
    opener: Callable[..., BinaryIO] = urlopen,
) -> PonyMonth:
    """Explicitly fetch and parse one Pony Mail month.

    Merely importing or parsing this module never accesses the network. The
    injectable opener keeps URL construction testable without live services.
    """

    _validate_source(list_name, domain, month)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    list_address = f"{list_name}@{domain}"
    mbox_query = urlencode({"list": list_address, "date": month})
    stats_query = urlencode(
        {"list": list_name, "domain": domain, "d": month, "quick": "true"}
    )
    mbox_url = f"{base_url.rstrip('/')}/mbox.lua?{mbox_query}"
    stats_url = f"{base_url.rstrip('/')}/stats.lua?{stats_query}"

    with opener(mbox_url, timeout=timeout_s) as response:
        target.write_bytes(response.read())
    with opener(stats_url, timeout=timeout_s) as response:
        stats = parse_stats_json(response.read())
    events = parse_mbox(
        target,
        list_name=list_name,
        domain=domain,
        month=month,
        mgtenant=mgtenant,
    )
    return PonyMonth(list_name, domain, month, tuple(events), stats)


def _record(message: Message, index: int) -> _MailRecord:
    message_id = _first_message_id(message.get("Message-ID"))
    if not message_id:
        seed = "\x1f".join(
            (_header(message, "Date"), _header(message, "From"), _header(message, "Subject"))
        )
        message_id = f"missing-{hashlib.sha256(seed.encode()).hexdigest()[:24]}"
    references = tuple(_message_ids(message.get("References")))
    in_reply_ids = _message_ids(message.get("In-Reply-To"))
    return _MailRecord(
        message_id=message_id,
        in_reply_to=in_reply_ids[0] if in_reply_ids else None,
        references=references,
        message=message,
        index=index,
    )


def _to_event(
    record: _MailRecord,
    *,
    thread_root: str,
    list_name: str,
    domain: str,
    month: str,
    mgtenant: str,
) -> EvalEvent:
    message = record.message
    subject = _header(message, "Subject")
    body = _plain_body(message)
    from_header = _header(message, "From")
    from_name, author_email = parseaddr(from_header)
    author = None
    if author_email:
        try:
            author = contributor_key(author_email)
        except ValueError:
            author = None
    tags = _subject_tags(subject)
    data: dict[str, Any] = {
        "list": f"{list_name}@{domain}",
        "month": month,
        "message_id": record.message_id,
        "in_reply_to": record.in_reply_to,
        "references": list(record.references),
        "thread_root": thread_root,
        "subject": subject,
        "subject_tags": tags,
        "body": body,
        "author": author,
        "author_name": from_name or None,
    }
    event_id = f"mail:{hashlib.sha256(record.message_id.encode()).hexdigest()[:24]}"
    return EvalEvent(
        id=event_id,
        source=f"mail:{list_name}@{domain}",
        type="org.apache.mail.message",
        subject=thread_subject(thread_root),
        time=_message_time(message),
        mgtenant=mgtenant,
        baseline_keys=derive_baseline_keys(
            author_emails=[author_email] if author_email else [], thread_root=thread_root
        ),
        data=data,
    )


def _subject_tags(subject: str) -> list[str]:
    bracketed = [f"[{match.group(1).upper()}]" for match in _BRACKET_TAG_RE.finditer(subject)]
    references = [*extract_kip_keys(subject), *extract_issue_keys(subject)]
    return list(dict.fromkeys([*bracketed, *references]))


def _plain_body(message: Message) -> str:
    if message.is_multipart():
        parts: list[str] = []
        for part in message.walk():
            if part.get_content_type() != "text/plain" or part.get_content_disposition() == "attachment":
                continue
            parts.append(_decode_payload(part))
        return "\n".join(part.strip() for part in parts if part.strip())
    return _decode_payload(message).strip()


def _decode_payload(message: Message) -> str:
    payload = message.get_payload(decode=True)
    if payload is None:
        value = message.get_payload()
        return value if isinstance(value, str) else ""
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, bytes):
        return ""
    charset = message.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def _message_time(message: Message) -> datetime:
    raw = message.get("Date")
    if not raw:
        raise ValueError("mail message is missing Date")
    parsed = parsedate_to_datetime(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _header(message: Message, name: str) -> str:
    value = message.get(name, "")
    try:
        return str(make_header(decode_header(value)))
    except (LookupError, UnicodeDecodeError):
        return value


def _message_ids(value: str | None) -> list[str]:
    if not value:
        return []
    bracketed = [normalize_message_id(match.group(1)) for match in _MESSAGE_ID_RE.finditer(value)]
    if bracketed:
        return bracketed
    return [normalize_message_id(token) for token in value.split() if token.strip()]


def _first_message_id(value: str | None) -> str | None:
    ids = _message_ids(value)
    return ids[0] if ids else None


def _validate_source(list_name: str, domain: str, month: str) -> None:
    if not list_name or "@" in list_name:
        raise ValueError("list_name must be the local part, for example 'dev'")
    if not domain or "." not in domain:
        raise ValueError("domain must be a mail domain")
    if not _MONTH_RE.fullmatch(month):
        raise ValueError("month must use YYYY-MM")


__all__ = [
    "PonyMonth",
    "fetch",
    "parse_mbox",
    "parse_stats_json",
    "stats_message_count",
]
