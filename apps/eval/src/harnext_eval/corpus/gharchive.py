"""Streaming GH Archive extractor for docs/evaluation-spec.md §3.1/§4.1."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO
from urllib.request import Request, urlopen

from harnext_eval.corpus.keys import (
    contributor_key,
    derive_baseline_keys,
    extract_issue_keys,
    extract_kip_keys,
    extract_text_keys,
    pr_subject,
    thread_subject,
)
from harnext_eval.types import EvalEvent

_SUPPORTED_TYPES = {
    "PullRequestEvent",
    "PullRequestReviewEvent",
    "PullRequestReviewCommentEvent",
    "IssueCommentEvent",
    "PushEvent",
}


def iter_archive(
    path: str | Path, *, repo: str, mgtenant: str = "kafka"
) -> Iterator[EvalEvent]:
    """Stream one hourly ``.json.gz`` archive, yielding only one repository."""

    with gzip.open(path, "rt", encoding="utf-8") as archive:
        for line_number, line in enumerate(archive, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid GH Archive JSON on line {line_number}") from exc
            if not isinstance(raw, Mapping):
                raise ValueError(f"GH Archive line {line_number} is not an object")
            event = parse_event(raw, repo=repo, mgtenant=mgtenant)
            if event is not None:
                yield event


def parse_event(
    raw: Mapping[str, Any], *, repo: str, mgtenant: str = "kafka"
) -> EvalEvent | None:
    """Normalize one supported GitHub event, or return ``None`` when filtered."""

    raw_repo = raw.get("repo")
    repo_name = raw_repo.get("name") if isinstance(raw_repo, Mapping) else None
    if not isinstance(repo_name, str) or repo_name.casefold() != repo.casefold():
        return None
    github_type = raw.get("type")
    if github_type not in _SUPPORTED_TYPES:
        return None
    payload = raw.get("payload")
    if not isinstance(payload, Mapping):
        return None
    created_at = _parse_time(_required_string(raw, "created_at"))
    github_id = str(raw.get("id") or _stable_fragment(raw))
    actor = raw.get("actor")
    actor_login = actor.get("login") if isinstance(actor, Mapping) else None
    common = {
        "repo": repo_name,
        "github_event_id": github_id,
        "actor_login": actor_login,
    }

    if github_type == "PullRequestEvent":
        return _pull_request_event(
            payload, common=common, created_at=created_at, github_id=github_id, mgtenant=mgtenant
        )
    if github_type == "PullRequestReviewEvent":
        return _review_event(
            payload, common=common, created_at=created_at, github_id=github_id, mgtenant=mgtenant
        )
    if github_type == "PullRequestReviewCommentEvent":
        return _review_comment_event(
            payload, common=common, created_at=created_at, github_id=github_id, mgtenant=mgtenant
        )
    if github_type == "IssueCommentEvent":
        return _issue_comment_event(
            payload, common=common, created_at=created_at, github_id=github_id, mgtenant=mgtenant
        )
    return _push_event(
        payload, common=common, created_at=created_at, github_id=github_id, mgtenant=mgtenant
    )


def fetch(
    hour: datetime,
    *,
    destination: str | Path,
    base_url: str = "https://data.gharchive.org",
    timeout_s: float = 120,
    opener: Callable[..., BinaryIO] = urlopen,
) -> Path:
    """Explicitly download one UTC GH Archive hour without parsing it in memory."""

    if hour.tzinfo is None:
        raise ValueError("hour must be timezone-aware")
    utc_hour = hour.astimezone(UTC)
    filename = f"{utc_hour:%Y-%m-%d}-{utc_hour.hour}.json.gz"
    target = Path(destination)
    if target.is_dir() or not target.suffix:
        target = target / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    request = Request(f"{base_url.rstrip('/')}/{filename}", headers={"Accept": "application/gzip"})
    with opener(request, timeout=timeout_s) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)
    return target


def _pull_request_event(
    payload: Mapping[str, Any],
    *,
    common: dict[str, Any],
    created_at: datetime,
    github_id: str,
    mgtenant: str,
) -> EvalEvent | None:
    pull = payload.get("pull_request")
    if not isinstance(pull, Mapping):
        return None
    action = str(payload.get("action") or "")
    if action == "opened":
        suffix = "opened"
    elif action == "closed" and pull.get("merged") is True:
        suffix = "merged"
    elif action == "closed":
        suffix = "closed"
    else:
        return None
    number = payload.get("number") or pull.get("number")
    if number is None:
        return None
    user = pull.get("user")
    email = _email(user)
    title = _optional_string(pull.get("title"))
    body = _optional_string(pull.get("body"))
    head_ref = _nested_string(pull, "head", "ref")
    join_surface = f"{title or ''}\n{head_ref or ''}"
    data = {
        **common,
        "number": int(number),
        "pull_request_number": int(number),
        "pull_request_id": pull.get("id"),
        "action": action,
        "title": title,
        "body": body,
        "author_login": _login(user),
        "author_association": pull.get("author_association"),
        "merged": pull.get("merged") is True,
        "merged_at": pull.get("merged_at"),
        "closed_at": pull.get("closed_at"),
        "merge_commit_sha": pull.get("merge_commit_sha"),
        "base_ref": _nested_string(pull, "base", "ref"),
        "head_ref": head_ref,
        # GH Archive does not promise a payload-level files array.  Preserve an
        # explicitly enriched PR object's files when one exists; code gold
        # otherwise resolves an exact merge-SHA-linked PushEvent.
        "changed_files": _changed_files(pull),
        "issue_keys": extract_issue_keys(join_surface),
        "kip_keys": extract_kip_keys(join_surface),
    }
    return _event(
        github_id=github_id,
        event_type=f"com.github.pull_request.{suffix}",
        subject=pr_subject(number),
        time=created_at,
        mgtenant=mgtenant,
        emails=[email] if email else [],
        data=data,
    )


def _review_event(
    payload: Mapping[str, Any],
    *,
    common: dict[str, Any],
    created_at: datetime,
    github_id: str,
    mgtenant: str,
) -> EvalEvent | None:
    pull = payload.get("pull_request")
    review = payload.get("review")
    if not isinstance(pull, Mapping) or not isinstance(review, Mapping):
        return None
    number = pull.get("number") or payload.get("number")
    if number is None:
        return None
    user = review.get("user")
    email = _email(user)
    data = {
        **common,
        "number": int(number),
        "pull_request_number": int(number),
        "review_id": review.get("id"),
        "action": payload.get("action"),
        "state": review.get("state"),
        "body": review.get("body"),
        "submitted_at": review.get("submitted_at"),
        "commit_id": review.get("commit_id"),
        "author_login": _login(user),
        "author_association": review.get("author_association"),
        "changed_files": _changed_files(payload, review, pull),
    }
    return _event(
        github_id=github_id,
        event_type="com.github.review",
        subject=pr_subject(number),
        time=created_at,
        mgtenant=mgtenant,
        emails=[email] if email else [],
        data=data,
    )


def _review_comment_event(
    payload: Mapping[str, Any],
    *,
    common: dict[str, Any],
    created_at: datetime,
    github_id: str,
    mgtenant: str,
) -> EvalEvent | None:
    pull = payload.get("pull_request")
    comment = payload.get("comment")
    if not isinstance(pull, Mapping) or not isinstance(comment, Mapping):
        return None
    number = pull.get("number") or payload.get("number")
    if number is None:
        return None
    user = comment.get("user")
    email = _email(user)
    data = {
        **common,
        "number": int(number),
        "pull_request_number": int(number),
        "comment_id": comment.get("id"),
        "action": payload.get("action"),
        "body": comment.get("body"),
        "path": comment.get("path"),
        "line": comment.get("line"),
        "position": comment.get("position"),
        "commit_id": comment.get("commit_id"),
        "created_at": comment.get("created_at"),
        "updated_at": comment.get("updated_at"),
        "author_login": _login(user),
        "author_association": comment.get("author_association"),
        "changed_files": _changed_files(payload, comment, pull),
    }
    return _event(
        github_id=github_id,
        event_type="com.github.review_comment",
        subject=pr_subject(number),
        time=created_at,
        mgtenant=mgtenant,
        emails=[email] if email else [],
        data=data,
    )


def _issue_comment_event(
    payload: Mapping[str, Any],
    *,
    common: dict[str, Any],
    created_at: datetime,
    github_id: str,
    mgtenant: str,
) -> EvalEvent | None:
    issue = payload.get("issue")
    comment = payload.get("comment")
    if not isinstance(issue, Mapping) or not isinstance(comment, Mapping):
        return None
    number = issue.get("number")
    if number is None:
        return None
    body = _optional_string(comment.get("body"))
    title = _optional_string(issue.get("title"))
    is_pull_request = isinstance(issue.get("pull_request"), Mapping)
    user = comment.get("user")
    email = _email(user)
    if is_pull_request:
        subject = pr_subject(number)
        pull_number: int | None = int(number)
    else:
        text_keys = extract_text_keys(f"{title or ''}\n{body or ''}")
        subject = text_keys[0] if text_keys else thread_subject(
            f"github-issue-{number}@{common['repo']}"
        )
        pull_number = None
    data = {
        **common,
        "number": int(number),
        "pull_request_number": pull_number,
        "comment_id": comment.get("id"),
        "action": payload.get("action"),
        "title": title,
        "body": body,
        "created_at": comment.get("created_at"),
        "updated_at": comment.get("updated_at"),
        "author_login": _login(user),
        "author_association": comment.get("author_association"),
        "issue_keys": extract_issue_keys(f"{title or ''}\n{body or ''}"),
        "kip_keys": extract_kip_keys(f"{title or ''}\n{body or ''}"),
    }
    return _event(
        github_id=github_id,
        event_type="com.github.issue_comment",
        subject=subject,
        time=created_at,
        mgtenant=mgtenant,
        emails=[email] if email else [],
        data=data,
    )


def _push_event(
    payload: Mapping[str, Any],
    *,
    common: dict[str, Any],
    created_at: datetime,
    github_id: str,
    mgtenant: str,
) -> EvalEvent:
    raw_commits = payload.get("commits", [])
    commits: list[dict[str, Any]] = []
    emails: list[str] = []
    all_text: list[str] = []
    if isinstance(raw_commits, list):
        for raw_commit in raw_commits:
            if not isinstance(raw_commit, Mapping):
                continue
            author = raw_commit.get("author")
            email = _email(author)
            if email:
                emails.append(email)
            message = _optional_string(raw_commit.get("message")) or ""
            all_text.append(message)
            commits.append(
                {
                    "sha": raw_commit.get("sha"),
                    "message": message,
                    "author": contributor_key(email) if email else None,
                    "author_name": _optional_string(
                        author.get("name") if isinstance(author, Mapping) else None
                    ),
                    "changed_files": _changed_files(raw_commit),
                }
            )
    text = "\n".join(all_text)
    references = extract_text_keys(text)
    contributor = contributor_key(emails[0]) if emails else None
    subject = references[0] if references else (
        contributor or thread_subject(f"github-push-{github_id}@{common['repo']}")
    )
    data = {
        **common,
        "ref": payload.get("ref"),
        "before": payload.get("before"),
        "head": payload.get("head"),
        "size": payload.get("size"),
        "distinct_size": payload.get("distinct_size"),
        "commits": commits,
        "changed_files": list(
            dict.fromkeys(file for commit in commits for file in commit["changed_files"])
        ),
        "issue_keys": extract_issue_keys(text),
        "kip_keys": extract_kip_keys(text),
    }
    return _event(
        github_id=github_id,
        event_type="com.github.push",
        subject=subject,
        time=created_at,
        mgtenant=mgtenant,
        emails=emails,
        data=data,
    )


def _event(
    *,
    github_id: str,
    event_type: str,
    subject: str,
    time: datetime,
    mgtenant: str,
    emails: list[str],
    data: dict[str, Any],
) -> EvalEvent:
    return EvalEvent(
        id=f"github:{github_id}",
        source=f"github:{data['repo']}",
        type=event_type,
        subject=subject,
        time=time,
        mgtenant=mgtenant,
        baseline_keys=derive_baseline_keys(author_emails=emails),
        data=data,
    )


def _changed_files(*objects: Mapping[str, Any]) -> list[str]:
    files: list[str] = []
    for value in objects:
        for field in ("changed_files", "files"):
            raw_files = value.get(field)
            if not isinstance(raw_files, list):
                continue
            for item in raw_files:
                if isinstance(item, str):
                    files.append(item)
                elif isinstance(item, Mapping):
                    filename = item.get("filename") or item.get("path") or item.get("name")
                    if isinstance(filename, str):
                        files.append(filename)
    return list(dict.fromkeys(files))


def _email(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    email = value.get("email")
    return email if isinstance(email, str) and "@" in email else None


def _login(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    login = value.get("login")
    return str(login) if login is not None else None


def _nested_string(value: Mapping[str, Any], outer: str, inner: str) -> str | None:
    nested = value.get(outer)
    return _optional_string(nested.get(inner)) if isinstance(nested, Mapping) else None


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"GH Archive event is missing string field {key!r}")
    return item


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _stable_fragment(value: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


__all__ = ["fetch", "iter_archive", "parse_event"]
