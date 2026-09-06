"""Cached GitHub GraphQL extraction for docs/evaluation-spec.md §3.1/§4.1.

Raw responses are immutable snapshots, not historical versions of edited text.
Discovery includes older PRs updated since the lower bound (even after until),
so later updates cannot hide activity in the requested historical window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from harnext_eval.corpus.build_replay import write_replay
from harnext_eval.corpus.gharchive import parse_event
from harnext_eval.corpus.keys import (
    contributor_key,
    extract_issue_keys,
    extract_kip_keys,
    pr_subject,
)
from harnext_eval.types import EvalEvent

Json = dict[str, Any]
PAGE = "pageInfo { hasNextPage endCursor }"
ACTOR = "author { login ... on User { email } } authorAssociation"
COMMENT = f"id {ACTOR} createdAt updatedAt body"
REVIEW_COMMENT = f"{COMMENT} path line position commit {{ oid }}"
REVIEW = (
    f"id {ACTOR} state submittedAt body commit {{ oid }} "
    f"comments(first:20) {{ nodes {{ {REVIEW_COMMENT} }} {PAGE} }}"
)
FIELDS = {
    "files": "path additions deletions",
    "labels": "name",
    "reviews": REVIEW,
    "comments": COMMENT,
}
PR = f"""id number title body {ACTOR} createdAt updatedAt mergedAt closedAt merged
mergedBy {{ login ... on User {{ email }} }} baseRefName headRefName mergeCommit {{ oid }}"""
DISCOVER = """query($owner:String!,$name:String!,$cursor:String) {
repository(owner:$owner,name:$name) {
 pullRequests(first:100,after:$cursor,orderBy:{field:CREATED_AT,direction:ASC}) {
 nodes { number createdAt updatedAt } pageInfo { hasNextPage endCursor }
} } }"""
DETAIL = (
    "query($owner:String!,$name:String!,$number:Int!) { "
    "repository(owner:$owner,name:$name) { pullRequest(number:$number) { "
    + PR
    + " "
    + " ".join(
        f"{field}(first:{20 if field == 'reviews' else 100}) "
        f"{{ nodes {{ {selection} }} {PAGE} }}"
        for field, selection in FIELDS.items()
    )
    + " } } }"
)
BRANCH = """query($owner:String!,$name:String!) {
repository(owner:$owner,name:$name) { defaultBranchRef { name target { oid } } } }"""
COMMITS = """query($owner:String!,$name:String!,$oid:GitObjectID!,$since:GitTimestamp!,
$until:GitTimestamp!,$cursor:String) { repository(owner:$owner,name:$name) {
object(oid:$oid) { ... on Commit { history(first:100,after:$cursor,since:$since,until:$until) {
nodes { oid author { name email user { login } } committedDate messageHeadline
associatedPullRequests(first:10) { nodes { number } } }
pageInfo { hasNextPage endCursor }
} } } } }"""


def timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


class GraphQLClient:
    """urllib transport with bounded retries and injectable clock/IO for tests."""

    def __init__(
        self,
        token: str | None = None,
        *,
        opener: Callable[..., Any] = urlopen,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        retries: int = 6,
    ) -> None:
        self.token = token or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        self.opener, self.sleep, self.clock, self.retries = opener, sleep, clock, retries

    def _pause(self, headers: Mapping[str, str], *, limited: bool = False) -> None:
        normalized = {key.lower(): value for key, value in headers.items()}
        remaining = int(normalized.get("x-ratelimit-remaining", "5000"))
        if remaining < 50:
            # Primary quota exhausted: wait for the window to reset.
            reset = float(normalized.get("x-ratelimit-reset", "0"))
            delay = max(0.0, reset - self.clock() + 1)
            delay = max(delay, float(normalized.get("retry-after", "0")))
            self.sleep(delay if delay > 1 else 1.0)
        elif limited:
            # Secondary (burst) limit with quota left: honour retry-after, else
            # wait for a near reset but never the full hour (that idled parallel
            # fetchers for ~1 h); fall back to one minute.
            retry_after = float(normalized.get("retry-after", "0"))
            reset_delay = float(normalized.get("x-ratelimit-reset", "0")) - self.clock() + 1
            if retry_after > 0:
                self.sleep(retry_after)
            elif reset_delay > 1:
                self.sleep(min(reset_delay, 300.0))
            else:
                self.sleep(60.0)

    def __call__(self, query: str, variables: Json) -> Json:
        if not self.token:
            raise ValueError("live fetch requires GH_TOKEN or GITHUB_TOKEN")
        request = Request(
            "https://api.github.com/graphql",
            data=json.dumps({"query": query, "variables": variables}).encode(),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "User-Agent": "harnext-eval",
            },
        )
        for attempt in range(self.retries + 1):
            try:
                with self.opener(request, timeout=120) as response:
                    payload = json.load(response)
                    self._pause(dict(response.headers))
                if not isinstance(payload, dict):
                    raise ValueError("GraphQL response is not an object")
                errors = payload.get("errors")
                if errors:
                    transient = all(
                        error.get("type") in {"RATE_LIMITED", "RATE_LIMIT", "INTERNAL", "SERVICE_UNAVAILABLE"}
                        or "something went wrong" in str(error.get("message", "")).lower()
                        for error in errors
                    )
                    if transient and attempt < self.retries:
                        self.sleep(
                            max(
                                60 if any(e.get("type") in {"RATE_LIMITED", "RATE_LIMIT"} for e in errors) else 0,
                                2**attempt,
                            )
                        )
                        continue
                    raise ValueError(f"GraphQL errors (response not cached): {errors}")
                if payload.get("data") is None:
                    raise ValueError("GraphQL response has no data")
                return payload
            except HTTPError as exc:
                if exc.code not in {403, 429, 500, 502, 503, 504} or attempt == self.retries:
                    raise
                if exc.code in {403, 429}:
                    self._pause(dict(exc.headers), limited=True)
                else:
                    self.sleep(min(60, 2**attempt))
            except (URLError, TimeoutError, ConnectionError):
                if attempt == self.retries:
                    raise
                self.sleep(min(60, 2**attempt))
        raise RuntimeError("retry loop exhausted")


class PageCache:
    """Query/variable-addressed raw pages; a cache miss offline is an error."""

    def __init__(
        self,
        raw_dir: str | Path,
        *,
        transport: Callable[[str, Json], Json] | None = None,
        parse_only: bool = False,
    ) -> None:
        self.root = Path(raw_dir) / "github"
        self.transport = transport
        self.parse_only = parse_only
        self.cached = self.fetched = 0

    def page(self, label: str, query: str, variables: Json) -> Json:
        digest = hashlib.sha256(json.dumps([query, variables], sort_keys=True).encode()).hexdigest()
        path = self.root / f"{label}-{digest[:24]}.json"
        if path.exists():
            self.cached += 1
            result = json.loads(path.read_text(encoding="utf-8"))
        else:
            if self.parse_only:
                raise FileNotFoundError(f"incomplete cache: {path}")
            result = (self.transport or GraphQLClient())(query, variables)
            if result.get("errors") or result.get("data") is None:
                raise ValueError("refusing to cache an incomplete GraphQL response")
            self.root.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
            self.fetched += 1
        if not isinstance(result, dict) or result.get("errors") or result.get("data") is None:
            raise ValueError(f"invalid cached GraphQL page: {path}")
        return result["data"]


def _next(connection: Json, seen: set[str]) -> str | None:
    info = connection["pageInfo"]
    if not info["hasNextPage"]:
        return None
    cursor = info.get("endCursor")
    if not isinstance(cursor, str) or not cursor or cursor in seen:
        raise ValueError("missing or repeated GraphQL pagination cursor")
    seen.add(cursor)
    return cursor


def _connection(
    cache: PageCache, node: Json, field: str, selection: str, node_type: str, label: str
) -> list[Json]:
    connection = node[field]
    result = list(connection["nodes"])
    seen: set[str] = set()
    while (cursor := _next(connection, seen)) is not None:
        query = (
            "query($id:ID!,$cursor:String) { node(id:$id) { ... on "
            + node_type
            + f" {{ {field}(first:100,after:$cursor) {{ nodes {{ {selection} }} {PAGE} }} }} }} }}"
        )
        connection = cache.page(label, query, {"id": node["id"], "cursor": cursor})["node"][field]
        result.extend(connection["nodes"])
    return result


def iter_prs(cache: PageCache, repo: str, since: datetime, until: datetime) -> Iterator[Json]:
    owner, name = repo.split("/", 1)
    variables: Json = {"owner": owner, "name": name, "cursor": None}
    seen: set[str] = set()
    while True:
        connection = cache.page("prs", DISCOVER, variables)["repository"]["pullRequests"]
        for item in connection["nodes"]:
            if timestamp(item["createdAt"]) >= until:
                return
            if timestamp(item["updatedAt"]) < since:
                continue
            number = item["number"]
            pr = cache.page(
                f"pr-{number}", DETAIL, {"owner": owner, "name": name, "number": number}
            )["repository"]["pullRequest"]
            for field, selection in FIELDS.items():
                pr[field] = _connection(
                    cache, pr, field, selection, "PullRequest", f"pr-{number}-{field}"
                )
            for review in pr["reviews"]:
                review["comments"] = _connection(
                    cache,
                    review,
                    "comments",
                    REVIEW_COMMENT,
                    "PullRequestReview",
                    f"pr-{number}-review-comments",
                )
            yield pr
        variables["cursor"] = _next(connection, seen)
        if variables["cursor"] is None:
            return


def iter_commits(
    cache: PageCache, repo: str, since: datetime, until: datetime
) -> Iterator[tuple[str, Json]]:
    owner, name = repo.split("/", 1)
    base = {"owner": owner, "name": name}
    branch = cache.page("branch", BRANCH, base)["repository"]["defaultBranchRef"]
    if branch is None:
        return
    variables = {
        **base,
        "oid": branch["target"]["oid"],
        "since": since.isoformat(),
        "until": until.isoformat(),
        "cursor": None,
    }
    seen: set[str] = set()
    while True:
        connection = cache.page("commits", COMMITS, variables)["repository"]["object"]["history"]
        for commit in connection["nodes"]:
            if since <= timestamp(commit["committedDate"]) < until:
                yield branch["name"], commit
        variables["cursor"] = _next(connection, seen)
        if variables["cursor"] is None:
            return


def _user(item: Json) -> Json:
    return item.get("author") or {}


def _normalize(
    kind: str,
    payload: Json,
    actor: Json,
    when: str,
    identity: str,
    repo: str,
    association: str | None,
    extra: Json,
) -> EvalEvent:
    event = parse_event(
        {
            "id": f"api:{repo}:{identity}",
            "type": kind,
            "repo": {"name": repo},
            "actor": actor,
            "created_at": when,
            "payload": payload,
        },
        repo=repo,
    )
    if event is None:
        raise ValueError(f"unsupported normalization: {kind}")
    login, email = actor.get("login"), actor.get("email")
    key = (
        contributor_key(email)
        if email and "@" in email
        else (f"contributor:gh:{login.casefold()}" if login else None)
    )
    data = dict(event.data or {})
    data.update(extra)
    data.update(
        {
            "actor": key,
            "author": key,
            "author_association": association,
            "is_committer": association in {"MEMBER", "OWNER"},
        }
    )
    if isinstance(data.get("body"), str):
        data["body"] = data["body"][:4000]
    return event.model_copy(update={"data": data, "baseline_keys": [key] if key else []})


def parse_pr(pr: Json, *, repo: str, since: datetime, until: datetime) -> list[EvalEvent]:
    """Pure GraphQL PR-to-GH-Archive conversion, with explicit enrichments."""
    number = pr["number"]
    surface = "\n".join(str(pr.get(key) or "") for key in ("title", "body", "headRefName"))
    extra = {
        "title": pr.get("title"),
        "issue_keys": extract_issue_keys(surface),
        "kip_keys": extract_kip_keys(surface),
        "base_ref": pr.get("baseRefName"),
        "merged_by": None,
    }
    pull = {
        "id": pr["id"],
        "number": number,
        "title": pr.get("title"),
        "body": pr.get("body"),
        "user": _user(pr),
        "author_association": pr.get("authorAssociation"),
        "head": {"ref": pr.get("headRefName")},
        "base": {"ref": pr.get("baseRefName")},
    }
    events: list[EvalEvent] = []

    def emit(
        kind: str,
        payload: Json,
        actor: Json,
        when: str | None,
        identity: str,
        association: str | None,
        additions: Json | None = None,
    ) -> None:
        if when and since <= timestamp(when) < until:
            event = _normalize(
                kind,
                payload,
                actor,
                when,
                identity,
                repo,
                association,
                {**extra, **(additions or {})},
            )
            events.append(event.model_copy(update={"subject": pr_subject(number)}))

    emit(
        "PullRequestEvent",
        {"action": "opened", "number": number, "pull_request": pull},
        _user(pr),
        pr["createdAt"],
        f"pr-{number}-opened",
        pr.get("authorAssociation"),
        {"merged_by": None},
    )
    terminal = pr.get("mergedAt") or pr.get("closedAt")
    merged = bool(pr.get("mergedAt")) and pr.get("merged") is True
    final_pull = {
        **pull,
        "merged": merged,
        "merged_at": pr.get("mergedAt"),
        "closed_at": pr.get("closedAt"),
        "merge_commit_sha": (pr.get("mergeCommit") or {}).get("oid"),
        "files": pr.get("files", []) if merged else [],
    }
    # GraphQL has no closing actor/merger association on the snapshot. Do not
    # misattribute the close to the PR author or invent MEMBER for the merger.
    emit(
        "PullRequestEvent",
        {"action": "closed", "number": number, "pull_request": final_pull},
        (pr.get("mergedBy") or {}) if merged else {},
        terminal,
        f"pr-{number}-{'merged' if merged else 'closed'}",
        None,
        {
            "merged_by": (pr.get("mergedBy") or {}).get("login") if merged else None,
            "pr_author_association": pr.get("authorAssociation"),
            "labels": [label["name"] for label in pr.get("labels", [])],
        },
    )
    for field, kind in (("reviews", "PullRequestReviewEvent"), ("comments", "IssueCommentEvent")):
        for item in pr.get(field, []):
            review = field == "reviews"
            if review and (item.get("state") == "PENDING" or not item.get("submittedAt")):
                continue
            obj = {
                "id": item["id"],
                "user": _user(item),
                "body": item.get("body"),
                "author_association": item.get("authorAssociation"),
                "created_at": item.get("createdAt"),
                "updated_at": item.get("updatedAt"),
                "state": str(item.get("state", "")).lower(),
                "submitted_at": item.get("submittedAt"),
                "commit_id": (item.get("commit") or {}).get("oid"),
            }
            payload = {
                "action": "submitted" if review else "created",
                "pull_request": {"number": number},
                "issue": {"number": number, "title": pr.get("title"), "pull_request": {}},
                "review" if review else "comment": obj,
            }
            emit(
                kind,
                payload,
                _user(item),
                item.get("submittedAt" if review else "createdAt"),
                f"{'review' if review else 'comment'}-{item['id']}",
                item.get("authorAssociation"),
            )
            if review:
                for comment in item.get("comments", []):
                    raw_comment = {
                        "id": comment["id"],
                        "user": _user(comment),
                        "body": comment.get("body"),
                        "path": comment.get("path"),
                        "line": comment.get("line"),
                        "position": comment.get("position"),
                        "created_at": comment.get("createdAt"),
                        "updated_at": comment.get("updatedAt"),
                        "commit_id": (comment.get("commit") or {}).get("oid"),
                        "author_association": comment.get("authorAssociation"),
                    }
                    emit(
                        "PullRequestReviewCommentEvent",
                        {
                            "action": "created",
                            "pull_request": {"number": number},
                            "comment": raw_comment,
                        },
                        _user(comment),
                        comment.get("createdAt"),
                        f"review-comment-{comment['id']}",
                        comment.get("authorAssociation"),
                    )
    return events


def parse_commit(commit: Json, *, repo: str, branch: str) -> EvalEvent:
    """Represent a trunk commit as one push, retaining GH Archive's subject rule."""
    author = commit.get("author") or {}
    actor = {**author, "login": (author.get("user") or {}).get("login")}
    sha, message = commit["oid"], commit["messageHeadline"]
    return _normalize(
        "PushEvent",
        {
            "ref": f"refs/heads/{branch}",
            "head": sha,
            "size": 1,
            "distinct_size": 1,
            "commits": [{"sha": sha, "message": message, "author": author}],
        },
        actor,
        commit["committedDate"],
        f"commit-{sha}",
        repo,
        None,
        {
            "sha": sha,
            "message": message,
            "branch": branch,
            "associated_pr_numbers": [
                pr["number"] for pr in (commit.get("associatedPullRequests") or {}).get("nodes", [])
            ],
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--since", required=True)
    parser.add_argument("--until", required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parse-only", action="store_true")
    args = parser.parse_args(argv)
    since, until = timestamp(args.since), timestamp(args.until)
    if since >= until or len(args.repo.split("/")) != 2:
        parser.error("require owner/repo and since < until")
    cache = PageCache(args.raw_dir, parse_only=args.parse_only)
    events: list[EvalEvent] = []
    prs = 0
    for pr in iter_prs(cache, args.repo, since, until):
        prs += 1
        events.extend(parse_pr(pr, repo=args.repo, since=since, until=until))
    events.extend(
        parse_commit(commit, repo=args.repo, branch=branch)
        for branch, commit in iter_commits(cache, args.repo, since, until)
    )
    events = sorted({event.id: event for event in events}.values(), key=lambda e: (e.time, e.id))
    artifact = write_replay(events, args.output)
    roster: dict[str, set[str]] = defaultdict(set)
    for event in events:
        data = event.data or {}
        if data.get("author_association") in {"MEMBER", "OWNER"} and data.get("actor_login"):
            roster[data["author_association"]].add(data["actor_login"])
    print(
        json.dumps(
            {
                "prs": prs,
                "events": artifact.event_count,
                "events_by_type": dict(Counter(e.type for e in events)),
                "cached_pages": cache.cached,
                "fetched_pages": cache.fetched,
                "roster": {role: len(roster[role]) for role in ("MEMBER", "OWNER")},
                "output": str(artifact.path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
