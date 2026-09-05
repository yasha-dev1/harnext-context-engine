"""Offline GitHub API contract tests for docs/evaluation-spec.md §3.1/§4.1."""

from __future__ import annotations

import copy
import io
import json
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import pytest
from harnext_eval.corpus.gharchive import parse_event
from harnext_eval.corpus.github_api import (
    BRANCH,
    COMMITS,
    DETAIL,
    DISCOVER,
    GraphQLClient,
    PageCache,
    iter_commits,
    iter_prs,
    main,
    parse_commit,
    parse_pr,
    timestamp,
)
from harnext_eval.corpus.keys import contributor_key

FIXTURES = Path(__file__).parent / "fixtures"
SINCE, UNTIL = timestamp("2020-01-01"), timestamp("2021-01-01")
REPO = "apache/kafka"


def fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"github-{name}.json").read_text())


def connection(nodes: list[Any], cursor: str | None = None) -> dict[str, Any]:
    return {"nodes": nodes, "pageInfo": {"hasNextPage": cursor is not None, "endCursor": cursor}}


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail_at: int | None = None

    def __call__(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((query, copy.deepcopy(variables)))
        if len(self.calls) == self.fail_at:
            raise URLError("interrupted")
        if query == DISCOVER:
            if variables["cursor"] is None:
                items = [{"number": 1, "createdAt": "2010-01-01", "updatedAt": "2011-01-01"}]
                data = {"repository": {"pullRequests": connection(items, "prs/one")}}
            else:
                assert variables["cursor"] == "prs/one"
                data = {"repository": {"pullRequests": connection([fixture("pr")])}}
        elif query == DETAIL:
            assert variables["number"] == 7
            data = {"repository": {"pullRequest": fixture("pr")}}
        elif "files(first:100,after:$cursor)" in query:
            assert variables == {"id": "PR_7", "cursor": "files/one"}
            data = {
                "node": {
                    "files": connection(
                        [
                            {"path": f"file-{i}.py", "additions": i, "deletions": 0}
                            for i in range(100)
                        ]
                    )
                }
            }
        elif "... on PullRequestReview " in query:
            assert variables == {"id": "R_1", "cursor": "rc/one"}
            comment = fixture("pr")["reviews"]["nodes"][0]["comments"]["nodes"][0]
            comment.update(id="RC_2", body="second page")
            data = {"node": {"comments": connection([comment])}}
        elif query == BRANCH:
            data = {"repository": {"defaultBranchRef": {"name": "trunk", "target": {"oid": "tip"}}}}
        elif query == COMMITS:
            assert variables["oid"] == "tip"
            if variables["cursor"] is None:
                commits = connection([fixture("commit")], "commit/one")
            else:
                assert variables["cursor"] == "commit/one"
                boundary = fixture("commit")
                boundary.update(oid="boundary", committedDate="2021-01-01T00:00:00Z")
                commits = connection([boundary])
            data = {"repository": {"object": {"history": commits}}}
        else:
            raise AssertionError(query)
        return {"data": data}


def load_pr(tmp_path: Path) -> dict[str, Any]:
    return list(iter_prs(PageCache(tmp_path, transport=FakeTransport()), REPO, SINCE, UNTIL))[0]


def test_pagination_resume_and_offline_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    transport = FakeTransport()
    transport.fail_at = 4
    cache = PageCache(tmp_path, transport=transport)
    with pytest.raises(URLError):
        list(iter_prs(cache, REPO, SINCE, UNTIL))
    assert cache.fetched == 3
    transport.fail_at = None
    resumed = PageCache(tmp_path, transport=transport)
    prs = list(iter_prs(resumed, REPO, SINCE, UNTIL))
    commits = list(iter_commits(resumed, REPO, SINCE, UNTIL))
    assert resumed.cached == 3
    assert resumed.fetched == 5
    assert len(prs) == len(commits) == 1
    assert len(prs[0]["files"]) == 101
    assert len(prs[0]["reviews"][0]["comments"]) == 2
    assert commits[0][0] == "trunk"
    offline = PageCache(tmp_path, parse_only=True)
    assert list(iter_prs(offline, REPO, SINCE, UNTIL)) == prs
    assert list(iter_commits(offline, REPO, SINCE, UNTIL)) == commits
    assert offline.cached == 8 and offline.fetched == 0
    output = tmp_path / "events.jsonl"
    assert (
        main(
            [
                "--repo",
                REPO,
                "--since",
                "2020-01-01",
                "--until",
                "2021-01-01",
                "--raw-dir",
                str(tmp_path),
                "--output",
                str(output),
                "--parse-only",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["prs"] == 1 and report["events"] == 7
    assert report["roster"] == {"MEMBER": 1, "OWNER": 1}
    assert report["fetched_pages"] == 0
    assert output.with_suffix(".jsonl.sha256").exists()
    first = output.read_bytes()
    main(
        [
            "--repo",
            REPO,
            "--since",
            "2020-01-01",
            "--until",
            "2021-01-01",
            "--raw-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--parse-only",
        ]
    )
    assert output.read_bytes() == first


def test_cache_missing_invalid_and_scope(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="incomplete cache"):
        PageCache(tmp_path, parse_only=True).page("test", "query", {})
    cache = PageCache(tmp_path, transport=lambda q, v: {"data": {"ok": True}})
    cache.page("test", "query", {"repo": "one"})
    with pytest.raises(FileNotFoundError):
        PageCache(tmp_path, parse_only=True).page("test", "query", {"repo": "two"})
    cache.transport = lambda q, v: {"data": {"partial": True}, "errors": [{"message": "oops"}]}
    with pytest.raises(ValueError, match="incomplete"):
        cache.page("error", "query", {})
    assert not list((tmp_path / "github").glob("error-*.json"))


def test_repeated_cursor_rejected(tmp_path: Path) -> None:
    def transport(query: str, variables: dict[str, Any]) -> dict[str, Any]:
        return {"data": {"repository": {"pullRequests": connection([], "repeat")}}}

    with pytest.raises(ValueError, match="repeated"):
        list(iter_prs(PageCache(tmp_path, transport=transport), REPO, SINCE, UNTIL))


def test_events_keys_truncation_and_boundaries(tmp_path: Path) -> None:
    pr = load_pr(tmp_path)
    pr["comments"][0]["body"] = "x" * 4500
    pr["reviews"][0]["body"] = "y" * 4500
    pr["reviews"][0]["comments"][0]["body"] = "z" * 4500
    events = parse_pr(pr, repo=REPO, since=SINCE, until=UNTIL)
    by_type = {e.type.removeprefix("com.github."): e for e in events}
    assert set(by_type) == {
        "pull_request.opened",
        "pull_request.merged",
        "review",
        "review_comment",
        "issue_comment",
    }
    for event in events:
        assert event.subject == "pr:7" and event.source == f"github:{REPO}"
        data = event.data or {}
        assert data["issue_keys"] == ["KAFKA-123", "KAFKA-456", "KAFKA-789"]
        assert data["kip_keys"] == ["KIP-42", "KIP-43"]
        assert data["number"] == data["pull_request_number"] == 7
        assert len(data.get("body") or "") <= 4000
        assert not any(key.startswith("component:") for key in event.baseline_keys)
    opened = by_type["pull_request.opened"]
    assert opened.baseline_keys == [contributor_key("alice@example.org")]
    assert (opened.data or {})["is_committer"] is True
    assert (opened.data or {})["merged"] is False
    assert (opened.data or {})["merged_at"] is None
    assert (opened.data or {})["changed_files"] == []
    assert len((by_type["pull_request.merged"].data or {})["changed_files"]) == 101
    assert (by_type["pull_request.merged"].data or {})["merged_by"] == "bob"
    assert by_type["review"].baseline_keys == ["contributor:gh:bob"]
    assert (by_type["review"].data or {})["state"] == "approved"
    assert len((by_type["issue_comment"].data or {})["body"]) == 4000
    assert all(
        e.time < timestamp("2020-01-03")
        for e in parse_pr(pr, repo=REPO, since=SINCE, until=timestamp("2020-01-03"))
    )
    pr.update(merged=False, mergedAt=None, mergedBy=None)
    closed = parse_pr(pr, repo=REPO, since=SINCE, until=UNTIL)[1]
    assert closed.type == "com.github.pull_request.closed"
    assert closed.baseline_keys == []
    assert (closed.data or {})["actor_login"] is None


@pytest.mark.parametrize(
    "suffix", ["opened", "merged", "closed", "review", "review_comment", "issue_comment", "push"]
)
def test_exact_archive_contract(tmp_path: Path, suffix: str) -> None:
    pr = load_pr(tmp_path)
    # Keep keys identical to the GH Archive join surface for this comparison.
    pr.update(body="plain body", headRefName="branch")
    if suffix == "closed":
        pr.update(merged=False, mergedAt=None, mergedBy=None)
    if suffix == "push":
        actual = parse_commit(fixture("commit"), repo=REPO, branch="trunk")
        raw_type = "PushEvent"
        payload = {
            "ref": "refs/heads/trunk",
            "head": "abc",
            "size": 1,
            "distinct_size": 1,
            "commits": [
                {
                    "sha": "abc",
                    "message": "Revert KAFKA-123 (#7)",
                    "author": {"name": "Alice", "email": "alice@example.org"},
                }
            ],
        }
    else:
        events = parse_pr(pr, repo=REPO, since=SINCE, until=UNTIL)
        actual = next(e for e in events if e.type.endswith(f".{suffix}"))
        data = actual.data or {}
        if suffix in {"opened", "merged", "closed"}:
            raw_type = "PullRequestEvent"
            pull = {
                "id": "PR_7",
                "number": 7,
                "title": pr["title"],
                "body": "plain body",
                "user": pr["author"],
                "author_association": data["author_association"],
                "head": {"ref": "branch"},
                "base": {"ref": "trunk"},
                "merged": suffix == "merged",
                "merged_at": data["merged_at"],
                "closed_at": data["closed_at"],
                "merge_commit_sha": data["merge_commit_sha"],
                "files": data["changed_files"],
            }
            payload = {"action": data["action"], "number": 7, "pull_request": pull}
        else:
            raw_type = {
                "review": "PullRequestReviewEvent",
                "review_comment": "PullRequestReviewCommentEvent",
                "issue_comment": "IssueCommentEvent",
            }[suffix]
            obj = {
                key: data.get(key)
                for key in (
                    "body",
                    "state",
                    "submitted_at",
                    "commit_id",
                    "created_at",
                    "updated_at",
                    "path",
                    "line",
                    "position",
                    "author_association",
                )
            }
            obj.update(
                id=data.get("review_id", data.get("comment_id")),
                user={"login": data["author_login"]},
            )
            payload = {
                "action": data["action"],
                "pull_request": {"number": 7},
                "issue": {"number": 7, "title": pr["title"], "pull_request": {}},
                "review" if suffix == "review" else "comment": obj,
            }
    data = actual.data or {}
    expected = parse_event(
        {
            "id": data["github_event_id"],
            "type": raw_type,
            "repo": {"name": REPO},
            "actor": {"login": data["actor_login"]},
            "created_at": actual.time.isoformat(),
            "payload": payload,
        },
        repo=REPO,
    )
    assert expected is not None
    assert (actual.id, actual.type, actual.source, actual.subject, actual.time) == (
        expected.id,
        expected.type,
        expected.source,
        expected.subject,
        expected.time,
    )
    # Every GH Archive field is preserved; explicitly requested enrichment is additive.
    for key, value in (expected.data or {}).items():
        assert data[key] == value, key


class Response(io.BytesIO):
    def __init__(self, payload: dict[str, Any], headers: dict[str, str] | None = None) -> None:
        super().__init__(json.dumps(payload).encode())
        self.headers = headers or {}


@pytest.mark.parametrize("code", [403, 429, 502])
def test_http_retries_and_rate_headers(code: int) -> None:
    sleeps: list[float] = []
    calls = 0
    headers = Message()
    headers["X-RateLimit-Reset"] = "120"

    def opener(request: Any, **kwargs: Any) -> Response:
        nonlocal calls
        calls += 1
        assert request.get_header("Authorization") == "Bearer secret"
        assert json.loads(request.data)["variables"] == {"cursor": None}
        if calls == 1:
            raise HTTPError(request.full_url, code, "retry", headers, None)
        return Response(
            {"data": {"ok": True}}, {"X-RateLimit-Remaining": "49", "X-RateLimit-Reset": "140"}
        )

    client = GraphQLClient("secret", opener=opener, sleep=sleeps.append, clock=lambda: 100)
    assert client("query", {"cursor": None}) == {"data": {"ok": True}}
    assert sleeps == ([21, 41] if code != 502 else [1, 41])


def test_graphql_errors_and_network_retry() -> None:
    sleeps: list[float] = []
    answers: list[Any] = [
        URLError("offline"),
        {"errors": [{"type": "RATE_LIMITED", "message": "limit"}]},
        {"data": {"ok": True}},
    ]

    def opener(*args: Any, **kwargs: Any) -> Response:
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return Response(answer)

    client = GraphQLClient("token", opener=opener, sleep=sleeps.append)
    assert client("query", {}) == {"data": {"ok": True}}
    assert sleeps == [1, 60]
    client.opener = lambda *a, **kw: Response({"errors": [{"type": "NOT_FOUND"}]})
    with pytest.raises(ValueError, match="GraphQL errors"):
        client("query", {})


def test_token_fallback_and_retry_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "fallback")
    assert GraphQLClient().token == "fallback"
    monkeypatch.setenv("GH_TOKEN", "preferred")
    assert GraphQLClient().token == "preferred"
    calls = 0

    def opener(*args: Any, **kwargs: Any) -> Response:
        nonlocal calls
        calls += 1
        raise URLError("failure")

    with pytest.raises(URLError):
        GraphQLClient("token", opener=opener, retries=2, sleep=lambda _: None)("query", {})
    assert calls == 3


@pytest.mark.parametrize("field", ["reviews", "comments", "labels"])
def test_other_pr_connections_paginate(tmp_path: Path, field: str) -> None:
    fake = FakeTransport()

    def transport(query: str, variables: dict[str, Any]) -> dict[str, Any]:
        if query == DETAIL:
            result = fake(query, variables)
            result["data"]["repository"]["pullRequest"][field]["pageInfo"] = {
                "hasNextPage": True,
                "endCursor": "extra/page",
            }
            return result
        if variables == {"id": "PR_7", "cursor": "extra/page"}:
            assert f"{field}(first:100,after:$cursor)" in query
            item = fixture("pr")[field]["nodes"][0]
            if field == "labels":
                item["name"] = "second label"
            else:
                item["id"] = "second item"
                if field == "reviews":
                    item["comments"] = connection([])
            return {"data": {"node": {field: connection([item])}}}
        return fake(query, variables)

    prs = list(iter_prs(PageCache(tmp_path, transport=transport), REPO, SINCE, UNTIL))
    assert len(prs[0][field]) == 2


def test_discovery_includes_old_pr_updated_after_until(tmp_path: Path) -> None:
    fake = FakeTransport()

    def transport(query: str, variables: dict[str, Any]) -> dict[str, Any]:
        if query == DISCOVER:
            return {
                "data": {
                    "repository": {
                        "pullRequests": connection(
                            [
                                {"number": 7, "createdAt": "2018-01-01", "updatedAt": "2022-01-01"},
                                {"number": 8, "createdAt": "2021-01-01", "updatedAt": "2022-01-01"},
                            ]
                        )
                    }
                }
            }
        result = fake(query, variables)
        if query == DETAIL:
            result["data"]["repository"]["pullRequest"]["createdAt"] = "2018-01-01"
        return result

    cache = PageCache(tmp_path, transport=transport)
    prs = list(iter_prs(cache, REPO, SINCE, UNTIL))
    assert len(prs) == 1 and prs[0]["number"] == 7
    events = parse_pr(prs[0], repo=REPO, since=SINCE, until=UNTIL)
    assert any(e.type.endswith(".merged") for e in events)
    assert not any(e.type.endswith(".opened") for e in events)


def test_pending_review_and_deleted_author(tmp_path: Path) -> None:
    pr = load_pr(tmp_path)
    pr["author"] = None
    pr["reviews"][0].update(submittedAt=None, state="PENDING")
    events = parse_pr(pr, repo=REPO, since=SINCE, until=UNTIL)
    opened = next(e for e in events if e.type.endswith(".opened"))
    assert opened.baseline_keys == []
    assert not any(e.type == "com.github.review" for e in events)
    assert not any(e.type == "com.github.review_comment" for e in events)
