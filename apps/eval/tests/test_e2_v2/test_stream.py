"""Stream rendering on a tiny synthetic replay: completeness, order, leak safety, checkpoints."""

import json

from harnext_eval.e2.v2 import stream

ISSUES = {
    "KAFKA-1": {
        "id": "100",
        "key": "KAFKA-1",
        "fields": {
            "summary": "FINAL summary",
            "description": "FINAL description",
            "status": {"name": "Resolved"},
            "priority": {"name": "Major"},
            "assignee": {"displayName": "Future Owner"},
            "components": [{"name": "core"}, {"name": "latecomer"}],
            "labels": [],
            "created": "2023-01-05T10:00:00.000+0000",
        },
        "changelog": {
            "startAt": 0,
            "total": 3,
            "histories": [
                {
                    "id": "11",
                    "created": "2023-01-06T09:00:00.500+0000",
                    "items": [
                        {
                            "field": "assignee",
                            "from": "a",
                            "fromString": None,
                            "to": "b",
                            "toString": "Future Owner",
                        }
                    ],
                },
                {
                    "id": "12",
                    "created": "2023-03-01T09:00:00.000+0000",
                    "items": [
                        {
                            "field": "summary",
                            "fromString": "ORIGINAL summary",
                            "toString": "FINAL summary",
                        },
                        {
                            "field": "description",
                            "fromString": "ORIGINAL description",
                            "toString": "FINAL description",
                        },
                        {"field": "Component", "fromString": None, "toString": "latecomer"},
                    ],
                },
                {
                    "id": "13",
                    "created": "2023-03-02T09:00:00.000+0000",
                    "items": [{"field": "status", "fromString": "Open", "toString": "Resolved"}],
                },
            ],
        },
    }
}


def cloud(id_, time, type_, data, subject="x"):
    return {"id": id_, "time": time, "type": type_, "source": "s", "subject": subject, "data": data}


def jira(extra):
    # Replay payloads carry harvest-time state on every JIRA event.
    final = {
        "summary": "FINAL summary",
        "description": "FINAL description",
        "components": ["latecomer"],
    }
    return {"issue_key": "KAFKA-1", **final, **extra}


def pull(extra):
    base = {"number": 7, "title": "KAFKA-1: fix", "issue_keys": ["KAFKA-1"], "kip_keys": ["KIP-9"]}
    return {**base, "base_ref": "trunk", **extra}


EVENTS = [
    cloud(
        "old",
        "2022-12-31T23:59:59Z",
        "com.github.issue_comment",
        pull({"author_login": "x", "body": "old"}),
    ),
    cloud(
        "jira:100:created",
        "2023-01-05T10:00:00Z",
        "org.apache.jira.issue.created",
        jira(
            {
                "creator_name": "Rita",
                "status": "Resolved",
                "assignee": "jira-user:b",
                "priority": "Major",
            }
        ),
    ),
    cloud(
        "jira:100:change:11:0",
        "2023-01-06T09:00:00.500000Z",
        "org.apache.jira.issue.transition",
        jira(
            {
                "actor_name": "Rita",
                "changelog_id": "11",
                "field": "assignee",
                "from": None,
                "to": "jira-user:b",
            }
        ),
    ),
    cloud(
        "jira:100:change:20:0",
        "2023-01-06T09:00:01Z",
        "org.apache.jira.issue.transition",
        jira(
            {
                "actor_name": "Rita",
                "changelog_id": "20",
                "field": "Workflow",
                "from": "a",
                "to": "b",
            }
        ),
    ),
    cloud(
        "jira:100:change:21:0",
        "2023-01-07T09:00:00Z",
        "org.apache.jira.issue.transition",
        jira(
            {
                "actor_name": "ASF GitHub Bot",
                "changelog_id": "21",
                "field": "RemoteIssueLink",
                "from": None,
                "to": 'This issue links to "GitHub Pull Request #7 (Web Link)"',
            }
        ),
    ),
    cloud(
        "jira:100:change:22:0",
        "2023-01-07T09:05:00Z",
        "org.apache.jira.issue.transition",
        jira(
            {
                "actor_name": "ASF GitHub Bot",
                "changelog_id": "22",
                "field": "RemoteIssueLink",
                "from": 'This issue links to "GitHub Pull Request #7 (Web Link)"',
                "to": 'This issue links to "GitHub Pull Request #7 (Web Link)"',
            }
        ),
    ),
    cloud(
        "jira:100:comment:1",
        "2023-01-07T10:00:00Z",
        "org.apache.jira.issue.comment",
        jira({"author_name": "ASF GitHub Bot", "author_account_id": "githubbot", "body": "mirror"}),
    ),
    cloud(
        "jira:100:comment:2",
        "2023-01-07T11:00:00Z",
        "org.apache.jira.issue.comment",
        jira(
            {
                "author_name": "Rita",
                "body": "I can reproduce this.",
                "updated": "2023-01-07T11:00:00.000+0000",
            }
        ),
    ),
    cloud(
        "gh:pr-7-opened",
        "2023-01-07T08:59:00Z",
        "com.github.pull_request.opened",
        pull(
            {
                "author_login": "dev",
                "head_ref": "fix",
                "merged": True,
                "merged_by": "LEAKED-MERGER",
                "merged_at": "2023-01-09T00:00:00Z",
                "closed_at": "2023-01-09T00:00:00Z",
                "merge_commit_sha": "LEAKEDSHA",
                "labels": ["LEAKED-LABEL"],
                "updated_at": "2024-01-01T00:00:00Z",
                "body": "Fixes it.\r\n\r\n### Committer Checklist (excluded from commit message)\r\n- [ ] Verify design",
            }
        ),
    ),
    cloud(
        "gh:review-1",
        "2023-01-08T08:00:00Z",
        "com.github.review",
        pull({"author_login": "rev", "state": "commented", "body": ""}),
    ),
    cloud(
        "gh:review-2",
        "2023-01-08T08:00:00Z",
        "com.github.review",
        pull(
            {"author_login": "rev", "state": "approved", "body": None, "merged_by": "LEAKED-MERGER"}
        ),
    ),
    cloud(
        "gh:comment-1",
        "2023-01-08T08:30:00Z",
        "com.github.issue_comment",
        pull({"author_login": "github-actions", "body": "This PR is being marked as stale"}),
    ),
    cloud(
        "gh:pr-7-merged",
        "2023-01-09T00:00:00Z",
        "com.github.pull_request.merged",
        pull(
            {
                "author_login": "dev",
                "merged_by": "boss",
                "merge_commit_sha": "abc123",
                "body": "Fixes it.",
                "changed_files": ["core/A.java", "core/ATest.java"],
            }
        ),
    ),
    cloud(
        "gh:commit-abc123",
        "2023-01-08T23:59:59Z",
        "com.github.push",
        {
            "branch": "trunk",
            "sha": "abc123",
            "message": "KAFKA-1: fix (#7)\n\nReviewers: rev",
            "associated_pr_numbers": [7],
            "issue_keys": ["KAFKA-1"],
            "actor_login": "boss",
            "commits": [
                {
                    "author_name": "Dev Eloper",
                    "message": "KAFKA-1: fix (#7)",
                    "changed_files": ["core/A.java"],
                }
            ],
        },
    ),
    cloud(
        "mail:1",
        "2023-01-10T00:00:00Z",
        "org.apache.mail.message",
        {
            "author_name": "Colin",
            "author_email": "c@apache.org",
            "list": "dev@kafka.apache.org",
            "subject": "[DISCUSS] KIP-9: things\n (KAFKA-1)",
            "thread_root": "root@x",
            "message_id": "m1",
            "body": "Hello\n> q1\n> q2\n> q3\n> q4\n> q5\nBye",
        },
    ),
    cloud(
        "mail:2",
        "2023-01-10T00:01:00Z",
        "org.apache.mail.message",
        {
            "author_name": "Rita (Jira)",
            "author_email": "jira@apache.org",
            "subject": "[jira] x",
            "thread_root": "j",
            "body": "",
        },
    ),
    cloud(
        "mail:3",
        "2023-01-10T00:02:00Z",
        "org.apache.mail.message",
        {
            "author_name": "Apache Jenkins Server",
            "author_email": "jenkins@builds.apache.org",
            "subject": "Build",
            "thread_root": "b",
            "body": "",
        },
    ),
    cloud(
        "late",
        "2026-07-01T00:00:00Z",
        "com.github.issue_comment",
        pull({"author_login": "x", "body": "late"}),
    ),
]


def built(tmp_path, scope="full"):
    replay = tmp_path / "replay.jsonl"
    replay.write_text("".join(json.dumps(e) + "\n" for e in reversed(EVENTS)))
    path = stream.build(scope, replay=replay, out=tmp_path / scope, issues=ISSUES)
    events = {e["id"]: e for e in stream.read_events(path)}
    return path, events, json.loads((path.parent / "stream-manifest.json").read_text())


def test_window_order_and_schema(tmp_path):
    path, events, manifest = built(tmp_path)
    rows = list(stream.read_events(path))
    assert [(e["time"], e["id"]) for e in rows] == sorted((e["time"], e["id"]) for e in rows)
    assert "old" in events and "late" not in events  # from the first event, up to 2026-06-30
    assert all(
        set(e) == {"id", "time", "type", "entities", "actor", "title", "text", "meta"} for e in rows
    )
    assert all(e["time"].endswith("Z") and len(e["time"]) == 20 for e in rows)
    assert manifest["events"] == len(rows) == sum(manifest["types"].values())
    assert manifest["months"] == {"2022-12": 1, "2023-01": len(rows) - 1}
    assert manifest["text_bytes"] == sum(len(e["text"]) for e in rows) and manifest["complete"]
    assert manifest["events_sha256"] == stream.hashlib.sha256(path.read_bytes()).hexdigest()


def test_nothing_is_filtered_or_trimmed(tmp_path):
    _, events, manifest = built(tmp_path)
    assert set(events) == {e["id"] for e in EVENTS} - {"late"} and manifest["filters"] == []
    assert "Workflow" in events["jira:100:change:20:0"]["title"]  # untracked fields too
    assert events["jira:100:comment:1"]["actor"] == "ASF GitHub Bot"
    assert "marked as stale" in events["gh:comment-1"]["text"]
    assert "- state: commented" in events["gh:review-1"]["text"]  # bodiless review
    assert events["mail:3"]["actor"] == "Apache Jenkins Server"
    assert "Committer Checklist" in events["gh:pr-7-opened"]["text"]
    assert "> q5" in events["mail:1"]["text"]
    link = events["jira:100:change:21:0"]
    assert link["text"].count("\n") == 1 and "pr:7" in link["entities"]


def test_no_future_information(tmp_path):
    _, events, _ = built(tmp_path)
    before_march = [e for e in events.values() if e["time"] < "2023-03-01"]
    for event in before_march:
        blob = json.dumps(event)
        assert "FINAL" not in blob and "LEAK" not in blob and "latecomer" not in blob, event["id"]
    created = events["jira:100:created"]
    assert "ORIGINAL summary" in created["title"] and "ORIGINAL description" in created["text"]
    assert "- status: Open" in created["text"] and "assignee" not in created["text"]
    assert created["entities"] == ["issue:KAFKA-1", "component:core"]
    opened = events["gh:pr-7-opened"]["text"]
    assert "Fixes it." in opened
    assert "2024-01-01" not in opened and "2023-01-09" not in opened


def test_renders_are_readable(tmp_path):
    _, events, _ = built(tmp_path)
    change = events["jira:100:change:11:0"]
    assert change["time"] == "2023-01-06T09:00:00Z"
    assert "2023-01-06T09:00:00.500000Z" in change["text"] and "Future Owner" in change["text"]
    assert change["meta"]["to"] == "Future Owner"
    merged = events["gh:pr-7-merged"]
    assert "`core/ATest.java`" in merged["text"] and merged["actor"] == "boss"
    assert merged["entities"] == ["pr:7", "issue:KAFKA-1", "kip:KIP-9"]
    push = events["gh:commit-abc123"]
    assert "Reviewers: rev" in push["text"] and "`core/A.java`" in push["text"]
    mail = events["mail:1"]
    assert mail["entities"] == ["thread:root@x", "issue:KAFKA-1", "kip:KIP-9"]
    assert "Bye" in mail["text"]
    assert not any(e["text"].lstrip().startswith("{") for e in events.values())


def test_read_events_until_is_exclusive(tmp_path):
    path, _, _ = built(tmp_path)
    ids = [e["id"] for e in stream.read_events(path, until="2023-01-07T09:00:00Z")]
    assert ids[-1] == "gh:pr-7-opened" and "jira:100:change:21:0" not in ids


def test_smoke_focus(tmp_path):
    events = [
        {"id": str(i), "type": "gh.review", "entities": [f"pr:{min(i, 3)}"], "meta": {}}
        for i in range(8)
    ]
    kept, chosen = stream.focus(events, top=1)
    assert chosen == ["pr:3"] and [e["id"] for e in kept] == ["3", "4", "5", "6", "7"]
    _, _, manifest = built(tmp_path, "smoke")
    assert manifest["window"]["until_exclusive"] == "2023-02-15T00:00:00Z" and "focus" in manifest


def test_checkpoints():
    all_ = stream.checkpoints()
    assert all_[0] == "2023-01" and all_[-1] == "2026-06" and len(all_) == 42
    assert stream.checkpoints("2023-01", "2023-02") == ["2023-01", "2023-02"]
    assert stream.boundary("2023-12") == "2024-01-01T00:00:00Z"
    assert stream.boundary("2023-06") == "2023-07-01T00:00:00Z"
    assert stream.checkpoint_for("2023-01-31T23:59:59Z") is None
    assert stream.checkpoint_for("2023-02-01T00:00:00Z") == "2023-01"
    assert stream.checkpoint_for("2024-03-15T12:00:00Z") == "2024-02"
    assert stream.checkpoint_for("2030-01-01T00:00:00Z") == "2026-06"
    assert stream.checkpoint_for("2023-02-10T00:00:00Z", "2023-01", "2023-02") == "2023-01"
