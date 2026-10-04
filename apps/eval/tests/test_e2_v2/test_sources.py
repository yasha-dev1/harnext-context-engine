"""Arm S-tools on a synthetic stream that crosses one checkpoint boundary: as-of state,
physical leak safety of the earlier snapshot, the three CLIs, resume, and the audit oracle."""

import json
import sqlite3
import zlib

import pytest
from harnext_eval.e2.v2 import runner, stream
from harnext_eval.e2.v2.agents import base, container
from harnext_eval.e2.v2.sources import audit, cli, index
from harnext_eval.e2.v2.stores import build as stores_build

LATER = "SENTINEL-ONLY-AFTER-THE-BOUNDARY"
LONG = "The log loader must move to the storage module. " * 12  # long enough to be compressed


def change(history, at, field, before, after, index_=0):
    return {"id": str(history), "created": at, "items": [{"field": "x"}] * index_ + [{"field": field, "fromString": before, "toString": after}]}  # fmt: skip


ISSUES = {
    "KAFKA-1": {
        "id": "100", "key": "KAFKA-1",
        "fields": {
            "summary": f"Final title {LATER}", "description": f"final text {LATER}",
            "status": {"name": "Closed"}, "priority": {"name": "Blocker"},
            "assignee": {"displayName": "Future Owner"}, "resolution": {"name": "Fixed"},
            "components": [{"name": "core"}], "labels": [], "created": "2023-01-05T10:00:00.000+0000",
        },
        "changelog": {"histories": [
            change(11, "2023-01-06T09:00:00.051+0000", "assignee", None, "Rita Skeeter"),
            change(12, "2023-01-07T09:00:00.000+0000", "status", "Open", "In Progress"),
            change(13, "2023-02-03T09:00:00.250+0000", "status", "In Progress", "Closed"),
            change(14, "2023-02-03T09:00:00.250+0000", "resolution", None, "Fixed"),
            change(15, "2023-02-04T09:00:00.000+0000", "summary", "Move LogLoader to storage", f"Final title {LATER}"),
            change(16, "2023-02-04T09:00:01.000+0000", "description", LONG, f"final text {LATER}"),
            change(17, "2023-02-05T09:00:00.000+0000", "assignee", "Rita Skeeter", "Future Owner"),
            change(18, "2023-02-05T09:00:00.000+0000", "priority", "Major", "Blocker"),
        ]},
    }
}  # fmt: skip


def cloud(id_, time, kind, data):
    return {"id": id_, "time": time, "type": kind, "source": "s", "subject": "x", "data": data}


def moved(history, time, field, before, after):
    data = {"issue_key": "KAFKA-1", "actor_name": "Rita Skeeter", "changelog_id": str(history), "field": field, "from": before, "to": after}  # fmt: skip
    return cloud(f"jira:100:change:{history}:0", time, "org.apache.jira.issue.transition", data)


def pull(number, **extra):
    return {"number": number, "title": f"KAFKA-1: part {number}", "issue_keys": ["KAFKA-1"], "kip_keys": [], "base_ref": "trunk", **extra}  # fmt: skip


def letter(id_, time, subject, body, thread="t1"):
    data = {"author_name": "Colin", "list": "dev@kafka.apache.org", "subject": subject, "thread_root": thread, "message_id": f"<{id_}@x>", "body": body}  # fmt: skip
    return cloud(f"mail:{id_}", time, "org.apache.mail.message", data)


REPLAY = [
    cloud("jira:100:created", "2023-01-05T10:00:00Z", "org.apache.jira.issue.created", {"issue_key": "KAFKA-1", "creator_name": "Rita Skeeter"}),
    moved(11, "2023-01-06T09:00:00.051000Z", "assignee", None, "hashed"),
    moved(12, "2023-01-07T09:00:00Z", "status", "Open", "In Progress"),
    cloud("jira:100:comment:1", "2023-01-08T09:00:00.500000Z", "org.apache.jira.issue.comment", {"issue_key": "KAFKA-1", "author_name": "Colin", "body": "Blocked by the segment refactoring."}),
    cloud("gh:pr-7-opened", "2023-01-09T00:00:00Z", "com.github.pull_request.opened", pull(7, author_login="dev", head_ref="dev/k1", body="## Summary\n\nFirst half.")),
    cloud("gh:review-1", "2023-01-10T00:00:00Z", "com.github.review", pull(7, author_login="rev", state="changes_requested", body="Please split this.")),
    cloud("gh:rc-1", "2023-01-10T00:00:01Z", "com.github.review_comment", pull(7, author_login="rev", path="core/src/main/java/A.java", line=12, body="Off by one here.")),
    cloud("gh:pr-8-opened", "2023-01-11T00:00:00Z", "com.github.pull_request.opened", pull(8, author_login="dev", head_ref="dev/k1b", body="Second half.")),
    cloud("gh:pr-7-merged", "2023-01-12T00:00:00Z", "com.github.pull_request.merged", pull(7, author_login="dev", merged_by="boss", merge_commit_sha="abc1234", body="## Summary\n\nFirst half.", changed_files=["core/src/main/java/A.java"])),
    cloud("gh:commit-abc1234", "2023-01-12T00:00:01Z", "com.github.push", {"branch": "trunk", "sha": "abc1234", "message": "KAFKA-1: part 7 (#7)\n\n## Notes\n\nReviewers: rev", "associated_pr_numbers": [7], "issue_keys": ["KAFKA-1"], "commits": [{"author_name": "Dev", "message": "x", "changed_files": ["core/src/main/java/A.java"]}]}),
    letter("m1", "2023-01-13T00:00:00Z", "[DISCUSS] KIP-9: tiered loading", "I propose tiered loading of segments."),
    # ---- boundary of checkpoint 2023-01 ----
    moved(13, "2023-02-03T09:00:00.250000Z", "status", "In Progress", "Closed"),
    moved(14, "2023-02-03T09:00:00.250000Z", "resolution", None, "Fixed"),
    moved(15, "2023-02-04T09:00:00Z", "summary", "Move LogLoader to storage", f"Final title {LATER}"),
    moved(16, "2023-02-04T09:00:01Z", "description", LONG, f"final text {LATER}"),
    moved(17, "2023-02-05T09:00:00Z", "assignee", "hashed", "hashed2"),
    moved(18, "2023-02-05T09:00:00Z", "priority", "Major", "Blocker"),
    cloud("gh:pr-8-merged", "2023-02-06T00:00:00Z", "com.github.pull_request.merged", pull(8, author_login="dev", merged_by="boss", merge_commit_sha="def5678", body=f"Second half. {LATER}", changed_files=["core/B.java"])),
    cloud("gh:comment-9", "2023-02-06T00:00:01Z", "com.github.issue_comment", pull(8, author_login="boss", body=f"Thanks! {LATER}")),
    letter("m2", "2023-02-07T00:00:00Z", "Re: [DISCUSS] KIP-9: tiered loading", f"+1 {LATER} " + "padding " * 60),
]  # fmt: skip


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    root = tmp_path_factory.mktemp("sources")
    jira = stream.Jira(ISSUES)
    events = sorted((stream.render(e, jira) for e in REPLAY), key=lambda e: (e["time"], e["id"]))
    (root / "stream-smoke").mkdir()
    (root / "stream-smoke/events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    assert stores_build.main(["--arm", "S-tools", "--scope", "smoke", "--export", "all", "--root", str(root)]) == 0  # fmt: skip
    return root / "stores-smoke"


def database(stores, checkpoint):
    return stores / "S-tools/export" / checkpoint / "sources.sqlite"


def everything(path):
    """Every value of every table, long texts decompressed."""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    values = [v for t in tables if not t.startswith("fts") for row in db.execute(f"SELECT * FROM {t}") for v in row]  # fmt: skip
    return "\n".join(index.unpack(v) if isinstance(v, bytes | str) else str(v) for v in values)


def run(capsys, monkeypatch, path, tool, *argv):
    monkeypatch.setenv("HARNEXT_SOURCES", str(path))
    code = cli.main(tool, list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_one_physical_database_per_checkpoint_and_the_earlier_one_holds_no_future(stores):
    early, late = database(stores, "2023-01"), database(stores, "2023-02")
    assert LATER.encode() not in early.read_bytes() and LATER not in everything(early)
    assert LATER in everything(late)  # the sentinel is real, only later
    stored = sqlite3.connect(early).execute("SELECT description FROM issues").fetchone()[0]
    assert (
        isinstance(stored, bytes) and zlib.decompress(stored).decode() == LONG.strip()
    )  # compressed
    found = sqlite3.connect(early).execute("SELECT count(*) FROM fts WHERE fts MATCH 'sentinel'")
    assert found.fetchone()[0] == 0
    assert audit.leaks(early) == [] and audit.leaks(late) == []
    manifest = json.loads((early.parent.parent / "2023-01.manifest.json").read_text())
    assert manifest["boundary"] == "2023-02-01T00:00:00Z" and manifest["events"] == 11
    assert manifest["sha256"] == index.file_sha(early) and not early.stat().st_mode & 0o222
    state = json.loads((stores / "S-tools/build-state.json").read_text())
    assert state["checkpoints"] == ["2023-01", "2023-02"]


def test_state_is_as_of_the_boundary(stores, capsys, monkeypatch):
    _, out, err = run(capsys, monkeypatch, database(stores, "2023-01"), "jira", "issue", "view", "KAFKA-1", "--history", "--comments", "5", "--raw")  # fmt: skip
    issue = json.loads(out)
    assert err.strip() == "# offline replica, data as of 2023-02-01T00:00:00Z"
    assert (issue["summary"], issue["status"], issue["resolution"], issue["assignee"], issue["priority"]) == ("Move LogLoader to storage", "In Progress", None, "Rita Skeeter", "Major")  # fmt: skip
    assert issue["description"] == LONG.strip() and issue["pullRequests"] == [7, 8]
    assert issue["reporter"] == "Rita Skeeter" and issue["components"] == ["core"]
    assert [(h["at"], h["field"], h["from"], h["to"]) for h in issue["history"]] == [
        (
            "2023-01-06T09:00:00.051000Z",
            "assignee",
            None,
            "Rita Skeeter",
        ),  # display name, microseconds
        ("2023-01-07T09:00:00Z", "status", "Open", "In Progress"),
    ]
    assert issue["comments"] == [{"author": "Colin", "created": "2023-01-08T09:00:00.500000Z", "body": "Blocked by the segment refactoring."}]  # fmt: skip
    _, out, _ = run(capsys, monkeypatch, database(stores, "2023-02"), "jira", "issue", "view", "KAFKA-1", "--history")  # fmt: skip
    assert (
        "status:\tClosed" in out
        and "assignee:\tFuture Owner" in out
        and "resolution:\tFixed" in out
    )
    assert "2023-02-03T09:00:00.250000Z\tRita Skeeter\tresolution: (none) → Fixed" in out
    assert out.index("status: In Progress → Closed") < out.index("resolution: (none) → Fixed")


def test_a_pr_merged_after_the_boundary_is_open(stores, capsys, monkeypatch):
    early, late = database(stores, "2023-01"), database(stores, "2023-02")
    _, out, _ = run(capsys, monkeypatch, early, "gh", "pr", "list", "--state", "all", "--json", "number,state,mergedAt,mergeCommit,files")  # fmt: skip
    assert json.loads(out) == [
        {"number": 8, "state": "OPEN", "mergedAt": None, "mergeCommit": None, "files": []},
        {"number": 7, "state": "MERGED", "mergedAt": "2023-01-12T00:00:00Z", "mergeCommit": {"oid": "abc1234"}, "files": [{"path": "core/src/main/java/A.java"}]},
    ]  # fmt: skip
    _, out, _ = run(capsys, monkeypatch, early, "gh", "pr", "list")
    assert out == "8\tKAFKA-1: part 8\tdev/k1b\tOPEN\t2023-01-11T00:00:00Z\n"
    code, _, err = run(capsys, monkeypatch, early, "gh", "pr", "diff", "8")
    assert code == 1 and "OPEN as of 2023-02-01" in err
    _, out, _ = run(capsys, monkeypatch, late, "gh", "pr", "list", "--state", "merged", "--search", "second half")  # fmt: skip
    assert out.split("\t")[0] == "8"
    _, out, _ = run(capsys, monkeypatch, late, "gh", "api", "repos/apache/kafka/pulls/8")
    assert json.loads(out)["merged"] is True and json.loads(out)["merge_commit_sha"] == "def5678"


def test_gh_reads(stores, capsys, monkeypatch):
    early = database(stores, "2023-01")
    _, out, _ = run(capsys, monkeypatch, early, "gh", "pr", "view", "7", "--comments")
    assert "state:\tMERGED" in out and "merge commit:\tabc1234" in out and "jira issues:\tKAFKA-1" in out  # fmt: skip
    assert (
        "--\n## Summary\n\nFirst half." in out
    )  # a `##` inside the body is content, not structure
    assert "review (CHANGES_REQUESTED) by rev\nPlease split this." in out
    assert "review comment by rev on core/src/main/java/A.java:12\nOff by one here." in out
    _, out, _ = run(capsys, monkeypatch, early, "gh", "pr", "diff", "7")
    assert "git show abc1234" in out and out.strip().endswith("core/src/main/java/A.java")
    _, out, _ = run(capsys, monkeypatch, early, "gh", "search", "commits", "reviewers")
    assert out.startswith("abc1234\tKAFKA-1: part 7 (#7)\tDev\t")
    _, out, _ = run(capsys, monkeypatch, early, "gh", "api", "/repos/apache/kafka/pulls/7/comments")
    assert json.loads(out)[0] | {"body": ""} == {"user": {"login": "rev"}, "created_at": "2023-01-10T00:00:01Z", "body": "", "path": "core/src/main/java/A.java", "line": "12"}  # fmt: skip
    _, out, _ = run(capsys, monkeypatch, early, "gh", "search", "prs", "KAFKA-1", "--state", "open")
    assert [line.split("\t")[0] for line in out.splitlines()] == ["8"]
    code, _, err = run(capsys, monkeypatch, early, "gh", "issue", "view", "1")
    assert code == 1 and "JIRA" in err and "jira issue view" in err
    code, _, err = run(capsys, monkeypatch, early, "gh", "pr", "checks", "7")
    assert code == 1 and "not available offline" in err
    code, _, err = run(capsys, monkeypatch, early, "gh", "release", "list")
    assert code == 2 and "offline replica as of 2023-02-01T00:00:00Z: supported commands are" in err
    code, out, _ = run(capsys, monkeypatch, early, "gh", "--version")
    assert (
        code == 0 and "offline-replica" in out and "2023-02-01" in out and "\n" not in out.strip()
    )
    code, _, err = run(capsys, monkeypatch, early, "gh", "pr", "view", "7", "--json", "nope")
    assert code == 1 and "Unknown JSON field" in err
    for argv in (["--help"], ["pr", "--help"], ["pr", "list", "--help"]):
        code, out, _ = run(capsys, monkeypatch, early, "gh", *argv)
        assert code == 0 and "usage: gh" in out


def test_jira_list_and_jql_lite(stores, capsys, monkeypatch):
    early, late = database(stores, "2023-01"), database(stores, "2023-02")
    keys = lambda out: [line.split("\t")[0] for line in out.splitlines()]  # noqa: E731
    for argv in (
        ["--summary", "Move LogLoader to storage"], ["--text", "segment refactoring"],
        ["loader", "storage"], ["--status", "in progress", "--assignee", "rita skeeter"],
        ["-q", 'project = KAFKA AND status = "In Progress" AND assignee = "Rita Skeeter" AND resolution = Unresolved ORDER BY created DESC'],
        ["-q", 'text ~ "\\"storage module\\"" AND created >= 2023-01-05 AND created <= 2023-01-05 AND component = core'],
        ["-q", "priority = Major AND resolution is EMPTY AND summary ~ logloader"],
    ):  # fmt: skip
        assert keys(run(capsys, monkeypatch, early, "jira", "issue", "list", *argv)[1]) == ["KAFKA-1"], argv  # fmt: skip
    for argv in (["--summary", "Final title"], ["--status", "Closed"], ["-q", "resolution = Fixed"], ["-q", 'text ~ "\\"module storage\\""'], ["-q", "created > 2023-01-05"]):  # fmt: skip
        assert keys(run(capsys, monkeypatch, early, "jira", "issue", "list", *argv)[1]) == [], argv
    assert keys(run(capsys, monkeypatch, late, "jira", "issue", "list", "-q", "status = Closed AND resolution = Fixed")[1]) == ["KAFKA-1"]  # fmt: skip
    # the summary was edited: the old words no longer find it, the index row was replaced
    assert keys(run(capsys, monkeypatch, late, "jira", "issue", "list", "--text", "LogLoader")[1]) == []  # fmt: skip
    code, _, err = run(capsys, monkeypatch, early, "jira", "issue", "list", "-q", "status in (Open, Closed) OR x = 1")  # fmt: skip
    assert code == 1 and "unsupported clause" in err and "JQL-lite" in err and "ORDER BY" in err
    code, _, err = run(capsys, monkeypatch, early, "jira", "issue", "view", "KAFKA-999")
    assert code == 1 and "not in the replica as of 2023-02-01T00:00:00Z" in err
    code, out, _ = run(capsys, monkeypatch, early, "jira", "issue", "list", "--help")
    assert code == 0 and "JQL-lite" in out


def test_mail(stores, capsys, monkeypatch):
    early, late = database(stores, "2023-01"), database(stores, "2023-02")
    _, out, _ = run(capsys, monkeypatch, early, "mail", "search", "tiered loading", "--list", "dev")
    assert [line.split("\t")[0] for line in out.splitlines()] == ["m1"]
    assert run(capsys, monkeypatch, early, "mail", "search", "tiered", "--list", "users")[1] == ""
    assert run(capsys, monkeypatch, early, "mail", "search", "tiered", "--after", "2023-01-14")[1] == ""  # fmt: skip
    _, out, _ = run(capsys, monkeypatch, early, "mail", "read", "m1")
    assert "subject:\t[DISCUSS] KIP-9: tiered loading" in out and out.endswith("I propose tiered loading of segments.\n")  # fmt: skip
    _, out, _ = run(capsys, monkeypatch, late, "mail", "thread", "m1")
    assert [line.split("\t")[0] for line in out.splitlines()] == ["m1", "m2"]
    _, out, _ = run(capsys, monkeypatch, early, "mail", "thread", "t1", "--full")
    assert "m2" not in out and "I propose" in out
    code, _, err = run(capsys, monkeypatch, early, "mail", "read", "m2")
    assert code == 1 and "no message" in err


def test_without_a_database_every_tool_says_so_and_exits_2(tmp_path, capsys, monkeypatch):
    for value in (None, str(tmp_path / "missing.sqlite")):
        monkeypatch.delenv("HARNEXT_SOURCES", raising=False)
        if value:
            monkeypatch.setenv("HARNEXT_SOURCES", value)
        for tool in ("gh", "jira", "mail"):
            assert cli.main(tool, ["--help"]) == 2
            captured = capsys.readouterr()
            assert captured.out == "" and f"{tool}: not available in this trial" in captured.err


def test_resume_continues_from_the_last_snapshot_and_matches_a_straight_build(stores, tmp_path):
    events = list(stream.read_events(stores.parent / "stream-smoke/events.jsonl"))
    out = tmp_path / "S-tools"
    index.SourcesStore().build(iter(events), out, until="2023-01")
    first = index.file_sha(out / "export/2023-01" / index.DB)
    index.SourcesStore().build(iter(events), out)
    assert index.file_sha(out / "export/2023-01" / index.DB) == first  # not rebuilt
    assert everything(out / "export/2023-02" / index.DB) == everything(database(stores, "2023-02"))
    # a wanted snapshot behind the cursor cannot be cut from a database that moved on: rebuild
    (out / "export/2023-01.manifest.json").unlink()
    index.SourcesStore().build(iter(events), out)
    assert everything(out / "export/2023-01" / index.DB) == everything(database(stores, "2023-01"))
    assert index.SourcesStore().export(out, "2023-02") == out / "export/2023-02"
    with pytest.raises(ValueError):
        index.SourcesStore().export(out, "2024-01")


def test_audit_oracle_reproduces_gold_through_the_cli_and_lists_defects(stores, tmp_path):
    history = [{"at": "2023-01-06T09:00:00.051000Z", "from": None, "to": "Rita Skeeter"}]
    tasks = [
        ("H-1", "status_at_snapshot", "What was the status of KAFKA-1?", "In Progress", False),
        ("H-2", "assignment_history", 'the Apache Kafka JIRA issue titled "Move LogLoader to storage" (created in 2023-01)', history, True),
        ("H-3", "closed_at_snapshot", "Was KAFKA-1 closed?", False, False),
        ("H-4", "resolution_at_snapshot", "Resolution of KAFKA-1?", None, False),
        ("H-5", "priority_at_snapshot", "Priority of KAFKA-1?", "Minor", False),  # wrong gold
        ("H-6", "status_at_snapshot", 'the Apache Kafka JIRA issue titled "No such title here" (created in 2023-01)', "Open", True),
    ]  # fmt: skip
    bench = tmp_path / "bench"
    bench.mkdir()
    public = [{"id": i, "kind": "historical", "family": f, "checkpoint": "2023-01", "prompt": p, "keyless": k} for i, f, p, _, k in tasks]  # fmt: skip
    private = [{"id": i, "expected": {"value": v}} for i, _, _, v, _ in tasks]
    (bench / "tasks.public.jsonl").write_text("".join(json.dumps(t) + "\n" for t in public))
    (bench / "tasks.private.jsonl").write_text("".join(json.dumps(t) + "\n" for t in private))
    report = audit.run("smoke", stores, bench)
    assert report["pass"] == 4 and report["instants_at_or_after_boundary"] == {}
    assert {d["task"]: d["reason"].split(":")[0] for d in report["defects"]} == {
        "H-5": "oracle answer differs from gold",
        "H-6": "0 issues carry the prompt's title as of the cutoff",
    }
    assert report["families"]["assignment_history"] == {"tasks": 1, "pass": 1, "keyless": 1, "keyless_pass": 1}  # fmt: skip
    assert json.loads((bench / "answerability.json").read_text()) == report
    assert private[4]["expected"]["value"] == "Minor"  # the gold is never adjusted


def test_audit_oracle_answers_the_new_families_through_gh_jira_and_mail(stores, tmp_path):
    title = 'the apache/kafka pull request titled "KAFKA-1: part 7" (opened in 2023-01)'
    issue = 'the Apache Kafka JIRA issue titled "Move LogLoader to storage" (created in 2023-01)'
    tasks = [
        ("N-1", "issue_prs", "2023-01", f"Which pull requests had been opened for {issue} by the snapshot?", [{"pr": 7, "state": "merged"}, {"pr": 8, "state": "open"}]),
        ("N-2", "issue_changed_files", "2023-01", "Which files were changed by the pull requests merged for KAFKA-1?", ["core/src/main/java/A.java"]),
        ("N-3", "issue_changed_files", "2023-02", "Which files were changed by the pull requests merged for KAFKA-1?", ["core/B.java", "core/src/main/java/A.java"]),
        ("N-4", "pr_approvers", "2023-01", f"Who had approved {title} by the snapshot?", []),
        ("N-5", "review_hotspot", "2023-01", "Which file received the most inline review comments on pull request #7?", "core/src/main/java/A.java"),
        ("N-6", "reopen_count", "2023-02", "How many times had KAFKA-1 been transitioned to Reopened?", 0),
        ("N-7", "fix_version_at_snapshot", "2023-02", "Which fix versions were set on KAFKA-1?", []),
        ("N-8", "resolution_vs_merge_order", "2023-02", "For KAFKA-1, exactly one pull request ...", {"first": "merge", "days": 0}),  # two merged: defect
        ("N-9", "open_blockers_unassigned", "2023-02", 'Which issues in component "core" were Blocker, unassigned ...', []),
        ("N-10", "author_activity", "2023-01", "How many pull requests authored by the GitHub user dev were merged (merged at or after 2022-11-03T00:00:00Z, through the cutoff)?", 1),
        ("N-11", "top_reviewer_for_author", "2023-02", "Among the pull requests authored by the GitHub user dev and merged (merged at or after 2022-03-01T00:00:00Z, through the cutoff), which login approved most?", "rev"),  # no approvals: defect
        ("N-12", "kip_thread", "2023-02", 'Consider the mailing-list discussion of KIP-9: the thread holding the earliest message whose subject contains "[DISCUSS] KIP-9".', {"starter": "Colin", "participants": 1}),
        ("N-13", "kip_thread", "2023-01", 'the earliest message whose subject contains "[DISCUSS] KIP-9"', {"starter": "Colin", "participants": 2}),  # wrong gold
        ("N-14", "issue_links", "2023-02", 'Which issues was KAFKA-1 linked to with the issue-link relation "blocks"?', []),
    ]  # fmt: skip
    bench = tmp_path / "bench"
    bench.mkdir()
    public = [{"id": i, "kind": "historical", "family": f, "checkpoint": c, "prompt": p, "keyless": "titled" in p} for i, f, c, p, _ in tasks]  # fmt: skip
    private = [{"id": i, "family": f, "expected": {"value": v}} for i, f, _, _, v in tasks]
    (bench / "tasks.public.jsonl").write_text("".join(json.dumps(t) + "\n" for t in public))
    (bench / "tasks.private.jsonl").write_text("".join(json.dumps(t) + "\n" for t in private))
    report = audit.run("smoke", stores, bench)
    reasons = {d["task"]: d["reason"] for d in report["defects"]}
    assert reasons.keys() == {"N-8", "N-11", "N-13"}, reasons
    assert reasons["N-8"].startswith("1 resolution settings and 2 merged PRs")
    assert reasons["N-11"] == "no unique maximum in the CLI output"
    assert reasons["N-13"].startswith("oracle answer differs from gold")
    assert report["pass"] == 11 and report["families"]["issue_prs"] == {"tasks": 1, "pass": 1, "keyless": 1, "keyless_pass": 1}  # fmt: skip
    assert report["families"]["pr_approvers"]["keyless_pass"] == 1


def test_arm_wiring_prompt_mount_and_cli_counting(stores, tmp_path):
    arms, server = runner.load_arms(runner.ROOT / "benchmarks/kafka-500-v2/arms.yaml", ["N", "S-tools", "F0"])  # fmt: skip
    arm = next(a for a in arms if a["id"] == "S-tools")
    assert arm == {"id": "S-tools", "access": "sources", "store": "S-tools"}
    prompts = {a["id"]: runner.build_prompt({"prompt": "Fix.", "kind": "coding"}, a, server) for a in arms}  # fmt: skip
    assert "`gh` (pull requests, reviews, commits), `jira`" in prompts["S-tools"]
    assert "/context" not in prompts["S-tools"] and "`gh`" not in prompts["F0"] + prompts["N"]
    assert all(runner.GIT_HISTORY in p and "no git history" not in p for p in prompts.values())
    assert runner.GIT_HISTORY not in runner.build_prompt({"prompt": "Q", "kind": "historical"}, arm, server)  # fmt: skip
    spec = base.TrialSpec("t", "S-tools", "historical", "p", {}, tmp_path, tmp_path, "m", "medium", 60, sources=database(stores, "2023-01"))  # fmt: skip
    argv = container.run_argv(spec, "name", ["true"], auth=[], home=tmp_path / "home")
    assert f"{database(stores, '2023-01').parent}:/sources:ro" in argv
    assert "HARNEXT_SOURCES=/sources/sources.sqlite" in argv
    plain = container.run_argv(base.TrialSpec("t", "N", "historical", "p", {}, tmp_path, tmp_path, "m", "medium", 60), "name", ["true"], auth=[], home=tmp_path / "home")  # fmt: skip
    assert not any("sources" in part.lower() for part in plain)
    commands = [
        "/bin/bash -lc 'jira issue view KAFKA-1 --history | head -50'",
        "bash -lc \"gh pr list --search 'x' && gh pr view 7; git log --oneline -- core/.git mail\"",
        "mail search kip | xargs -n1 mail read", "grep -rn 'git ' docs/jira.md", "cat .github/gh/x",
        "cd /workspace && git blame A.java", None,
    ]  # fmt: skip
    assert base.cli_calls(commands) == {"gh": 2, "git": 2, "jira": 1, "mail": 2}
