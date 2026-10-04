"""The eleven families of families.py on tiny synthetic issues and stream facts: gold at the
boundary, temporal traps, one entity per task, wording, and the streaming facts pass."""

import gzip
import json

import pytest
from harnext_eval.e2.benchmark import canonical, sha, timestamp
from harnext_eval.e2.v2 import families, stream, tasks

START = timestamp("2023-01-01T00:00:00Z")
CKPTS = ["2023-03", "2023-04", "2023-06"]


def issue(number, rows, created="2023-01-15T00:00:00.000+0000", **fields):
    """rows: (iso time, field, from, to) -> raw JIRA issue with a complete changelog."""
    histories = [
        {"id": str(number * 100 + i), "created": at, "items": [{"field": f, "fromString": a, "toString": b}]}
        for i, (at, f, a, b) in enumerate(rows)
    ]  # fmt: skip
    return {
        "id": str(number),
        "key": f"KAFKA-{number}",
        "fields": {"summary": f"A sufficiently unique title number {number}", "created": created, **fields},
        "changelog": {"startAt": 0, "total": len(histories), "histories": histories},
    }


ISSUES = {
    "KAFKA-1": issue(1, [
        ("2023-02-01T10:00:00.000+0000", "status", "Open", "Resolved"),
        ("2023-02-01T12:00:00.000+0000", "resolution", None, "Fixed"),
        ("2023-02-01T10:00:00.000+0000", "Fix Version", None, "3.5.0"),
        ("2023-02-10T10:00:00.000+0000", "Link", None, "This issue blocks KAFKA-2"),
        ("2023-03-01T10:00:00.000+0000", "Link", None, "This issue blocks KAFKA-3"),
        ("2023-03-02T10:00:00.000+0000", "Fix Version", "3.5.0", None),
        ("2023-03-02T10:00:00.000+0000", "Fix Version", None, "3.6.0"),
        ("2023-03-05T10:00:00.000+0000", "Link", "This issue blocks KAFKA-2", None),
        ("2023-03-06T10:00:00.000+0000", "Link", None, "This issue relates to KAFKA-4"),
        ("2023-03-15T10:00:00.000+0000", "status", "Resolved", "Reopened"),
        ("2023-03-15T10:00:00.000+0000", "resolution", "Fixed", None),
        ("2023-03-20T10:00:00.000+0000", "status", "Reopened", "Resolved"),
        ("2023-03-20T10:00:00.000+0000", "resolution", None, "Fixed"),
        ("2023-05-10T10:00:00.000+0000", "Fix Version", None, "3.5.1"),
        ("2023-07-01T10:00:00.000+0000", "status", "Resolved", "Reopened"),
    ], fixVersions=[{"name": "3.6.0"}, {"name": "3.5.1"}], priority={"name": "Major"}),
    "KAFKA-2": issue(2, [
        ("2023-05-01T10:00:00.000+0000", "status", "Open", "Closed"),
        ("2023-06-25T10:00:00.500+0000", "resolution", None, "Fixed"),
    ], fixVersions=[{"name": "9.9.9"}], priority={"name": "Major"}),
    "KAFKA-3": issue(3, [], priority={"name": "Blocker"}, status={"name": "Open"}, components=[{"name": "core"}]),
    "KAFKA-4": issue(4, [
        ("2023-02-01T10:00:00.000+0000", "Fix Version", None, "1.0"),
    ], priority={"name": "Blocker"}, status={"name": "Open"}, assignee={"displayName": "Someone"}, components=[{"name": "core"}], fixVersions=[{"name": "2.0"}]),
}  # fmt: skip


def pr(title, author, issues, events, **extra):
    return {"title": title, "author": author, "issues": issues, "tags": sorted(issues), "events": events, **extra}


FACTS = {
    "sha": "x",
    "prs": {
        "7": pr("KAFKA-1: part seven of the move", "dev", ["KAFKA-1"], [["2023-03-01T00:00:00Z", "open", "e7o"], ["2023-03-10T00:00:00Z", "merged", "e7m"]], merge_commit="abc", files=["a/A.java", "b/B.java"]),
        "8": pr("KAFKA-1: part eight of the move", "dev", ["KAFKA-1"], [["2023-03-05T00:00:00Z", "open", "e8o"], ["2023-05-02T00:00:00Z", "closed", "e8c"]]),
        "9": pr("KAFKA-2: the other thing", "dev", ["KAFKA-2"], [["2023-06-01T00:00:00Z", "open", "e9o"], ["2023-06-20T00:00:00Z", "merged", "e9m"]], merge_commit="def", files=["c/C.java"]),
    },
    "reviews": {
        "7": [["2023-03-02T00:00:00Z", "rev1", "approved", "r1"], ["2023-03-03T00:00:00Z", "rev2", "approved", "r2"], ["2023-03-04T00:00:00Z", "rev1", "approved", "r3"], ["2023-05-01T00:00:00Z", "rev3", "approved", "r4"]],
        "9": [["2023-06-10T00:00:00Z", "rev1", "approved", "r5"], ["2023-06-11T00:00:00Z", "rev1", "commented", "r6"]],
    },
    "review_comments": {
        "7": [["2023-03-02T00:00:00Z", "rev1", "a/A.java", "c1"], ["2023-03-02T00:00:01Z", "rev1", "a/A.java", "c2"], ["2023-03-03T00:00:00Z", "rev2", "b/B.java", "c3"]],
        "9": [["2023-06-10T00:00:00Z", "rev1", "x", "c4"], ["2023-06-10T00:00:01Z", "rev2", "y", "c5"]],
    },
    "mails": sorted([
        ["2023-03-01T00:00:00Z", "m1", "t1", "Alice", "[DISCUSS] KIP-99: thing"],
        ["2023-03-02T00:00:00Z", "m2", "t1", "Bob", "Re: [DISCUSS] KIP-99: thing"],
        ["2023-03-03T00:00:00Z", "m3", "t1", "Alice", "Re: [DISCUSS] KIP-99: thing"],
        ["2023-03-04T00:00:00Z", "m4", "t1", "Carol", "Re: [DISCUSS] KIP-99: thing"],
        ["2023-03-05T00:00:00Z", "m5", "t1", "Bob", "Re: thing"],
        ["2023-05-01T00:00:00Z", "m6", "t1", "Dave", "Re: [DISCUSS] KIP-99: thing"],
        ["2023-03-06T00:00:00Z", "m7", "t2", "Eve", "[DISCUSS] KIP-990: other"],
    ]),
}  # fmt: skip


@pytest.fixture
def ctx():
    return families.Context(ISSUES, FACTS, CKPTS, START)


def by_ckpt(found):
    return {t["checkpoint"]: t for t in found}


def test_issue_prs_state_at_boundary_and_trap(ctx):
    found = by_ckpt(t for t in families.issue_prs(ctx) if t["issue"] == "KAFKA-1")
    early = found["2023-03"] if "2023-03" in found else found["2023-04"]
    assert early["expected"]["value"] == [{"pr": 7, "state": "merged"}, {"pr": 8, "state": "open"}]
    assert early["temporal_trap"] and early["preferred"] and early["issue"] == "KAFKA-1"
    assert set(early["evidence_ids"]) == {"e7o", "e7m", "e8o"}
    late = found["2023-06"]
    assert late["expected"]["value"][1] == {"pr": 8, "state": "closed"} and not late["temporal_trap"]
    assert families.state(FACTS["prs"]["8"], "2023-03-01T00:00:00Z") is None  # not yet opened


def test_issue_changed_files_union_and_clone_verification(ctx):
    found = [t for t in families.issue_changed_files(ctx)]
    mine = {t["issue"]: t for t in found}
    assert mine["KAFKA-1"]["expected"]["value"] == ["a/A.java", "b/B.java"]
    assert mine["KAFKA-1"]["merge_commits"] == ["abc"] and mine["KAFKA-2"]["expected"]["value"] == ["c/C.java"]  # fmt: skip
    cache = {"abc": ["b/B.java", "a/A.java"], "def": None}
    assert families.verify_files(mine["KAFKA-1"], cache) is True
    assert families.verify_files(mine["KAFKA-2"], cache) is False  # commit not in the clone
    cache["abc"] = ["a/A.java"]
    assert families.verify_files(mine["KAFKA-1"], cache) is False  # stream and clone disagree


def test_pr_approvers_dedupes_and_traps(ctx):
    found = by_ckpt(t for t in families.pr_approvers(ctx) if t["pr"] == 7)
    early = next(t for t in found.values() if t["temporal_trap"])
    assert early["expected"]["value"] == ["rev1", "rev2"] and early["preferred"]
    assert set(early["evidence_ids"]) == {"r1", "r2", "r3"} and early["issues"] == ["KAFKA-1"]
    assert found["2023-06"]["expected"]["value"] == ["rev1", "rev2", "rev3"]
    nine = [t for t in families.pr_approvers(ctx) if t["pr"] == 9]
    assert [t["expected"]["value"] for t in nine] == [["rev1"]] and not nine[0]["preferred"]


def test_review_hotspot_skips_ties(ctx):
    found = list(families.review_hotspot(ctx))
    assert {t["pr"] for t in found} == {7}
    assert found[0]["expected"]["value"] == "a/A.java" and set(found[0]["evidence_ids"]) == {"c1", "c2"}
    assert families.hotspot(FACTS["review_comments"]["9"], "2023-07-01T00:00:00Z") is None


def test_reopen_count_counts_visible_reopens_only(ctx):
    found = {(t["issue"], t["checkpoint"]): t for t in families.reopen_count(ctx)}
    one = next(t for (k, _), t in found.items() if k == "KAFKA-1")
    assert one["expected"]["value"] == 1 and one["temporal_trap"] and one["preferred"]
    assert one["evidence_ids"] == ["jira:1:change:109:0"]
    zero = found[("KAFKA-2", "2023-06")]
    assert zero["expected"]["value"] == 0 and not zero["temporal_trap"]
    assert not any(k == "KAFKA-3" for k, _ in found)  # never resolved: trivial, not asked


def test_links_replay_honours_removals(ctx):
    rows = ctx.log("KAFKA-1", "Link")
    assert families.links(rows) == {"blocks": ["KAFKA-3"], "relates to": ["KAFKA-4"]}
    assert families.links(rows[:2]) == {"blocks": ["KAFKA-2", "KAFKA-3"]}
    found = list(families.issue_links(ctx))
    assert len(found) == 1 and found[0]["relation"] in families.RELATIONS  # one relation per issue
    assert found[0]["expected"]["value"] == {"blocks": ["KAFKA-3"], "relates to": ["KAFKA-4"]}[found[0]["relation"]]  # fmt: skip
    assert families.link({"from": None, "to": "This issue is blocked by KAFKA-9"}) == ("is blocked by", "KAFKA-9", True)  # fmt: skip
    assert families.link({"from": "This issue Blocked KAFKA-9", "to": None}) == ("Blocked", "KAFKA-9", False)


def test_fix_versions_forward_replay_must_meet_the_export(ctx):
    found = by_ckpt(families.fix_version_at_snapshot(ctx))
    early = next(t for t in found.values() if t["temporal_trap"])
    assert early["expected"]["value"] == ["3.6.0"] and early["preferred"] and early["issue"] == "KAFKA-1"
    assert found["2023-06"]["expected"]["value"] == ["3.5.1", "3.6.0"] and not found["2023-06"]["temporal_trap"]  # fmt: skip
    assert not any(t["issue"] == "KAFKA-4" for t in found.values())  # changelog says 1.0, export 2.0
    assert families.fix_versions(ctx.log("KAFKA-1", "Fix Version")) == ["3.5.1", "3.6.0"]


def test_resolution_vs_merge_order(ctx):
    found = list(families.resolution_vs_merge_order(ctx))
    assert [(t["issue"], t["checkpoint"], t["expected"]["value"]) for t in found] == [
        ("KAFKA-2", "2023-06", {"first": "merge", "days": 5})
    ]  # KAFKA-1 set its resolution twice
    assert set(found[0]["evidence_ids"]) == {"jira:2:change:201:0", "e9m"}
    assert families.order_and_days(timestamp("2023-01-01T00:00:00Z"), "2023-01-03T23:59:59Z") == {"first": "resolution", "days": 2}  # fmt: skip
    assert families.order_and_days(timestamp("2023-01-05T00:00:00Z"), "2023-01-03T00:00:00Z") == {"first": "merge", "days": 2}  # fmt: skip


def test_open_blockers_unassigned_replays_every_issue(ctx):
    found = list(families.open_blockers_unassigned(ctx))
    assert found and all(t["component"] == "core" and t["expected"]["value"] == ["KAFKA-3"] for t in found)
    assert found[0]["evidence_ids"] == ["jira:3:created"] and found[0]["issue"] is None
    assert families.reserved(found[0]) == {f"slot:blockers:core:{found[0]['checkpoint']}"}
    keys = families.ever_blocker(ctx)
    assert set(keys) == {"KAFKA-3", "KAFKA-4"}  # KAFKA-4 is assigned, never listed
    later = families.blockers(ctx, timestamp("2022-12-01T00:00:00Z"), keys)
    assert later == {}  # not created yet


def test_author_activity_and_top_reviewer(ctx):
    counts = list(families.author_activity(ctx))
    assert [(t["login"], t["expected"]["value"]) for t in counts] == [("dev", 1)]
    assert counts[0]["since"] == "2023-01-01T00:00:00Z" or counts[0]["since"].startswith("2023-04-02")
    top = list(families.top_reviewer_for_author(ctx))
    assert [(t["checkpoint"], t["expected"]["value"]) for t in top] == [("2023-06", "rev1")]  # earlier: tie
    assert top[0]["preferred"] and set(top[0]["evidence_ids"]) == {"r1", "r3", "r5"}
    assert families.reserved(top[0]) == {"login:dev"} == families.reserved(counts[0])


def test_kip_thread_starter_and_participants(ctx):
    found = by_ckpt(families.kip_thread(ctx))
    early = next(t for t in found.values() if t["temporal_trap"])
    assert early["kip"] == "KIP-99" and early["expected"]["value"] == {"starter": "Alice", "participants": 3}
    assert set(early["evidence_ids"]) == {"m1", "m2", "m3", "m4", "m5"}
    assert found["2023-06"]["expected"]["value"] == {"starter": "Alice", "participants": 4}
    assert not any(t["kip"] == "KIP-990" for t in found.values())  # too short; KIP-99 never matches it
    rows, later = families.discussion(FACTS["mails"], "KIP-990", "2023-07-01T00:00:00Z")
    assert [m[1] for m in rows] == ["m7"] and not later


def test_selection_reserves_entities_once_and_is_deterministic(ctx):
    counts = {f: 2 for f in families.NEW}
    pools = families.candidates(ctx, counts)
    assert set(pools) == set(families.GENERATORS) and pools["author_activity"][0] == 1  # 2 split over variants
    taken, chosen = set(), {}
    verify = lambda t, cache: True  # noqa: E731
    for family, (count, found) in sorted(pools.items()):
        chosen[family], _ = families.choose(family, found, count, taken, "seed", verify=verify)
    again = set()
    for family, (count, found) in sorted(pools.items()):
        assert families.choose(family, found, count, again, "seed", verify=verify)[0] == chosen[family]
    everything = [t for mine in chosen.values() for t in mine]
    ids = [families.reserved(t) for t in everything]
    assert all(not a & b for i, a in enumerate(ids) for b in ids[i + 1 :])  # nothing reserved twice
    assert chosen["kip_thread"] and chosen["author_activity"] and chosen["open_blockers_unassigned"]
    # two issues only: the issue-centred families that pick first take both, and a PR task
    # reserves its PR's issues too, so later issue/PR families find nothing (never padded)
    assert sum(len(v) for v in chosen.values()) < sum(c for c, _ in pools.values())
    alone = set()
    assert families.choose("issue_prs", pools["issue_prs"][1], 2, alone, "seed", verify=verify)[0]
    assert not families.choose("pr_approvers", pools["pr_approvers"][1], 2, alone, "seed", verify=verify)[0]  # fmt: skip


def test_wording_schema_and_grading_hints():
    schema = families.answer_schema
    for family in families.GENERATORS:
        assert schema(family)["required"] == ["value", "evidence"]
    fake = {"family": "issue_links", "relation": "is blocked by"}
    text = families.question(fake, "KAFKA-1")
    assert '"is blocked by"' in text and "KAFKA-1" in text and "tool" not in text.lower()
    fake = {"family": "author_activity", "login": "dev", "since": "2023-01-01T00:00:00Z"}
    assert "GitHub user dev were merged" in families.question(fake, None)
    fake = {"family": "kip_thread", "kip": "KIP-99"}
    assert '"[DISCUSS] KIP-99"' in families.question(fake, None)
    with pytest.raises(KeyError):
        families.question({"family": "nope"}, None)


def page(title, facts, body=""):
    return stream.page(title, facts, body)


def test_facts_streams_github_and_mail_events_and_caches(tmp_path):
    events = [
        {"id": "j1", "time": "2023-01-01T00:00:00Z", "type": "jira.created", "entities": ["issue:KAFKA-1"], "actor": "x", "title": "t", "text": "# t\n", "meta": {"issue": "KAFKA-1"}},
        {"id": "e7o", "time": "2023-01-02T00:00:00Z", "type": "gh.pr_opened", "entities": ["pr:7", "issue:KAFKA-1"], "actor": "dev", "title": "PR #7 opened: Part seven", "text": page("PR #7 opened: Part seven", [("author", "dev"), ("at", "2023-01-02T00:00:00Z")]), "meta": {"pr": 7}},
        {"id": "r1", "time": "2023-01-03T00:00:00Z", "type": "gh.review", "entities": ["pr:7", "issue:KAFKA-1"], "actor": "rev", "title": "x", "text": page("x", [("by", "rev")]), "meta": {"pr": 7, "state": "approved"}},
        {"id": "c1", "time": "2023-01-03T00:00:01Z", "type": "gh.review_comment", "entities": ["pr:7", "issue:KAFKA-1"], "actor": "rev", "title": "x", "text": page("x", [("by", "rev")], "nit"), "meta": {"pr": 7, "path": "a/A.java"}},
        {"id": "e7m", "time": "2023-01-04T00:00:00Z", "type": "gh.pr_merged", "entities": ["pr:7", "issue:KAFKA-1"], "actor": "boss", "title": "PR #7 merged: Part seven", "text": page("PR #7 merged: Part seven", [("author", "dev"), ("merged by", "boss")], "## Description\n\nbody\n\n## Changed files\n\n- `a/A.java`\n- `b/B.java`"), "meta": {"pr": 7, "merge_commit": "abc"}},
        {"id": "p1", "time": "2023-01-04T00:00:01Z", "type": "gh.push", "entities": ["pr:7"], "actor": "dev", "title": "x", "text": "# x\n", "meta": {"sha": "abc"}},
        {"id": "m1", "time": "2023-01-05T00:00:00Z", "type": "mail.message", "entities": ["thread:t1", "kip:KIP-9"], "actor": "Alice", "title": "[DISCUSS] KIP-9: x", "text": page("[DISCUSS] KIP-9: x", [("from", "Alice"), ("thread", "t1")], "hi"), "meta": {"list": "dev@kafka.apache.org"}},
    ]  # fmt: skip
    text = "".join(canonical(e) + "\n" for e in events)
    (tmp_path / "events.jsonl").write_text(text)
    (tmp_path / "stream-manifest.json").write_text(json.dumps({"events_sha256": sha(text)}))
    found = families.facts(tmp_path / "events.jsonl")
    assert found["prs"]["7"] == {
        "title": "Part seven", "author": "dev", "issues": ["KAFKA-1"], "tags": ["KAFKA-1"],
        "events": [["2023-01-02T00:00:00Z", "open", "e7o"], ["2023-01-04T00:00:00Z", "merged", "e7m"]],
        "merge_commit": "abc", "files": ["a/A.java", "b/B.java"],
    }  # fmt: skip
    assert found["reviews"] == {"7": [["2023-01-03T00:00:00Z", "rev", "approved", "r1"]]}
    assert found["review_comments"] == {"7": [["2023-01-03T00:00:01Z", "rev", "a/A.java", "c1"]]}
    assert found["mails"] == [["2023-01-05T00:00:00Z", "m1", "t1", "Alice", "[DISCUSS] KIP-9: x"]]
    cache = tmp_path / "facts.json.gz"
    assert json.loads(gzip.decompress(cache.read_bytes())) == found
    (tmp_path / "events.jsonl").write_text("")  # a stale stream is never re-read while the sha matches
    assert families.facts(tmp_path / "events.jsonl") == found
    (tmp_path / "stream-manifest.json").write_text(json.dumps({"events_sha256": "other"}))
    assert families.facts(tmp_path / "events.jsonl")["prs"] == {}


def test_historical_words_new_families_with_keyless_titles(monkeypatch):
    config = {
        "first": CKPTS[0], "last": CKPTS[-1],
        "families": {f: (field, 1) for f, field in tasks.FIELDS.items()},
        "extra": {f: 2 for f in families.NEW},
    }  # fmt: skip
    monkeypatch.setattr(families, "verify_files", lambda task, cache=None: True)  # no clone here
    chosen, short = tasks.historical(ISSUES, config, set(), facts=FACTS)
    assert chosen == tasks.historical(ISSUES, config, set(), facts=FACTS)[0]
    mine = {t["family"]: t for t in chosen}
    assert {"kip_thread", "author_activity", "open_blockers_unassigned", "resolution_vs_merge_order"} <= set(mine)  # fmt: skip
    assert set(mine) & (families.ISSUE_CENTRED | families.PR_CENTRED)  # scarcity order gives them a turn
    assert all(t["prompt"].startswith(f"Context snapshot cutoff (inclusive): {t['cutoff']}.") for t in chosen)
    assert all("tool" not in t["prompt"].lower() and '{"value": <answer>, "evidence": [' in t["prompt"] for t in chosen)  # fmt: skip
    keyless = [t for t in chosen if t["keyless"]]
    assert keyless and all(t["family"] in families.ISSUE_CENTRED | families.PR_CENTRED for t in keyless)
    for task in keyless:
        public = json.dumps(tasks.public(task))
        if task["family"] in families.PR_CENTRED:
            assert f'pull request titled "{task["title_at_cutoff"]}"' in task["prompt"] and f"#{task['pr']}" not in task["prompt"]  # fmt: skip
        else:
            assert task["issue"] not in public and task["summary_at_cutoff"] in task["prompt"]
    for task in chosen:
        if task["family"] in families.GENERATORS:
            assert task["answer_schema"] == families.answer_schema(task["family"])
    ids = [families.reserved(t) for t in chosen]
    assert all(not a & b for i, a in enumerate(ids) for b in ids[i + 1 :])
    assert all(isinstance(v, dict) and "missing" in v for f, v in short.items() if f in families.GENERATORS)
