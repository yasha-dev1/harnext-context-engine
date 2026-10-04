"""Task curation on synthetic issues: gold at the boundary, traps, keyless, coding, order."""

import json

from harnext_eval.e2.benchmark import timestamp
from harnext_eval.e2.v2 import tasks

START = timestamp("2023-01-01T00:00:00Z")


def issue(number, summary, rows, created="2023-02-01T00:00:00.000+0000"):
    """rows: (iso time, field, from, to) -> raw JIRA issue with a complete changelog."""
    histories = [
        {
            "id": str(number * 100 + i),
            "created": at,
            "items": [{"field": f, "fromString": a, "toString": b}],
        }
        for i, (at, f, a, b) in enumerate(rows)
    ]
    return {
        "id": str(number),
        "key": f"KAFKA-{number}",
        "fields": {"summary": summary, "created": created},
        "changelog": {"startAt": 0, "total": len(histories), "histories": histories},
    }


def flow(number, summary="", *, reopen=True, month="08"):
    done = "Closed" if number % 2 == 0 else "Resolved"
    rows = [
        (f"2023-{month}-02T10:00:00.250+0000", "status", "Open", "In Progress"),
        (f"2023-{month}-03T10:00:00.000+0000", "assignee", None, f"Dev {number}"),
        (f"2023-{month}-04T10:00:00.000+0000", "priority", "Major", "Minor"),
        (f"2023-{month}-20T10:00:00.000+0000", "resolution", None, "Fixed"),
        (f"2023-{month}-20T10:00:00.000+0000", "status", "In Progress", done),
    ]
    if reopen:
        rows += [
            ("2024-02-10T10:00:00.000+0000", "status", done, "Reopened"),
            ("2024-02-10T10:00:00.000+0000", "resolution", "Fixed", None),
            ("2024-02-11T10:00:00.000+0000", "assignee", f"Dev {number}", None),
            ("2024-02-12T10:00:00.000+0000", "priority", "Minor", "Blocker"),
        ]
    return issue(number, summary or f"A sufficiently unique title number {number}", rows)


def test_gold_is_replayed_to_the_boundary():
    found = tasks.candidates(
        flow(1), "status_at_snapshot", "status", ["2023-07", "2023-08", "2024-02"], START
    )
    by_ckpt = {t["checkpoint"]: t for t in found}
    assert "2023-07" not in by_ckpt  # nothing visible yet
    first = by_ckpt["2023-08"]
    assert first["cutoff"] == "2023-08-31T23:59:59Z" and first["expected"] == {"value": "Resolved"}
    assert first["temporal_trap"] and first["later_value"] == "Reopened"
    assert first["evidence_ids"] == ["jira:1:change:104:0"]
    assert (
        by_ckpt["2024-02"]["expected"] == {"value": "Reopened"}
        and not by_ckpt["2024-02"]["temporal_trap"]
    )
    history = tasks.candidates(flow(1), "status_transition_history", "status", ["2023-08"], START)[
        0
    ]
    assert history["expected"]["value"][0] == {
        "at": "2023-08-02T10:00:00.250000Z",
        "from": "Open",
        "to": "In Progress",
    }
    assert len(history["evidence_ids"]) == 2 and not history["long_history"]


def test_answers_outside_the_stream_window_are_rejected():
    old = issue(
        2,
        "An old issue with history",
        [("2022-05-01T00:00:00.000+0000", "status", "Open", "Resolved")],
    )
    assert tasks.candidates(old, "status_at_snapshot", "status", ["2023-08"], START) == []
    broken = issue(3, "x", [
        ("2023-08-01T00:00:00.000+0000", "status", "Open", "Resolved"),
        ("2023-08-02T00:00:00.000+0000", "status", "Open", "Closed"),
    ])  # fmt: skip
    assert tasks.candidates(broken, "status_at_snapshot", "status", ["2023-08"], START) == []
    # Smoke: a change between the truncated stream end and the boundary hides the gold.
    end = timestamp("2023-08-15T00:00:00Z")
    assert tasks.candidates(flow(1), "status_at_snapshot", "status", ["2023-08"], START, end) == []
    assert tasks.candidates(flow(1), "assignee_at_snapshot", "assignee", ["2023-08"], START, end)


def corpus():
    issues = {
        f"KAFKA-{n}": flow(n, reopen=n % 3 > 0, month=f"{6 + n % 6:02d}") for n in range(1, 41)
    }
    issues["KAFKA-41"] = flow(41, "Duplicated title of two issues")
    issues["KAFKA-42"] = flow(42, "duplicated  title of two issues")
    return issues


def config(count, coding=2):
    families = {family: (field, count) for family, (field, _) in tasks.FAMILIES.items()}
    return {**tasks.SCOPES["full"], "families": families, "coding": coding, "extra": {}}


def test_selection_is_disjoint_trapped_keyless_and_deterministic():
    issues = corpus()
    chosen, short = tasks.historical(issues, config(4), {"KAFKA-5"})
    again, _ = tasks.historical(issues, config(4), {"KAFKA-5"})
    assert chosen == again and short == {}
    assert len(chosen) == 32 and len({t["issue"] for t in chosen}) == 32
    assert "KAFKA-5" not in {t["issue"] for t in chosen}
    state = [t for t in chosen if not t["family"].endswith("history")]
    assert sum(t["temporal_trap"] for t in state) / len(state) >= tasks.TRAP_SHARE
    for family in tasks.FAMILIES:
        mine = [t for t in chosen if t["family"] == family]
        assert sum(t["keyless"] for t in mine) == 2
    for task in chosen:
        assert "MCP" not in task["prompt"] and "tool" not in task["prompt"].lower()
        assert task["prompt"].startswith(f"Context snapshot cutoff (inclusive): {task['cutoff']}.")
        assert '{"value": <answer>, "evidence": [' in task["prompt"]
        after = tasks.iso(timestamp(task["cutoff"]) + tasks.timedelta(seconds=1))
        assert tasks.stream.boundary(task["checkpoint"]) == after
        public = tasks.public(task)
        assert task["issue"] not in public["id"]
        if task["keyless"]:
            assert (
                task["issue"] not in json.dumps(public)
                and task["summary_at_cutoff"] in task["prompt"]
            )
            assert task["issue"] not in {"KAFKA-41", "KAFKA-42"}  # ambiguous title
        else:
            assert task["issue"] in task["prompt"]


def row(pr, key, created, **extra):
    return {
        "id": f"C-KAFKA-PR-{pr}", "pr": pr, "issue": key, "title": "Fix the thing", "description": "It breaks.",
        "base_sha": "b" * 40, "context_cutoff": created, "pr_created_at": created, "reference_commit": "r" * 40,
        "reference_patch": "SECRET-REFERENCE", "test_patch": "SECRET-TESTS",
        "tests": [{"class": "a.FooTest", "module": ":core"}], **extra,
    }  # fmt: skip


def test_coding_takes_only_admitted_rows_and_hides_tests(tmp_path):
    rows = [
        row(1, "KAFKA-100", "2023-01-20T00:00:00Z"),  # before the first checkpoint
        row(2, "KAFKA-101", "2023-09-10T00:00:00Z"),
        row(3, "KAFKA-101", "2024-01-10T00:00:00Z"),  # same issue again
        row(4, "KAFKA-102", "2024-05-10T00:00:00Z"),  # rejected
        row(5, "KAFKA-103", "2024-06-10T00:00:00Z"),  # historical owns the issue
        row(6, "KAFKA-104", "2025-06-10T00:00:00Z"),
    ]
    bench, results = tmp_path / "bench", tmp_path / "results"
    bench.mkdir()
    (bench / "coding-candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        status = "rejected_no_fail_to_pass" if r["pr"] == 4 else "admitted"
        tasks.save(
            results / f"{r['pr']}.json",
            {"status": status, "fail_to_pass": ["a.FooTest.x"], "pass_to_pass": []},
        )
    everything, passed = tasks.admitted(bench, results)
    assert len(everything) == 6 and [r["pr"] for r in passed] == [1, 2, 3, 5, 6]
    chosen = tasks.coding(passed, config(1, coding=250), {"KAFKA-103"})
    assert [(t["pr"], t["checkpoint"]) for t in chosen] in (
        [(2, "2023-08"), (6, "2025-05")],
        [(3, "2023-12"), (6, "2025-05")],
    )
    public = tasks.public(chosen[0])
    assert set(public) == {
        "id",
        "kind",
        "family",
        "checkpoint",
        "cutoff",
        "prompt",
        "answer_schema",
        "base_sha",
        "tests",
    }
    # Visible tests: the prompt names the test classes; the patch TEXT stays private.
    assert "SECRET" not in json.dumps(public) and "FooTest" in public["prompt"]
    assert "have already been added to the workspace" in public["prompt"]
    assert "MCP" not in public["prompt"]
    assert (
        chosen[0]["fail_to_pass"] == ["a.FooTest.x"] and chosen[0]["test_patch"] == "SECRET-TESTS"
    )
    # Smoke relaxation: any admitted row, pinned to the smoke checkpoint.
    assert [
        t["checkpoint"] for t in tasks.coding(passed[:1], config(1), set(), force="2023-02")
    ] == ["2023-02"]


def test_build_records_the_coding_shortfall(tmp_path):
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "coding-candidates.jsonl").write_text(
        json.dumps(row(9, "KAFKA-7", "2024-01-01T00:00:00Z")) + "\n"
    )
    tasks.SCOPES["tiny"] = config(1)
    try:
        out = tasks.build(
            "tiny",
            corpus(),
            tmp_path / "out",
            bench,
            tmp_path / "none",
            tmp_path / "no-events.jsonl",
        )
    finally:
        del tasks.SCOPES["tiny"]
    manifest = json.loads((out / "manifest.json").read_text())
    public = [json.loads(line) for line in (out / "tasks.public.jsonl").open()]
    private = [json.loads(line) for line in (out / "tasks.private.jsonl").open()]
    assert manifest["coding"] == 0 and manifest["coding_shortfall"]["missing"] == 2
    assert (
        manifest["historical"] == 8 == len(public)
        and "Visible tests" in manifest["test_patch_policy"]
    )
    assert "KAFKA-7" not in {t["issue"] for t in private}  # reserved for coding
    assert sorted(manifest["run_order"]) == sorted(t["id"] for t in public)
    assert all("expected" not in t and "issue" not in t and "evidence_ids" not in t for t in public)


def test_order_balances_every_prefix():
    families = list(tasks.FAMILIES)
    items = [
        {
            "id": f"H{i}",
            "kind": "historical",
            "family": families[i % 8],
            "checkpoint": f"{2023 + i // 8 % 4}-08",
        }
        for i in range(32)
    ] + [
        {
            "id": f"C{i}",
            "kind": "coding",
            "family": "pr_implementation",
            "checkpoint": f"{2023 + i % 4}-08",
        }
        for i in range(32)
    ]
    run = tasks.order(items)
    by_id = {t["id"]: t for t in items}
    assert sorted(run) == sorted(by_id) and run == tasks.order(list(reversed(items)))
    assert [by_id[i]["kind"] for i in run[:6]] == ["coding", "historical"] * 3
    head = [by_id[i] for i in run[:16]]
    assert {t["family"] for t in head if t["kind"] == "historical"} == set(families)
    assert {t["checkpoint"][:4] for t in head if t["kind"] == "coding"} == {
        "2023",
        "2024",
        "2025",
        "2026",
    }
    assert tasks.order(items[:3]) and tasks.order([]) == []
