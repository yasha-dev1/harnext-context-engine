"""Independent frozen-data audit and exact-answer scorer for the review benchmark."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict

from harnext_eval.e2.benchmark import BENCH, canonical, read, save, sha, timestamp


def typed_equal(actual, expected):
    """No bool/int coercion, unordered-history credit, or stringified JSON credit."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            typed_equal(actual[k], v) for k, v in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            typed_equal(a, e) for a, e in zip(actual, expected, strict=True)
        )
    return actual == expected


def score_answer(task, answer):
    if task["kind"] != "historical":
        raise ValueError(
            "Coding correctness requires admitted test execution, not JSON or patch equality"
        )
    return {
        "value_correct": isinstance(answer, dict)
        and "value" in answer
        and typed_equal(answer["value"], task["expected_answer"]["value"]),
        "retrieval_scored": False,
        "note": "Score retrieval separately from actual MCP response spans; supplied evidence IDs are not proof.",
    }


def audit():
    bundle = read(BENCH / "data/review-bundle.json")
    tasks, manifest = bundle["tasks"], bundle["manifest"]
    archived = None
    archive_path = BENCH / "data/audit-inputs.json.gz"
    if archive_path.exists():
        archived = json.loads(gzip.decompress(archive_path.read_bytes()))
    issues = archived["issues"] if archived else read(BENCH / "cache/issues.json")
    failures, checks = [], Counter()

    def check(condition, label, task_id="dataset"):
        checks[label] += 1
        if not condition:
            failures.append({"task": task_id, "check": label})

    check(len(tasks) == 1000, "exact_1000")
    check(len({t["id"] for t in tasks}) == 1000, "unique_ids")
    check(len({t["issue"] for t in tasks}) == 1000, "unique_issues")
    check(sha(canonical(tasks)) == manifest["tasks_sha256"], "dataset_hash")
    check(Counter(t["kind"] for t in tasks) == {"historical": 800, "coding": 200}, "kind_counts")
    check(
        Counter(t["expected_answer"]["value"] for t in tasks if t["family"] == "closed_at_snapshot")
        == {False: 50, True: 50},
        "balanced_closed_answers",
    )
    lineages = defaultdict(set)
    for task in tasks:
        tid = task["id"]
        cutoff = timestamp(task["cutoff"])
        lineages[task["lineage"]].add(task["split"])
        issue = issues[task["issue"]]
        hs = sorted(
            issue["changelog"]["histories"], key=lambda h: (timestamp(h["created"]), int(h["id"]))
        )
        check(issue["changelog"]["total"] == len(hs), "complete_changelog", tid)
        check(timestamp(issue["fields"]["created"]) <= cutoff, "issue_precedes_cutoff", tid)
        check(
            not task["admission"]["execution_ready"] and task["agent_runs"] == [],
            "execution_held",
            tid,
        )
        for ev in task["evidence"]:
            check(timestamp(ev["at"]) <= cutoff, "evidence_precedes_cutoff", tid)
            check(sha(ev["text"]) == ev["sha256"], "evidence_text_hash", tid)
        check(
            task["tools"]["network"] is False
            and task["tools"]["browser"] is False
            and task["tools"]["other_mcp_servers"] is False,
            "external_data_forbidden",
            tid,
        )
        if task["kind"] == "historical":
            check(
                task["tools"]["shell"] is False
                and task["tools"]["native_file_access"] is False
                and "shell" not in task["tools"]["allowed"],
                "historical_mcp_only",
                tid,
            )
            field = {
                "assignee_at_snapshot": "assignee",
                "status_at_snapshot": "status",
                "resolution_at_snapshot": "resolution",
                "closed_at_snapshot": "status",
                "priority_at_snapshot": "priority",
                "status_transition_history": "status",
                "assignment_history": "assignee",
                "resolution_transition_history": "resolution",
            }[task["family"]]
            raw = []
            for h in hs:
                for index, item in enumerate(h["items"]):
                    if item["field"].lower() == field:
                        raw.append((h, index, item))
            visible = [(h, i, item) for h, i, item in raw if timestamp(h["created"]) <= cutoff]
            if task["family"].endswith("history"):
                expected = [
                    {
                        "at": timestamp(h["created"]).isoformat().replace("+00:00", "Z"),
                        "from": item.get("fromString"),
                        "to": item.get("toString"),
                    }
                    for h, _, item in visible
                ]
            else:
                later = [item for h, _, item in raw if timestamp(h["created"]) > cutoff]
                # Independent oracle: next transition's prior value where available.
                expected = later[0].get("fromString") if later else visible[-1][2].get("toString")
                if task["family"] == "closed_at_snapshot":
                    expected = expected == "Closed"
            check(
                typed_equal(expected, task["expected_answer"]["value"]),
                "gold_recomputed_from_raw",
                tid,
            )
            check(len(visible) == len(task["evidence"]), "complete_visible_field_evidence", tid)
            for ev, (h, index, item) in zip(task["evidence"], visible, strict=False):
                proof = json.loads(ev["text"])
                check(
                    proof["history_id"] == str(h["id"])
                    and proof["item_index"] == index
                    and proof["from"] == item.get("fromString")
                    and proof["to"] == item.get("toString"),
                    "proof_matches_raw_changelog",
                    tid,
                )
        else:
            check(
                task["tools"]["shell"] is True
                and task["tools"]["native_file_access"] == "workspace_only",
                "coding_shell_scoped",
                tid,
            )
            check(
                cutoff < timestamp(task["pr_created_at"]) < timestamp(task["pr_merged_at"]),
                "base_before_future_pr",
                tid,
            )
            check(len(task["base_sha"]) == 40, "pinned_base_sha", tid)
            check(
                sha(task["reference_patch"]) == task["reference_patch_sha256"]
                and sha(task["test_patch"]) == task["test_patch_sha256"],
                "patch_hashes",
                tid,
            )
            check(
                task["reference_patch"] not in task["agent_prompt"]
                and task["reference_url"] not in task["agent_prompt"],
                "reference_not_in_prompt",
                tid,
            )
            check(
                task["admission"]["test_execution"] == "not_run"
                and task["grading"]["fail_to_pass"] is None,
                "test_status_honest",
                tid,
            )
            cache = BENCH / "cache/coding" / str(task["pr_number"])
            archived_case = archived["coding"][str(task["pr_number"])] if archived else None
            commit = archived_case["commit"] if archived_case else read(cache / "commit.json")
            check(commit["sha"] == task["reference_commit"], "reference_commit_identity", tid)
            for file in commit["files"]:
                lines = file["patch"].splitlines()
                check(
                    sum(line.startswith("+") for line in lines) == file["additions"]
                    and sum(line.startswith("-") for line in lines) == file["deletions"],
                    "patch_not_truncated",
                    tid,
                )
            for path, expected_hash in task["base_file_sha256"].items():
                check(
                    sha(
                        archived_case["base_files"][path]
                        if archived_case
                        else (cache / "base" / path).read_bytes()
                    )
                    == expected_hash,
                    "base_file_hash",
                    tid,
                )
    check(all(len(s) == 1 for s in lineages.values()), "lineage_not_split")
    public = [
        json.loads(line) for line in (BENCH / "data/tasks.public.jsonl").read_text().splitlines()
    ]
    check(len(public) == 1000, "public_task_count")
    for record in public:
        check(
            not {
                "expected_answer",
                "reference_patch",
                "reference_url",
                "reference_commit",
                "evidence",
                "grading",
                "context_events",
            }.intersection(record),
            "public_fields_exclude_gold",
            record["id"],
        )
    report = {
        "passed": not failures,
        "checks": dict(checks),
        "check_count": sum(checks.values()),
        "failures": failures,
        "tasks_sha256": manifest["tasks_sha256"],
        "historical": 800,
        "coding": 200,
        "test_execution": "not_run",
        "agent_execution": "not_run",
        "scope": "source/chronology/patch-integrity/data-isolation audit, not coding test admission",
    }
    save(BENCH / "data/validation.json", report)
    print(canonical(report))
    if failures:
        raise ValueError(f"{len(failures)} benchmark audit failures")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate"])
    parser.parse_args()
    audit()


if __name__ == "__main__":
    main()
