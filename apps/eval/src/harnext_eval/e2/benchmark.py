"""Deterministic Kafka benchmark curation; never runs an answering agent.

Cached acquisition is separate from deterministic generation. Coding admission
here proves source/test patch applicability, not test execution or task success.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks/kafka-1000-v1"
CORPUS = ROOT / "out/corpus/kafka/raw"
FIELDS = {"assignee", "status", "resolution", "priority"}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def iso(value):
    return value.isoformat().replace("+00:00", "Z")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(value) + "\n")


def read(path):
    return json.loads(path.read_text())


def configuration():
    return yaml.safe_load((BENCH / "benchmark.yaml").read_text())


def inventory():
    cache = BENCH / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    # Canonical frozen raw exports. Rebuild explicitly to change the inventory.
    if (cache / "inventory.json").exists():
        return read(cache / "issues.json"), read(cache / "prs.json"), read(cache / "commits.json")
    issues, prs, commits, hashes = {}, {}, {}, {}
    for path in sorted((CORPUS / "jira").glob("*.json")):
        content = path.read_bytes()
        hashes[str(path.relative_to(CORPUS))] = sha(content)
        for issue in json.loads(content).get("issues", []):
            issues[issue["key"]] = issue
    for path in sorted((CORPUS / "github").glob("*.json")):
        if not (path.name.startswith("pr-") or path.name.startswith("commits-")):
            continue
        content = path.read_bytes()
        hashes[str(path.relative_to(CORPUS))] = sha(content)
        repo = json.loads(content).get("data", {}).get("repository") or {}
        pr = repo.get("pullRequest")
        if pr:
            pr["_raw_path"] = str(path.relative_to(ROOT))
            prs[pr["number"]] = pr
        for commit in (repo.get("object") or {}).get("history", {}).get("nodes", []):
            commits[commit["oid"]] = commit
    save(cache / "issues.json", issues)
    save(cache / "prs.json", list(prs.values()))
    save(cache / "commits.json", list(commits.values()))
    save(
        cache / "inventory.json",
        {"raw_sha256": hashes, "issues": len(issues), "prs": len(prs), "commits": len(commits)},
    )
    return issues, list(prs.values()), list(commits.values())


def histories(issue):
    log = issue["changelog"]
    if log.get("startAt", 0) != 0 or log["total"] != len(log["histories"]):
        raise ValueError("incomplete changelog")
    return sorted(log["histories"], key=lambda h: (timestamp(h["created"]), int(h["id"])))


def changes(issue, field):
    return [
        {
            "at": iso(timestamp(h["created"])),
            "history_id": str(h["id"]),
            "item_index": index,
            "field": field,
            "from": item.get("fromString"),
            "to": item.get("toString"),
            "from_id": item.get("from"),
            "to_id": item.get("to"),
        }
        for h in histories(issue)
        for index, item in enumerate(h["items"])
        if item["field"].lower() == field.lower()
    ]


def body_at(issue, cutoff):
    """Undo later edits, verifying the text chain against the exported final value."""
    result = {}
    for field in ("summary", "description"):
        value = issue["fields"].get(field)
        for change in reversed(changes(issue, field)):
            if timestamp(change["at"]) <= cutoff:
                break
            if value != change["to"]:
                raise ValueError(f"{field} reverse chain inconsistent")
            value = change["from"]
        result[field] = value or ""
    return result


def evidence(issue, change):
    value = {"issue": issue["key"], **change}
    text = canonical(value)
    return {
        "id": f"jira:{issue['key']}:{change['history_id']}:{change['item_index']}",
        "at": change["at"],
        "text": text,
        "sha256": sha(text),
        "url": f"https://issues.apache.org/jira/browse/{issue['key']}?page=com.atlassian.jira.plugin.system.issuetabpanels:changehistory-tabpanel",
        "locator": f"changelog history {change['history_id']}, item {change['item_index']}",
    }


def ordered(values, seed, key):
    return sorted(values, key=lambda v: sha(seed + "|" + str(key(v))))


def stratified(values, seed, key, stratum):
    groups = defaultdict(list)
    for value in ordered(values, seed, key):
        groups[stratum(value)].append(value)
    result = []
    while any(groups.values()):
        for group in sorted(groups):
            if groups[group]:
                result.append(groups[group].pop())
    return result


def coding_candidates(issues, prs, commits, config):
    timeline = sorted((timestamp(c["committedDate"]), c["oid"]) for c in commits)
    dates = [item[0] for item in timeline]
    values = []
    for pr in prs:
        keys = set(re.findall(r"KAFKA-\d+", pr["title"]))
        files = pr["files"]
        paths = [f["path"] for f in files["nodes"]]
        if not (
            pr["merged"]
            and pr["baseRefName"] == "trunk"
            and len(keys) == 1
            and not files["pageInfo"]["hasNextPage"]
            and pr.get("mergeCommit")
            and 2 <= len(paths) <= config["coding_max_changed_files"]
            and all("/src/main/" in p or "/src/test/" in p for p in paths)
            and all(p.endswith((".java", ".scala")) for p in paths)
            and any("/src/test/" in p for p in paths)
            and any("/src/main/" in p for p in paths)
            and sum(f["additions"] + f["deletions"] for f in files["nodes"])
            <= config["coding_max_changed_lines"]
        ):
            continue
        key = next(iter(keys))
        if key not in issues:
            continue
        issue = issues[key]
        cutoff = timestamp(pr["createdAt"]) - timedelta(seconds=1)
        position = bisect.bisect_right(dates, cutoff) - 1
        if position < 0:
            continue
        base_time, base_sha = timeline[position]
        if timestamp(issue["fields"]["created"]) > base_time:
            continue
        try:
            body = body_at(issue, base_time)
        except ValueError:
            continue
        if len(body["description"]) < config["coding_min_issue_characters"]:
            continue
        # Reject issue text explicitly pointing to this future solution.
        if f"/pull/{pr['number']}" in body["description"]:
            continue
        values.append(
            {"pr": pr, "issue": key, "base_sha": base_sha, "cutoff": iso(base_time), "body": body}
        )
    return stratified(
        values, config["seed"], lambda v: v["pr"]["number"], lambda v: v["pr"]["createdAt"][:4]
    )


def acquire_json(endpoint, path):
    if path.exists():
        return read(path)
    error = "no response"
    for attempt in range(3):
        process = subprocess.run(
            ["gh", "api", endpoint], capture_output=True, text=True, timeout=90
        )
        if process.returncode == 0:
            value = json.loads(process.stdout)
            save(path, value)
            return value
        error = process.stderr[:180]
        time.sleep(1 + attempt)
    raise ValueError(f"GitHub acquisition failed: {endpoint}: {error}")


def acquire_raw(url, path):
    if path.exists():
        return path.read_bytes()
    for attempt in range(3):
        try:
            with urlopen(
                Request(url, headers={"User-Agent": "harnext-thesis-benchmark"}), timeout=45
            ) as response:
                content = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            return content
        except HTTPError as exc:
            if exc.code == 404:
                raise ValueError("base file absent") from exc
        except (URLError, TimeoutError):
            pass
        time.sleep(attempt + 1)
    raise ValueError(f"raw acquisition failed: {url}")


def diff_part(file):
    path = file["filename"]
    old = "/dev/null" if file["status"] == "added" else f"a/{path}"
    new = "/dev/null" if file["status"] == "removed" else f"b/{path}"
    return f"diff --git a/{path} b/{path}\n--- {old}\n+++ {new}\n{file['patch']}\n"


def check_patch(workspace, patch, apply=False):
    return subprocess.run(
        ["git", "apply", "--whitespace=nowarn", *([] if apply else ["--check"]), "-"],
        input=patch,
        text=True,
        capture_output=True,
        cwd=workspace,
        timeout=30,
    )


def acquire_coding(candidate):
    pr, key = candidate["pr"], candidate["issue"]
    cache = BENCH / "cache/coding" / str(pr["number"])
    result_path = cache / "admission.json"
    if result_path.exists():
        return read(result_path)
    cache.mkdir(parents=True, exist_ok=True)
    try:
        commit = acquire_json(
            f"repos/apache/kafka/commits/{pr['mergeCommit']['oid']}", cache / "commit.json"
        )
        files = commit["files"]
        if len(files) != len(pr["files"]["nodes"]) or {f["filename"] for f in files} != {
            f["path"] for f in pr["files"]["nodes"]
        }:
            raise ValueError("merge diff and PR file inventory differ")
        if not all(
            f.get("patch") and f["status"] in {"modified", "added", "removed"} for f in files
        ):
            raise ValueError("missing textual patch or unsupported rename")
        # Verify chronology independently from public PR metadata.
        if timestamp(commit["commit"]["committer"]["date"]) <= timestamp(candidate["cutoff"]):
            raise ValueError("target solution does not follow snapshot")
        base_files = {}
        for file in files:
            if file["status"] != "added":
                path = file["filename"]
                content = acquire_raw(
                    f"https://raw.githubusercontent.com/apache/kafka/{candidate['base_sha']}/{path}",
                    cache / "base" / path,
                )
                base_files[path] = content.decode("utf-8")
        test_patch = "".join(diff_part(f) for f in files if "/src/test/" in f["filename"])
        reference_patch = "".join(diff_part(f) for f in files if "/src/main/" in f["filename"])
        with tempfile.TemporaryDirectory(prefix="harnext-pr-admission-") as workspace:
            for path, content in base_files.items():
                target = Path(workspace) / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
            for patch in (test_patch, reference_patch):
                checked = check_patch(workspace, patch)
                if checked.returncode:
                    raise ValueError(
                        "patch does not apply to pre-PR snapshot: " + checked.stderr[:240]
                    )
                checked = check_patch(workspace, patch, apply=True)
                if checked.returncode:
                    raise ValueError("patch application failed")
        test_paths = [f["filename"] for f in files if "/src/test/" in f["filename"]]
        tests = []
        for path in test_paths:
            module, relative = path.split("/src/test/", 1)
            language, relative = relative.split("/", 1)
            tests.append(
                {
                    "path": path,
                    "module": module,
                    "class": relative.rsplit(".", 1)[0].replace("/", "."),
                    "language": language,
                }
            )
        commands = [
            [
                "./gradlew",
                f":{t['module'].replace('/', ':')}:test",
                "--tests",
                t["class"],
                "--no-daemon",
            ]
            for t in tests
        ]
        result = {
            "status": "patch_verified_tests_pending",
            "candidate": candidate,
            "commit_sha": commit["sha"],
            "commit_parent_sha": commit["parents"][0]["sha"],
            "commit_time": commit["commit"]["committer"]["date"],
            "test_patch": test_patch,
            "reference_patch": reference_patch,
            "test_patch_sha256": sha(test_patch),
            "reference_patch_sha256": sha(reference_patch),
            "base_file_sha256": {p: sha(v) for p, v in base_files.items()},
            "tests": tests,
            "test_commands": commands,
            "patch_check": "both patches apply sequentially to fetched files at pre-PR base SHA",
            "test_execution": "not_run",
            "fail_to_pass": None,
            "pass_to_pass": None,
        }
    except Exception as exc:
        result = {"status": "rejected", "pr": pr["number"], "issue": key, "reason": str(exc)}
    save(result_path, result)
    return result


def acquire(limit=None):
    config = configuration()
    issues, prs, commits = inventory()
    candidates = coding_candidates(issues, prs, commits, config)
    save(
        BENCH / "cache/candidate_inventory.json",
        {"eligible": len(candidates), "candidates": candidates},
    )
    selected, seen, results = [], set(), []
    cap = limit or config["coding_candidate_limit"]
    # Stable batches, ordered results: completion timing never selects tasks.
    for offset in range(0, min(cap, len(candidates)), 8):
        with ThreadPoolExecutor(max_workers=4) as pool:
            batch = list(pool.map(acquire_coding, candidates[offset : offset + 8]))
        results.extend(batch)
        for result in batch:
            if result["status"] != "rejected" and result["candidate"]["issue"] not in seen:
                selected.append(result)
                seen.add(result["candidate"]["issue"])
        print(
            canonical(
                {
                    "checked": len(results),
                    "admitted_static": len(selected),
                    "rejected": sum(r["status"] == "rejected" for r in results),
                }
            ),
            flush=True,
        )
        if len(selected) >= config["coding_count"]:
            break
    save(
        BENCH / "cache/selection.json",
        {
            "selected": selected[: config["coding_count"]],
            "attempts": results,
            "config_sha256": sha(canonical(config)),
        },
    )
    if len(selected) < config["coding_count"]:
        raise ValueError(
            f"Only {len(selected)} coding tasks pass static admission; no synthetic padding"
        )


def historical_task(issue, family, config, target_value=...):
    field = {
        "assignee_at_snapshot": "assignee",
        "status_at_snapshot": "status",
        "resolution_at_snapshot": "resolution",
        "closed_at_snapshot": "status",
        "priority_at_snapshot": "priority",
        "status_transition_history": "status",
        "assignment_history": "assignee",
        "resolution_transition_history": "resolution",
    }[family]
    rows = changes(issue, field)
    if not rows:
        return None
    # Restrict to internally consistent full field history, never silently repair gaps.
    if any(b["from"] != a["to"] for a, b in zip(rows, rows[1:], strict=False)):
        return None
    options = [
        r
        for r in rows
        if config["historical_year_min"] <= timestamp(r["at"]).year <= config["historical_year_max"]
    ]
    if target_value is not ...:
        options = [
            r
            for r in options
            if (r["to"] == "Closed" if family == "closed_at_snapshot" else r["to"]) == target_value
        ]
    if family.endswith("history"):
        longer = [r for r in options if sum(v["at"] <= r["at"] for v in rows) >= 3]
        options = longer or options
    if not options:
        return None
    chosen = ordered(options, config["seed"] + family, lambda r: r["history_id"])[0]
    cutoff = timestamp(chosen["at"]) + timedelta(microseconds=1)
    visible = [r for r in rows if timestamp(r["at"]) <= cutoff]
    if family.endswith("history") and len(visible) < 2:
        return None
    key = issue["key"]
    if family == "assignee_at_snapshot":
        question = f"Who was assigned to {key} at the snapshot? Return the exact assignee display name in the latest recorded assignment, or null for unassigned."
        answer = visible[-1]["to"]
    elif family == "status_at_snapshot":
        question = f"What was the recorded workflow status of {key} at the snapshot? Return the exact status label."
        answer = visible[-1]["to"]
    elif family == "resolution_at_snapshot":
        question = f"What was the recorded resolution of {key} at the snapshot? Return the exact resolution label, or null if unresolved."
        answer = visible[-1]["to"]
    elif family == "closed_at_snapshot":
        question = f"Was {key} in the exact workflow status Closed at the snapshot? Return true only for Closed; Resolved is a distinct status and must return false."
        answer = visible[-1]["to"] == "Closed"
    elif family == "priority_at_snapshot":
        question = f"What was the recorded priority of {key} at the snapshot? Return the exact priority label."
        answer = visible[-1]["to"]
    else:
        question = f"List every recorded {field} change for {key} through the snapshot in chronological order. Return an array of objects with at, from, and to. Preserve null values and repeated states; break equal-time ties by numeric changelog history ID and item index."
        answer = [{k: r[k] for k in ("at", "from", "to")} for r in visible]
    proof = [evidence(issue, r) for r in visible]
    # Every pre-cutoff transition in other tracked fields is part of the context, not gold.
    context = [
        evidence(issue, r)
        for f in sorted(FIELDS)
        for r in changes(issue, f)
        if timestamp(r["at"]) <= cutoff
    ]
    prompt = f'Context snapshot cutoff (inclusive): {iso(cutoff)}.\n\n{question}\n\nUse only the Harnext context-engine MCP tools. You have no shell, native filesystem, browser or other data source. Return JSON {{"value": <answer>, "evidence_ids": [<source IDs>]}}. Do not substitute today\'s issue state.'
    required = proof if family.endswith("history") else [proof[-1]]
    return {
        "id": f"H-{family}-{key}",
        "kind": "historical",
        "family": family,
        "issue": key,
        "repository": "apache/kafka",
        "lineage": key,
        "cutoff": iso(cutoff),
        "title": question,
        "agent_prompt": prompt,
        "tool_profile": "historical_mcp_only",
        "expected_answer": {"value": answer},
        "answer_type": type(answer).__name__,
        "grading": "typed_exact_json; ordered history arrays; evidence exposure scored separately",
        "evidence": proof,
        "required_evidence_ids": [p["id"] for p in required],
        "context_events": context,
        "source_url": f"https://issues.apache.org/jira/browse/{key}",
        "difficulty": "multi_event" if family.endswith("history") else "state_lookup",
        "visible_change_count": len(context),
        "has_withheld_later_change": any(timestamp(r["at"]) > cutoff for r in rows),
        "admission": {
            "changelog_complete": True,
            "field_chain_consistent": True,
            "gold_recomputed": True,
            "review": "pending",
            "execution_ready": False,
        },
        "time_caveat": "Snapshot uses source-recorded changelog timestamps; historical ingestion/observation times are unavailable.",
    }


def coding_task(result, issues):
    candidate = result["candidate"]
    key, pr = candidate["issue"], candidate["pr"]
    cutoff = timestamp(candidate["cutoff"])
    issue = issues[key]
    description = candidate["body"]["description"]
    tests = result["test_commands"]
    prompt = (
        f"Implement the following issue in the supplied apache/kafka workspace.\n"
        f"Repository base commit: {candidate['base_sha']}\nContext snapshot: {candidate['cutoff']}\n\n"
        f"Issue: {key}\nTitle: {candidate['body']['summary']}\n\n{description}\n\n"
        "The supplied workspace contains the base source and the visible acceptance-test changes. "
        "Implement the behavior, preserve existing behavior, and run the supplied tests. "
        "Use Harnext MCP for historical context and the sandbox shell to inspect/edit code and run tests. "
        "Network, future Git history, the reference solution and private evaluator files are unavailable. "
        "Do not change the grading tests. Submit a unified implementation patch and a brief test summary.\n\n"
        "Proposed test commands (environment admission pending):\n"
        + "\n".join(" ".join(c) for c in tests)
    )
    context = [
        evidence(issue, r)
        for f in sorted(FIELDS)
        for r in changes(issue, f)
        if timestamp(r["at"]) <= cutoff
    ]
    body_evidence = {
        "id": f"jira:{key}:body:{candidate['base_sha'][:12]}",
        "at": candidate["cutoff"],
        "text": canonical({"issue": key, **candidate["body"]}),
        "url": f"https://issues.apache.org/jira/browse/{key}",
        "locator": "body reconstructed by reversing later changelog edits",
    }
    body_evidence["sha256"] = sha(body_evidence["text"])
    context.append(body_evidence)
    return {
        "id": f"C-KAFKA-PR-{pr['number']}",
        "kind": "coding",
        "family": "historical_pr_implementation",
        "repository": "apache/kafka",
        "issue": key,
        "lineage": key,
        "cutoff": candidate["cutoff"],
        "title": candidate["body"]["summary"],
        "agent_prompt": prompt,
        "tool_profile": "coding_mcp_shell",
        "base_sha": candidate["base_sha"],
        "base_url": f"https://github.com/apache/kafka/tree/{candidate['base_sha']}",
        "pr_number": pr["number"],
        "pr_title": pr["title"],
        "pr_created_at": pr["createdAt"],
        "pr_merged_at": pr["mergedAt"],
        "source_url": f"https://issues.apache.org/jira/browse/{key}",
        "reference_url": f"https://github.com/apache/kafka/pull/{pr['number']}",
        "reference_commit_url": f"https://github.com/apache/kafka/commit/{result['commit_sha']}",
        "reference_commit": result["commit_sha"],
        "expected_answer": "Any implementation passing admitted acceptance and regression tests; reference patch is not the unique correct answer.",
        "reference_patch": result["reference_patch"],
        "test_patch": result["test_patch"],
        "reference_patch_sha256": result["reference_patch_sha256"],
        "test_patch_sha256": result["test_patch_sha256"],
        "base_file_sha256": result["base_file_sha256"],
        "tests": result["tests"],
        "test_commands": tests,
        "changed_files": pr["files"]["nodes"],
        "context_events": context,
        "evidence": context,
        "required_evidence_ids": [],
        "retrieval_gold_status": "requires independent context-dependence review; issue text already appears in prompt",
        "grading": {
            "primary": "all admitted fail-to-pass and pass-to-pass tests pass in pristine evaluator workspace",
            "fail_to_pass": None,
            "pass_to_pass": None,
            "reference_comparison": "secondary changed-file overlap and diff statistics; never exact-patch equality as correctness",
        },
        "difficulty": "unmeasured_real_change",
        "visible_change_count": len(context),
        "admission": {
            "patch_applicability": "passed",
            "pre_pr_snapshot": True,
            "test_execution": "not_run",
            "base_fails": None,
            "reference_passes": None,
            "environment_pinned": False,
            "review": "pending",
            "execution_ready": False,
        },
        "time_caveat": "Base commit precedes target PR creation. Issue text is reconstructed at base time; later PR/test information is evaluator-held except explicitly supplied visible tests.",
    }


def generate():
    config = configuration()
    issues, prs, commits = inventory()
    selection = read(BENCH / "cache/selection.json")
    if selection["config_sha256"] != sha(canonical(config)):
        raise ValueError("selection configuration changed; reacquire deterministically")
    if len(selection["selected"]) != config["coding_count"]:
        raise ValueError("coding selection incomplete")
    tasks = [coding_task(r, issues) for r in selection["selected"]]
    used = {t["issue"] for t in tasks}
    for family in config["historical_families"]:
        candidates = []
        for key, issue in issues.items():
            if key in used:
                continue
            targets = [...]
            if family in {"status_at_snapshot", "resolution_at_snapshot", "closed_at_snapshot"}:
                field = "resolution" if family == "resolution_at_snapshot" else "status"
                targets = sorted(
                    {
                        r["to"] == "Closed" if family == "closed_at_snapshot" else r["to"]
                        for r in changes(issue, field)
                    },
                    key=canonical,
                )
            for target in targets:
                try:
                    task = historical_task(issue, family, config, target)
                except (ValueError, KeyError):
                    continue
                if task:
                    candidates.append(task)
        candidates = stratified(
            candidates,
            config["seed"] + family,
            lambda t: t["id"] + t["cutoff"],
            lambda t, family=family: (
                (canonical(t["expected_answer"]["value"]), t["cutoff"][:4])
                if family in {"status_at_snapshot", "resolution_at_snapshot", "closed_at_snapshot"}
                else (t["cutoff"][:4],)
            ),
        )
        chosen, chosen_issues, balance = [], set(), Counter()
        for candidate in candidates:
            value = canonical(candidate["expected_answer"]["value"])
            if candidate["issue"] in chosen_issues:
                continue
            if (
                family == "closed_at_snapshot"
                and balance[value] >= config["historical_per_family"] // 2
            ):
                continue
            chosen.append(candidate)
            chosen_issues.add(candidate["issue"])
            balance[value] += 1
            if len(chosen) == config["historical_per_family"]:
                break
        if len(chosen) < config["historical_per_family"]:
            raise ValueError(f"not enough independent eligible issues for {family}: {len(chosen)}")
        tasks.extend(chosen)
        used.update(t["issue"] for t in chosen)
    # Connected issue/PR lineage, including multi-issue PRs, prevents obvious split leakage.
    parents = {key: key for key in issues}

    def root(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for pr in prs:
        keys = sorted(set(re.findall(r"KAFKA-\d+", pr["title"])))
        for key in keys[1:]:
            a, b = root(keys[0]), root(key)
            parents[max(a, b)] = min(a, b)
    for task in tasks:
        task["lineage"] = root(task["issue"])
        task["split"] = (
            "confirmation_candidate"
            if int(sha(config["seed"] + root(task["issue"]))[:8], 16) / 2**32
            < config["confirmation_fraction"]
            else "development"
        )
        task["tools"] = config["tool_profiles"][task["tool_profile"]]
        task["agent_runs"] = []
    tasks.sort(key=lambda t: (t["kind"] != "historical", t["family"], t["id"]))
    manifest = {
        "protocol": config["protocol"],
        "benchmark": "Kafka Context Benchmark — 1,000 review instances",
        "config_sha256": sha(canonical(config)),
        "tasks_sha256": sha(canonical(tasks)),
        "source_inventory_sha256": sha((BENCH / "cache/inventory.json").read_bytes()),
        "task_count": len(tasks),
        "unique_issues": len(used),
        "unique_lineages": len({t["lineage"] for t in tasks}),
        "families": dict(Counter(t["family"] for t in tasks)),
        "splits": dict(Counter(t["split"] for t in tasks)),
        "years": dict(sorted(Counter(t["cutoff"][:4] for t in tasks).items())),
        "repository": "apache/kafka",
        "review_status": "pending",
        "experiment_runs": 0,
        "coding_admission": "200 patch-verified candidates; base/reference test execution pending",
        "limitations": [
            "Single-repository benchmark; do not infer cross-repository generalization.",
            "Most historical state questions are calibration tasks, not established hard reasoning questions.",
            "Recorded event times proxy observation times; real ingestion clocks unavailable.",
            "Current harvested descriptions are reversed through edit history; no archived web-page proof is implied.",
            "Coding tests are visible; hidden independent tests and fail/pass admission remain pending.",
            "Public historical PRs may be present in model training data.",
            "Review SPA contains gold and reference solutions: evaluator only, never agent-accessible.",
            "Confirmation split is a candidate split, not sealed against the human reviewer viewing it.",
            "No harness executions authorized until user review and technical admission complete.",
        ],
    }
    data = BENCH / "data"
    save(data / "manifest.json", manifest)
    save(data / "review-bundle.json", {"manifest": manifest, "tasks": tasks})
    with (data / "tasks.private.jsonl").open("w") as handle:
        for task in tasks:
            handle.write(canonical(task) + "\n")
    with (data / "tasks.public.jsonl").open("w") as handle:
        for task in tasks:
            public = {
                k: task[k]
                for k in ("id", "kind", "family", "cutoff", "agent_prompt", "tool_profile", "tools")
            }
            if task["kind"] == "coding":
                public.update(
                    base_sha=task["base_sha"],
                    test_patch=task["test_patch"],
                    test_commands=task["test_commands"],
                )
            handle.write(canonical(public) + "\n")
    # Shared safe event corpus includes distractor issues; no exported current description/status injection.
    with (data / "context-sources.jsonl").open("w") as handle:
        for key, issue in sorted(issues.items()):
            try:
                records = [
                    evidence(issue, r) for field in sorted(FIELDS) for r in changes(issue, field)
                ]
            except ValueError:
                continue
            for record in sorted(records, key=lambda r: (r["at"], r["id"])):
                change = json.loads(record["text"])
                predicate = {
                    "assignee": "assigned_to",
                    "status": "has_status",
                    "resolution": "has_resolution",
                    "priority": "has_priority",
                }[change["field"]]
                target = (
                    ("person:" + (change["to_id"] or "UNASSIGNED"))
                    if change["field"] == "assignee"
                    else str(change["to"] or "UNSET")
                )
                source = {
                    "id": record["id"],
                    "entity": f"issue:{key}",
                    "observed_at": record["at"],
                    "valid_at": record["at"],
                    "text": record["text"],
                    "facts": [
                        {
                            "id": record["id"],
                            "subject": f"issue:{key}",
                            "predicate": predicate,
                            "object": target,
                            "text": record["text"],
                            "quote": record["text"],
                            "topic": "issue_history",
                        }
                    ],
                }
                handle.write(canonical(source) + "\n")
        by_number = {pr["number"]: pr for pr in prs}
        for commit in sorted(commits, key=lambda c: (c["committedDate"], c["oid"])):
            for associated in commit.get("associatedPullRequests", {}).get("nodes", []):
                pr = by_number.get(associated["number"])
                if (
                    not pr
                    or not pr["merged"]
                    or pr["mergeCommit"]["oid"] != commit["oid"]
                    or pr["files"]["pageInfo"]["hasNextPage"]
                ):
                    continue
                text = canonical(
                    {
                        "commit": commit["oid"],
                        "committed_at": commit["committedDate"],
                        "message": commit["messageHeadline"],
                        "pr": pr["number"],
                        "issue_keys": sorted(
                            set(re.findall(r"KAFKA-\d+", commit["messageHeadline"]))
                        ),
                        "changed_files": [f["path"] for f in pr["files"]["nodes"]],
                    }
                )
                facts = []
                for key in sorted(set(re.findall(r"KAFKA-\d+", commit["messageHeadline"]))):
                    facts.append(
                        {
                            "id": f"commit:{commit['oid']}:issue:{key}",
                            "subject": f"pr:{pr['number']}",
                            "predicate": "references_issue",
                            "object": f"issue:{key}",
                            "text": f"PR #{pr['number']} at commit {commit['oid']} references {key}: {commit['messageHeadline']}",
                            "quote": key,
                            "topic": "implementation_history",
                        }
                    )
                for index, file in enumerate(pr["files"]["nodes"]):
                    facts.append(
                        {
                            "id": f"commit:{commit['oid']}:file:{index}",
                            "subject": f"pr:{pr['number']}",
                            "predicate": "changes_file",
                            "object": f"file:{file['path']}",
                            "text": f"PR #{pr['number']} at commit {commit['oid']} changed {file['path']}.",
                            "quote": file["path"],
                            "topic": "implementation_history",
                        }
                    )
                if facts:
                    handle.write(
                        canonical(
                            {
                                "id": f"commit:{commit['oid']}",
                                "entity": f"pr:{pr['number']}",
                                "observed_at": commit["committedDate"],
                                "valid_at": commit["committedDate"],
                                "text": text,
                                "facts": facts,
                            }
                        )
                        + "\n"
                    )
    with (data / "context-sources.jsonl").open("rb") as context_file:
        manifest["context_sources_sha256"] = hashlib.file_digest(context_file, "sha256").hexdigest()
    manifest["generator_sha256"] = sha(Path(__file__).read_bytes())
    save(data / "manifest.json", manifest)
    save(data / "review-bundle.json", {"manifest": manifest, "tasks": tasks})
    save(
        data / "acquisition-report.json",
        {
            "attempts": [
                {k: v for k, v in r.items() if k in {"status", "pr", "issue", "reason"}}
                for r in selection["attempts"]
            ]
        },
    )
    print(canonical(manifest), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inventory", "acquire", "generate"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "inventory":
        issues, prs, commits = inventory()
        print(canonical({"issues": len(issues), "prs": len(prs), "commits": len(commits)}))
    elif args.command == "acquire":
        acquire(args.limit)
    else:
        generate()


if __name__ == "__main__":
    main()
