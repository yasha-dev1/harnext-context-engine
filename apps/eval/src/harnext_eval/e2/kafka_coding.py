"""Kafka coding tasks for benchmark v2: local-clone selection and executed admission.

Selection reads the frozen PR/issue inventory and a bare apache/kafka clone; it makes
no network or model calls. Admission executes the PR's own tests in Docker at the
merge parent (must fail) and with the golden patch (must pass). Every candidate's
outcome is saved atomically, so an interrupted sweep resumes where it stopped.
"""

from __future__ import annotations

import argparse
import os
import queue
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

from harnext_eval.e2.benchmark import (
    ROOT,
    body_at,
    canonical,
    inventory,
    iso,
    read,
    save,
    sha,
    stratified,
    timestamp,
)

REPO = ROOT / "out/kafka-repo/kafka.git"
WORK = ROOT / "out/kafka-admission-v2"
BENCH = ROOT / "benchmarks/kafka-500-v2"
INIT = ROOT / "admission/kafka-legacy.init.gradle"
WINDOW = ("2023-01-01T00:00:00Z", "2026-06-30T23:59:59Z")
LIMITS = {"files": (2, 8), "lines": 600, "issue_characters": 100}
SEED = "kafka-500-v2"
TEST_CLASS = re.compile(r"(Test|Tests|IT|Suite|Spec)\.(java|scala)$")
PACKAGE = re.compile(r"^\s*package\s+([\w.]+)", re.MULTILINE)
COMPILE_TEST = re.compile(r"Task (:\S+):compileTest(Java|Scala) FAILED")
FAILED_TASK = re.compile(r"Task (:\S+) FAILED")
CACHES: queue.Queue[int] = queue.Queue()  # one Gradle user home per concurrent worker
CACHES.put(0)


def git(*args, text=True, check=True, **kwargs):
    return subprocess.run(
        ["git", "--git-dir", str(REPO), *args],
        capture_output=True,
        text=text,
        check=check,
        **kwargs,
    ).stdout


def is_test(path):
    return "/src/test/" in path


def diff(parent, merge, paths):
    return git("diff", "--no-renames", "--full-index", parent, merge, "--", *paths)


def candidate(pr, issues):
    """One PR -> candidate record, or a rejection reason. Pure function of clone+inventory."""
    number = pr["number"]
    keys = set(re.findall(r"KAFKA-\d+", pr["title"]))
    if not (pr["merged"] and pr["baseRefName"] == "trunk" and pr.get("mergeCommit")):
        return {"pr": number, "rejected": "not merged to trunk"}
    if not WINDOW[0] <= pr["createdAt"] <= WINDOW[1]:
        return {"pr": number, "rejected": "outside window"}
    if len(keys) != 1 or next(iter(keys)) not in issues:
        return {"pr": number, "rejected": "not exactly one known issue key"}
    key = next(iter(keys))
    merge = pr["mergeCommit"]["oid"]
    try:
        parents = git("rev-list", "--parents", "-n", "1", merge).split()[1:]
    except subprocess.CalledProcessError:
        return {"pr": number, "rejected": "merge commit absent from clone"}
    if len(parents) != 1:
        return {"pr": number, "rejected": "merge commit is not a single-parent squash"}
    parent = parents[0]
    status = [
        line.split("\t")
        for line in git("diff", "--no-renames", "--name-status", parent, merge).splitlines()
    ]
    paths = [p for _, p in status]
    low, high = LIMITS["files"]
    if not (
        low <= len(paths) <= high
        and all(p.endswith((".java", ".scala")) for p in paths)
        and all(is_test(p) or "/src/main/" in p for p in paths)
        and any(is_test(p) for p in paths)
        and any(not is_test(p) for p in paths)
    ):
        return {"pr": number, "rejected": "changed files outside the main+test source shape"}
    numstat = git("diff", "--no-renames", "--numstat", parent, merge).splitlines()
    if sum(int(a) + int(d) for a, d, _ in (n.split("\t") for n in numstat)) > LIMITS["lines"]:
        return {"pr": number, "rejected": "too many changed lines"}
    tests = []
    for state, path in status:
        if is_test(path) and state != "D" and TEST_CLASS.search(path):
            module, relative = path.split("/src/test/", 1)
            # Scala test trees (core/src/test/scala/unit/...) do not mirror packages.
            declared = PACKAGE.search(git("show", f"{merge}:{path}"))
            name = Path(path).stem
            tests.append(
                {
                    "path": path,
                    "module": ":" + module.replace("/", ":"),
                    "class": f"{declared.group(1)}.{name}" if declared else name,
                    "language": relative.split("/", 1)[0],
                }
            )
    if not tests:
        return {"pr": number, "rejected": "no runnable test class among changed test files"}
    issue = issues[key]
    # Context stops before the PR exists; the workspace is the merge parent.
    cutoff = timestamp(pr["createdAt"]) - timedelta(seconds=1)
    if timestamp(issue["fields"]["created"]) > cutoff:
        return {"pr": number, "rejected": "issue created after the PR"}
    try:
        body = body_at(issue, cutoff)
    except ValueError as exc:
        return {"pr": number, "rejected": str(exc)}
    if len(body["description"]) < LIMITS["issue_characters"]:
        return {"pr": number, "rejected": "issue description too short"}
    if f"/pull/{number}" in body["description"]:
        return {"pr": number, "rejected": "issue text links the solution"}
    test_patch = diff(parent, merge, [p for p in paths if is_test(p)])
    reference_patch = diff(parent, merge, [p for p in paths if not is_test(p)])
    return {
        "id": f"C-KAFKA-PR-{number}",
        "pr": number,
        "issue": key,
        "title": body["summary"],
        "description": body["description"],
        "pr_created_at": pr["createdAt"],
        "pr_merged_at": pr["mergedAt"],
        "context_cutoff": iso(cutoff),
        "base_sha": parent,
        "reference_commit": merge,
        "changed_files": paths,
        "tests": tests,
        "test_patch": test_patch,
        "reference_patch": reference_patch,
        "test_patch_sha256": sha(test_patch),
        "reference_patch_sha256": sha(reference_patch),
    }


def select():
    issues, prs, _ = inventory()
    rows = [candidate(pr, issues) for pr in prs]
    kept = stratified(
        [r for r in rows if "id" in r], SEED, lambda r: r["pr"], lambda r: r["pr_created_at"][:7]
    )
    BENCH.mkdir(parents=True, exist_ok=True)
    (BENCH / "coding-candidates.jsonl").write_text("".join(canonical(r) + "\n" for r in kept))
    report = {
        "window": WINDOW,
        "limits": LIMITS,
        "kept": len(kept),
        "by_year": Counter(r["pr_created_at"][:4] for r in kept),
        "rejections": Counter(r["rejected"] for r in rows if "rejected" in r),
        "repo_head": git("rev-parse", "HEAD").strip(),
    }
    save(BENCH / "coding-selection.json", report)
    print(canonical(report))


def wrapper_version(sha_):
    text = git("show", f"{sha_}:gradle/wrapper/gradle-wrapper.properties")
    return tuple(int(x) for x in re.search(r"gradle-(\d+)\.(\d+)", text).groups())


def image_for(sha_):
    # Gradle < 8.5 cannot run on JDK 21; Kafka trunk builds on 17 across the window.
    return "eclipse-temurin:17-jdk"


def outcomes(workspace, tests):
    """JUnit XML -> {class#method: passed|failed|skipped} for the targeted classes."""
    wanted = {t["class"] for t in tests}
    result = {}
    for path in workspace.glob("**/build/test-results/**/*.xml"):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        for case in root.iter("testcase"):
            if case.get("classname") not in wanted:
                continue
            state = "passed"
            if case.find("failure") is not None or case.find("error") is not None:
                state = "failed"
            elif case.find("skipped") is not None:
                state = "skipped"
            result[f"{case.get('classname')}#{case.get('name')}"] = state
    return result


def gradle(
    workspace, tests, image, log, *, offline=False, timeout=3000, cpus=4, memory="8g",
    home=None, ro_cache=None, goals=None,
):  # fmt: skip
    """Run the targeted tests in a throwaway container; returns (returncode|None, seconds).

    Admission uses one writable user home per worker slot. A grader passes its own throwaway
    `home` plus an admission cache as read-only dependency cache (`ro_cache`), so grading never
    shares a WRITABLE Gradle home with a concurrently running admission sweep or another grader
    (Gradle's cross-process locks do not span containers: "Could not create service …")."""
    for stale in workspace.glob("**/build/test-results"):
        shutil.rmtree(stale, ignore_errors=True)
    # Gradle's cross-process locks do not span containers: one user home per slot.
    slot = None if home else CACHES.get()
    cache = Path(home) if home else WORK / f"gradle-cache-{slot}"
    cache.mkdir(parents=True, exist_ok=True)
    shared = (
        ["-e", "GRADLE_RO_DEP_CACHE=/gradle-ro/caches", "-v", f"{ro_cache}:/gradle-ro:ro"]
        if ro_cache
        else []
    )
    args = list(goals or [])
    for module in [] if goals else sorted({t["module"] for t in tests}):
        # cleanTest: an up-to-date test task is skipped and writes no results.
        args += [f"{module}:cleanTest", f"{module}:test"]
        for t in tests:
            if t["module"] == module:
                args += ["--tests", t["class"]]
    name = f"harnext-admit-{workspace.parent.name}-{os.getpid()}-{time.time_ns() % 10**6}"
    argv = [
        "docker", "run", "--rm", "--name", name,
        "--memory", memory, "--cpus", str(cpus), "--pids-limit", "2048", "--cap-drop", "ALL",
        "--user", f"{os.getuid()}:{os.getgid()}",
        *(["--network", "none"] if offline else []),
        "-e", "GRADLE_USER_HOME=/gradle", "-e", "HOME=/tmp",
        "-v", f"{cache}:/gradle", *shared, "-v", f"{workspace}:/work", "-v", f"{INIT}:/init.gradle:ro",
        "-w", "/work", image,
        "./gradlew", *args, "--no-daemon", "--continue", "--console=plain", "--max-workers=2",
        "-PmaxParallelForks=2", "-PignoreFailures=false", *(["--offline"] if offline else []),
    ]  # fmt: skip
    started = time.monotonic()
    with log.open("w") as out:
        try:
            code = subprocess.run(argv, stdout=out, stderr=subprocess.STDOUT, timeout=timeout)
            return code.returncode, time.monotonic() - started
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True)
            return None, time.monotonic() - started
        finally:
            if slot is not None:
                CACHES.put(slot)


def classify(log_text, tests, touched=()):
    """Separate 'new tests do not compile on base' (a valid failure) from a broken build.

    A test-source compile failure counts as the expected base failure in every module
    whose test tree the PR touched (helpers such as MockAdminClient live in other modules
    than the test classes that use them)."""
    failed = set(FAILED_TASK.findall(log_text))
    test_compile = {m for m, _ in COMPILE_TEST.findall(log_text)}
    modules = {t["module"] for t in tests} | {
        ":" + p.split("/src/test/", 1)[0].replace("/", ":") for p in touched if is_test(p)
    }
    other = {
        f
        for f in failed
        if not any(f == f"{m}:test" for m in modules)
        and not any(f.startswith(m + ":compileTest") for m in test_compile)
    }
    return test_compile & modules, other


def admit(row, *, keep=False):
    """Base must fail, golden patch must pass (twice). Saved result is final unless retryable."""
    out = WORK / "results" / f"{row['pr']}.json"
    if out.exists() and not read(out).get("retryable"):
        return read(out)
    root = WORK / "trees" / str(row["pr"])
    shutil.rmtree(root, ignore_errors=True)
    workspace = root / "src"
    workspace.mkdir(parents=True)
    logs = WORK / "logs" / str(row["pr"])
    logs.mkdir(parents=True, exist_ok=True)
    result = {"id": row["id"], "pr": row["pr"], "base_sha": row["base_sha"], "runs": []}

    def finish(status, retryable=False, **extra):
        result.update(status=status, retryable=retryable, **extra)
        save(out, result)
        if not keep:
            shutil.rmtree(root, ignore_errors=True)
        print(canonical({"pr": row["pr"], "status": status}), flush=True)
        return result

    def apply(patch):
        # The tree lives inside this project's repository: without a ceiling git treats
        # it as a subdirectory, skips every path and still exits 0.
        env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(root)}
        done = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"], input=patch, text=True, cwd=workspace,
            capture_output=True, env=env,
        )  # fmt: skip
        reverse = subprocess.run(
            ["git", "apply", "--check", "--reverse", "--whitespace=nowarn", "-"], input=patch,
            text=True, cwd=workspace, capture_output=True, env=env,
        )  # fmt: skip
        if done.returncode == 0 and reverse.returncode:
            raise RuntimeError("patch reported applied but the tree is unchanged")
        return done

    try:
        archive = subprocess.Popen(
            ["git", "--git-dir", str(REPO), "archive", row["base_sha"]], stdout=subprocess.PIPE
        )
        subprocess.run(["tar", "-x", "-C", str(workspace)], stdin=archive.stdout, check=True)
        if archive.wait():
            return finish("environment_error", True, detail="git archive failed")
        if apply(row["test_patch"]).returncode:
            return finish("rejected_test_patch_conflict")
        image = image_for(row["base_sha"])
        result.update(image=image, gradle=".".join(map(str, wrapper_version(row["base_sha"]))))

        def run(label, offline=False):
            log = logs / f"{label}.log"
            code, seconds = gradle(workspace, row["tests"], image, log, offline=offline)
            text = log.read_text(errors="replace")
            compile_failed, broken = classify(text, row["tests"], row["changed_files"])
            cases = outcomes(workspace, row["tests"])
            result["runs"].append(
                {
                    "label": label,
                    "returncode": code,
                    "seconds": round(seconds, 1),
                    "test_compile_failed": sorted(compile_failed),
                    "other_failed_tasks": sorted(broken),
                    "counts": Counter(cases.values()),
                }
            )
            return code, compile_failed, broken, cases

        code, base_compile, broken, base = run("base")
        if code is None:
            return finish("environment_error", True, detail="base run timed out")
        if broken or (code and not base_compile and "failed" not in base.values()):
            return finish("environment_error", True, detail="base build broken outside the tests")
        if not base_compile and "failed" not in base.values():
            return finish("rejected_passes_on_base")
        if apply(row["reference_patch"]).returncode:
            return finish("rejected_reference_patch_conflict")
        golden = []
        for label in ("reference-1", "reference-2"):
            code, compile_failed, broken, cases = run(label)
            if code is None or compile_failed or broken:
                return finish("rejected_reference_does_not_build")
            golden.append(cases)
        stable = {k for k, v in golden[0].items() if v == "passed" and golden[1].get(k) == "passed"}
        unstable = {k for g in golden for k, v in g.items() if v == "failed"}
        if unstable:
            return finish("rejected_reference_fails_or_flaky", unstable=sorted(unstable)[:20])
        fail_to_pass = sorted(k for k in stable if base.get(k) != "passed")
        pass_to_pass = sorted(k for k in stable if base.get(k) == "passed")
        if not fail_to_pass:
            return finish("rejected_no_fail_to_pass")
        return finish(
            "admitted",
            base_failure="test_compile" if base_compile else "test_failure",
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
        )
    except Exception as exc:  # keep the sweep alive; the candidate is retried on resume
        return finish("environment_error", True, detail=f"{type(exc).__name__}: {exc}"[:500])


def sweep(workers, limit, target, prs=None):
    rows = [read_line for read_line in map(_loads, (BENCH / "coding-candidates.jsonl").open())]
    if prs:  # an explicit list, admitted in the given order (e.g. strongest dependency chains first)
        by_pr = {str(r["pr"]): r for r in rows}
        rows = [by_pr[p] for p in prs if p in by_pr]
    rows = rows[:limit] if limit else rows
    done = Counter()

    def one(row):
        if target and done["admitted"] >= target:
            return
        done[admit(row)["status"]] += 1

    for slot in range(1, workers):
        CACHES.put(slot)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, rows))
    summarize()


def _loads(line):
    import json

    return json.loads(line)


def summarize():
    results = [read(p) for p in sorted((WORK / "results").glob("*.json"))]
    report = {
        "checked": len(results),
        "status": Counter(r["status"] for r in results),
        "base_failure": Counter(
            r.get("base_failure") for r in results if r["status"] == "admitted"
        ),
        "seconds": round(sum(run["seconds"] for r in results for run in r["runs"])),
    }
    save(BENCH / "coding-admission.json", report)
    print(canonical(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("select")
    sub.add_parser("summary")
    run = sub.add_parser("admit")
    run.add_argument("--workers", type=int, default=3)
    run.add_argument("--limit", type=int)
    run.add_argument("--target", type=int, help="stop launching once this many are admitted")
    run.add_argument("--pr", type=int, help="admit one candidate and keep its tree")
    run.add_argument("--prs", help="comma-separated PR numbers to admit, in this order")
    args = parser.parse_args()
    if args.command == "select":
        select()
    elif args.command == "summary":
        summarize()
    elif args.pr:
        row = next(
            r for r in map(_loads, (BENCH / "coding-candidates.jsonl").open()) if r["pr"] == args.pr
        )
        print(canonical(admit(row, keep=True)))
    else:
        sweep(args.workers, args.limit, args.target, args.prs.split(",") if args.prs else None)


if __name__ == "__main__":
    main()
