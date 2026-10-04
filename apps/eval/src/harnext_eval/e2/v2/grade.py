"""Grading for E2 v2. Historical: v1's typed exact match. Coding: the agent's main-source diff
plus the hidden test patch on a fresh base tree, targeted tests run offline in Docker.

A grading failure that is the machine's (docker, offline dependency resolution, a build that
breaks without any agent change, a timeout before tests start) raises PauseRun; a build or
test failure that follows from the agent's patch is the model's result (resolved = false).
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

from harnext_eval.e2 import kafka_coding as kc
from harnext_eval.e2.benchmark_audit import typed_equal
from harnext_eval.e2.v2.errors import PauseRun

FILE_HEADER = re.compile(r"^diff --git a/(.+?) b/(.+)$", re.MULTILINE)
INFRA_LOG = re.compile(
    r"Could not resolve |No cached version|available for offline mode|Could not (GET|HEAD|download)|"
    r"No space left on device|Java heap space|OutOfMemoryError|daemon disappeared|"
    r"Could not create service|Unable to (find|locate) a Java|Timeout waiting to lock|"
    r"docker: Error|Cannot connect to the Docker daemon",
    re.IGNORECASE,
)
TEST_STARTED = re.compile(r"> Task :\S+:test\b")


NORMALISED = {  # families.py answers: strings casefolded/whitespace-normalised, arrays sorted
    "issue_prs", "issue_changed_files", "pr_approvers", "review_hotspot", "reopen_count",
    "issue_links", "fix_version_at_snapshot", "resolution_vs_merge_order",
    "open_blockers_unassigned", "author_activity", "top_reviewer_for_author", "kip_thread",
    "implicit_link",  # mail thread -> JIRA key (2026-09-27 implicit-link benchmark)
    "design_decision",  # final value of a KIP design decision (2026-09-27, human-reviewed)
}  # fmt: skip


def normalised(value):
    """Case/whitespace-insensitive strings, order-insensitive arrays (duplicates still count),
    dict keys and every type kept: int stays int, bool stays bool, null stays null."""
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    if isinstance(value, list):
        return sorted(map(normalised, value), key=lambda v: (type(v).__name__, json.dumps(v, sort_keys=True)))  # fmt: skip
    if isinstance(value, dict):
        return {k: normalised(v) for k, v in value.items()}
    return value


def grade_historical(private, answer):
    """Typed equality: no null/"null", bool/int or stringified-JSON credit; v1 histories
    ordered; the families of families.py compare normalised (see `normalised`)."""
    ok = isinstance(answer, dict) and "value" in answer
    expected = private["expected"]["value"]  # tasks.private.jsonl: {"expected": {"value": ...}}
    loose = private.get("family") in NORMALISED
    actual = normalised(answer["value"]) if ok and loose else answer.get("value") if ok else None
    return {
        "correct": bool(ok and typed_equal(actual, normalised(expected) if loose else expected)),
        "expected": expected,
        "actual": answer.get("value") if ok else None,
        "evidence": answer.get("evidence") if ok else None,
    }


def extract(base_sha, destination, repo=None):
    """`git archive` of the base commit: no .git, no history, hidden tests absent."""
    destination.mkdir(parents=True, exist_ok=True)
    archive = subprocess.Popen(
        ["git", "--git-dir", str(repo or kc.REPO), "archive", base_sha], stdout=subprocess.PIPE
    )
    unpack = subprocess.run(["tar", "-x", "-C", str(destination)], stdin=archive.stdout)
    if archive.wait() or unpack.returncode:
        raise PauseRun(f"git archive {base_sha} failed")
    provision_wrapper(destination)
    return destination


def provision_wrapper(tree, cache=None):
    """Kafka does not version gradle-wrapper.jar: its gradlew curls it from GitHub, which no
    trial or offline grading container can reach. Take it from the cached Gradle distribution
    (it ships inside lib/plugins/gradle-wrapper*.jar). The path is git-ignored by Kafka, so it
    never shows up in an agent's diff."""
    properties = tree / "gradle/wrapper/gradle-wrapper.properties"
    jar = tree / "gradle/wrapper/gradle-wrapper.jar"
    if not properties.exists() or jar.exists():
        return
    found = re.search(r"(gradle-[\d.]+)-(?:all|bin)", properties.read_text())
    name = found.group(1) if found else "gradle-?"
    caches = [cache] if cache else sorted(kc.WORK.glob("gradle-cache*"))
    jars = f"wrapper/dists/{name}-*/*/{name}/lib/plugins/gradle-wrapper*.jar"
    for holder in (j for c in caches for j in c.glob(jars)):
        with zipfile.ZipFile(holder) as archive:
            if "gradle-wrapper.jar" in archive.namelist():
                jar.write_bytes(archive.read("gradle-wrapper.jar"))
                return
    raise PauseRun(f"{name} is not in the offline Gradle cache ({kc.WORK}); admit the task first")


def workspace_patch(base_sha, workspace, tmp, repo=None):
    """Final workspace vs base as a git patch. A throwaway index and object directory layered
    over the bare clone: the clone is never written, and the tree's own .gitignore keeps
    build output out of the diff."""
    repo = Path(repo or kc.REPO)
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "objects").mkdir(exist_ok=True)
    env = {
        **os.environ,
        "GIT_DIR": str(repo),
        "GIT_WORK_TREE": str(workspace),
        "GIT_INDEX_FILE": str(tmp / "index"),
        "GIT_OBJECT_DIRECTORY": str(tmp / "objects"),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(repo / "objects"),
    }

    def git(*args):
        done = subprocess.run(
            ["git", "-c", "core.bare=false", "-c", "core.fileMode=true", *args],
            env=env, cwd=workspace, capture_output=True, text=True, errors="replace",
        )  # fmt: skip
        if done.returncode:
            raise PauseRun(f"git {args[0]} failed while diffing: {done.stderr[:500]}")
        return done.stdout

    git("read-tree", base_sha)
    git("add", "-A", ".")
    patch = git("diff", "--cached", "--no-renames", "--full-index", "--binary", base_sha)
    shutil.rmtree(tmp, ignore_errors=True)
    return patch


def split_patch(patch):
    """[(path, section)] per file of a git patch."""
    starts = [m.start() for m in FILE_HEADER.finditer(patch)] + [len(patch)]
    sections = [patch[a:b] for a, b in zip(starts, starts[1:], strict=False)]
    return [(m.group(2), s) for s in sections if (m := FILE_HEADER.match(s))]


HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", re.MULTILINE)


def _f1(hit, mine, gold):
    precision, recall = (hit / mine if mine else 0.0), (hit / gold if gold else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def _changes(patch):
    """{path: (added lines, removed lines, touched base-line numbers)}; lines are compared
    with whitespace collapsed, and blank / brace-only lines carry no signal."""
    result = {}
    for path, section in split_patch(patch):
        added, removed, touched = Counter(), Counter(), set()
        for line in section.splitlines():
            text = " ".join(line[1:].split())
            if line.startswith(("+++", "---")) or text in {"", "{", "}", "});", ");"}:
                continue
            if line.startswith("+"):
                added[text] += 1
            elif line.startswith("-"):
                removed[text] += 1
        for start, length in HUNK.findall(section):
            first, count = int(start), int(length or 1)
            touched.update(range(first, first + max(count, 1)))
        result[path] = (added, removed, touched)
    return result


def similarity(agent_patch, reference_patch):
    """How close the agent's change is to the golden one. Descriptive only: a different correct
    fix scores low here, so it never decides `resolved`. Files = which files were changed;
    lines = identical added / removed lines; location = how much of the golden change's place in
    the base files the agent also edited (hunk ranges incl. context)."""
    mine, gold = _changes(agent_patch), _changes(reference_patch)
    shared = set(mine) & set(gold)

    def lines(index):
        hit = sum(sum((mine[p][index] & gold[p][index]).values()) for p in shared)
        return _f1(
            hit,
            sum(sum(v[index].values()) for v in mine.values()),
            sum(sum(v[index].values()) for v in gold.values()),
        )

    touched_gold = sum(len(v[2]) for v in gold.values())
    return {
        "files": _f1(len(shared), len(mine), len(gold))
        | {"agent": len(mine), "golden": len(gold), "shared": len(shared)},
        "added_lines": lines(0),
        "removed_lines": lines(1),
        "location_recall": round(
            sum(len(mine[p][2] & gold[p][2]) for p in shared) / touched_gold, 3
        )
        if touched_gold
        else 0.0,
        "size": {
            "agent_added": sum(sum(v[0].values()) for v in mine.values()),
            "agent_removed": sum(sum(v[1].values()) for v in mine.values()),
            "golden_added": sum(sum(v[0].values()) for v in gold.values()),
            "golden_removed": sum(sum(v[1].values()) for v in gold.values()),
        },
    }


def without_paths(patch, hidden):
    """(patch minus sections touching the hidden test files, dropped paths)."""
    kept = [(p, s) for p, s in split_patch(patch) if p not in hidden]
    dropped = sorted(p for p, _ in split_patch(patch) if p in hidden)
    return "".join(s for _, s in kept), dropped


def apply(patch, workspace):
    """`git apply` inside another repository's tree (apps/eval/out is inside this one) treats
    cwd as a subdirectory, skips every path and exits 0. The ceiling stops repository discovery
    and the reverse check proves the patch really landed."""
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(Path(workspace).resolve().parent)}
    argv = ["git", "apply", "--whitespace=nowarn"]
    done = subprocess.run([*argv, "-"], input=patch, text=True, cwd=workspace, env=env, capture_output=True)  # fmt: skip
    if done.returncode == 0 and patch.strip():
        check = subprocess.run([*argv, "--check", "--reverse", "-"], input=patch, text=True, cwd=workspace, env=env, capture_output=True)  # fmt: skip
        if check.returncode:
            return subprocess.CompletedProcess(
                done.args, 1, "", "patch was not applied: " + check.stderr
            )
    return done


def ensure_offline_ready(task, tree, scratch):
    """Once per task, before its first trial: prove that the task's modules resolve and compile
    OFFLINE from the shared read-only Gradle cache; if they do not (the task was admitted on a
    cache that is gone, or a plugin was never fetched), warm the cache ONLINE once. Serialised
    by a lock: the writable cache must never be shared by two containers."""
    import fcntl

    from harnext_eval.e2.v2.agents import container

    marker = kc.WORK / "warmed" / f"{task['id']}.json"
    if marker.exists():
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    modules = sorted({t["module"] for t in task["tests"]})
    goals = [f"{m}:testClasses" for m in modules]
    wrapper = tree / "gradle/wrapper/gradle-wrapper.properties"
    found = re.search(r"(gradle-[\d.]+)-", wrapper.read_text()) if wrapper.exists() else None
    dist = found.group(1) if found else None
    image = kc.image_for(task["base_sha"])
    with (marker.parent / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists():
            return
        home = scratch / "warm" / "gradle-home"
        container.seed_gradle_home(home, dist)
        log = scratch / "warm" / "offline.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        kc.gradle(tree, task["tests"], image, log, offline=True, timeout=1800, home=home,
                  ro_cache=container.gradle_cache(dist), goals=goals)  # fmt: skip
        text = log.read_text(errors="replace") if log.exists() else ""
        warmed = bool(INFRA_LOG.search(text))
        if warmed:  # online, into the writable admission cache (slot 0; admission must be idle)
            log = scratch / "warm" / "online.log"
            kc.gradle(tree, task["tests"], image, log, offline=False, timeout=2400, goals=goals)
            again = log.read_text(errors="replace") if log.exists() else ""
            if INFRA_LOG.search(again):
                found = INFRA_LOG.search(again)
                raise PauseRun(f"cannot warm the Gradle cache for {task['id']}: {found.group(0)}")
        marker.write_text(json.dumps({"task": task["id"], "warmed_online": warmed}))
    shutil.rmtree(scratch / "warm", ignore_errors=True)
    for built in tree.glob("**/build"):  # leave the workspace as staged: no compiled output
        shutil.rmtree(built, ignore_errors=True)
    shutil.rmtree(tree / ".gradle", ignore_errors=True)


def decide(*, code, log, cases, private, tests, patch_empty):
    """The decision table. Returns the grade dict or raises PauseRun."""
    compile_failed, other = kc.classify(log, tests)
    if code in {125, 126, 127} or INFRA_LOG.search(log):
        found = INFRA_LOG.search(log)
        raise PauseRun(f"grading infrastructure: {found.group(0) if found else f'docker {code}'}")
    if code is None and not (TEST_STARTED.search(log) or cases):
        raise PauseRun("grading timed out before any targeted test started")
    if patch_empty and other:
        raise PauseRun(f"untouched base does not build: {sorted(other)}")
    if code == 0 and not cases:
        raise PauseRun("Gradle succeeded but produced no targeted test results")
    failed = [k for k in private["fail_to_pass"] if cases.get(k) != "passed"]
    broken = [k for k in private["pass_to_pass"] if cases.get(k) != "passed"]
    build_failed = bool(compile_failed or other)
    return {
        "resolved": not failed and not broken and not build_failed and code == 0,
        "fail_to_pass_failed": failed,
        "pass_to_pass_failed": broken,
        "test_compile_failed": sorted(compile_failed),
        "other_failed_tasks": sorted(other),
        "grading_timed_out": code is None,
        "gradle_returncode": code,
        "counts": dict(Counter(cases.values())),
        "outcomes": dict(sorted(cases.items())),
    }


def grade_coding(task, private, workspace, out, *, scratch, timeout=3000, repo=None):
    """Diff -> fresh base + main-source changes + hidden tests -> offline targeted tests."""
    patch = workspace_patch(task["base_sha"], workspace, scratch / "diff", repo)
    (out / "patch.diff").write_text(patch)
    hidden = {p for p, _ in split_patch(private["test_patch"])}
    main, dropped = without_paths(patch, hidden)
    grade = {"patch_empty": not main.strip(), "patch_bytes": len(patch), "dropped_paths": dropped}
    if private.get("reference_patch"):  # descriptive; never part of `resolved`
        grade["similarity"] = similarity(main, private["reference_patch"])
    if grade["patch_empty"]:
        # Admission executed exactly this state (base + hidden tests) and it fails.
        return grade | {"resolved": False, "graded_by": "empty_patch", "seconds": 0.0}
    tree = extract(task["base_sha"], scratch / "grade" / "src", repo)
    if (done := apply(main, tree)).returncode:
        raise PauseRun(f"agent diff does not re-apply to its own base: {done.stderr[:300]}")
    if (done := apply(private["test_patch"], tree)).returncode:
        raise PauseRun(f"hidden test patch does not apply: {done.stderr[:300]}")
    log = scratch / "grade" / "gradle.log"
    # A throwaway writable Gradle home + the admission cache read-only: see kc.gradle.
    from harnext_eval.e2.v2.agents import container

    wrapper = tree / "gradle/wrapper/gradle-wrapper.properties"
    found = re.search(r"(gradle-[\d.]+)-", wrapper.read_text()) if wrapper.exists() else None
    dist = found.group(1) if found else None
    home = scratch / "grade" / "gradle-home"
    container.seed_gradle_home(home, dist)
    code, seconds = kc.gradle(
        tree, task["tests"], kc.image_for(task["base_sha"]), log, offline=True, timeout=timeout,
        home=home, ro_cache=container.gradle_cache(dist),
    )  # fmt: skip
    text = log.read_text(errors="replace") if log.exists() else ""
    (out / "gradle.log.gz").write_bytes(gzip.compress(text.encode(), mtime=0))
    cases = kc.outcomes(tree, task["tests"])
    decision = decide(
        code=code, log=text, cases=cases, private=private, tests=task["tests"], patch_empty=False
    )
    return grade | decision | {"graded_by": "gradle", "seconds": round(seconds, 1)}
