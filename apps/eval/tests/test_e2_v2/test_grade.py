import gzip

import pytest
from harnext_eval.e2 import kafka_coding as kc
from harnext_eval.e2.v2 import grade
from harnext_eval.e2.v2.errors import PauseRun

from .test_runner import CASE, MAIN, TEST, make_repo

TESTS = [{"path": TEST, "module": ":core", "class": "ATest", "language": "java"}]
PRIVATE = {"fail_to_pass": ["T#new"], "pass_to_pass": ["T#old"]}


@pytest.mark.parametrize(
    ("expected", "value", "correct"),
    [
        (None, None, True),
        (None, "null", False),
        ("null", None, False),
        ("Open", "Open", True),
        ("Open", "open", False),
        (True, 1, False),
        (True, "true", False),
        (False, False, True),
        ([{"at": "1", "from": None, "to": "A"}, {"at": "2", "from": "A", "to": "B"}],
         [{"at": "1", "from": None, "to": "A"}, {"at": "2", "from": "A", "to": "B"}], True),
        ([{"at": "1", "from": None, "to": "A"}, {"at": "2", "from": "A", "to": "B"}],
         [{"at": "2", "from": "A", "to": "B"}, {"at": "1", "from": None, "to": "A"}], False),
        ([{"at": "1", "from": None, "to": "A"}], [{"at": "1", "from": "null", "to": "A"}], False),
        ([], [], True),
        ([], None, False),
        (["a"], '["a"]', False),
    ],
)  # fmt: skip
def test_historical_typed_equality(expected, value, correct):
    graded = grade.grade_historical(
        {"expected": {"value": expected}}, {"value": value, "evidence": ["e1"]}
    )
    assert graded["correct"] is correct and graded["evidence"] == ["e1"]


@pytest.mark.parametrize("answer", [None, {}, {"evidence": []}, "Open", ["Open"]])
def test_historical_without_a_value_is_wrong_not_an_error(answer):
    assert grade.grade_historical({"expected": {"value": None}}, answer)["correct"] is False


@pytest.mark.parametrize(
    ("family", "expected", "value", "correct"),
    [
        ("issue_prs", [{"pr": 7, "state": "merged"}, {"pr": 8, "state": "open"}],
         [{"pr": 8, "state": "OPEN"}, {"pr": 7, "state": "merged"}], True),  # order, case
        ("issue_prs", [{"pr": 7, "state": "merged"}], [{"pr": "7", "state": "merged"}], False),  # int stays int
        ("issue_prs", [{"pr": 7, "state": "merged"}], [{"pr": 7, "state": "merged"}, {"pr": 7, "state": "merged"}], False),
        ("issue_changed_files", ["b/B.java", "a/A.java"], ["a/A.java", " b/B.java "], True),
        ("issue_changed_files", ["a/A.java"], ["a/A.java", "b/B.java"], False),
        ("pr_approvers", ["rev1", "rev2"], ["Rev2", "rev1"], True),
        ("review_hotspot", "a/A.java", "A/a.JAVA", True),
        ("review_hotspot", "a/A.java", ["a/A.java"], False),
        ("reopen_count", 0, 0, True),
        ("reopen_count", 1, "1", False),
        ("reopen_count", 1, True, False),
        ("reopen_count", 0, False, False),
        ("issue_links", ["KAFKA-2", "KAFKA-3"], ["kafka-3", "KAFKA-2"], True),
        ("fix_version_at_snapshot", ["3.6.0"], ["3.6.0", "3.6.0"], False),
        ("resolution_vs_merge_order", {"first": "merge", "days": 5}, {"days": 5, "first": "Merge"}, True),
        ("resolution_vs_merge_order", {"first": "merge", "days": 5}, {"first": "merge", "days": 5.0}, False),
        ("resolution_vs_merge_order", {"first": "merge", "days": 5}, {"first": "merge"}, False),
        ("open_blockers_unassigned", ["KAFKA-3"], [], False),
        ("author_activity", 3, 3, True),
        ("top_reviewer_for_author", "ijuma", " IJUMA ", True),
        ("kip_thread", {"starter": "Gantigmaa Selenge", "participants": 7}, {"starter": "gantigmaa  selenge", "participants": 7}, True),
        ("kip_thread", {"starter": "A", "participants": 7}, {"starter": "A", "participants": "7"}, False),
    ],
)  # fmt: skip
def test_new_families_grade_normalised(family, expected, value, correct):
    private = {"family": family, "expected": {"value": expected}}
    assert grade.grade_historical(private, {"value": value, "evidence": []})["correct"] is correct
    # v1 families keep the strict rule: no case or order credit
    strict = {"family": "status_at_snapshot", "expected": {"value": "Open"}}
    assert grade.grade_historical(strict, {"value": "open", "evidence": []})["correct"] is False


OK = "> Task :core:test\nBUILD SUCCESSFUL\n"
MAIN_BROKEN = "> Task :core:compileJava FAILED\nerror: ';' expected\nBUILD FAILED\n"
TEST_BROKEN = "> Task :core:compileTestJava FAILED\nerror: cannot find symbol\nBUILD FAILED\n"
TEST_FAILED = "> Task :core:test FAILED\nBUILD FAILED\n"


@pytest.mark.parametrize(
    ("code", "log", "cases", "resolved"),
    [
        (0, OK, {"T#new": "passed", "T#old": "passed"}, True),
        (1, TEST_FAILED, {"T#new": "failed", "T#old": "passed"}, False),
        (1, TEST_FAILED, {"T#new": "passed", "T#old": "failed"}, False),  # regression
        (0, OK, {"T#new": "skipped", "T#old": "passed"}, False),
        (0, OK, {"T#old": "passed"}, False),  # fail_to_pass never ran
        (1, MAIN_BROKEN, {}, False),  # the agent's patch does not compile: model failure
        (1, TEST_BROKEN, {}, False),  # hidden tests need an API the patch did not add
        (None, "> Task :core:test\n", {"T#old": "passed"}, False),  # the patched code hangs
    ],
)
def test_coding_decision_table_model_results(code, log, cases, resolved):
    graded = grade.decide(code=code, log=log, cases=cases, private=PRIVATE, tests=TESTS, patch_empty=False)  # fmt: skip
    assert graded["resolved"] is resolved
    assert graded["grading_timed_out"] is (code is None)


@pytest.mark.parametrize(
    ("code", "log", "cases", "patch_empty"),
    [
        (125, "docker: Error response from daemon: no such image\n", {}, False),
        (1, "> Could not resolve com.x:y:1.\n  > No cached version available for offline mode\n", {}, False),
        (1, "java.lang.OutOfMemoryError: Java heap space\n", {}, False),
        (1, "write error: No space left on device\n", {}, False),
        (None, "> Task :clients:compileJava\n", {}, False),  # timed out compiling, no test started
        (1, MAIN_BROKEN, {}, True),  # nothing was changed and main still does not compile
        (0, "BUILD SUCCESSFUL\n", {}, False),  # green build, no XML: the harness lost the results
    ],
)  # fmt: skip
def test_coding_decision_table_infrastructure_pauses(code, log, cases, patch_empty):
    with pytest.raises(PauseRun):
        grade.decide(code=code, log=log, cases=cases, private=PRIVATE, tests=TESTS, patch_empty=patch_empty)  # fmt: skip


@pytest.fixture
def repo(tmp_path, monkeypatch):
    path, base, agent_patch, test_patch = make_repo(tmp_path / "repo")
    monkeypatch.setattr(kc, "REPO", path)
    task = {"base_sha": base, "tests": TESTS}
    private = {"fail_to_pass": [CASE], "pass_to_pass": [], "test_patch": test_patch}
    return tmp_path, task, private, agent_patch


def test_workspace_has_no_git_and_no_hidden_tests_and_diff_ignores_build_output(repo):
    tmp, task, _, agent_patch = repo
    workspace = grade.extract(task["base_sha"], tmp / "ws")
    assert not (workspace / ".git").exists() and "works" not in (workspace / TEST).read_text()
    assert grade.workspace_patch(task["base_sha"], workspace, tmp / "d") == ""
    grade.apply(agent_patch, workspace)
    (workspace / "build").mkdir()
    (workspace / "build/A.class").write_bytes(b"\xca\xfe")
    (workspace / "core/src/main/java/B.java").write_text("class B {}\n")
    patch = grade.workspace_patch(task["base_sha"], workspace, tmp / "d")
    assert {p for p, _ in grade.split_patch(patch)} == {MAIN, "core/src/main/java/B.java"}
    assert not list(kc.REPO.glob("objects/??/*")) or "return 2" in patch  # clone untouched


def test_coding_grade_drops_edits_to_hidden_test_paths_and_applies_the_real_tests(
    repo, monkeypatch
):
    tmp, task, private, agent_patch = repo
    workspace = grade.extract(task["base_sha"], tmp / "ws")
    grade.apply(agent_patch, workspace)
    (workspace / TEST).write_text("class ATest { void works() { /* rigged */ } }\n")
    seen = {}

    def gradle(tree, tests, image, log, *, offline, timeout, home=None, ro_cache=None):
        seen.update(main=(tree / MAIN).read_text(), test=(tree / TEST).read_text(), offline=offline)
        log.write_text(OK)
        return 0, 12.3

    monkeypatch.setattr(kc, "gradle", gradle)
    monkeypatch.setattr(kc, "outcomes", lambda tree, tests: {CASE: "passed"})
    out = tmp / "out"
    out.mkdir()
    graded = grade.grade_coding(task, private, workspace, out, scratch=tmp / "scratch")
    assert graded["resolved"] and graded["dropped_paths"] == [TEST] and not graded["patch_empty"]
    assert "return 2" in seen["main"] and "rigged" not in seen["test"] and "works" in seen["test"]
    assert seen["offline"] is True and graded["seconds"] == 12.3
    assert gzip.decompress((out / "gradle.log.gz").read_bytes()).decode() == OK
    assert "rigged" in (out / "patch.diff").read_text()  # the full diff is kept for audit


def test_only_test_edits_count_as_an_empty_patch_and_skip_gradle(repo, monkeypatch):
    tmp, task, private, _ = repo
    workspace = grade.extract(task["base_sha"], tmp / "ws")
    (workspace / TEST).write_text("class ATest { void works() {} }\n")
    monkeypatch.setattr(kc, "gradle", lambda *a, **k: pytest.fail("gradle must not run"))
    out = tmp / "out"
    out.mkdir()
    graded = grade.grade_coding(task, private, workspace, out, scratch=tmp / "scratch")
    assert graded["patch_empty"] and not graded["resolved"] and graded["graded_by"] == "empty_patch"


def test_wrapper_jar_comes_from_the_cached_distribution_or_pauses(tmp_path):
    import zipfile

    tree, cache = tmp_path / "tree", tmp_path / "cache"
    (tree / "gradle/wrapper").mkdir(parents=True)
    (tree / "gradle/wrapper/gradle-wrapper.properties").write_text(
        "distributionUrl=https\\://services.gradle.org/distributions/gradle-8.10.2-all.zip\n"
    )
    with pytest.raises(PauseRun, match="gradle-8.10.2 is not in the offline Gradle cache"):
        grade.provision_wrapper(tree, cache)
    plugins = cache / "wrapper/dists/gradle-8.10.2-all/abc123/gradle-8.10.2/lib/plugins"
    plugins.mkdir(parents=True)
    with zipfile.ZipFile(plugins / "gradle-wrapper-main-8.10.2.jar", "w") as holder:
        holder.writestr("gradle-wrapper.jar", b"JAR")
    grade.provision_wrapper(tree, cache)
    assert (tree / "gradle/wrapper/gradle-wrapper.jar").read_bytes() == b"JAR"
    grade.provision_wrapper(tree, tmp_path / "gone")  # already present: nothing to do


def test_apply_works_in_a_directory_nested_inside_this_repository(repo):
    """Regression: under apps/eval/out (inside the project repo) plain `git apply` skips every
    path and exits 0."""
    import shutil
    import tempfile
    from pathlib import Path

    _, task, _, agent_patch = repo
    inside = Path(tempfile.mkdtemp(prefix=".e2v2-apply-", dir=Path(__file__).parent))
    try:
        tree = grade.extract(task["base_sha"], inside / "src")
        assert grade.apply(agent_patch, tree).returncode == 0
        assert "return 2" in (tree / MAIN).read_text()
        assert grade.apply(agent_patch, tree).returncode != 0  # already applied: reported
    finally:
        shutil.rmtree(inside)


def test_similarity_to_the_golden_patch_is_descriptive_and_bounded():
    golden = (
        "diff --git a/m/src/main/java/A.java b/m/src/main/java/A.java\n--- a/m/src/main/java/A.java\n"
        "+++ b/m/src/main/java/A.java\n@@ -10,3 +10,4 @@\n ctx\n-old()\n+fresh()\n+extra()\n ctx\n"
    )
    other = golden.replace("fresh()", "different()").replace("A.java", "B.java")
    same, far = grade.similarity(golden, golden), grade.similarity(other, golden)
    assert same["files"]["f1"] == same["added_lines"]["f1"] == same["location_recall"] == 1.0
    assert far["files"]["f1"] == far["location_recall"] == 0.0
    assert grade.similarity("", golden)["files"] == {
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "agent": 0, "golden": 1, "shared": 0,
    }  # fmt: skip
