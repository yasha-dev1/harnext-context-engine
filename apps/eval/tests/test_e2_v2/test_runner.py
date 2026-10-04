import json
import subprocess

import pytest
import yaml
from harnext_eval.e2 import kafka_coding as kc
from harnext_eval.e2.benchmark import sha
from harnext_eval.e2.v2 import report, runner
from harnext_eval.e2.v2.agents.fake import FakeAgent

MAIN, TEST = "core/src/main/java/A.java", "core/src/test/java/ATest.java"
CASE = "ATest#works"


def sh(*args, cwd, **kwargs):
    return subprocess.run(
        args, cwd=cwd, check=True, capture_output=True, text=True, **kwargs
    ).stdout


def make_repo(tmp):
    """Bare clone stand-in with one commit; returns (repo, base sha, agent patch, test patch)."""
    src = tmp / "src"
    for path, text in {MAIN: "class A { int f() { return 1; } }\n", TEST: "class ATest {}\n", ".gitignore": "build/\n"}.items():  # fmt: skip
        (src / path).parent.mkdir(parents=True, exist_ok=True)
        (src / path).write_text(text)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    sh(*git, "init", "-q", cwd=src)
    sh(*git, "add", "-A", cwd=src)
    sh(*git, "commit", "-qm", "base", cwd=src)
    base = sh("git", "rev-parse", "HEAD", cwd=src).strip()
    patches = []
    for path, text in ((MAIN, "class A { int f() { return 2; } }\n"), (TEST, "class ATest { void works() {} }\n")):  # fmt: skip
        (src / path).write_text(text)
        patches.append(sh("git", "diff", cwd=src))
        sh("git", "checkout", "-q", "--", path, cwd=src)
    sh("git", "clone", "-q", "--bare", str(src), str(tmp / "kafka.git"), cwd=tmp)
    return tmp / "kafka.git", base, *patches


@pytest.fixture
def bench(tmp_path, monkeypatch):
    repo, base, agent_patch, test_patch = make_repo(tmp_path / "repo")
    monkeypatch.setattr(kc, "REPO", repo)
    calls = []

    def gradle(workspace, tests, image, log, **kwargs):
        calls.append(workspace)
        log.write_text("BUILD SUCCESSFUL\n")
        return 0, 2.0

    monkeypatch.setattr(kc, "gradle", gradle)
    monkeypatch.setattr(kc, "outcomes", lambda workspace, tests: {CASE: "passed"})
    directory = tmp_path / "benchmarks/mini"
    directory.mkdir(parents=True)
    public, private = [], []
    for i in range(10):
        if i in (2, 7):
            public.append({"id": f"C-{i}", "kind": "coding", "family": "coding", "checkpoint": "2023-06", "prompt": f"Fix {i}", "answer_schema": None, "base_sha": base, "tests": [{"path": TEST, "module": ":core", "class": "ATest", "language": "java"}]})  # fmt: skip
            private.append({"id": f"C-{i}", "fail_to_pass": [CASE], "pass_to_pass": [], "test_patch": test_patch, "reference_patch": agent_patch, "evidence_ids": []})  # fmt: skip
        else:
            family = "assignee_at_snapshot" if i % 2 else "status_history"
            public.append({"id": f"H-{i}", "kind": "historical", "family": family, "checkpoint": "2023-06", "prompt": f"Q{i}?", "answer_schema": {"type": "object"}})  # fmt: skip
            private.append({"id": f"H-{i}", "expected": {"value": f"v{i}"}, "evidence_ids": []})
    for name, rows in (("tasks.public.jsonl", public), ("tasks.private.jsonl", private)):
        (directory / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest = {
        "files": {n: sha((directory / n).read_bytes()) for n in ("tasks.public.jsonl", "tasks.private.jsonl")},
        "run_order": [t["id"] for t in public],
    }  # fmt: skip
    (directory / "manifest.json").write_text(json.dumps(manifest))
    arms = [
        {"id": "N", "access": "none"},
        {"id": "F1", "access": "files", "store": "F1"},
        {"id": "F5-mcp", "access": "mcp", "mcp_kind": "research", "store": "F5", "subset": "control"},
    ]
    (directory / "arms.yaml").write_text(yaml.safe_dump({"mcp_server": {"name": "history"}, "arms": arms}))  # fmt: skip
    for store in ("F1", "F5"):
        export = tmp_path / "stores" / store / "export/2023-06"
        export.mkdir(parents=True)
        (export / "INDEX.md").write_text(f"# {store}\n")

    def config(run_id, **extra):
        return runner.Config(
            protocol="harnext-e2-v2", run_id=run_id, benchmark=directory, arms_file=directory / "arms.yaml",
            agent="fake", model="m", reader="fake", stores=tmp_path / "stores", results=tmp_path / "results" / run_id,
            scratch=tmp_path / "scratch" / run_id, workspaces=tmp_path / "workspaces", workers=1, min_free_gib=0, store_hash="tree",
            checkouts=tmp_path / "checkouts", **extra,
        )  # fmt: skip

    # F1 knows every answer, closed book knows the even ones, the control is always wrong.
    answers = {f"H-{i}|F1": f"v{i}" for i in range(10)} | {
        f"H-{i}|N": f"v{i}" for i in range(0, 10, 2)
    }
    patches = {"C-2|F1": agent_patch, "C-7|F1": agent_patch, "C-7|N": agent_patch}
    return {
        "config": config,
        "answers": answers,
        "patches": patches,
        "gradle_calls": calls,
        "tmp": tmp_path,
    }


def agent(bench, **modes):
    return FakeAgent(bench["answers"], bench["patches"], **modes)


def stable(summary):
    return {k: v for k, v in summary.items() if k != "updated_at_unix"}


def summary_of(config):
    return json.loads((config.results / "summary.json").read_text())


def test_plan_rotates_arms_and_runs_control_on_every_fifth_task():
    arms = [{"id": "N"}, {"id": "A"}, {"id": "B"}, {"id": "K", "subset": "control"}]
    rows = runner.plan([f"t{i}" for i in range(7)], arms)
    assert rows[0][2] == ["N", "A", "B", "K"]
    assert (
        rows[1][2] == ["A", "B", "N"]
        and rows[2][2] == ["B", "N", "A"]
        and rows[3][2] == ["N", "A", "B"]
    )
    assert rows[5][2] == ["A", "B", "K", "N"]  # index 5: control present, rotated by 5 % 4
    assert [i for i, _, a in rows if "K" in a] == [0, 5]
    firsts = {a[0] for _, _, a in rows}
    assert firsts >= {"N", "A", "B"}


def test_access_paragraph_is_identical_within_an_access_type():
    server = {"name": "history"}
    task = {"prompt": "Q?", "kind": "historical"}
    files = {runner.build_prompt(task, {"id": i, "access": "files"}, server) for i in ("F0", "F6")}
    mcp = {
        runner.build_prompt(task, {"id": i, "access": "mcp", "mcp_kind": "rag"}, server)
        for i in ("R-hyb", "R2")
    }
    none = runner.build_prompt(task, {"id": "N", "access": "none"}, server)
    assert len(files) == 1 and len(mcp) == 1
    assert "/context" in next(iter(files)) and "ls, grep/rg, cat" in next(iter(files))
    assert "`history`" in next(iter(mcp)) and "search, read_event" in next(iter(mcp))
    assert "No project-history context is provided" in none and "/context" not in none
    research = {
        runner.build_prompt(task, {"id": i, "access": "mcp", "mcp_kind": "research"}, server)
        for i in ("F0", "F5", "F6")
    }
    assert len(research) == 1
    paragraph = next(iter(research)).split("\n\n")[1]
    assert paragraph == (
        "The MCP server `history` holds the project's recorded history up to the cutoff. Its tool "
        "`context_research(question)` asks Harnext's research agent, which reads the stored "
        "context and returns a cited answer; ask as many focused questions as you need. Nothing "
        "later than the cutoff is in it and there is no network access."
    )
    assert "context_research" in paragraph and "list_dir" not in paragraph and "grep" not in paragraph


def test_shipped_arms_serve_every_file_arm_through_context_research(tmp_path):
    arms, _ = runner.load_arms(runner.ROOT / "benchmarks/kafka-500-v2/arms.yaml")
    kinds = {a["id"]: a.get("mcp_kind") for a in arms}
    assert {kinds[f] for f in ("F0", "F1", "F2", "F3", "F4-fact", "F4-entity", "F4-topic", "F5", "F6")} == {"research"}  # fmt: skip
    assert kinds["R-hyb"] == "rag" and "G-graphiti" not in kinds and "files" not in kinds.values()
    bad = tmp_path / "arms.yaml"
    bad.write_text(yaml.safe_dump({"arms": [{"id": "X", "access": "mcp", "mcp_kind": "files", "store": "X"}]}))
    with pytest.raises(ValueError, match="unknown mcp_kind"):
        runner.load_arms(bad)


def test_reader_infra_row_in_receipts_pauses_and_records_nothing(bench):
    """The research reader's provider refusal reaches the runner only through the receipts
    (the agent itself finished normally): the trial is not recorded and resumes cleanly."""
    reference = bench["config"]("ref-reader")
    assert runner.Runner(reference, agent(bench)).live() == 0
    receipts = reference.results / "trials/H-0/F5-mcp/mcp-receipts.jsonl"
    rows = [json.loads(line) for line in receipts.read_text().splitlines()]
    assert rows[1]["tool"] == "context_research" and rows[1]["research"]["files"] == ["INDEX.md"]
    assert rows[1]["args"]["question"].startswith("Q0?") and "infra" not in rows[1]
    config = bench["config"]("reader-down")
    seen = agent(bench, reader_infra_at=3)  # task 0 runs N, F1, F5-mcp: the third call is the MCP arm
    assert runner.Runner(config, seen).live() == runner.EXIT_PAUSED
    assert seen.seen[2] == ("H-0", "F5-mcp")
    assert len(list(config.results.glob("trials/*/*/result.json"))) == 2
    paused = summary_of(config)
    assert paused["status"] == "paused" and "research reader infrastructure error" in paused["reason"]
    assert "usage limit" in paused["reason"] and paused["not_recorded"] == ["H-0/F5-mcp"]
    assert not list(config.scratch.glob("*/*"))
    assert runner.Runner(config, agent(bench)).live() == 0
    assert stable(summary_of(config)) == stable(summary_of(reference))


def test_full_run_records_everything_then_resume_reruns_nothing(bench):
    config = bench["config"]("full")
    first = agent(bench)
    assert runner.Runner(config, first).live() == 0
    assert first.calls == 10 * 2 + 2  # N and F1 everywhere, control on run_order index 0 and 5
    summary = summary_of(config)
    assert summary["status"] == "complete" and summary["recorded"] == summary["target"] == 22
    assert summary["arms"]["F1"]["success"] == 10 and summary["arms"]["N"]["success"] == 5
    assert summary["arms"]["F5-mcp"]["n"] == 2
    assert summary["paired_vs_N"]["F1"] == {
        "pairs": 10,
        "helped": 5,
        "harmed": 0,
        "both": 5,
        "neither": 0,
    }
    trial = config.results / "trials/C-7/F1"
    result = json.loads((trial / "result.json").read_text())
    assert result["success"] and result["grade"]["resolved"] and not result["grade"]["patch_empty"]
    assert result["prompt_sha256"] == sha((trial / "prompt.txt").read_text())
    manifest = json.loads((trial / "archive-manifest.json").read_text())["files_sha256"]
    assert manifest["result.json"] == sha((trial / "result.json").read_bytes())
    assert "return 2" in (trial / "patch.diff").read_text()
    assert (
        json.loads((config.results / "trials/C-2/N/result.json").read_text())["grade"]["graded_by"]
        == "empty_patch"
    )
    assert len(bench["gradle_calls"]) == 3 and not any(config.scratch.iterdir())
    second = agent(bench)
    assert runner.Runner(config, second).live() == 0
    assert second.calls == 0 and stable(summary_of(config)) == stable(summary)


@pytest.mark.parametrize("k", [1, 4, 9])
def test_infra_error_at_trial_k_pauses_with_k_minus_1_results_and_resume_matches(bench, k, capsys):
    reference = bench["config"]("reference")
    assert runner.Runner(reference, agent(bench)).live() == 0
    config = bench["config"]("paused")
    assert runner.Runner(config, agent(bench, infra_at=k)).live() == runner.EXIT_PAUSED
    assert len(list(config.results.glob("trials/*/*/result.json"))) == k - 1
    paused = summary_of(config)
    assert paused["status"] == "paused" and "usage limit" in paused["reason"]
    assert (
        len(paused["not_recorded"]) == 1
        and "resume with the same command" in capsys.readouterr().err
    )
    assert not list((config.scratch).glob("*/*"))  # the unit in flight left nothing behind
    resumed = agent(bench)
    assert runner.Runner(config, resumed).live() == 0
    assert resumed.calls == 22 - (k - 1)
    assert stable(summary_of(config)) == stable(summary_of(reference))


def test_grading_pause_is_not_a_failed_trial(bench, monkeypatch):
    config = bench["config"]("gradle-down")

    def offline(workspace, tests, image, log, **kwargs):
        log.write_text(
            "> Could not resolve org.foo:bar:1.0.\n  No cached version available for offline mode\n"
        )
        return 1, 3.0

    monkeypatch.setattr(kc, "gradle", offline)
    assert runner.Runner(config, agent(bench)).live() == runner.EXIT_PAUSED
    assert not (config.results / "trials/C-2/F1").exists()
    assert "grading infrastructure" in summary_of(config)["reason"]


def test_timeout_is_recorded_as_a_model_result(bench):
    config = bench["config"]("timeout")
    assert runner.Runner(config, agent(bench, timeout_at=2)).live() == 0
    rows = [json.loads(p.read_text()) for p in config.results.glob("trials/*/*/result.json")]
    timed = [r for r in rows if r["status"] == "timeout"]
    assert len(timed) == 1 and not timed[0]["success"]


def test_leftover_scratch_and_staging_from_a_crash_are_removed(bench):
    config = bench["config"]("crash")
    (config.scratch / "H-0/N/out").mkdir(parents=True)
    (config.results / "trials/H-0/N.staging").mkdir(parents=True)
    assert runner.Runner(config, agent(bench)).live() == 0
    assert not (config.results / "trials/H-0/N.staging").exists()
    assert (config.results / "trials/H-0/N/result.json").exists()


def test_preflight_refuses_changed_tasks_and_changed_stores(bench):
    config = bench["config"]("pins")
    assert runner.Runner(config, agent(bench, infra_at=3)).live() == runner.EXIT_PAUSED
    (bench["tmp"] / "stores/F1/export/2023-06/INDEX.md").write_text("tampered\n")
    with pytest.raises(ValueError, match="stores changed since the run began"):
        runner.Runner(config, agent(bench)).live()
    with (config.benchmark / "tasks.private.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="differs from the frozen manifest"):
        runner.Runner(config, agent(bench))


def test_git_store_pins_tree_hash_and_resume_refuses_a_moved_tag(bench):
    tmp = bench["tmp"]
    tree = tmp / "stores/F1/tree"
    tree.mkdir()
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    sh(*git, "init", "-q", "-b", "main", cwd=tree)
    (tree / "INDEX.md").write_text("# F1 from git\n")
    sh(*git, "add", "-A", cwd=tree)
    sh(*git, "commit", "-qm", "ckpt", cwd=tree)
    sh("git", "tag", "ckpt-2023-06", cwd=tree)
    expected = sh("git", "rev-parse", "ckpt-2023-06^{tree}", cwd=tree).strip()
    (tmp / "stores/F1/export/2023-06/INDEX.md").unlink()  # no materialised export at all
    config = bench["config"]("git-pins", keep_per_arm=1)
    seen = agent(bench, infra_at=4)
    assert runner.Runner(config, seen).live() == runner.EXIT_PAUSED
    manifest = json.loads((config.results / "manifest.json").read_text())
    assert manifest["stores"]["F1/2023-06"] == expected
    assert (tmp / "checkouts/F1/2023-06/INDEX.md").read_text() == "# F1 from git\n"
    assert json.loads((tmp / "checkouts/F1/2023-06.manifest.json").read_text())["tree"] == expected  # fmt: skip
    assert [c for _, c in seen.seen if c == "F1"], "F1 trials ran from the checkout"
    (tree / "INDEX.md").write_text("rebuilt\n")
    sh(*git, "commit", "-qam", "rebuilt", cwd=tree)
    sh("git", "tag", "-f", "ckpt-2023-06", cwd=tree)
    with pytest.raises(ValueError, match="stores changed since the run began"):
        runner.Runner(config, agent(bench)).live()
    sh("git", "tag", "-d", "ckpt-2023-06", cwd=tree)
    with pytest.raises(ValueError, match="F1: tag ckpt-2023-06 does not exist"):
        runner.Runner(config, agent(bench)).check_resume()


def test_check_resume_makes_no_calls_and_counts_per_arm(bench, capsys):
    config = bench["config"]("check")
    assert runner.Runner(config, agent(bench, infra_at=6)).live() == runner.EXIT_PAUSED
    capsys.readouterr()
    idle = agent(bench)
    assert runner.Runner(config, idle).check_resume() == 0
    printed = json.loads(capsys.readouterr().out)
    assert idle.calls == 0 and printed["recorded"] == 5 and printed["remaining"] == 17
    assert sum(v["recorded"] for v in printed["per_arm"].values()) == 5


def test_arm_subset_and_version_pin(bench):
    config = bench["config"]("subset", arms=["N"])
    only = agent(bench, infra_at=2)
    assert runner.Runner(config, only).live() == runner.EXIT_PAUSED
    newer = agent(bench)
    setattr(newer, "version", lambda: "fake-2")  # noqa: B010
    with pytest.raises(ValueError, match="version"):
        runner.Runner(config, newer).live()
    assert {arm for _, arm in only.seen} == {"N"}


def test_report_on_partial_run(bench):
    config = bench["config"]("partial")
    assert runner.Runner(config, agent(bench, infra_at=12)).live() == runner.EXIT_PAUSED
    built = report.write(config.results)
    assert built["partial"] and built["recorded"] == 11
    page = (config.results / "report.html").read_text()
    assert "Partial run" in page and "http" not in page.replace("http-equiv", "")
    cell = built["paired"]["N"]["F1"]
    both = {t["task_id"] for t in built["trials"] if t["arm"] == "N"} & {
        t["task_id"] for t in built["trials"] if t["arm"] == "F1"
    }
    assert cell["pairs"] == len(both) == 5 and cell["helped"] == 3 and cell["harmed"] == 0
    assert 0 <= cell["mcnemar_exact_p"] <= 1


def test_statistics():
    interval = report.wilson(8, 10)
    assert interval
    low, high = interval
    assert 0.49 < low < 0.50 and 0.94 < high < 0.95
    assert report.wilson(0, 0) is None
    assert report.mcnemar_exact(0, 0) == 1.0
    assert report.mcnemar_exact(5, 0) == pytest.approx(0.0625)
    assert report.mcnemar_exact(3, 3) == 1.0


def test_torn_trial_archive_is_quarantined_not_loaded(tmp_path):
    from harnext_eval.e2.v2.runner import atomic_json, intact, quarantine

    trial = tmp_path / "trials" / "T-1" / "N"
    atomic_json(trial / "result.json", {"status": "ok"})
    from harnext_eval.e2.benchmark import sha

    atomic_json(trial / "archive-manifest.json",
                {"files_sha256": {"result.json": sha((trial / "result.json").read_bytes())}})
    assert intact(trial) == {"status": "ok"}
    (trial / "result.json").write_text("")  # what a power cut can leave behind
    assert intact(trial) is None
    quarantine(tmp_path, trial)
    assert not trial.exists() and len(list((tmp_path / "quarantine").iterdir())) == 1


def test_server_started_needs_a_start_receipt(tmp_path):
    receipts = tmp_path / "receipts.jsonl"
    assert not runner.server_started(receipts)
    receipts.write_text('{"kind": "call"}\n')
    assert not runner.server_started(receipts)
    receipts.write_text('{"kind": "start"}\n{"kind": "call"}\n')
    assert runner.server_started(receipts)


def test_split_queues_record_the_same_trials_as_one_queue(bench):
    reference = bench["config"]("one-queue")
    assert runner.Runner(reference, agent(bench)).live() == 0
    split = bench["config"]("two-queues").model_copy(update={"workers": 3, "coding_workers": 1})
    assert runner.Runner(split, agent(bench)).live() == 0
    assert stable(summary_of(split)) == stable(summary_of(reference))


def test_drain_finishes_in_flight_trials_and_records_them(bench):
    config = bench["config"]("drain")
    fake = agent(bench)
    run = runner.Runner(config, fake)
    original = fake.run

    def run_then_drain(spec):
        result = original(spec)
        if fake.calls == 3:
            run.drain()  # arrives while trial 3 is in flight
        return result

    fake.run = run_then_drain
    assert run.live() == runner.EXIT_PAUSED and run.draining and not run.interrupted
    assert len(list(config.results.glob("trials/*/*/result.json"))) == 3  # the in-flight one kept
    rest = agent(bench)
    assert runner.Runner(config, rest).live() == 0 and rest.calls == 22 - 3
