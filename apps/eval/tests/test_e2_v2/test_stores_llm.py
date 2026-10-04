"""LLM-store guards with stub harnesses: F2 byte identity, PauseRun, failed windows, F5 product lane."""

import json

import pytest
from harnext_builder.harness.base import ConversationTranscript, TranscriptTurn
from harnext_eval.e2.v2.errors import PauseRun
from harnext_eval.e2.v2.stores import build, product, prompts, synthetic
from harnext_eval.e2.v2.stores.base import Repo, tree_files
from harnext_eval.e2.v2.stores.fake import FakeBuilder
from harnext_eval.e2.v2.stores.llm_files import ARMS, outside, verify_moves

EVENTS = synthetic.events(40, hours=12)  # January only


class Stub(FakeBuilder):
    """Fake builder that misbehaves on chosen calls: `script[n]` is an error message, a
    callable(tree) run after the honest work, or None."""

    def __init__(self, arm, script):
        super().__init__(arm)
        self.script, self.calls = script, 0

    async def run(self, req):
        step = self.script.get(self.calls)
        self.calls += 1
        if isinstance(step, str):
            (__import__("pathlib").Path(req.working_dir) / "half.md").write_text("partial\n")
            turns = [TranscriptTurn(role="assistant", content="working")]
            return ConversationTranscript(
                harness="fake", turns=turns, stop_reason="error", error=step
            )
        transcript = await super().run(req)
        if step:
            step(__import__("pathlib").Path(req.working_dir))
        return transcript


def store(arm, script, **kwargs):
    return ARMS[arm](builder="fake", harness=Stub(arm, script), window_events=10, **kwargs)


def state(out):
    return json.loads((out / "build-state.json").read_text())


def edit_one(tree):
    victim = sorted(tree.rglob("syn-*.md"))[0]
    victim.write_text(victim.read_text() + " ")


def test_move_verifier_catches_edit_loss_and_foreign_files():
    pre = {"events/a.md": "h1", "events/b.md": "h2"}
    assert verify_moves(pre, {"x/a.md": "h1", "y/b.md": "h2"}) is None
    assert "1 file(s) lost or edited" in str(
        verify_moves(pre, {"x/a.md": "h1", "y/b.md": "EDITED"})
    )
    assert "foreign" in str(
        verify_moves(pre, {"x/a.md": "h1", "y/b.md": "h2", "x/README.md": "h3"})
    )
    assert verify_moves(pre, {"x/a.md": "h1", "y/b.md": "h2", "z/b.md": "h2"})  # a copy


def test_f2_edit_is_rolled_back_retried_then_falls_back_to_f0_paths(tmp_path):
    f2 = store("F2", {3: edit_one, 4: edit_one})  # last window: both attempts edit a file
    f2.build(iter(EVENTS), tmp_path)
    log = [json.loads(line) for line in (tmp_path / "build-log.jsonl").open()]
    assert [a["problem"] is None for a in log[3]["attempts"]] == [False, False]
    assert "move-only violated" in log[3]["failed"]
    assert state(tmp_path)["failed_windows"] == [{"window": 3, "reason": log[3]["failed"]}]
    from harnext_eval.e2.v2.stores.base import render

    files = tree_files(tmp_path / "tree")
    assert sorted(files.values()) == sorted(
        __import__("hashlib").sha256(render(e).encode()).hexdigest() for e in EVENTS
    )
    assert sum(path.startswith("events/") for path in files) == 10  # the failed window, F0 paths


def test_f2_single_violation_recovers_on_retry(tmp_path):
    store("F2", {0: edit_one}).build(iter(EVENTS), tmp_path)
    assert state(tmp_path)["failed_windows"] == []
    assert not any(p.startswith("events/") for p in tree_files(tmp_path / "tree"))


def replan(change):
    """Misbehaviour applied to the honest move plan the fake builder wrote."""

    def step(tree):
        plan = json.loads((tree / "_moves.json").read_text())
        (tree / "_moves.json").write_text(json.dumps(change(plan)))

    return step


def first(plan, target):
    source = sorted(plan)[0]
    return {**plan, source: target(plan, source) if callable(target) else target}


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda p: first(p, "../escaped.md"), "outside the store"),
        (lambda p: first(p, "/tmp/escaped.md"), "outside the store"),
        (lambda p: first(p, ".git/hooks/x.md"), "reserved location"),
        (lambda p: first(p, "a/./b.md"), "not normalized"),
        (lambda p: first(p, "notes/x.txt"), "not .md"),
        (lambda p: first(p, lambda plan, s: sorted(plan)[1]), "target exists"),
        (lambda p: first(p, lambda plan, s: plan[sorted(plan)[1]]), "named twice"),
        (lambda p: first(p, lambda plan, s: plan[sorted(plan)[1]] + "/x.md"), "collides"),
        (lambda p: {"events/nowhere.md": "x/nowhere.md"}, "no change to the store"),  # mistyped sources are skipped
        (lambda p: [], "not a JSON object"),
        (lambda p: first(p, "_moves-9.json"), "reserved location"),
        (lambda p: {}, "no change to the store"),
    ],
)
def test_f2_bad_move_plan_fails_the_window_and_moves_nothing(tmp_path, change, reason):
    f2 = store("F2", {0: replan(change), 1: replan(change)})
    f2.build(iter(EVENTS[:10]), tmp_path)
    log = [json.loads(line) for line in (tmp_path / "build-log.jsonl").open()]
    assert reason in log[0]["failed"]
    files = tree_files(tmp_path / "tree")
    assert len(files) == 10 and all(path.startswith("events/") for path in files)


def test_f2_any_write_besides_the_plan_fails_the_window(tmp_path):
    def note(tree):
        (tree / "README.md").write_text("taxonomy notes\n")

    f2 = store("F2", {0: note, 1: edit_one})
    f2.build(iter(EVENTS[:10]), tmp_path)
    log = [json.loads(line) for line in (tmp_path / "build-log.jsonl").open()]
    assert ["foreign" in a["problem"] or "edited" in a["problem"] for a in log[0]["attempts"]] == [True, True]  # fmt: skip
    assert all(p.startswith("events/") for p in tree_files(tmp_path / "tree"))


def test_f2_plan_may_name_folders_and_continue_in_numbered_files(tmp_path):
    def split(tree):
        plan = json.loads((tree / "_moves.json").read_text())
        items = sorted(plan.items())
        (tree / "_moves.json").write_text(json.dumps(dict(items[:4])))
        (tree / "_moves-2.json").write_text(json.dumps({a: "kept-names/" for a, _ in items[4:]}))

    store("F2", {0: split}).build(iter(EVENTS[:10]), tmp_path)
    files = tree_files(tmp_path / "tree")
    assert state(tmp_path)["failed_windows"] == [] and len(files) == 10
    assert sum(p.startswith("kept-names/syn-") for p in files) == 6
    assert sum(p.startswith("by-entity/") for p in files) == 4


def test_f2_plan_is_executed_removed_and_pins_its_own_digest(tmp_path):
    from harnext_eval.e2.v2.stores import foldered

    f2 = store("F2", {})
    f2.build(iter(EVENTS), tmp_path)
    files = tree_files(tmp_path / "tree")
    assert len(files) == len(EVENTS) and all(p.startswith("by-entity/") for p in files)
    assert not (tmp_path / "tree/events").exists() and "_moves.json" not in files
    assert state(tmp_path)["config"]["prompts_sha256"] == foldered.digest() != prompts.digest()
    assert f2.tools == ["Read(./**)", "Edit(./_moves*.json)"] and "Bash" in f2.denied
    assert "_moves.json" in foldered.system() and "Zero byte edits" in foldered.system()
    assert "- `by-entity/" in foldered.folders(files)


def test_infra_error_pauses_with_clean_tree_and_cursor_then_resumes_identically(tmp_path):
    paused = store("F5", {2: "API Error: 429 rate limit exceeded"})
    with pytest.raises(PauseRun):
        paused.build(iter(EVENTS), tmp_path / "paused")
    repo, cursor = Repo(tmp_path / "paused/tree"), state(tmp_path / "paused")
    assert cursor["window"] == 2 and cursor["last_event_id"] == EVENTS[19]["id"]
    assert repo.git("status", "--porcelain") == "" and repo.head() == cursor["head"]
    assert (
        not (tmp_path / "paused/tree/half.md").exists()
        and not (tmp_path / "paused/tree/_event").exists()
    )
    store("F5", {}).build(iter(EVENTS), tmp_path / "paused")
    store("F5", {}).build(iter(EVENTS), tmp_path / "straight")
    assert repo.head() == Repo(tmp_path / "straight/tree").head()


def test_model_failure_retries_once_then_keeps_events_under_unprocessed(tmp_path):
    f4 = store("F4-entity", {1: "error_max_turns", 2: "model wrote garbage", 3: "error_max_turns"})
    f4.build(iter(EVENTS), tmp_path)
    final = state(tmp_path)
    assert [f["window"] for f in final["failed_windows"]] == [1]  # window 2 recovered on retry
    kept = sorted(p.name for p in (tmp_path / "tree/_unprocessed").rglob("*.md"))
    assert kept == sorted(f"{e['id']}.md" for e in EVENTS[10:20])
    assert not (tmp_path / "tree/half.md").exists()
    assert final["audit"] == {
        "events": 40,
        "processed": 30,
        "unprocessed": 10,
        "through": EVENTS[-1]["id"],
    }


def test_unprocessed_is_guarded_per_window_and_checked_by_the_audit(tmp_path):
    def wipe(tree):
        __import__("shutil").rmtree(tree / "_unprocessed")

    f4 = store("F4-entity", {1: "error_max_turns", 2: "error_max_turns", 4: wipe})
    f4.build(iter(EVENTS), tmp_path)  # window 3's first attempt wipes, is rolled back, retried
    log = [json.loads(line) for line in (tmp_path / "build-log.jsonl").open()]
    assert log[3]["attempts"][0]["problem"] == "builder changed files under _unprocessed/"
    ids = [e["id"] for e in EVENTS[10:20]]
    assert f4.kept_verbatim(tmp_path / "tree", ids) == []
    wipe(tmp_path / "tree")
    assert f4.kept_verbatim(tmp_path / "tree", ids) == ids
    with pytest.raises(RuntimeError, match="not kept verbatim"):
        f4.audit(tmp_path, state(tmp_path), [(e["time"], e["id"]) for e in EVENTS])


def test_three_consecutive_failed_windows_pause_the_run(tmp_path):
    with pytest.raises(PauseRun, match="consecutive"):
        store("F6", {n: "model wrote garbage" for n in range(6)}).build(iter(EVENTS), tmp_path)
    assert state(tmp_path)["window"] == 2


def test_silent_cli_failure_is_infrastructure_not_a_model_failure(tmp_path):
    class Dead:
        name = "fake"

        async def run(self, req):
            return ConversationTranscript(
                harness="fake", stop_reason="error", error="ProcessError: exit 1"
            )

    with pytest.raises(PauseRun):
        ARMS["F6"](builder="fake", harness=Dead(), window_events=10).build(iter(EVENTS), tmp_path)


def test_f5_is_the_product_batch_lane_seed_prompt_tools_and_per_entity_instruction(tmp_path):
    from harnext_builder.agentfs.seed import SEED_FILES
    from harnext_builder.harness.base import ALLOWED_TOOLS, DENIED_TOOLS
    from harnext_builder.prompts import _BATCH_TMPL, SYSTEM_PROMPT

    seen = []

    class Spy(Stub):
        async def run(self, req):
            seen.append(req)
            return await super().run(req)

    f5 = ARMS["F5"](builder="fake", harness=Spy("F5", {}), window_events=10)
    f5.build(iter(EVENTS[:10]), tmp_path)
    tree = tmp_path / "tree"
    assert Repo(tree).git("log", "-1", "--format=%s", "HEAD~1") == "store: seed"
    for path, text in SEED_FILES.items():  # the product's genesis files, byte for byte
        assert Repo(tree).git("show", f"HEAD~1:{path}") == text.strip()
    req = seen[0]
    assert req.system_prompt is SYSTEM_PROMPT and req.system_prompt == f5.system()
    # the product's policy with Bash (unusable here) swapped for the eval's delete/rename tools
    assert req.allowed_tools == [t for t in ALLOWED_TOOLS if t != "Bash"] + ["mcp__fs__delete", "mcp__fs__rename"]  # fmt: skip
    assert req.disallowed_tools == [*DENIED_TOOLS, "Bash"]
    assert req.event_files == [] and "Zhou" not in req.instruction + req.system_prompt
    # EVENTS[:10]: 5 JIRA events of KAFKA-9000 + 1 of KAFKA-9001, 3 PR events, 1 mail thread
    wording = _BATCH_TMPL.split("```")[0]
    subjects = ["issue:KAFKA-9000", "pr:apache/kafka/14000", "thread:dev-0", "issue:KAFKA-9001"]
    sizes = [5, 3, 1, 1]
    for subject, n in zip(subjects, sizes, strict=True):
        assert wording.format(n=n, subjects=subject) in req.instruction
    assert req.instruction.count("A batch window closed for this organization") == 4
    blocks = [json.loads(b) for b in __import__("re").findall(r"```json\n(.*?)\n```", req.instruction, flags=16)]
    assert [len(b) for b in blocks] == sizes
    event = blocks[1][0]
    assert {k: event[k] for k in ("id", "source", "type", "subject", "mgtenant")} == {
        "id": EVENTS[3]["id"], "source": "github:apache/kafka", "type": "com.github.pr_opened",
        "subject": "pr:apache/kafka/14000", "mgtenant": "apache-kafka"}  # fmt: skip
    assert event["data"] == {k: EVENTS[3][k] for k in ("title", "text", "actor", "meta")} | {
        "related": ["issue:KAFKA-9000"]
    }
    for name in ("OVERVIEW.md", "facts.md", "timeline.md"):
        assert (tree / "entities/pr/apache__kafka__14000" / name).exists()
    timeline = (tree / "entities/issue/KAFKA-9000/timeline.md").read_text().splitlines()
    assert len(timeline) == 5 and timeline[0].startswith(f"- 2023-01-02 [jira:KAFKA#{EVENTS[0]['id']}]")
    assert "| `issue:KAFKA-9000` | issue |" in (tree / "INDEX.md").read_text()
    assert state(tmp_path)["config"]["prompts_sha256"] == product.digest() != prompts.digest()


def test_f5_slice_too_large_to_inline_mounts_the_events_and_keeps_the_batches(monkeypatch):
    monkeypatch.setattr(product, "INLINE_LIMIT", 3000)
    instruction, files = product.request(EVENTS[:10])
    assert instruction.count("A batch window closed for this organization") == 4
    assert "`_event/MANIFEST.md`" in instruction and EVENTS[0]["text"] not in instruction
    paths = [f.path for f in files]
    assert paths == ["_event/issue_KAFKA-9000/events.json", "_event/pr_apache_kafka_14000/events.json",
                     "_event/thread_dev-0/events.json", "_event/issue_KAFKA-9001/events.json",
                     "_event/MANIFEST.md"]  # fmt: skip
    assert f'"events_file": "{paths[0]}"' in instruction and EVENTS[0]["id"] in instruction
    assert [e["data"]["text"] for e in json.loads(files[0].content)][0] == EVENTS[0]["text"]


def test_f5_fake_build_reads_mounted_batches_into_the_same_tree(tmp_path, monkeypatch):
    ARMS["F5"](builder="fake", window_events=10).build(iter(EVENTS), tmp_path / "inline")
    monkeypatch.setattr(product, "INLINE_LIMIT", 3000)
    ARMS["F5"](builder="fake", window_events=10).build(iter(EVENTS), tmp_path / "mounted")
    assert tree_files(tmp_path / "inline/tree") == tree_files(tmp_path / "mounted/tree")
    assert "entities/thread/dev-0/timeline.md" in tree_files(tmp_path / "mounted/tree")


def test_f5_model_failure_rolls_back_retries_and_falls_back_to_unprocessed(tmp_path):
    f5 = store("F5", {1: "error_max_turns", 2: "error_max_turns", 3: "model wrote garbage"})
    f5.build(iter(EVENTS), tmp_path)
    final = state(tmp_path)
    assert [f["window"] for f in final["failed_windows"]] == [1]  # window 2 recovered on retry
    kept = sorted(p.name for p in (tmp_path / "tree/_unprocessed").rglob("*.md"))
    assert kept == sorted(f"{e['id']}.md" for e in EVENTS[10:20])
    assert not (tmp_path / "tree/half.md").exists() and not (tmp_path / "tree/_event").exists()
    assert final["audit"]["unprocessed"] == 10 and final["audit"]["processed"] == 30

    def drop_schema(tree):
        (tree / "_meta/schema.md").unlink()

    store("F5", {0: drop_schema}).build(iter(EVENTS[:10]), tmp_path / "layout")
    log = json.loads((tmp_path / "layout/build-log.jsonl").read_text().splitlines()[0])
    assert [a["problem"] for a in log["attempts"]] == ["layout file removed: ['_meta/schema.md']", None]


def test_f3_reorganizes_on_schedule_and_at_the_checkpoint(tmp_path):
    f3 = ARMS["F3"](builder="fake", window_events=10, reorg_every=3)
    f3.build(iter(EVENTS), tmp_path)
    log = [json.loads(line) for line in (tmp_path / "build-log.jsonl").open()]
    assert [("attempts" in r, r["intake_left"]) for r in log] == [
        (False, 10),
        (False, 20),
        (True, 0),
        (True, 0),
    ]
    text = "\n".join(p.read_text() for p in (tmp_path / "tree/subjects").rglob("*.md"))
    assert all(e["text"] in text for e in EVENTS)  # keep every fact


def test_outside_paths_are_detected(tmp_path):
    def call(tool, **inputs):
        return TranscriptTurn(role="tool_use", tool_name=tool, content=json.dumps(inputs))

    inside = [
        call("Read", file_path=f"{tmp_path}/a.md"),
        call("Bash", command="mkdir -p x/y && mv a.md x/y/ 2>/dev/null"),
    ]
    assert outside(tmp_path, ConversationTranscript(harness="x", turns=inside)) == []
    bad = [
        call("Read", file_path="/etc/passwd"),
        call("Bash", command="cat ../../benchmarks/tasks.private.jsonl"),
    ]
    assert outside(tmp_path, ConversationTranscript(harness="x", turns=bad)) == [
        "../../benchmarks/tasks.private.jsonl",
        "/etc/passwd",
    ]
    said = lambda text: TranscriptTurn(role="tool_result", content=text)  # noqa: E731
    refused = [bad[0], said("Permission to use Read has been denied because …"),
               call("mcp__fs__delete", path="/x.md"), said("delete refused (outside the store): '/x.md'")]  # fmt: skip
    assert outside(tmp_path, ConversationTranscript(harness="x", turns=refused)) == []
    answered = [bad[0], said("root:x:0:0")]  # the same call, not refused: still a violation
    assert outside(tmp_path, ConversationTranscript(harness="x", turns=answered)) == ["/etc/passwd"]


def test_prompts_name_the_reader_quote_the_contract_and_leak_nothing():
    for arm in set(ARMS) - {"F5"}:  # F5 sends the product's prompt, asserted above
        text = prompts.system(arm)
        assert "grep" in text and "Apache Kafka" in text
        assert "Distance mirrors relatedness" in text and "Structure serves the search" in text
        assert "benchmark" not in text.lower() and "kafka-500" not in text
    assert "Keep every fact, faithfully." in prompts.system("F3")
    assert "Zero byte edits" in prompts.system("F2")
    assert len(prompts.digest()) == 64


def test_cli_exits_75_on_pause(tmp_path, monkeypatch, capsys):
    events = synthetic.dump(tmp_path / "events.jsonl", 20)
    monkeypatch.setattr(build, "make", lambda arm, **kw: store("F6", {0: "401 unauthorized"}))
    assert (
        build.main(
            ["--arm", "F6", "--builder", "fake", "--events", str(events), "--root", str(tmp_path), "--no-wait"]
        )
        == 75
    )
    assert "PAUSED" in capsys.readouterr().err


def test_fs_tools_delete_and_rename_inside_the_tree_only(tmp_path):
    from harnext_eval.e2.v2.stores import fs_tools

    tree = tmp_path / "tree"
    for name in ("a/one.md", "a/two.md", "b/three.md", "_unprocessed/raw.md", ".git/config"):
        (tree / name).parent.mkdir(parents=True, exist_ok=True)
        (tree / name).write_text(name)
    (tmp_path / "secret.md").write_text("outside")
    (tree / "link.md").symlink_to(tmp_path / "secret.md")

    assert fs_tools.rename(tree, "a/one.md", "topics/deep/one.md") == "renamed a/one.md -> topics/deep/one.md"
    assert (tree / "topics/deep/one.md").read_text() == "a/one.md"
    fs_tools.rename(tree, "b", "c/d")  # a whole folder; the emptied parent is pruned
    assert (tree / "c/d/three.md").exists() and not (tree / "b").exists()
    fs_tools.delete(tree, "a/two.md")
    assert not (tree / "a").exists()  # last file gone, folder gone

    refused = [
        lambda: fs_tools.delete(tree, "../secret.md"),
        lambda: fs_tools.delete(tree, str(tmp_path / "secret.md")),
        lambda: fs_tools.delete(tree, "link.md"),
        lambda: fs_tools.delete(tree, "c/../../secret.md"),
        lambda: fs_tools.delete(tree, ".git/config"),
        lambda: fs_tools.delete(tree, "_unprocessed/raw.md"),
        lambda: fs_tools.delete(tree, "c/d"),
        lambda: fs_tools.delete(tree, "nowhere.md"),
        lambda: fs_tools.delete(tree, None),
        lambda: fs_tools.rename(tree, "topics/deep/one.md", "../escaped.md"),
        lambda: fs_tools.rename(tree, "topics/deep/one.md", "/tmp/escaped.md"),
        lambda: fs_tools.rename(tree, "topics/deep/one.md", ".git/hooks/x.md"),
        lambda: fs_tools.rename(tree, "topics/deep/one.md", "c/d/three.md"),
        lambda: fs_tools.rename(tree, "topics/deep/one.md", "c/d/three.md/x.md"),
        lambda: fs_tools.rename(tree, "_unprocessed/raw.md", "raw.md"),
        lambda: fs_tools.rename(tree, "c", "c/inner"),
        lambda: fs_tools.rename(tree, "link.md", "mine.md"),
    ]
    for call in refused:
        with pytest.raises(ValueError, match="refused"):
            call()
    assert (tmp_path / "secret.md").read_text() == "outside"
    assert sorted(tree_files(tree)) == sorted(
        ["_unprocessed/raw.md", "c/d/three.md", "link.md", "topics/deep/one.md"]
    )


def test_no_builder_has_a_shell_and_every_curating_arm_gets_delete_and_rename():
    from harnext_eval.e2.v2.stores import fs_tools

    for arm, cls in ARMS.items():
        assert "Bash" not in cls.tools and "Bash" in cls.denied, arm
        assert (fs_tools.DELETE in cls.tools and fs_tools.RENAME in cls.tools) == (arm != "F2")
    for arm in ("F3", "F4-fact", "F6"):
        assert "`delete` tool" in prompts.system(arm) and "Bash" not in prompts.system(arm)
    text, _ = product.request(EVENTS[:3])
    assert product.FS_NOTE in text and "`rename` tool" in product.FS_NOTE


def test_outside_guard_ignores_contents_patterns_and_spills_but_catches_escapes(tmp_path):
    from pathlib import Path
    from types import SimpleNamespace as NS

    from harnext_eval.e2.v2.stores.llm_files import outside

    def calls(*pairs):
        return NS(turns=[NS(role="tool_use", tool_name=n, content=c) for n, c in pairs])

    spill = f"{Path.home()}/.claude/projects/-x-tree/0ac58c7f-96e5/tool-results/toolu_01.txt"
    benign = calls(
        ("Glob", '{"pattern": "entities/issue/KAFKA-*/OVERVIEW.md"}'),
        ("Grep", '{"pattern": "https://cwiki.apache.org/confluence/x", "path": "issues"}'),
        ("Read", '{"file_path": "%s"}' % spill),
        ("Write", '{"file_path": "a.md", "content": "see /etc/default/kafka and https://reviews.apache.org/r/1 ... [trunc'),  # fmt: skip
    )
    assert outside(tmp_path, benign) == []
    for bad in (
        ("Write", '{"file_path": "/etc/x.md", "content": "hi"}'),
        ("Edit", '{"file_path": "../other/x.md", "old_string": "a ... [trunc'),
        ("Glob", '{"pattern": "/home/*/secrets"}'),
        ("Read", '{"file_path": "%s/../../x"}' % spill.rsplit("/", 1)[0]),
    ):
        assert outside(tmp_path, calls(bad)), bad
