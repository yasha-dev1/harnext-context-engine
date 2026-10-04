"""Store mechanics: windows, checkpoints, resume, export — fake builder, synthetic events."""

import json
import os
import re
import subprocess
from typing import Any

import pytest
from harnext_eval.e2.v2.stores import build, synthetic
from harnext_eval.e2.v2.stores.base import InjectedKill, Repo, next_month, tree_files, windows
from harnext_eval.e2.v2.stores.build import make

EVENTS = synthetic.events(130, hours=12)  # 2023-01-02 … 2023-03-08: three months
ARMS = ["F0", "F1", "F2", "F3", "F4-fact", "F4-entity", "F4-topic", "F5", "F6"]


def built(arm, out, *, until=None, kill=None, point="mid_window", events=EVENTS):
    store: Any = make(arm, builder="fake", window_events=10)
    store.kill_after, store.kill_point = kill, point
    store.build(iter(events), out, until=until)
    return store


def test_windows_never_cross_a_month_and_flag_its_last_window():
    parts = list(windows(EVENTS, 10))
    assert [e["id"] for _, batch, _ in parts for e in batch] == [e["id"] for e in EVENTS]
    assert all(
        len({e["time"][:7] for e in batch}) == 1 and len(batch) <= 10 for _, batch, _ in parts
    )
    assert [ym for ym, _, last in parts if last] == ["2023-01", "2023-02", "2023-03"]
    assert next_month("2023-12") == "2024-01"


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("point", ["mid_window", "after_commit"])
def test_killed_build_resumes_to_the_identical_store(arm, point, tmp_path):
    built(arm, tmp_path / "straight")
    with pytest.raises(InjectedKill):
        built(arm, tmp_path / "killed", kill=7, point=point)
    cursor = json.loads((tmp_path / "killed/build-state.json").read_text())
    assert (
        cursor["window"] == 7 and cursor["last_event_id"] == EVENTS[69]["id"]
    )  # 6 January windows + 1
    built(arm, tmp_path / "killed")
    straight, resumed = Repo(tmp_path / "straight/tree"), Repo(tmp_path / "killed/tree")
    assert resumed.head() == straight.head()  # same trees, messages, identities and dates
    assert resumed.git("tag", "--list") == straight.git("tag", "--list")
    assert resumed.git("status", "--porcelain") == ""
    states = [
        json.loads((tmp_path / d / "build-state.json").read_text()) for d in ("straight", "killed")
    ]
    assert states[0]["checkpoints"] == states[1]["checkpoints"] == ["2023-01", "2023-02", "2023-03"]


def test_commit_messages_carry_the_window_bounds(tmp_path):
    built("F0", tmp_path)
    subjects = Repo(tmp_path / "tree").git("log", "--reverse", "--format=%s").split("\n")
    assert subjects[2] == f"window 0: {EVENTS[0]['id']} .. {EVENTS[9]['id']} (10 events)"


@pytest.mark.parametrize("arm", ["F0", "F1", "F2", "F5"])
def test_checkpoint_export_holds_nothing_after_its_boundary(arm, tmp_path):
    store = built(arm, tmp_path)
    target = store.export(tmp_path, "2023-01")
    text = "\n".join(p.read_text() for p in target.rglob("*.md"))
    seen = set(re.findall(r"syn-\d{5}-[a-z_-]+", text))
    january = {e["id"] for e in EVENTS if e["time"] < "2023-02-01T00:00:00Z"}
    assert seen == january
    assert not re.search(r"2023-0[23]-\d\d", text)
    manifest = json.loads((tmp_path / "export/2023-01.manifest.json").read_text())
    files = tree_files(target)
    assert manifest["files"] == len(files) and manifest["bytes"] == sum(
        (target / f).stat().st_size for f in files
    )
    assert len(manifest["content_sha256"]) == 64 and not (target / ".git").exists()
    assert all(not os.access(target / f, os.W_OK) for f in files)
    assert store.export(tmp_path, "2023-01") == target  # re-export over a read-only tree
    assert json.loads((tmp_path / "export/2023-01.manifest.json").read_text()) == manifest


def test_until_stops_after_that_checkpoint_and_continues_later(tmp_path):
    built("F0", tmp_path / "a", until="2023-02")
    state = json.loads((tmp_path / "a/build-state.json").read_text())
    assert state["checkpoints"] == ["2023-01", "2023-02"] and not state["done"]
    built("F0", tmp_path / "a")
    built("F0", tmp_path / "b")
    assert Repo(tmp_path / "a/tree").head() == Repo(tmp_path / "b/tree").head()


def test_a_month_without_events_is_still_a_checkpoint(tmp_path):
    events = [e for e in EVENTS if e["time"][:7] != "2023-02"]
    built("F0", tmp_path, events=events)
    repo = Repo(tmp_path / "tree")
    assert repo.git("rev-parse", "ckpt-2023-02") == repo.git("rev-parse", "ckpt-2023-01")


def test_changed_configuration_refuses_to_resume(tmp_path):
    built("F0", tmp_path, until="2023-01")
    with pytest.raises(ValueError, match="window_events"):
        make("F0", window_events=25).build(iter(EVENTS), tmp_path)


def test_f0_is_verbatim_and_f1_is_derived_from_meta(tmp_path):
    built("F0", tmp_path / "f0")
    path = tmp_path / "f0/tree/events/2023/01/02/syn-00000-jira-created.md"
    head, body = path.read_text().split("---\n")[1:]
    assert body == f"\n{EVENTS[0]['text']}\n" and "id: syn-00000-jira-created" in head
    built("F1", tmp_path / "f1")
    overview = (tmp_path / "f1/tree/entities/issue/KAFKA-9000/OVERVIEW.md").read_text()
    assert (
        "status: Resolved" in overview
        and "resolution: Fixed" in overview
        and "- pr:14000" in overview
    )
    assert "status: merged" in (tmp_path / "f1/tree/entities/pr/14000/OVERVIEW.md").read_text()
    timeline = (tmp_path / "f1/tree/entities/issue/KAFKA-9000/timeline.md").read_text()
    assert timeline.count("\n- ") == 9 and "http" not in timeline


def test_cli_builds_exports_and_returns_zero(tmp_path, capsys):
    events = synthetic.dump(tmp_path / "events.jsonl", 40)
    args = [
        "--builder",
        "fake",
        "--events",
        str(events),
        "--root",
        str(tmp_path),
        "--export",
        "all",
    ]
    assert build.main(["--arm", "all-deterministic", *args]) == 0
    assert (tmp_path / "stores-smoke/F1/export/2023-01.manifest.json").exists()
    assert subprocess.run(["git", "-C", str(tmp_path / "stores-smoke/F0/tree"), "tag"],
                          capture_output=True, text=True).stdout.split() == ["ckpt-2023-01"]  # fmt: skip


def test_byte_cap_closes_a_window_first_and_never_drops_an_event():
    sized = [e | {"text": "x" * (900 if n % 7 else 5000)} for n, e in enumerate(EVENTS)]
    parts = list(windows(sized, 10, 2000))
    assert [e["id"] for _, batch, _ in parts for e in batch] == [e["id"] for e in EVENTS]
    for _, batch, _ in parts:
        weight = sum(len(e["text"]) for e in batch)
        assert len(batch) <= 10 and (weight <= 2000 or len(batch) == 1)  # oversized event: alone
    assert max(len(batch) for _, batch, _ in parts) == 2


@pytest.mark.parametrize("arm", ["F0", "F2", "F3", "F5"])
def test_build_end_audit_accounts_for_every_stream_event(arm, tmp_path):
    built(arm, tmp_path, until="2023-02")
    state = json.loads((tmp_path / "build-state.json").read_text())
    through = [e for e in EVENTS if e["time"] < "2023-03"]
    assert state["audit"] == {"events": len(through), "processed": len(through), "unprocessed": 0,
                              "through": through[-1]["id"]}  # fmt: skip
    log = [json.loads(line) for line in (tmp_path / "build-log.jsonl").open()]
    assert [i for r in log for i in r["ids"]] == [e["id"] for e in through]


def test_audit_fails_when_the_log_does_not_cover_the_stream(tmp_path):
    built("F0", tmp_path, until="2023-01")
    lines = (tmp_path / "build-log.jsonl").read_text().splitlines()
    record = json.loads(lines[2])
    record["ids"] = record["ids"][:-1]  # one event unaccounted for
    (tmp_path / "build-log.jsonl").write_text(
        "\n".join([*lines[:2], json.dumps(record), *lines[3:]]) + "\n"
    )
    with pytest.raises(RuntimeError, match="1 missing"):
        built("F0", tmp_path)


def test_unsorted_or_duplicated_stream_is_refused(tmp_path):
    with pytest.raises(ValueError, match="strictly sorted"):
        built("F0", tmp_path, events=[EVENTS[0], EVENTS[1], EVENTS[1]])


def test_tree_scan_cache_rereads_only_changed_files(tmp_path):
    (tmp_path / "a.md").write_text("one\ntwo\n")
    (tmp_path / "b.md").write_text("x")
    cache: dict = {}
    first = tree_files(tmp_path, cache)
    assert first == tree_files(tmp_path) and [cache["a.md"][3], cache["b.md"][3]] == [2, 1]
    (tmp_path / "b.md").write_text("changed\n")
    (tmp_path / "a.md").unlink()
    assert tree_files(tmp_path, cache) == tree_files(tmp_path) and list(cache) == ["b.md"]
