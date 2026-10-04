import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from harnext_eval.e2.v2 import checkouts as co


def git(tree, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(tree), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()  # fmt: skip


def commit(tree, files, tag):
    for name, text in files.items():
        (tree / name).parent.mkdir(parents=True, exist_ok=True)
        (tree / name).write_text(text)
    git(tree, "add", "-A")
    git(tree, "commit", "-qm", tag)
    git(tree, "tag", "-f", tag)


@pytest.fixture
def store(tmp_path):
    """A git store with tags ckpt-2023-01 (one file) and ckpt-2023-02 (two files)."""
    tree = tmp_path / "stores/F9/tree"
    tree.mkdir(parents=True)
    git(tree, "init", "-q", "-b", "main")
    commit(tree, {"events/a.md": "a\n"}, "ckpt-2023-01")
    commit(tree, {"events/b.md": "b\n"}, "ckpt-2023-02")
    return tree.parent


@pytest.fixture
def checkouts(tmp_path):
    return co.Checkouts(tmp_path / "checkouts", keep_per_arm=2)


def test_identity_is_the_tag_tree_hash_and_missing_tags_are_named(store):
    tree = git(store / "tree", "rev-parse", "ckpt-2023-01^{tree}")
    assert co.identity(store, "2023-01") == tree
    assert co.identity(store, "2023-02") != tree
    with pytest.raises(LookupError, match="ckpt-2024-01 does not exist"):
        co.identity(store, "2024-01")


def test_acquire_extracts_once_read_only_and_is_shared(store, checkouts, monkeypatch):
    extractions = []
    real, go = checkouts.extract, threading.Event()

    def slow(*args, **kwargs):
        extractions.append(args[1])
        assert go.wait(timeout=5)  # hold .build so the second acquirer must wait
        real(*args, **kwargs)

    monkeypatch.setattr(checkouts, "extract", slow)
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(checkouts.acquire, "F9", store, "2023-01") for _ in range(2)]
        assert not any(f.done() for f in futures) and len(extractions) <= 1
        go.set()
        a, b = (f.result(timeout=10) for f in futures)
    assert a.path == b.path == checkouts.root / "F9/2023-01"
    assert extractions == ["2023-01"], "the second acquirer waited on .build, then reused"
    assert (a.path / "events/a.md").read_text() == "a\n" and not (a.path / "events/b.md").exists()
    assert a.identity == git(store / "tree", "rev-parse", "ckpt-2023-01^{tree}")
    manifest = json.loads((checkouts.root / "F9/2023-01.manifest.json").read_text())
    assert manifest["tree"] == a.identity and manifest["commit"] == git(store / "tree", "rev-parse", "ckpt-2023-01")  # fmt: skip
    assert not os.access(a.path / "events/a.md", os.W_OK) and not os.access(a.path, os.W_OK)
    with pytest.raises(PermissionError):
        (a.path / "events/c.md").write_text("no\n")
    checkouts.release(a)
    checkouts.release(b)


def test_second_acquire_reuses_without_extracting(store, checkouts, monkeypatch):
    first = checkouts.acquire("F9", store, "2023-02")
    checkouts.release(first)
    monkeypatch.setattr(checkouts, "extract", lambda *a, **k: pytest.fail("re-extracted"))
    again = checkouts.acquire("F9", store, "2023-02")
    assert again.path == first.path and (again.path / "events/b.md").exists()
    checkouts.release(again)


def test_eviction_respects_leases_and_keep_per_arm(tmp_path, checkouts):
    tags = ["2023-01", "2023-02", "2023-03", "2023-04"]
    tree = tmp_path / "stores/F8/tree"
    tree.mkdir(parents=True)
    git(tree, "init", "-q", "-b", "main")
    for tag in tags:
        commit(tree, {f"{tag}.md": tag + "\n"}, f"ckpt-{tag}")
    store = tree.parent
    held = checkouts.acquire("F8", store, "2023-01")  # leased: never evicted
    checkouts.release(checkouts.acquire("F8", store, "2023-02"))
    assert checkouts.present("F8") == ["2023-01", "2023-02"]
    checkouts.release(checkouts.acquire("F8", store, "2023-03"))
    # total bounded at keep=2: 01 is leased, so the LRU unleased one (02) goes instead
    assert checkouts.present("F8") == ["2023-01", "2023-03"]
    assert not (checkouts.root / "F8/2023-02").exists()
    kept = checkouts.acquire("F8", store, "2023-03")  # refresh 03, lease it too
    fourth = checkouts.acquire("F8", store, "2023-04")
    assert checkouts.present("F8") == ["2023-01", "2023-03", "2023-04"], "all leased: none evicted"  # fmt: skip
    checkouts.release(fourth)
    assert checkouts.present("F8") == ["2023-01", "2023-03"], "the only unleased one goes"
    checkouts.release(held)
    checkouts.release(checkouts.acquire("F8", store, "2023-04"))
    assert checkouts.present("F8") == ["2023-03", "2023-04"], "01 evicted once its lease ended"
    checkouts.release(kept)


def test_stale_staging_is_removed_and_crash_leaves_no_partial_checkout(store, checkouts, monkeypatch):  # fmt: skip
    arm_dir = checkouts.root / "F9"
    arm_dir.mkdir(parents=True)
    stale = arm_dir / "2023-01.staging/events"
    stale.mkdir(parents=True)
    (stale / "junk.md").write_text("half\n")
    os.chmod(stale, 0o555)  # a crash after chmod, before the rename
    real = subprocess.run

    def crash(argv, **kwargs):
        if argv[0] == "tar":
            raise RuntimeError("disk full")
        return real(argv, **kwargs)

    monkeypatch.setattr(co.subprocess, "run", crash)
    with pytest.raises(RuntimeError, match="disk full"):
        checkouts.acquire("F9", store, "2023-01")
    assert not (arm_dir / "2023-01").exists() and not (arm_dir / "2023-01.manifest.json").exists()
    monkeypatch.setattr(co.subprocess, "run", real)
    got = checkouts.acquire("F9", store, "2023-01")
    assert not (arm_dir / "2023-01.staging").exists()
    assert sorted(p.name for p in (got.path / "events").iterdir()) == ["a.md"]
    checkouts.release(got)


def test_moved_tag_replaces_an_unleased_checkout(store, checkouts):
    old = checkouts.acquire("F9", store, "2023-02")
    checkouts.release(old)
    commit(store / "tree", {"events/c.md": "c\n"}, "ckpt-2023-02")  # store rebuilt, tag moved
    new = checkouts.acquire("F9", store, "2023-02")
    assert new.identity != old.identity and (new.path / "events/c.md").exists()
    with pytest.raises(ValueError, match="differs from the pinned"):
        checkouts.acquire("F9", store, "2023-02", expect=old.identity)
    checkouts.release(new)


def test_non_git_store_returns_the_export_in_place(tmp_path, checkouts):
    export = tmp_path / "stores/R-hyb/export/2023-01"
    export.mkdir(parents=True)
    (export / "chunks.jsonl").write_text("{}\n")
    export.with_name("2023-01.manifest.json").write_text('{"content_sha256": "x"}')
    got = checkouts.acquire("R-hyb", tmp_path / "stores/R-hyb", "2023-01")
    assert got.path == export and got.lease is None
    assert got.identity == co.identity(tmp_path / "stores/R-hyb", "2023-01")
    assert co.identity(tmp_path / "stores/R-hyb", "2023-01", "tree") != got.identity
    with pytest.raises(LookupError, match="is missing"):
        co.identity(tmp_path / "stores/R-hyb", "2023-02")
    checkouts.release(got)
    assert not (checkouts.root / "R-hyb").exists()
