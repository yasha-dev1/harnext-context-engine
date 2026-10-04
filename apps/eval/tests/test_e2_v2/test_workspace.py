import json
import os
import shutil
import subprocess

import pytest
from harnext_eval.e2.v2 import grade
from harnext_eval.e2.v2 import workspace as ws

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
MAIN = "core/src/main/java/A.java"


def sh(*args, cwd, date=None):
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(cwd.parent)}
    if date:
        env |= {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    return subprocess.run(args, cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def clone(tmp_path):
    """Bare clone with history old -> base -> FUTURE (the golden commit) and a future tag."""
    src = tmp_path / "src"
    (src / MAIN).parent.mkdir(parents=True)
    sh(*GIT, "init", "-q", "-b", "trunk", cwd=src)
    shas = []
    for text, date in (("v1", "2023-01-01T00:00:00Z"), ("v2", "2023-02-01T00:00:00Z"), ("GOLDEN-FUTURE-FIX", "2023-03-01T00:00:00Z")):  # fmt: skip
        (src / MAIN).write_text(f"class A {{ /* {text} */ }}\n")
        (src / ".gitignore").write_text("build/\n")
        sh(*GIT, "add", "-A", cwd=src)
        sh(*GIT, "commit", "-qm", f"commit {text}", cwd=src, date=date)
        shas.append(sh("git", "rev-parse", "HEAD", cwd=src).strip())
    sh(*GIT, "tag", "-a", "release", "-m", "future tag", cwd=src, date="2023-03-02T00:00:00Z")
    bare = tmp_path / "kafka.git"
    sh("git", "clone", "-q", "--bare", str(src), str(bare), cwd=tmp_path)
    old, base, future = shas
    return {"bare": bare, "task": {"id": "C-1", "base_sha": base}, "old": old, "future": future, "root": tmp_path / "workspaces"}  # fmt: skip


def test_task_repo_holds_exactly_the_ancestors_of_base(clone):
    repo = ws.task_repo(clone["task"], clone["future"], root=clone["root"], source=clone["bare"])
    has = lambda sha: ws.run_git(repo, "cat-file", "-e", sha, check=False).returncode == 0  # noqa: E731
    assert has(clone["task"]["base_sha"]) and has(clone["old"]) and not has(clone["future"])
    assert "v2" in (repo / MAIN).read_text()
    log = ws.run_git(repo, "log", "--format=%s").stdout.split("\n")
    assert log[:2] == ["commit v2", "commit v1"]
    assert not ws.run_git(repo, "remote").stdout and not ws.run_git(repo, "tag").stdout
    packed = b"".join(p.read_bytes() for p in (repo / ".git").rglob("*") if p.is_file())
    assert str(clone["bare"]).encode() not in packed  # not even the clone's path stays behind
    manifest = json.loads((repo.parent / "repo-manifest.json").read_text())
    assert all(manifest["checks"].values()) and manifest["objects"]["commit"] == 2
    assert manifest["newest_commit"]["sha"] == clone["task"]["base_sha"]
    assert manifest["reference_commit"] == clone["future"] and manifest["reflog_entries"] == 0
    assert "allowanysha1inwant" not in (clone["bare"] / "config").read_text().lower()
    assert ws.task_repo(clone["task"], root=clone["root"], source=clone["bare"]) == repo  # cached


def test_verification_catches_a_newer_object(clone):
    repo = ws.task_repo(clone["task"], clone["future"], root=clone["root"], source=clone["bare"])
    ws.run_git(
        repo, "fetch", "-q", "--no-tags", f"file://{clone['bare']}", "trunk"
    )  # the future arrives unreferenced
    with pytest.raises(ws.RepoCheckFailed) as caught:
        ws.verify(repo, clone["task"]["base_sha"], clone["future"])
    for check in ("every_commit_is_an_ancestor_of_base", "newest_committer_date_le_base", "reference_commit_absent"):  # fmt: skip
        assert check in str(caught.value)


def test_a_clone_that_would_leak_is_never_published(clone, monkeypatch):
    monkeypatch.setattr(ws, "build_repo", lambda base, repo, source=None: shutil.copytree(clone["bare"].parent / "src", repo))  # fmt: skip
    with pytest.raises(ws.RepoCheckFailed):
        ws.task_repo(clone["task"], clone["future"], root=clone["root"], source=clone["bare"])
    assert not (clone["root"] / "C-1/repo").exists()
    assert not (clone["root"] / "C-1/repo-manifest.json").exists()


def test_staged_workspace_is_independent_of_the_task_repo(clone, tmp_path):
    options = {"root": clone["root"], "source": clone["bare"]}
    repo = ws.task_repo(clone["task"], clone["future"], **options)
    before = ws.object_files(repo)
    trial = ws.stage_workspace(clone["task"], tmp_path / "trial/workspace", **options)
    pack = next((trial / ".git/objects/pack").glob("*.pack"))
    assert pack.stat().st_ino == (repo / ".git/objects/pack" / pack.name).stat().st_ino  # linked
    assert (trial / ".git/config").stat().st_ino != (repo / ".git/config").stat().st_ino  # copied
    (trial / MAIN).write_text("class A { /* agent */ }\n")
    (trial / ".git/config").open("a").write("[user]\n\tname = agent\n")
    sh(*GIT, "commit", "-qam", "agent work", cwd=trial)
    sh(
        *GIT, "gc", "-q", "--aggressive", "--prune=now", cwd=trial
    )  # repacks: old pack unlinked here
    assert not (trial / ".git/objects/pack" / pack.name).exists()
    assert ws.object_files(repo) == before and "agent" not in (repo / ".git/config").read_text()
    assert ws.run_git(repo, "log", "--format=%s").stdout.split("\n")[0] == "commit v2"
    assert ws.run_git(repo, "status", "--porcelain").stdout == ""
    other = ws.stage_workspace(clone["task"], tmp_path / "trial2/workspace", **options)
    assert "v2" in (other / MAIN).read_text() and "agent" not in (other / MAIN).read_text()


def test_a_pack_overwritten_in_place_is_detected_and_rebuilt(clone, tmp_path):
    options = {"root": clone["root"], "source": clone["bare"]}
    trial = ws.stage_workspace(clone["task"], tmp_path / "t1", **options)
    pack = next((trial / ".git/objects/pack").glob("*.pack"))
    pack.chmod(0o644)
    pack.write_bytes(b"garbage")  # same inode as the task repo's pack: deliberate sabotage
    again = ws.stage_workspace(clone["task"], tmp_path / "t2", **options)
    assert ws.run_git(again, "log", "--format=%s").stdout.split("\n")[0] == "commit v2"
    assert ws.intact(clone["root"] / "C-1/repo", clone["root"] / "C-1/repo-manifest.json")


def test_patch_is_the_tree_against_base_whatever_the_agent_did_to_its_git(clone, tmp_path):
    options = {"root": clone["root"], "source": clone["bare"]}
    trial = ws.stage_workspace(clone["task"], tmp_path / "trial", **options)
    (trial / MAIN).write_text("class A { /* fixed */ }\n")
    (trial / "core/src/main/java/B.java").write_text("class B {}\n")  # new, untracked
    (trial / "build").mkdir()
    (trial / "build/out.class").write_text("ignored build output\n")
    sh(*GIT, "add", MAIN, cwd=trial)
    sh(*GIT, "commit", "-qm", "agent commit", cwd=trial)  # committed or not makes no difference
    (trial / ".git/info").mkdir(exist_ok=True)
    (trial / ".git/info/exclude").write_text("B.java\n")  # the trial's git config is not trusted
    patch = grade.workspace_patch(clone["task"]["base_sha"], trial, tmp_path / "diff", clone["bare"])  # fmt: skip
    assert [p for p, _ in grade.split_patch(patch)] == [MAIN, "core/src/main/java/B.java"]
    assert "fixed" in patch and ".git/" not in patch and "build/" not in patch


def test_arms_of_one_task_can_stage_at_the_same_time(clone, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    options = {"root": clone["root"], "source": clone["bare"]}
    with ThreadPoolExecutor(6) as pool:
        staged = list(pool.map(lambda n: ws.stage_workspace(clone["task"], tmp_path / f"arm{n}/workspace", **options), range(6)))  # fmt: skip
    for dest in staged:
        head = ws.run_git(dest, "rev-parse", "HEAD").stdout.strip()
        assert head == clone["task"]["base_sha"] and not ws.run_git(dest, "status", "--porcelain").stdout.strip()
