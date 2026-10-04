"""Coding workspaces with real, pruned git history (E2-V2-DESIGN.md "Genuine baseline").

`task_repo` builds once per coding task a repository holding exactly the ancestors of the
task's base commit: `git init` + `git fetch <bare clone> <base_sha>` (nothing else is ever
asked of the clone), no remote, no tags, reflog expired, `gc --prune=now`. The result is
verified object by object and the evidence is written to `repo-manifest.json`; a repository
that fails a check is deleted and the call raises.

`stage_workspace` materializes the per-trial copy. Pack files are HARDLINKED, everything
else in `.git` is copied, the working tree is checked out fresh:
  - git never rewrites an object or pack file in place (packs are written under a temporary
    name, renamed, and chmod 0444); `git gc`/`repack`/`prune` in a trial create new files and
    unlink old names, and unlinking one name of a hardlinked inode leaves the other names
    intact. So nothing an agent does *through git* can reach the task repo.
  - An agent runs under the same uid, so it could still `chmod +w` and overwrite a pack
    byte-for-byte on purpose. That is caught, not prevented: the manifest pins the sha256 of
    every object file and `stage_workspace` re-verifies it before each trial (a mismatch
    rebuilds the task repo). `config`, `HEAD`, refs, index and hooks are small mutable files
    that tools do edit in place, hence copied, never linked.
The grader does not trust the trial's `.git` at all (see grade.workspace_patch).

Every directory under apps/eval/out lies inside this project's own git repository, so each
git call here pins --git-dir/--work-tree and GIT_CEILING_DIRECTORIES.
"""

from __future__ import annotations

import hashlib
import fcntl
import os
import shutil
import subprocess
from pathlib import Path

from harnext_eval.e2 import kafka_coding as kc
from harnext_eval.e2.benchmark import ROOT, read, save
from harnext_eval.e2.v2.errors import PauseRun

WORKSPACES = ROOT / "out/kafka-500-v2/workspaces"
BRANCH = "task"
QUIET = ["-c", "gc.auto=0", "-c", "maintenance.auto=false", "-c", "advice.detachedHead=false"]
BOUNDED = ["-c", "pack.threads=2", "-c", "pack.windowMemory=256m"]  # the machine is short on RAM


class RepoCheckFailed(ValueError):  # noqa: N818
    """The built repository holds something it must not (or lacks the base commit)."""


def run_git(repo, *args, check=True, stdin=None):
    """git pinned to `repo` (a non-bare repository directory); user/system config ignored."""
    env = {
        **os.environ,
        "GIT_CEILING_DIRECTORIES": str(Path(repo).resolve().parent),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY"):
        env.pop(name, None)
    argv = ["git", "--git-dir", str(Path(repo) / ".git"), "--work-tree", str(repo), *QUIET, *args]
    done = subprocess.run(argv, cwd=repo, env=env, input=stdin, capture_output=True, text=True)
    if check and done.returncode:
        raise PauseRun(f"git {' '.join(args[:2])} failed in {repo}: {done.stderr.strip()[:500]}")
    return done


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def object_files(repo):
    objects = Path(repo) / ".git/objects"
    return {str(p.relative_to(objects)): file_sha(p) for p in sorted(objects.rglob("*")) if p.is_file()}  # fmt: skip


def verify(repo, base_sha, reference_commit=None):
    """The evidence dict for repo-manifest.json; raises RepoCheckFailed on any violation."""
    listing = run_git(repo, "cat-file", "--batch-all-objects", "--batch-check").stdout.split("\n")
    rows = [line.split() for line in listing if line]
    kinds = {kind: sum(1 for r in rows if r[1] == kind) for kind in sorted({r[1] for r in rows})}
    commits = {r[0] for r in rows if r[1] == "commit"}
    dated = run_git(
        repo, "log", "--no-walk=unsorted", "--stdin", "--format=%ct %H %cI",
        stdin="".join(f"{c}\n" for c in sorted(commits)),
    ).stdout.splitlines()  # fmt: skip
    newest = max((line.split() for line in dated), key=lambda r: int(r[0]), default=None)
    base = run_git(repo, "log", "-1", "--format=%ct %H %cI", base_sha, check=False).stdout.split()
    ancestors = set(run_git(repo, "rev-list", base_sha, check=False).stdout.split())
    refs = run_git(repo, "for-each-ref", "--format=%(refname) %(objectname)").stdout.splitlines()
    logs = Path(repo) / ".git/logs"
    reflog = [
        line.split()[1]
        for path in (logs.rglob("*") if logs.exists() else ())
        if path.is_file()
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    counted = dict(
        line.split(": ") for line in run_git(repo, "count-objects", "-v").stdout.splitlines()
    )
    golden_present = bool(reference_commit) and (
        run_git(repo, "cat-file", "-e", reference_commit, check=False).returncode == 0
    )
    checks = {
        "base_commit_present": len(base) == 3 and base[1] == base_sha,
        "only_commits_trees_blobs": set(kinds) <= {"commit", "tree", "blob"},
        "every_commit_is_an_ancestor_of_base": bool(commits) and commits == ancestors,
        "newest_committer_date_le_base": newest is not None and bool(base) and int(newest[0]) <= int(base[0]),
        "reference_commit_absent": not golden_present,
        "no_remotes": not run_git(repo, "remote").stdout.strip()
        and not run_git(repo, "config", "--get-regexp", r"^(remote|branch)\.", check=False).stdout.strip(),
        "only_ref_is_task_branch_at_base": refs == [f"refs/heads/{BRANCH} {base_sha}"],
        "reflog_points_nowhere_else": set(reflog) <= {base_sha},
        "no_fetch_head_or_alternates": not any(
            (Path(repo) / ".git" / name).exists()
            for name in ("FETCH_HEAD", "ORIG_HEAD", "objects/info/alternates", "shallow")
        ),
        "nothing_loose_or_garbage": counted.get("count") == "0" and counted.get("garbage") == "0",
    }  # fmt: skip
    evidence = {
        "base_sha": base_sha,
        "base_committer_date": base[2] if len(base) == 3 else None,
        "newest_commit": {"sha": newest[1], "committer_date": newest[2]} if newest else None,
        "reference_commit": reference_commit,
        "objects": kinds,
        "objects_total": len(rows),
        "commits_reachable_from_base": len(ancestors),
        "packs": int(counted.get("packs", 0)),
        "size_pack_kib": int(counted.get("size-pack", 0)),
        "refs": refs,
        "reflog_entries": len(reflog),
        "checks": checks,
    }
    if failed := sorted(name for name, ok in checks.items() if not ok):
        raise RepoCheckFailed(f"{repo}: {', '.join(failed)} ({evidence['newest_commit']})")
    return evidence


def build_repo(base_sha, repo, source=None):
    """init + fetch-by-sha + branch; leaves no remote, no tags, no reflog, no unreachable object."""
    source = Path(source or kc.REPO).resolve()
    repo.mkdir(parents=True)
    run_git(repo, "init", "-q", "-b", BRANCH)
    # Local transports spawn upload-pack as a child: the -c reaches it through the environment,
    # so the bare clone's own config is never touched.
    run_git(
        repo, "-c", "uploadpack.allowAnySHA1InWant=true", "-c", "fetch.writeCommitGraph=false",
        *BOUNDED, "fetch", "-q", "--no-tags", "--no-recurse-submodules", f"file://{source}", base_sha,
    )  # fmt: skip
    run_git(repo, "checkout", "-q", "-B", BRANCH, base_sha)
    for name in ("FETCH_HEAD", "ORIG_HEAD"):
        (repo / ".git" / name).unlink(missing_ok=True)
    run_git(repo, "reflog", "expire", "--expire=now", "--expire-unreachable=now", "--all")
    shutil.rmtree(repo / ".git/logs", ignore_errors=True)
    run_git(repo, *BOUNDED, "gc", "-q", "--prune=now")
    shutil.rmtree(repo / ".git/hooks", ignore_errors=True)  # samples only; nothing runs in trials


def task_repo(task, reference_commit=None, *, root=None, source=None) -> Path:
    """The verified per-task repository (built once; later calls return the cached one).
    Serialised per task: several arms of one task may stage at the same time."""
    home = Path(root or WORKSPACES) / task["id"]
    home.mkdir(parents=True, exist_ok=True)
    with (home / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _task_repo(task, home, reference_commit, source)


def _task_repo(task, home, reference_commit, source) -> Path:
    repo, manifest = home / "repo", home / "repo-manifest.json"
    base_sha = task["base_sha"]
    reference_commit = reference_commit or task.get("reference_commit")
    if repo.exists() and manifest.exists() and read(manifest).get("base_sha") == base_sha:
        return repo
    shutil.rmtree(repo, ignore_errors=True)
    staging = home / "repo.building"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        build_repo(base_sha, staging, source)
        evidence = verify(staging, base_sha, reference_commit)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    staging.replace(repo)
    save(manifest, evidence | {"task_id": task["id"], "object_files_sha256": object_files(repo)})
    return repo


def intact(repo, manifest):
    return manifest.exists() and object_files(repo) == read(manifest).get("object_files_sha256")


def stage_workspace(task, dest, reference_commit=None, *, root=None, source=None) -> Path:
    """Per-trial workspace: hardlinked packs, copied git metadata, fresh checkout at base."""
    home = Path(root or WORKSPACES) / task["id"]
    home.mkdir(parents=True, exist_ok=True)
    with (home / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        repo = _task_repo(task, home, reference_commit, source)
        if not intact(repo, repo.parent / "repo-manifest.json"):  # a trial overwrote a shared pack
            shutil.rmtree(repo)
            repo = _task_repo(task, home, reference_commit, source)
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError(f"workspace is not empty: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    objects = repo / ".git/objects"

    def place(src, dst):
        if Path(src).is_relative_to(objects):
            try:
                os.link(src, dst)
                return
            except OSError:
                pass  # other filesystem: fall through to a real copy
        shutil.copy2(src, dst)

    shutil.copytree(repo / ".git", dest / ".git", copy_function=place, symlinks=True)
    run_git(dest, "reset", "-q", "--hard", task["base_sha"])
    if run_git(dest, "status", "--porcelain").stdout.strip():
        raise PauseRun(f"staged workspace is not clean at {task['base_sha']}: {dest}")
    return dest


def discard(task_id, *, root=None):
    """Free the task repo once every arm of the task is recorded; the manifest stays."""
    shutil.rmtree(Path(root or WORKSPACES) / task_id / "repo", ignore_errors=True)
