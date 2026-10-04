"""On-demand, read-only checkouts of store checkpoints, shared by concurrent trials.

A git-tree store (`<stores>/<arm>/tree`, tag `ckpt-YYYY-MM` per checkpoint) is far too big
to materialise for every checkpoint up front (F0 alone: ~630k files per checkpoint, 42
checkpoints, 9 file arms — tens of GB per arm). A trial therefore extracts the checkpoint it
needs into `<root>/<arm>/<YYYY-MM>/` the first time it is asked for, later trials share it,
and the least recently used unleased checkouts beyond `keep_per_arm` are evicted.

Identity: the run pins the tag's TREE hash (`git rev-parse ckpt-X^{tree}`). It names the
exact content without hashing the files and changes whenever the store is rebuilt.

Locking (fcntl.flock; every call opens its own file description, so threads of one process
and separate processes are treated alike):
- `<arm>/.lock`, exclusive and short, serialises the check-then-lease step of `acquire`
  with eviction and the removal of stale checkouts.
- `<arm>/<ckpt>.lease` is held SHARED by every live lease; an evictor takes EX|NB and skips
  the checkout when that fails. A crashed holder loses its lease with its descriptor.
- `<arm>/<ckpt>.build` is held EXCLUSIVE during extraction: one extractor works, the others
  wait, then find the checkout complete.
Extraction goes into `<ckpt>.staging` and is renamed into place before the manifest is
written, so a manifest means a complete checkout; a stale staging directory (crash
mid-extract) is removed by the next extractor.

Non-git stores (R-hyb index dirs, Graphiti snapshots, S-tools sqlite) are already one small
directory per checkpoint: `acquire` returns `<stores>/<arm>/export/<ckpt>` after checking
its identity, no copy, no lease.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from harnext_eval.e2.benchmark import canonical, sha
from harnext_eval.e2.v2.stores.base import lock as read_only
from harnext_eval.e2.v2.stores.base import save_json
from harnext_eval.e2.v2.stores.base import unlock as writable

MANIFEST = ".manifest.json"
DEFAULT_ROOT = Path(__file__).resolve().parents[4] / "out/kafka-500-v2/checkouts"


def is_git_store(store_dir: Path) -> bool:
    return (store_dir / "tree" / ".git").exists()


def tag_hashes(store_dir: Path, checkpoint: str) -> tuple[str, str] | None:
    """(commit, tree) of `ckpt-<checkpoint>`; None when the tag does not exist."""
    tag = f"ckpt-{checkpoint}"
    done = subprocess.run(
        ["git", "-C", str(store_dir / "tree"), "rev-parse", f"{tag}^{{commit}}", f"{tag}^{{tree}}"],
        capture_output=True,
        text=True,
    )
    if done.returncode:
        return None
    commit, tree = done.stdout.split()
    return commit, tree


def tree_hash(directory: Path) -> str:
    rows = [
        (str(p.relative_to(directory)), sha(p.read_bytes()))
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    ]
    return sha(canonical(rows))


def identity(store_dir: Path, checkpoint: str, mode: str = "manifest") -> str:
    """The hash a run pins for (store, checkpoint). Git stores: the tag's tree hash. Others:
    the sha256 of the builder's export manifest (`manifest`) or of the export's files
    (`tree`, small stores and smoke runs). LookupError names what is missing."""
    if is_git_store(store_dir):
        hashes = tag_hashes(store_dir, checkpoint)
        if hashes is None:
            raise LookupError(f"{store_dir.name}: tag ckpt-{checkpoint} does not exist")
        return hashes[1]
    export = store_dir / "export" / checkpoint
    if not export.is_dir():
        raise LookupError(f"{store_dir.name}: export {export} is missing")
    if mode == "tree":
        return tree_hash(export)
    for candidate in (
        export.with_name(checkpoint + MANIFEST),
        export / "manifest.json",
        export / ".manifest.json",
        export / "meta.json",  # Graphiti snapshots carry their hash inside
    ):
        if candidate.exists():
            return sha(candidate.read_bytes())
    raise LookupError(f"{store_dir.name}: export {export} has no manifest to pin")


@dataclass
class Checkout:
    arm: str
    checkpoint: str
    path: Path
    identity: str
    lease: IO | None = None  # open, SH-locked lease file; None for non-git stores


class Checkouts:
    def __init__(self, root: Path = DEFAULT_ROOT, keep_per_arm: int = 3):
        self.root, self.keep_per_arm = root, keep_per_arm

    # -- paths ---------------------------------------------------------------------
    def target(self, arm: str, checkpoint: str) -> Path:
        return self.root / arm / checkpoint

    def manifest(self, arm: str, checkpoint: str) -> Path:
        return self.root / arm / (checkpoint + MANIFEST)

    @contextmanager
    def arm_lock(self, arm: str):
        (self.root / arm).mkdir(parents=True, exist_ok=True)
        with (self.root / arm / ".lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    # -- leases --------------------------------------------------------------------
    def acquire(
        self,
        arm: str,
        store_dir: Path,
        checkpoint: str,
        *,
        expect: str | None = None,
        mode: str = "manifest",
    ) -> Checkout:
        """A read-only directory holding `checkpoint` of the store, leased until `release`.
        `expect` is the identity the run pinned; a store that changed since is refused."""
        current = identity(store_dir, checkpoint, mode)
        if expect is not None and current != expect:
            raise ValueError(
                f"{arm}/{checkpoint}: store identity {current} differs from the pinned {expect}"
            )
        if not is_git_store(store_dir):
            return Checkout(arm, checkpoint, store_dir / "export" / checkpoint, current)
        target, manifest = self.target(arm, checkpoint), self.manifest(arm, checkpoint)
        with self.arm_lock(arm):
            if manifest.exists() and json.loads(manifest.read_text())["tree"] != current:
                if not self.remove(arm, checkpoint):
                    raise ValueError(
                        f"{arm}/{checkpoint}: checkout holds another tree and is leased by "
                        "another run; wait for it or delete it"
                    )
            lease = target.with_name(checkpoint + ".lease").open("a")
            fcntl.flock(lease, fcntl.LOCK_SH)
            complete = manifest.exists()
        if not complete:
            with target.with_name(checkpoint + ".build").open("a") as build:
                fcntl.flock(build, fcntl.LOCK_EX)
                if not manifest.exists():
                    self.extract(store_dir, checkpoint, target, manifest)
        os.utime(manifest)  # LRU clock
        self.evict(arm)
        return Checkout(arm, checkpoint, target, current, lease)

    def release(self, checkout: Checkout) -> None:
        if checkout.lease is not None:
            fcntl.flock(checkout.lease, fcntl.LOCK_UN)
            checkout.lease.close()
            checkout.lease = None
            self.evict(checkout.arm)

    # -- materialisation -----------------------------------------------------------
    def extract(self, store_dir: Path, checkpoint: str, target: Path, manifest: Path) -> None:
        """`git archive ckpt-X | tar -x` into a staging sibling, made read-only, renamed into
        place, then the manifest (its presence is the completion mark). Caller holds `.build`."""
        hashes = tag_hashes(store_dir, checkpoint)
        if hashes is None:
            raise LookupError(f"{store_dir.name}: tag ckpt-{checkpoint} does not exist")
        commit, tree = hashes
        staging = target.with_name(checkpoint + ".staging")
        for debris in (staging, target):  # crash mid-extract, or before the manifest was written
            if debris.exists():
                writable(debris)
                shutil.rmtree(debris)
        staging.mkdir(parents=True)
        archive = subprocess.Popen(
            ["git", "-C", str(store_dir / "tree"), "archive", f"ckpt-{checkpoint}"],
            stdout=subprocess.PIPE,
        )
        subprocess.run(["tar", "-x", "-C", str(staging)], stdin=archive.stdout, check=True)
        if archive.wait():
            raise RuntimeError(f"git archive ckpt-{checkpoint} failed in {store_dir}")
        read_only(staging)
        os.sync()  # the extracted files reach the disk before the manifest marks them complete
        os.rename(staging, target)
        save_json(
            manifest,
            {"arm": store_dir.name, "checkpoint": checkpoint, "commit": commit, "tree": tree},
        )

    def remove(self, arm: str, checkpoint: str) -> bool:
        """Delete one checkout unless a lease is held on it. Caller holds the arm lock."""
        target, manifest = self.target(arm, checkpoint), self.manifest(arm, checkpoint)
        with target.with_name(checkpoint + ".lease").open("a") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return False
            manifest.unlink(missing_ok=True)
            if target.exists():
                writable(target)
                shutil.rmtree(target)
        return True

    def evict(self, arm: str) -> list[str]:
        """Drop unleased checkouts beyond `keep_per_arm`, least recently acquired first."""
        evicted = []
        with self.arm_lock(arm):
            manifests = sorted(
                (self.root / arm).glob(f"*{MANIFEST}"), key=lambda p: p.stat().st_mtime_ns
            )
            excess = len(manifests) - self.keep_per_arm
            for manifest in manifests:
                if excess <= 0:
                    break
                checkpoint = manifest.name[: -len(MANIFEST)]
                if self.remove(arm, checkpoint):  # a leased one is skipped for the next-oldest
                    evicted.append(checkpoint)
                    excess -= 1
        return evicted

    def present(self, arm: str) -> list[str]:
        return sorted(p.name[: -len(MANIFEST)] for p in (self.root / arm).glob(f"*{MANIFEST}"))
