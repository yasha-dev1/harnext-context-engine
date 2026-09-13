"""Validate a stopped benchmark before resuming it on another machine."""

from pathlib import Path

from harnext_eval.e2.benchmark import read, sha


def portable_manifest(manifest):
    """Relocate only the five known repository paths; preserve every other field."""
    import copy

    result = copy.deepcopy(manifest)
    root = Path(manifest["protocol"]["pilot"]).parents[3]
    for section, names in {
        "protocol": ("pilot", "output", "scratch"),
        "pilot": ("benchmark", "engine_profile"),
    }.items():
        for name in names:
            result[section][name] = str(Path(result[section][name]).relative_to(root))
    return result


def verify_files(root, hashes):
    for relative, expected in hashes.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"checkpoint path escapes root: {relative}")
        if not path.is_file() or sha(path.read_bytes()) != expected:
            raise ValueError(f"checkpoint file missing or changed: {relative}")


def validate_checkpoint(output, repo):
    checkpoint = read(output / "checkpoint.json")
    verify_files(repo, checkpoint["implementation_files_sha256"])
    verify_files(output, checkpoint["frozen_files_sha256"])
    for archive in sorted(output.glob("cases/*/*/archive-manifest.json")):
        verify_files(archive.parent, read(archive)["files_sha256"])
    for archive in sorted(output.glob("interruptions/*/*/archive-manifest.json")):
        verify_files(archive.parent, read(archive)["files_sha256"])
    completed = set(checkpoint["completed_reports"])
    observed = {str(p.relative_to(output)) for p in output.glob("cases/*/*/report.json")}
    if not completed <= observed:
        raise ValueError("completed checkpoint trials are missing")
    for relative in observed:
        report = output / relative
        if not (report.parent / "archive-manifest.json").is_file():
            raise ValueError(f"unverified trial report: {relative}")
    return checkpoint
