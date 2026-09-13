"""Disposable, network-disabled execution of development task test suites."""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath

IMAGE = "python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea"
RUNNER = """import json, sys, unittest
sys.path.insert(0, "/work")
class Result(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outcomes = {}
    def addSuccess(self, test):
        super().addSuccess(test)
        self.outcomes[test.id()] = "passed"
    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.outcomes[test.id()] = "failed"
    def addError(self, test, err):
        super().addError(test, err)
        self.outcomes[test.id()] = "error"
suite = unittest.defaultTestLoader.discover("/grader", pattern="test_*.py")
result = unittest.TextTestRunner(verbosity=2, resultclass=Result).run(suite)
print(json.dumps({"tests": result.outcomes, "count": result.testsRun,
                  "passed": result.wasSuccessful()}), flush=True)
sys.exit(0 if result.wasSuccessful() else 1)
"""


def safe_path(name: str) -> str:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or any(part in {"..", ".git"} for part in path.parts):
        raise ValueError("unsafe repository path")
    return str(path)


def resolve_image(image: str = IMAGE) -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        check=True,
    )
    digest = result.stdout.strip()
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise RuntimeError("test image did not resolve to an immutable ID")
    return digest


def run_tests(
    files: dict[str, str], visible: str, hidden: str | None, *, image: str, timeout_s: int = 30
) -> dict:
    if not image.startswith("sha256:"):
        raise ValueError("test execution requires a resolved immutable image ID")
    name = f"harnext-e2-test-{uuid.uuid4().hex}"
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="harnext-e2-test-") as directory:
        root = Path(directory)
        work, grader = root / "work", root / "grader"
        work.mkdir()
        grader.mkdir()
        for path, content in files.items():
            target = work / safe_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        (grader / "run.py").write_text(RUNNER)
        (grader / "test_visible.py").write_text(visible)
        if hidden is not None:
            (grader / "test_hidden.py").write_text(hidden)
        cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65534:65534",
            "--pids-limit",
            "64",
            "--memory",
            "256m",
            "--cpus",
            "1",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--mount",
            f"type=bind,source={work},target=/work,readonly",
            "--mount",
            f"type=bind,source={grader},target=/grader,readonly",
            "--workdir",
            "/work",
            "--entrypoint",
            "python",
            image,
            "-I",
            "-B",
            "/grader/run.py",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
            # Bound retained logs. The fixture limits task runtime and container
            # memory; a production grader also needs streaming output limits.
            lines = proc.stdout.splitlines()
            try:
                result = json.loads(lines[-1]) if lines else {}
            except json.JSONDecodeError:
                result = {}
            valid = (
                proc.returncode in {0, 1}
                and isinstance(result.get("tests"), dict)
                and result.get("count", 0) > 0
                and len(result["tests"]) == result["count"]
                and result.get("passed") == (proc.returncode == 0)
            )
            return {
                "status": "completed" if valid else "infrastructure_error",
                "passed": bool(valid and result["passed"]),
                "tests": result.get("tests", {}),
                "exit_code": proc.returncode,
                "stdout": proc.stdout[-16000:],
                "stderr": proc.stderr[-16000:],
                "latency_s": time.monotonic() - started,
                "image_id": image,
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "passed": False,
                "tests": {},
                "latency_s": time.monotonic() - started,
                "image_id": image,
            }
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=10)
