"""Finish the first legacy Kafka base/reference admission; never invokes a model."""

import argparse
import gzip
import hashlib
import json
import subprocess
import tarfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outcomes(workspace, classes):
    tests = {}
    for path in workspace.glob("**/build/test-results/test/TEST-*.xml"):
        for case in ET.parse(path).getroot().iter("testcase"):
            classname = case.get("classname", "")
            if not any(classname == name or classname.startswith(name + "$") for name in classes):
                continue
            identity = classname + "::" + case.get("name", "")
            if identity in tests:
                raise ValueError("duplicate test identity: " + identity)
            tests[identity] = (
                "failed"
                if case.find("failure") is not None or case.find("error") is not None
                else "skipped"
                if case.find("skipped") is not None
                else "passed"
            )
    return tests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.probe.resolve(), args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + 1500
    while time.monotonic() < deadline:
        probe = json.loads((root / "probe-repaired.json").read_text())
        if probe["status"] != "base_legacy_environment_probe_running":
            break
        time.sleep(5)
    else:
        raise TimeoutError("base admission probe did not finish")
    bundle = json.loads(
        Path("apps/eval/benchmarks/kafka-1000-v1/data/review-bundle.json").read_text()
    )
    task = next(t for t in bundle["tasks"] if t["id"] == probe["task_id"])
    classes = [t["class"] for t in task["tests"]]
    base = outcomes(Path(probe["workspace"]), classes)
    result = {
        "task_id": task["id"],
        "base_sha": task["base_sha"],
        "base": probe,
        "base_tests": base,
        "reference_tests": {},
        "status": "not_admitted",
        "execution_ready": False,
        "reason": "base did not produce a reproducible failing test",
        "reference": None,
        "test_patch_sha256": task["test_patch_sha256"],
        "reference_patch_sha256": task["reference_patch_sha256"],
    }
    if base and any(v == "failed" for v in base.values()) and probe["returncode"] == 1:
        reference = root / "reference-source"
        reference.mkdir(exist_ok=False)
        with tarfile.open(root / "base.tar.gz") as archive:
            archive.extractall(reference, filter="data")
        workspace = reference / ("kafka-" + task["base_sha"])
        for patch in (root / "visible.patch", root / "reference.patch"):
            subprocess.run(["git", "apply", "--check", str(patch)], cwd=workspace, check=True)
            subprocess.run(["git", "apply", str(patch)], cwd=workspace, check=True)
        cmd = [
            str(arg)
            .replace(probe["workspace"], str(workspace))
            .replace("harnext-kafka-admission-10040", "harnext-kafka-reference-10040")
            for arg in probe["argv"]
        ]
        cmd[2:2] = ["--network", "none"]
        cmd += ["--offline"]
        started = time.monotonic()
        try:
            with (root / "reference-build.log").open("w") as log:
                completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
            ref = outcomes(workspace, classes)
            result.update(
                reference={
                    "argv": cmd,
                    "returncode": completed.returncode,
                    "elapsed_seconds": time.monotonic() - started,
                },
                reference_tests=ref,
            )
            same = set(base) == set(ref)
            f2p = [k for k, value in base.items() if value == "failed"]
            p2p = [k for k, value in base.items() if value == "passed"]
            ok = completed.returncode == 0 and same and all(ref[k] == "passed" for k in f2p + p2p)
            result.update(
                execution_ready=False,
                status="base_reference_passed_requires_offline_base_replay"
                if ok
                else "not_admitted",
                reason="reference passed offline; base must still be replayed offline from a clean export before model trials"
                if ok
                else "reference run failed or test populations differ",
                fail_to_pass=f2p,
                pass_to_pass=p2p,
            )
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["docker", "rm", "-f", "harnext-kafka-reference-10040"], capture_output=True
            )
            result["reason"] = "reference test timeout"
    for path in root.glob("*.log"):
        (output / (path.name + ".gz")).write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    result["archive_sha256"] = digest(root / "base.tar.gz")
    (output / "admission.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "reason": result["reason"],
                "base_tests": len(base),
                "reference_tests": len(result["reference_tests"]),
            }
        )
    )


if __name__ == "__main__":
    main()
