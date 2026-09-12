"""Build a standalone result viewer through apply_patch; no model calls."""

import argparse
import base64
import gzip
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / "report.json").read_text())
    if not report["finished"]:
        raise ValueError("do not render an incomplete pilot as final")
    details = []
    for index, row in enumerate(report["results"], 1):
        folder = args.run / f"{index:02}-{row['family']}"
        logs = list((folder / "store/trials").glob("*.jsonl"))
        log = next(p for p in logs if not p.stem.endswith("capability"))
        calls = [json.loads(line) for line in log.read_text().splitlines()][1:]
        details.append(
            {
                "prompt": (folder / "reader/prompt.txt").read_text(),
                "calls": calls,
                "snapshot_bytes": (folder / "store/snapshot.json").stat().st_size,
            }
        )
    report["details"] = details
    encoded = base64.b64encode(
        gzip.compress(json.dumps(report, ensure_ascii=False).encode(), mtime=0)
    ).decode()
    template = Path(__file__).with_name("benchmark_pilot_report.template.html").read_text()
    html = template.replace("__PILOT_DATA__", encoded)
    path = args.out.resolve()
    if path.exists():
        raise ValueError("use a new report path")
    patch = (
        f"*** Begin Patch\n*** Add File: {path}\n"
        + "\n".join("+" + line for line in html.splitlines())
        + "\n*** End Patch\n"
    )
    subprocess.run(["apply_patch"], input=patch, text=True, capture_output=True, check=True)
    print(path)


if __name__ == "__main__":
    main()
