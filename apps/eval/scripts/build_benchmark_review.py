"""Embed the frozen review bundle in a standalone HTML SPA using apply_patch."""

import base64
import gzip
import subprocess

from harnext_eval.e2.benchmark import BENCH


def main():
    source = BENCH / "data/review-bundle.json"
    encoded = base64.b64encode(gzip.compress(source.read_bytes(), mtime=0)).decode()
    template = (BENCH / "review.template.html").read_text()
    marker = "__BENCHMARK_GZIP_BASE64__"
    assert template.count(marker) == 1
    html = template.replace(marker, encoded)
    destination = BENCH / "review.html"
    # All HTML creation/updates go through apply_patch, including generated data.
    old = destination.read_text() if destination.exists() else None
    if old == html:
        print(f"Unchanged: {destination}")
        return
    if old is None:
        patch = (
            f"*** Begin Patch\n*** Add File: {destination}\n"
            + "\n".join("+" + line for line in html.splitlines())
            + "\n*** End Patch\n"
        )
    else:
        patch = (
            f"*** Begin Patch\n*** Update File: {destination}\n@@\n"
            + "\n".join("-" + line for line in old.splitlines())
            + "\n"
            + "\n".join("+" + line for line in html.splitlines())
            + "\n*** End Patch\n"
        )
    subprocess.run(["apply_patch"], input=patch, text=True, check=True, capture_output=True)
    assert destination.read_text().startswith("<!doctype html>")
    assert marker not in destination.read_text()
    print(f"Created {destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
