"""Offline E1 scale measurement for docs/evaluation-spec.md §7 E1.

Generate a replay with ``harnext-eval corpus`` first; no hourly world states are
built here. Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 before invocation.
This is a non-evidentiary performance run, with every policy/budget and metric.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from dataclasses import replace
from pathlib import Path

from harnext_eval.cli import _handle_for_replay, _write_result
from harnext_eval.config import load_config
from harnext_eval.e1.run import E1Experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("apps/eval/configs/e1-kafka.yaml"))
    args = parser.parse_args()
    corpus = _handle_for_replay(args.replay, "synthetic")
    corpus = replace(corpus, meta={**corpus.meta, "e1_only": True, "smoke": True})
    start = time.monotonic()
    result = E1Experiment().run(load_config(args.config).engine, corpus, args.out / "e1", 1)
    _write_result(result, args.out / "e1")
    metrics = {
        "seconds": time.monotonic() - start,
        "parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "child_peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "events": corpus.meta["event_count"],
    }
    (args.out / "performance.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
