"""Build a standalone result viewer through apply_patch; no model calls."""

import argparse
from pathlib import Path

from harnext_eval.e2.benchmark_report import build_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, action="append")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(build_report(args.run, args.out))


if __name__ == "__main__":
    main()
