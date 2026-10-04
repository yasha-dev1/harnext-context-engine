"""Build and export context stores.

    python -m harnext_eval.e2.v2.stores.build --arm F5 --scope smoke --builder claude \\
        [--until 2023-02] [--window-events 25] [--window-bytes 60000] [--export all|2023-06,2023-07]
    python -m harnext_eval.e2.v2.stores.build --arm all-deterministic --scope full
    python -m harnext_eval.e2.v2.stores.build --arm S-tools --scope full --export all

Exit code 75 (EX_TEMPFAIL) means the build paused on a quota/auth/rate-limit/infra error:
the cursor is saved and the tree is clean; run the same command again to resume.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from harnext_eval.e2.v2.errors import PauseRun, reset_delay
from harnext_eval.e2.v2.stores.base import OUT, read_events
from harnext_eval.e2.v2.stores.deterministic import F0, F1
from harnext_eval.e2.v2.stores.llm_files import ARMS as LLM_ARMS

DETERMINISTIC = {"F0": F0, "F1": F1}
PAUSED = 75


def make(arm: str, *, builder: str = "claude", window_events: int | None = None,
         window_bytes: int | None = None, **options):  # fmt: skip
    """`window_events`/`window_bytes` default per arm: 1000/no cap for code-built stores,
    25 events or 60 kB of event text (whichever first) per builder run for LLM stores."""
    if arm in DETERMINISTIC:
        return DETERMINISTIC[arm](window_events=window_events, window_bytes=window_bytes)
    if arm in LLM_ARMS:
        return LLM_ARMS[arm](builder=builder, window_events=window_events,
                             window_bytes=window_bytes, **options)  # fmt: skip
    if arm == "S-tools":  # snapshots are taken while building: the wanted set goes in up front
        from harnext_eval.e2.v2.sources.index import SourcesStore

        export = options.get("export")
        return SourcesStore(None if export in (None, "all") else export.split(","))
    if arm == "R-hyb":
        from harnext_eval.e2.v2.stores.rag import RagStore  # loads numpy/fastembed lazily

        return RagStore(builder=builder)
    if arm == "RAW":  # raw data per source in ClickHouse (chdb) + file catalog; one export per month
        from harnext_eval.e2.v2.stores.raw.store import RawStore

        return RawStore()
    raise ValueError(f"unknown arm: {arm!r}")


def paths(scope: str, root: Path = OUT) -> tuple[Path, Path]:
    """(events.jsonl, stores dir): smoke builds never share a directory with full ones."""
    suffix = "" if scope == "full" else f"-{scope}"
    return root / f"stream{suffix}/events.jsonl", root / f"stores{suffix}"


def run(arm: str, args) -> None:
    stream, stores = paths(args.scope, args.root)
    if args.events:
        stream = args.events
    if not stream.exists():
        raise SystemExit(f"no event stream at {stream}" + (": with --root the stream is looked up under that root; pass --events <events.jsonl>" if args.root != OUT else ""))  # fmt: skip
    store = make(arm, builder=args.builder, window_events=args.window_events,
                 window_bytes=args.window_bytes, **({"export": args.export} if arm == "S-tools" else {}))  # fmt: skip
    out = stores / arm
    store.build(read_events(stream), out, until=args.until)
    state = json.loads((out / "build-state.json").read_text())
    done = sorted(state["checkpoints"])
    wanted = done if args.export == "all" else [c for c in (args.export or "").split(",") if c]
    for checkpoint in wanted:
        store.export(out, checkpoint)
    failed = len(state.get("failed_windows", []))
    print(f"{arm}: {len(done)} checkpoints ({done[0] if done else '-'} … {done[-1] if done else '-'}), "
          f"{failed} failed windows, exported {len(wanted)}; usage {state.get('usage', {})}")  # fmt: skip
    if not wanted and (out / "tree/.git").exists():
        print(f"{arm}: no export materialised; trials extract checkpoints on demand from the "
              "tags (runner `checkouts:`). Use --export only for small/smoke stores.")  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--arm", required=True, help="F0 … F6, R-hyb, S-tools, RAW, all-deterministic")
    parser.add_argument("--scope", choices=("smoke", "full", "w2026"), default="smoke")
    parser.add_argument("--builder", choices=("claude", "fake"), default="claude")
    parser.add_argument("--until", help="stop after this checkpoint, YYYY-MM")
    parser.add_argument("--window-events", type=int, help="max events per window (arm default)")
    parser.add_argument("--window-bytes", type=int, help="max event-text bytes per window")
    parser.add_argument("--export", help="'all' or comma-separated checkpoints to materialize "
                        "(small/smoke stores only; by default trials extract on demand)")  # fmt: skip
    parser.add_argument("--events", type=Path, help="override the scope's events.jsonl")
    parser.add_argument("--root", type=Path, default=OUT, help="override out/kafka-500-v2")
    parser.add_argument("--no-wait", action="store_true",
                        help="exit 75 on a usage-limit pause instead of sleeping until the reset")  # fmt: skip
    args = parser.parse_args(argv)
    arms = list(DETERMINISTIC) if args.arm == "all-deterministic" else [args.arm]
    for arm in arms:
        while True:
            try:
                run(arm, args)
                break
            except PauseRun as pause:
                if args.no_wait:
                    print(f"PAUSED (exit {PAUSED}): {pause}\nThe cursor is saved and the tree is "
                          "clean; rerun the same command to resume.", file=sys.stderr)  # fmt: skip
                    return PAUSED
                # Subscription limits: every window is already committed with its cursor, so
                # sleeping until the reset and calling run() again continues exactly there.
                delay = reset_delay(str(pause))
                print(f"PAUSED: {pause}\nwaiting {delay // 60} min, then resuming from the saved "
                      "cursor", file=sys.stderr, flush=True)  # fmt: skip
                time.sleep(delay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
