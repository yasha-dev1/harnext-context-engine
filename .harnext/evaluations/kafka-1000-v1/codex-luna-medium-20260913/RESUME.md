# Resume on another PC

Stopped at the user's request on 2026-09-13. **159/1,600 historical trials saved;
1,441 remain.** One interrupted MCP attempt (`H-assignee_at_snapshot-KAFKA-10520`)
is retained unscored in `interruptions/`. Resume skips completed trials and starts
a fresh session for the unfinished trial. Keep that interruption visible in analysis.
No benchmark processes remain running on the original PC.

The 200 coding candidates still require admission. Claude Code Haiku will use
native settings and remains on hold for Luna review. This command never starts Haiku.

## Setup

Use Linux or WSL2, Python 3.12+, uv, Node.js/npm and Git. From the repository root:

```bash
git fetch origin
git switch eval/e2-configurable-engine-kafka-benchmark
git pull --ff-only
uv sync --locked --all-packages
npm install -g @openai/codex@0.154.0
codex --version
codex login
```

Authenticate with an account that can access `gpt-5.6-luna`. No credentials are
included. Configure Git identity for automatic commits and allow at least 15 GiB
free disk. Restore the frozen corpus from its tracked gzip:

```bash
.venv/bin/python - <<'PY'
import gzip
from pathlib import Path
p = Path('apps/eval/benchmarks/kafka-1000-v1/data/context-sources.jsonl')
if not p.exists():
    p.write_bytes(gzip.decompress(p.with_suffix('.jsonl.gz').read_bytes()))
PY
```

## Validate, then resume

This verifies checkpoint hashes and CLI version without model calls or result writes:

```bash
.venv/bin/python -m harnext_eval.e2.benchmark_full \
  --config apps/eval/configs/benchmark-codex-full.yaml --check-resume
```

Expect `resume_validated`, `completed: 159`, `remaining: 1441` on first transfer.
Then, when ready to start the remaining Luna trials:

```bash
.venv/bin/python -m harnext_eval.e2.benchmark_full \
  --config apps/eval/configs/benchmark-codex-full.yaml --live
```

Keep the terminal/session alive. The runner commits checkpoints locally; push with
`git push`. Do not run the same checkpoint concurrently on two PCs. This resumes
the benchmark, not an ephemeral Codex conversation.

## Provenance

`manifest.json` and `implementation.json` retain original paths and code hashes.
`checkpoint.json` pins the compatible resume implementation and completed/interrupted
archives. Only orchestration changed: five repository paths may relocate, with
evidence validation before resume. Pilot, prompts, model/effort, engine and scoring
are unchanged. New machine details and configuration go under `resume-sessions/`;
analyze latency by machine segment because hardware changes affect timing.
`stop-record.json` records the interruption, and `summary.json` records live progress.
The stop checkpoint stays immutable as later results arrive.
