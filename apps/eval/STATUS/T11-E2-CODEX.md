# T11 integration: merged E2 Codex smoke

The merged E2/E3 infrastructure path is now `python -m harnext_eval.e2.context`.
See [the runbook](../E2-CONTEXT.md) and `configs/e2-context-codex-smoke.yaml`.
The legacy E2/E3 experiment-registry commands remain separate.

## Implemented

- `apps/builder/src/harnext_builder/harness/codex.py`: pinned Codex CLI invocation,
  structured event parsing, usage, completion/error handling, deadline termination,
  native tool limits, explicit model/effort, no sampling-seed fiction.
- `harness/codex_files.py`: explicit file-tool policy with JSON actions, allowlisted
  paths, symlink rejection, read/write byte limits and complete operation traces.
- Builder request/registry/settings/runner integration; `ConversationTranscript.ok`
  now rejects timeouts, max-turn stops and errors rather than committing partial work.
- Eval configuration, factory, store runtime and S3 integration carry model, effort
  and tool policy. S3 rejects a Codex completion with no context changes. Rollback
  and no-publication behavior are covered by failure fixtures.
- Provider identity-cache fix: stale/reused Python object IDs cannot silently
  transfer another configuration's offline state or provider summary.
- `providers/codex.py`: real, isolated Codex completions, no fake fallback, raw
  provider logs; actual usage separate from smoke byte accounting.
- `e2/context.py`: six-event/two-window fixture, ten independent gold probes,
  S0/S1/S3 stores, exact-SHA host retrieval, typed grades, final answer slot,
  immutable outputs, model/effort/source manifests, preserved-snapshot reader retests.
- `e2/audit_smoke.py`: deterministic replay of every tool response and budget,
  snapshot/hash checks, model/effort checks, and a separately labeled exact
  provenance-wrapper diagnostic that preserves original grades.
- Offline/live YAML profiles; runbook and README entry points; protocol, path,
  deadline, cache identity, budget, grading, rollback and audit-tampering tests.

## Validation

```bash
.venv/bin/pytest apps/eval/tests apps/builder/tests -q
```

**363 passed**, 10 existing numerical/multiprocessing warnings, 165.74 seconds.
Changed Python files pass Ruff. Pyright with `--pythonpath .venv/bin/python`
reports zero errors for the new adapters and merged E2 modules. Repository-wide
Ruff still reports two pre-existing E1 issues (`e1_human_sample.py` unused loop
variable; `test_scale.py` import ordering), left outside this change.

The authenticated CLI preflight succeeded with `gpt-5.6-luna`, medium effort,
Codex CLI 0.154.0. Native-shell trials exposed a bwrap setup failure on this host.
The explicit file-tool policy completed two real incremental context builds;
source code mounts were read and removed before snapshots. No sandbox bypass
was used. Native-shell harness performance is not established by this smoke.

Live outputs are under `apps/eval/out/e2-context/`. Exploratory runs and failures
are retained. Run 4 built the six snapshots and was interrupted during reader
testing after exposing the missing final-answer slot. Run 5 retests the three
arms separately against those exact SHAs, with no additional builder calls.
The arms run concurrently for wall-clock convenience; their response latencies
are not a serial performance benchmark.

Final verification: **30/30** typed completions and budget checks pass; **30/30**
saved tool traces reproduce exactly. Values: S3 9/10, S1 7/10, S0 9/10.
Strict evidence: S3 8/10, S1 8/10, S0 9/10. See the
[verified smoke report](E2-SMOKE-2026-09-12.md) for raw artifact links and limitations.

## Remaining study gates

No fake implementation substitutes for semantic retrieval or a knowledge graph
in this merged path. Those condition adapters, pinned token budgets, real causal
development probes, the 12 file cells, graph/retrieval matrices, harness parity,
repeated builds, selection statistics and held-out confirmation remain to be
implemented/frozen. The byte-budget fixture is explicitly smoke-only. USD prices
are unknown, not zero. Neither E1 routing nor Kafka transport is varied here.

S1's current fixed projection omits arbitrary fields such as `freeze_date`.
Include the planned lossless/ledger deterministic baseline before interpreting
any store difference as a benefit of model curation.

The next step is to freeze exact token and citation accounting, then run a small
causally audited real-data development pilot before expanding the configuration
matrix. The ten smoke questions must not be reused as confirmation evidence.
