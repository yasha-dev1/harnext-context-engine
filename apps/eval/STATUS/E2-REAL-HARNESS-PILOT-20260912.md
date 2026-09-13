# Native Codex benchmark pilot — 2026-09-12

The historical-task evaluation path completed a real native-harness pilot on
eight frozen Kafka development tasks, one deterministically selected task from
each historical family. This is the merged E2/E3 experiment's integration gate.
It does not admit coding tasks or approve the full benchmark run.

[Interactive report: six graphs and all task/tool details](../../../.harnext/artifacts/e2-native-benchmark-pilot-001.html)

## Configuration and execution

- Native Codex CLI 0.154.0, `gpt-5.6-luna`, medium reasoning effort.
- [Pilot YAML](../configs/benchmark-codex-pilot.yaml) fixes selection and harness settings.
- [Engine YAML](../benchmarks/kafka-1000-v1/context-profile.yaml) fixes the existing hybrid store, BM25 retrieval, and typed graph condition.
- Each task receives a fresh native session and a corpus snapshot at its own cutoff.
- Historical tasks have seven read-only Harnext MCP tools, with native shell,
  web, apps, skills, and subagents disabled. Native events are checked against
  the allowlist, and violations terminate the run.
- Gold is held outside the reader workspace; output schemas use public task
  families without revealing expected values or history lengths.
- A separate negative native-access canary returned `unavailable`, with no
  native tool execution. This is a behavioral check plus configuration/event
  enforcement, not a formal proof of operating-system isolation.

Run from the repository root with a fresh output directory:

```bash
.venv/bin/python -m harnext_eval.e2.benchmark_pilot \
  --config apps/eval/configs/benchmark-codex-pilot.yaml \
  --out apps/eval/out/benchmark-pilot/codex-luna-medium-002 --live
```

The completed run is `apps/eval/out/benchmark-pilot/codex-luna-medium-001`.
It retains exact prompts, launch commands, native transcripts, answers, frozen
snapshots, hash-chained server receipts, scores, and runner provenance.
`source-pin-validation.json` independently confirms that all eight snapshot
source hashes match the frozen benchmark corpus. Benchmark tasks and splits
were not changed.

## Observations and interpretation

| Measurement | Observed value |
| --- | ---: |
| Deterministic typed answer correctness | 8/8 |
| Required-support recall | 1.0 on every task |
| Completed MCP calls | 25 |
| Tool errors | 0 |
| Native calls/responses matching server receipts | All 25 |
| Returned context bytes | 134,759 |
| Snapshot build time, summed | 183.6 s |
| Reader time including startup, summed | 256.9 s |
| Native input tokens | 467,895 |
| Cached input tokens, included in input total | 253,952 |
| Native output tokens | 3,491 |
| Eight materialized snapshots | 2,758,367,720 bytes |

Calls comprise 10 searches, six source fetches, six entity searches, two context
listings, and one context read. No graph-neighbor or assertion calls were used.
Success therefore does not establish that graph traversal improves answers.
Raw source access is enabled in this condition, so representation comparisons
must explicitly control that access.

Answer scoring compares typed values or ordered transitions to frozen gold.
Retrieval scoring credits complete required evidence found in actual returned
content; source identifiers alone are insufficient. Recall measures exposure
to required support, not causal use by the model. Precision is not reported
because relevance annotation is incomplete. Eight tasks under one condition
cannot establish a winning configuration or full-benchmark performance.

The report shows answer correctness/support recall, tool counts, response bytes
against budget, build/reader latency, native token usage, and per-call cumulative
support recall. Each task exposes the exact prompt, expected and actual answer,
and executed tool arguments/responses.

## Validation

- All 52 E2 tests passed, including four new evidence, schema, and native-boundary tests.
- Targeted runner Ruff and Pyright checks passed.
- Chromium checked eight result rows, six charts, task inspection and curve
  selection, desktop/mobile layouts, no mobile overflow, no JavaScript errors,
  and no external requests. Screenshots and verification JSON accompany the HTML.

## Next gates

1. Admit a small set of real coding candidates: provision their pinned Kafka
   checkout and Gradle environment, then prove supplied tests fail on the base
   and pass with the reference PR. Existing patch-application checks alone are
   insufficient. Keep reference patches and gold outside the agent workspace.
2. Run those admitted tasks through Codex with MCP plus isolated shell access.
   Save test outcomes, implementation diff, all native/MCP calls, and retrieval
   scores; compare behavior to the reference without requiring identical code.
3. Run paired development comparisons across context conditions and a second
   harness with matched prompts, budgets, source visibility, and task selection.
4. Reduce duplicated snapshot storage before expanding the configuration matrix:
   this eight-task run alone materialized about 2.76 GB. Retain provenance and
   cutoff checks while investigating reusable immutable corpus/index data.
5. Complete benchmark review and freeze the chosen protocol before running the
   untouched confirmation split. This pilot does not replace user approval.
