# Known limitations of the offline (smoke) evaluation profile

This file records, honestly and in one place, what the smoke profile does NOT evidence.
The framework is built so each item below is a configuration away, not a rewrite; the
"real profile" column names the switch. Reviews R1/R2 (apps/eval/REVIEW/) flagged these;
they are documented here rather than papered over.

| # | Limitation | Why it exists in smoke | Real profile switch |
|---|---|---|---|
| 1 | All LLM/embedding providers are deterministic fakes; no result is evidence about a real model | Offline-by-design (see api-isolation-audit.md) | `builder.harness: claude_code`, `reader.provider: anthropic`, pinned embeddings adapter, `offline: false` |
| 2 | The corpus is synthetic (scenario engine with injected situations); no external validity | Real extractors exist but network fetches are a deliberate separate step | `harnext-eval corpus --replay` over Pony Mail/JIRA/GH Archive extractions (Kafka R-H1 / R-long) |
| 3 | Sample sizes are tiny (120 events / 10 probes per family / 1 seed); CIs are wide or NaN and several statistical checks are N/A | Smoke must finish in minutes | default profile sizes (2000+ events, 300 probes, 150 tasks, 3 seeds) |
| 4 | Human-in-the-loop validity items (30-probe pilot κ, 200-pair judge calibration, S1 fairness review) cannot run offline | They require people | G1/G4 gate procedure in docs/evaluation-spec.md §10 |
| 5 | E6 measures an in-process pipeline model (partitions/workers simulated), not a live broker on separate hardware | No broker in smoke | E6 Kafka transport config + separate load host |
| 6 | PREREG chronology is recorded but not externally timestamped | Git timestamps only | commit PREREG.md before first real run |
| 7 | Some R2 review findings about spec literalism at real-profile scale remain open by design; each is listed in the review files with a disposition | Convergence traded against runtime | re-run R2 review set on the first real-profile run |

Every experiment's results.json carries a `checks` map where each smoke-N/A item appears
as a structured reason, never a silent pass.

## Outstanding items from FINAL-VERIFICATION.md (deliberately deferred)

The final verification (apps/eval/REVIEW/FINAL-VERIFICATION.md) audited 49 R2 blockers:
33 verified fixed, 6 partial, 10 not fixed. The unfixed set is, by decision, the
**real-profile evidentiary package**, deferred until the first real-corpus run:

- PREREG.md creation and enforced chronology/hash inputs on `run` (E1-8, E3-5).
- The frozen 300-probe / 150-task / 3-seed evidentiary populations wired through the CLI
  (E2-2, E3-4, E5-4) — the generators support the sizes; the frozen-package plumbing does not exist yet.
- Corpus-S temporal/update/multi-source probe generators for non-JIRA sources (E2-3).
- The E6 Kafka transport's route-decision table for guard comparisons on a live broker.
- Human-audit input artifacts (pilot κ, judge calibration, S1 review) as typed inputs.

These do not affect the offline smoke story; they gate the first *evidentiary* run and
are the first work items of that phase.
