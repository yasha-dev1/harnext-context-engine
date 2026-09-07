# Final verification — Round 3

Date: 2026-08-31

## Overall verdict

**FAIL — the Round-3 patch materially improves and correctly labels the offline smoke path, but it does not fix every Round-2 blocker and the fresh reports violate the requirement that every false check carry a reason.**

The requested static gates and both fresh smoke commands exit 0. The two runs use byte-identical frozen inputs (`replay_hash=6c9f4987b7831f3184483881451a0646154e42a2723ab77e4e98f8e0255f8b06`, `probe_hash=439a987459fa5ce263f82965895825ce367f91a2a214abc152cd324358668932`). E2–E6 publish the requested headline structures, and E5/E6 do not claim validity when their gates fail. However:

- 49 blocker findings were audited: **33 VERIFIED, 6 PARTIALLY, 10 NOT-FIXED**.
- The registered non-smoke path still cannot supply/enforce the complete frozen probe, raw-gold, human-audit, G4, and preregistration package required by §7.
- Corpus-S temporal/update/multi-source probe generation remains absent.
- Builder seeds still label stores but do not reach `HarnessRequest` or the Claude model settings.
- The E6 Kafka result still returns no route-decision table, so the research guard comparison remains unable to execute as declared.
- Both fresh reports contain bare false checks with no reason: E5 `shared_e2=false` and E6 `repetitions_p99_within_20pct=false`. The HTML check table has no reason column. Parent primary objects name the failed gates, but that is not the requested per-check reason.

Verdict meanings below: **VERIFIED** = the blocker-level behavior exists and the focused regression test passes; **PARTIALLY** = the core patch exists but an explicit part of the blocker remains; **NOT-FIXED** = the required path/evidence is still absent.

## Blocker-by-blocker verification

### R2-e1.md

| Blocker | Verdict | One-line evidence |
|---|---|---|
| E1-1 — R7 was budgeted instead of always-fast | VERIFIED | `e1/run.py::_admit_month` has an R7 bypass that admits all rows; the focused end-to-end test asserts 240/240 R7 admissions and passed. |
| E1-2 — mandatory rules could exceed the common monthly capacity | VERIFIED | `rank_with_budget` now applies one total capacity to R0–R6 and records `budget_feasible=False` on a rule-floor overrun; the overfull-floor test passed. |
| E1-4 — non-reference/mismatched VUS-PR | VERIFIED | `e1/score.py` implements the 250-threshold range-AUC PR surface and the same five-buffer setting is used for reporting/floors; three pinned reference-value cases passed. |
| E1-5 — approximate/partial affiliation and missing NAB | VERIFIED | Every arm/budget metric row now includes affiliation P/R and `nab_low_fn`; Huet-reference, timestamped-affiliation, and end-to-end coverage tests passed. |
| E1-7 — incomplete/non-tri-state validity gating | VERIFIED | `e1/run.py` builds all gates before a single `valid` conjunction and serializes pass/fail/N/A with reasons; fresh E1 false/N/A rows carry structured reasons. |
| E1-8 — no executable real/simulated profile and prereg chronology | NOT-FIXED | `run_command` still has no typed prereg/profile/audit inputs, generates one synthetic replay from `cfg.seeds[0]`, and `build_manifest` is still called without `prereg_ref`; no `apps/eval/PREREG.md` exists. |
| E1-11 — harm check trusted metadata rather than executing paired S3/E4 work | PARTIALLY | Executable S3/E4-shaped harm code and its focused planted test pass, but both fresh runs promote one event and produce zero pairs, with `harm_paired_coverage`, leakage, non-vacuity, and provider gates failed. |
| E1-15 — replay seam discarded R5 eligibility/attribution | VERIFIED | `replay/driver.py` consumes `eligible`, selected key, rules and guard outcomes without the former second guards; the multi-key/two-month negative-path test passed. |
| E1-16 — quadratic LF evaluation / exact-gold path still ran weak labels | PARTIALLY | Indexed/bisected LF evaluation and exact-sidecar weak-label bypass exist and are tested, but no scale test demonstrates the specified ~350k-event profile is operational. |

### R2-e2.md

| Blocker | Verdict | One-line evidence |
|---|---|---|
| E2-1 — no configurable pinned real embeddings | VERIFIED | Strict config accepts Voyage/OpenAI with mandatory model+revision, `s3-curated.yaml` pins `voyage-3-large@2025-01-07`, and no-network A3/S4 factory/manifest tests passed. |
| E2-2 — registered run cannot construct the 300/60/150 evidentiary population | NOT-FIXED | `run_command` still defaults to 10/family and calls `_generate_probes` without evidentiary/raw-Jira/join/A0/pilot inputs; only the standalone library validator has those arguments. |
| E2-3 — Corpus-S temporal/update/link probe generators absent | NOT-FIXED | `gen_temporal.py` and `gen_update.py` still skip any history whose `source_kind != "jira"`, while `gen_multisource.py` remains regex-event based rather than world-state/event-graph based. |
| E2-4 — smoke passed S0 as A4 and emitted no primary | VERIFIED | CLI supplies built S3 as `store_handle`; `E2Experiment` requires layout S3 and a non-empty A4-A3 row. Both fresh runs publish the 50-probe A4-A3 contrast. |

### R2-e3.md

| Blocker | Verdict | One-line evidence |
|---|---|---|
| E3-1 — changing/incomplete erosion panel could emit a zero slope | VERIFIED | E3 freezes checkpoint-safe templates, re-golds the same panel at each checkpoint, requires panel/family completeness, and emits NaN when incomplete; both fresh runs have 10/10 complete rows at all five smoke checkpoints. |
| E3-2 — primary validity ignored E2/E3 gates | PARTIALLY | `consolidated_gates` now invalidates every contrast with named reasons, but human/gold/review evidence is still accepted as unauthenticated scalar metadata and is not configurable as frozen artifacts. |
| E3-3 — no pinned real embeddings/S4 identity | VERIFIED | Voyage/OpenAI adapters, strict model/revision validation, real-profile pinning, and A3/S4 adapter propagation are present and the focused tests passed. |
| E3-4 — registered run does not enforce the frozen R-H1/Corpus-S probe population | NOT-FIXED | The normal CLI still generates probes across the replay with `per_family=10` by default and exposes no frozen 300-probe/audit input path. |
| E3-5 — preregistration/input hashes not enforced | NOT-FIXED | No `apps/eval/PREREG.md` exists and `run_command` still omits `prereg_ref`, prompt/CLAUDE/schema/threshold audit inputs from the manifest contract. |

### R2-e4.md

| Blocker | Verdict | One-line evidence |
|---|---|---|
| E4-1 — directional hypothesis result was treated as validity | VERIFIED | Non-vacuity is now a constant-output plumbing check; signed contrasts are published independently of direction and the planted “everything wins” falsification test passed. |
| E4-2 — Corpus-S action gold was trusted metadata, not world state | VERIFIED | Synthetic corpus emits versioned world-state snapshots with stable fact IDs and task loading derives action/owner/required IDs from them; both focused tests passed. |
| E4-3 — batch evaluation started from a post-window snapshot | VERIFIED | `_batch_fold_plan` resolves the closing fold and its parent, then `_scratch_batch_fold` folds the real window only on scratch; the immutability/changed-answer test passed. |

### R2-e5.md

| Blocker | Verdict | One-line evidence |
|---|---|---|
| E5-1 — deviation admissions ignored R5 eligibility/threshold | VERIFIED | `_event_clock_replay` requires both `policy_eligible` and `score >= threshold`; the ineligible-high-score regression test passed. |
| E5-2 — cadence accuracy discarded non-A4 store arms | VERIFIED | `_run_e2` retains `store_arm` and all downstream filters use it; both fresh `cost.csv` files contain finite cadence `macro_acc` values. |
| E5-3 — shared E2 gates could never clear correctly | PARTIALLY | E5 now passes audit scalars and checks an explicit positive requirement list, but those values are unverified metadata rather than frozen artifacts; fresh `shared_e2` remains false because all cadence retrieve-everything floors fail. |
| E5-4 — no claim-eligible real-profile CLI/config path | NOT-FIXED | CLI/config still have no frozen probes, E1 labels, raw-gold/world-state, G4, or prereg inputs; `_claim_profile` remains a truthiness check over metadata strings. |
| E5-5 — authentic nested SDK usage was priced as zero | VERIFIED | `_tokens` recursively descends nested `usage`, requires input/output tokens and exact model identity, and the hand-calculated nested-usage/config-price test passed. |
| E5-6 — frozen gold silently replaced by non-evidentiary derivation | NOT-FIXED | `run_cadences` still calls `_rederive_probes`, which constructs Python/SQL adapters from normalized events and ends with `audit.require_valid(evidentiary=False)`. |
| E5-8 — invalid primary ratio still published | VERIFIED | Point/CI are null unless every gate passes, diagnostic ratio is separately named, and both fresh runs include a labelled `unavailable_reason`. |
| E5-12 — fake projections masqueraded as authentic usage/bypassed prices | VERIFIED | Fake rows are labelled `usage_kind=deterministic_projection`, priced from config, and excluded from `authentic_provider_usage`; the frozen fake-price regression passed. |

### R2-e6.md

| Blocker | Verdict | One-line evidence |
|---|---|---|
| E6-1 — finite HBOS score was treated as an R5 anomaly | VERIFIED | Production two-lane routing owns `GuardedHBOSPolicy` and consumes threshold plus `eligible`; the below/above-threshold R5 test passed. |
| E6-2 — batch-fold starts counted as fast urgent SLO hits | VERIFIED | Urgent two-lane rows use finite latency only for lane `fast`; batch rows become infinite-latency misses. In-process and fixture-backed Kafka tests passed. |
| E6-3 — Kafka research matrix crashes at guard route comparison | NOT-FIXED | `external_records_to_run` still returns the default empty `route_decisions`; the guard loop unconditionally selects `result.route_decisions[["event_id", "lane"]]`, so the declared research path remains incomplete. |
| E6-4 — applicable validity failures did not gate the primary | VERIFIED | `_applicable_validity_failures` drives `primary.valid/invalid_reasons`, and invalid headline charts are annotated; both focused validity tests passed and fresh primaries are `valid=false`. |
| E6-5 — rule admissions bypassed the total causal budget | VERIFIED | Allowance is now `floor(events_seen * budget_pct)`, applies to rule and deviation admissions, and overflow is recorded; the every-prefix rule-heavy test passed. |

### R2-graders.md

| Blocker | Verdict | One-line evidence |
|---|---|---|
| GR-1 — registry E2 ignored S3 `store_handle` | VERIFIED | `E2Experiment.run` requires S3 `store_handle`, and the CLI integration test plus both fresh runs prove an A4 arm and A4-A3 contrast. |
| GR-2 — only fake embeddings were configurable | VERIFIED | Strict Voyage/OpenAI adapters and real profile pinning exist; config/factory/manifest/A3/S4 no-network tests passed. |
| GR-3 — pilot/claim-grader evidence unreachable in production | PARTIALLY | Registry now forwards `pilot_kappa` and `claim_disagreement` metadata, but there is still no typed/hash-verified artifact input or production caller for repeated claim grading. |

### R2-probes.md

| Blocker | Verdict | One-line evidence |
|---|---|---|
| PR-1 — normal run cannot build/audit the evidentiary population | NOT-FIXED | The library supports evidentiary arguments, but `run_command` supplies none of raw-Jira, resolutions, join audit, A0 audit, code allocation, 150-entity minimum, or frozen reports. |
| PR-2 — Jira initial state copied from current search snapshot | VERIFIED | Both Python and raw-SQL derivations reconstruct earliest changelog `from` values; the realistic Closed-snapshot/Open-before-transition test passed. |
| PR-3 — real GH Archive path could not supply merged-PR file gold | VERIFIED | Code joins merged PR merge SHA to push files and audits title/head keys; extractor-to-generator and branch-only key tests passed. |
| PR-4 — no pinned embedding provider | VERIFIED | Voyage/OpenAI provider support and the pinned real profile are present; no-network construction and integration tests passed. |
| PR-5 — partial code file sets scored as zero | VERIFIED | E2 and A0 use file-set F1 keyed by `gold_type="files"`; partial-set `2/3` regression tests passed. |
| PR-6 — smoke omitted A4-A3 without failing | VERIFIED | Registry requires S3 and the mandatory contrast row; both fresh runs contain exactly one A4-A3 row. |

### R2-stores.md

| Blocker | Verdict | One-line evidence |
|---|---|---|
| ST-1 — E3 used a second non-E2 retrieval path | VERIFIED | `_evaluate_condition` calls `evaluate_e2` for every store/budget; fresh shared-E2 answer logs cover S0/S1/S3/S4, with S3 recorded as literal A4. |
| ST-2 — E3 preselected/truncated with a fake tokenizer | VERIFIED | The active E3 path delegates material selection, provider tokenizer, accounting and grading to E2; the old `_store_material` helper remains but is not called by evaluation. |
| ST-3 — no real configurable embedding provider | VERIFIED | Voyage/OpenAI strict adapters and the pinned Voyage real profile exist; A3/S4 propagation tests passed. |
| ST-4 — erosion used a shrinking panel | VERIFIED | Fixed checkpoint-safe templates are re-golded at every point and incomplete matrices suppress slopes; fresh smoke rows are all panel/family complete. |
| ST-5 — S1 omitted structured state and had no review gate | PARTIALLY | Source/world-state reducers now populate the required fields and tests pass, but the 20-folder review is still only an optional scalar audit and the CLI cannot supply a frozen review artifact. |
| ST-6 — S3 seeds did not affect builder requests | NOT-FIXED | `HarnessRequest` still has no seed/temperature fields and `run_builder_harness` passes neither; CLI seed changes labels/roots only. |

## Focused regression commands

All focused commands were run from the repository root without source edits.

```text
E1 + replay focused nodes:
........                                                                 [100%]
8 passed in 20.84s

E2/E3/providers/stores focused nodes:
...............                                                          [100%]
15 passed in 12.15s

Probe focused nodes:
.....                                                                    [100%]
5 passed in 0.95s

E4 focused nodes:
......                                                                   [100%]
6 passed in 5.46s

E5 focused nodes:
....                                                                     [100%]
4 passed in 3.40s

E6 focused nodes:
.......                                                                  [100%]
7 passed in 10.25s
```

## Required command outputs

### `uv run pytest apps/eval/tests -q`

```text
........................................................................ [ 28%]
........................................................................ [ 57%]
........................................................................ [ 86%]
..................................                                       [100%]
250 passed in 82.67s (0:01:22)
```

Exit code: 0.

### `uv run ruff check apps/eval`

```text
All checks passed!
```

Exit code: 0.

### `uv run pyright apps/eval/src`

```text
0 errors, 0 warnings, 0 informations
WARNING: there is a new pyright version available (v1.1.410 -> v1.1.411).
Please install the new version or set PYRIGHT_PYTHON_FORCE_VERSION to `latest`
```

Exit code: 0.

### `make eval-smoke`

```text
uv run harnext-eval run --config apps/eval/configs/baseline-minimal.yaml --corpus synthetic --all --event-count 120 --entity-count 12 --per-family 10 --smoke
building S0 from the shared E3 replay (120 events)
building S1 from the shared E3 replay (120 events)
building S4 from the shared E3 replay (120 events)
building S3-sonnet-seed-1 from the shared E3 replay (120 events)
running e3 across configured seeds [1]
building S0 store for seed 1 (120 events)
building S3 store for seed 1 (120 events)
running e1 seed 1
running e2 seed 1
running e4 seed 1
running e5 seed 1
running e6 seed 1
report: /home/yasha/Desktop/uni/masters/harnext-context-engine/apps/eval/out/20260830T230156Z-baseline-minimal/report.html
apps/eval/out/20260830T230156Z-baseline-minimal
```

Exit code: 0.

### S1 comparable run

Command:

```text
UV_NO_SYNC=1 uv run harnext-eval run --config apps/eval/configs/s1-templated.yaml --corpus synthetic --all --event-count 120 --entity-count 12 --per-family 10 --smoke
```

Output:

```text
building S0 from the shared E3 replay (120 events)
building S1 from the shared E3 replay (120 events)
building S4 from the shared E3 replay (120 events)
building S3-sonnet-seed-1 from the shared E3 replay (120 events)
running e3 across configured seeds [1]
building S0 store for seed 1 (120 events)
building S1 store for seed 1 (120 events)
building S3 store for seed 1 (120 events)
running e1 seed 1
running e2 seed 1
running e4 seed 1
running e5 seed 1
running e6 seed 1
report: /home/yasha/Desktop/uni/masters/harnext-context-engine/apps/eval/out/20260830T230641Z-s1-templated/report.html
apps/eval/out/20260830T230641Z-s1-templated
```

Exit code: 0.

## Fresh-run artifact verification

| Requirement | Baseline fresh run | S1 fresh run | Verdict |
|---|---|---|---|
| Comparable frozen inputs | Replay/probe hashes `6c9f…8b06` / `439a…8932` | Same hashes | PASS |
| E2 publishes A4(S3)−A3 | Delta `-0.1315873`, BCa CI `[-0.2953785, 0.0025455]`, 50 probes, 16 entities, 10,000 resamples | Identical | PASS |
| E3 covers S0/S1/S3/S4 at the same reader path | `curve.csv` has all four at 2k/8k/32k; all 8k answer logs are under `e3/shared-e2/budget-8000/accuracy`; S3 rows are arm A4 | Identical condition/path coverage | PASS |
| E3 fixed smoke erosion panel | All stores/checkpoints have `panel_size=n=10`, `panel_complete=True`, `family_complete=True`; 60-probe gate is labelled N/A | Identical | PASS |
| E4 publishes signed V3−V1 and V3−V6 with CIs | V3−V1-N20 `+0.25 [0.25,0.25]`; V3−V1-N100 `+0.25 [0.25,0.25]`; V3−V6 `+0.25 [0.25,0.25]` | Identical quality effects/CIs | PASS |
| E5 CI-gated primary or labelled reason | Primary point/CI null; diagnostic `0.9456978`; unavailable reason names claim profile, usage provenance, E2 checks, non-triviality | Primary null; reason additionally names failed equal-accuracy CI | PASS |
| E6 publishes SLO attainment with validity | Gap `-1.0`, `valid=false`, invalid reason `repetitions_p99_within_20pct` | Identical | PASS |
| Every false check carries a reason | Bare `E5.shared_e2=false` and `E6.repetitions_p99_within_20pct=false` | Same two bare failures | **FAIL** |

The E5 bare failure is backed only indirectly: `e2_checks.csv` shows `floor_retrieve_everything_ge_0_9=0` for W1, W20+rules, and W20+rules+deviation. The E6 parent primary names the failing repetition gate, but neither the raw check nor the rendered HTML row explains why the repetitions exceed 20%.

## Per-experiment smoke trust verdict

“PASS” here means the plumbing result is trustworthy within the limitations explicitly documented in `apps/eval/LIMITATIONS.md`; it does not turn fake-provider/synthetic/tiny-sample output into scientific evidence.

| Experiment | Verdict | Evidence |
|---|---|---|
| E1 | **FAIL** | Routing/metric fixes are real and invalidity is explicit, but the fresh S3 harm path has 1 promotion and 0 paired results, with real coverage/leakage/non-vacuity failures; this is more than the documented tiny/fake-provider limitation. |
| E2 | **PASS** | The required A4(S3)−A3 plumbing, leakage/budget records, 10k paired CI, and non-evidentiary label are present in both runs. |
| E3 | **PASS** | All four mandatory layouts use the shared E2 evaluator, the smoke panel is fixed/complete, and evidentiary-only gates are explicitly N/A. |
| E4 | **PASS** | Signed primary contrasts and CIs are present; the fake/tiny/one-run restrictions are explicit and do not directionally gate publication. |
| E5 | **FAIL** | Primary suppression is correct, but every cadence fails the retrieve-everything floor and the top-level `shared_e2=false` check has no reason. |
| E6 | **FAIL** | SLO_att is published with `valid=false`, but the applicable repetition-stability check fails and is rendered as a bare false check without an explanatory reason. |

## Final conclusion

The narrow Round-3 smoke fixes for A4 wiring, shared E3 reading, E4 signed contrasts, E5 primary suppression, and E6 validity gating are real. The global claim that the Round-2 blocker set is fixed is not: ten blocker paths remain absent, six are only partial, three experiments fail the fresh smoke trust assessment, and the “every False check carries a reason” requirement fails identically in both new reports. No result from these runs should be promoted beyond the documented smoke/plumbing scope, and the final verification verdict is **FAIL**.
