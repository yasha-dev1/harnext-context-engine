# F-synthetic status

## Built

- `corpus/synthetic.py`: deterministic v2 scenario engine with 40 default
  issues, 2,000 events over 60 days, ON/OFF bursts, evolving issue state,
  linked KAFKA/KIP/PR/thread records, two-segment changed-file paths, actor
  roles/bots, exact injected-situation metadata, four urgent archetypes,
  detectable post-onset outcomes, and outcome-free benign flash crowds.
- `providers/llm.py`: deterministic time-aware structured reader for raw JSON,
  S1 overview/facts/timeline text, links, code locations, and schema-constrained
  E4 actions.
- `providers/embeddings.py`: feature hashing augmented with token bigrams,
  trigrams, and strongly weighted exact-ID features.
- `stores/fake_usage.py`: deterministic byte-derived input/output token and fake
  price model. The E5 fold handle calls it when fake S0/S1/S4 folds persist
  `usage.jsonl`; E5 reads the resulting usage cost and reports the raw cadence
  ratio even when reportability gates correctly keep the primary claim invalid.
- `tests/test_core`: coverage for v2 corpus construction, exact scenarios and
  hard negatives, state/cross-source structure, role balance, structured-reader
  formats/cutoffs/action JSON, ID-sensitive embeddings, and fake usage batching.

## Minimal probe fixture adjustments

1. `test_update_probes_require_a_real_transition` now recognizes the transition
   from its changelog instead of requiring `transition` in the event type. The
   silent-burst warm-up intentionally preserves the Jira transition payload
   while exposing it as telemetry.
2. `test_code_location_probe_requires_cross_source_linkage` still verifies the
   per-probe source union, but no longer requires an arbitrary probe to contain
   more than one source. Corpus v2 creates one merged PR per issue/horizon, so
   the linkage is carried by embedded KAFKA/KIP keys and changed files.

## Smoke before / after

Final run: `apps/eval/out/20260830T193119Z-baseline-minimal`.

| Check | Before | After |
|---|---:|---:|
| E1 prevalence | 0 | 0.0333 |
| E1 recall@2%, R0 / R5 | NaN / NaN | 0.0 / 1.0 |
| E2 retrieve-everything | 0.60 | 0.96 |
| E2 A0 / A1 / A3 / A4 | vacuous / vacuous / vacuous / 0.20 | 0.20 / 0.88 / 0.38 / 0.20 |
| E3 S0 / S1 / S3 / S4 at 8k | all 0.20 | 0.82 / 0.68 / 0.64 / 0.50 |
| E4 V3-V1 / V3-V6 | 0 / 0 | -0.3333 / -0.3333 |
| E4 max archetype share | 1.0 | 0.3333 |
| E4 V6 median >= 3x V3 | failed | passed |
| E5 W20+rules / W1 cost | 1.0 | 0.9456 |
| E6 fitted burstiness B | vacuous | 0.6204 |
| E6 urgent SLO gap at 1.5x knee | 0 | 0 |

All E1 policy recall values are finite in the final smoke. E2 A4 differs from
A1, A0 remains at the 0.20 prior, and retrieve-everything exceeds 0.90. E3 is
non-flat. E5 writes 120 W1 usage rows versus 35 W20+rules rows and derives the
reported non-unit ratio from those records.

## Verification

```bash
uv run pytest apps/eval/tests/test_core apps/eval/tests/test_probes -q
# 49 passed

uv run pytest apps/eval/tests -q
# 197 passed

uv run ruff check apps/eval
# All checks passed

make eval-smoke
# completed; report at the final run path above
```

`git diff --check` passes for the owned files. No git state-changing command was
run.

## Remaining integration issue

E6 consumes `meta.injected_situations`, reports nonempty urgent gold, and fits a
nonzero burstiness value, but its current in-process smoke runner still gives
both lane designs 100% urgent attainment against a 2-second SLO; consequently
the measured gap remains exactly zero. The corpus metadata contract has no
smoke queue/service-time override beyond choosing the smoke/research profile.
Fixing that result requires changing the concurrently owned E6 load/queue
model, which was explicitly outside F-synthetic ownership. No E6 source was
edited here.
