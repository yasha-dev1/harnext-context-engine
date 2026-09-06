# E1 preregistration

Commit this document before the first evidentiary run. Amendments must be appended with dates.

```json
{
  "budgets_pct": [
    1.0,
    2.0,
    5.0,
    10.0
  ],
  "config_sha256": "15062b5bc4c678f64f45af461ddf278e5c165e3c48b20d34bccf2eefd89311d1",
  "created_at": "2026-09-06T18:53:26.693583+00:00",
  "excluded_label_functions": [
    "github_trunk_ci_failure_fix_6h",
    "jira_fix_version_in_flight_later",
    "dev_vote_cancelled_recast_later"
  ],
  "exclusion_rules": [
    "First two observed months are tuning only (one for a two-month pilot).",
    "Window is half-open [START, END); no events outside it reach labels or tuning.",
    "Censored or inapplicable labeling functions abstain; unknown labels excluded from quality metrics.",
    "Rule-negative is the primary population; full-population results are secondary.",
    "Failed LF accuracy/coverage gates invalidate evidence; functions are never silently retained as valid.",
    "Rules exceeding monthly capacity invalidate that month; no hidden capacity is added.",
    "Harm is N/A: no real action provider / no S3 store in this profile.",
    "Labeling functions listed in excluded_label_functions are dropped before fitting; they are uncomputable from the registered sources and are not gated.",
    "R10 fits one global HBOS per source and ranks by within-source percentile; R11-R13 are R2 with IsolationForest, ECOD and LOF; all share the R0-R6 monthly capacity.",
    "An LF whose positive support is below label_positive_support_min fails its gate.",
    "Secondary metrics are kept only if the design condition (R10) is separated from the random floor by more than twice the paired monthly standard error; the decision is recorded in metric_remediation.csv before any interpretation.",
    "Lateness metrics use timestamped, subject-separated situations derived from consecutive positives (label_situation_gap_hours); row-index affiliation/NAB are reported as *_rowindex only."
  ],
  "feature_history": "single chronological fold within requested window; four-week rolling baselines",
  "fit_window_months": 12,
  "git_head": "fef1f6ca6e03cd75baf279359c5a4f8259faeec0",
  "label_accuracy_min": 0.6,
  "label_coverage_min": 0.01,
  "label_definition_sensitivity": [
    "weighted_model_p50",
    "any_outcome_lf",
    "two_outcome_lfs",
    "half_of_votes"
  ],
  "label_positive_support_min": 20,
  "label_situation_gap_hours": 24.0,
  "policies": [
    "R0",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
    "R7",
    "R10",
    "R11",
    "R12",
    "R13"
  ],
  "primary_contrasts": [
    "R5-R1",
    "R5-R2",
    "R2-R0",
    "R4-R0",
    "R10-R0",
    "R10-R2",
    "R11-R2",
    "R12-R2",
    "R13-R2"
  ],
  "primary_metric": "recall_at_2pct_rule_negative",
  "replay_sha256": "d2d43a2204169cd1a4ee97ca08c64d2ae9dfe8caf759b149a6fc2465717784b6",
  "rules": {
    "dedup_per_subject": true,
    "vote_thread_start_only": true
  },
  "sanity_relative_tolerance": 0.5,
  "schema": "e1-prereg-v1",
  "window": [
    "2022-01-01T00:00:00+00:00",
    "2026-07-01T00:00:00+00:00"
  ]
}
```

Run 4 registration (2026-09-06). Same replay (d2d43a22…) and config as run 3; differs by the
"Amendment 2026-09-06, run 4" section of `PREREG.md`: R8/R9 removed, R10 per-source global HBOS
added as the design condition, R11–R13 global detector variants of R2, contrasts updated.
