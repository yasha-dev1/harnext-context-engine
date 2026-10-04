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
  "config_sha256": "009b2cc8e6b2185b9578f65b1a86a2d49e1d096b24c50e327d34c419d4473a5c",
  "created_at": "2026-10-04T09:54:01.961123+00:00",
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
  "git_head": "a6579cf9c865d35f5ee1ade0638e1361a8c0e471",
  "jev": {
    "admission": "R2-style monthly top-b% over all events, shared R0-R6 capacity, no guards",
    "categories": [
      "regression_or_outage",
      "security",
      "build_break",
      "release_risk",
      "minor_bug",
      "feature_work",
      "design_discussion",
      "code_review",
      "docs_or_chore",
      "user_question",
      "bookkeeping"
    ],
    "dev_set": "2021-10..2021-12 (before the window); prompt never scored on window months before registration",
    "fast_categories": [
      "regression_or_outage",
      "security",
      "build_break",
      "release_risk"
    ],
    "limitations": [
      "comment bodies and PR titles/bodies are export-time text (never versioned by the sources)",
      "the Jev build (2026-09-17) postdates the window: memorised project history cannot be excluded",
      "fast-lane categories name security problems, broken builds and release risks, which the outcome labeling functions also reflect; per-source results are reported"
    ],
    "model": "typesafe/jev-1.13",
    "prompt_sha256": "5d4d9014d24864f3bf6ef1fa886f899b1ba3df91cfaa1577a288e379f31da65f",
    "prompt_version": "se-triage-v2",
    "required_gates": [
      "jev_decisions_complete (failed <= 0.1 %)",
      "jev_single_model",
      "jev_decided_after_registration"
    ],
    "scores": {
      "R14": "sum of category probabilities over fast_categories + 0.001 * urgency",
      "R15": "urgency (noul)",
      "failed_decision": -1.0
    },
    "view": "JIRA summary/description/labels rebuilt as of the event from the changelog; issue keys, PR numbers, hashes and links removed; author shown only as person/automation"
  },
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
    "R13",
    "R14",
    "R15"
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
    "R13-R2",
    "R14-R0",
    "R14-R2",
    "R15-R0",
    "R15-R2",
    "R14-R15",
    "R14-R11"
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
