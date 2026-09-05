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
  "config_sha256": "3a72eece16c37b03af233834231f942ae8340eb040be4dd7d4101b9a996f9445",
  "created_at": "2026-09-05T16:30:24.146733+00:00",
  "excluded_label_functions": [
    "github_trunk_ci_failure_fix_6h"
  ],
  "exclusion_rules": [
    "First two observed months are tuning only (one for a two-month pilot).",
    "Window is half-open [START, END); no events outside it reach labels or tuning.",
    "Censored or inapplicable labeling functions abstain; unknown labels excluded from quality metrics.",
    "Rule-negative is the primary population; full-population results are secondary.",
    "Failed LF accuracy/coverage gates invalidate evidence; functions are never silently retained as valid.",
    "Rules exceeding monthly capacity invalidate that month; no hidden capacity is added.",
    "Harm is N/A: no real action provider / no S3 store in this profile.",
    "Labeling functions listed in excluded_label_functions are dropped before fitting; they are uncomputable from the registered sources and are not gated."
  ],
  "feature_history": "single chronological fold within requested window; four-week rolling baselines",
  "fit_window_months": 12,
  "git_head": "8998791dc6f9bb75db002b28fee8caa92f3e2852",
  "label_accuracy_min": 0.6,
  "label_coverage_min": 0.01,
  "policies": [
    "R0",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
    "R7"
  ],
  "primary_contrasts": [
    "R5-R1",
    "R5-R2"
  ],
  "primary_metric": "recall_at_2pct_rule_negative",
  "replay_sha256": "397da1ac548aa2c68dc2008417a03b783ae5a02d8aedc006fbddd48c6b9f8a28",
  "schema": "e1-prereg-v1",
  "window": [
    "2022-01-01T00:00:00+00:00",
    "2026-07-01T00:00:00+00:00"
  ]
}
```
