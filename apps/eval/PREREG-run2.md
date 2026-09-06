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
  "config_sha256": "c89e62a81b323120de6aa7dd811c0b3a9d01e7f13b0dd6f650ebcf167e4947f5",
  "created_at": "2026-09-06T07:24:54.953956+00:00",
  "excluded_label_functions": [
    "github_trunk_ci_failure_fix_6h",
    "jira_fix_version_in_flight_later"
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
  "git_head": "84405faa3c141d8e3b4b089ecb2812abe8c9a94c",
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

Run 2 registration (2026-09-06). Differs from `PREREG.md` only by the amendments listed in that
file's appended amendment section: excluded LFs, `[VOTE]` thread-start rule, declared-priority
transition detection. Same replay, same window, same policies, budgets, primary metric and
label model.
