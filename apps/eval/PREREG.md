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

## Amendment 2026-09-06 (appended; the registration block above is unchanged)

Run 1 (`20260905T185046Z-e1-kafka`) was executed against the block above and is retained as
exploratory/invalid (`evidence_status = non-evidentiary`). The independent review
(`apps/eval/STATUS/REVIEW-E1-KAFKA.md`) found that `jira_fix_version_in_flight_later` cannot be
computed from the sources (no release-state fields; 145,425 spurious negative votes), that the
declared-priority rule missed `priority → Critical` changelog transitions, and that `[VOTE]`
fired on every reply. These are corrected prospectively and registered as **run 2** in
`apps/eval/PREREG-run2.md` (new config hash): `exclude_label_functions` gains
`jira_fix_version_in_flight_later`; the LF abstains when release-state fields are absent;
`rules.vote_thread_start_only = true`; changelog `to ∈ {Blocker, Critical}` counts as declared
priority. The label-fusion model is **unchanged** in run 2 (registered weighted model, p ≥ 0.5);
alternative fusion rules are reported as post-hoc sensitivity only
(`apps/eval/scripts/e1_label_sensitivity.py`) pending the human sanity sample.

## Amendment 2026-09-06, run 3 (appended; the registration block above is unchanged)

Run 2 (registered in `PREREG-run2.md`) executed on the laptop; its outputs are not on the machine
that runs run 3 and it is superseded. Run 3 addresses the remaining review findings
(`apps/eval/STATUS/REVIEW-E1-KAFKA.md`) prospectively and is registered in `PREREG-run3.md`
against a freshly rebuilt replay (the GitHub snapshot is re-fetched, so the replay hash changes).

Changes registered for run 3, each tied to a review finding:

1. **Finding 7 (provenance).** JIRA `components` and the `component:` baseline keys are replayed
   as of each event from the changelog (`corpus/jira.py::_components_timeline`), never taken from
   the export-time snapshot. Summary/description/comment/PR text remains snapshot text; the run
   records an informational `historical_text_provenance` gate with the count of payloads edited
   after event time and the thesis states it as a limitation.
2. **Finding 4 (rule semantics).** In addition to run 2's `[VOTE]` thread-start and
   `→ Critical` transition fixes, `rules.dedup_per_subject = true`: each (subject, rule) fires once
   per evaluation month, so repeated notifications of one incident do not re-consume the lane.
3. **Findings 4/5 and the rule-accounting question (rule cost vs. scorer budget).** Two new
   registered conditions: **R8** = R5's scorer and guards with rule hits admitted *outside* the
   budget (`rules_outside_budget` reported; deviation admissions capped at the same monthly
   capacity as R0–R6), and **R9** = R8 with any-key guard eligibility. R5 is retained unchanged as
   the original design. Registered contrasts: R5−R1, R5−R2, R8−R2, R9−R2, R8−R5, and the
   scorer bake-off against the random floor (R2−R0, R4−R0, R8−R0, R9−R0). The primary metric
   and population are unchanged (recall@2 % on rule-negative events).
4. **Finding 3 (vacuous coverage gate).** New gate `label_positive_support_min = 20` positive
   votes per LF; `dev_vote_cancelled_recast_later` (7 positives in run 1) is excluded before
   fitting. Diagnostics add `positive_rate` and `accuracy_at_cap` (0.95 is the model's clip).
5. **Finding 2 (label fusion).** The registered label stays the weighted model at p ≥ 0.5. Four
   label definitions are registered for post-hoc sensitivity of the primary metric
   (`prereg.LABEL_DEFINITIONS`: weighted_model_p50, any_outcome_lf, two_outcome_lfs,
   half_of_votes; script `scripts/e1_label_definition_sensitivity.py` over `label_votes.parquet`).
   None replaces the registered label without the human sanity sample.
6. **Finding 6 (sanity tolerance, remediation).** Random-scorer gates use a relative tolerance
   (`sanity_relative_tolerance = 0.5`) and the random VUS uses the same timestamp geometry as the
   reported metric. `metric_remediation.csv` records kept/dropped for every secondary metric by a
   mechanical criterion (R8 − R0 paired over months at 2 %/rule-negative, |mean| > 2 SE).
7. **Finding 8 (lateness geometry).** Row-index affiliation/NAB are reported only as
   `*_rowindex`. Timestamped, subject-separated situations are derived from consecutive positives
   (`label_situation_gap_hours = 24`) and drive affiliation P/R and detection delay per policy and
   budget for the full and rule-negative populations.
8. **Finding 9 (clustering, human sample).** The paired bootstrap is reported with subject
   clusters and, as sensitivity, calendar-month clusters. The run writes the blind
   `human_sanity_sample.csv` (50 top R8-eligible + 50 top R2 rule-negative events) and a key file;
   the gate stays not-run until two annotators fill it in.
9. **Finding 10.** Calibration adds the binary empirical rate per decile; robustness adds a
   prevalence-preserving swap perturbation next to the registered uniform flip.

Not changed: budgets, window, fit window, primary metric/population, label model, the R5
guards (retained as the registered ablation; not tuned on these months).

## Amendment 2026-09-06, run 4 (appended; the registration block above is unchanged)

Run 3 (`20260906T145312Z-e1-kafka`, `PREREG-run3.md`) showed R8 and R9 identical to R5 in every
admission (the guards, not the budget, block the deviation layer) and the urgency signal to be
source-specific (global HBOS on JIRA only, the gap scorer on dev@, per-entity HBOS on GitHub).
Run 4 is registered in `PREREG-run4.md` against the same replay and config:

1. **R8 and R9 removed** (redundant with R5; the rule-exempt accounting question is settled).
2. **R10 per-source global HBOS** (new design condition): one global HBOS fitted per source
   (jira / github / mail) on the trailing 12 months; each event is ranked by the percentile of
   its raw score within its source's training-score distribution, so the monthly top-b % cut
   allocates the budget across sources by within-source anomaly rather than by one detector's
   scale. Unseen sources fall back to the global model.
3. **R11, R12, R13 = R2 with a different global detector**: IsolationForest (100 trees, seeded),
   ECOD (empirical-CDF tail score, O(log n) per event, same rule as pyod's ECOD), and LOF
   (k = 20, standardised features, novelty mode). COPOD was considered and dropped as a
   near-duplicate of ECOD on this feature vector.
4. Registered contrasts: R5−R1, R5−R2, R2−R0, R4−R0, R10−R0, R10−R2, R11−R2, R12−R2, R13−R2.
   Design condition for the metric-remediation record: R10. Human sanity sample: 50 top R10 +
   50 top R2 rule-negative events.
5. Everything else (budgets, window, fit window, primary metric and population, label model,
   label-definition sensitivity, rules, gates) is unchanged from run 3. The candidate set was
   chosen after seeing run 3 and is stated as such; no parameter was tuned on evaluation months.
