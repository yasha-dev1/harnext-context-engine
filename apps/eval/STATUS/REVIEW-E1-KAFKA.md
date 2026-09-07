# Independent review: Kafka E1, 20260905T185046Z

Reviewed 2026-09-05. Scope: `docs/evaluation-spec.md` §4.1 and §7 E1, `apps/eval/PREREG.md`, `STATUS/K2.md`, implementation, and the completed Kafka artifacts. No source or `apps/eval/out` files were changed; no git state-changing commands were used. The replay was sampled with `rg` into `/tmp`; only March 2024's 7,483 JSONL events were loaded. Parquet reads used PyArrow with selected columns and filters, not a whole-table load of all columns.

## Verdict

**The negative C1 result is not trustworthy as thesis evidence as-is. The arithmetic is reproducible, but the evaluation is contaminated by an uncomputable LF counted as negative evidence, poorly validated label fusion, and historical-payload provenance problems.** The run itself correctly reports `primary.valid=false`, `evidence_status=non-evidentiary`. It is a useful descriptive failure of this configured router against these generated labels, not an established failure of routing against real urgency. Fixing the evaluation does **not** guarantee that C1 will become positive.

The actual results JSON is `apps/eval/out/20260905T185046Z-e1-kafka/e1/seed-1/results.json`; there is no `results.json` at the run-directory root. All artifact paths below are relative to that seed directory unless otherwise stated.

Independent Parquet counts: **12,396,896 rows = 387,403 evaluation events × 32 conditions**, 52 months. There are 421 full-population positives (0.108725%), of which **378 are rule-negative**, on **103 distinct subjects**. Recomputing the paired subject-cluster BCa interval, 10,000 draws, seed 1, exactly reproduces the recorded contrast.

| Policy, 2% | Rule-negative positive admissions / 378 | Recall |
|---|---:|---:|
| R0 random | 9 | 0.023810 |
| R1 rules | 0 | 0 |
| R2 global HBOS | 27 | 0.071429 |
| R3 gap | 8 | 0.021164 |
| R4 HBOS without guards | 4 | 0.010582 |
| R5 guarded HBOS | 0 | 0 |
| R6 LOF | 6 | 0.015873 |
| R7 always-fast | 378 | 1 |

R5−R2 = **−0.07142857**, 95% CI **[−0.10334347, −0.04675550]**. This interval is conditional on fixed weak labels and the selected subject clusters; it does not quantify uncertainty in the label model or corpus reconstruction.

Two corrections to the supplied headline: **R5 and R1 have equal primary recall but differ on 83 admission decisions at 2%**; and **five**, not six, of the 12 LF accuracies are exactly 0.95. The stated mail vote count is correct: 3,962 mail hits; the all-source vote-rule count is 3,965.

## 1. Blocker — Missing release observability becomes 145,425 confident negative votes

**Evidence.** In `src/harnext_eval/e1/labels.py`, `_fix_version_in_flight` requires a nonempty target and either `in_flight` / `is_in_flight_release`, or equality with `in_flight_release` / `current_release`. `corpus/jira.py:parse_issue` emits `fixVersion` changes but never supplies those release-state fields. A read-only search of the entire replay for those JSON keys returned **zero matching lines**:

```sh
rg -c '"(in_flight_release|current_release|is_in_flight_release|in_flight)"\s*:' \
  apps/eval/out/corpus/kafka/replay/kafka-rlong.jsonl
# No output, exit 1: no matches.
```

Nevertheless `_apply_with_observability` treats every applicable JIRA event as observable for this horizon-None LF. `label_diagnostics.csv` reports **0 positive, 145,425 negative, coverage 0.3684061195, accuracy 0.95**. March alone has **150 fixVersion transitions**, no release-state keys, and **3,198 negative votes** from this LF. This is not evidence that none of the fixes targets an in-flight release; the required information is absent.

This is a real observability bug, analogous to the missing CI stream that the preregistration explicitly excluded. Wrong-source and finite-window censoring do abstain correctly: the March reapplication found **zero non-abstaining votes on rows marked unobservable**. The defect is that the release LF's observability predicate is too weak, not that the EM loop directly counts ABSTAIN as NEGATIVE.

**Fix.** Derive release-in-flight status from timestamped release information, or prospectively register this LF as uncomputable and exclude it. Make observability depend on required source fields, not just source membership and elapsed time. Refit labels and rerun all policies after a dated amendment; retain this run as invalid, not as the final negative result.

## 2. Major — Fusion collapses toward negative consensus; 0.95 is a cap, not measured accuracy

**Evidence.** `WeightedLabelModel.fit_predict` is a deterministic independent-LF, symmetric-accuracy EM-like model, not a supervised accuracy estimate. It:

- initializes each accuracy at 0.7 and prevalence from the fraction with at least half of non-abstaining votes positive, clipped to [0.01, 0.99];
- adds `log(a/(1-a))` per POSITIVE and subtracts it per NEGATIVE from prior log-odds;
- re-estimates accuracy as `(sum(expected agreement with its own posterior)+2)/(covered+4)`, i.e. Beta(2,2)-style smoothing;
- clips accuracies to **[0.05, 0.95]** and re-estimates prevalence, also clipped to [0.01, 0.99]. The optional `prior` only initializes prevalence; it is not held fixed.

Thus **0.95 is the hard upper cap**. Constant/rare-positive LFs can score very well by agreeing with an overwhelmingly negative posterior. It is not external evidence of 95% correctness. Five full-run columns hit this cap: priority-raised, fix-version, three-responders, vote-cancel/recast, hotfix-reference.

A JIRA event normally receives **six votes**, not just whichever positive outcome occurs. A committer reply alone is outweighed by five negative votes, including the impossible release LF. Using the reported learned accuracies and 1% prevalence bound, that pattern yields **p ≈ 7.36×10⁻⁸**. The 32,758 committer-positive votes are event-level LF firings across the fit corpus, not 32,758 independent urgent incidents.

Recorded evaluation posterior quantiles are median **0.0000284685**, p99 **0.00983005**, p99.9 **0.51953494**, max **0.97865914**. Thresholds 0.5 / 0.3 / 0.1 / 0.01 yield **421 / 422 / 574 / 1,856** positives. Merely lowering 0.5 to 0.1 barely changes the main problem.

I reapplied the 12 registered LFs to **March 2024 only**, end `2024-04-01T00:00:00Z`, then fitted the same model:

| Label rule on the same 7,483-event slice | Positives | Prevalence |
|---|---:|---:|
| Any outcome-positive vote, excluding declared-priority | 1,541 | 20.5933% |
| Any positive vote, including declared-priority | 1,553 | 20.7537% |
| At least half of non-abstaining votes positive, ties positive | 361 | 4.8243% |
| Strict majority of non-abstaining votes | 1 | 0.01336% |
| WeightedLabelModel ≥ 0.5 | 1 | 0.01336% |
| WeightedLabelModel after removing only fix-version LF | 6 | 0.08018% |

The saved full-window model also labels one March event positive. The slice's “later” horizons stop at April 1, however, so these are **slice sensitivities, not alternate full-corpus prevalence estimates**. Finite horizons crossing April 1 abstain. Ties matter enormously because GitHub normally has only two active LFs.

The non-collapse alternative is not automatically “any vote is truth”: weak outcomes can be false positives. However, treating absence of *each alternative urgency manifestation* as independent evidence of non-urgency effectively requires conjunctions of rare outcomes. Correlated human reactions, repeated event candidates, disjoint source-specific LF groups, and the always-negative release function make this assumption particularly questionable. I found no literal duplicate-column summation bug; the problem is model specification and observability.

**Fix.** Validate event-level posterior and LF precision/recall against the required independent human sample, including negative and source-stratified cases. Report positive firing rates, vote-pattern counts, source-specific posteriors, and leave-one-LF-out sensitivity. Choose and preregister an appropriate positive-evidence/abstention or dependency-aware model based on that validation; do not pick the rule that makes R5 win.

## 3. Major — Coverage gate accepts a completely non-informative LF

**Evidence.** `fit_label_model` actually calculates coverage as **`mean(vote != ABSTAIN)`**, not directly as `mean(observable)`. But application emits POSITIVE or NEGATIVE on *every observable row*, making those quantities equal in this implementation. All JIRA LFs therefore cover roughly **145,425 / 394,741 = 36.8406%**; the small 1h/24h differences are censoring. The diagnostic denominator includes the two tuning months, whereas quality metrics contain 387,403 evaluation events.

The 1% gate still detects a missing source or explicitly unobservable LF, but is **vacuous as a test of useful positive signal within an available source**. The release LF with zero positives passes both coverage and accuracy. The retained vote-cancel/recast LF has only **7 positive votes / 394,741 events = 0.00177%**, yet coverage is 8.9454%. Declared/outcome agreement **0.6434657** is measurable but is not human validation; it compares declared-positive to any outcome-positive over comparable events.

**Fix.** Keep conventional non-abstain coverage, but label it explicitly. Separately gate source/field observability, positive support, class-conditional performance, and informative variation. Define denominators and minimum support before the replacement run.

## 4. Major — Repeated lexical rule hits exhaust capacity; negation boundaries are better than the headline suggests

**Evidence.** `e1/policies.py:match_rule` scans flattened payload keys/values, except explicitly historical fields, with this precedence: declared priority → `[vote]` → CVE → blocker → on-call → dispute.

- **CVE is not an arbitrary substring match.** Its regex is `(?<![\w-])cve(?:-\d{4}-\d+)?(?![\w-])`. It rejects CVE embedded in ordinary word/hyphen tokens, but accepts actual CVE IDs anywhere in payload prose/URLs, including routine dependency changes and repeated descriptions.
- `[vote]` has **no first-message, thread, subject-only, or dedup condition**. Quotes and replies match too.
- `blocker` uses equivalent word/hyphen boundaries. Literal **`non-blocker` does not match**. There is no semantic negation handling for “not a blocker” or uncertainty, and quoted notifications match.

Sample command:

```sh
rg '"time":"2024-03-' apps/eval/out/corpus/kafka/replay/kafka-rlong.jsonl \
  > /tmp/review-e1-kafka-2024-03.jsonl
wc -l /tmp/review-e1-kafka-2024-03.jsonl  # 7483
```

Reapplying the actual rule function to that grep-selected month yields **168 hits**: vote 51, CVE 69, blocker 23, declared priority 25. Of the **51 vote mail hits**, **43 have `Re:` subjects and in-reply-to identifiers**; there are only **15 distinct threads**, with 15 first-observed-in-month messages. “First observed in month” is not necessarily the historical thread root.

The **69 CVE hits concern 12 subjects**; **49** also contain update/upgrade/bump/jline/dependency language. There are 48 JIRA, 14 mail, and 7 GitHub CVE hits. `issue:KAFKA-15882` contributes 19, `KAFKA-16347` 16, and `KAFKA-16322` 12. Example: `jira:13570572:created`, “Fix CVE-2023-50572 by updating jline from 3.22.0 to 3.25.1”; subsequent RemoteIssueLink transitions repeat the same description and trigger again. Those may be legitimate security work; the demonstrated problem is repeated lexical eligibility, not a claim that all dependency CVEs are harmless.

There are **11 raw `non-blocker` occurrences**: four have no rule and seven match CVE instead; none is assigned `blocker_word`. But `github:api:apache/kafka:comment-IC_kwDOACG9q854sIit` is flagged for “**Not sure if it's blocker priority though**.” Ten of the 23 blocker hits are actual priority transitions, so not all blocker hits are irrelevant prose. Conversely, all **8 March transitions `field=priority,to=Critical` are missed**, because declared detection only looks for fields named `priority`/`severity`, not transition `to`; the ten `to=Blocker` transitions happen to match the lexical fallback.

Across evaluation there are **8,982 rule hits / 387,403 = 2.3185%**: vote 3,965, CVE 2,360, blocker 2,094, declared 501, on-call 62.

| R5 budget | Infeasible months / 52 | Sum of excess rule hits | Sum unused capacity |
|---|---:|---:|---:|
| 1% | 52 | 5,105 | 0 |
| 2% | 33 | 1,808 | 493 |
| 5% | 2 | 55 | 9,625 |
| 10% | 0 | 0 | 26,969 |

These totals use one `source=all,population=full` metrics row per month, avoiding multiplication across report slices. The gate's **278** infeasible conditions counts both R1 and R5 across budgets: `2×(52+33+2)`; it is not 278 distinct months. At 2%, **221 of 378 primary positives fall in R5-infeasible months**; `_paired_primary` still includes them. Its valid flag prevents this from being silently evidentiary, but the headline interval is not restricted to feasible months.

**Fix.** Define event-vs-incident rule semantics prospectively: first genuine vote initiation, meaningful security escalation, structured priority transitions including Critical, and dedup/cooldown or explicit handling of repeated issue notifications. Audit semantic false positives before changing rules. Resolve the mandatory-floor/budget incompatibility explicitly; do not silently drop bad months or add hidden budget.

## 5. Major — Volume is attainable; adjacent anomalous-window confirmation is the dominant bottleneck

**Evidence.** `features.py` counts the **current fixed UTC 5-minute bucket up to and including the candidate**, not the eventual completed bucket and not a sliding five-minute interval. `GuardedHBOSPolicy.score` uses `count_5m >= 3`. With multi-window enabled, confirmation requires an anomalous raw HBOS score in the **immediately preceding bucket (`window−300`)**, then an anomalous event in the current bucket; confirmed state persists within that bucket. It does **not** mean independent evidence from the 5-minute and 1-hour feature windows, nor just any earlier anomaly. The preceding anomaly need not itself have volume ≥3. Rule-hit events return before updating anomaly-confirmation state; state starts afresh with each monthly policy instance.

The code chooses the **maximum raw-score key first**, then uses that key's two guards. A lower-scoring key that passes both guards cannot rescue an event when the winning key fails. That is a substantive multi-key semantics choice, not “admit if any entity baseline confirms.” Training threshold pools per-key and fallback training-score distributions; their comparability is also an unvalidated assumption.

March completed **occupied** bucket distributions, counted separately for each `baseline_key` (an event can belong to multiple keys; empty keys use the extractor's `__global__` fallback):

| Key type | Keys | Occupied buckets | Count=1 | Count=2 | Count≥3 | p50 / p90 / p95 / p99 / max | Adjacent pair with both counts≥3 |
|---|---:|---:|---:|---:|---:|---|---:|
| component | 29 | 1,637 | 1,111 | 268 | 258 | 1 / 3 / 4 / 11 / 35 | 45 |
| contributor | 231 | 2,722 | 1,694 | 788 | 240 | 1 / 2 / 3 / 6 / 14 | 19 |
| thread | 332 | 633 | 626 | 5 | 2 | 1 / 1 / 1 / 1.68 / 3 | 0 |
| global fallback | 1 | 831 | 557 | 149 | 125 | 1 / 3 / 4 / 6 / 45 | 10 |
| all | 593 | 5,823 | 3,988 | 1,210 | 625 | 1 / 3 / 4 / 6 / 45 | 74 |

Empty buckets are omitted from that distribution; including every March bucket for every observed key would make it overwhelmingly zero. Counts are event records, including separate JIRA changelog items, not unique human actions. At decision time **1,221 / 7,483 events (16.32%)** have some key with prefix count ≥3. Therefore real contributor/component entities absolutely **can** reach the volume floor. Thread keys almost never do. **1,414 / 7,483 (18.90%)** have no baseline keys and are pooled globally; these are not truly per-entity decisions.

Whole-run rule-negative guard diagnostics:

| Budget | Rule-negative events | Chosen key passes volume | Confirmed | Both / eligible | Admitted | Positive admitted |
|---|---:|---:|---:|---:|---:|---:|
| 2% | 378,421 | 62,804 (16.60%) | 1,299 | 255 (0.0674%) | 83 | 0 |
| 10% | 378,421 | 62,804 (16.60%) | 12,492 | 2,787 (0.7365%) | 2,787 | 0 |

The 0.74% headline is verified. R5=R1 is true for primary recall, **not decisions**. Zero positives among 83 deviations is unsurprising at 0.09989% rule-negative prevalence; even 2,787 uniformly sampled deviations would have zero positives with probability about **6.11%** (hypergeometric). This does not explain away the policy, but puts the sparse-label zero in perspective.

**Fix.** Specify whether confirmation means adjacent anomalous buckets or corroboration at two time scales, and whether any eligible key can qualify. Tune guard operating rates only on prior months, with source/key-type diagnostics and explicit sparse-entity handling. Audit how many counts are duplicated notifications/changelog items. Retain the registered configuration as an ablation; do not retrospectively tune it on these test outcomes.

## 6. Major — Sanity checks pass numerically but do not establish metric discrimination

**Evidence.** `validity.csv` confirms:

| Sanity quantity | Value |
|---|---:|
| 200-repeat uniform-random precision | 0.0010982696 |
| Full-population prevalence / always-flag precision | 0.0010872457 |
| Always-flag recall | 1.0 |
| Random VUS-PR | 0.0022255034 |

R0's pooled rule-negative recall is 9/378 = 2.381%, reasonably near 2%; its precision is 0.11870% against rule-negative prevalence 0.09989%. The repeated-random recall is not a separate row in validity.csv. Do not substitute the unweighted mean of monthly recall for the pooled primary.

The precision and VUS gates use **absolute tolerance 0.05**. At 0.001087 prevalence this permits an error about **46× prevalence**. Random VUS is already **2.05× prevalence** and passes. Moreover sanity VUS uses row-index coordinates over pooled events, whereas reported monthly VUS uses timestamp coordinates in median-cadence units; the identical `max_buffer=5` does not make these the same geometry.

There is no automatic near-floor test: `metric_remediation_recorded` only tests whether metadata is truthy. It is missing and required. The spec's “near” has no numerical margin. Concrete candidates for remediation, from **unweighted monthly means, 2%, full/all**: affiliation precision **R5 0.53181, R0 0.49252, R7 0.50774**; affiliation recall **R5 0.70879, R0 0.74645**. R5 is not distinguished from those floors as a useful lateness measure. On rule-negatives, VUS is **R5 0.0038390, R0 0.0030342, R7 0.0025031**; R5 and R4's VUS are **exactly equal**, because guards change admission eligibility but not rule-negative raw scores. VUS therefore cannot establish the guards' benefit.

**Fix.** Record explicit kept/dropped decisions and a predeclared discrimination criterion; secondary affiliation/NAB should be withheld pending the semantic repair in finding 8, and VUS should be flagged pending matched-geometry sanity checks. Use simulation intervals or prevalence-scaled tolerances. **Do not silently delete the preregistered primary recall because R5 loses to random**: preserve that failed result and distinguish policy failure from a defective metric. The spec's broad dropping rule needs a prospective clarification, not outcome-dependent metric selection.

## 7. Blocker — Timestamp firewall passes a 20-event audit, but historical payloads are not certified causal

**Evidence.** I selected 20 evenly spaced, time-ordered rows of March 2024 R5/2% with PyArrow (`event_id,t,decision_ts,baseline_key_used,features_fired`), matched them to the sampled replay, and independently recomputed each chosen key's in-bucket prefix count. **20/20 decision timestamps equal event time; 20/20 saved counts equal recomputed causal counts.** `_OutcomeIndex` gives strictly later related events in all 20 cases (19 nonempty March suffixes; the last has none). Equal-time events are excluded by `bisect_right` / `event.time > candidate_time`.

Examples, including the endpoints of the sample:

| Event | t / decision_ts (UTC) | Saved/recomputed count | First related March event strictly after t |
|---|---|---:|---|
| `jira:13570372:change:21895797:0` | Mar 1 00:25:48.622 | 1 / 1 | Mar 1 19:22:54.664 |
| `github:api:apache/kafka:comment-IC_kwDOACG9q853rHyg` | Mar 19 17:48:40 | 3 / 3 | Mar 19 17:48:44.161 |
| `jira:13568939:comment:17832665` | Mar 31 22:25:55.656 | 2 / 2 | none before Apr 1 |

For a candidate at t, finite LF windows are **(t,t+h]**; “later” LFs use **(t,2026-07-01]** against events already filtered to before July 1. March tuning is March 2023 through February 2024. `run.py` folds features chronologically and fits models on earlier months; `p_urgent` is attached only after policy scoring. I found no direct future-label-to-router-feature path. The full-window label model is fitted offline across all months; that is consistent with building evaluation labels, but its fitted parameters are not an online predictor.

**The stronger firewall still fails provenance scrutiny.** `corpus/jira.py:parse_issue` copies snapshot `summary`, `description`, and components into historical events. It uses final snapshot components for `baseline_keys` even though it reconstructs creation-state components for the creation payload. Comment `body` is the exported version, emitted at `created`, while `updated` can be later. In March, **31 of 317 JIRA comments (9.78%) have updated > created**. Example `jira:13570804:comment:17823930`: event **2024-03-06 09:10:43.039Z**, updated **12:19:35.391Z**. That proves post-creation revision metadata, not the exact words changed, but the parser has no historical body reconstruction. March contains no component transitions, so this sample does not quantify actual component misassignment.

`corpus/github_api.py:parse_pr` likewise places the fetched PR title/body at `createdAt`, and reuses snapshot PR title on subsequent historical records. Rules scan these payloads; label relationship matching also scans their text. Timestamp ordering cannot prove these strings were known at t. K2 additionally documents a **2026-09-05 committer roster applied retrospectively**; that affects outcome-positive identities and cannot establish historical committer status.

**Fix.** Reconstruct fields from dated change/edit histories where available, otherwise omit or mark mutable snapshot fields unavailable for historical rules/features/labels. Derive baseline components as of each event, not export time. Use historical committer membership or clearly registered contemporary-association evidence. Add field-availability provenance to the leakage gate. The 20-event audit validates count mechanics, not the full nine-feature vector or historical textual truth.

A separate qualification: `budgeted_decisions` uses the entire month's scores and count to choose top-b%; theta is only diagnostic. `decision_ts=t` is therefore a score/event timestamp, **not evidence that the final admission was executable online at t**. This retrospective ranking is explicitly permitted by the spec's exact top-b% formula, but must not be presented as a deployed causal admission schedule or measured zero-latency router.

## 8. Major — Real-corpus lateness metrics can credit unrelated entities and use event-index time

**Evidence.** `_metric_rows` sorts each month/source population and calls `affiliation_precision_recall(labels, admitted)` and `nab_low_fn_score(labels, admitted)` without subject or timestamps. Those functions build intervals from **adjacent row indices**; NAB considers the previous **five rows**, not elapsed minutes or an entity's response window. An admission on another issue/PR/thread can receive credit for a nearby positive. VUS uses timestamps, but still buffers the pooled month/source stream rather than separated entities. The entity-separated timestamped affiliation implementation is reached for constructed situation metadata, not these Kafka event labels.

For example, full-population R0 affiliation recall averages **0.74645** despite its approximately budget-level exact recall; always-fast affiliation precision averages **0.50774** despite actual precision only about 0.0011. These are properties of the distance metric and pooled coordinate system, not evidence of timely urgent routing. NAB is explicitly described in its own docstring as **“NAB-like”**, not an exact reference benchmark.

**Fix.** Define real-corpus situation onsets/endpoints, separate entities, and measure elapsed-time detection with matching randomized floors. Until then drop these outputs from lateness claims, retaining exact event-level recall/precision as descriptive results. Reference toy tests verify formulas on toy series; they do not validate Kafka's population/coordinate construction.

## 9. Major — Required evidence remains missing; subject clustering is narrower than the causal dependency structure

**Evidence.** Required nonpasses in `validity.csv` are rule-floor feasibility, `human_sanity_100_two_annotators`, `metric_remediation_recorded`, and `corpus_preflight`. Human sanity/remediation are displayed `not_applicable` but have **required=True** and invalidate the result. Flink replication and Corpus S's 200 situations × 3 seeds are supported but not run. Harm is separately and legitimately N/A for this no-provider/no-S3 profile; C1's “promoting changes the outcome” part is untested.

There are primary positives in **41/52 months** and 103 subjects. Subjects are issue/PR/thread IDs, **not the contributor/component baseline keys**. The bootstrap correctly clusters repeat positives within a subject, but linked JIRA/PR/mail copies and common contributors can span clusters. March's 642 mail events include **221 `[jira]` notifications** in the dev stream. Those are real observations, but duplication and automated notifications undermine an interpretation as independent human urgency reactions. The largest primary-positive subject contributes **18 of 378** events (`issue:KAFKA-14024`).

**Fix.** Complete the registered human and replication requirements or explicitly limit the thesis claim to an invalid exploratory Kafka run. Report sensitivity to linked-incident and calendar clustering; audit human-vs-automated outcome provenance. Preserve the current interval as conditional on the current subject definition, not as uncertainty over independent urgent incidents.

## 10. Minor — Calibration and noise robustness answer narrower questions than their names suggest

**Evidence.** `calibration.csv` contains **520 rows**, R5/2%, rule-negative, all-source, ten bins per month. `urgency_rate` is **mean posterior**, while recall/precision threshold it at 0.5. Weighted mean posterior is **0.00123196**, versus binary rule-negative prevalence **0.000998887**. Across months median Spearman rho is **0.030303**, positive in **26/52**. Pooled weighted bin means fall from decile 1 **0.00159964** to decile 10 **0.000923583**; this is not a demonstrated positive urgency gradient. It evaluates raw scores, including ineligible events, not admitted deviations.

`flip_labels` flips 10% of all binary event labels. At 0.1087% initial prevalence that creates approximately **10.087%** positives, roughly **93×** the original rate. The robustness artifact's full-population mean random precision becomes **0.09936** and R5's **0.11383**, versus their very small original precisions. That is an extreme synthetic noise test, not a mild 10% uncertainty perturbation of the sparse positive set. It is implemented as specified and should remain reported, but cannot validate the original labels.

**Fix.** Name calibration as posterior-vs-score binning and also show binary empirical rates with uncertainty and eligibility. Keep the registered flip test, accompanied by class-conditional LF/label uncertainty sensitivity that preserves or explicitly varies prevalence.

## Reproduction commands and audit code

Read-only inputs were inspected with `cat`, `sed -n`, `rg -n`, and selected-column PyArrow reads. The two scripts below were run from the repository root as:

```sh
UV_NO_SYNC=1 uv run python /tmp/review_e1.py > /tmp/review_e1_output.txt
UV_NO_SYNC=1 uv run python /tmp/review_e1_extra.py > /tmp/review_e1_extra_output.txt
```

They write no source or output artifacts. Recreate the March `/tmp` slice using the command in finding 4. Their source is embedded here so the quantitative audit does not depend on temporary files surviving. The full 20-event firewall audit prints each selected ID, timestamp, key, stored/recomputed count, and strict-future check. The first version of the supplemental script hit a JSON serialization error for tuple dictionary keys; the embedded version fixes that reporting-only error and completed successfully.

<details>
<summary>Primary artifact, rule, bucket, LF, and 20-event audit</summary>

```python
import json,re
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,UTC
import numpy as np,pandas as pd,pyarrow.parquet as pq
from harnext_eval.types import EvalEvent
from harnext_eval.e1.labels import apply_labeling_functions,fit_label_model,DEFAULT_LABELING_FUNCTIONS,_field,_OutcomeIndex
from harnext_eval.e1.policies import match_rule,_walk
from harnext_eval.corpus.committers import stamp_events
p=Path('apps/eval/out/20260905T185046Z-e1-kafka/e1/seed-1')
def out(k,v): print(k,json.dumps(v,default=str))
cols=['event_id','policy','p_urgent','admitted','rule_negative','subject','source','rule','month']
d=pq.read_table(p/'scores.parquet',columns=cols,filters=[('budget_pct','=',2.)]).to_pandas()
b=d[d.policy=='R0']; out('evaluated',{'n':len(b),'positive':int((b.p_urgent>=.5).sum()),'thresholds':{t:int((b.p_urgent>=t).sum()) for t in [.5,.3,.1,.01]},'quantiles':b.p_urgent.quantile([.5,.9,.99,.999,1]).to_dict(),'rules':b.rule.value_counts().to_dict(),'positive_by_source':b[b.p_urgent>=.5].source.value_counts().to_dict()})
for pol,g in d.groupby('policy',observed=True):
 n=g[g.rule_negative]; pos=n.p_urgent>=.5
 out('primary '+pol,{'n':len(n),'positive':int(pos.sum()),'hits':int((pos&n.admitted).sum()),'recall':float(n.loc[pos,'admitted'].mean()),'precision':float((n.loc[n.admitted,'p_urgent']>=.5).mean()),'clusters':n[pos].subject.nunique()})
a=d[d.policy=='R1'].set_index('event_id'); c=d[d.policy=='R5'].set_index('event_id');out('R1 R5 different decisions 2pct',int((a.admitted!=c.admitted).sum()))
m=pd.read_csv(p/'metrics.csv'); f=m[(m.policy=='R5')&(m.source=='all')&(m.population=='full')];out('R5 capacities',f.groupby('budget_pct').agg(months=('month','size'),infeasible=('budget_feasible',lambda x:int((~x).sum())),excess=('rules_over_budget','sum'),unused=('unused_capacity','sum')).to_dict('index'))
for budget in [2.,10.]:
 g=pq.read_table(p/'scores.parquet',columns=['admitted','eligible','rule_negative','p_urgent','features_fired'],filters=[('policy','=','R5'),('budget_pct','=',budget)]).to_pandas();n=g[g.rule_negative];out('R5 guards '+str(budget),{'n':len(n),'eligible':int(n.eligible.sum()),'admitted':int(n.admitted.sum()),'positive_admitted':int(((n.p_urgent>=.5)&n.admitted).sum()),'volume':sum(json.loads(x).get('volume_guard',False) for x in n.features_fired),'confirmed':sum(json.loads(x).get('multi_window_confirmed',False) for x in n.features_fired)})
# Only the selected month is loaded from JSONL.
events=sorted([EvalEvent.model_validate_json(l) for l in open('/tmp/review-e1-kafka-2024-03.jsonl')],key=lambda e:(e.time,e.id)); events,_=stamp_events(events)
out('month sources',Counter(e.source.split(':')[0] for e in events))
out('month rules',Counter(match_rule(e) for e in events))
mail=[e for e in events if '.mail.' in e.type]; votes=[e for e in mail if match_rule(e)=='vote'];out('mail vote',{'n':len(votes),'threads':len(set(e.subject for e in votes)),'reply_subject':sum(bool(re.match(r'\s*re:',str(e.data.get('subject','')),re.I)) for e in votes),'with_in_reply_to':sum(bool(e.data.get('in_reply_to')) for e in votes),'first_month_thread':sum(e.id==next(x.id for x in mail if x.subject==e.subject) for e in votes)})
for rule in ['cve','blocker_word']:
 hit=[e for e in events if match_rule(e)==rule];out('month '+rule,{'n':len(hit),'source':Counter(e.source.split(':')[0] for e in hit),'subjects':len(set(e.subject for e in hit))})
 for e in hit[:5]:
  tx=' '.join(_walk(e.data)); ma=re.search('cve|blocker',tx,re.I);out('example '+rule,{'id':e.id,'title':e.data.get('subject') or e.data.get('title') or e.data.get('summary'),'excerpt':tx[max(0,ma.start()-100):ma.end()+140] if ma else ''})
for pattern in ['cve','non-blocker','blocker','in_flight_release','current_release','is_in_flight_release','in_flight']:
 out('month text '+pattern,sum(bool(re.search(pattern,' '.join(_walk(e.data)),re.I)) for e in events))
out('fixversion transitions',sum(e.data.get('field')=='fixVersion' for e in events))
counts=Counter(); prefix=Counter(); eventprefix=[]
for e in events:
 keys=sorted(set(e.baseline_keys)) or ['__global__']; v=[]
 for k in keys:
  kw=(k,int(e.time.timestamp())//300);counts[kw]+=1;prefix[k.split(':')[0]]+=1;v.append(counts[kw])
 eventprefix.append(max(v))
out('month baseline memberships',prefix);out('empty keys',sum(not e.baseline_keys for e in events));out('event any volume',{'ge3':sum(x>=3 for x in eventprefix),'n':len(eventprefix)})
for kind in ['all','component','contributor','thread','__global__']:
 vals=[v for (k,w),v in counts.items() if kind=='all' or k.split(':')[0]==kind]; keys=set(k for k,w in counts if kind=='all' or k.split(':')[0]==kind)
 out('bucket '+kind,{'keys':len(keys),'occupied':len(vals),'hist':dict(sorted(Counter(vals).items())),'quantiles':np.quantile(vals,[.5,.9,.95,.99,1]).tolist() if vals else [],'ge3':sum(v>=3 for v in vals),'adjacent_ge3':sum(v>=3 and counts.get((k,w-1),0)>=3 for (k,w),v in counts.items() if kind=='all' or k.split(':')[0]==kind)})
# Reapply LFs within one month, explicitly truncated later horizon.
func=[f for f in DEFAULT_LABELING_FUNCTIONS if f.name!='github_trunk_ci_failure_fix_6h']
v=apply_labeling_functions(events,func,observation_end=datetime(2024,4,1,tzinfo=UTC));res=fit_label_model(v);mat=v.to_numpy();pos=(mat==1).sum(1);fired=(mat!=-1).sum(1);out('month vote models',{'n':len(v),'any_all':int((pos>0).sum()),'any_outcome':int((v.drop(columns='declared_blocker_critical')==1).any(axis=1).sum()),'majority_ties_positive':int(((2*pos>=fired)&(fired>0)).sum()),'strict_majority':int((2*pos>fired).sum()),'model_pos':int((res.probabilities>=.5).sum()),'unobservable_nonabstain':int(((~v.attrs['observability'].to_numpy())&(mat!=-1)).sum()),'pquant':res.probabilities.quantile([.5,.99,.999,1]).to_dict()});out('month diagnostics',res.diagnostics.to_dict('index'))
# sensitivity: same votes, remove structurally uncomputable LF
nod=fit_label_model(v.drop(columns='jira_fix_version_in_flight_later'));out('month drop fix LF positives',int((nod.probabilities>=.5).sum()))
# Twenty evenly spaced events from one policy/month, compare causal bucket prefix and strict suffix.
s=pq.read_table(p/'scores.parquet',columns=['event_id','t','decision_ts','features_fired','baseline_key_used','p_urgent','admitted','above_tuning_theta'],filters=[('month','=','2024-03'),('policy','=','R5'),('budget_pct','=',2.)]).to_pandas().sort_values(['t','event_id']);sample=s.iloc[np.linspace(0,len(s)-1,20,dtype=int)]; index=_OutcomeIndex(events); lookup={e.id:i for i,e in enumerate(events)}; seen=Counter(); causal={}
for e in events:
 for k in sorted(set(e.baseline_keys)) or ['__global__']:
  seen[(k,int(e.time.timestamp())//300)]+=1;causal[(e.id,k)]=seen[(k,int(e.time.timestamp())//300)]
for row in sample.to_dict('records'):
 e=events[lookup[row['event_id']]]; feat=json.loads(row['features_fired']);suffix=index.suffix(lookup[e.id],datetime(2024,4,1,tzinfo=UTC),relationships_only=True)
 out('firewall sample',{'id':e.id,'t':str(e.time),'decision_eq_t':row['decision_ts']==e.time,'key':row['baseline_key_used'],'count_saved':feat.get('count_5m'),'count_recomputed':causal[(e.id,row['baseline_key_used'])],'first_related_future':str(suffix[0].time) if suffix else None,'all_future_strict':all(x.time>e.time for x in suffix),'jira_updated':e.data.get('updated')})
# outcome provenance examples
out('JIRA updated later sample',[(e.id,e.time,e.data.get('updated')) for e in events if '.jira.issue.comment' in e.type and e.data.get('updated') and pd.Timestamp(e.data['updated'])>e.time][:5])
```

</details>

<details>
<summary>Supplemental floor, sensitivity, provenance, and CI audit</summary>

```python
import json,re
from pathlib import Path
from collections import Counter,defaultdict
import pandas as pd,numpy as np,pyarrow.parquet as pq
from harnext_eval.types import EvalEvent
from harnext_eval.e1.policies import match_rule,_walk
from harnext_eval.e1.labels import _field,_text,_OutcomeIndex,_references,_is_github
from harnext_eval.stats.stats import paired_difference_bca
p=Path('apps/eval/out/20260905T185046Z-e1-kafka/e1/seed-1')
def out(k,v): print(k,json.dumps({str(a):b for a,b in v.items()} if isinstance(v,dict) else v,default=str),flush=True)
events=sorted([EvalEvent.model_validate_json(l) for l in open('/tmp/review-e1-kafka-2024-03.jsonl')],key=lambda e:(e.time,e.id))
for word in ['non-blocker','blocker','cve']:
 hit=[e for e in events if word in ' '.join(_walk(e.data)).lower()];out('raw '+word,Counter(match_rule(e) for e in hit))
 if word=='non-blocker':
  for e in hit[:3]: out('nonblocker example',{'id':e.id,'rule':match_rule(e),'title':e.data.get('subject') or e.data.get('title') or e.data.get('summary')})
block=[e for e in events if match_rule(e)=='blocker_word'];out('blocker kinds',Counter(e.type for e in block));out('blocker priority transitions',sum(e.data.get('field')=='priority' for e in block))
for e in block:
 tx=' '.join(_walk(e.data));ma=re.search(r'(?<![\w-])blocker(?![\w-])',tx,re.I)
 if ma and ('not' in tx[max(0,ma.start()-70):ma.start()].lower() or '.github.' in e.type):out('blocker prose',{'id':e.id,'text':tx[max(0,ma.start()-100):ma.end()+100]})
cve=[e for e in events if match_rule(e)=='cve'];out('CVE bump surface',sum(bool(re.search('upgrad|updat|bump|jline|dependenc', ' '.join(_walk(e.data)),re.I)) for e in cve));out('CVE subjects',Counter(e.subject for e in cve).most_common(6))
j=[e for e in events if '.jira.' in e.type];comments=[e for e in j if '.comment' in e.type];out('JIRA snapshots',{'comments':len(comments),'updated_later':sum(bool(e.data.get('updated')) and pd.Timestamp(e.data['updated'])>e.time for e in comments),'component_transitions':sum(e.data.get('field')=='components' for e in j),'created':sum(e.type.endswith('.created') for e in j)})
for e in j:
 if e.data.get('field')=='components':out('component transition',{'id':e.id,'t':e.time,'keys':e.baseline_keys,'from':e.data.get('from'),'to':e.data.get('to'),'common_components':e.data.get('components')})
# sanity metric comparisons are unweighted month means, not pooled primary.
m=pd.read_csv(p/'metrics.csv');m=m[(m.source=='all')&(m.budget_pct==2.)]
out('sanity metrics means',m.groupby(['population','policy'])[['prevalence','recall_at_b','precision_at_b','vus_pr','affiliation_precision','affiliation_recall','nab_low_fn']].mean().to_dict('index'))
r=pd.read_csv(p/'robustness.csv');out('robustness at 2',r[r.budget_pct==2].groupby(['population','policy'])[['recall_label_flip','precision_label_flip']].mean().to_dict('index'))
c=pd.read_csv(p/'calibration.csv');out('calibration',{'n':len(c),'mean_p':np.average(c.urgency_rate,weights=c.n),'rho_median':c.drop_duplicates('month').spearman_rho.median(),'rho_positive_months':int((c.drop_duplicates('month').spearman_rho>0).sum()),'lift_non_null':int(c.lift_over_rules.notna().sum()),'deciles_mean_p':c.groupby('decile').apply(lambda x:np.average(x.urgency_rate,weights=x.n)).to_dict()})
t=pq.read_table(p/'scores.parquet',columns=['event_id','subject','policy','admitted','source','rule','p_urgent','budget_feasible','month'],filters=[('budget_pct','=',2.),('rule_negative','=',True),('p_urgent','>=',.5)]).to_pandas();v=t.pivot_table(index=['event_id','subject'],columns='policy',values='admitted',aggfunc='first',observed=True).reset_index();out('recomputed CI',paired_difference_bca(v.R5.astype(float),v.R2.astype(float),v.subject,n_resamples=10000,random_state=1));out('positive clusters largest',v.subject.value_counts().head(10).to_dict());out('pos by month counts',{'positive_months':t[t.policy=='R5'].month.nunique(),'R5_infeasible_positive':int((~t[t.policy=='R5'].budget_feasible).sum())})
# Entire month scored guard diagnostics.
s=pq.read_table(p/'scores.parquet',columns=['event_id','features_fired','p_urgent','rule_negative','eligible','admitted','above_tuning_theta','source'],filters=[('month','=','2024-03'),('policy','=','R5'),('budget_pct','=',2.)]).to_pandas();out('March score labels',{'positives':int((s.p_urgent>=.5).sum()),'mean_p':s.p_urgent.mean(),'n':len(s),'rule_negative_eligible':int((s.rule_negative&s.eligible).sum())})
# Make diagnostic symmetric-LF logit examples from full-run learned accuracies and fitted 1% lower-bound prevalence.
a=pd.read_csv(p/'label_diagnostics.csv').set_index('function').accuracy;out('accuracy cap count',int((a==.95).sum()));w=np.log(a/(1-a));jira=[x for x in a.index if x.startswith('jira') or x.startswith('declared')];logit=np.log(.01/.99)-w[jira].sum()+2*w['jira_committer_comment_1h'];out('one jira committer vote posterior at prior floor',1/(1+np.exp(-logit)))
```

</details>

Additional small checks used in findings 4, 5, and 9:

```sh
UV_NO_SYNC=1 uv run python - <<'PYCHECK'
from collections import Counter
from scipy.stats import hypergeom
import pyarrow.parquet as pq
from harnext_eval.types import EvalEvent
from harnext_eval.e1.policies import match_rule
es = [EvalEvent.model_validate_json(line)
      for line in open('/tmp/review-e1-kafka-2024-03.jsonl')]
mail = [e for e in es if '.mail.' in e.type]
print('JIRA notifications / mail:',
      sum('[jira]' in str(e.data.get('subject', '')).lower() for e in mail), len(mail))
for target in ['Critical', 'Blocker']:
    rows = [e for e in es if e.data.get('field') == 'priority' and e.data.get('to') == target]
    print(target, len(rows), Counter(match_rule(e) for e in rows))
print('Random zero-positive probability:', hypergeom.pmf(0, 378421, 378, 2787))
t = pq.read_table('apps/eval/out/20260905T185046Z-e1-kafka/e1/seed-1/scores.parquet',
                  columns=['source', 'rule'],
                  filters=[('policy', '=', 'R0'), ('budget_pct', '=', 2.), ('rule', '=', 'vote')])
print(t.to_pandas().source.value_counts())
PYCHECK
```
