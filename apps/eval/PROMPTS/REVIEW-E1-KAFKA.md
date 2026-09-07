You are an independent reviewer (fresh eyes). Do NOT modify source code or anything under apps/eval/out. Write your findings to apps/eval/STATUS/REVIEW-E1-KAFKA.md. No git state-changing commands.

Context: read docs/evaluation-spec.md §4.1 and §7 E1, apps/eval/PREREG.md, apps/eval/STATUS/K2.md, then the real-corpus E1 run in apps/eval/out/20260905T185046Z-e1-kafka/ (results.json, e1/seed-1/{metrics,label_diagnostics,validity,calibration,robustness}.csv, scores.parquet — 12.4M rows, read with pyarrow selecting columns). Replay: apps/eval/out/corpus/kafka/replay/kafka-rlong.jsonl (629k events; read-only; sample with head/grep, do not load fully).

Headline observed by the orchestrator (verify independently, do not take on trust):
- Primary recall@2% on rule-negatives: R5 = 0.000, R1 = 0.000, R2 (global HBOS) = 0.071, R0 random = 0.024, R6 LOF = 0.016. R5−R2 = −0.071, CI [−0.103, −0.047], 103 entity clusters.
- Rules floor flags 2.32% of events (mail `vote` 3,962; `cve` 2,360; `blocker_word` 2,094; declared_priority 501) so at b ≤ 2% the mandatory rule hits exceed capacity in 33/52 months (rules_over_budget total 1,808) and R5 ≡ R1.
- Even at b = 10% with 26,969 unused slots, only 0.74% of rule-negatives pass R5's guards (absolute_floor 3 in 5-min bucket + multi-window) and none of the 2,787 admitted is positive.
- Label model: 421 positives among 387,403 events (0.11%); p_urgent 99.9th percentile = 0.52; threshold sensitivity 0.5→0.1 changes positives 421→574. Yet raw positive votes are plentiful (jira_committer_comment_1h fires 32,758 times). Six of 12 LF accuracies are exactly 0.95.

Questions to answer with evidence (commands + numbers):
1. Is the label model collapsing? Inspect e1/labels.py WeightedLabelModel: how are accuracies estimated, is 0.95 a cap/prior, how do NEGATIVE votes enter, why does p_urgent stay near 0 despite tens of thousands of positive votes? Is there a bug (e.g. negative votes from *unobservable* windows counted, or double-counting) or a defensible modeling choice? What would prevalence be under a plain majority/any-outcome-vote rule (compute it from votes if you can re-apply LFs on a 1-month slice; otherwise reason from the diagnostics).
2. Rule floor over-matching: inspect e1/policies.py rule detection. Does `cve` match as a substring (e.g. inside other words / dependency bumps)? Does `vote` flag every message in a [VOTE] thread rather than the thread's first message? Does `blocker_word` fire on "non-blocker"/"blocker" in unrelated prose? Quantify with grep on the replay for a sample month.
3. Guards: inspect GuardedHBOSPolicy — is `eligible` computed as intended (≥3 events in the entity's 5-min bucket AND confirmation in a second window)? Would a real Kafka contributor/component entity ever reach 3 events per 5 minutes? Report the empirical distribution of 5-min bucket counts per baseline_key from a sample month of the replay.
4. Are the sanity floors informative: R0 recall ≈ budget, always-flag = 1, random precision ≈ prevalence — confirm from validity.csv. Are any metrics "floor-near-R5" that the spec says must be dropped (metric_remediation gate)?
5. Coverage column: why does every JIRA LF show coverage 0.3684 (= JIRA's share of events)? Is coverage measuring "observable" rather than "voted non-abstain"? Does that make the ≥1% coverage gate vacuous?
6. Check the leakage/temporal firewall for one policy month: features use only ≤ t, labels only > t (sample 20 events from scores.parquet and reason about decision_ts vs label windows).
7. Anything else that would make a thesis examiner distrust these numbers.

Structure the report: Verdict (is the negative C1 result trustworthy as-is, or contaminated by an eval bug?), then numbered findings each with severity (blocker / major / minor), evidence, and a concrete fix suggestion. Be specific and quantitative. Finish within ~60 minutes of work.
