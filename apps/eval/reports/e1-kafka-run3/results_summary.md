# E1 results summary — run `20260906T145312Z-e1-kafka`

Window ['2022-01-01 00:00:00+00:00', '2026-07-01 00:00:00+00:00'] · events 387,403 · rule-negative positives 839 (219 subjects, 48 months) · rule hit rate 0.0077 · evidence status **non-evidentiary**

## Primary metric: recall@2 % on rule-negative events

| policy | recall_2pct |
|---|---|
| R0 | 0.020 |
| R1 | 0.000 |
| R2 | 0.089 |
| R3 | 0.014 |
| R4 | 0.005 |
| R5 | 0.000 |
| R6 | 0.013 |
| R7 | 1.000 |
| R8 | 0.000 |
| R9 | 0.000 |

## Registered contrasts (paired BCa, subject clusters; month-cluster CI as sensitivity)

| contrast | effect | ci_low | ci_high | subject_clusters | month_ci_low | month_ci_high |
|---|---|---|---|---|---|---|
| R5-R1 | 0.000 | 0.000 | 0.000 | 219 | 0.000 | 0.000 |
| R5-R2 | -0.089 | -0.111 | -0.070 | 219 | -0.117 | -0.066 |
| R8-R2 | -0.089 | -0.111 | -0.070 | 219 | -0.117 | -0.066 |
| R9-R2 | -0.089 | -0.111 | -0.070 | 219 | -0.117 | -0.066 |
| R8-R5 | 0.000 | 0.000 | 0.000 | 219 | 0.000 | 0.000 |
| R2-R0 | 0.069 | 0.047 | 0.093 | 219 | 0.047 | 0.097 |
| R4-R0 | -0.015 | -0.028 | -0.006 | 219 | -0.029 | -0.005 |
| R8-R0 | -0.020 | -0.033 | -0.012 | 219 | -0.033 | -0.011 |
| R9-R0 | -0.020 | -0.033 | -0.012 | 219 | -0.033 | -0.011 |

## Recall at every budget (pooled, rule-negative)

| policy | condition | 1 % | 2 % | 5 % | 10 % |
|---|---|---|---|---|---|
| R0 | R0 random | 0.011 | 0.020 | 0.055 | 0.098 |
| R1 | R1 rules only | 0.000 | 0.000 | 0.000 | 0.000 |
| R2 | R2 global HBOS | 0.045 | 0.089 | 0.137 | 0.166 |
| R3 | R3 gap robust-z | 0.010 | 0.014 | 0.043 | 0.081 |
| R4 | R4 per-entity HBOS | 0.005 | 0.005 | 0.019 | 0.046 |
| R5 | R5 guarded HBOS (rules share budget) | 0.000 | 0.000 | 0.000 | 0.001 |
| R6 | R6 per-entity LOF | 0.011 | 0.013 | 0.056 | 0.138 |
| R7 | R7 always fast | 1.000 | 1.000 | 1.000 | 1.000 |
| R8 | R8 guarded HBOS (rules outside budget) | 0.000 | 0.000 | 0.000 | 0.001 |
| R9 | R9 = R8 + any-key guards | 0.000 | 0.000 | 0.000 | 0.001 |

## Guard funnel (rule-negative)

| policy | budget_pct | rule_negative_events | eligible | admitted | positives_among_admitted | positives_among_eligible |
|---|---|---|---|---|---|---|
| R5 | 1 | 384413 | 118 | 102 | 0 | 0 |
| R5 | 2 | 384413 | 256 | 256 | 0 | 0 |
| R5 | 5 | 384413 | 851 | 851 | 0 | 0 |
| R5 | 10 | 384413 | 2792 | 2792 | 1 | 1 |
| R8 | 1 | 384413 | 118 | 118 | 0 | 0 |
| R8 | 2 | 384413 | 256 | 256 | 0 | 0 |
| R8 | 5 | 384413 | 851 | 851 | 0 | 0 |
| R8 | 10 | 384413 | 2792 | 2792 | 1 | 1 |
| R9 | 1 | 384413 | 118 | 118 | 0 | 0 |
| R9 | 2 | 384413 | 256 | 256 | 0 | 0 |
| R9 | 5 | 384413 | 851 | 851 | 0 | 0 |
| R9 | 10 | 384413 | 2796 | 2796 | 1 | 1 |

## Per-source recall@2 % (rule-negative)

| policy | github | jira | mail |
|---|---|---|---|
| R0 | 0.014 | 0.020 | 0.030 |
| R1 | 0.000 | 0.000 | 0.000 |
| R2 | 0.000 | 0.107 | 0.000 |
| R3 | 0.014 | 0.000 | 0.164 |
| R4 | 0.043 | 0.000 | 0.015 |
| R5 | 0.000 | 0.000 | 0.000 |
| R6 | 0.000 | 0.007 | 0.090 |
| R7 | 1.000 | 1.000 | 1.000 |
| R8 | 0.000 | 0.000 | 0.000 |
| R9 | 0.000 | 0.000 | 0.000 |

## Lateness, timestamped subject-separated situations (rule-negative, 2 %)

| policy | n_situations | affiliation_precision | affiliation_recall | delay_p50_s | delay_p95_s | detected_rate |
|---|---|---|---|---|---|---|
| R0 | 255.000 | 0.673 | 0.278 | 100.231 | 51699.762 | 0.067 |
| R1 | 255.000 | 0.730 | 0.592 | 30618.063 | 58983.980 | 0.027 |
| R2 | 255.000 | 0.760 | 0.399 | 0.000 | 0.000 | 0.294 |
| R3 | 255.000 | 0.609 | 0.098 | 138.000 | 3993.243 | 0.051 |
| R4 | 255.000 | 0.476 | 0.054 | 709.000 | 22588.000 | 0.016 |
| R5 | 255.000 | 0.730 | 0.592 | 30618.063 | 58983.980 | 0.027 |
| R6 | 255.000 | 0.693 | 0.159 | 109.500 | 43360.870 | 0.039 |
| R7 | 255.000 | 0.652 | 0.988 | 0.000 | 0.000 | 1.000 |
| R8 | 255.000 | 0.730 | 0.592 | 30618.063 | 58983.980 | 0.027 |
| R9 | 255.000 | 0.730 | 0.592 | 30618.063 | 58983.980 | 0.027 |

## Label diagnostics

| function | accuracy | accuracy_at_cap | coverage | overlap | conflict | positive_votes | negative_votes | positive_rate | unknown |
|---|---|---|---|---|---|---|---|---|---|
| jira_committer_comment_1h | 0.7759 | False | 0.3684 | 0.3684 | 0.4026 | 32758 | 112666 | 0.2253 | 249317 |
| jira_priority_raised_later | 0.9500 | True | 0.3684 | 0.3684 | 0.4026 | 6325 | 139100 | 0.0435 | 249316 |
| jira_resolved_24h | 0.8650 | False | 0.3683 | 0.3683 | 0.4026 | 19400 | 125989 | 0.1334 | 249352 |
| jira_linked_pr_24h | 0.9171 | False | 0.3683 | 0.3683 | 0.4026 | 12058 | 133331 | 0.0829 | 249352 |
| declared_blocker_critical | 0.9500 | True | 0.3684 | 0.3684 | 0.4026 | 7937 | 137488 | 0.0546 | 249316 |
| dev_committer_reply_1h | 0.9354 | False | 0.0895 | 0.0895 | 0.1319 | 2451 | 32860 | 0.0694 | 359430 |
| dev_three_responders_2h | 0.9500 | True | 0.0895 | 0.0895 | 0.1319 | 545 | 34765 | 0.0154 | 359431 |
| dev_cve_blocker_later | 0.9264 | False | 0.0895 | 0.0895 | 0.1319 | 2620 | 32691 | 0.0742 | 359430 |
| github_reverted_48h | 0.9492 | False | 0.5412 | 0.5412 | 0.0530 | 11009 | 202629 | 0.0515 | 181103 |
| github_hotfix_reference_24h | 0.9500 | True | 0.5417 | 0.5412 | 0.0530 | 452 | 213367 | 0.0021 | 180922 |

## Metric remediation record

| metric | design | floor | mean_R0 | mean_R5 | mean_R8 | mean_R7 | paired_months | design_minus_floor | standard_error | decision | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| precision_at_b | R8 | R0 | 0.00191 | 0.00000 | 0.00000 | 0.00229 | 31 | -0.00195 | 0.00082 | kept | design separated from random floor by > 2 paired SE |
| vus_pr | R8 | R0 | 0.00526 | 0.00523 | 0.00523 | 0.00475 | 48 | -0.00003 | 0.00024 | dropped | design not separated from the random floor (|mean| <= 2 SE or < 10 months); reported, not interpreted |
| affiliation_precision_rowindex | R8 | R0 | 0.50094 | 0.53532 | 0.53532 | 0.50965 | 29 | 0.03276 | 0.03122 | dropped | design not separated from the random floor (|mean| <= 2 SE or < 10 months); reported, not interpreted |
| affiliation_recall_rowindex | R8 | R0 | 0.65847 | 0.07228 | 0.07228 | 1.00000 | 48 | -0.58619 | 0.02665 | kept | design separated from random floor by > 2 paired SE |
| nab_low_fn_rowindex | R8 | R0 | -4.49335 | -1.93145 | -1.93145 | -145.14033 | 52 | 2.56190 | 0.51137 | kept | design separated from random floor by > 2 paired SE |

## Validity gates not passed

| gate | status | required | reason |
|---|---|---|---|
| random_vus_at_prevalence | fail | True | uniform random same-geometry VUS-PR must be within 50% (relative) of prevalence |
| rule_floor_budget_feasible | fail | True | months whose rule floor exceeds capacity are invalid, never over-admitted |
| injected_non_trivial | not_applicable | False | Corpus-S R1 recall@2% must not exceed 0.9 |
| human_sanity_100_two_annotators | not_applicable | True | 100 top rule-negative items require two annotators and reported kappa |
| historical_text_provenance | not_applicable | False | rule/feature/label text is export-time snapshot text (review finding 7); components are replayed as-of-event, text edits cannot be; declared limitation |
| human_sanity_sample_written | not_applicable | False | blind annotator sheet for the 100-item human sanity check; annotate then rerun with human_sanity meta |
| corpus_preflight | fail | True | every declared corpus profile requirement must run |
| harm_store_s3 | not_applicable | False | no real action provider / no S3 store in this profile |
| harm_minimum_five_promotions | not_applicable | False | no real action provider / no S3 store in this profile |
| harm_paired_coverage | not_applicable | False | no real action provider / no S3 store in this profile |
| harm_leakage | not_applicable | False | no real action provider / no S3 store in this profile |
| harm_non_vacuous | not_applicable | False | no real action provider / no S3 store in this profile |
| harm_real_provider | not_applicable | False | no real action provider / no S3 store in this profile |
| excluded_label_functions | nan | False | labeling functions dropped by registered amendment (config e1.exclude_label_functions) |
| valid | fail | True | failed required E1 validity items: ['random_vus_at_prevalence (fail): uniform random same-geometry VUS-PR must be within 50% (relative) of prevalence', 'rule_floor_budget_feasible (fail): months whose rule floor exceeds capacity are invalid, never over-admitted', 'human_sanity_100_two_annotators (not_applicable): 100 top rule-negative items require two annotators and reported kappa', 'corpus_preflight (fail): every declared corpus profile requirement must run'] |

## LaTeX

```latex
\begin{table}[t]
\centering
\small
\begin{tabular}{lrrrr}
\toprule
policy & 1 \% & 2 \% & 5 \% & 10 \% \\
\midrule
R0 & 0.011 & 0.020 & 0.055 & 0.098 \\
R1 & 0.000 & 0.000 & 0.000 & 0.000 \\
R2 & 0.045 & 0.089 & 0.137 & 0.166 \\
R3 & 0.010 & 0.014 & 0.043 & 0.081 \\
R4 & 0.005 & 0.005 & 0.019 & 0.046 \\
R5 & 0.000 & 0.000 & 0.000 & 0.001 \\
R6 & 0.011 & 0.013 & 0.056 & 0.138 \\
R7 & 1.000 & 1.000 & 1.000 & 1.000 \\
R8 & 0.000 & 0.000 & 0.000 & 0.001 \\
R9 & 0.000 & 0.000 & 0.000 & 0.001 \\
\bottomrule
\end{tabular}
\caption{Recall at budget on rule-negative revealed-urgent events (Apache Kafka).}
\label{tab:e1-recall}
\end{table}

\begin{table}[t]
\centering
\small
\begin{tabular}{lrrrr}
\toprule
contrast & effect & ci\_low & ci\_high & subject\_clusters \\
\midrule
R5-R1 & 0.000 & 0.000 & 0.000 & 219 \\
R5-R2 & -0.089 & -0.111 & -0.070 & 219 \\
R8-R2 & -0.089 & -0.111 & -0.070 & 219 \\
R9-R2 & -0.089 & -0.111 & -0.070 & 219 \\
R8-R5 & 0.000 & 0.000 & 0.000 & 219 \\
R2-R0 & 0.069 & 0.047 & 0.093 & 219 \\
R4-R0 & -0.015 & -0.028 & -0.006 & 219 \\
R8-R0 & -0.020 & -0.033 & -0.012 & 219 \\
R9-R0 & -0.020 & -0.033 & -0.012 & 219 \\
\bottomrule
\end{tabular}
\caption{Registered paired contrasts of recall@2\% (BCa bootstrap, subject clusters).}
\label{tab:e1-contrasts}
\end{table}

\begin{table}[t]
\centering
\small
\begin{tabular}{lrrrrrr}
\toprule
policy & budget\_pct & rule\_negative\_events & eligible & admitted & positives\_among\_admitted & positives\_among\_eligible \\
\midrule
R5 & 1 & 384413 & 118 & 102 & 0 & 0 \\
R5 & 2 & 384413 & 256 & 256 & 0 & 0 \\
R5 & 5 & 384413 & 851 & 851 & 0 & 0 \\
R5 & 10 & 384413 & 2792 & 2792 & 1 & 1 \\
R8 & 1 & 384413 & 118 & 118 & 0 & 0 \\
R8 & 2 & 384413 & 256 & 256 & 0 & 0 \\
R8 & 5 & 384413 & 851 & 851 & 0 & 0 \\
R8 & 10 & 384413 & 2792 & 2792 & 1 & 1 \\
R9 & 1 & 384413 & 118 & 118 & 0 & 0 \\
R9 & 2 & 384413 & 256 & 256 & 0 & 0 \\
R9 & 5 & 384413 & 851 & 851 & 0 & 0 \\
R9 & 10 & 384413 & 2796 & 2796 & 1 & 1 \\
\bottomrule
\end{tabular}
\caption{Guard funnel for the guarded conditions on rule-negative events.}
\label{tab:e1-funnel}
\end{table}

```
