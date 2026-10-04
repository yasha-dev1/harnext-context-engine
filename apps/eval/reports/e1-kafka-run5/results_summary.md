# E1 run 5 — Jev routing arms

Run `apps/eval/out/20261004T115316Z-e1-kafka-jev-only/e1/seed-1`; run-4 arms from `apps/eval/out/20260907T000315Z-e1-kafka/e1/seed-1` (R2 identical across runs: True).
Required gates not passed (all pre-existing, identical to run 4): corpus_preflight, human_sanity_100_two_annotators, random_vus_at_prevalence, rule_floor_budget_feasible; Jev gates: jev_decided_after_registration pass, jev_decisions_complete pass, jev_single_model pass

Jev: 394741 decisions, failed 0, $18.086511 total, models ['typesafe/jev-1.13-20260917'], prompt se-triage-v2 `5d4d9014d248`.

## Recall on rule-negative urgent events (pooled)

| arm | 1 % | 2 % | 5 % | 10 % |
|---|---|---|---|---|
| R0 random | 0.011 | 0.020 | 0.055 | 0.098 |
| R2 global HBOS | 0.045 | 0.089 | 0.137 | 0.166 |
| R4 per-entity HBOS | 0.005 | 0.005 | 0.019 | 0.046 |
| R5 guarded HBOS | 0.000 | 0.000 | 0.000 | 0.001 |
| R10 per-source HBOS | 0.027 | 0.058 | 0.128 | 0.181 |
| R11 IsolationForest | 0.035 | 0.072 | 0.117 | 0.154 |
| R12 ECOD | 0.023 | 0.044 | 0.108 | 0.154 |
| R13 global LOF | 0.006 | 0.014 | 0.044 | 0.080 |
| R14 Jev SE-category router | 0.083 | 0.139 | 0.250 | 0.433 |
| R15 Jev urgency | 0.089 | 0.155 | 0.280 | 0.440 |

## Paired contrasts at 2 % (BCa, subject clusters; month clusters as sensitivity)

| contrast | effect | 95 % CI subjects | 95 % CI months | cross-run |
|---|---|---|---|---|
| R14-R0 | +0.119 | [+0.085, +0.161] | [+0.082, +0.166] |  |
| R14-R2 | +0.050 | [+0.015, +0.095] | [+0.003, +0.099] |  |
| R14-R10 | +0.081 | [+0.049, +0.123] | [+0.043, +0.129] | yes |
| R14-R11 | +0.068 | [+0.034, +0.110] | [+0.027, +0.118] | yes |
| R14-R15 | -0.015 | [-0.038, +0.011] | [-0.045, +0.011] |  |
| R15-R0 | +0.135 | [+0.100, +0.177] | [+0.096, +0.177] |  |
| R15-R2 | +0.066 | [+0.033, +0.106] | [+0.023, +0.110] |  |
| R15-R10 | +0.097 | [+0.065, +0.136] | [+0.060, +0.139] | yes |
| R15-R11 | +0.083 | [+0.051, +0.123] | [+0.042, +0.124] | yes |

## By source, recall at 2 %

| source | positives | R0 | R2 | R4 | R5 | R10 | R11 | R12 | R13 | R14 | R15 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| github | 69 | 0.014 | 0.000 | 0.043 | 0.000 | 0.014 | 0.000 | 0.000 | 0.000 | 0.145 | 0.232 |
| jira | 703 | 0.020 | 0.107 | 0.000 | 0.000 | 0.068 | 0.085 | 0.053 | 0.016 | 0.125 | 0.148 |
| mail | 67 | 0.030 | 0.000 | 0.015 | 0.000 | 0.000 | 0.000 | 0.000 | 0.015 | 0.284 | 0.149 |

## By year (memorisation check), recall at 2 %

| year | positives | R0 | R2 | R4 | R5 | R10 | R11 | R12 | R13 | R14 | R15 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | 159 | 0.013 | 0.050 | 0.000 | 0.000 | 0.044 | 0.075 | 0.031 | 0.019 | 0.101 | 0.145 |
| 2023 | 149 | 0.000 | 0.114 | 0.007 | 0.000 | 0.094 | 0.067 | 0.054 | 0.007 | 0.087 | 0.121 |
| 2024 | 313 | 0.032 | 0.099 | 0.003 | 0.000 | 0.058 | 0.077 | 0.064 | 0.013 | 0.141 | 0.144 |
| 2025 | 163 | 0.031 | 0.098 | 0.012 | 0.000 | 0.049 | 0.061 | 0.018 | 0.018 | 0.166 | 0.160 |
| 2026 | 55 | 0.000 | 0.055 | 0.000 | 0.000 | 0.036 | 0.073 | 0.018 | 0.018 | 0.309 | 0.327 |

## Unbudgeted operating point

| rule | admission_rate | recall_rule_negative | precision_rule_negative | prevalence_rule_negative |
|---|---|---|---|---|
| argmax category in fast set | 0.0507 | 0.2682 | 0.0120 | 0.0022 |
| urgency >= 0.5 | 0.0599 | 0.3278 | 0.0126 | 0.0022 |

## Category mix

| category | all_events | rule_negative_positives | r14_admitted_2pct | precision_rule_negative | fast_lane |
|---|---|---|---|---|---|
| code_review | 0.3801 | 0.0524 | 0.0000 | 0.0003 | False |
| bookkeeping | 0.3715 | 0.5256 | 0.0000 | 0.0031 | False |
| docs_or_chore | 0.0585 | 0.0226 | 0.0000 | 0.0008 | False |
| feature_work | 0.0580 | 0.0346 | 0.0000 | 0.0013 | False |
| minor_bug | 0.0467 | 0.0608 | 0.0000 | 0.0028 | False |
| design_discussion | 0.0312 | 0.0298 | 0.0000 | 0.0021 | False |
| regression_or_outage | 0.0274 | 0.1967 | 0.5052 | 0.0161 | True |
| build_break | 0.0152 | 0.0381 | 0.2849 | 0.0055 | True |
| release_risk | 0.0051 | 0.0334 | 0.1350 | 0.0165 | True |
| user_question | 0.0034 | 0.0060 | 0.0000 | 0.0038 | False |
| security | 0.0029 | 0.0000 | 0.0750 | 0.0000 | True |
