---
title: "Evaluation Specification — Two-Lane Streaming Context Engine"
subtitle: "Everything the implementer needs to build and run the evaluation system"
date: 2026-08-30
---

> **Status.** Implementer-facing specification, v1. Supersedes every earlier evaluation note in this repository (`research/evaluation-methodology.md`, `research/are-time-aware-evaluation.md`, the *Evaluation* section of `proposal.md`). Nothing below depends on SWE-bench, SPARK, ProAgentBench or MCP.
> **Companions.** The literature behind each choice is in the *Two-Lane Context Engine Research Brief*; the defense Q&A is in the *Evaluation Protocol* artifact. This file is the build spec.

## 0. Purpose and how to read this

The system under test (SUT) ingests a high-throughput stream of heterogeneous company events (chat, issue tracker, code activity, payments, tickets, clicks), keyed by entity. A **router** promotes a small budgeted share of events to a **fast lane**, where an AI harness acts immediately; everything else is folded on a **batch lane**, per entity and per window, into a maintained **per-entity state** (markdown files in a versioned filesystem). The batch lane is the *write path*; the fast lane is the *read-and-act path*.

The evaluation answers six claims with six experiments (E1–E6), on two corpora, with ground truth that is always either **derived from the log** or **true by construction**. Read §1–§5 once (they define shared vocabulary and infrastructure), then treat each experiment in §7 as a self-contained runbook.

Everything the implementer builds lives under `eval/` in the harnext repository (§11).

---

## 1. Claims under test

| ID | Claim | Falsified if | Primary experiment |
|---|---|---|---|
| C1 Routing | A budgeted router (rules floor + per-entity deviation + guards) admits high-cost-of-delay events better than equally budgeted baselines, and promoting them changes the outcome. | Rules-only or random-at-budget matches it on the rule-negative subset; or promoted events are handled no better now than at window close. | E1 |
| C2 Fidelity | The batch lane maintains per-entity state that is correct and current with respect to facts re-derivable from the log. | Probe accuracy no better than reading the raw last-*N* events; supersession errors common. | E2 |
| C3 Organisation | Organising the store (per-entity structure, curation, cross-links) yields more correct context per token than unorganised alternatives, and does not erode over weeks of stream. | A code-templated store or a vector index over raw events ties at equal read budget; or health degrades until probes fail. | E3 |
| C4 Envelope | The composition of the context handed to the agent changes action quality; a small structured envelope beats both nothing and everything. | "Everything" wins outright, or envelope choice makes no difference. | E4 |
| C5 Cadence | Per-window folding amortises cost without losing fidelity; the fast lane buys freshness only where budgeted. | Per-event and per-window cost the same, or folding loses fidelity. | E5 |
| C6 System | The pipeline sustains target throughput with fast-lane tail latency under SLO during bursts; guards stop the fast lane from amplifying a burst. | Fast-lane p99 collapses with aggregate load; guard removal changes nothing. | E6 |

---

## 2. Principles (every experiment obeys these)

1. **Ground truth is derived or constructed, never opined.** LLM judges are used only for free-text similarity, always position-swapped and κ-checked against a human sample.
2. **One variable per experiment.** Separate baselines per experiment; failures are localisable.
3. **Everything is paired on a frozen replay.** Identical stream, identical order; a leakage gate proves no post-T event reached the store; comparisons are paired differences with entity-clustered CIs.
4. **Two corpora for two reasons.** Real (external validity) and simulated (sources no public data provides; exact state; injected situations; arbitrary load). A claim counts only if it holds on both where both apply.
5. **Cost is a result.** Every quality number is reported next to tokens, dollars and latency.
6. **Pre-registration.** `eval/PREREG.md` is committed before the first full run and lists every condition, primary metric, threshold and exclusion rule. Amendments are appended with dates, never edited in place.

---

## 3. Corpora

### 3.1 Corpus R — Apache Kafka (real, live, multi-source)

**Sources and extractors**

| Source | Extractor | Event types produced | Notes |
|---|---|---|---|
| `dev@kafka.apache.org` | `eval/corpus/pony_mail.py` (Pony Mail `mbox.lua` monthly archives; `stats.lua` for counts) | `org.apache.mail.message` with `in_reply_to`, `thread_root`, `subject_tags` (`[VOTE]`, `[DISCUSS]`, `KIP-N`, `KAFKA-N`) | `users@` is ~40 msg/month → excluded |
| KAFKA JIRA | `eval/corpus/jira.py` (public REST, `expand=changelog`, comments) | `org.apache.jira.issue.created`, `.transition` (one per changelog item), `.comment` | changelog items are the supersession ground truth |
| GitHub `apache/kafka` | `eval/corpus/gharchive.py` (GH Archive hourly files, filtered by repo) | `com.github.pull_request.{opened,merged,closed}`, `.review`, `.review_comment`, `.issue_comment`, `.push` | GH Archive has no stated data licence; store only IDs and derived fields beyond the replay |
| kafka.apache.org docs | existing `sitemap` connector | `com.web.page` with `lastmod` | small; gives the store something non-event to cite |

**Windows**

| Window | Use | Why |
|---|---|---|
| **R-H1**: 2026-01-01 → 2026-06-30 | All LLM experiments (E2–E5). Jan–Apr = warm-up (store is built), May–Jun = probe period. | Post-training-cutoff for the models used, so the model-prior arm is meaningful. |
| **R-long**: 2022-01-01 → 2026-06-30 | E1 only (no LLM in the loop, so contamination is irrelevant). Rolling month-ahead evaluation. | 66 Blocker/Critical issues per half-year is too few; the long window gives ≈ 600 declared-critical and thousands of revealed-urgent events. |

**Measured volume (R-H1, queried 2026-08-30)**

| Stream | Count |
|---|---|
| JIRA issues created / touched / resolved | 706 / 969 / 416 |
| JIRA Blocker + Critical created | 66 |
| JIRA changelog + comment activity | ≈ 15,000 (1.8–3.7 k / month) |
| `dev@` messages | ≈ 4,000 (560–760 / month) |
| Commits | ≈ 1,000 |
| PRs created / merged | 1,480 / 1,052 (+ review comments ≈ 10–15 k) |
| **Total** | **≈ 35–45 k events, ≈ 250 / day** |

Flink is the replication project (1,180 issues, 61 Blocker/Critical, 1,220 PRs, dev@ ≈ 450 / month in the same window) — used for E1 and, if budget remains, one E3 store.

**Entity keys (`subject`)**

| Key | Example | Lifetime | Used for |
|---|---|---|---|
| `issue:KAFKA-N` | `issue:KAFKA-19876` | days–weeks | state (E2–E5) |
| `kip:N` | `kip:1150` | weeks–months | state |
| `pr:N` | `pr:20412` | days | state |
| `thread:<root-message-id>` | `thread:CA+abc@mail` | days | state; routing baseline |
| `contributor:<sha256(email)[:12]>` | `contributor:9f1c2a…` | years | **routing baselines (E1)**, state |
| `component:<jira component>` | `component:streams` | years | **routing baselines (E1)**, state |

Anomaly baselines must be keyed on long-lived entities (`contributor`, `component`, `thread`); `issue`/`pr` are too short-lived and would leave the deviation layer permanently cold. The event's `subject` is the entity the state is about; the router additionally computes `baseline_keys[]` from the event (author, component) and scores against each, taking the max.

### 3.2 Corpus S — OrgForge (simulated, ground truth by construction)

OrgForge (MIT, arXiv 2603.14997) runs a deterministic engine that owns every fact and emits an append-only ISO-timestamped **SimEvent** bus: Slack channels/DMs, Jira tickets, GitHub PRs and reviews, Confluence, Zendesk tickets, Salesforce records, invoices, NPS, Datadog metrics/alerts. An LLM writes prose only.

**Extensions the implementer adds**

| Extension | Module | What it does |
|---|---|---|
| Clickstream | `eval/corpus/orgforge_clicks.py` | Per-customer sessions of `view / cart / purchase / support_page` events; inter-arrival, session length and conversion fitted from the REES46 public dataset; customers are OrgForge's Salesforce accounts. |
| Payments | `eval/corpus/orgforge_stripe.py` | Stripe-shaped events (`invoice.paid/payment_failed`, `charge.refunded`, `charge.dispute.created/updated/closed`) generated from OrgForge invoices, with Stripe's documented dispute lifecycle timings; optionally validated against Stripe sandbox fixtures. |
| Injected situations | `eval/corpus/orgforge_inject.py` | ≥ 4 archetypes × ≥ 50 each with onset time and cost weight: VIP dispute, production incident with paging, churn signal (downgrade after failed invoice), security report. Hard negatives: benign volume spikes and chit-chat bursts on the same entities. |
| World-state dump | `eval/corpus/orgforge_state.py` | Hourly JSON snapshot of every entity's true state (plan, open tickets, unpaid invoices, incident status, owner). |

Scale target: ≥ 90 simulated days, ≥ 20 k events. Seeds recorded.

### 3.3 Replay file, clock and snapshots

One JSONL per corpus/window: `eval/replay/<corpus>-<window>.jsonl`, one CloudEvent per line, sorted by event time, SHA-256 recorded in `PREREG.md`.

```jsonc
{ "specversion":"1.0", "id":"…", "source":"jira:KAFKA", "type":"org.apache.jira.issue.transition",
  "subject":"issue:KAFKA-19876", "time":"2026-05-03T10:12:44Z",
  "mgtenant":"kafka", "baseline_keys":["contributor:9f1c2a…","component:streams"],
  "data":{ "field":"status", "from":"Open", "to":"In Progress", "actor":"contributor:9f1c2a…" } }
```

`eval/replay.py` produces the file to `cms.events.raw.v1` at a chosen speed-up (60× for builds; open-loop fixed rates for E6), stamping `intended_send_ts`. It can **stop at a cutoff T**. The store is git-backed; after each fold the commit SHA is appended to `eval/replay/snapshots-<store>.csv` as `(T_last_event, sha, last_event_id, lane)`. `snapshot(T)` = the last commit whose `T_last_event ≤ T`.

---

## 4. Ground truth catalogue

| Quantity | Corpus R (derived) | Corpus S (constructed) | Used by |
|---|---|---|---|
| High cost-of-delay event | **Revealed urgency** (§4.1): label model over post-*t* outcomes | Injected situations with onset and cost weight; hard negatives | E1 |
| Entity state as of T | JIRA changelog replayed to T (status, assignee, priority, fixVersion, components); PR merged/closed; KIP vote outcome; thread answered by whom | World-state dump at T | E2 E3 E5 |
| Cross-source link | Regex join on mandated keys (`KAFKA-N` in commits/threads/PR titles; `KIP-N` in threads/PRs); join precision/recall reported | Event-graph edges | E2 E3 |
| Supersession | Every changelog transition = a superseded value with timestamp | Every state mutation | E2 E3 |
| Gold action | Human next step after T: assignee/component/duplicate-of/priority set by a human within 24 h; first committer reply; reviewer of the linked PR | Scripted correct handling per situation (deterministic rule over world state); fact coverage against world state | E4, E1 harm check |
| Code location | Files (and modules = first two path segments) changed by the PR(s) whose title carries the issue key, merged within 14 d of T; union over PRs; formatting-only PRs excluded | n/a | E4 localisation, E2 multi-source (code) |
| Freshness | Event time → first snapshot whose diff references the event id | same | E5 |
| Latency | `intended_send_ts` → fast-lane agent-start; window close → fold commit | same | E6 |

### 4.1 Revealed urgency (Corpus R, E1)

Labelling functions, each computed **only from events after *t***:

| Source | Labelling function |
|---|---|
| JIRA | committer comment within 1 h; priority raised later; fixVersion set to the in-flight release; resolved within 24 h; linked PR opened within 24 h; priority ∈ {Blocker, Critical} (declared — kept as one noisy function among many) |
| `dev@` | first committer reply within 1 h; ≥ 3 distinct responders within 2 h; subject `[VOTE]` cancelled/re-cast; CVE or "blocker" in thread |
| GitHub | PR reverted within 48 h; hotfix PR referencing it within 24 h; trunk CI failure followed by fix commit within 6 h |

Fused with a Snorkel-style label model (`eval/e1/labels.py`) → probabilistic label per event plus each function's estimated accuracy and coverage (both reported).

**Temporal firewall.** Router features use only data ≤ *t*; labels use only data > *t*. **No circularity:** stream statistics (bursts, gaps, type mix) are features and may never be labels; labels are human reactions or consequences.

### 4.2 Leakage gate (every probe and task)

```
assert max(event_time for e in delivered_to_builder_before(sha)) <= T
assert no token of probe.question occurs only in events with time > T
assert gold_action.time > T and gold_action not in any envelope
log probe_id, T, sha, last_event_id, PASS|FAIL   → eval/out/gate.csv
```

A failure excludes the item and is counted; the count is reported.

---

## 5. Shared infrastructure (build once)

| Component | Path | Responsibility |
|---|---|---|
| Replay producer | `eval/replay.py` | JSONL → Kafka at speed-up or open-loop rate; cutoff; `intended_send_ts` |
| Snapshot index | `eval/snapshots.py` | `snapshot(T)` over the git-backed store; materialise to a temp dir |
| Probe generators | `eval/probes/gen_{extraction,temporal,update,multisource,abstention}.py` | emit `probes/<corpus>.jsonl`: `{probe_id, family, entity, T, question, gold, gold_type, superseded_values[], source_event_ids[]}` |
| Graders | `eval/grade/exact.py`, `claims.py`, `links.py`, `action.py` | deterministic normalised exact match; claim-level precision/recall; link set P/R; action composite |
| Store health | `eval/health/store_health.py` | file/byte counts; over-cap files (> 200 lines); INDEX resolution rate; dangling cross-refs; MinHash near-duplicate facts; superseded-value leakage |
| Read agent | `eval/agents/reader.py` | fixed model/prompt; token-budgeted material; "answer only from material; UNKNOWN if absent; cite IDs" |
| Envelope builder | `eval/agents/envelope.py` | V0–V8 from a snapshot; per-section token log |
| Label model | `eval/e1/labels.py` | labelling functions + label model + diagnostics |
| Router harness | `eval/e1/policies.py` | R0–R6 policies over the replay; per-event score/lane/decision-time log |
| Load generator | `eval/e6/loadgen.py` | Pareto ON/OFF per entity, Zipf popularity, calibrated burstiness B; runs on a separate host |
| Stats | `eval/stats.py` | paired entity-clustered bootstrap, McNemar, Holm, power |
| Pre-registration | `eval/PREREG.md` | frozen decisions, hashes, thresholds, model IDs, prices |

Store variants are built by `eval/stores/build_<S0|S1|S2|S3|S4|S5>.py`; S0 uses the `fake` harness, S1 is pure code, S4/S5 use a pinned embedding model.

---

## 6. Phases

| Phase | Weeks | Deliverables | Gate |
|---|---|---|---|
| **0 — Foundations** | 1–2 | Fix the two pre-run bugs (§12 D12). `PREREG.md`. Corpus R extractors → `replay/kafka-H1.jsonl` and `replay/kafka-long.jsonl` (hashes in PREREG). Probe generators, graders, leakage gate, `store_health.py` with unit tests on planted defects. Pilot of 30 probes graded by two humans. | **G1** |
| **1 — Baselines and system** | 3–6 | E6 with the fake harness. E1 on R-long (no LLM) incl. calibration curve. Stores S0, S1, S4 → E2. Then S3 × 3 seeds (+ Opus tier on seed 1) with health checkpoints → E2, E3. | **G2** after baselines, **G3** after S3 seed 1 |
| **2 — Simulated corpus and agent-facing evals** | 7–9 | OrgForge + extensions → `replay/orgforge.jsonl`, probes, injected situations, world-state dumps (G1 repeated on S). E1 exact on S. E2/E3-S3 on S. E4 on both corpora; E1 harm check. E5 cadence sweep. S2/S5 if budget remains. | **G4** before E4/E5 |
| **3 — Audit and write-up** | 10 | Human audit (200 probes, 100 actions, two annotators, κ). Analysis notebooks, figures, tables, reproducibility package. | — |

**Order rationale:** cheapest real results first (E6, E1 need no LLM), baselines before the expensive curated stores, simulated corpus only once the real one has validated the tooling.

---

## 7. Experiments

Each experiment lists: type of evaluation · population · conditions (the variations) · procedure · measurements · baselines and floors · sample · validity checks · outputs.

### E1 — Routing under a budget

**Type.** Offline ranking evaluation with weak / constructed labels; no LLM in the loop (except the harm check).

**Population.** Corpus R-long (Kafka 2022-01 → 2026-06, plus Flink replication), rolling month-ahead: for each month *m* ≥ 2022-03, tune θ on months < *m*, evaluate on *m*. Corpus S: all events, injected situations as positives.
The **primary population is the rule-negative subset** (events the rules floor does not flag) — the only place the deviation layer can add value. Full-population numbers are reported too.

**Conditions (router policies), each at admission budgets b ∈ {1, 2, 5, 10} %:**

| Policy | Definition | Isolates |
|---|---|---|
| R0 | random at budget | metric floor |
| R1 | rules only (declared Blocker/Critical; `[VOTE]`, `CVE`, "blocker"; dispute ≥ N; on-call page) | the floor's own value |
| R2 | global z-score / global HBOS over the feature vector, no entity keying | necessity of per-entity keying |
| R3 | per-entity robust-z on inter-arrival gap only (current implementation) | single-feature baseline |
| R4 | per-entity HBOS over the full feature vector, no guards | the scorer alone |
| R5 | **rules + per-entity HBOS + absolute-volume floor + multi-window confirmation** (ours) | the design |
| R6 | LOF / kNN per entity (local-density reference) | ceiling for local anomalies |
| R7 | always-fast | cost ceiling |

Feature vector (computed from data ≤ *t*, per `baseline_keys`): inter-arrival gap (log), 5-min and 1-h event counts as ratios to baseline, event-type mix divergence (JS), actor novelty, subject/thread novelty, priority field (if any), money field (if any), time-of-day bucket. Robust-z uses median/MAD over a 4-week rolling window in 5-min buckets.

**Procedure.**

1. Build labels (`eval/e1/labels.py`) on R-long; on S read injected labels. Store per-event `y_hat`, per-function accuracy/coverage.
2. For each policy and budget, replay the evaluation months in order; log `{event_id, score, lane, decision_ts, baseline_key_used, features_fired}`.
3. Score (`eval/e1/score.py`).
4. Robustness: jitter S onsets ±5 min; flip 10 % of R labels; re-score. Run uniform-random and always-flag scorers through every metric.
5. Harm check (Phase 2, needs store S3): for each R5-promoted event, run the fast-lane task at admission and again at that entity's next window close; grade with `grade/action.py`; paired delta.

**Measurements.**

| Metric | Definition | Role |
|---|---|---|
| recall@budget | share of positives admitted at budget b, per source | **primary** (b = 2 %, rule-negative subset, R5 vs R1 and R5 vs R2) |
| precision@budget | share of admissions that are positive | secondary |
| VUS-PR | volume under the PR surface with tolerance buffers (reference implementation) | secondary, threshold-free |
| affiliation P/R | time-distance-based per situation | secondary (lateness) |
| NAB score, low-FN profile | early-detection reward | secondary |
| detection delay | onset → first fast admission on the entity, p50/p95 | secondary |
| **calibration curve** | revealed-urgency rate by score decile | secondary; the "real-world" figure |
| lift over rules | positive rate among deviation admissions ÷ positive rate among rule-negatives | secondary |
| harm delta | action-quality(now) − action-quality(window close), paired | secondary |
| feature attribution | HBOS terms firing on true positives; 10 case studies | qualitative |

Never point-adjusted F1.

**Exact scores.** With `admitted_b(e)` = the top *b* % of scores within the evaluation month and `y(e)` the label: `recall@b = |{e : admitted_b(e) ∧ y(e)=1}| / |{e : y(e)=1}|` (**primary**, b = 2 %, rule-negative subset, R5 − R1 and R5 − R2); `precision@b = |{e : admitted_b(e) ∧ y(e)=1}| / |{e : admitted_b(e)}|`; VUS-PR and affiliation from the reference implementations; `delay = t_first_admission(entity) − t_onset`; calibration `rate(d) = mean(y(e) : e ∈ decile d)` with Spearman ρ(d, rate); `lift = rate(deviation admissions) / rate(all rule-negatives)`; `harm_delta = Q_action(now) − Q_action(window close)` using the E4 composite.

**Baselines and floors.** R0, R1, R7; random and always-flag scorers.

**Sample.** R-long: ≈ 350 k events, ≈ 600 declared-critical, thousands of revealed positives; per-month evaluation gives ≥ 50 evaluation months. S: ≥ 200 situations × 3 seeds.

**Validity checks.**

- Label-model diagnostics: no function with estimated accuracy < 0.6 or coverage < 1 %; report agreement between outcome-based functions and the declared-priority function (this is the thesis's own measurement of label noise).
- Random scorer's precision@budget ≈ prevalence and VUS-PR ≈ prevalence; always-flag recall = 1, precision = prevalence. Any metric that ranks either near R5 is dropped.
- Injected situations non-trivial: R1 must not reach recall@2 % > 0.9 on them; otherwise regenerate with subtler features.
- θ tuned only on months < *m*; `PREREG.md` commit predates first evaluation run.
- Metric implementations validated on three hand-built toy series with known VUS/affiliation values.
- Human sanity sample: 100 top-scored rule-negative events, two annotators answer "would you want to be interrupted for this?"; report κ vs score bucket.

**Outputs.** `eval/out/e1/scores.parquet`, `metrics.csv`, `calibration.png`, `operating_curves.png`, `attribution.md`, `harm.csv`.

### E2 — State fidelity

**Type.** Probe-based question answering against the store, with a fixed read agent and read budget; deterministic grading.

**Population.** Corpus R-H1 probe period (May–Jun 2026) and Corpus S. 300 probes per corpus, 60 per family, ≥ 150 entities.

**Conditions (what is read):**

| Arm | Material at snapshot(T), truncated to budget |
|---|---|
| A0 | nothing (model prior) |
| A1 | raw last-*N* events of the entity, N ∈ {20, 100} |
| A2 | BM25 top-k over all events ≤ T |
| A3 | embedding top-k over all events ≤ T (pinned model) |
| A4 | **curated store** S3 — read agent starts at `INDEX.md` / `OVERVIEW.md`, may open files within budget |
| floors | retrieve-everything (entity's full history, no budget); retrieve-nothing |

**Probe families.**

| Family | Question shape | Gold | Grader |
|---|---|---|---|
| Extraction | current value of a field (component, assignee, fixVersion, KIP status, plan) | field value at T | exact |
| Temporal | value of field *f* of entity *e* **as of T′ < T** | changelog replayed to T′ | exact |
| Knowledge update | field with ≥ 2 transitions before T: latest value; answer must not contain superseded values | latest + `superseded_values[]` | exact + supersession check |
| Multi-source | which PR / thread / ticket relates to *e* | regex join | links P/R |
| Multi-source (code) | which files / modules were changed for issue *e* (T after the PR merged) | linked PR diff (`--name-only`) | file P/R, module hit |
| Abstention | field or entity absent from the window | `UNKNOWN` | exact |

**Exact scores.** `correct(p) = 1[norm(answer) = norm(gold)]` for extraction / temporal / update / abstention; `link_F1(p)` for multi-source; `acc_family = mean(correct)`; **`macro_acc = mean over families`** (primary, A4 − A3 at 8 k, entity-clustered bootstrap CI + McNemar); `supersession_error = |{p ∈ update : any superseded value in answer}| / |update|`; `abstention_precision = |{p ∈ abstention : answer = UNKNOWN}| / |abstention|`; per probe `tokens_read`, `tool_calls`, `latency`.

**Procedure.**

1. Generate and freeze probes (hash in PREREG).
2. For each probe: leakage gate; materialise each arm at `snapshot(T)`; run the read agent; log tokens read, tool calls, latency.
3. Grade; compute per-family accuracy; supersession error rate; abstention precision.
4. Aggregate: macro accuracy over families; paired A4 − A3 with entity-clustered bootstrap CI; McNemar.

**Measurements.** **Primary:** macro accuracy over the five families, A4 vs A3 at 8 k budget. **Secondary:** per-family accuracy; supersession error rate; abstention precision; tokens read; tool calls per probe; latency per probe.

**Validity checks.**

- Temporal/update gold computed by two independent implementations (Python changelog replay; SQL over the raw JIRA export); disagreements resolved before a probe is kept.
- Retrieve-everything ≥ 0.9 macro accuracy (probe is answerable); A0 ≤ 0.3 on temporal/update/multi-source (probe is not answerable from memory — A0-correct probes flagged and reported separately).
- Graders reproducible: `exact` identical on re-run; `claims` run twice, ≤ 2 % disagreement, human-resolved.
- Pilot: 30 probes, two humans, pipeline-vs-human κ ≥ 0.8 before scaling.
- Leakage gate 100 % on kept probes; exclusions counted.
- Equal budget is real: logged tokens within ±10 % of 8 k for arms that fill the budget.

**Outputs.** `eval/out/e2/answers.jsonl`, `metrics.csv` (per arm × family), `contrasts.csv` (paired deltas + CIs).

### E3 — Store organisation

**Type.** Ablation over store designs built from the identical stream; E2 probes as the yardstick at three read budgets; longitudinal health measurement.

**Population.** Corpus R-H1 (both warm-up and probe period), Corpus S. Probes from E2.

**Conditions (stores):**

| Store | Built by | Layout | Isolates |
|---|---|---|---|
| S0 dump | `fake` harness | `events/YYYY/MM/DD/<id>.md`, one file per event | the filesystem alone |
| S1 templated | code, no LLM (`build_S1.py`) | seeded layout; OVERVIEW from structured fields; every event → one timeline line; no synthesis | whether the LLM adds anything over a rule-based Customer-360 |
| S2 curated flat | LLM builder | per-entity folder only; no INDEX/topics | curation |
| S3 curated + index | LLM builder | seeded layout with `INDEX.md`, `topics/`, `_meta/schema.md`, cross-links | global organisation (**headline**) |
| S4 vector | pinned embedding model | index over raw events, no files | the RAG alternative |
| S5 hybrid | S3 + embeddings over the store's files | — | retrieval on top of curation |

S3 is built with 3 seeds (`claude-sonnet-5`) plus one `claude-opus-5` build (seed 1). S2 and S5 only if budget remains (order of cuts: S5, then S2).

**Procedure.**

1. Build S0, S1, S4 from `replay/kafka-H1.jsonl` at 60×; commit per window so `snapshot(T)` exists.
2. Build S3 × 3 seeds (+ Opus); builder prompt and `CLAUDE.md` hashed in PREREG; usage logged per run.
3. Health checkpoints at replay weeks 1, 2, 4, 8 and end: `store_health.py` + a fixed 60-probe subset (erosion curve).
4. Run E2 against every store at budgets 2 k / 8 k / 32 k.
5. Cost accounting from provider usage records.
6. Analyse: accuracy-vs-budget per store; S3 vs S1 and S3 vs S4 at 8 k (primary), paired per probe, clustered by entity, across seeds; between-seed variance of accuracy and of health metrics; erosion slope per store.

**Measurements.**

| Metric | Definition |
|---|---|
| accuracy@budget | E2 macro accuracy at 2 k / 8 k / 32 k — **primary at 8 k** |
| build cost | builder tokens and $ per 1,000 events; wall time; files touched per run |
| store size | files, bytes, files per entity |
| over-cap share | files > 200 lines |
| INDEX accuracy | share of INDEX entries resolving to existing files |
| dangling refs | cross-references to non-existent files/IDs |
| duplicate-fact rate | MinHash near-duplicates across `facts.md` lines |
| supersession leakage | superseded values still present in `OVERVIEW.md` |
| erosion slope | change in 60-probe accuracy per replay week |
| builder failure rate | DLQ / rolled-back folds per store |
| seed spread | between-seed SD of accuracy@8k |

**Exact scores.** `acc(S, B)` = E2 `macro_acc` for store S at budget B; **primary** `acc(S3, 8k) − acc(S1, 8k)` and `acc(S3, 8k) − acc(S4, 8k)`, each valid only if `|Δ| > seed_spread`; `build_cost(S) = Σ(tokens_in + tokens_out)` and $ per 1,000 events; `index_acc = resolving INDEX entries / entries`; `dangling = unresolved refs / refs`; `dup_rate = near-duplicate fact lines (MinHash, Jaccard ≥ 0.8) / fact lines`; `leak_rate = entities with a superseded value in OVERVIEW / entities with ≥ 1 supersession`; `over_cap = files > 200 lines / files`; `erosion_slope` = OLS slope of 60-probe accuracy over replay weeks.

**Validity checks.**

- Same input, provably: replay hash and delivered-event-id list identical across stores (ledger diff).
- `store_health.py` unit-tested on synthetic stores with planted defects (dangling link, duplicate fact, superseded value in OVERVIEW) — each must be detected.
- S1 fairness: human review of 20 S1 entity folders confirms every structured field from the events is present; gaps fixed before comparison.
- S4 fairness: embedding model and chunking stated; recall@10 of gold `source_event_ids` reported.
- Seed rule: if between-seed spread exceeds the S3 − S1 gap, the result is stated as "no reliable difference".
- Builder failures counted; a store with > 5 % failed folds is rebuilt after the cause is fixed (logged amendment).

**Outputs.** `eval/out/e3/curve.png` (accuracy vs budget per store, both corpora), `health.csv` (per store × checkpoint), `erosion.png`, `cost.csv`, `contrasts.csv`.

### E4 — Context envelope

**Type.** Ablation over the context handed to a fixed agent on a fixed store (S3), graded against human actions (R) and constructed correct handling (S).

**Population.** Fast tasks: 150 per corpus. Batch windows: 150 per corpus.
- R fast tasks: events anywhere in R-H1 (Jan–Jun; the store is frozen at each task's own T) that the rules floor promotes (new Blocker/Critical issue, `[VOTE]` thread, CVE mention) and for which at least one human decision exists after T. Kafka H1 has ≈ 100–150 such events; Flink adds a similar number if needed.

**Gold per task (Corpus R) — four groups of human decisions, all after T, none ordered:**

| Group | Gold | Source | Score |
|---|---|---|---|
| People | assignee set within 24 h; reviewer(s) of the linked PR | JIRA changelog; GitHub reviews | `assignee_hit@3`, `reviewer_hit@3` (any value set within the window counts) |
| Category | component(s); duplicate-of link within 7 d; priority change within 24 h | JIRA changelog + links | exact match |
| Place | files / modules changed by PR(s) with the issue key, merged within 14 d | GitHub diff `--name-only` | `file_hit@5`, `file_recall`, `file_precision`, `module_hit` (Agentless superset convention) |
| Text | first committer comment / reply | JIRA comments; `dev@` | ROUGE-L; position-swapped pairwise judge, κ-checked |

Bot accounts excluded (published list). Tasks without a decision in a group have no gold for that group; coverage per group is reported (expect Place ≈ 70–80 %). The diff *content* is never gold — the agent routes, contextualises and localises at T; it does not patch.
- S fast tasks: injected situations with scripted correct handling.
- Batch windows: sampled from the probe period; each paired with its entity's E2 probes at T = window close.

**Conditions (envelopes):**

| Envelope | Contents | Tests |
|---|---|---|
| V0 | triggering event only | floor / prior |
| V1 | event + raw last-*N* entity events (N = 20, 100) | recency without curation |
| V2 | event + `OVERVIEW.md` | minimal curated |
| V3 | event + OVERVIEW + timeline tail (20 lines) + top-10 matched facts | **recommended small envelope** |
| V4 | V3 + the agent's own action log (last 10) | continuity / idempotency |
| V5 | V3 + just-in-time tools (`read_state`, `search_facts`, `recent_events`) | pull vs push |
| V6 | all files of the entity, verbatim | "stuff everything" ceiling |
| V7 | V3 + `superseded.md` bodies | distractor effect |
| V8 | V3 with sections shuffled, state mid-prompt | position effect |

Prefix (role, lane, tool docs, output schema) is byte-identical across envelopes.

**Procedure.**

1. Select tasks; exclude gold set by bot accounts (published list); leakage gate incl. gold-after-T.
2. Materialise V0–V8 from `snapshot(T)`; log tokens per section.
3. Fast: run the harness with the typed output schema (`assignee_candidates[≤3]`, `reviewer_candidates[≤3]`, `component`, `duplicate_of?`, `priority_change?`, `suspected_locations[≤5]`, `draft_reply`, `cited_ids[]`, `action`), 3 runs per task. The agent has the store, never the repository. Batch: run the builder with envelope V-x; apply the delta to a scratch copy; ask the entity's E2 probes against it.
4. Grade. Fast: assignee@3 / component / duplicate-of exact; required-ID coverage; reply ROUGE-L vs human reply; pairwise judge (different model family) in both orders. Batch: E2 accuracy of the post-delta state; evidence-citation validity (every `evidence_event_ids` exists and predates T).
5. Analyse: composite per envelope; V3 vs V1 and V3 vs V6 (primary), paired per task; V7 vs V3 and V8 vs V3 as named secondaries; cost per envelope; pass^3.

**Measurements.** **Primary:** action-quality composite `Q = mean(field_em, id_cov)` where `field_em` = mean over available gold fields of `1[pred = gold]` (assignee/reviewer as hit@3) and `id_cov = |required_ids ∩ cited_ids| / |required_ids|` (Corpus S: required IDs from world state; Corpus R: the issue key, linked PR/KIP keys) — contrasts V3 vs V1 and V3 vs V6. **Secondary:** localisation `file_hit@5`, `file_recall`, `module_hit`; ROUGE-L; `judge_win = 1[preferred in both orders]`; batch-delta E2 accuracy; `evidence_valid`; tokens; latency; tool calls; `pass^3 = 1[all 3 runs correct]`.

**Validity checks.**

- Gold actions are human (bot accounts excluded) and after T; PR-key join precision/recall reported; multiple PRs → union of files; test files count, formatting-only PRs excluded.
- Judge calibration: 200 pairwise judgements also made by two humans; κ(judge, human) ≥ 0.6 per corpus, else judge dropped and ROUGE-L only.
- Position bias cancelled: a win counts only if it survives both orders.
- Envelopes land in intended sizes: V3 median ≤ 12 k tokens; V6 median ≥ 3 × V3.
- Task balance: per-archetype and per-source counts reported; no archetype > 40 %.

**Outputs.** `eval/out/e4/runs.jsonl`, `metrics.csv` (per envelope), `contrasts.csv`, `judge_kappa.csv`, `sizes.csv`.

### E5 — Cadence and economics

**Type.** Ablation over incorporation cadence; same stream, builder, layout; cost from usage records.

**Population.** Corpus R-H1 and S; E2 probes; every event for freshness.

**Conditions:**

| Cadence | Setting |
|---|---|
| W1 | every event is its own run (all fast) — last 1,000 events only |
| W5 / W20 / W50 | session windows with those caps; gap and max-age scaled |
| W20+rules | window 20; rules floor promotes to fast |
| W20+rules+deviation | **ours** (full router R5) |

**Procedure.** Build each cadence's store (2 seeds) with usage logging; compute freshness per event (first commit whose diff references the id, replay-clock delay); run E2 probes and `store_health.py` on the final stores; plot cost per 1,000 events against accuracy and against freshness.

**Measurements.** **Primary:** cost per 1,000 events at equal E2 accuracy (within CI), W20+rules vs W1. **Secondary:** agent runs per 1,000 events; freshness p50/p95 for revealed-urgent vs routine events; E2 accuracy; fragmentation (files per entity, duplicate facts).

**Exact scores.** `cost_1k(W) = $ / (events / 1000)` from usage records at frozen prices; `runs_1k(W)`; `fresh(e) = t_commit_first_mentioning(e.id) − t(e)` on the replay clock, p50/p95 split by label; `acc(W)` = E2 `macro_acc` of the final store; `frag(W)` = files per entity and `dup_rate`; **primary** `cost_1k(W20+rules) / cost_1k(W1)` reported only where `acc(W20+rules) ≥ acc(W1) − CI`.

**Validity checks.** Cost from provider usage records with frozen prices; freshness on the replay clock (pass-through calibration ≈ 0 delay); builder-run count equals the classifier's window-close count; if W1 is significantly more accurate, report the price of freshness rather than hiding it.

**Outputs.** `eval/out/e5/pareto.png`, `cost.csv`, `freshness.csv`.

### E6 — Throughput, tail latency and bursts

**Type.** Systems benchmark; fake harness (deterministic, constant service time); open-loop load from a separate host; HdrHistogram.

**Population.** Synthetic streams fitted to Corpus R (inter-arrival per entity, Zipf popularity, burstiness B with finite-size correction) and, in Phase 2, Corpus S burst shapes.

**Conditions.**

| Factor | Levels |
|---|---|
| lane design | two-lane (router R5) vs single lane |
| load | steady at {0.25, 0.5, 1, 2, 4} × fitted mean; then {1.0, 1.5} × knee |
| shape | steady; benign flash crowd (×5 volume, mix unchanged, 10 min); anomalous burst (type-mix shift on 3 hot entities); Zipf-skewed hot entities; flash crowd + worker kill at peak; Poisson (B → 0) control |
| topology | partitions {8, 32} × workers {1, 4} |
| guards | full; minus absolute floor; minus multi-window; minus situation dedup |

**Procedure.**

1. Fit workload parameters from R; record them. Calibrate the ON/OFF generator until realised B matches.
2. Calibration run: pass-through pipeline with a fixed 10 ms sleep → p50 ≈ 10 ms, flat p99.
3. Steady sweep to find the **knee**: highest rate with ~zero p99 trend and non-growing lag; 25 % warm-up; ≥ 3 repetitions.
4. Burst shapes at 1.0× and 1.5× knee for both lane designs and all topologies.
5. Guard ablation under the anomalous burst.
6. Measure at the boundaries: producer→Kafka for throughput; fast-lane agent-start and batch fold-commit vs `intended_send_ts` for latency; per-partition lag; DLQ; duplicates.

**Measurements.**

| Metric | Definition |
|---|---|
| fast-lane latency p50/p99/p99.9 | `intended_send_ts` → agent-run start, during the burst window — **primary: p99.9 SLO attainment for injected urgent events at 1.5× knee, two-lane vs single-lane, and the gap as a function of B** |
| batch fold latency, staleness | window close → commit; event → commit |
| sustainable throughput | the knee |
| resource demand | minimal (partitions, workers) meeting the SLO at each load and entity cardinality |
| lag spike, drain time | peak `records-lag-max`; time to baseline after burst |
| recovery time | kill → pre-failure lag |
| partition-lag Gini | skew |
| self-amplification chart | fast-lane admission rate and urgent-event SLO compliance on one time axis |
| cross-entity fairness | SLO attainment for urgent events on cold entities during a hot-entity burst |
| duplicates / missed | per lane |

SLOs (D10): fast-lane p99 ≤ 2 s at up to 1.5× the single-lane knee; batch fold p99 ≤ 5 min, lag not trending up.

**Exact scores.** `lat(e) = t_agent_start(e) − intended_send_ts(e)`; p50/p99/p99.9 from HdrHistogram over the burst window; **primary** `SLO_att = |{urgent e : lat(e) ≤ 2 s}| / |urgent e|` at 1.5× knee, two-lane vs single-lane, and `gap(B) = SLO_att_two − SLO_att_single`; `knee` = max steady rate with p99-trend slope ≈ 0 and `records-lag-max` slope ≤ threshold; `demand(l)` = min (partitions, workers) meeting the SLO at load *l*; `drain = t(lag at baseline) − t(burst end)`; `recovery = t(lag at pre-kill) − t(kill)`; partition-lag Gini; duplicates and missed per lane.

**Validity checks.** Generator actual-vs-intended send timestamps within 1 ms at p99 (no coordinated omission); calibration run passes; fake-harness service time constant and logged; broker CPU/disk monitored; two-lane advantage must shrink toward zero under Poisson; repetitions' p99 within 20 %.

**Outputs.** `eval/out/e6/knee.csv`, `latency_hdr/*.hgrm`, `burst_slo.png`, `self_amplification.png`, `demand_curve.png`, `guards.csv`.

---

## 8. Statistics and reporting standard

- Paired per probe / task / situation on the identical stream; paired difference with BCa bootstrap 95 % CI (10,000 resamples), **clustered by entity**.
- McNemar for binary correctness between two conditions.
- One primary metric per experiment (named above); Holm–Bonferroni across that experiment's secondaries.
- Power: paired, 10-point effect near 0.6, α 0.05, power 0.8 → n ≈ 150; sample sizes above derive from this.
- ≥ 3 seeds for every LLM-built store; report pass^3 for tasks and between-seed spread of accuracy and health metrics.
- Sanity floors (random router, retrieve-everything, retrieve-nothing, model prior) run through the same code and are shown in every table.
- Every quality number is printed beside tokens, $ and latency.
- "Significantly better" = CI excludes zero **and** the delta is practically meaningful (stated per experiment in PREREG).

---

## 9. Meta-evaluation checklist (how we know the evaluation is right)

| Risk | Control | Evidence reported |
|---|---|---|
| Gold wrong | two derivations; human audit (200 probes, 100 actions, two annotators) | derivation agreement; κ; corrected-probe list |
| Probes trivial / unanswerable | retrieve-everything ≥ 0.9; model prior ≤ 0.3; pilot κ ≥ 0.8 | floor scores per family; dropped probes |
| Leakage | per-item gate; θ tuned on earlier months; PREREG predates runs | gate pass counts; commit timestamps |
| Grader noise / bias | deterministic re-run identity; claim grader twice; judge from other family, position-swapped, κ-checked | reproducibility; κ(judge, human) |
| Broken metric | random and always-flag scorers; toy-series validation; no point adjustment | floor table; dropped metrics |
| Strawman baselines | S1 human-reviewed; S4 retriever recall reported; same reader and budget everywhere | review notes; recall@10 |
| Randomness as effect | 3 seeds; spread vs effect; pass^3 | spread table |
| Overclaiming | paired clustered bootstrap; McNemar; one primary per experiment; Holm; power | CIs everywhere; PREREG |
| Hidden cost | usage-record accounting; frozen prices | cost beside every number |
| Irreproducible | frozen replay + hash; versioned prompts/schemas; snapshot SHAs; notebooks; model IDs/dates | reproducibility package |

---

## 10. Go / no-go gates

- **G1 (after Phase 0 pilot):** two-way gold agreement ≥ 98 %; pipeline-vs-human κ ≥ 0.8 on 30 probes; leakage gate 30/30; retrieve-everything ≥ 0.9; model prior ≤ 0.3 on temporal/update/multi-source. No store is built until G1 passes.
- **G2 (after baseline stores):** S0/S1/S4 from the same replay hash; S1 review complete; S4 recall@10 ≥ 0.7; E2 shows spread between A1 and A3 (if all baselines tie, probes are too easy → re-weight families).
- **G3 (after S3 seed 1):** failed folds ≤ 5 %; health metrics stable; pilot S3 − S1 CI excludes zero on ≥ 1 family. Then seeds 2–3 and the Opus tier.
- **G4 (before E4/E5):** OrgForge replay passes G1 on its probes; injected situations pass the E1 non-triviality check.
- **Stop rule:** if at G3 the between-seed spread exceeds the S3 − S1 gap, the central result becomes "curation does not reliably beat templating at this scale", and Phase 2 prioritises E1/E6 over E4/E5.

---

## 11. Repository layout for the evaluation system (harnext repo)

```
eval/
  PREREG.md
  README.md                      # points here
  corpus/      pony_mail.py  jira.py  gharchive.py  orgforge_{clicks,stripe,inject,state}.py
  replay/      kafka-H1.jsonl  kafka-long.jsonl  orgforge.jsonl  snapshots-<store>.csv
  replay.py    snapshots.py    stats.py
  probes/      gen_*.py  kafka.jsonl  orgforge.jsonl
  grade/       exact.py  claims.py  links.py  action.py
  health/      store_health.py  tests/
  agents/      reader.py  envelope.py
  stores/      build_S0.py … build_S5.py
  e1/          labels.py  policies.py  score.py  calibration.py
  e2/ e3/ e4/ e5/                run.py per experiment
  e6/          loadgen.py  run.py  hdr/
  out/         e1/ … e6/   gate.csv
  notebooks/   analysis per experiment
```

Conventions: every script takes `--corpus`, `--window`, `--seed`, `--out`; writes a `manifest.json` (inputs' hashes, model IDs, prompt hashes, prices, git SHA of harnext) beside its output; nothing reads the network during an experiment except the model APIs.

---

## 12. Decisions register

| ID | Decision |
|---|---|
| D1 | Corpus R = Apache Kafka. R-H1 = 2026-01-01 → 06-30 for LLM experiments (Jan–Apr warm-up, May–Jun probes). R-long = 2022-01 → 2026-06 for E1 with rolling month-ahead evaluation. Flink = replication. `users@` excluded. |
| D2 | Corpus S = OrgForge in Phase 2, extended with clickstream and Stripe-shaped events, injected situations, hourly world-state dumps. |
| D3 | Entity keys as in §3.1; anomaly baselines keyed on `contributor`, `component`, `thread` via `baseline_keys[]`. |
| D4 | Builder and read agent `claude-sonnet-5` everywhere; `claude-opus-5` on S3 seed 1 only; judge from a non-Anthropic family; temperature 0; prompt/schema hashes logged. |
| D5 | Budget headline b = 2 %; sweep {1, 2, 5, 10} %; θ per source tuned only on earlier months. The claim is about *per-entity deviation scoring with guards*; HBOS is one scorer among the compared policies. |
| D6 | Curated layout = seeded harnext layout on the git backend, one commit per run. |
| D7 | Session window 30 s gap / 20 events / 120 s max age at 60× replay; scaled proportionally otherwise. |
| D8 | Read budgets 2 k / **8 k** / 32 k tokens, enforced by truncation, counted with the provider tokeniser. |
| D9 | 300 probes and 150 + 150 tasks per corpus; 200 injected situations; 3 seeds per LLM-built store; 3 runs per task. |
| D10 | SLOs: fast-lane p99 ≤ 2 s at ≤ 1.5× single-lane knee; batch fold p99 ≤ 5 min; lag not trending up. |
| D11 | `eval/PREREG.md` committed before the first full run; amendments appended. |
| D12 | Pre-run fixes: every writer to the store goes through the per-org lock (the MCP path had its own `BuildRunner`); builder tool set = read-only sandboxed Bash allowed, prompt text aligned. |
| D13 | E1's primary population is the rule-negative subset; the calibration-by-decile curve is a named secondary. |
| D14 | E4 gold = four groups of human decisions after T (people, category, place, text); the linked PR's changed files are the localisation gold; the diff content is never gold (no patch generation). E4 tasks are drawn from all of R-H1. |

---

## 13. Glossary

- **Entity / subject** — the thing a state file is about (`issue:`, `pr:`, `contributor:` …). **Baseline key** — the long-lived entity an anomaly baseline is kept for.
- **Fold** — one batch-lane builder run that incorporates a window into the store. **Snapshot(T)** — the store commit after the last fold before T.
- **Revealed urgency** — a label derived from what humans did *after* the event.
- **Rule-negative subset** — events the rules floor did not promote.
- **Read budget** — the maximum tokens of material the read agent may load.
- **Envelope** — the exact context handed to the agent for one run.
- **Probe** — a question with derived gold, asked against a snapshot.
- **Gate** — a pre-registered pass/fail check that must hold before the next phase spends money.
