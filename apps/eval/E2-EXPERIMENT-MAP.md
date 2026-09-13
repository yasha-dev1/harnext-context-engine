# Merged E2/E3: questions, controls, factors, and decisions

Updated 2026-09-12. This is the current benchmark execution map, adding the
no-MCP control to the earlier thesis blueprint. It distinguishes implemented
configuration options from planned studies. It does not approve execution of
the full benchmark or change its frozen tasks, gold, or candidate splits.

The experiment asks two separate questions:

1. Does access to historical context improve an outside agent's answers and code?
2. Given access, which storage, writing, retrieval, and harness settings give the
   best quality for a declared cost and time allowance?

The comparison unit is the same task under different conditions. We do not give
easy questions to one configuration and difficult questions to another. Context
configuration is selected for the declared workload, not separately using the
correct answer to each test question.

## 1. Conditions applied to the same questions

| ID | What the reader receives | Historical tools | Coding tools | What comparison establishes |
| --- | --- | --- | --- | --- |
| N: no MCP | Question only for history; identical base repository and visible tests for coding | None | Isolated shell, no MCP | Reference answer/test success without historical context |
| R: raw history | Eligible original records, searched with BM25; no learned curation | MCP search/read | Same shell + MCP | R − N measures the benefit of providing searchable historical data |
| F: organized files | Facts/entity pages/index/history ledger | MCP list/read; search is a controlled factor | Same shell + permitted MCP tools | F − R tests whether organization helps beyond raw access |
| V: retrieval store | Same eligible records with lexical, semantic, or hybrid ranking | MCP search/read | Same shell + MCP | Ranking/chunking/reranking effects on a fixed corpus |
| G: graph | Supported entities, temporal assertions, and relations | MCP entity lookup/assertions/traversal | Same shell + MCP | Whether explicit relationships help on the tasks that need them |
| H: combined engine | Shortlisted file + graph + search configuration | Selected MCP tools | Same shell + MCP | Whether combining useful components improves the complete system |
| O: supplied evidence, diagnostic | Reviewed sufficient historical evidence supplied directly in the prompt | None | Same shell; historical evidence only | Separates finding evidence from interpreting it; never supplies a coding solution |

N and R are mandatory anchors in primary comparisons. R can be deduplicated
with an identical V/BM25 cell. O is a diagnostic, not a deployable contender or a
guaranteed empirical upper bound. H is a whole-system comparison: it changes
several components, so it cannot by itself establish a single-component effect.

In F/G mechanism tests, `expose_sources: false` prevents fetching the original
archive as an escape route. Complete-system finalists are also compared with
raw-source access on/off. Always report this setting. Access to a source must
never bypass its temporal cutoff, tool allowlist, or response allowance.

Historical N has neither shell nor filesystem access. Coding N retains the
same repository, visible tests, and shell as coding MCP arms. Removing those
would confound historical-context value with the ability to inspect or run code.

## 2. Actual task population and examples

Examples below are shortened from the frozen task prompts. The reviewer SPA
contains the full cutoff, instructions, gold, and proof. Each main condition
above uses the same tasks; targeted studies can use a preselected, shared
development panel. O is used on a small diagnostic panel.

| Task family | Count | Example actually in v1 | Deterministic score | Most informative factor comparisons |
| --- | ---: | --- | --- | --- |
| Assignee at snapshot | 100 | Who was assigned to KAFKA-10101 at the snapshot? | Exact name or null | N/R/F; canonical IDs versus supported aliases; current versus retained state |
| Status at snapshot | 100 | What was the workflow status of KAFKA-10017? | Exact label, e.g. Reopened | Temporal updates; stale/current-state errors; file versus graph |
| Resolution at snapshot | 100 | What was KAFKA-10303's resolution? | Exact label or null | Supported extraction; replacement versus supersession |
| Closed at snapshot | 100 | Was KAFKA-10064 exactly Closed, rather than Resolved? | Exact boolean | N baseline, source-state fidelity; this is a calibration task |
| Priority at snapshot | 100 | What priority did KAFKA-10167 have? | Exact label | Retrieval depth, raw versus organized lookup; calibration |
| Assignment history | 100 | List every assignee change for KAFKA-10049, in order | Exact ordered timestamp/from/to array | All/current/window retention; summaries; ledger; byte budget |
| Status history | 100 | List every status change for KAFKA-10021 through the cutoff | Exact ordered transition array | Append/replace/supersede; history length; chunk boundaries |
| Resolution history | 100 | List every resolution change for KAFKA-10043, including resets to null | Exact ordered transition array | Lossy summaries; chronology; retained supporting evidence |
| PR implementation | 200 candidates total | Bugfix: KAFKA-12259 / PR 10040, prevent the consolidated connector-status endpoint returning HTTP 500 when config lookup fails. Feature example: PR 11695, allow null-valued records in Console Producer | Admitted acceptance and regression tests in a clean evaluator | Repository/tests-only versus +raw context versus +Harnext; reader harness, retrieval, budget |

Bugfix and feature examples are both within the 200 coding candidates; there is
not yet a separately validated bugfix/feature quota. Candidate patches apply to
the chosen bases, but their Gradle environments and base-fail/reference-pass
checks still need admission. Identical source code to the reference PR is not
required. Changed-file overlap is descriptive, not correctness.

**Coverage gap:** v1 does not contain dedicated issue→PR→file multi-hop QA,
implementation-history explanations, alias-heavy questions, independently
reviewed semantic paraphrases, or unanswerable questions. Graph superiority and
semantic robustness cannot be claimed from simple ID lookups. Add those as a
separately versioned, reviewed development extension before using them to select
graph/semantic settings; do not silently count them among the frozen 1,000.

| Proposed extension, not currently in the 1,000 | Sample question | Construction/scoring requirement |
| --- | --- | --- |
| Multi-hop implementation history | Which merged PRs addressed issue X before the cutoff, and which files did they change? | Join explicit issue links, merge records, and changed-file lists; exact reviewed sets |
| Historical implementation lookup | At this snapshot, which commit introduced behavior X, and where was it implemented? | Explicit frozen issue/commit mapping and source evidence; exclude ambiguous intent claims |
| Semantic/alias query | Ask the same supported question without its literal issue ID, or with a supported alias | Independent review that meaning and answerability are unchanged; keep in the same lineage |
| Missing evidence | What was decided about X when no eligible record contains that decision? | Exhaustively verify absence in the allowed corpus; explicit abstention schema and score |

## 3. Factor catalog: what changes and what remains fixed

These are staged comparisons, not one enormous Cartesian product. Levels in
brackets below are proposed targeted sweeps. Existing enum support does not mean
that all combinations are valid or that their scientific comparisons have run.

| Factor | Levels / YAML controls | Tasks and decision | Readiness |
| --- | --- | --- | --- |
| Historical context access | Pilot `context_access: none / mcp` | All; value of providing context | Historical Codex executable; coding baseline awaits admission/adapter |
| Representation | `representation: files / graph / hybrid`; raw baseline via file layout | All; compare complete systems first | Engine implemented |
| File layout | `verbatim_dump`, `foldered_sessions`, `flat_facts`, `indexed_entities`, `indexed_temporal`, `topic_hierarchy`, `agent_curated` | All; organization/navigation versus build cost | Implemented; learned layouts need builder |
| Writing instructions | `files.prompt: preserve_supported / concise_supported / domain_focused` | State/history/coding; detail versus compactness | Implemented |
| Fact extraction | `structured / atomic_llm` | Same input text; deterministic provided-fact parsing versus learned extraction | Implemented; report that structured facts are dataset-supplied |
| Retention | `all / current / window`; window [30, 90, 365] days | Especially history; quality lost by pruning | Implemented; loss is a result, not an exclusion |
| Summarization | `none / extractive / abstractive`; sentence count [3, 8] | History/coding; compression versus evidence preservation | Implemented; abstractive needs learned builder |
| Updates | `append / replace / supersede` | State + transition questions; accuracy versus stale/duplicate facts | Implemented |
| File granularity | `page_bytes` [8192, 32768, 65536] | Navigation calls, truncation and storage | Implemented; bytes, not the old blueprint's token units |
| Reader file tools | List/read only versus +search | Same frozen file store; does search remove layout differences? | Implemented `trial.tools` |
| Retrieval method | `literal / bm25 / dense / hybrid` | ID lookup and semantic extension/coding | Implemented; literal is diagnostic, BM25 main lexical anchor |
| Chunks | `document / fixed / entity_bundle`; bytes [1024, 2048, 4096], overlap [0, 256] | Boundary loss and support per response | Implemented; invalid combinations rejected |
| Depth/fusion | Candidates [20, 40, 80], top-k [5, 10, 20]; freeze BM25/RRF defaults initially | Recall versus bytes/latency | Implemented; enforce top-k ≤ candidates |
| Reranking | `none / cross_encoder`, one pinned model | Same candidate pool; quality versus extra query cost | Implemented |
| Embeddings | One pinned primary and one pinned alternative; provider/model/revision | Dense/hybrid finalists on identical chunks | Implemented; exact alternate identity frozen before running |
| Graph schema | `generic_attributed / typed_assertions` | Relationship + temporal questions | Implemented; dedicated relationship QA extension pending |
| Graph extraction | `broad / domain_predicates` | Coverage versus irrelevant relationships | Implemented |
| Entity resolution | `canonical_only / evidence_aliases` | Alias recall versus incorrect merges | Implemented; alias-rich task coverage pending |
| Traversal algorithm | `neighbors / paths / personalized_pagerank / dual_level` | Multi-hop/coding; relevant relationships per read | Implemented adaptations, not complete paper-package replications |
| Traversal size | Hops [1, 2], returned edges [20, 50]; fixed expansion cap | Recall versus noisy neighborhoods | Implemented |
| Raw archive access | `expose_sources: false / true`, with appropriate tool allowlist | Curated retention versus raw fallback | Implemented; keep matched within mechanism contrasts |
| Builder model/effort | Same harness with two pinned models; baseline versus higher supported effort/allowance | Does a better writer change store rankings? | Builder adapters exist; live configuration admission required |
| Builder harness/tools | Harnext versus baseline; native file operations versus shell | Layout × builder interface | Separate builder-adapter study; not the reader MCP-tools grid |
| Reader model/effort | Economical, primary, stronger pinned recipes | Same stored snapshots; store × reader strength | Codex pilot verified; expanded recipes pending |
| Reader harness | Codex / Claude Code / Harnext adapter where supported | Same task/store/tools; harness robustness | Only native Codex verified for this benchmark route |
| Retrieval allowance | Total bytes [16384, 65536, 262144], primary 65536; fixed per-response cap initially | Quality–budget curve | Implemented; record actual tokenizer usage separately |
| Calls/deadline | Primary max 100 MCP calls and 240 s reader; targeted caps predeclared | Failures, efficiency, slow-tool sensitivity | MCP and historical Codex controls exist; coding needs its own feasible fixed deadline |
| Replicates/run order | Three independent reader runs on a fixed diagnostic panel; three builds for learned finalists | Variability, paired intervals | Study orchestration pending; repeat IDs are not model sampling seeds |
| Corpus growth | Shared eligible prefixes/checkpoints and history-length strata | Storage/build growth, long-history failures | Cutoff builds supported; full longitudinal study pending |
| Paraphrases | Original versus reviewed meaning-preserving variants | Lexical versus semantic robustness | Versioned extension pending |
| Consolidation/cadence | Incremental, daily, weekly; same delivered data | Update cost/freshness bridge to E5 | Separate schedule study, not currently a pilot YAML treatment |
| Delays/duplicates/corrections | Controlled delays [0, 1, 24] h; duplicates [0%, 10%] | Temporal robustness/idempotence | Separate synthetic replay study; source time is only a proxy in v1 |
| Package challengers | Pinned Graphiti/LightRAG as whole systems | External-system robustness | Optional integrations; current enums do not reproduce those packages |

Fixed controls: exact question core, typed answer schema, eligible source corpus,
cutoff, source authority/order, coding base/test hashes, model identity within a
mechanism comparison, machine/resources, and enforcement policy. Use independent
sessions/workspaces; no future history, reference solutions, evaluator files,
network, or native tools beyond the declared profile. Run conditions in blocked
random order when scaling; keep event order inside each build chronological.

Only the availability instruction changes between N and MCP prompts; the
question, cutoff, and output schema remain the same. This avoids instructing a
no-tool agent to call nonexistent tools. Save both full prompt hashes and the
shared question-core hash. Correct guesses, nulls, and booleans can score without
evidence; label answer correctness separately from evidence support. In v1, null
means a real field value, not an abstention.

Some harnesses cannot run the same model. A Codex/model-A versus Claude/model-B
comparison is a **whole-stack comparison**, not a clean harness effect. Builder
and reader harnesses are separate factors; do not change both together when
claiming either one's effect.

## 4. Execution order and run counts

| Stage | Run | Gate / resulting decision |
| --- | --- | --- |
| Instrument pilot | Same eight historical development tasks with MCP and without MCP | Verify capability boundaries, scoring, trace accounting, and baseline comparison |
| Coding admission | Small reviewed coding sample; base and reference in the same build environment | Base fail-to-pass tests fail; reference passes; identify pass-to-pass regressions |
| Paired coding pilot | Admitted tasks under N, R, and current H | Check actual added context value and coding trace coverage |
| Broad development screening | Same preselected development panel across 12 file, 12 retrieval, 8 graph cells, plus N/R controls | Shortlist within families; inspect family-specific failures and feasibility |
| Targeted comparisons | Shortlisted stores × the applicable factor blocks above | Resolve interactions without crossing everything with everything |
| Development selection | Candidate configurations on the reviewed development population | Freeze candidate, cheap comparator, settings, workload weights and contrasts |
| Confirmation | Untouched eligible confirmation tasks; N/R + frozen finalists; independent repeats | Estimate paired effects and uncertainty; no retuning on these outcomes |

The current split has 694 development and 306 confirmation candidates, not the
older blueprint's 160/300/300 probe counts. Coding admission and lineage review
may reduce usable counts; disclose that before any comparison. The old blueprint
also used token budgets and full-stream build trajectories; do not label this
byte-budget snapshot pilot as execution of that complete design.

The existing 32-cell generator is a starting screen. Its file grid varies
reader navigation/search access, not builder shell/native-file tools. Its graph
grid does not yet vary traversal. Add targeted blocks explicitly instead of
claiming those questions were covered by the initial 32 configurations.

For a shared panel of n admitted tasks, 32 distinct screening cells + N + R give
at most 34n reader trials before deduplication and repeats. For example, n=90
means at most 3,060 trials, not 90 model calls. Each reader trial may involve
multiple model turns. Do not commit to a panel size or full grid cost until the
coding pilot and development power/cost estimates are available.

## 5. Measurements, graphs, and interpretation

| Graph / output | Exactly what is measured | Decision and prediction to test |
| --- | --- | --- |
| Correctness by condition and task family | Exact-answer fraction for history; admitted test success for coding | N < R for obscure history; H may or may not beat R |
| Paired improvement over N and R | Per-task outcome differences, including helped/harmed/tied counts; grouped confidence intervals at scale | Is context useful, and is Harnext better than cheap raw search? |
| Coding outcome matrix | Fail-to-pass and pass-to-pass results per task/condition, timeouts separately | Context may help cross-file fixes; self-contained fixes may show no gain |
| Store → retrieve → answer diagnostic | Corpus-required evidence coverage in store; fraction actually returned; final correctness | Locate writer loss, reader miss, or reasoning error |
| Quality versus allowed bytes | Family scores at matched 16/64/256 KiB caps | Good organization may matter most under tight budgets |
| Recall over MCP calls/bytes | Union of returned gold evidence spans, grouped by required support | Earlier support with fewer repeated reads is preferable |
| Tools and context use | Completed tool counts, arguments, errors, repeated reads/bytes, zero-use rate | More MCP calls are not inherently better; show what was fetched |
| File-factor heatmap | Paired layout × prompt × tool-access scores and cost | Ledger may help histories; search may shrink layout differences |
| Retrieval-factor heatmap | Chunking × ranker × reranker, equal eligible data/caps | BM25 may win literal IDs; semantic methods need nonliteral tasks to show value |
| Graph-factor heatmap | Schema/extraction/resolution/traversal scores and support, on appropriate tasks | Typed graphs may help multi-hop tasks; extra edges may add noise |
| Harness/model interactions | Same stored snapshots, task panels and permitted tools; explicit stack labels | Store rankings may change with reader/tool behavior |
| Build/query latency and tokens | Native input/cache/output tokens, index/extraction work, wall time, failures, storage | Query improvements may cost more to build or maintain |
| Quality–cost frontier | Correctness versus measured lifecycle cost for stated query volumes | Select a conditional default; do not assume the most elaborate engine wins |
| Growth and robustness curves | Same-corpus checkpoints; controlled delayed/duplicate/correction scenarios separately | Compression can save space while damaging long-history answers |

Retrieval precision remains unavailable without exhaustive relevance labels.
Required-support recall uses reviewed representations, not the source IDs the
agent claims to have used. For N, MCP calls and evidence exposure are zero;
retrieval algorithm precision/recall is not applicable because no retrieval
system ran. Evidence exposure does not prove causal use by the answering model.

The current pilot binder requires complete source records in a stored view. That
is sufficient for the raw-enabled pilot, but not a valid universal gold mapping
for summarized or graph-only stores. Before those comparisons, review equivalent
fact/relationship spans and measure storage coverage independently. Do not drop
a lossy configuration or task simply because its required history is missing.

Report historical macro-average over its eight families and coding success
separately. Do not let 500 simple field questions hide coding/history failures.
Choose any combined workload weights before selection; show sensitivity to those
weights. Pair outcomes by task/lineage and account for learned-store repeats in
uncertainty estimates. Ordinary model failures remain failures in the denominator.
Admission defects are resolved across all arms, not excluded selectively.

Before confirmation, freeze the small set of superiority/noninferiority contrasts,
practical effect margins, grouped uncertainty method, replicate counts and
multiple-comparison handling. Use development variance to check whether the
remaining confirmation sample can resolve those margins. Current reviewer
access to all gold means this is not a cryptographically sealed test set; do not
claim otherwise. Public PR training contamination remains a limitation even when
the runtime receives no future commits.

## 6. Current entry points

- [Frozen 1,000-task reviewer SPA](benchmarks/kafka-1000-v1/review.html)
- [Supported engine settings and MCP protocol](E2-STRATEGIES.md)
- [MCP pilot configuration](configs/benchmark-codex-pilot.yaml)
- [No-MCP pilot configuration](configs/benchmark-codex-no-mcp-pilot.yaml)
- [Measured paired MCP/no-MCP results](STATUS/E2-NO-MCP-PILOT-20260912.md)
- [Native MCP pilot report](../../.harnext/artifacts/e2-native-benchmark-pilot-001.html)
- Earlier thesis planning reference: `masters-streaming-ai-agent-architecture/eval/artifacts/06-e3-experiment-blueprint.html`

No full benchmark execution or production-default claim follows from these pilots.
