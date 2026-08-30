# R1 — E2/E4 graders, statistics, and reporting review

1. **blocker — `apps/eval/src/harnext_eval/probes/gen_code_location.py:23-45,56-84`; `apps/eval/src/harnext_eval/probes/gen.py:41-46`**  
   **Spec:** E2's code multi-source probe has `T` after the linked PR has merged and grades facts available at `snapshot(T)`; future changed files are E4 localisation gold, not E2 state gold (§4, §7 E2).  
   **Code:** the shared E2 probe set deliberately samples `T` before a merge, then derives files from PRs in `(T, T+14d]` and asks which PRs “merge within 14 days after the snapshot.” The answer is post-T information.  
   **Minimal fix:** split E2 and E4 code-location gold. Generate E2 probes only after qualifying merges and include only files known by T; retain the future 14-day union only for E4 tasks, and run the real leakage gate over the resulting E2 probes.

2. **blocker — `apps/eval/src/harnext_eval/e2/run.py:66-74,259-298`; `apps/eval/src/harnext_eval/replay/gate.py:43-75`**  
   **Spec:** every probe must prove that the builder received no event after T, that no question token comes only from post-T events, and that post-T gold/action material is absent; failures are excluded and counted (§4.2).  
   **Code:** E2 passes the gate a delivery log already filtered to `event.time <= T`, so the delivered-after-T assertion is incapable of failing. It also fabricates `gold_action_time = T + 1 microsecond`, sends no evaluated material/envelope, and discards the gate's detailed log to `/dev/null`. The separately written E2 `gate.csv` contains only PASS/FAIL and, on raw fallback, even leaves `sha`/`last_event_id` blank.  
   **Minimal fix:** pass the actual builder delivery log through the selected snapshot SHA, real gold metadata where applicable, and the exact material shown to the reader; preserve the canonical gate reasons/identifiers in `gate.csv` and reject any item lacking data needed to prove a gate condition.

3. **blocker — `apps/eval/src/harnext_eval/e2/arms.py:193-244,247-261`; `apps/eval/tests/test_e2e3/test_e2e3.py:48-50,78-90`**  
   **Spec:** A4 starts at `INDEX.md`/`OVERVIEW.md` and navigates the requested entity's curated state at `snapshot(T)` (§7 E2).  
   **Code:** entity selection searches paths/links for the literal canonical entity, e.g. `issue:HNX-1`, but store paths are `entities/issue/HNX-1/...`; neither `issue:hnx-1 in entities/issue/hnx-1` nor the same check against a link target succeeds. A4 therefore commonly reads only the root index and never the entity state. The test masks this by using entity `HNX-1` without the required `issue:` prefix.  
   **Minimal fix:** parse canonical entity kind/key and resolve the exact `entities/<kind>/<key>/` path (and normalised index links); add a test with a real canonical entity string and assert that `OVERVIEW.md` content is actually read.

4. **blocker — `apps/eval/configs/s3-curated.yaml:8-12`; `apps/eval/src/harnext_eval/e2/run.py:373-391`; `apps/eval/src/harnext_eval/e2/arms.py:173-184`; `apps/eval/src/harnext_eval/providers/llm.py:58-63,98-129`**  
   **Spec:** D4 fixes the E2 reader to `claude-sonnet-5`, and A3 uses a pinned embedding model; model/prompt identity is part of the reproducibility package.  
   **Code:** the supplied S3 experiment profile uses the fake reader and fake embeddings. The E2 registry adapter injects neither a real reader nor an embedding provider, and A3 silently constructs `FakeEmbeddings` regardless of the configured provider. The fake reader has no link-answering mode and does not recognise the code-location question, so multi-source/link, code-location, and all abstention scores are effectively identical across arms; it also answers temporal questions with the last lexical field value rather than the value at T'. A0's `<=0.3` check is tautological because empty material always yields `UNKNOWN`.  
   **Minimal fix:** refuse publishable E2 runs with fake providers; instantiate and pin the configured real reader/embedding providers, record their IDs, and label fake runs as smoke-only. If the fake path remains, implement every probe family and add a non-triviality check that requires score spread across arms/families.

5. **blocker — `apps/eval/src/harnext_eval/e4/run.py:388-397,470-495`**  
   **Spec:** for each batch condition, run the builder with V-x, apply its delta to a scratch copy, then ask the paired E2 probes against the resulting state (§7 E4 procedure 3–4).  
   **Code:** no builder is called, no delta is produced/applied, and no read agent or E2 grader runs. `batch_e2_acc` is merely the fraction of gold strings found as substrings in the pre-existing envelope. This makes the batch experiment vacuous and can directly reward material containing gold strings without demonstrating a correct state update.  
   **Minimal fix:** materialise an isolated snapshot per task/variant, invoke the actual builder, apply only its delta, run the shared E2 reader/graders on the paired probes, and log the resulting state SHA, usage, and grades.

6. **blocker — `apps/eval/src/harnext_eval/e4/tasks.py:201-290,293-324,396-421`**  
   **Spec:** Corpus R gold is derived from specified human decisions; Corpus S gold is scripted correct handling from world state with fact coverage (§4, §7 E4).  
   **Code:** `build_tasks` uses the same event-following human-action heuristic for every corpus, including synthetic/OrgForge, and has no world-state rule or fact-coverage derivation. Thus Corpus S does not have the constructed ground truth the claim requires.  
   **Minimal fix:** dispatch gold derivation by corpus: keep audited R derivation, and for S load the world-state dump/injected-situation rule, emit required fact IDs and coverage, and cross-check it before tasks enter E4.

7. **blocker — `apps/eval/src/harnext_eval/e4/run.py:549-574,665-681`**  
   **Spec:** V3−V1 and V3−V6 are paired per task on the identical stream with 10,000-resample entity-clustered BCa 95% CIs; binary correctness also gets McNemar (§7 E4, §8).  
   **Code:** E4 writes only `n` and an unclustered arithmetic `mean_delta_Q`. There are no confidence intervals, no entity clusters, no McNemar result, and no practical-significance decision, yet these bare deltas are returned as the primary result.  
   **Minimal fix:** pair task/run outcomes, aggregate the preregistered task score, call the shared 10,000-resample entity-clustered BCa routine with a fixed run seed, add McNemar for named binary outcomes, and write CIs/sample/entity counts and practical-significance status to `contrasts.csv`.

8. **major — `apps/eval/src/harnext_eval/e2/run.py:108-118,158-215`**  
   **Spec:** literally, `macro_acc = mean over families`, and the primary effect is the difference in those family-macro scores (§7 E2 Exact scores).  
   **Code:** `macro_accuracy` silently averages whatever families happen to be present instead of requiring all five. More importantly, `_paired_contrast` computes a micro-average of per-probe differences, not a paired difference of five equally weighted family means. It also treats continuous link/file F1 as binary only when exactly 1.0 for McNemar without specifying that convention.  
   **Minimal fix:** validate the five required macro families, compute per-family arm means and then their equal-weight macro, and bootstrap a paired statistic that preserves that exact family weighting; restrict McNemar to a preregistered binary correctness definition.

9. **major — `apps/eval/src/harnext_eval/e2/run.py:184-195`**  
   **Spec:** all primary CIs are entity-clustered; insufficient clusters make a contrast invalid (§8).  
   **Code:** E2 uses only 1,000 rather than 10,000 resamples, and if fewer than two entities exist it fabricates a zero-width CI equal to the point estimate.  
   **Minimal fix:** use 10,000 resamples and fail/mark the contrast invalid when cluster count is below the minimum; never report a point estimate as both CI bounds.

10. **major — `apps/eval/src/harnext_eval/probes/gen.py:21-46`; `apps/eval/src/harnext_eval/e2/run.py:108-118`**  
    **Spec:** E2 has 300 probes per corpus, 60 per each of five families; code localisation is a multi-source subtype (§7 E2).  
    **Code:** the generator creates six independently sized families—360 probes at the specified default—and later merges `code_location` into `multisource` only for macro scoring. That gives the multi-source family twice the samples and a hidden 50/50 subtype weighting.  
    **Minimal fix:** generate 300 total with 60 macro-family slots and a preregistered split inside the multi-source family; report both subtype scores while giving the combined family exactly one macro weight.

11. **major — `apps/eval/src/harnext_eval/grade/links.py:12-25`; `apps/eval/src/harnext_eval/probes/gen_multisource.py:13-27,88-96`; `apps/eval/tests/test_grade/test_exact_links.py:23-29`**  
    **Spec:** multi-source uses set precision/recall/F1 over PR/thread/ticket links (§4, §7 E2 Exact scores).  
    **Code:** gold identifiers are emitted as `pr:<n>` and `thread:<id>`, but the free-text parser recognises only ticket-like `ABC-123` and `#123`. A prediction containing multiple `pr:`/`thread:` IDs becomes one normalised blob rather than a set, so its P/R/F1 is wrong. Tests cover only KAFKA/KIP ticket keys and miss the actual gold dialect.  
    **Minimal fix:** define and parse every canonical link type emitted by the gold generator (including multiple comma/newline-separated PR and thread IDs), then add literal P/R/F1 cases using those identifiers.

12. **major — `apps/eval/src/harnext_eval/e2/arms.py:90-107,142-184`; `apps/eval/src/harnext_eval/config.py:62-80`**  
    **Spec:** A1 includes both N=20 and N=100 variations, and all snapshot material is visible through T (§7 E2 conditions).  
    **Code:** the runner has one A1 label/default (20), the strict configuration exposes no `last_n` variation, and A1 uniquely excludes events exactly at T (`< T`) while A2/A3 use `<= T`. Results cannot distinguish A1-20 from A1-100 and can unfairly omit the state-setting event.  
    **Minimal fix:** run separately labelled `A1-N20` and `A1-N100` arms and use `event.time <= T`, with deterministic tie ordering.

13. **major — `apps/eval/src/harnext_eval/providers/tokenizer.py:1-13`; `apps/eval/src/harnext_eval/agents/reader.py:32-48,109-119`**  
    **Spec:** D8 requires truncation and accounting with the provider tokenizer.  
    **Code:** all budgets use a regex token approximation, including real Anthropic runs. Therefore neither the 8k cap nor the ±10% fill check is in provider-token units.  
    **Minimal fix:** bind token counting/truncation to the pinned reader model's tokenizer and record that tokenizer ID/version; retain the approximation only for explicitly non-publishable smoke runs.

14. **major — `apps/eval/src/harnext_eval/e2/run.py:324-359`; `apps/eval/src/harnext_eval/grade/claims.py:158-168`**  
    **Spec:** E2 validity includes dual gold, retrieve-everything ≥0.9, A0 flags, exact rerun identity, claim grading twice with ≤2% disagreement and human resolution, pilot human κ≥0.8, 100% leakage, and real equal budgets (§7 E2 validity; G1).  
    **Code:** E2 implements only aggregate floor/prior/leakage/budget booleans. `grade_claims_twice` is never called by E2; there is no exact rerun, disagreement gate, human κ, or dual-gold agreement output. `all(budget_checks)` passes vacuously when no arm fills the budget. Failed checks do not prevent a primary result from being emitted.  
    **Minimal fix:** implement and persist every validity item, require non-empty denominators, flag A0-correct probes individually, and gate publishable primary results/G1 progression on all mandatory checks.

15. **major — `apps/eval/src/harnext_eval/probes/gold.py:88-177`; `apps/eval/src/harnext_eval/probes/gen_temporal.py:25-44`; `apps/eval/src/harnext_eval/probes/gen_update.py:25-41`**  
    **Spec:** temporal/update gold must agree between Python changelog replay and an independent SQL computation over the raw JIRA export, with agreement/disagreements reported (§7 E2 validity, §9).  
    **Code:** the “SQL” path queries the same normalised `EvalEvent.model_dump_json()` records consumed by Python, not the raw JIRA export, and generators silently discard disagreements without reporting agreement or a corrected-probe list. Shared normalisation bugs can therefore agree in both paths.  
    **Minimal fix:** implement the SQL derivation against independently loaded raw JIRA tables/export, record agreement and every excluded/resolved item, and enforce the preregistered threshold.

16. **major — `apps/eval/src/harnext_eval/grade/action.py:95-106`**  
    **Spec:** literally, `Q = mean(field_em, id_cov)`, with required IDs defined for both corpora (§7 E4 Exact score).  
    **Code:** missing `field_em` or `id_cov` is silently removed and Q is reweighted over the remaining component; if both are missing Q becomes 0. This changes the metric instead of treating malformed/missing gold as invalid.  
    **Minimal fix:** require both composite components for every scored fast task, reject/flag tasks without valid required IDs or available field gold, and compute exactly `(field_em + id_cov) / 2`.

17. **major — `apps/eval/src/harnext_eval/e4/tasks.py:181-198,261-271`; `apps/eval/src/harnext_eval/e4/run.py:240-256`**  
    **Spec:** Corpus R `required_ids` are the issue key plus linked PR/KIP keys (§7 E4 Exact score).  
    **Code:** required IDs are frozen from the trigger alone; linked PR/KIP identifiers discovered during gold derivation are never added. The primary `id_cov` therefore omits required evidence and over-scores incomplete citations.  
    **Minimal fix:** canonicalise and union the trigger key with every linked PR/KIP key found by the audited join before grading.

18. **blocker — `apps/eval/src/harnext_eval/e4/run.py:359-379,436-469,491-493`**  
    **Spec:** E4 uses the full §4.2 per-item leakage gate; evidence is valid only when every cited event exists and predates T (§7 E4 procedure).  
    **Code:** the custom gate checks only snapshot metadata, optional decision timestamps, and whether future gold event IDs appear verbatim. It does not inspect actual deliveries through the SHA, question/material-only-post-T tokens, or leaked gold values/text; missing decision times pass. `evidence_valid` merely checks whether citation text occurs anywhere in the envelope, not whether it is an existing pre-T event.  
    **Minimal fix:** use the shared gate with the real delivery log and rendered envelope/gold actions, require complete fast-task timestamps, and validate each citation against the replay event index with `event.time <= T`.

19. **major — `apps/eval/src/harnext_eval/e4/tasks.py:84-89,293-324`; `apps/eval/tests/test_e4e5/test_tasks_envelopes.py:78-82`**  
    **Spec:** R fast triggers are new Blocker/Critical issues, `[VOTE]` threads, or CVE mentions; sampling must report balance and no archetype may exceed 40% (§7 E4).  
    **Code:** any later priority transition to Blocker/Critical is selected as another task, creating overlapping/dependent tasks, and tasks are the first 150 chronological matches rather than a balanced sample. The test explicitly enshrines the wrong extra priority-transition task. A post-hoc `max_archetype` flag does not repair the population.  
    **Minimal fix:** identify the specified trigger event types, deduplicate situations/entities as preregistered, stratify/sample deterministically to the source/archetype cap, and fail task construction when balance cannot be met.

20. **major — `apps/eval/src/harnext_eval/e4/tasks.py:149-170,201-278`; `apps/eval/src/harnext_eval/e4/run.py:616-629`**  
    **Spec:** gold must be human, the text target is the first committer reply, formatting-only PRs alone are excluded, and PR-key join precision/recall is reported (§4, §7 E4 validity).  
    **Code:** a reply with no role metadata is assumed to be from a committer; any all-documentation PR is classified “formatting only”; reviewer/file events are accepted by loose subject/key relatedness without demonstrating the linked-PR join. The reported “join precision” and “join recall” checks are just booleans for presence of event IDs/files, not precision or recall.  
    **Minimal fix:** require verified committer membership, use an audited formatting-only classification rather than file suffixes, evaluate the mandated key join against annotated pairs, and output numerical P/R plus exclusions.

21. **major — `apps/eval/src/harnext_eval/agents/envelope.py:222-255`; `apps/eval/src/harnext_eval/e4/run.py:130-153`**  
    **Spec:** V1 has N=20 and N=100 conditions; V5 provides callable `read_state/search_facts/recent_events`; V7 adds only the task entity's superseded bodies (§7 E4 conditions).  
    **Code:** V1 is always the unconfigurable default 20 in the runner; V5 sends exactly the same prompt/sections as V3 and merely logs three tool names—no tool can be called—so V3 and V5 scores are identical for every provider; V7 searches all store files and can inject other entities' `superseded.md` bodies.  
    **Minimal fix:** create separately labelled V1-20/V1-100 conditions, implement actual bounded tool dispatch/transcripts for V5, and scope V7 to the current entity snapshot.

22. **major — `apps/eval/src/harnext_eval/agents/envelope.py:37-45,255-258`; `apps/eval/src/harnext_eval/e4/run.py:455-465,593-604`**  
    **Spec:** per-section tokens and the V3≤12k/V6≥3×V3 validity checks use the actual provider token count (§7 E4).  
    **Code:** `token_count` sums prefix and section bodies but omits every rendered `## <section-name>` heading and separator, and uses the approximate tokenizer. Size validity is therefore calculated on text different from what the provider receives.  
    **Minimal fix:** render the exact prompt once, count it with the pinned provider tokenizer (with auditable per-section overhead allocation), and use those counts for `sizes.csv` and validity checks.

23. **major — `apps/eval/src/harnext_eval/e4/run.py:498-546,630-663`; `apps/eval/src/harnext_eval/grade/action.py:185-207`**  
    **Spec:** E4 reports order-swapped pairwise `judge_win`; 200 judgements are calibrated against two humans and judge-human κ must be ≥0.6 or the judge is dropped (§7 E4).  
    **Code:** although a two-order helper exists, the runner never calls it, `judge_win` is absent from `metric_names`, position-swapping is hard-coded false, and `judge_kappa.csv` is a permanent placeholder saying calibration is unavailable. The named file exists but the named measurement does not.  
    **Minimal fix:** run the non-Anthropic judge in both orders, ingest the 200 dual-human labels, compute/report κ by corpus, and include `judge_win` only when the gate passes.

24. **major — `apps/eval/src/harnext_eval/e4/run.py:88-143,470-495,527-534`**  
    **Spec:** three runs per task support `pass^3` and randomness/reliability analysis (§7 E4, §8).  
    **Code:** the fake path bypasses `FakeLLM` and deterministically parses the envelope, so all three repetitions for a task/variant are byte-identical and `pass3` collapses to the one-run perfect-score rate. Together with V3/V5's identical prompts, this produces guaranteed identical arm scores and a vacuous reliability measure in the default offline experiment.  
    **Minimal fix:** mark fake E4 output smoke-only and never use it for claims; for real runs preserve three separately logged completions, and add a test/non-triviality gate detecting identical predictions/scores across all repetitions or supposedly distinct arms.

25. **major — `apps/eval/src/harnext_eval/stats/stats.py:47-55`; `apps/eval/src/harnext_eval/e2/run.py:337-368`; `apps/eval/src/harnext_eval/e4/run.py:549-681`**  
    **Spec:** Holm–Bonferroni applies across each experiment's secondaries, and at least three LLM-store seeds must report between-seed accuracy/health spread (§8).  
    **Code:** the Holm helper and seed-spread helper are not used by E2 or E4. Runs are written seed-by-seed with no cross-seed aggregation, no corrected secondary p-values, and no spread-vs-effect result. The generic BCa helper also defaults to an unseeded RNG, permitting non-reproducible callers.  
    **Minimal fix:** add deterministic run-level aggregation over the three seeds, calculate/report spread and Holm-adjusted secondaries, and require an explicit recorded RNG seed for every resampling call.

26. **major — `apps/eval/src/harnext_eval/e2/run.py:327-368`; `apps/eval/src/harnext_eval/e4/run.py:498-546,640-680`; `apps/eval/src/harnext_eval/report/templates/report.html.j2:61-78`**  
    **Spec:** every quality number is printed beside tokens, dollars, and latency; E2 separately reports A0-correct probes, and E4 reports all named secondary measurements (§7 E2/E4 outputs, §8).  
    **Code:** neither E2 nor E4 computes dollars. E2 does not output per-probe A0-correct flags as a validity population. E4 omits judge wins and its contrast table omits resource measures. The report shows primary mappings/contrast CSVs independently from cost/latency tables, so quality is not beside tokens/$/latency.  
    **Minimal fix:** derive cost from logged provider usage and frozen prices, add the missing flags/metrics, and generate joined quality-resource tables in both CSV and HTML.

27. **major — `apps/eval/src/harnext_eval/report/report.py:99-133`; `apps/eval/src/harnext_eval/e2/run.py:350-359`; `apps/eval/src/harnext_eval/e4/run.py:597-639`; `apps/eval/tests/test_report/test_report.py:48-51,78-87`**  
    **Spec:** validity evidence must be visibly reported as pass/fail (§8–§9).  
    **Code:** actual experiment checks are emitted as floats `0.0/1.0`, but the report recognises only Python booleans (or mappings) as pass/fail; all real E2/E4 checks render as neutral INFO. The report test supplies a fabricated boolean and therefore misses the integration bug.  
    **Minimal fix:** emit typed booleans (and observed values separately) or teach the report schema to interpret explicit numeric check records; test against real `ExperimentResult` serialization from E2/E4.

28. **major — `apps/eval/tests/test_e2e3/test_e2e3.py:48-75,184-207`; `apps/eval/tests/test_report/test_stats.py:18-49`**  
    **Spec:** E2 tests must validate all five-family exact scores, leakage/validity failures, equal budgets, and the 10,000-resample entity-clustered primary analysis.  
    **Code:** the end-to-end fixture has only extraction, update, and abstention probes, all for one entity; it asserts only that outputs exist and checks say PASS. It consequently validates neither temporal/link/file grading nor the literal macro formula and accepts the fabricated one-entity CI. Stats tests exercise a standalone helper with 2,000 resamples but never assert the E2 runner uses the required 10,000.  
    **Minimal fix:** add adversarial five-family, multi-entity fixtures with unequal family sizes, planted leakage, post-T gold, filled/unfilled budgets, and literal expected macro/CI inputs; assert invalid runs cannot publish a primary result.

29. **major — `apps/eval/tests/test_e4e5/test_runs.py:95-150`; `apps/eval/tests/test_e4e5/test_tasks_envelopes.py:101-123,126-173`**  
    **Spec:** E4 tests must exercise fast and batch procedures, all conditions/variations, exact Q, paired statistics, judge calibration/order, leakage/evidence validity, and size/balance gates.  
    **Code:** the run test covers only fast tasks and V1/V3/V6 and asserts row counts, schema caps, no reported leakage, and file existence—never scores, CIs, pairing, costs, or judge behavior. The batch test only checks task construction, not running a builder/delta/E2. Envelope tests omit V1-100 and V2, check merely V6>V3 rather than ≥3×, do not execute V5 tools, and have no second entity to expose V7 contamination.  
    **Minimal fix:** add end-to-end oracle tests for every named condition and validity failure, including a spy builder/read agent/tool dispatcher, two entities, both V1 Ns, exact expected Q/contrasts, calibrated position-swapped judge behavior, and threshold boundary cases.

30. **minor — `apps/eval/src/harnext_eval/e4/run.py:555-564`**  
    **Spec:** reruns must be deterministic enough for paired comparisons and reproducible artifacts (§2, §8–§9).  
    **Code:** contrast task IDs are held in a set and iterated without sorting. Hash-randomised order can change floating summation at the last bits and therefore CSV bytes across processes.  
    **Minimal fix:** sort paired task IDs before constructing deltas and serialising every paired result.

## Verdict

The implementation is not faithful enough to support trusted E2/E4 conclusions: E2 contains post-T code-location gold, a non-proving leakage gate, a broken canonical A4 traversal, fake providers that make several families invariant, and a primary statistic that is not the literal macro contrast; E4's batch experiment is not implemented, Corpus S gold is wrong, the leakage/evidence gates are incomplete, and its primary contrasts have no required inference. Before results can be trusted, separate E2/E4 gold semantics, enforce the temporal firewall with real delivery/material evidence, run the pinned real providers and all specified conditions, implement the actual batch and judge procedures, compute literal scores with 10,000-resample entity-clustered inference/McNemar/Holm and seed aggregation, enforce every validity gate, and replace the smoke-level tests with adversarial full-family/full-condition tests.

## Fixes applied

- **2** → `apps/eval/src/harnext_eval/replay/gate.py:46` replaces filtered-event and fabricated-action evidence with exact store/SHA ledger proof, probe-specific source/question/material checks, recursive task gold/time checks, and canonical audit reasons. Historical snapshot calls resolve only through a uniquely registered live store and otherwise fail closed → `apps/eval/tests/test_replay/test_gate.py:55`, `:79`, `:109`, `:131`, `:153`, and `:173`.
- E2/E4 call-site integration outside FX-core ownership was deliberately left to the designated module owners; the shared gate API and compatibility adapter needed by those callers are complete.
