# Merged E2: implementation understanding and held-out coding tasks

Status: development extension, 2026-09-12. This does not replace the earlier
smoke results or register confirmation hypotheses retroactively.

The research question is now also: **does a context strategy help an agent make
the correct code change at a particular historical snapshot, and what evidence
does the agent retrieve while doing so?** Answer quality, executable task
success, retrieval quality, and resource usage remain separate outcomes.

Actual fixture counts and test results are recorded in the
[development verification record](STATUS/E2-DEVELOPMENT-20260912.md).

## What exists now

- 32 constructed QA probes: 16 contract/history/absence probes plus 16 scenario
  probes requiring application of the contracts to supplied inputs.
- 5 constructed coding tasks with independent pre-change source trees, visible
  regression/acceptance tests, private extra tests, and evaluator-only reference
  implementations. They concern pagination, idempotency, retry policy, and
  two-clock snapshot visibility, and a batch retry planning feature.
- 29 annotated context paragraphs: 17 applicable or superseded source documents
  and 12 similar but irrelevant dashboard-policy documents. The index is
  navigation overhead rather than a relevant fact.
- Public/private dataset export; deterministic span-exposure scoring with exact
  tool replay; a model-driven QA/coding development runner; local candidate
  patches and PR descriptions; containerized test execution.
- Executable initial controls: repository/tests without context (`none`) and
  the raw immutable development context (`raw`). The latter uses the same
  `SnapshotTools` implementation as the merged smoke.

These are **synthetic development fixtures**, not real PRs, a production MCP
benchmark, or evidence that the eventual tasks are sufficiently difficult for
frontier models. Four coding fixtures have one implementation file; the batch
feature has separate planner and worker modules. The historical population must include larger multi-file changes and realistic
repositories. No new live model scores were produced when adding this package.

The original smoke tested only the recorded model/configuration. A claim that
most models are at ceiling needs a multi-model pilot on the same task set.

## Two task tracks

### A. Understand what was implemented

Ask questions whose gold is decomposable into inspectable facts:

| Family | Example task | Deterministic scoring |
|---|---|---|
| Implemented versus discussed | Which accepted proposals actually reached the code by T? | Exact set of change/capability IDs, plus evidence exposure |
| Temporal reconstruction | Which behavior applied before a rollback, and which applied after it? | Typed fields at each cutoff; no credit for citing a later implementation |
| Multi-step dependency | Join an issue, decision, PR, module move, and compatibility constraint | Required relationship/claim sets and supporting source groups |
| Conflict resolution | Resolve superseded decisions, rejected proposals, and stale documentation | Current/historical contract fields scored separately |
| Behavioral reasoning | Given inputs, derive pagination, retry, or visibility behavior from several contracts | Exact computed values or order-independent sets |
| Scope and impact | Identify affected entrypoints and contracts without guessing the reference patch's exact file set | Reviewed symbol/contract labels with alternative valid locations |
| Absence | A requested approval, performance number, or implemented feature is missing | Explicit abstention; no invented values |

Keep simple checks as anchors. Add difficulty through genuine dependencies,
conflicting versions, large relevant histories, and realistic distractors, not
merely longer wording. Analyze anchor and scenario questions separately.

For open explanations, request a structured claim table alongside prose. Grade
the claims and source spans, not the style of the prose. An LLM judge can be a
secondary diagnostic but cannot silently become the deterministic primary.

### B. Implement a withheld change

For a historical task choose a pinned base revision **B**, observation cutoff
**T**, and target PR **P**. The base must not contain P or an equivalent merged
change. Do not blindly assume the merge commit's first parent is the original
development base; record and validate the chosen revision explicitly.

The agent receives:

1. A clean source tree exported from B, with no Git history, remote, sibling
   checkout, reference implementation, or future files.
2. A versioned issue/feature request available at T. For reconstructed requests,
   label the task as reconstructed and keep it separate from authentic historical
   requests. Never derive a prompt that accidentally enumerates the target diff.
3. Visible tests: the base regression suite and a frozen subset of feature/bug
   acceptance tests. Every experimental condition sees identical tests.
4. Optional context through the assigned engine, containing only information
   observable at T. The agent decides when and how to retrieve it.

The evaluator retains P, the reference patch, private acceptance/regression
checks, gold evidence annotations, and all later events. It grades the final
candidate in a separate fresh test environment after the agent has finished.
The hidden result is never fed back during that trial.

The output is a **local candidate patch and PR description**, not a published
PR. Correct alternative implementations pass even if they differ from P.

“Withheld” means absent from this run's inputs. It does not establish that a
public PR was absent from model training. Record possible contamination and use
newly authored/private tasks as a separate panel where suitable access exists.

## Historical task admission and leakage controls

Repository choice is pending. Apache Kafka matches the thesis corpus and is a
provisional candidate, not a selected or downloaded PR population. Building the
development fixture does not depend on that choice.

Every admitted real task needs a manifest with:

- repository URL and license; immutable base and reference revisions;
- cutoff, target issue/PR identity, task request version/hash;
- source-tree hash and reproducible dependency/test image digest;
- exact public and private test files, hashes and commands;
- test IDs that fail on B and pass on the reference; preserved regression IDs;
- exported context hash, event/source versions and availability timestamps;
- held-out change lineage, split/cluster identity, source-to-fact annotations;
- budgets, model and harness versions, and exclusion/admission decisions.

Audit PR bodies, comments, issue edits, commit messages, patches, code snapshots,
and generated summaries for target-change leakage. Current GitHub/Jira text can
contain edits made after T even when the record was created earlier. Use a
historical version or exclude that field. Require all store builds, indexes,
graph edges and caches to respect the same cutoff. Purge target PR descendants,
backports, near-duplicate fixes and shared test adaptations from other splits.

Reject a task if the base cannot build, the target bug is not reproduced by a
specific assertion, the reference fails, the tests are flaky, or a missing
requirement makes the expected result unknowable from permitted inputs. Record
these exclusions before model trials; never remove a valid task because a
particular model performed poorly.

Private tests can be newly authored; their requirements must be supported by the
public task or permitted pre-T context. They cannot introduce a surprise product
requirement learned only from the future PR. Keep visible and hidden tests
separate in reports, since visible tests themselves convey implementation clues.

## Measuring coding success

Freeze two test sets after base/reference validation:

- **Fail-to-pass (F2P):** assertions failing on B and passing on the reference.
- **Pass-to-pass (P2P):** selected regression checks passing on both.

Primary task resolution requires a completed candidate, all required F2P checks
passing, all required P2P checks still passing, and valid test execution with no
unauthorized test/config changes. Preserve per-test outcomes. Report F2P
fraction, P2P regression count, visible-suite success, private-suite success and
full resolution separately. An import/build error is not an ordinary reproduced
bug; infrastructure failures need their own status and cannot disappear from
the denominator without a prespecified rule.

Measure one attempt per trial initially. Repeated trials are independent fresh
attempts, not a hidden best-of-N selection. Run a candidate from a clean source
export and pristine tests; do not grade an agent's edited test files. The pilot
prevents test edits through its file allowlist and exposes only a fixed test
command. Container execution is network-disabled, non-root, resource-limited,
and mounts only the trial source and selected test suite. It does not inherit
host credentials or a Docker socket.

The pilot is a trusted-fixture runner, not a formal adversarial grader. Production
admission still needs streaming process-output limits, test discovery/report
integrity checks, artifact limits, and repo-specific build/test adapters. A test
suite passing alone is not proof against an implementation designed to tamper
with its runtime. Candidate review and protected test execution remain necessary.

## Measuring retrieval during implementation

Log each tool action with run/task/model/condition identity, global step,
snapshot/content hash, query/path/offset, exact returned content and byte count,
truncation, elapsed time, and phase (before first edit, after edits, after tests).
Keep context retrieval separate from repository inspection, file edits and test
execution. Record unsuccessful and empty retrieval attempts too.

Gold is **task-relevant supporting content**, not “the files the reference patch
changed.” The agent may need an API contract in mail without editing that file;
it may edit a code file without ever retrieving it through Harnext.

Annotate stable evidence units in raw sources. For each store representation,
map actual supporting passages to those units in an evaluator-only sidecar.
A curated summary must contain the supporting proposition, not just an event ID.
Graph edges need actual relation/provenance mappings; vector results need exact
returned chunk spans. Freeze/review mappings independently of reader outputs.
If evidence was lost during building, that is a storage-retention failure; if
present but not returned, that is a retrieval failure. The current raw fixture
has exhaustive paragraph annotations; other representations are not mapped yet.

For task evidence requirements G1...Gn, each group contains alternative valid
support units. A group is covered when at least one alternative is fully exposed.
Current scoring unions byte ranges across partial reads, validates quotes against
the snapshot, and replays actual tool output before awarding exposure.

| Metric | Definition | Interpretation |
|---|---|---|
| Context use rate | Trials with at least one context attempt / all trials | Adoption, not success |
| Context calls and tool mix | Counts of list/read/search/etc., with empty/error calls separate | Retrieval behavior; fewer is not automatically better |
| Required-group recall | Covered required evidence groups / required groups | How much annotated support reached the agent |
| Unit precision | Relevant fully exposed units / all fully exposed annotated units | Requires exhaustive judgments; undefined if none returned |
| Relevant-byte precision | Returned bytes overlapping relevant support / all returned context bytes | Charges wrappers, irrelevant text, and duplicate reads; repeated relevant reads are charged too |
| First relevant retrieval | First context call exposing a complete relevant support unit | Undefined if support never arrives |
| Recall by calls/bytes | Recalculate cumulative group coverage at each retrieval step | Retrieval efficiency under the budget |
| Pre-edit coverage | Required evidence exposed before the first code edit | Evidence available when implementation starts |
| Repeated material | Re-read bytes / all retrieved bytes, measured by span overlap | Redundant retrieval; planned addition to the pilot scorer |
| Storage retention | Required groups represented in the stored view / source-supported groups | Distinguishes builder loss from reader failure; pending curated mappings |

Recall for tasks without a positive evidence requirement is undefined, not 100%.
Unjudged material is not automatically irrelevant: when annotations are partial,
precision is undefined and annotation coverage must be shown. Gold exposure
does not prove that the model used a fact internally. Causal usefulness comes
from matched experimental conditions, not self-reported citations or call counts.

The pilot currently measures calls/tool mix/bytes, span precision and recall,
first relevant context call, cumulative recall, and call counts before/after
first edit. Global action timing and exact traces allow later phase analyses.
Full pre-edit *coverage*, repeated-byte summaries and citations-to-claims scoring
are planned additions, not fields already reported as measured results.

## Experimental comparisons

Use matched tasks, base sources, visible tests, models, effort and total budgets.

| Condition | Purpose | Current coding runner |
|---|---|---|
| Repository + tests, no context | Baseline capability and information available in code/tests | Runnable (`none`) |
| Repository + tests + raw context | Is source history useful without curation? | Runnable (`raw`) |
| Repository + tests + selected Harnext file store | Does the engine preserve and expose useful evidence? | Pending historical snapshots and reviewed span mappings |
| Selected lexical/semantic/graph strategy | Does retrieval/storage strategy change task success? | Later finalists from the merged E2 screen |
| Oracle relevant evidence | Diagnostic upper bound when the needed support is supplied | Planned diagnostic; evaluator gold is deliberately supplied only in this labeled arm |

Initially tell agents that context exists and how to use it, without requiring a
call or rewarding call counts. A guided “consult context before edits” instruction
is a separate prompting factor. Do not conflate forced usage with voluntary
adoption. Give the no-context arm equal overall execution opportunity; charge
retrieval against the same overall model/action budget in context arms.

Screen store/template/retrieval configurations on the cheaper QA development
panel first. Send a small number of frozen finalists to coding tasks. Compare
harnesses only after normalizing repository tools, tests, context boundaries and
budgets. Avoid crossing every template, prompt, graph, retriever, model and
harness with every expensive coding task at once.

A provisional historical development pilot is 8–12 distinct PR tasks spanning
bugfixes and features, with paired no-context/raw/Harnext trials and a small
number of independent repetitions. This is an engineering pilot, not a powered
confirmation sample. Use pilot variance and paired discordance to choose the
confirmation size before inspecting held-out results. Cluster by issue/change
lineage; several questions about one PR do not become independent samples.

## Planned graphs for the coding extension

These are chart specifications; no live coding results exist yet.

| Graph | Measurement | Hypothesis to test, not an expected result to enforce |
|---|---|---|
| Resolution by condition and task family | Fully resolved / all admitted trials; paired uncertainty | Useful context may improve context-dependent changes |
| Paired task outcome matrix | Success/failure for the same PR across conditions | Show wins, regressions and shared failures, rather than only an average |
| F2P versus P2P | New requirements fixed versus preserved regressions | A seeming fix can introduce compatibility failures |
| Visible versus private tests | Both test panels on the final patch | Visible-only success may overstate generalization |
| Retrieval precision versus recall | Span-based scores per trial, sized by retrieved bytes | Broader retrieval may improve coverage while adding irrelevant material |
| Recall versus cumulative budget | Gold exposure after each context action | Good layouts may expose required support earlier |
| Resolution versus retrieval recall | Final executable result against support actually received | Evidence access may help, but does not guarantee implementation ability |
| Task action timeline | Context/repo reads, edits, tests and finalization on a shared clock | Agents may retrieve after failures as well as before edits |
| Context adoption and tool mix | Use rate, empty calls, tools and repeated material | Guided prompting may change behavior without improving outcomes |
| Latency/token tradeoff | Total builder + agent + test work, cached usage separated | A context benefit may or may not justify extra work |
| Context-dependence breakdown | Matched effects on code-local versus history-dependent tasks | Benefits may be concentrated in history/decision-heavy tasks |
| Difficulty and distractor sensitivity | Outcomes by dependencies, change span and distractor load | Robust strategies should degrade gradually rather than collapse |
| Harness interaction | Paired condition differences within each harness | An engine benefit may depend on tool-use behavior |
| Failure attribution | Storage missing / retrieval missed / retrieved but code wrong / test infra | Identify the next engine or harness improvement |

Correlations in these graphs do not establish causal use of evidence. The paired
condition comparison provides the main test of incremental usefulness. Context
that merely restates everything in visible tests may have little incremental
effect; keep code-local tasks as a control rather than excluding them afterward.

## Run the development gate

From the engine repository root, with Docker available:

```bash
docker pull python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

.venv/bin/python -m harnext_eval.e2.development export \
  --out apps/eval/out/e2-development/dataset-NEW

# No model calls. Verify every buggy base/reference against the same tests.
.venv/bin/python -m harnext_eval.e2.development validate \
  --out apps/eval/out/e2-development/validation-NEW

# Explicit live opt-in; runnable commands, not results already obtained.
.venv/bin/python -m harnext_eval.e2.development qa --arm raw \
  --model gpt-5.6-luna --effort medium --live \
  --out apps/eval/out/e2-development/qa-NEW

.venv/bin/python -m harnext_eval.e2.development code \
  --task dev-code-retry --arm raw --model gpt-5.6-luna --effort medium --live \
  --out apps/eval/out/e2-development/code-NEW
```

Use a fresh output directory and fresh agent history for every trial. Public
export files are separate from private answers/reference tests; never mount the
whole export or the evaluator package inside a model-controlled environment.
The provider receives only host-mediated actions with native tools disabled.

Pilot budgets: 12 QA completions, 30 coding completions, 6 visible test runs,
65,536 context bytes and 12,000 bytes per context response. Test invocations have
a 30-second timeout; model completions have 120-second timeouts. These are not
a production token-cap or total trial deadline. Output-token requests remain
advisory with the existing CLI adapter. Do not use this development runner for
large billable studies before total deadline/token/output bounds are implemented.

The [shared YAML strategy and external MCP route](E2-STRATEGIES.md) now provides
the context-engine connection separately from this older host-mediated prototype.
Its external Codex smoke and real neural retrieval checks are recorded in
[the verification record](STATUS/E2-STRATEGIES-20260912.md). Historical task
curation and reviewed representation mappings are still pending.

The next real-data step is to choose a repository, curate and validate a small
historical task population, and connect its pinned Harnext snapshots to reviewed
evidence mappings. Keep this synthetic fixture as the admission/regression gate.
