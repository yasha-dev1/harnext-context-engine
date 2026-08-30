# Eval framework — implementation plan

Source of truth for *what* is measured: `docs/evaluation-spec.md` (§7 has the E1–E6 runbooks with exact scores; §11 the target layout; §12 the decisions). This file is the *how*: package design, shared interfaces, the minimal baseline engine, and the task split for parallel implementation.

## 1. Goals for this iteration

1. Every experiment E1–E6 from the spec exists in code, with its conditions, exact scores, validity checks and outputs.
2. Everything runs **end to end offline** against a *minimal baseline context engine* — no API keys, no Kafka (except E6) — via `make eval-smoke`. That baseline is the reference every later configuration is compared against.
3. The engine under test is fully described by one **YAML config** ("nudges"); the run writes a manifest and a **report** (charts for every measurement + the config it ran with).
4. Real-corpus extractors exist and are tested on fixtures; running them against the network is a later step.

Out of scope now: the Next.js UI (a static HTML report is produced instead), OrgForge integration, real LLM runs at scale.

## 2. Package

`apps/eval` → distribution `harnext-eval`, import name `harnext_eval`, registered as a uv workspace member and in the root `dev` dependency group. Console script `harnext-eval`.

Dependencies: `harnext-shared`, `harnext-builder` (harness protocol, `FakeHarness`, `GitBackend`, `SEED_FILES`, `BuildRunner`), `harnext-classifier` (`WindowManager`, `rules_match`), `pydantic`, `pyyaml`, `typer`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `pyod`, `rank-bm25`, `matplotlib`, `hdrh` (HdrHistogram), `jinja2`, `datasketch` (MinHash). Kafka (`aiokafka`) only inside `replay/kafka_replay.py` and `e6/`.

```
apps/eval/
  pyproject.toml  README.md  PLAN.md
  configs/                       baseline-minimal.yaml  s1-templated.yaml  s3-curated.yaml  e6-twolane.yaml  e6-single.yaml
  src/harnext_eval/
    __init__.py
    config.py                    EngineConfig (YAML → pydantic), ExperimentConfig, load_config()
    types.py                     shared records (below)
    manifest.py                  RunManifest writer (hashes, model ids, prices, git sha)
    registry.py                  Experiment protocol + registry (e1..e6)
    cli.py                       typer app: corpus | probes | stores | run | report
    providers/                   llm.py (LLMProvider protocol, FakeLLM, AnthropicLLM), embeddings.py (EmbeddingsProvider, FakeEmbeddings(hash), OpenAI/…), tokenizer.py (count_tokens)
    corpus/                      synthetic.py (T0) · keys.py · pony_mail.py · jira.py · gharchive.py · build_replay.py (T1)
    replay/                      driver.py (in-process pipeline over JSONL) · kafka_replay.py · snapshots.py · gate.py (T2)
    probes/                      schema.py · gen_extraction.py … gen_abstention.py · gen_code_location.py · gold.py (T3)
    grade/                       exact.py · claims.py · links.py · action.py · localisation.py (T4)
    health/                      store_health.py (T4)
    stores/                      base.py · build_s0.py … build_s5.py · templated.py · vector_index.py (T5)
    e1/                          labels.py · features.py · policies.py · score.py · calibration.py · run.py (T6)
    agents/                      reader.py · envelope.py (T7 reader; T8 envelope)
    e2/ e3/                      run.py each (T7)
    e4/ e5/                      run.py each (T8)
    e6/                          loadgen.py · run.py · metrics.py (T9)
    stats/                       stats.py (paired clustered bootstrap, McNemar, Holm, power) (T10)
    report/                      charts.py · report.py · templates/report.html.j2 (T10)
  tests/                         one directory per task: test_core/ test_corpus/ test_replay/ test_probes/ test_grade/ test_stores/ test_e1/ test_e2e3/ test_e4e5/ test_e6/ test_report/
  STATUS/                        <task>.md written by each implementer at the end
```

Every run writes to `out/<run_id>/` (git-ignored): `manifest.json`, `config.yaml` (resolved), per-experiment `results.json` + CSV/Parquet, `charts/*.png`, `report.html`.

## 3. The minimal baseline context engine

Defined in `configs/baseline-minimal.yaml`. It is the *reference configuration* every experiment can run against without keys or brokers, and the lower anchor for all comparisons:

```yaml
engine:
  router:
    rules: { enabled: true }                # declared priority / [VOTE] / CVE / money threshold
    deviation: { enabled: false }           # no per-entity scoring in the baseline
    budget_pct: 2.0
    guards: { absolute_floor: 0, multi_window: false, situation_dedup: false }
  window: { gap_s: 30, max_events: 20, max_age_s: 120 }
  store:
    layout: S0                              # dump: one file per event
    backend: git
  builder:
    harness: fake                           # deterministic; no LLM
    model: null
    prompt_version: v1
  reader:
    provider: fake                          # deterministic reader (keyword/ID matcher)
    budget_tokens: 8000
  envelope: V3
  embeddings: { provider: fake, dim: 64 }   # hash-based; only used by S4/S5
budgets: { read_tokens: [2000, 8000, 32000] }
seeds: [1]
```

Other profiles change one section at a time (`s1-templated.yaml`: `store.layout: S1`; `s3-curated.yaml`: `layout: S3`, `builder.harness: claude_code`, `builder.model: claude-sonnet-5`; `e6-*.yaml`: lane design). The config loader validates that every knob the spec names exists here; unknown keys are errors.

**Fake providers must be deterministic and non-trivial.** `FakeLLM` answers reader questions by lexical matching over the material it was given (find the line containing the probe's entity id and field name; return the last value; `UNKNOWN` otherwise) — so the offline pipeline exercises real budget truncation, grading and statistics with plausible (not perfect, not random) scores. `FakeEmbeddings` = feature-hashed bag-of-words, L2-normalised.

## 4. Shared interfaces (owned by T0; other tasks import, never redefine)

`harnext_eval.types` (pydantic models):

- `EvalEvent(CloudEvent)` — adds `baseline_keys: list[str]`, `intended_send_ts: datetime | None`.
- `Probe(probe_id, family: Literal[extraction,temporal,update,multisource,code_location,abstention], entity, T, question, gold: Any, gold_type: Literal[exact,links,files], superseded_values: list[str], source_event_ids: list[str])`.
- `Task(task_id, corpus, T, trigger_event_id, entity, kind: Literal[fast,batch], gold: dict[str, Any], gold_coverage: dict[str,bool])` — gold groups `people/category/place/text`.
- `RouterRecord(event_id, t, score, lane, policy, budget_pct, baseline_key_used, features_fired: dict)`.
- `Answer(probe_id, arm, text, cited_ids, tokens_read, tool_calls, latency_s)`.
- `GradeResult(item_id, metric, value, details: dict)`.
- `SnapshotRef(sha, T_last_event, last_event_id, lane)`.
- `RunManifest(run_id, created_at, config_hash, replay_hash, probe_hash, git_sha, model_ids, prices, seeds, prereg_ref)`.

`harnext_eval.providers.llm.LLMProvider`: `complete(system: str, user: str, *, json_schema: dict | None = None, max_tokens: int) -> LLMResult(text, json, usage)`.
`harnext_eval.providers.embeddings.EmbeddingsProvider`: `embed(texts: list[str]) -> np.ndarray`.
`harnext_eval.providers.tokenizer.count_tokens(text) -> int` (approximate, deterministic).

`harnext_eval.registry.Experiment` protocol: `name: str`; `run(cfg: EngineConfig, corpus: CorpusHandle, out_dir: Path, seed: int) -> ExperimentResult`; `chart(result: ExperimentResult, out_dir: Path) -> list[Path]`. `ExperimentResult(name, metrics: dict[str, float], tables: dict[str, pd.DataFrame], artifacts: list[Path], primary: dict)`.

`harnext_eval.corpus.CorpusHandle(name, replay_path, probes_path | None, tasks_path | None, window, meta: dict)`.

`harnext_eval.stores.base.StoreHandle(layout, org_id, root, backend, snapshots_csv)` with `snapshot(T) -> SnapshotRef`, `materialise(ref) -> Path` (temp checkout), `read(ref, relpath)`, `list_files(ref)`.

## 5. Task split (ownership = directories; no task edits another's directory)

| Task | Owns | Depends on |
|---|---|---|
| **T0 foundation** (sequential, first) | `apps/eval/{pyproject.toml,README.md,configs/}`, `src/harnext_eval/{__init__,config,types,manifest,registry,cli}.py`, `providers/`, `corpus/synthetic.py`, `corpus/__init__.py` (CorpusHandle), `stores/base.py`, `tests/test_core/`; root `pyproject.toml` workspace registration; `Makefile` targets `eval-smoke`, `eval-test`; `.gitignore` for `out/` | — |
| T1 corpus (real) | `corpus/{keys,pony_mail,jira,gharchive,build_replay}.py`, `tests/test_corpus/` + fixtures | T0 |
| T2 replay | `replay/`, `tests/test_replay/` | T0, uses `stores/base.py` |
| T3 probes | `probes/`, `tests/test_probes/` | T0 |
| T4 grade + health | `grade/`, `health/`, `tests/test_grade/` | T0 |
| T5 stores | `stores/build_s*.py`, `stores/templated.py`, `stores/vector_index.py`, `tests/test_stores/` | T0 (uses `replay/driver.py` interface described in §6 — if absent, T5 implements a minimal local fold loop inside `stores/`) |
| T6 E1 | `e1/`, `tests/test_e1/` | T0 |
| T7 E2 + E3 | `agents/reader.py`, `e2/`, `e3/`, `tests/test_e2e3/` | T0; consumes `probes`, `grade`, `health`, `stores` via the interfaces in §4/§6 |
| T8 E4 + E5 | `agents/envelope.py`, `e4/`, `e5/`, `tests/test_e4e5/` | T0; consumes `grade/action.py`, `stores` |
| T9 E6 | `e6/`, `tests/test_e6/` | T0; Kafka optional (must degrade to an in-process queue for tests) |
| T10 stats + report | `stats/`, `report/`, `tests/test_report/` | T0 |
| **T11 integration** (after wave 1) | `cli.py` wiring, `Makefile`, cross-module fixes, `make eval-smoke` green on the synthetic corpus with `baseline-minimal.yaml` | all |
| **R1–R3 reviews** | read-only review agents per module against the spec; fix tasks scoped to the findings | T11 |

Rules for every implementer: read `PLAN.md` and the relevant spec sections first; only create/modify files under your owned paths; import shared interfaces from T0, never copy them; keep everything runnable offline with the fake providers; write tests under your own `tests/` dir and make `uv run pytest apps/eval/tests/<yours>` pass; do not run `git commit`; do not reformat files outside your paths; finish by writing `apps/eval/STATUS/<task>.md` (what was built, how to run it, what is stubbed, open questions).

## 6. Cross-task contracts beyond §4

- **In-process replay driver** (`replay/driver.py`, T2): `run_pipeline(events: Iterable[EvalEvent], cfg: EngineConfig, store: StoreHandle, *, cutoff: datetime | None, on_decision: Callable[[RouterRecord], None] | None) -> DriverStats`. It applies the router (rules + optional deviation scorer from `e1.policies` if enabled; T2 wires a `RouterPolicy` protocol `score(event) -> float` and `rules(event) -> str | None` so E1's policies plug in later), the session window (`WindowManager` semantics on the *event clock*, not wall clock), and for each closed window / fast event calls `store.fold(events, lane)`; records a `SnapshotRef` per fold. No Kafka.
- **Store fold** (`stores/base.py`, T0 defines; T5 implements per layout): `StoreHandle.fold(events: list[EvalEvent], lane: str) -> SnapshotRef` — S0 writes files directly; S1 runs the templater; S2/S3 run the builder harness via `BuildRunner`-equivalent local call with the layout's seed/prompt; S4 updates the vector index; S5 both.
- **Reader** (`agents/reader.py`, T7): `answer(probe: Probe, material: Material, cfg) -> Answer` where `Material` is produced by arm builders in `e2/arms.py` (`A0..A4`) honouring `budget_tokens` via `count_tokens`.
- **Envelope** (`agents/envelope.py`, T8): `build(task: Task, snapshot: SnapshotRef, variant: str, cfg) -> Envelope(sections: dict[str,str], tokens_by_section, tools: list[str])`.
- **Graders** (T4) are pure functions over `(prediction, gold)` returning `GradeResult`; E2/E4 call them.
- **Charts** (T10): `report/charts.py` exposes one function per chart named in the spec outputs (`calibration`, `operating_curves`, `e2_family_bars`, `e3_curve`, `erosion`, `e4_envelopes`, `e5_pareto`, `e6_burst_slo`, `self_amplification`, `demand_curve`); experiments call them from `chart()`.

## 7. Validity checks in code

Each experiment's `run()` must emit `checks: dict[str, bool | float]` in `ExperimentResult.metrics` for the spec's validity checks that are computable (e.g. `e2.floor_retrieve_everything >= 0.9`, `e2.prior_leq_0.3`, `e1.random_scorer_at_prevalence`, `e3.same_input_hash`, `e6.generator_p99_skew_ms <= 1`). The report shows them as a pass/fail table. The leakage gate is applied by `replay/gate.py` and its pass count is a check in every experiment that uses snapshots.

## 8. Definition of done for this iteration

- `make eval-test` (all `apps/eval/tests`) green; `ruff` and `pyright` clean on `apps/eval`.
- `make eval-smoke` runs E1–E6 on the synthetic corpus with `baseline-minimal.yaml` in < 10 min on a laptop, writes `out/<run>/report.html` with charts and the check table.
- `harnext-eval run --config configs/s1-templated.yaml` produces a second report; the two are comparable (same probe hash).
- Every experiment's code references the spec section it implements in its module docstring.
