# API isolation audit: baseline-minimal and s1-templated

Date: 2026-08-30  
Scope: `apps/eval/src`, the eval tests, and the imported code in `apps/builder`, `apps/classifier`, and `packages/shared`.  
Question: whether the default `baseline-minimal.yaml` and `s1-templated.yaml` evaluation runs contact an AI/network API or read API credentials.

## Verdict: ISOLATED

For both audited configurations, no reachable path contacts OpenAI, Anthropic, OpenRouter, NVIDIA NIM, another network LLM/embedding service, Kafka, or a corpus-download endpoint. No reachable path reads an API key. The baseline and S1 all-experiment smoke runs and the complete eval test suite succeeded with all API-key variables absent and with network egress removed. Syscall traces contained zero `AF_INET`/`AF_INET6` sockets and zero non-Unix `connect()` calls.

This verdict is configuration-specific, not a claim that the repository contains no network-capable code. Real Anthropic/Claude Code, Harnext/OpenRouter/NIM, Kafka, and corpus-fetch implementations exist behind explicit provider, harness, transport, or function choices. Those paths and their activation conditions are documented below.

One defense-in-depth caveat does not change the verdict: the git builder backend passes the parent environment to every harness subprocess (`apps/builder/src/harnext_builder/agentfs/git_backend.py:57-64`). Thus, in an ordinary shell that contains API keys, a fake-harness child receives those environment entries even though the audited fake code never reads them. The sanitized dynamic runs proved absence of reads/requirements, but the subprocess environment should still be narrowed.

## 1. Configuration and reachability trace

### Baseline and S1 selections

- `apps/eval/configs/baseline-minimal.yaml:8-19` selects `layout: S0`, `builder.harness: fake`, `builder.model: null`, `reader.provider: fake`, and `embeddings.provider: fake`.
- `apps/eval/configs/s1-templated.yaml:8-12` differs only in the relevant area by selecting `layout: S1`; builder, reader, and embeddings remain fake.
- The schema permits builder `fake|claude_code`, reader `fake|anthropic`, and embeddings `fake|openai` at `apps/eval/src/harnext_eval/config.py:45-69`. A non-fake builder requires a model at lines 55-59.

### Providers actually constructed

1. The CLI loads the YAML at `apps/eval/src/harnext_eval/cli.py:310-320`.
2. Every store is configured at `apps/eval/src/harnext_eval/cli.py:146-161`. It passes `cfg.builder.harness` and `cfg.builder.model` verbatim, but constructs `FakeEmbeddings(dim=...)` unconditionally at line 159. Consequently, `FakeEmbeddings` is the only embedding provider constructed by the CLI under both audited configs. In fact, the currently accepted `embeddings.provider: openai` value is ignored and there is no OpenAI embedding adapter in `providers/embeddings.py`.
3. Store runtime defaults are also fake at `apps/eval/src/harnext_eval/stores/layouts.py:24-32,38-67`.
4. Reader selection occurs at `apps/eval/src/harnext_eval/agents/reader.py:65-77`. `provider == "fake"` returns `FakeLLM` at lines 70-71. `AnthropicLLM` is constructed only in the separate `provider == "anthropic"` branch at lines 72-76.
5. E4's evaluator independently defaults to `FakeLLM` at `apps/eval/src/harnext_eval/e4/run.py:412-430`. Claims grading special-cases `FakeLLM` without a provider call at `apps/eval/src/harnext_eval/grade/claims.py:52-63,77-91`.

A sanitized runtime constructor check printed:

```text
FakeLLM FakeLLM
fake fake
FakeEmbeddings
anthropic_sdk_loaded False
claude_harness_loaded False
harnext_harness_loaded False
```

### S2/S3/S5 with `harness=fake`

S2 calls the shared builder runner at `apps/eval/src/harnext_eval/stores/build_s2.py:43-61`; S3 creates a `HarnessRequest(harness=runtime.harness)` and invokes the local runner subprocess at `apps/eval/src/harnext_eval/stores/build_s3.py:48-80`; S5 calls S3 first at `apps/eval/src/harnext_eval/stores/build_s5.py:59-67`.

The subprocess reads the request and calls `get_harness(req.harness)` at `apps/builder/src/harnext_builder/harness/runner.py:78-95`. The registry uses mutually exclusive lazy imports:

- `claude_code` imports/constructs `ClaudeCodeHarness` at `apps/builder/src/harnext_builder/harness/registry.py:9-12`.
- `fake` imports/constructs only `FakeHarness` at lines 13-16.
- `harnext` imports/constructs `HarnextHarness` at lines 17-20.

Therefore `harness=fake` cannot instantiate either real harness through this path. This was also checked dynamically:

- `harnext-eval stores --layouts S2,S3,S5` over one synthetic event exited 0.
- All three `usage.jsonl` rows recorded `"harness": "fake"`, `"model": "fake"`, zero tokens, and zero cost.
- A combined file/network syscall trace opened `harnext_builder/harness/fake.py` for each store. It did not access `harnext_builder/harness/claude_code.py`, `harnext_builder/harness/harnext.py`, `.env`, `.claude`, credential paths, or key-named files. It contained zero internet-family sockets and zero non-Unix connects.
- The full test suite also explicitly parametrizes all layouts with `configure_store(..., harness="fake", embeddings=FakeEmbeddings(...))` and asserts every S2/S3/S5 usage row says fake at `apps/eval/tests/test_stores/test_layouts.py:56-87`.

The all-experiment `--smoke` command deliberately builds S3 even when the configured layout is S0 or S1 (`apps/eval/src/harnext_eval/cli.py:173-182`). That does not weaken isolation: the same fake harness value is passed into this additional S3 store. Each completed audit run recorded 60 S3 usage rows, all `harness=fake`, `model=fake`.

### Anthropic lazy import

`apps/eval/src/harnext_eval/providers/llm.py:145-192` defines `AnthropicLLM`, but merely importing this module does not import the Anthropic SDK. The SDK import is inside `AnthropicLLM.complete()` at line 161; client construction is line 166 and the network request is `client.messages.create(...)` at lines 170-176. Because both audited configs select `FakeLLM`, this method is never entered. The runtime module check above confirmed that `anthropic` was absent from `sys.modules`.

## 2. Static network/API/key scan

Commands searched Python, TOML, and YAML under the scoped directories for `anthropic`, `openai`, `claude_agent_sdk`, Harnext SDK references, `httpx`, `requests`, `aiohttp`, `urllib`, `socket`, API-key/environment names, dynamic imports, Kafka constructors, and URL literals. The meaningful hits are exhaustively classified here. There were no source imports of the OpenAI SDK, `httpx`, `requests`, `aiohttp`, or `socket` in the scoped directories.

### LLM/embedding SDK and network-library hits

| Hit | Capability | Reachable in baseline/S1? |
|---|---|---|
| `apps/eval/src/harnext_eval/providers/llm.py:5,161-176` | Dynamic Anthropic SDK import, client, and messages API call | No; only inside `AnthropicLLM.complete()` |
| `apps/eval/src/harnext_eval/agents/reader.py:9,65-77,100-119` | Imports both provider classes and selects/calls one | Yes, but selects only `FakeLLM` |
| `apps/eval/src/harnext_eval/providers/__init__.py:4` | Re-exports `AnthropicLLM`; does not load Anthropic SDK | Importable but inert |
| `apps/builder/pyproject.toml:9,14` | Declares `claude-agent-sdk` and the Harnext SDK distribution as dependencies | Installed/available, but availability alone does not select either lazy harness branch |
| `apps/builder/src/harnext_builder/harness/claude_code.py:22-33,72-102` | Imports Claude Agent SDK and invokes `query()` | No; lazy registry branch not selected |
| `apps/builder/src/harnext_builder/harness/harnext.py:26-38,78-117` | Imports `harnext_sdk`, constructs options, invokes `query()` | No; not selectable by current eval YAML schema and registry branch not selected |
| `apps/eval/src/harnext_eval/corpus/jira.py:11-12,195-235` | `urllib.parse`, `Request`, `urlopen`-defaulted explicit Jira fetch | No; synthetic corpus path does not call `fetch()` |
| `apps/eval/src/harnext_eval/corpus/pony_mail.py:17-18,138-169` | `urllib.parse` and explicit Pony Mail download | No |
| `apps/eval/src/harnext_eval/corpus/gharchive.py:13,101-121` | `Request` and explicit GH Archive download | No |
| `apps/eval/src/harnext_eval/health/store_health.py:10` | `urllib.parse.unquote/urlsplit` only | Reachable but purely local string parsing |
| `apps/eval/src/harnext_eval/e6/run.py:535-569` | Lazy `aiokafka` producer/consumer and broker start | No; default `RunnerConfig.transport` is `in-process` at lines 50-69 |
| `apps/eval/src/harnext_eval/replay/kafka_replay.py:33-61` | Lazy Kafka producer when no producer is injected | No; eval `run` uses the in-process replay driver |
| `apps/classifier/src/harnext_classifier/kafka.py:5-11` and `main.py:9,42-54` | Kafka producer/consumer | No; eval imports only `harnext_classifier.rules` |
| `apps/builder/src/harnext_builder/consumer.py:12,35-46`, `dlq.py:7-15`, `main.py:40-55` | Kafka consumer/producer | No; eval imports builder store/harness protocol modules, not builder service startup |

`urllib.parse` imports do not themselves perform I/O. All three corpus network operations are in explicitly named `fetch()` functions with injectable openers. The synthetic/replay-only eval CLI path at `apps/eval/src/harnext_eval/cli.py:85-109` never calls them.

### Environment/API-key hits

There are no API-key environment reads in `apps/eval/src`, `apps/classifier/src`, or `packages/shared/src`.

The builder package has the following credential-capable settings:

| Hit | Behavior | Reachable in baseline/S1? |
|---|---|---|
| `apps/builder/src/harnext_builder/settings.py:7-8` | `BaseSettings` can read process env and `.env` when instantiated | No; fake harness does not instantiate `BuilderSettings` |
| `apps/builder/src/harnext_builder/settings.py:22-23` | `HARNEXT_HARNESS` alias and automatic `ANTHROPIC_API_KEY` field name | No |
| `apps/builder/src/harnext_builder/settings.py:34-44` | Reads `HARNEXT_PROVIDER`, `HARNEXT_MODEL`, `HARNEXT_API_KEY`, `OPENROUTER_API_KEY`, `NVIDIA_NIM_API_KEY`, `NVIDIA_API_KEY`, `HARNEXT_API_KEY_ENV`, and `HARNEXT_PERMISSION_MODE` | No; only a `HarnextHarness` construction instantiates these settings |
| `apps/builder/src/harnext_builder/harness/harnext.py:81-93` | Constructs `BuilderSettings`, reads the resolved key field, and forwards it under the configured env name | No |
| `apps/builder/src/harnext_builder/harness/claude_code.py:3-4` | Documents that the external SDK/CLI uses `ANTHROPIC_API_KEY` or OAuth credentials | No; real harness not imported |
| `apps/builder/src/harnext_builder/agentfs/git_backend.py:57-64` | Inherits the complete process environment into any harness subprocess | Reachable, but the fake harness does not inspect key variables; sanitized audit environment had none |
| `apps/builder/src/harnext_builder/harness/runner.py:92-95` | Reads only `REQUEST_PATH` and `RESULT_PATH` | Yes; these are temporary local paths, not credentials |
| `packages/shared/src/harnext_shared/migrations/env.py:42` | Reads `DATABASE_URL` for Alembic | No; migrations are not invoked |

`AnthropicLLM` passes its optional `api_key` (which is `None` when built by the reader factory) to the external Anthropic client at `providers/llm.py:148-166`; the SDK may then consult `ANTHROPIC_API_KEY`. That client is never constructed in the audited configs.

### Hard-coded URLs/endpoints

Every hard-coded URL-like hit in scoped source is below:

| Hit | Classification and reachability |
|---|---|
| `apps/eval/src/harnext_eval/corpus/pony_mail.py:145` — `https://lists.apache.org/api` | Real network base URL, only in explicit `fetch()`; unreachable |
| `apps/eval/src/harnext_eval/corpus/gharchive.py:105` — `https://data.gharchive.org` | Real network base URL, only in explicit `fetch()`; unreachable |
| `apps/eval/src/harnext_eval/replay/kafka_replay.py:36` — `localhost:9092` | Local broker default, only broker-backed replay; unreachable |
| `apps/builder/src/harnext_builder/settings.py:11` and `apps/classifier/src/harnext_classifier/settings.py:9` — `localhost:9092` | Service defaults, service startup unreachable from eval |
| `apps/builder/src/harnext_builder/settings.py:14`, `apps/classifier/src/harnext_classifier/settings.py:10`, `packages/shared/src/harnext_shared/migrations/env.py:33` — `sqlite+aiosqlite:///...` | Local filesystem database URLs, not network |
| `packages/shared/src/harnext_shared/db.py:122` — `https://x.ladesk.com` | Example value in a model docstring, not an executed request |
| `packages/shared/src/harnext_shared/envelope.py:3` — CloudEvents GitHub URL | Documentation citation only |

Jira's endpoint is constructed from the caller-supplied `base_url` at `apps/eval/src/harnext_eval/corpus/jira.py:195-234`, so it is network-capable without containing a fixed host.

No LLM provider base URL is hard-coded in the audited application source; endpoint selection is delegated to the external Anthropic, Claude Agent, or Harnext SDK/CLI.

## 3. Dynamic evidence

`unshare -rn true` succeeded with exit code 0, so every dynamic audit was run in a new network namespace. `strace -f -e trace=network` was used inside the namespace as an additional check for attempted socket activity. The environment was created with `env -i` and contained no `*_API_KEY`, `*_KEY`, `OPENAI*`, `ANTHROPIC*`, `OPENROUTER*`, `NVIDIA*`, or `HARNEXT_*` variables. `UV_NO_SYNC=1` prevented dependency synchronization/downloads.

### Baseline all-experiment smoke run

Command (the CLI exposes `--smoke`; size flags were added to keep the audit bounded):

```bash
env -i PATH="$PATH" HOME="$HOME" UV_NO_SYNC=1 \
  unshare -rn strace -f -e trace=network -o /tmp/audit-baseline-strace.txt \
  uv run harnext-eval run \
  --config apps/eval/configs/baseline-minimal.yaml \
  --corpus synthetic --all --smoke \
  --event-count 60 --entity-count 8 --per-family 2 \
  --out /tmp/audit-run
```

- Exit code: **0**.
- Completed E1-E6 and wrote `/tmp/audit-run/20260830T165445Z-baseline-minimal/report.html`.
- Trace: 6,949 lines; 50 `AF_UNIX` sockets/connects (failed local `/var/run/nscd/socket` lookups), **0 `AF_INET`/`AF_INET6` occurrences**, **0 non-Unix `connect()` calls**.
- The forced S3 store recorded 60/60 folds as `harness=fake`, `model=fake`.

### S1 all-experiment smoke run

```bash
env -i PATH="$PATH" HOME="$HOME" UV_NO_SYNC=1 \
  unshare -rn strace -f -e trace=network -o /tmp/audit-s1-strace.txt \
  uv run harnext-eval run \
  --config apps/eval/configs/s1-templated.yaml \
  --corpus synthetic --all --smoke \
  --event-count 60 --entity-count 8 --per-family 1 \
  --out /tmp/audit-run-s1
```

- Exit code: **0**.
- Completed E1-E6 and wrote `/tmp/audit-run-s1/20260830T165927Z-s1-templated/report.html`.
- Trace: 6,430 lines; 50 `AF_UNIX` sockets/connects, **0 `AF_INET`/`AF_INET6` occurrences**, **0 non-Unix `connect()` calls**.
- The forced S3 store recorded 60/60 folds as `harness=fake`, `model=fake`.

An earlier attempt with only 24 events exited 1 before store construction because the generated corpus had zero temporal-probe candidates. This was a too-small smoke input, not a credential/network failure; the corrected 60-event run above is the audit result.

### Eval pytest suite

```bash
env -i PATH="$PATH" HOME="$HOME" UV_NO_SYNC=1 \
  unshare -rn strace -f -e trace=network -o /tmp/audit-pytest-strace.txt \
  uv run pytest apps/eval/tests -q
```

- Exit code: **0**.
- Result: **121 passed in 119.01s**.
- Trace: 2,542 lines; 52 `AF_UNIX` sockets/connects, **0 `AF_INET`/`AF_INET6` occurrences**, **0 non-Unix `connect()` calls**.

### Direct fake-store file/network trace

After generating one synthetic event locally, this command exercised all three builder-backed layouts:

```bash
env -i PATH="$PATH" HOME="$HOME" UV_NO_SYNC=1 \
  unshare -rn strace -f -e trace=network,file \
  -o /tmp/audit-fake-stores-file-strace.txt \
  uv run harnext-eval stores \
  --config apps/eval/configs/baseline-minimal.yaml \
  --replay /tmp/audit-one-event.jsonl \
  --out /tmp/audit-fake-stores --layouts S2,S3,S5 --seed 1
```

- Exit code: **0**.
- S2, S3, and S5 each built one snapshot and recorded a fake/fake, zero-token, zero-cost usage row.
- Trace: 22,381 lines; **0 internet-family sockets**, **0 non-Unix connects**.
- No sensitive file/path match and no real-harness source access; `fake.py` was the selected harness module.

## 4. First-use libraries and test fixtures

### pandas / Parquet

Pandas itself made no network call in the full runs or tests. A fresh-cache, network-isolated direct `DataFrame.to_parquet()` attempt exited 1 because neither `pyarrow` nor `fastparquet` is installed; its 409-line trace had zero internet-family sockets and zero non-Unix connects. This is expected by eval code: `_write_scores()` catches `ImportError` and writes a pandas-table JSON fallback plus a format marker at `apps/eval/src/harnext_eval/e1/run.py:196-214`. The successful baseline output included that fallback marker. There is therefore no pyarrow first-use network path in the installed environment.

### matplotlib font cache

A separate first-use run forced new `MPLCONFIGDIR` and `XDG_CACHE_HOME` directories, imported pandas/matplotlib, rendered a plot, and saved a PNG. It exited 0, created one matplotlib cache file, and its 409-line trace contained 50 local Unix sockets but zero internet-family sockets and zero non-Unix connects. The full eval runs also rendered all report charts successfully while network-isolated.

Matplotlib's first use scans local fonts/builds a local cache; no font download or remote lookup was observed.

### Other libraries and fixtures

- The full E1-E6 runs exercised NumPy, SciPy, scikit-learn/PyOD, rank-bm25, datasketch, hdrhistogram, Jinja2, pandas, and matplotlib without any internet-family syscall.
- Pony Mail tests read only `apps/eval/tests/test_corpus/fixtures/pony-dev-2026-01.mbox` and `pony-stats-2026-01.json` (`test_pony_mail.py:8-36`).
- Jira tests read `jira-page-1.json` and `jira-page-2.json` and inject a local `fetch_page` callable (`test_jira.py:9-29`). They do not call the module's network `fetch()`.
- GH Archive tests gzip and parse the local `gharchive-hour.jsonl` fixture (`test_gharchive.py:9-19`). They do not call its network `fetch()`.
- Kafka replay tests inject `FakeProducer` and fake clock/sleep functions (`test_replay/test_kafka_replay.py:14-48`), so no broker is used.
- No test or fixture contains a `urlopen`, `requests`, `httpx`, `aiohttp`, or live URL call. The syscall trace independently confirms no internet access anywhere in 121 tests.

## 5. Network-capable paths and exact activation conditions

| Path | What enables it | Credential behavior |
|---|---|---|
| Eval Anthropic reader | Change `engine.reader.provider: fake` to `anthropic`. Any E2 reader answer then constructs `AnthropicLLM`; `builder.model` supplies its model if non-null, otherwise it defaults to `claude-sonnet-5`. | External Anthropic client may read `ANTHROPIC_API_KEY` because eval passes `api_key=None`. |
| Claude Code builder | Change `engine.builder.harness: fake` to `claude_code` **and** set `engine.builder.model` to a non-null model. Also build S2/S3/S5: set `engine.store.layout` accordingly, use `stores --layouts ...`, or select E4 with `--smoke` (which adds S3). `configs/s3-curated.yaml:8-9` is an existing enabling profile. | Claude Agent SDK/CLI uses `ANTHROPIC_API_KEY` or seeded OAuth credentials. |
| Harnext/OpenRouter/NIM builder | No valid eval YAML change currently enables it: `BuilderConfig.harness` rejects `harnext`. It is reachable only by direct programmatic `configure_store(harness="harnext")` or after extending the schema, then building S2/S3/S5. | `BuilderSettings` reads `HARNEXT_API_KEY`, `OPENROUTER_API_KEY`, `NVIDIA_NIM_API_KEY`, or `NVIDIA_API_KEY` and forwards the selected key to Harnext CLI. |
| OpenAI embeddings | No working code path exists. `engine.embeddings.provider: openai` is accepted by validation but ignored by the CLI, which still constructs `FakeEmbeddings`. | No OpenAI SDK import or `OPENAI_API_KEY` read exists in scoped source. |
| E6 Kafka | Programmatically pass `RunnerConfig(transport="kafka", kafka_bootstrap_servers="host:port", ...)` to the E6 runner. This option is not exposed by the audited YAML/CLI run path. | No AI key; opens broker network connections. |
| Replay Kafka | Call `replay_jsonl(...)` without injecting a producer. | No AI key; defaults to `localhost:9092`. |
| Pony Mail download | Directly call `harnext_eval.corpus.pony_mail.fetch(...)`. | No AI key; opens HTTPS to the configured base URL. |
| Jira download | Directly call `harnext_eval.corpus.jira.fetch(base_url=..., ...)`. | No AI key; opens HTTPS/HTTP to caller-selected Jira. |
| GH Archive download | Directly call `harnext_eval.corpus.gharchive.fetch(...)`. | No AI key; opens HTTPS to GH Archive. |
| Builder/classifier services | Start `harnext-builder` or `harnext-classifier` service entry points. Eval does not do so. | Kafka/database environment may be read; builder harness selection may additionally enable AI credentials. |

## 6. Recommendations

1. Add an explicit top-level `offline: true` setting, default it to true in eval profiles, and enforce it in a single provider/transport factory. Raise before constructing any non-fake reader, non-fake builder harness, non-fake embedding provider, Kafka transport, or network corpus fetcher.
2. Stop passing the full parent environment to fake harness subprocesses. Use an allowlist (`PATH`, locale, temporary request/result paths, and required runtime variables) so API keys are not even present in a fake child.
3. Make embeddings selection honest and centralized. Either implement an explicit OpenAI adapter behind `offline: false`, or reject `embeddings.provider: openai`; silently constructing `FakeEmbeddings` for an `openai` config can produce misleading manifests.
4. Add a startup summary that records resolved provider classes (`FakeLLM`, `FakeEmbeddings`, `FakeHarness`) and an `offline_enforced` flag in the manifest, without recording environment values.
5. Add an offline CI job using a network namespace or equivalent egress-deny wrapper. Assert zero internet-family syscalls for `baseline-minimal.yaml`, `s1-templated.yaml`, and the eval test suite.
6. Add unit tests that poison real-provider constructors/imports and fail if baseline/S1 touches them. Preserve the existing S2/S3/S5 assertion that all usage rows identify the fake harness.
7. Keep corpus downloads in separate explicit commands and require `offline: false` for them. The present `fetch()` separation is good, but a central guard would prevent future accidental wiring from `run`.
8. Prefer narrow imports such as `from harnext_shared.envelope import CloudEvent` where practical. `from harnext_shared import CloudEvent` executes the broad shared package initializer and imports DB/auth modules even though it does not currently open a connection.

## Bottom line

Under `baseline-minimal.yaml` and `s1-templated.yaml`, including the extra S3 path forced by `--all --smoke`, the evaluation framework is offline: only `FakeLLM`, `FakeEmbeddings`, and `FakeHarness` are selected; Anthropic/Claude/Harnext lazy imports are not triggered; API-key settings are not instantiated; and dynamic network traces show no non-local network activity.
