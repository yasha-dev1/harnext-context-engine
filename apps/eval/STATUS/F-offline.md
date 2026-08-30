# F-offline status

## Built

- `src/harnext_eval/config.py`: added top-level `offline: bool = True` and limited
  embeddings to the implemented `fake` adapter.
- `src/harnext_eval/providers/factory.py`: added the central offline guard,
  `OfflineViolation`, reader/embeddings/harness resolution, Kafka/corpus-fetch
  guard inputs, and a non-secret resolved-provider summary.
- `src/harnext_eval/cli.py`: routes store/provider startup through the factory,
  guards runs/stores/network-fetch requests, and writes the provider summary to
  `manifest.json`.
- `src/harnext_eval/manifest.py`: added the manifest provider-summary field.
- `src/harnext_eval/agents/reader.py` and `src/harnext_eval/e4/run.py`: route
  default reader and E4 builder-harness construction through the factory.
- `apps/builder/src/harnext_builder/agentfs/{git_backend,agentfs_backend}.py`:
  narrowed harness subprocess environments to the requested runtime allowlist;
  provider-prefixed variables are included only for a parsed non-fake harness.
- `tests/test_core/test_offline.py`: covers poisoned real-provider imports,
  offline Anthropic rejection, unsupported embeddings, Kafka/fetch guards,
  manifest resolution metadata, and fake-child API-key exclusion.
- `configs/*.yaml`: explicitly set offline true for baseline/S1/E6 and false for
  S3 curated.

## Run

```bash
uv run pytest apps/eval/tests apps/builder/tests -q
uv run pytest apps/builder/tests -q
uv run ruff check apps/eval apps/builder
```

## Verification

- `uv run pytest apps/eval/tests apps/builder/tests -q`:
  **159 passed in 75.67s**.
- `uv run pytest apps/builder/tests -q`:
  **31 passed in 15.36s**.
- Ruff over every owned changed Python file: **all checks passed**.
- `uv run ruff check apps/eval apps/builder`: one pre-existing failure remains in
  unowned `apps/builder/tests/test_harnext.py:5` (`I001` import ordering). Per the
  task's ownership restriction, that file was not modified.
- `git diff --check`: passed.

## Stubbed or deferred

- Network corpus extractors remain intentionally separate and are not newly
  wired into the eval CLI. A `corpus --fetch` request must supply a config and
  passes through the offline guard before the command reports that network
  extractor wiring is unavailable.

## Integration notes

- No secrets or environment values are written to the manifest; it contains
  only resolved class names and `offline_enforced`.
- After the successful combined test run, an external workspace change deleted
  unowned `src/harnext_eval/corpus/synthetic.py`. It was preserved rather than
  restored, but fresh eval imports will remain blocked until its owner resolves
  that deletion.
- No git state-changing command was run.
