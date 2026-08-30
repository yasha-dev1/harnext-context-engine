# F-embed — Round 3 pinned embeddings

## Built

- `src/harnext_eval/providers/embeddings.py`: lazy Voyage and OpenAI embedding adapters with pinned model/revision identities and validated, normalised response matrices.
- `src/harnext_eval/providers/factory.py`: strict provider construction behind the existing offline guard.
- `src/harnext_eval/config.py`: embedding provider/model/revision validation.
- `configs/s3-curated.yaml`: pinned Voyage embedding selection.
- `src/harnext_eval/manifest.py`: configured embedding model and revision in `model_ids`, alongside the resolved provider class summary.
- Provider, offline, manifest, A3, and S4 regression tests.

## Run

Focused regression command:

`uv run pytest apps/eval/tests/test_core/test_offline.py apps/eval/tests/test_core/test_providers.py apps/eval/tests/test_stores/test_layouts.py::test_configured_real_adapter_reaches_a3_and_s4_without_network -q`

Result: 25 passed.

Owned-file quality checks:

`uv run ruff check apps/eval/src/harnext_eval/providers/embeddings.py apps/eval/src/harnext_eval/providers/factory.py apps/eval/src/harnext_eval/config.py apps/eval/src/harnext_eval/manifest.py apps/eval/tests/test_core/test_offline.py apps/eval/tests/test_core/test_providers.py apps/eval/tests/test_stores/test_layouts.py`

Result: all checks passed. Targeted pyright over the four changed source modules: 0 errors.

## Deferred / open questions

- The optional `voyageai` or `openai` SDK must be installed only on hosts that execute a paid real-provider run; imports remain lazy so offline smoke needs neither package.
- No paid API call or evidentiary corpus run was performed, as required by the Round 3 scope.
