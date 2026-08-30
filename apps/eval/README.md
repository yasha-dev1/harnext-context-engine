# harnext evaluation framework

This package implements the evaluation design in
[`docs/evaluation-spec.md`](../../docs/evaluation-spec.md). It is offline-first:
the synthetic corpus and fake providers require no model keys or Kafka broker.

From the repository root:

```bash
uv sync
make eval-smoke
make eval-test
```

`make eval-smoke` runs the currently registered experiments against the
deterministic synthetic corpus. Evaluation artifacts are written beneath
`apps/eval/out/`.
