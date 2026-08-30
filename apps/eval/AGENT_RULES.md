# Rules for implementer agents

1. Read `apps/eval/PLAN.md` fully, then the sections of `docs/evaluation-spec.md` your task implements (§7 experiment cards, §4 ground truth, §5 shared infrastructure, §11 layout, §12 decisions).
2. Only create or modify files under the paths your task owns (listed in PLAN.md §5). Never edit another task's directory. If you need something from a directory you do not own, code against the interface described in PLAN.md §4/§6 and, if it does not exist yet, add a *minimal* local stub under your own directory with a `# TODO(integration)` comment.
3. Import shared types/interfaces from `harnext_eval.types`, `harnext_eval.config`, `harnext_eval.registry`, `harnext_eval.providers`, `harnext_eval.stores.base`. Never redefine them.
4. Everything must run offline with the fake providers and the synthetic corpus. No network calls in tests. Real-network code paths must be behind explicit flags and covered by fixture-based tests.
5. Python 3.12+, type hints, pydantic v2, ruff-clean (`uv run ruff check apps/eval`), pyright standard mode. Module docstrings cite the spec section implemented (e.g. "Implements docs/evaluation-spec.md §7 E2").
6. Tests: under `apps/eval/tests/<your dir>/`; `uv run pytest apps/eval/tests/<your dir> -q` must pass before you finish. Use `uv run --package harnext-eval ...` or plain `uv run` from the repo root.
7. Do NOT run `git commit`, `git add`, `git stash`, or any git command that changes state. Do not run repo-wide formatters. Do not modify root `pyproject.toml`, `uv.lock`, or `Makefile` unless your task explicitly owns them.
8. Finish by writing `apps/eval/STATUS/<task-id>.md`: what was built (file list), how to run it, what is stubbed/deferred, open questions for integration, and the exact test command you ran with its result.
