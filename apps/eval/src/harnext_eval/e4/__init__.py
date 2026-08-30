"""E4 context-envelope evaluation (docs/evaluation-spec.md §7 E4)."""

from harnext_eval.e4.run import E4Experiment, run_e4
from harnext_eval.e4.tasks import build_batch_tasks, build_tasks, select_fast_tasks

__all__ = [
    "E4Experiment",
    "build_batch_tasks",
    "build_tasks",
    "run_e4",
    "select_fast_tasks",
]
