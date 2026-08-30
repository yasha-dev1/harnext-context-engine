"""Store-organisation experiment from docs/evaluation-spec.md §7 E3."""

from harnext_eval.e3.run import (
    E3Experiment,
    StoreCondition,
    compute_erosion_slope,
    evaluate_e3,
)

__all__ = ["E3Experiment", "StoreCondition", "compute_erosion_slope", "evaluate_e3"]
