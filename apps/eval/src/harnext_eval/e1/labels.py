"""Post-outcome urgency labels for docs/evaluation-spec.md §4.1 and §7 E1.

Every labelling function in this module receives only the strict suffix after the
candidate event.  The candidate payload is deliberately unavailable to the
functions, which makes the temporal firewall structural rather than conventional.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from harnext_eval.types import EvalEvent

ABSTAIN = -1
NEGATIVE = 0
POSITIVE = 1
Vote = int
OutcomeFunction = Callable[[Sequence[EvalEvent]], Vote]


@dataclass(frozen=True)
class LabelingFunction:
    """Named weak-supervision function evaluated on events strictly after t."""

    name: str
    horizon: timedelta
    apply: OutcomeFunction


@dataclass(frozen=True)
class LabelModelResult:
    """Probabilistic labels plus label-function diagnostics."""

    probabilities: pd.Series
    votes: pd.DataFrame
    diagnostics: pd.DataFrame
    declared_outcome_agreement: float


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _strings(item)
    elif value is not None:
        yield str(value)


def _text(event: EvalEvent) -> str:
    return " ".join(_strings(event.data or {})).casefold()


def _field(event: EvalEvent, *names: str) -> Any:
    data = event.data or {}
    wanted = {name.casefold() for name in names}
    for key, value in data.items():
        if str(key).casefold() in wanted:
            return value
    return None


def _is_committer(event: EvalEvent) -> bool:
    data = event.data or {}
    role = str(data.get("author_association", data.get("role", ""))).casefold()
    actor = str(data.get("author", data.get("from", data.get("actor", "")))).casefold()
    return (
        bool(data.get("is_committer"))
        or role in {"member", "owner", "committer"}
        or ("committer" in actor)
    )


def committer_reply(events: Sequence[EvalEvent]) -> Vote:
    """A committer comment/reply occurred in the supplied one-hour suffix."""

    for event in events:
        kind = event.type.casefold()
        if ("comment" in kind or "mail" in kind or "message" in kind) and _is_committer(event):
            return POSITIVE
    return ABSTAIN


def resolved(events: Sequence[EvalEvent]) -> Vote:
    """The entity was resolved/closed in the supplied 24-hour suffix."""

    for event in events:
        field = str(_field(event, "field", "status")).casefold()
        target = str(_field(event, "to", "state", "status")).casefold()
        if target in {"resolved", "closed", "done", "fixed"} and (
            field in {"", "status", "state"} or "resolved" in _text(event)
        ):
            return POSITIVE
    return ABSTAIN


_PRIORITY = {"trivial": 0, "minor": 1, "normal": 2, "major": 3, "critical": 4, "blocker": 5}


def priority_raised(events: Sequence[EvalEvent]) -> Vote:
    """A later priority transition moves upward."""

    for event in events:
        if str(_field(event, "field")).casefold() != "priority":
            continue
        old = str(_field(event, "from", "old", "previous")).casefold()
        new = str(_field(event, "to", "new", "priority")).casefold()
        if _PRIORITY.get(new, -1) > _PRIORITY.get(old, -1):
            return POSITIVE
    return ABSTAIN


def linked_pr(events: Sequence[EvalEvent]) -> Vote:
    """A pull request was opened/linked in the supplied 24-hour suffix."""

    for event in events:
        kind = event.type.casefold()
        state = str(_field(event, "state", "action")).casefold()
        if ("pull_request" in kind or "pull request" in kind or "github" in event.source) and (
            state in {"open", "opened", "merged"} or any(word in kind for word in ("open", "merge"))
        ):
            return POSITIVE
    return ABSTAIN


def reverted(events: Sequence[EvalEvent]) -> Vote:
    """A PR/commit was explicitly reverted in the supplied 48-hour suffix."""

    for event in events:
        if "revert" in event.type.casefold() or "revert" in _text(event):
            return POSITIVE
    return ABSTAIN


def declared_blocker_critical(events: Sequence[EvalEvent]) -> Vote:
    """A later event declares Blocker/Critical; retained as one noisy LF."""

    for event in events:
        data = event.data or {}
        candidates = [data.get("priority"), data.get("to")]
        state = data.get("state")
        if isinstance(state, dict):
            candidates.append(state.get("priority"))
        if any(str(value).casefold() in {"blocker", "critical"} for value in candidates):
            return POSITIVE
    return ABSTAIN


DEFAULT_LABELING_FUNCTIONS: tuple[LabelingFunction, ...] = (
    LabelingFunction("committer_reply_1h", timedelta(hours=1), committer_reply),
    LabelingFunction("resolved_24h", timedelta(hours=24), resolved),
    LabelingFunction("priority_raised", timedelta(days=30), priority_raised),
    LabelingFunction("linked_pr_24h", timedelta(hours=24), linked_pr),
    LabelingFunction("reverted_48h", timedelta(hours=48), reverted),
    LabelingFunction("declared_blocker_critical", timedelta(days=30), declared_blocker_critical),
)


def apply_labeling_functions(
    events: Sequence[EvalEvent],
    functions: Sequence[LabelingFunction] = DEFAULT_LABELING_FUNCTIONS,
) -> pd.DataFrame:
    """Return LF votes, grouping suffixes by subject and enforcing event_time > t."""

    ordered = sorted(events, key=lambda event: (event.time, event.id))
    by_subject: dict[str, list[EvalEvent]] = defaultdict(list)
    for event in ordered:
        by_subject[event.subject].append(event)
    rows: list[dict[str, int | str]] = []
    for event in ordered:
        subject_events = by_subject[event.subject]
        row: dict[str, int | str] = {"event_id": event.id}
        for function in functions:
            cutoff = event.time + function.horizon
            suffix = tuple(later for later in subject_events if event.time < later.time <= cutoff)
            row[function.name] = function.apply(suffix)
        rows.append(row)
    return pd.DataFrame(rows).set_index("event_id")


class WeightedLabelModel:
    """Small one-sided generative label model for sparse outcome LFs.

    Accuracies are iteratively estimated from agreement with the current soft
    latent label.  Abstentions contribute no likelihood.  A Beta(7, 3) prior
    keeps rare, non-overlapping functions identifiable and encodes the E1
    preregistration expectation that retained LFs are better than chance.
    """

    def __init__(self, *, prior: float | None = None, max_iter: int = 100) -> None:
        self.prior = prior
        self.max_iter = max_iter
        self.accuracies_: np.ndarray | None = None
        self.prevalence_: float | None = None

    def fit_predict(self, votes: pd.DataFrame) -> np.ndarray:
        matrix = votes.to_numpy(dtype=int)
        fired = matrix != ABSTAIN
        positive = matrix == POSITIVE
        observed_rate = float(positive.any(axis=1).mean()) if len(matrix) else 0.0
        prevalence = (
            self.prior if self.prior is not None else float(np.clip(observed_rate, 0.01, 0.5))
        )
        accuracy = np.full(matrix.shape[1], 0.7, dtype=float)
        probabilities = np.full(matrix.shape[0], prevalence, dtype=float)
        for _ in range(self.max_iter):
            logit = np.full(len(matrix), np.log(prevalence / (1.0 - prevalence)))
            for column in range(matrix.shape[1]):
                weight = np.log(accuracy[column] / (1.0 - accuracy[column]))
                logit += np.where(positive[:, column], weight, 0.0)
                logit -= np.where(matrix[:, column] == NEGATIVE, weight, 0.0)
            updated = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
            next_accuracy = accuracy.copy()
            for column in range(matrix.shape[1]):
                covered = fired[:, column]
                if covered.any():
                    expected_correct = np.where(
                        positive[covered, column], updated[covered], 1.0 - updated[covered]
                    ).sum()
                    next_accuracy[column] = (expected_correct + 7.0) / (covered.sum() + 10.0)
            next_accuracy = np.clip(next_accuracy, 0.501, 0.99)
            # Sparse positive-only LFs do not identify class balance. Keep the
            # observed noisy-or prior fixed rather than letting EM inflate it.
            if np.max(np.abs(next_accuracy - accuracy), initial=0.0) < 1e-7:
                probabilities = updated
                accuracy = next_accuracy
                break
            probabilities, accuracy = updated, next_accuracy
        self.accuracies_ = accuracy
        self.prevalence_ = prevalence
        return probabilities


def fit_label_model(votes: pd.DataFrame, *, prior: float | None = None) -> LabelModelResult:
    """Fuse LF votes and report accuracy, coverage, overlap, and conflict."""

    model = WeightedLabelModel(prior=prior)
    probabilities = model.fit_predict(votes)
    assert model.accuracies_ is not None
    matrix = votes.to_numpy(dtype=int)
    diagnostics: list[dict[str, float | str]] = []
    for column, name in enumerate(votes.columns):
        covered = matrix[:, column] != ABSTAIN
        other = np.delete(matrix, column, axis=1)
        overlap = covered & np.any(other != ABSTAIN, axis=1)
        conflict = covered & np.any((other != ABSTAIN) & (other != matrix[:, column, None]), axis=1)
        diagnostics.append(
            {
                "function": name,
                "accuracy": float(model.accuracies_[column]),
                "coverage": float(covered.mean()) if len(matrix) else 0.0,
                "overlap": float(overlap.mean()) if len(matrix) else 0.0,
                "conflict": float(conflict.sum() / max(covered.sum(), 1)),
            }
        )
    declared = "declared_blocker_critical"
    outcome_columns = [name for name in votes.columns if name != declared]
    agreement = float("nan")
    if declared in votes and outcome_columns:
        declared_vote = votes[declared].to_numpy()
        outcome_positive = (votes[outcome_columns].to_numpy() == POSITIVE).any(axis=1)
        comparable = declared_vote != ABSTAIN
        if comparable.any():
            agreement = float(
                (outcome_positive[comparable] == (declared_vote[comparable] == 1)).mean()
            )
    return LabelModelResult(
        probabilities=pd.Series(probabilities, index=votes.index, name="p_urgent"),
        votes=votes.copy(),
        diagnostics=pd.DataFrame(diagnostics).set_index("function"),
        declared_outcome_agreement=agreement,
    )


def build_labels(
    events: Sequence[EvalEvent],
    functions: Sequence[LabelingFunction] = DEFAULT_LABELING_FUNCTIONS,
    *,
    prior: float | None = None,
) -> LabelModelResult:
    """Apply strict post-t labelling functions and fit the label model."""

    return fit_label_model(apply_labeling_functions(events, functions), prior=prior)
