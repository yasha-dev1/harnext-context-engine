"""Post-outcome urgency labels for docs/evaluation-spec.md §4.1 and §7 E1.

Candidates are evaluated only against the strict event-time suffix. A finite
labelling horizon crossing ``observation_end`` is censored (ABSTAIN), while an
applicable, fully observed horizon with no outcome casts a NEGATIVE vote.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from harnext_eval.types import EvalEvent

ABSTAIN = -1
NEGATIVE = 0
POSITIVE = 1
Vote = int
OutcomePredicate = Callable[[EvalEvent, Sequence[EvalEvent]], bool]
Applicability = Callable[[EvalEvent], bool]


@dataclass(frozen=True)
class LabelingFunction:
    """One source-specific weak-supervision function.

    ``horizon=None`` means later through the preregistered observation end; it
    never introduces an arbitrary elapsed-time cutoff.
    """

    name: str
    horizon: timedelta | None
    source: str
    applies_to: Applicability
    outcome: OutcomePredicate
    declared: bool = False


@dataclass(frozen=True)
class LabelModelResult:
    """Probabilistic labels plus per-function and agreement diagnostics."""

    probabilities: pd.Series
    votes: pd.DataFrame
    observability: pd.DataFrame
    diagnostics: pd.DataFrame
    declared_outcome_agreement: float
    declared_outcome_comparable: int


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
    wanted = {name.casefold() for name in names}
    stack: list[Any] = [event.data or {}]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).casefold() in wanted and not isinstance(item, (dict, list)):
                    return item
                stack.append(item)
        elif isinstance(value, list):
            stack.extend(value)
    return None


def _is_jira(event: EvalEvent) -> bool:
    return "jira" in event.source.casefold() or ".jira." in event.type.casefold()


def _is_dev_mail(event: EvalEvent) -> bool:
    source = event.source.casefold()
    return "dev@" in source or "pony" in source or ".mail." in event.type.casefold()


def _is_github(event: EvalEvent) -> bool:
    return "github" in event.source.casefold() or ".github." in event.type.casefold()


def _is_committer(event: EvalEvent) -> bool:
    role = str(_field(event, "author_association", "role") or "").casefold()
    actor = str(_field(event, "author", "from", "actor") or "").casefold()
    return bool(_field(event, "is_committer")) or role in {
        "member",
        "owner",
        "committer",
    } or "committer" in actor


def _identity_tokens(candidate: EvalEvent) -> set[str]:
    tokens = {candidate.id.casefold(), candidate.subject.casefold()}
    tokens.update(re.findall(r"(?:kafka|flink|kip)-\d+|#\d+|pr:\d+", candidate.subject.casefold()))
    for name in ("issue_key", "key", "number", "pr_number", "pull_request"):
        value = _field(candidate, name)
        if value is not None:
            tokens.add(str(value).casefold())
            if str(value).isdigit():
                tokens.add(f"#{value}")
    return {token for token in tokens if token}


def _same_thread(candidate: EvalEvent, later: EvalEvent) -> bool:
    if candidate.subject == later.subject:
        return True
    candidate_roots = {
        str(_field(candidate, "thread_root", "message_id") or ""),
        candidate.subject.removeprefix("thread:"),
    }
    later_roots = {
        str(_field(later, "thread_root", "in_reply_to", "message_id") or ""),
        later.subject.removeprefix("thread:"),
    }
    return bool((candidate_roots - {""}) & (later_roots - {""}))


def _references(candidate: EvalEvent, later: EvalEvent) -> bool:
    if candidate.subject == later.subject:
        return True
    text = f" {later.subject.casefold()} {_text(later)} "
    return any(
        re.search(rf"(?<![\w-]){re.escape(token)}(?![\w-])", text)
        for token in _identity_tokens(candidate)
    )


def _related(candidate: EvalEvent, events: Sequence[EvalEvent]) -> list[EvalEvent]:
    if _is_dev_mail(candidate):
        return [event for event in events if _is_dev_mail(event) and _same_thread(candidate, event)]
    return [event for event in events if _references(candidate, event)]


def _committer_comment(candidate: EvalEvent, events: Sequence[EvalEvent]) -> bool:
    for event in _related(candidate, events):
        kind = event.type.casefold()
        if ("comment" in kind or "message" in kind or "mail" in kind) and _is_committer(event):
            return True
    return False


def _resolved(candidate: EvalEvent, events: Sequence[EvalEvent]) -> bool:
    for event in _related(candidate, events):
        field = str(_field(event, "field") or "").casefold()
        target = str(_field(event, "to", "status", "state") or "").casefold()
        if target in {"resolved", "closed", "done", "fixed"} and field in {
            "",
            "status",
            "state",
        }:
            return True
    return False


_PRIORITY = {"trivial": 0, "minor": 1, "normal": 2, "major": 3, "critical": 4, "blocker": 5}


def _priority_raised(candidate: EvalEvent, events: Sequence[EvalEvent]) -> bool:
    for event in _related(candidate, events):
        if str(_field(event, "field") or "").casefold() != "priority":
            continue
        old = str(_field(event, "from", "old", "previous") or "").casefold()
        new = str(_field(event, "to", "new", "priority") or "").casefold()
        if _PRIORITY.get(new, -1) > _PRIORITY.get(old, -1):
            return True
    return False


def _fix_version_in_flight(candidate: EvalEvent, events: Sequence[EvalEvent]) -> bool:
    for event in _related(candidate, events):
        field = str(_field(event, "field") or "").casefold().replace("_", "")
        if field not in {"fixversion", "fixversions"}:
            continue
        target = str(_field(event, "to", "new", "fix_version", "fixVersion") or "")
        current = str(_field(event, "in_flight_release", "current_release") or "")
        marked = bool(_field(event, "in_flight", "is_in_flight_release"))
        if target and (marked or (current and target == current)):
            return True
    return False


def _linked_pr(candidate: EvalEvent, events: Sequence[EvalEvent]) -> bool:
    for event in events:
        kind = event.type.casefold()
        action = str(_field(event, "action", "state") or "").casefold()
        opened = "pull_request.opened" in kind or (
            "pull_request" in kind and action in {"open", "opened"}
        )
        if _is_github(event) and opened and _references(candidate, event):
            return True
    return False


def _declared_priority(candidate: EvalEvent, events: Sequence[EvalEvent]) -> bool:
    for event in _related(candidate, events):
        field = str(_field(event, "field") or "").casefold()
        value = str(_field(event, "priority", "to", "severity") or "").casefold()
        if value in {"blocker", "critical"} and field in {"", "priority", "severity"}:
            return True
    return False


def _three_responders(candidate: EvalEvent, events: Sequence[EvalEvent]) -> bool:
    responders = {
        str(_field(event, "author", "from", "actor", "sender") or "").casefold()
        for event in _related(candidate, events)
    }
    responders.discard("")
    return len(responders) >= 3


def _vote_cancelled_recast(candidate: EvalEvent, events: Sequence[EvalEvent]) -> bool:
    related = _related(candidate, events)
    cancelled = [event for event in related if re.search(r"\b(cancelled|canceled)\b", _text(event))]
    if not cancelled:
        return False
    return any(
        event.time > cancelled[0].time
        and ("[vote]" in _text(event) or re.search(r"\bre[- ]?cast\b", _text(event)))
        for event in related
    )


def _thread_cve_blocker(candidate: EvalEvent, events: Sequence[EvalEvent]) -> bool:
    return any(
        re.search(r"(?<![\w-])(?:cve(?:-\d{4}-\d+)?|blocker)(?![\w-])", _text(event))
        is not None
        for event in _related(candidate, events)
    )


def _reverted(candidate: EvalEvent, events: Sequence[EvalEvent]) -> bool:
    return any(
        _is_github(event)
        and _references(candidate, event)
        and re.search(r"(?<!\w)revert(?:ed|ing)?(?!\w)", f"{event.type} {_text(event)}")
        for event in events
    )


def _hotfix_reference(candidate: EvalEvent, events: Sequence[EvalEvent]) -> bool:
    return any(
        _is_github(event)
        and "pull_request" in event.type.casefold()
        and "hotfix" in _text(event)
        and _references(candidate, event)
        for event in events
    )


def _is_trunk_failure(event: EvalEvent) -> bool:
    text = f"{event.type} {_text(event)}"
    branch = str(_field(event, "branch", "ref") or "").casefold()
    return ("ci" in text or "check" in text) and any(
        token in text for token in ("fail", "failure", "failed")
    ) and branch in {"", "main", "master", "trunk", "refs/heads/main", "refs/heads/trunk"}


def _is_fix_commit(event: EvalEvent) -> bool:
    text = f"{event.type} {_text(event)}"
    return ("push" in event.type.casefold() or "commit" in event.type.casefold()) and re.search(
        r"(?<!\w)(fix|fixed|repair)(?!\w)", text
    ) is not None


def _trunk_failure_fixed(candidate: EvalEvent, events: Sequence[EvalEvent]) -> bool:
    related = _related(candidate, events)
    failures = [candidate] if _is_trunk_failure(candidate) else []
    failures.extend(event for event in related if _is_trunk_failure(event))
    return any(
        _is_fix_commit(event)
        and _references(candidate, event)
        and any(failure.time < event.time <= failure.time + timedelta(hours=6) for failure in failures)
        for event in events
    )


# Compatibility helpers for focused predicate tests. Source and censoring
# semantics are applied by ``apply_labeling_functions``.
def committer_reply(events: Sequence[EvalEvent]) -> Vote:
    return POSITIVE if any(_is_committer(event) for event in events) else ABSTAIN


def resolved(events: Sequence[EvalEvent]) -> Vote:
    return POSITIVE if events and _resolved(events[0], events) else ABSTAIN


def priority_raised(events: Sequence[EvalEvent]) -> Vote:
    return POSITIVE if events and _priority_raised(events[0], events) else ABSTAIN


def linked_pr(events: Sequence[EvalEvent]) -> Vote:
    return POSITIVE if events and any("pull_request" in event.type for event in events) else ABSTAIN


def reverted(events: Sequence[EvalEvent]) -> Vote:
    return POSITIVE if any("revert" in f"{event.type} {_text(event)}" for event in events) else ABSTAIN


def declared_blocker_critical(events: Sequence[EvalEvent]) -> Vote:
    return POSITIVE if any(str(_field(event, "priority", "to") or "").casefold() in {"blocker", "critical"} for event in events) else ABSTAIN


DEFAULT_LABELING_FUNCTIONS: tuple[LabelingFunction, ...] = (
    LabelingFunction("jira_committer_comment_1h", timedelta(hours=1), "jira", _is_jira, _committer_comment),
    LabelingFunction("jira_priority_raised_later", None, "jira", _is_jira, _priority_raised),
    LabelingFunction("jira_fix_version_in_flight_later", None, "jira", _is_jira, _fix_version_in_flight),
    LabelingFunction("jira_resolved_24h", timedelta(hours=24), "jira", _is_jira, _resolved),
    LabelingFunction("jira_linked_pr_24h", timedelta(hours=24), "jira", _is_jira, _linked_pr),
    LabelingFunction("declared_blocker_critical", None, "jira", _is_jira, _declared_priority, declared=True),
    LabelingFunction("dev_committer_reply_1h", timedelta(hours=1), "dev", _is_dev_mail, _committer_comment),
    LabelingFunction("dev_three_responders_2h", timedelta(hours=2), "dev", _is_dev_mail, _three_responders),
    LabelingFunction("dev_vote_cancelled_recast_later", None, "dev", _is_dev_mail, _vote_cancelled_recast),
    LabelingFunction("dev_cve_blocker_later", None, "dev", _is_dev_mail, _thread_cve_blocker),
    LabelingFunction("github_reverted_48h", timedelta(hours=48), "github", _is_github, _reverted),
    LabelingFunction("github_hotfix_reference_24h", timedelta(hours=24), "github", _is_github, _hotfix_reference),
    LabelingFunction("github_trunk_ci_failure_fix_6h", timedelta(hours=6), "github", _is_github, _trunk_failure_fixed),
)


_REFERENCE_TOKEN_RE = re.compile(r"[A-Za-z0-9_:#./-]+")


class _OutcomeIndex:
    """One-pass relationship index for the 350k-event Corpus-R profile."""

    def __init__(self, events: Sequence[EvalEvent]) -> None:
        self.events = events
        self.times = [event.time for event in events]
        self.by_subject: defaultdict[str, set[int]] = defaultdict(set)
        self.by_thread: defaultdict[str, set[int]] = defaultdict(set)
        self.by_token: defaultdict[str, set[int]] = defaultdict(set)
        self._cache: dict[int, tuple[int, ...]] = {}
        for index, event in enumerate(events):
            self.by_subject[event.subject].add(index)
            roots = {
                event.subject.removeprefix("thread:"),
                str(_field(event, "thread_root", "in_reply_to", "message_id") or ""),
            }
            for root in roots - {""}:
                self.by_thread[root].add(index)
            surface = f"{event.subject} {_text(event)}"
            for token in _REFERENCE_TOKEN_RE.findall(surface.casefold()):
                self.by_token[token].add(index)

    def related_indices(self, candidate_index: int) -> tuple[int, ...]:
        cached = self._cache.get(candidate_index)
        if cached is not None:
            return cached
        candidate = self.events[candidate_index]
        related = set(self.by_subject.get(candidate.subject, set()))
        if _is_dev_mail(candidate):
            roots = {
                candidate.subject.removeprefix("thread:"),
                str(_field(candidate, "thread_root", "message_id") or ""),
            }
            for root in roots - {""}:
                related.update(self.by_thread.get(root, set()))
        else:
            for token in _identity_tokens(candidate):
                related.update(self.by_token.get(token.casefold(), set()))
        candidate_time = candidate.time
        result = tuple(
            sorted(
                index
                for index in related
                if index > candidate_index and self.events[index].time > candidate_time
            )
        )
        self._cache[candidate_index] = result
        return result

    def suffix(
        self,
        candidate_index: int,
        cutoff: datetime,
        *,
        relationships_only: bool,
    ) -> tuple[EvalEvent, ...]:
        stop = bisect_right(self.times, cutoff)
        if relationships_only:
            indices = self.related_indices(candidate_index)
            return tuple(self.events[index] for index in indices if index < stop)
        start = bisect_right(self.times, self.events[candidate_index].time)
        return tuple(self.events[start:stop])


def _apply_with_observability(
    events: Sequence[EvalEvent],
    functions: Sequence[LabelingFunction],
    *,
    observation_end: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = sorted(events, key=lambda event: (event.time, event.id))
    index = _OutcomeIndex(ordered)
    default_outcomes = {function.outcome for function in DEFAULT_LABELING_FUNCTIONS}
    vote_rows: list[dict[str, int | str]] = []
    observed_rows: list[dict[str, bool | str]] = []
    for candidate_index, candidate in enumerate(ordered):
        vote_row: dict[str, int | str] = {"event_id": candidate.id}
        observed_row: dict[str, bool | str] = {"event_id": candidate.id}
        for function in functions:
            applicable = function.applies_to(candidate)
            complete = function.horizon is None or candidate.time + function.horizon <= observation_end
            observable = applicable and complete
            observed_row[function.name] = observable
            if not observable:
                vote_row[function.name] = ABSTAIN
                continue
            cutoff = observation_end if function.horizon is None else candidate.time + function.horizon
            suffix = index.suffix(
                candidate_index,
                cutoff,
                relationships_only=function.outcome in default_outcomes,
            )
            vote_row[function.name] = POSITIVE if function.outcome(candidate, suffix) else NEGATIVE
        vote_rows.append(vote_row)
        observed_rows.append(observed_row)
    votes = pd.DataFrame(vote_rows).set_index("event_id")
    observability = pd.DataFrame(observed_rows).set_index("event_id")
    return votes, observability


def apply_labeling_functions(
    events: Sequence[EvalEvent],
    functions: Sequence[LabelingFunction] = DEFAULT_LABELING_FUNCTIONS,
    *,
    observation_end: datetime | None = None,
) -> pd.DataFrame:
    """Return source-specific votes; censored or wrong-source rows abstain."""

    if not events:
        return pd.DataFrame(columns=[function.name for function in functions], dtype=int)
    end = observation_end or max(event.time for event in events)
    votes, observability = _apply_with_observability(events, functions, observation_end=end)
    votes.attrs["observability"] = observability
    return votes


class WeightedLabelModel:
    """Deterministic Snorkel-style independent-LF generative model."""

    def __init__(self, *, prior: float | None = None, max_iter: int = 100) -> None:
        self.prior = prior
        self.max_iter = max_iter
        self.accuracies_: np.ndarray | None = None

    def fit_predict(self, votes: pd.DataFrame) -> np.ndarray:
        matrix = votes.to_numpy(dtype=int)
        if not len(matrix):
            self.accuracies_ = np.full(matrix.shape[1], np.nan)
            return np.asarray([], dtype=float)
        fired = matrix != ABSTAIN
        positive = matrix == POSITIVE
        known_rows = fired.any(axis=1)
        majority = np.divide(
            positive.sum(axis=1),
            fired.sum(axis=1),
            out=np.zeros(len(matrix), dtype=float),
            where=fired.sum(axis=1) > 0,
        )
        observed = float(np.mean(majority[known_rows] >= 0.5)) if known_rows.any() else 0.01
        prevalence = float(np.clip(self.prior if self.prior is not None else observed, 0.01, 0.99))
        accuracy = np.full(matrix.shape[1], 0.7, dtype=float)
        probabilities = np.full(len(matrix), prevalence, dtype=float)
        for _ in range(self.max_iter):
            logit = np.full(len(matrix), np.log(prevalence / (1.0 - prevalence)))
            for column in range(matrix.shape[1]):
                weight = np.log(accuracy[column] / (1.0 - accuracy[column]))
                logit += np.where(matrix[:, column] == POSITIVE, weight, 0.0)
                logit -= np.where(matrix[:, column] == NEGATIVE, weight, 0.0)
            updated = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
            next_accuracy = accuracy.copy()
            for column in range(matrix.shape[1]):
                covered = fired[:, column]
                if covered.any():
                    expected = np.where(positive[covered, column], updated[covered], 1.0 - updated[covered])
                    next_accuracy[column] = (expected.sum() + 2.0) / (covered.sum() + 4.0)
            next_accuracy = np.clip(next_accuracy, 0.05, 0.95)
            next_prevalence = float(np.clip(updated[known_rows].mean(), 0.01, 0.99)) if known_rows.any() else prevalence
            if max(np.max(np.abs(next_accuracy - accuracy)), abs(next_prevalence - prevalence)) < 1e-7:
                probabilities, accuracy, prevalence = updated, next_accuracy, next_prevalence
                break
            probabilities, accuracy, prevalence = updated, next_accuracy, next_prevalence
        probabilities[~known_rows] = np.nan
        self.accuracies_ = accuracy
        return probabilities


def fit_label_model(
    votes: pd.DataFrame,
    *,
    prior: float | None = None,
    observability: pd.DataFrame | None = None,
) -> LabelModelResult:
    """Fuse votes and report per-function accuracy/coverage/conflicts."""

    if observability is None:
        stored = votes.attrs.get("observability")
        observability = stored if isinstance(stored, pd.DataFrame) else votes != ABSTAIN
    model = WeightedLabelModel(prior=prior)
    probabilities = model.fit_predict(votes)
    assert model.accuracies_ is not None
    matrix = votes.to_numpy(dtype=int)
    diagnostics: list[dict[str, float | str | int]] = []
    for column, name in enumerate(votes.columns):
        covered = matrix[:, column] != ABSTAIN
        other = np.delete(matrix, column, axis=1)
        overlap = covered & np.any(other != ABSTAIN, axis=1)
        conflict = covered & np.any((other != ABSTAIN) & (other != matrix[:, column, None]), axis=1)
        observed = observability[name].to_numpy(dtype=bool)
        diagnostics.append(
            {
                "function": name,
                "accuracy": float(model.accuracies_[column]),
                "coverage": float(covered.mean()) if len(matrix) else 0.0,
                "overlap": float(overlap.mean()) if len(matrix) else 0.0,
                "conflict": float(conflict.sum() / max(overlap.sum(), 1)),
                "positive_votes": int(np.sum(matrix[:, column] == POSITIVE)),
                "negative_votes": int(np.sum(matrix[:, column] == NEGATIVE)),
                "unknown": int(np.sum(~observed)),
            }
        )
    declared_columns = [function.name for function in DEFAULT_LABELING_FUNCTIONS if function.declared and function.name in votes]
    outcome_columns = [name for name in votes.columns if name not in declared_columns]
    comparable = np.zeros(len(votes), dtype=bool)
    agreement = float("nan")
    if declared_columns and outcome_columns:
        declared_matrix = votes[declared_columns].to_numpy(dtype=int)
        outcome_matrix = votes[outcome_columns].to_numpy(dtype=int)
        comparable = np.any(declared_matrix != ABSTAIN, axis=1) & np.any(outcome_matrix != ABSTAIN, axis=1)
        if comparable.any():
            declared_label = np.any(declared_matrix == POSITIVE, axis=1)
            outcome_label = np.any(outcome_matrix == POSITIVE, axis=1)
            agreement = float(np.mean(declared_label[comparable] == outcome_label[comparable]))
    return LabelModelResult(
        probabilities=pd.Series(probabilities, index=votes.index, name="p_urgent"),
        votes=votes.copy(),
        observability=observability.copy(),
        diagnostics=pd.DataFrame(diagnostics).set_index("function"),
        declared_outcome_agreement=agreement,
        declared_outcome_comparable=int(comparable.sum()),
    )


def build_labels(
    events: Sequence[EvalEvent],
    functions: Sequence[LabelingFunction] = DEFAULT_LABELING_FUNCTIONS,
    *,
    prior: float | None = None,
    observation_end: datetime | None = None,
) -> LabelModelResult:
    """Apply strict post-t functions and fit the label model."""

    if not events:
        votes = apply_labeling_functions(events, functions, observation_end=observation_end)
        return fit_label_model(votes, prior=prior)
    end = observation_end or max(event.time for event in events)
    votes, observability = _apply_with_observability(events, functions, observation_end=end)
    return fit_label_model(votes, prior=prior, observability=observability)
