"""Budgeted router policies R0--R7 from docs/evaluation-spec.md §7 E1."""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from sklearn.preprocessing import StandardScaler

from harnext_eval.config import RouterConfig
from harnext_eval.e1.features import FEATURE_NAMES, CausalFeatureExtractor, FeatureVector
from harnext_eval.types import EvalEvent


@runtime_checkable
class RouterPolicy(Protocol):
    """The replay-driver seam described in PLAN.md §6."""

    name: str

    def rules(self, event: EvalEvent) -> str | None: ...

    def score(self, event: EvalEvent) -> float: ...


@dataclass(frozen=True)
class RuleSettings:
    enabled: bool = True
    dispute_amount: float = 1_000.0
    vote_thread_start_only: bool = False
    dedup_per_subject: bool = False
    """Amendment 2026-09-06 (review finding 4): a rule fires once per (subject, rule)
    within a policy instance's stream, so repeated notifications of one incident do
    not re-consume the fast lane."""


_DEFAULT_RULE_SETTINGS = RuleSettings()


_HISTORICAL_RULE_FIELDS = {
    "changelog",
    "from",
    "fromstring",
    "old",
    "old_value",
    "previous",
}


def _walk(value: Any) -> list[str]:
    if isinstance(value, dict):
        result: list[str] = []
        for key, item in value.items():
            if str(key).casefold().replace("_", "") in {
                name.replace("_", "") for name in _HISTORICAL_RULE_FIELDS
            }:
                continue
            result.append(str(key))
            result.extend(_walk(item))
        return result
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(_walk(item))
        return result
    return [] if value is None else [str(value)]


def _field_values(value: Any, names: set[str]) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).casefold() in names and not isinstance(item, (dict, list)):
                found.append(item)
            found.extend(_field_values(item, names))
    elif isinstance(value, list):
        for item in value:
            found.extend(_field_values(item, names))
    return found


def _is_reply(data: dict[str, Any]) -> bool:
    """A mail message that answers an earlier one: has ``in_reply_to``/``references`` or a ``Re:`` subject."""

    if data.get("in_reply_to"):
        return True
    refs = data.get("references")
    if isinstance(refs, list) and refs:
        return True
    if isinstance(refs, str) and refs.strip() not in {"", "[]"}:
        return True
    subject = str(data.get("subject") or "")
    return bool(re.match(r"\s*(re|aw|fwd?)\s*:", subject, re.IGNORECASE))


def match_rule(event: EvalEvent, settings: RuleSettings = _DEFAULT_RULE_SETTINGS) -> str | None:
    """Rules floor configured by enablement and the dispute amount threshold."""

    if not settings.enabled:
        return None
    data = event.data or {}
    text = " ".join(_walk(data)).casefold()
    declared = _field_values(data, {"priority", "severity"})
    # Amendment 2026-09-06: changelog transitions carry the value under ``to`` with ``field=priority``.
    if str(data.get("field") or "").casefold() in {"priority", "severity"}:
        declared = [*declared, data.get("to"), data.get("new")]
    if any(str(value).strip().casefold() in {"critical", "blocker"} for value in declared if value is not None):
        return "declared_priority"
    if "[vote]" in text and not (settings.vote_thread_start_only and _is_reply(data)):
        return "vote"
    if re.search(r"(?<![\w-])cve(?:-\d{4}-\d+)?(?![\w-])", text):
        return "cve"
    if re.search(r"(?<![\w-])blocker(?![\w-])", text):
        return "blocker_word"
    if "on-call" in text or "oncall" in text or "pagerduty" in text:
        return "on_call_page"
    if "dispute" in text:
        for amount in _field_values(data, {"amount", "dispute_amount", "value", "total"}):
            try:
                if float(amount) >= settings.dispute_amount:
                    return "large_dispute"
            except (TypeError, ValueError):
                continue
    return None


class _PolicyBase:
    name = "base"

    def __init__(self, *, rules: RuleSettings = _DEFAULT_RULE_SETTINGS) -> None:
        self.rule_settings = rules
        self.extractor = CausalFeatureExtractor()
        self.feature_cache: dict[str, list[FeatureVector]] | None = None
        self.baseline_key_used: str | None = None
        self.features_fired: dict[str, Any] = {}
        self._rule_seen: set[tuple[str, str]] = set()
        self._rule_cache: dict[str, str | None] = {}

    def fit(self, events: Sequence[EvalEvent]) -> _PolicyBase:
        del events
        return self

    def rules(self, event: EvalEvent) -> str | None:
        """Rule verdict for one event; idempotent per event id so dedup state is stable."""

        cached = self._rule_cache.get(event.id, "__unset__")
        if cached != "__unset__":
            return cached  # type: ignore[return-value]
        rule = match_rule(event, self.rule_settings)
        if rule and self.rule_settings.dedup_per_subject:
            key = (event.subject, rule)
            if key in self._rule_seen:
                rule = None
            else:
                self._rule_seen.add(key)
        self._rule_cache[event.id] = rule
        return rule

    def score(self, event: EvalEvent) -> float:
        raise NotImplementedError

    def _vectors(self, event: EvalEvent) -> list[FeatureVector]:
        if self.feature_cache is not None:
            return self.feature_cache[event.id]
        return self.extractor.update(event)

    def _record(self, vector: FeatureVector, score: float, **extra: Any) -> float:
        self.baseline_key_used = vector.baseline_key
        self.features_fired = {**vector.values, **vector.context, **extra}
        return float(score)


class RandomPolicy(_PolicyBase):
    """R0: deterministic pseudo-random score, independent of replay order."""

    name = "R0"

    def __init__(self, *, seed: int = 0, rules: RuleSettings = _DEFAULT_RULE_SETTINGS) -> None:
        super().__init__(rules=rules)
        self.seed = seed

    def score(self, event: EvalEvent) -> float:
        digest = hashlib.sha256(f"{self.seed}:{event.id}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / float(2**64)


class RulesOnlyPolicy(_PolicyBase):
    """R1: rules floor only."""

    name = "R1"

    def score(self, event: EvalEvent) -> float:
        rule = self.rules(event)
        self.features_fired = {"rule": rule} if rule else {}
        return float(rule is not None)


def _matrix(vectors: Sequence[FeatureVector]) -> np.ndarray:
    if not vectors:
        return np.empty((0, len(FEATURE_NAMES)), dtype=float)
    return np.vstack([vector.as_array() for vector in vectors])


_GLOBAL_METHODS = ("z", "hbos", "iforest", "ecod", "lof")


class _ECOD:
    """Empirical-CDF outlier score (Li et al., ECOD) with O(log n) per-event scoring.

    pyod's ECOD re-concatenates the training matrix on every ``decision_function``
    call, which is prohibitive for 387k single-event calls; this keeps the sorted
    training columns and their skewness and applies the same scoring rule:
    the maximum of the left-tail, right-tail and skewness-selected tail sums of
    -log(ECDF).
    """

    def __init__(self) -> None:
        self.columns: np.ndarray | None = None
        self.skew_negative: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> _ECOD:
        self.columns = np.sort(x, axis=0)
        centred = x - x.mean(axis=0)
        std = np.maximum(centred.std(axis=0), 1e-12)
        self.skew_negative = (np.mean(centred**3, axis=0) / std**3) < 0
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        assert self.columns is not None and self.skew_negative is not None
        n = len(self.columns)
        out = np.empty(len(x))
        width = self.columns.shape[1]
        for row_index, row in enumerate(x):
            below = np.array([np.searchsorted(self.columns[:, j], row[j], side="right") for j in range(width)], dtype=float)
            above = n - np.array([np.searchsorted(self.columns[:, j], row[j], side="left") for j in range(width)], dtype=float)
            left_tail = -np.log(np.clip(below / n, 1.0 / n, 1.0))
            right_tail = -np.log(np.clip(above / n, 1.0 / n, 1.0))
            auto_tail = np.where(self.skew_negative, left_tail, right_tail)
            out[row_index] = max(float(left_tail.sum()), float(right_tail.sum()), float(auto_tail.sum()))
        return out


def _global_model(method: str, count: int, seed: int) -> Any:
    if method == "hbos":
        return HBOS(n_bins=min(10, max(3, int(math.sqrt(count)))), contamination=0.1)
    if method == "iforest":
        return IForest(n_estimators=100, contamination=0.1, random_state=seed, n_jobs=1)
    if method == "ecod":
        return _ECOD()
    if method == "lof":
        return LOF(n_neighbors=min(20, max(2, count - 1)), contamination=0.1, novelty=True)
    raise ValueError(f"unknown global method {method!r}")


class GlobalPolicy(_PolicyBase):
    """R2 and its registered variants: one global detector over the feature vector, no entity keying."""

    name = "R2"
    global_features = True

    def __init__(
        self,
        *,
        method: str = "hbos",
        rules: RuleSettings = _DEFAULT_RULE_SETTINGS,
        seed: int = 0,
        name: str | None = None,
    ) -> None:
        super().__init__(rules=rules)
        self.extractor = CausalFeatureExtractor(global_only=True)
        if method not in _GLOBAL_METHODS:
            raise ValueError(f"global method must be one of {_GLOBAL_METHODS}")
        if name is not None:
            self.name = name
        self.method = method
        self.seed = seed
        self.model: Any = None
        self.scaler: StandardScaler | None = None
        self.median = np.zeros(len(FEATURE_NAMES))
        self.scale = np.ones(len(FEATURE_NAMES))

    def _fit_matrix(self, x: np.ndarray) -> Any:
        if self.method == "z" or len(x) < 2:
            return None
        if self.method == "lof":
            self.scaler = StandardScaler().fit(x)
            x = np.asarray(self.scaler.transform(x), dtype=float)
        model = _global_model(self.method, len(x), self.seed)
        model.fit(x)
        return model

    def _score_matrix(self, model: Any, x: np.ndarray) -> float:
        if self.method == "lof" and self.scaler is not None:
            x = np.asarray(self.scaler.transform(x), dtype=float)
        result = model.decision_function(x)
        assert result is not None
        return float(result[0])

    def fit(self, events: Sequence[EvalEvent]) -> GlobalPolicy:
        vectors = [self._vectors(event)[0] for event in events]
        x = _matrix(vectors)
        if len(x):
            self.median = np.median(x, axis=0)
            mad = np.median(np.abs(x - self.median), axis=0)
            self.scale = np.maximum(1.4826 * mad, 1e-6)
        self.model = self._fit_matrix(x)
        return self

    def score(self, event: EvalEvent) -> float:
        vector = self._vectors(event)[0]
        x = vector.as_array()[None, :]
        if self.model is not None:
            value = self._score_matrix(self.model, x)
        else:
            value = float(np.max(np.abs((x[0] - self.median) / self.scale)))
        return self._record(vector, value, scorer=f"global_{self.method}")


def _event_source(event: EvalEvent) -> str:
    return event.source.split(":", 1)[0]


class PerSourceGlobalPolicy(GlobalPolicy):
    """R10: one global HBOS per source, scores calibrated to within-source percentiles.

    Run 3 showed the urgency signal is source-specific (global HBOS on JIRA, the
    gap on dev@, per-entity HBOS on GitHub). Fitting one detector per source and
    ranking by the percentile of the raw score within that source's training
    distribution makes the monthly top-b% cut allocate the budget across sources
    in proportion to how anomalous each source's events are, not to the scale of
    one detector's scores.
    """

    name = "R10"

    def __init__(self, *, rules: RuleSettings = _DEFAULT_RULE_SETTINGS, seed: int = 0) -> None:
        super().__init__(method="hbos", rules=rules, seed=seed, name="R10")
        self.source_models: dict[str, Any] = {}
        self.source_scores: dict[str, np.ndarray] = {}

    def fit(self, events: Sequence[EvalEvent]) -> PerSourceGlobalPolicy:
        # One causal pass over the events: the extractor is stateful and ordered.
        vectors = [self._vectors(event)[0] for event in events]
        x = _matrix(vectors)
        if len(x):
            self.median = np.median(x, axis=0)
            mad = np.median(np.abs(x - self.median), axis=0)
            self.scale = np.maximum(1.4826 * mad, 1e-6)
        self.model = self._fit_matrix(x)
        groups: defaultdict[str, list[np.ndarray]] = defaultdict(list)
        for event, vector in zip(events, vectors, strict=True):
            groups[_event_source(event)].append(vector.as_array())
        for source, rows in groups.items():
            x = np.vstack(rows)
            model = self._fit_matrix(x)
            if model is None:
                continue
            self.source_models[source] = model
            scores = model.decision_function(x)
            assert scores is not None
            self.source_scores[source] = np.sort(np.asarray(scores, dtype=float))
        return self

    def score(self, event: EvalEvent) -> float:
        vector = self._vectors(event)[0]
        x = vector.as_array()[None, :]
        source = _event_source(event)
        model = self.source_models.get(source, self.model)
        reference = self.source_scores.get(source)
        if model is None:
            value = float(np.max(np.abs((x[0] - self.median) / self.scale)))
            return self._record(vector, value, scorer="per_source_hbos", source_model=False)
        raw = self._score_matrix(model, x)
        if reference is None or not len(reference):
            value = raw
        else:
            value = float(np.searchsorted(reference, raw, side="right") / len(reference))
        return self._record(vector, value, scorer="per_source_hbos", raw_score=raw, source_model=source in self.source_models)


class RobustGapPolicy(_PolicyBase):
    """R3: per-baseline-key robust z-score on log inter-arrival gap only."""

    name = "R3"

    def __init__(self, *, rules: RuleSettings = _DEFAULT_RULE_SETTINGS) -> None:
        super().__init__(rules=rules)
    def fit(self, events: Sequence[EvalEvent]) -> RobustGapPolicy:
        for event in events:
            self._vectors(event)
        return self

    def score(self, event: EvalEvent) -> float:
        scored = []
        for vector in self._vectors(event):
            median = vector.context["baseline_median_log_gap"]
            scale = max(1.4826 * vector.context["baseline_mad_log_gap"], 0.1)
            # Bursts have unexpectedly short gaps, hence the one-sided sign.
            value = max(0.0, (median - vector.values["log_gap_s"]) / scale)
            scored.append((value, vector))
        value, vector = max(scored, key=lambda item: item[0])
        return self._record(vector, value, scorer="per_entity_gap_robust_z")


class EntityHBOSPolicy(_PolicyBase):
    """R4: per-key HBOS over the complete E1 feature vector, unguarded."""

    name = "R4"

    def __init__(self, *, rules: RuleSettings = _DEFAULT_RULE_SETTINGS) -> None:
        super().__init__(rules=rules)
        self.models: dict[str, HBOS] = {}
        self.fallback: HBOS | None = None

    def fit(self, events: Sequence[EvalEvent]) -> EntityHBOSPolicy:
        groups: defaultdict[str, list[FeatureVector]] = defaultdict(list)
        all_vectors: list[FeatureVector] = []
        for event in events:
            for vector in self._vectors(event):
                groups[vector.baseline_key].append(vector)
                all_vectors.append(vector)
        for key, vectors in groups.items():
            if len(vectors) >= 5:
                model = HBOS(
                    n_bins=min(10, max(3, int(math.sqrt(len(vectors))))), contamination=0.1
                )
                model.fit(_matrix(vectors))
                self.models[key] = model
        if len(all_vectors) >= 5:
            self.fallback = HBOS(n_bins=10, contamination=0.1).fit(_matrix(all_vectors))
        return self

    def _score_vector(self, vector: FeatureVector) -> float:
        model = self.models.get(vector.baseline_key, self.fallback)
        if model is None:
            return 0.0
        result = model.decision_function(vector.as_array()[None, :])
        assert result is not None
        return float(result[0])

    def _hbos_terms(self, vector: FeatureVector) -> dict[str, float]:
        model = self.models.get(vector.baseline_key, self.fallback)
        if model is None or model.hist_ is None or model.bin_edges_ is None:
            return {}
        terms: dict[str, float] = {}
        all_edges = np.asarray(model.bin_edges_, dtype=float)
        all_histograms = np.asarray(model.hist_, dtype=float)
        for index, name in enumerate(FEATURE_NAMES):
            edges = all_edges[:, index]
            hist = all_histograms[:, index]
            bin_index = int(np.clip(np.searchsorted(edges, vector.as_array()[index], side="right") - 1, 0, len(hist) - 1))
            terms[name] = float(-np.log(max(hist[bin_index], 1e-12)))
        return terms

    def score(self, event: EvalEvent) -> float:
        scored = [(self._score_vector(vector), vector) for vector in self._vectors(event)]
        value, vector = max(scored, key=lambda item: item[0])
        return self._record(
            vector, value, scorer="per_entity_hbos", hbos_terms=self._hbos_terms(vector)
        )


class GuardedHBOSPolicy(EntityHBOSPolicy):
    """R5: rules + per-key HBOS + absolute volume and confirmation guards."""

    name = "R5"

    def __init__(
        self,
        *,
        absolute_floor: float = 3.0,
        multi_window: bool = True,
        budget_pct: float = 2.0,
        rules: RuleSettings = _DEFAULT_RULE_SETTINGS,
        any_key: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(rules=rules)
        if name is not None:
            self.name = name
        self.absolute_floor = absolute_floor
        self.multi_window = multi_window
        self.budget_pct = budget_pct
        # Amendment 2026-09-06 (review finding 5): with ``any_key`` an event is
        # eligible when any of its baseline keys passes both guards (registered
        # R5 uses only the maximum-scoring key's guards).
        self.any_key = any_key
        self.threshold = float("inf")
        self._previous_anomaly_window: dict[str, float] = {}
        self._confirmed_windows: set[tuple[str, float]] = set()

    def fit(self, events: Sequence[EvalEvent]) -> GuardedHBOSPolicy:
        super().fit(events)
        training_scores = [
            float(score)
            for model in [*self.models.values(), self.fallback]
            if model is not None and model.decision_scores_ is not None
            for score in model.decision_scores_
        ]
        if training_scores:
            self.threshold = float(np.quantile(training_scores, 1.0 - self.budget_pct / 100.0))
        return self

    def score(self, event: EvalEvent) -> float:
        rule = self.rules(event)
        vectors = self._vectors(event)
        if rule:
            vector = vectors[0]
            return self._record(vector, 1_000_000.0, rule=rule, guard="rule_floor")
        candidates: list[tuple[float, FeatureVector, bool, bool]] = []
        for vector in vectors:
            raw = self._score_vector(vector)
            enough_volume = vector.context["count_5m"] >= self.absolute_floor
            anomalous = raw >= self.threshold
            window = vector.context["window_epoch_5m"]
            previous = self._previous_anomaly_window.get(vector.baseline_key)
            confirmation_key = (vector.baseline_key, window)
            confirmed = not self.multi_window or confirmation_key in self._confirmed_windows
            if anomalous and previous != window:
                confirmed = not self.multi_window or previous == window - 300.0
                self._previous_anomaly_window[vector.baseline_key] = window
                if confirmed:
                    self._confirmed_windows.add(confirmation_key)
            elif not anomalous and previous is not None and previous != window:
                self._previous_anomaly_window.pop(vector.baseline_key, None)
            candidates.append((raw, vector, confirmed, enough_volume))
        value, vector, confirmed, enough_volume = max(candidates, key=lambda item: item[0])
        if self.any_key:
            passing = [item for item in candidates if item[2] and item[3]]
            if passing:
                value, vector, confirmed, enough_volume = max(passing, key=lambda item: item[0])
        return self._record(
            vector,
            value,
            scorer="guarded_per_entity_hbos",
            volume_guard=enough_volume,
            multi_window_confirmed=confirmed,
            eligible=enough_volume and confirmed,
            hbos_terms=self._hbos_terms(vector),
        )


@dataclass
class _LOFBundle:
    scaler: StandardScaler
    model: LOF


class EntityLOFPolicy(_PolicyBase):
    """R6: per-key local outlier factor over the full feature vector."""

    name = "R6"

    def __init__(self, *, rules: RuleSettings = _DEFAULT_RULE_SETTINGS) -> None:
        super().__init__(rules=rules)
        self.models: dict[str, _LOFBundle] = {}

    def fit(self, events: Sequence[EvalEvent]) -> EntityLOFPolicy:
        groups: defaultdict[str, list[FeatureVector]] = defaultdict(list)
        for event in events:
            for vector in self._vectors(event):
                groups[vector.baseline_key].append(vector)
        for key, vectors in groups.items():
            if len(vectors) < 5:
                continue
            x = _matrix(vectors)
            scaler = StandardScaler().fit(x)
            neighbors = min(20, len(x) - 1)
            model = LOF(n_neighbors=neighbors, contamination=0.1, novelty=True).fit(
                scaler.transform(x)
            )
            self.models[key] = _LOFBundle(scaler, model)
        return self

    def score(self, event: EvalEvent) -> float:
        scored = []
        for vector in self._vectors(event):
            bundle = self.models.get(vector.baseline_key)
            if bundle:
                result = bundle.model.decision_function(
                    bundle.scaler.transform(vector.as_array()[None, :])
                )
                assert result is not None
                value = float(result[0])
            else:
                value = 0.0
            scored.append((value, vector))
        value, vector = max(scored, key=lambda item: item[0])
        return self._record(vector, value, scorer="per_entity_lof")


class AlwaysFastPolicy(_PolicyBase):
    """R7: cost ceiling that admits every event."""

    name = "R7"

    def score(self, event: EvalEvent) -> float:
        del event
        return 1.0


POLICY_CLASSES = {
    "R0": RandomPolicy,
    "R1": RulesOnlyPolicy,
    "R2": GlobalPolicy,
    "R3": RobustGapPolicy,
    "R4": EntityHBOSPolicy,
    "R5": GuardedHBOSPolicy,
    "R6": EntityLOFPolicy,
    "R7": AlwaysFastPolicy,
    "R10": PerSourceGlobalPolicy,
}
# Registered R2 variants (run 4): the same global fit with a different detector.
GLOBAL_VARIANTS = {"R11": "iforest", "R12": "ecod", "R13": "lof"}

# Guarded per-entity HBOS conditions whose operating threshold depends on the budget.
GUARDED_POLICIES = frozenset({"R5"})
# Conditions whose rule hits are admitted outside the budget (R8/R9 in run 3; none registered for run 4).
RULE_EXEMPT_POLICIES: frozenset[str] = frozenset()


def make_policy(
    name: str, cfg: RouterConfig, *, seed: int = 0, budget_pct: float | None = None
) -> _PolicyBase:
    """Construct a preregistered policy from shared router configuration."""

    normalized = name.upper()
    settings = RuleSettings(
        enabled=cfg.rules.enabled,
        dispute_amount=float(getattr(cfg.rules, "dispute_amount", 1_000.0)),
        vote_thread_start_only=bool(getattr(cfg.rules, "vote_thread_start_only", False)),
        dedup_per_subject=bool(getattr(cfg.rules, "dedup_per_subject", False)),
    )
    if normalized == "R0":
        return RandomPolicy(seed=seed, rules=settings)
    if normalized in GUARDED_POLICIES:
        return GuardedHBOSPolicy(
            absolute_floor=max(cfg.guards.absolute_floor, 3.0),
            # R5 is the guarded condition even when the engine profile under
            # evaluation is the R1 baseline with its deviation guard disabled.
            multi_window=True,
            budget_pct=budget_pct or cfg.budget_pct,
            rules=settings,
            name=normalized,
        )
    if normalized in GLOBAL_VARIANTS:
        return GlobalPolicy(method=GLOBAL_VARIANTS[normalized], rules=settings, seed=seed, name=normalized)
    if normalized == "R10":
        return PerSourceGlobalPolicy(rules=settings, seed=seed)
    try:
        policy_class = POLICY_CLASSES[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown E1 policy {name!r}") from exc
    return policy_class(rules=settings)


def budgeted_decisions(
    event_ids: Sequence[str],
    scores: Sequence[float],
    *,
    budget_pct: float,
    tuning_scores: Sequence[float],
    eligible: Sequence[bool] | None = None,
    mandatory: Sequence[bool] | None = None,
    exempt: bool = False,
) -> pd.DataFrame:
    """Select the stable top b% in a month; theta comes only from tuning scores.

    The monthly capacity is exact (up to integer rounding).  `theta` is an
    out-of-sample diagnostic operating threshold and never reads evaluation
    labels; ties at the capacity boundary are broken by event id.

    With ``exempt=True`` the mandatory rows are admitted without consuming the
    capacity (rule hits outside the budget); ``rules_outside_budget`` records
    how many such admissions were made so the rule cost stays visible.
    """

    if len(event_ids) != len(scores):
        raise ValueError("event_ids and scores must have equal length")
    if not 0 <= budget_pct <= 100:
        raise ValueError("budget_pct must be in [0, 100]")
    tuning = np.asarray(tuning_scores, dtype=float)
    theta = float(np.quantile(tuning, 1.0 - budget_pct / 100.0)) if len(tuning) else float("inf")
    count = len(scores)
    capacity = min(count, int(round(count * budget_pct / 100.0)))
    if budget_pct > 0 and count and capacity == 0:
        capacity = 1
    allowed = (
        np.ones(count, dtype=bool)
        if eligible is None
        else np.asarray(eligible, dtype=bool).copy()
    )
    required = (
        np.zeros(count, dtype=bool)
        if mandatory is None
        else np.asarray(mandatory, dtype=bool).copy()
    )
    if len(allowed) != count or len(required) != count:
        raise ValueError("eligible and mandatory must match event_ids")
    allowed |= required
    order = sorted(
        range(count),
        key=lambda index: (
            -int(required[index]),
            -float(scores[index]),
            str(event_ids[index]),
        ),
    )
    # R0--R6 are one total-budget comparison.  Rule hits rank first in R1/R5
    # but never create hidden capacity.  If the rule floor alone exceeds the
    # month capacity, the month is explicitly infeasible and deterministic
    # event-id tie breaking selects the capacity-sized audit sample.
    admitted: set[int] = set()
    outside: set[int] = set()
    for index in order:
        if exempt and required[index]:
            outside.add(index)
            continue
        if len(admitted) >= capacity:
            break
        if allowed[index]:
            admitted.add(index)
    rank = {index: position + 1 for position, index in enumerate(order)}
    required_in_budget = 0 if exempt else int(required.sum())
    return pd.DataFrame(
        {
            "event_id": list(event_ids),
            "score": np.asarray(scores, dtype=float),
            "admitted": [index in admitted or index in outside for index in range(count)],
            "rank": [rank[index] for index in range(count)],
            "theta": theta,
            "above_tuning_theta": [float(value) >= theta for value in scores],
            "eligible": allowed,
            "mandatory": required,
            "capacity": capacity,
            "unused_capacity": max(capacity - len(admitted), 0),
            "rules_over_budget": max(required_in_budget - capacity, 0),
            "budget_feasible": required_in_budget <= capacity,
            "rules_outside_budget": len(outside),
        }
    )


top_budget_decisions = budgeted_decisions
