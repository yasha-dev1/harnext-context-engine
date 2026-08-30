"""Budgeted router policies R0--R7 from docs/evaluation-spec.md §7 E1."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from pyod.models.hbos import HBOS
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


_DEFAULT_RULE_SETTINGS = RuleSettings()


def _walk(value: Any) -> list[str]:
    if isinstance(value, dict):
        result: list[str] = []
        for key, item in value.items():
            result.append(str(key))
            result.extend(_walk(item))
        return result
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(_walk(item))
        return result
    return [] if value is None else [str(value)]


def match_rule(event: EvalEvent, settings: RuleSettings = _DEFAULT_RULE_SETTINGS) -> str | None:
    """Rules floor configured by enablement and the dispute amount threshold."""

    if not settings.enabled:
        return None
    data = event.data or {}
    text = " ".join(_walk(data)).casefold()
    if any(token in text for token in ("critical", "blocker")):
        return "declared_priority"
    if "[vote]" in text:
        return "vote"
    if "cve-" in text or " cve " in f" {text} ":
        return "cve"
    if "on-call" in text or "oncall" in text or "pagerduty" in text:
        return "on_call_page"
    if "dispute" in text:
        for key in ("amount", "dispute_amount", "value"):
            try:
                if float(data.get(key, 0)) >= settings.dispute_amount:
                    return "large_dispute"
            except (TypeError, ValueError):
                continue
    return None


class _PolicyBase:
    name = "base"

    def __init__(self, *, rules: RuleSettings = _DEFAULT_RULE_SETTINGS) -> None:
        self.rule_settings = rules
        self.extractor = CausalFeatureExtractor()
        self.baseline_key_used: str | None = None
        self.features_fired: dict[str, Any] = {}

    def fit(self, events: Sequence[EvalEvent]) -> _PolicyBase:
        del events
        return self

    def rules(self, event: EvalEvent) -> str | None:
        return match_rule(event, self.rule_settings)

    def score(self, event: EvalEvent) -> float:
        raise NotImplementedError

    def _vectors(self, event: EvalEvent) -> list[FeatureVector]:
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


class GlobalPolicy(_PolicyBase):
    """R2: global robust z or HBOS, with no baseline-key conditioning."""

    name = "R2"

    def __init__(
        self, *, method: str = "hbos", rules: RuleSettings = _DEFAULT_RULE_SETTINGS
    ) -> None:
        super().__init__(rules=rules)
        if method not in {"z", "hbos"}:
            raise ValueError("R2 method must be 'z' or 'hbos'")
        self.method = method
        self.model: HBOS | None = None
        self.median = np.zeros(len(FEATURE_NAMES))
        self.scale = np.ones(len(FEATURE_NAMES))

    def fit(self, events: Sequence[EvalEvent]) -> GlobalPolicy:
        vectors = [vector for event in events for vector in self._vectors(event)]
        x = _matrix(vectors)
        if len(x):
            self.median = np.median(x, axis=0)
            mad = np.median(np.abs(x - self.median), axis=0)
            self.scale = np.maximum(1.4826 * mad, 1e-6)
        if self.method == "hbos" and len(x) >= 2:
            self.model = HBOS(n_bins=min(10, max(3, int(math.sqrt(len(x))))), contamination=0.1)
            self.model.fit(x)
        return self

    def score(self, event: EvalEvent) -> float:
        vectors = self._vectors(event)
        scored: list[tuple[float, FeatureVector]] = []
        for vector in vectors:
            x = vector.as_array()[None, :]
            if self.model is not None:
                result = self.model.decision_function(x)
                assert result is not None
                value = float(result[0])
            else:
                value = float(np.max(np.abs((x[0] - self.median) / self.scale)))
            scored.append((value, vector))
        value, vector = max(scored, key=lambda item: item[0])
        return self._record(vector, value, scorer=f"global_{self.method}")


class RobustGapPolicy(_PolicyBase):
    """R3: per-baseline-key robust z-score on log inter-arrival gap only."""

    name = "R3"

    def __init__(self, *, rules: RuleSettings = _DEFAULT_RULE_SETTINGS) -> None:
        super().__init__(rules=rules)
        self.stats: dict[str, tuple[float, float]] = {}

    def fit(self, events: Sequence[EvalEvent]) -> RobustGapPolicy:
        values: defaultdict[str, list[float]] = defaultdict(list)
        for event in events:
            for vector in self._vectors(event):
                values[vector.baseline_key].append(vector.values["log_gap_s"])
        for key, samples in values.items():
            median = float(np.median(samples))
            mad = float(np.median(np.abs(np.asarray(samples) - median)))
            self.stats[key] = (median, max(1.4826 * mad, 0.1))
        return self

    def score(self, event: EvalEvent) -> float:
        scored = []
        for vector in self._vectors(event):
            median, scale = self.stats.get(vector.baseline_key, (0.0, 1.0))
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

    def score(self, event: EvalEvent) -> float:
        scored = [(self._score_vector(vector), vector) for vector in self._vectors(event)]
        value, vector = max(scored, key=lambda item: item[0])
        return self._record(vector, value, scorer="per_entity_hbos")


class GuardedHBOSPolicy(EntityHBOSPolicy):
    """R5: rules + per-key HBOS + absolute volume and confirmation guards."""

    name = "R5"

    def __init__(
        self,
        *,
        absolute_floor: float = 1.0,
        multi_window: bool = True,
        rules: RuleSettings = _DEFAULT_RULE_SETTINGS,
    ) -> None:
        super().__init__(rules=rules)
        self.absolute_floor = absolute_floor
        self.multi_window = multi_window
        self.thresholds: dict[str, float] = {}
        self._previous_anomaly: defaultdict[str, bool] = defaultdict(bool)

    def fit(self, events: Sequence[EvalEvent]) -> GuardedHBOSPolicy:
        super().fit(events)
        # Thresholds are learned solely from the tuning vectors already retained
        # by the fitted PyOD estimators.
        for key, model in self.models.items():
            assert model.decision_scores_ is not None
            self.thresholds[key] = float(np.quantile(model.decision_scores_, 0.95))
        return self

    def score(self, event: EvalEvent) -> float:
        rule = self.rules(event)
        vectors = self._vectors(event)
        if rule:
            vector = vectors[0]
            return self._record(vector, 1_000_000.0, rule=rule, guard="rule_floor")
        candidates: list[tuple[float, FeatureVector, bool]] = []
        for vector in vectors:
            raw = self._score_vector(vector)
            enough_volume = vector.context["count_5m"] >= self.absolute_floor
            anomalous = raw >= self.thresholds.get(vector.baseline_key, float("inf"))
            confirmed = not self.multi_window or (
                anomalous and self._previous_anomaly[vector.baseline_key]
            )
            self._previous_anomaly[vector.baseline_key] = anomalous
            guarded = raw if enough_volume and confirmed else -float("inf")
            candidates.append((guarded, vector, confirmed))
        value, vector, confirmed = max(candidates, key=lambda item: item[0])
        if not np.isfinite(value):
            value = -1e12
        return self._record(
            vector,
            value,
            scorer="guarded_per_entity_hbos",
            volume_guard=vector.context["count_5m"] >= self.absolute_floor,
            multi_window_confirmed=confirmed,
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
}


def make_policy(name: str, cfg: RouterConfig, *, seed: int = 0) -> _PolicyBase:
    """Construct a preregistered policy from shared router configuration."""

    normalized = name.upper()
    settings = RuleSettings(enabled=cfg.rules.enabled)
    if normalized == "R0":
        return RandomPolicy(seed=seed, rules=settings)
    if normalized == "R5":
        return GuardedHBOSPolicy(
            absolute_floor=max(cfg.guards.absolute_floor, 1.0),
            # R5 is the guarded condition even when the engine profile under
            # evaluation is the R1 baseline with its deviation guard disabled.
            multi_window=True,
            rules=settings,
        )
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
) -> pd.DataFrame:
    """Select the stable top b% in a month; theta comes only from tuning scores.

    The monthly capacity is exact (up to integer rounding).  `theta` is an
    out-of-sample diagnostic operating threshold and never reads evaluation
    labels; ties at the capacity boundary are broken by event id.
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
    order = sorted(range(count), key=lambda index: (-float(scores[index]), str(event_ids[index])))
    admitted = set(order[:capacity])
    rank = {index: position + 1 for position, index in enumerate(order)}
    return pd.DataFrame(
        {
            "event_id": list(event_ids),
            "score": np.asarray(scores, dtype=float),
            "admitted": [index in admitted for index in range(count)],
            "rank": [rank[index] for index in range(count)],
            "theta": theta,
            "above_tuning_theta": [float(value) >= theta for value in scores],
        }
    )


top_budget_decisions = budgeted_decisions
