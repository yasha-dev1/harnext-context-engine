"""Rolling month-ahead E1 experiment from docs/evaluation-spec.md §7 E1."""

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportGeneralTypeIssues=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportReturnType=false

from __future__ import annotations

import gc
import json
import math
import multiprocessing
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from harnext_eval.agents.envelope import build as build_envelope
from harnext_eval.config import EngineConfig
from harnext_eval.corpus import CorpusHandle
from harnext_eval.e1.calibration import calibration_spearman, decile_rates, lift_over_rules
from harnext_eval.e1.features import CausalFeatureExtractor, FeatureVector
from harnext_eval.e1.labels import (
    ABSTAIN,
    DEFAULT_LABELING_FUNCTIONS,
    LabelModelResult,
    build_labels,
)
from harnext_eval.e1.policies import (
    GUARDED_POLICIES,
    RULE_EXEMPT_POLICIES,
    budgeted_decisions,
    make_policy,
)
from harnext_eval.e1.prereg import PRIMARY_CONTRASTS
from harnext_eval.e1.score import (
    affiliation_precision_recall,
    always_flag_sanity_scorer,
    delay_summary,
    detection_delays,
    flip_labels,
    jitter_onsets,
    label_situations,
    nab_low_fn_score,
    precision_at_budget,
    random_sanity_scorer,
    recall_at_budget,
    swap_labels,
    timestamped_affiliation_precision_recall,
    vus_pr,
)
from harnext_eval.grade.action import grade_action
from harnext_eval.providers.factory import make_llm
from harnext_eval.providers.llm import FakeLLM, LLMProvider
from harnext_eval.registry import ExperimentResult, register_experiment
from harnext_eval.replay.gate import leakage_gate
from harnext_eval.stats.stats import paired_difference_bca
from harnext_eval.stores.base import StoreHandle
from harnext_eval.types import EvalEvent, RouterRecord, Task

_BUDGETS = (1.0, 2.0, 5.0, 10.0)
_POLICIES = (*(f"R{index}" for index in range(8)), "R10", "R11", "R12", "R13")
_SHARED_BUDGET_POLICIES = tuple(f"R{index}" for index in range(7))
_ROWINDEX_SECONDARY = (
    "precision_at_b",
    "vus_pr",
    "affiliation_precision_rowindex",
    "affiliation_recall_rowindex",
    "nab_low_fn_rowindex",
)
_VUS_MAX_BUFFER = 5
_SCORE_ROW_GROUP_SIZE = 100_000
# Same causal fold reused by policies/budgets within a rolling month. Fit/model
# state is never shared. The calibration suffix has the same earlier history.
_FEATURE_CACHE: dict[bool, dict[str, list[FeatureVector]]] = {}


def _fitted_policy(
    name: str, tuning: list[EvalEvent], cfg: EngineConfig, seed: int, budget: float,
):
    policy = make_policy(name, cfg.router, seed=seed, budget_pct=budget)
    policy.feature_cache = _FEATURE_CACHE.get(bool(getattr(policy, "global_features", False)))
    return policy.fit(tuning)


def _tuning_start(month: str) -> str:
    """Fit on the trailing 12 calendar months, strictly before evaluation."""
    return f"{int(month[:4]) - 1:04d}{month[4:]}"

_GOLD_ONLY_FIELDS = {
    "cost_weight",
    "hard_negative",
    "injected_positive",
    "is_urgent",
    "situation_archetype",
    "situation_label",
    "situation_onset",
    "urgent",
}


def _month(event: EvalEvent) -> str:
    return f"{event.time.year:04d}-{event.time.month:02d}"


def _source(event: EvalEvent) -> str:
    return event.source.split(":", 1)[0]


def _without_gold(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_gold(item)
            for key, item in value.items()
            if str(key).casefold() not in _GOLD_ONLY_FIELDS
        }
    if isinstance(value, list):
        return [_without_gold(item) for item in value]
    return value


def _router_events(events: Iterable[EvalEvent]) -> list[EvalEvent]:
    """Return policy-visible copies with evaluation-only gold stripped."""

    return [event.model_copy(update={"data": _without_gold(event.data or {})}) for event in events]


def _as_timestamp(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True)


def _situation_gold(
    corpus: CorpusHandle, events: list[EvalEvent]
) -> tuple[dict[str, float] | None, pd.DataFrame]:
    """Read Corpus S's sidecar manifest as exact gold when it is present."""

    raw: Any = corpus.meta.get("injected_situations", corpus.meta.get("situations"))
    if not isinstance(raw, list) and events and all(
        isinstance((event.data or {}).get("injected_positive"), bool) for event in events
    ):
        # The synthetic replay carries exact construction flags even when its
        # in-memory world-state sidecar is absent after a CLI round trip.
        raw = [
            {"event_id": event.id, "entity": event.subject, "onset": event.time,
             "cost_weight": (event.data or {}).get("cost_weight", 1.0)}
            for event in events if (event.data or {})["injected_positive"]
        ]
    if not isinstance(raw, list):
        return None, pd.DataFrame(
            columns=["situation_id", "entity", "onset", "end", "label", "cost_weight"]
        )
    labels = {event.id: 0.0 for event in events}
    event_by_id = {event.id: event for event in events}
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        hard_negative = bool(item.get("hard_negative", False))
        label_value = item.get("label", item.get("positive", not hard_negative))
        if isinstance(label_value, str):
            positive = label_value.casefold() in {"1", "positive", "true", "urgent"}
        else:
            positive = bool(label_value)
        event_ids = item.get("event_ids", item.get("event_id", []))
        if isinstance(event_ids, str):
            event_ids = [event_ids]
        onset = _as_timestamp(item.get("onset", event_by_id.get(event_ids[0]).time if event_ids else events[0].time))
        end_value = item.get("end")
        end = _as_timestamp(end_value) if end_value is not None else onset
        entity = str(item.get("entity", event_by_id.get(event_ids[0]).subject if event_ids else ""))
        if not event_ids and entity:
            event_ids = [
                event.id
                for event in events
                if event.subject == entity and onset <= pd.Timestamp(event.time) <= end
            ]
        for event_id in event_ids:
            if str(event_id) in labels:
                labels[str(event_id)] = float(positive)
        rows.append(
            {
                "situation_id": str(item.get("situation_id", item.get("id", f"situation-{index:04d}"))),
                "event_id": str(event_ids[0]) if event_ids else "",
                "entity": entity,
                "onset": onset,
                "end": end,
                "label": positive,
                "cost_weight": float(item.get("cost_weight", 1.0 if positive else 0.0)),
            }
        )
    situations = pd.DataFrame(rows)
    if not situations.empty:
        situations = situations[situations["label"].astype(bool)].reset_index(drop=True)
    return labels, situations


def _constructed_label_result(
    labels: dict[str, float],
) -> LabelModelResult:
    """Represent exact Corpus-S gold without running the quadratic LF pipeline."""

    event_ids = list(labels)
    columns = [function.name for function in DEFAULT_LABELING_FUNCTIONS]
    votes = pd.DataFrame(ABSTAIN, index=event_ids, columns=columns, dtype=int)
    votes.index.name = "event_id"
    observability = pd.DataFrame(False, index=event_ids, columns=columns, dtype=bool)
    observability.index.name = "event_id"
    diagnostics = pd.DataFrame(
        [
            {
                "function": name,
                "accuracy": float("nan"),
                "coverage": 0.0,
                "overlap": 0.0,
                "conflict": 0.0,
                "positive_votes": 0,
                "negative_votes": 0,
                "unknown": len(event_ids),
            }
            for name in columns
        ]
    ).set_index("function")
    return LabelModelResult(
        probabilities=pd.Series(labels, name="p_urgent", dtype=float),
        votes=votes,
        observability=observability,
        diagnostics=diagnostics,
        declared_outcome_agreement=float("nan"),
        declared_outcome_comparable=0,
    )


def _calibration_scores(
    name: str,
    tuning: list[EvalEvent],
    cfg: EngineConfig,
    seed: int,
    budget: float,
) -> list[float]:
    """Score a tuning suffix using a model fitted only on an earlier prefix."""

    split = max(1, int(len(tuning) * 0.7))
    fit_events = tuning[:split]
    score_events = tuning[split:]
    if not score_events:
        score_events = fit_events[-1:]
        fit_events = fit_events[:-1]
    policy = _fitted_policy(name, fit_events, cfg, seed, budget)
    values = [policy.score(event) for event in score_events]
    return [float(value) for value in values if np.isfinite(value)]


def _score_month(
    name: str,
    tuning: list[EvalEvent],
    evaluation: list[EvalEvent],
    cfg: EngineConfig,
    seed: int,
    budget: float,
) -> tuple[pd.DataFrame, list[float]]:
    tuning_scores = _calibration_scores(name, tuning, cfg, seed, budget)
    policy = _fitted_policy(name, tuning, cfg, seed, budget)
    rows: list[dict[str, Any]] = []
    for event in evaluation:
        score = policy.score(event)
        rule = policy.rules(event)
        record = RouterRecord(
            event_id=event.id,
            t=event.time,
            score=score,
            lane="unranked",
            policy=name,
            budget_pct=budget,
            baseline_key_used=policy.baseline_key_used,
            features_fired=policy.features_fired,
        )
        rows.append(
            {
                **record.model_dump(),
                "decision_ts": event.time,
                "decision_latency_ms": 0.0,
                "routing_tokens": 0,
                "routing_dollars": 0.0,
                "source": _source(event),
                "subject": event.subject,
                "rule_flag": rule is not None,
                "rule": rule,
                "month": _month(event),
            }
        )
    return pd.DataFrame(rows), tuning_scores


def _admit_month(
    scored: pd.DataFrame,
    *,
    name: str,
    budget: float,
    tuning_scores: list[float],
) -> pd.DataFrame:
    """Rank once across the whole evaluation month, then derive report slices."""

    eligible = np.ones(len(scored), dtype=bool)
    decisions = pd.DataFrame()
    if name == "R7":
        decisions = pd.DataFrame(
            {
                "event_id": scored["event_id"].tolist(),
                "admitted": True,
                "rank": np.arange(1, len(scored) + 1),
                "theta": float("-inf"),
                "above_tuning_theta": True,
                "eligible": True,
                "mandatory": False,
                "capacity": len(scored),
                "unused_capacity": 0,
                "rules_over_budget": 0,
                "budget_feasible": True,
                "rules_outside_budget": 0,
            }
        )
    elif name == "R1":
        eligible = scored["rule_flag"].astype(bool).to_numpy()
    elif name in GUARDED_POLICIES:
        eligible = np.asarray(
            [
                bool(features.get("eligible", False)) or bool(rule)
                for features, rule in zip(scored["features_fired"], scored["rule_flag"], strict=True)
            ]
        )
    if name != "R7":
        mandatory = (
            scored["rule_flag"].astype(bool).to_numpy()
            if name == "R1" or name in GUARDED_POLICIES
            else None
        )
        decisions = budgeted_decisions(
            scored["event_id"].tolist(),
            scored["score"].tolist(),
            budget_pct=budget,
            tuning_scores=tuning_scores,
            eligible=eligible,
            mandatory=mandatory,
            exempt=name in RULE_EXEMPT_POLICIES,
        ).drop(columns="score")
    admitted = scored.merge(decisions, on="event_id", how="left", validate="one_to_one")
    admitted["lane"] = np.where(admitted["admitted"], "fast", "batch")
    full = admitted.assign(population="full")
    rule_negative = admitted[~admitted["rule_flag"].astype(bool)].copy()
    rule_negative["population"] = "rule_negative"
    result = pd.concat([full, rule_negative], ignore_index=True)
    for column in ("event_id", "policy", "lane", "source", "subject", "rule", "month", "population", "baseline_key_used"):
        result[column] = result[column].astype("category")
    return result


def _population_groups(frame: pd.DataFrame):
    """Derive report slices without retaining duplicate population score rows."""
    full = frame[frame["population"] == "full"]
    for identifiers, group in full.groupby(["month", "policy", "budget_pct"], sort=True, observed=True):
        yield (*identifiers, "full"), group
        negative = group[~group["rule_flag"].astype(bool)]
        if not negative.empty:
            yield (*identifiers, "rule_negative"), negative


_WORKER_STATE: dict[str, Any] = {}


def _init_worker(
    tuning: list[EvalEvent],
    evaluation: list[EvalEvent],
    cfg: EngineConfig,
    seed: int,
    event_labels: dict[str, float],
    constructed: bool,
    caches: dict[bool, dict[str, list[FeatureVector]]],
) -> None:
    """Receive one month's inputs once per spawned worker (no parent-heap inheritance).

    A forked worker copied the parent's whole, ever-growing heap through
    copy-on-write (about 7 GB per worker by month 15); a spawned worker holds
    only this month's events, labels and feature cache.
    """

    _WORKER_STATE.update(
        tuning=tuning, evaluation=evaluation, cfg=cfg, seed=seed,
        event_labels=event_labels, constructed=constructed,
    )
    _FEATURE_CACHE.clear()
    _FEATURE_CACHE.update(caches)


def _policy_month(args):
    if len(args) == 2:
        name, budgets = args
        state = _WORKER_STATE
        tuning, evaluation, cfg, seed = state["tuning"], state["evaluation"], state["cfg"], state["seed"]
        event_labels, constructed = state["event_labels"], state["constructed"]
    else:
        name, tuning, evaluation, cfg, seed, event_labels, constructed, *selection = args
        budgets = selection[0] if selection else _BUDGETS
    parts = []
    budget_scores = {}
    guarded = name in GUARDED_POLICIES
    for budget in (budgets if guarded else (_BUDGETS[0],)):
        budget_scores[budget] = _score_month(name, tuning, evaluation, cfg, seed, budget)
    for budget in budgets:
        scored, tuning_scores = budget_scores[budget if guarded else _BUDGETS[0]]
        scored = scored.copy()
        scored["budget_pct"] = budget
        scored["p_urgent"] = scored["event_id"].map(event_labels).astype(float)
        scored["label"] = scored["p_urgent"]
        scored["constructed_label"] = constructed
        part = _admit_month(scored, name=name, budget=budget, tuning_scores=tuning_scores)
        part = part[part["population"] == "full"].copy()
        # Compact the per-event feature payload here, in the worker: one JSON
        # string per row instead of a ~20-key dict kept alive for 15M rows.
        part["features_fired"] = part["features_fired"].map(
            lambda value: value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _metric_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for identifiers, month_group in _population_groups(frame):
        month, policy, budget, population = identifiers
        source_groups = [("all", month_group), *month_group.groupby("source", sort=True, observed=True)]
        for source, group in source_groups:
            known = group[group["label"].notna()]
            ordered = known.sort_values(["t", "event_id"])
            labels = ordered["label"].to_numpy(dtype=float)
            admitted = ordered["admitted"].to_numpy(dtype=bool)
            scores = (
                ordered["score"].replace([np.inf, -np.inf], np.nan).fillna(-1e30).to_numpy()
            )
            affiliation_p, affiliation_r = (
                affiliation_precision_recall(labels, admitted)
                if len(labels) and np.any(labels >= 0.5)
                else (float("nan"), float("nan"))
            )
            rows.append(
                {
                    "month": month,
                    "policy": policy,
                    "budget_pct": budget,
                    "population": population,
                    "source": source,
                    "n": len(group),
                    "n_known": len(known),
                    "unknown_labels": len(group) - len(known),
                    "prevalence": float(np.mean(labels >= 0.5)) if len(labels) else float("nan"),
                    "admission_rate": float(np.mean(group["admitted"])) if len(group) else float("nan"),
                    "recall_at_b": recall_at_budget(labels, admitted),
                    "precision_at_b": precision_at_budget(labels, admitted),
                    "zero_admissions": int(not admitted.any()),
                    "vus_pr": vus_pr(
                        labels,
                        scores,
                        max_buffer=_VUS_MAX_BUFFER,
                        timestamps=ordered["t"],
                    )
                    if len(labels)
                    else float("nan"),
                    # Row-index geometry (amendment 2026-09-06, review finding 8):
                    # these credit adjacent rows regardless of entity or elapsed
                    # time and are reported only under the *_rowindex suffix. The
                    # timestamped, subject-separated versions live in
                    # robustness.csv / delays.csv (label-derived situations).
                    "affiliation_precision_rowindex": affiliation_p,
                    "affiliation_recall_rowindex": affiliation_r,
                    "nab_low_fn_rowindex": nab_low_fn_score(labels, admitted),
                    "decision_latency_ms": float(group["decision_latency_ms"].mean()),
                    "tokens": int(group["routing_tokens"].sum()),
                    "dollars": float(group["routing_dollars"].sum()),
                    "unused_capacity": int(group["unused_capacity"].iloc[0]),
                    "rules_over_budget": int(group["rules_over_budget"].iloc[0]),
                    "budget_feasible": bool(group["budget_feasible"].iloc[0]),
                    "rules_outside_budget": int(group["rules_outside_budget"].iloc[0]),
                    "rule_hits": int(group["rule_flag"].astype(bool).sum()),
                }
            )
    return rows


def _paired_primary(scores: pd.DataFrame, seed: int) -> dict[str, Any]:
    subset = scores[
        (scores["budget_pct"] == 2.0)
        & (~scores["rule_flag"].astype(bool))
        & scores["label"].notna()
        & (scores["label"] >= 0.5)
    ]
    pivot = subset.pivot_table(
        index=["event_id", "subject", "month"], columns="policy", values="admitted", aggfunc="first", observed=True
    ).reset_index()
    by_policy = {
        str(name): float(pivot[name].mean()) for name in _POLICIES if name in pivot.columns
    }
    result: dict[str, Any] = {
        "metric": "recall_at_2pct_rule_negative",
        "by_policy": by_policy,
        "n_positives": int(len(pivot)),
        "n_subjects": int(pivot["subject"].nunique()) if not pivot.empty else 0,
        "n_months": int(pivot["month"].nunique()) if not pivot.empty else 0,
        "contrasts": list(PRIMARY_CONTRASTS),
    }
    for contrast_name in PRIMARY_CONTRASTS:
        left, right = contrast_name.split("-")
        key = f"{left.casefold()}_minus_{right.casefold()}"
        if {left, right} <= set(pivot.columns) and pivot["subject"].nunique() >= 2:
            contrast = paired_difference_bca(
                pivot[left].astype(float),
                pivot[right].astype(float),
                pivot["subject"],
                n_resamples=10_000,
                random_state=seed,
            )
            result[key] = contrast.effect
            result[f"{key}_ci_low"] = contrast.ci_low
            result[f"{key}_ci_high"] = contrast.ci_high
            result[f"{key}_n_entities"] = contrast.n_clusters
            # Sensitivity (review finding 9): calendar-month clusters absorb the
            # linked-incident and notification-burst dependence that subject
            # clusters cannot see.
            if pivot["month"].nunique() >= 2:
                by_month = paired_difference_bca(
                    pivot[left].astype(float),
                    pivot[right].astype(float),
                    pivot["month"].astype(str),
                    n_resamples=10_000,
                    random_state=seed,
                )
                result[f"{key}_month_ci_low"] = by_month.ci_low
                result[f"{key}_month_ci_high"] = by_month.ci_high
        else:
            result[key] = float("nan")
            result[f"{key}_ci_low"] = float("nan")
            result[f"{key}_ci_high"] = float("nan")
            result[f"{key}_n_entities"] = int(pivot["subject"].nunique()) if not pivot.empty else 0
    return result


def _situation_metrics(
    scores: pd.DataFrame, situations: pd.DataFrame, *, seed: int, population: str = "full"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if situations.empty:
        return pd.DataFrame(), pd.DataFrame()
    delay_parts: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    full = scores[scores["population"] == "full"]
    for (policy, budget), relevant in full.groupby(["policy", "budget_pct"], sort=True, observed=True):
        relevant = relevant.drop_duplicates("event_id")
        condition_situations = situations[
            situations["event_id"].isin(relevant["event_id"])
        ].copy()
        if condition_situations.empty:
            continue
        admissions = relevant.rename(columns={"subject": "entity", "t": "time"})
        admissions = admissions[admissions["entity"].isin(condition_situations["entity"])]
        delays = detection_delays(condition_situations, admissions)
        delays["policy"] = policy
        delays["budget_pct"] = budget
        delays["population"] = population
        delay_parts.append(delays)
        summary = delay_summary(delays["delay_s"])
        affiliation_p, affiliation_r = timestamped_affiliation_precision_recall(
            condition_situations, admissions
        )
        jittered = condition_situations.copy()
        jittered["onset"] = jitter_onsets(jittered["onset"], seed=seed)
        jittered_p, jittered_r = timestamped_affiliation_precision_recall(
            jittered, admissions
        )
        jittered_delays = detection_delays(jittered, admissions)
        jittered_summary = delay_summary(jittered_delays["delay_s"])
        rows.append(
            {
                "policy": policy,
                "budget_pct": budget,
                "population": population,
                "n_situations": int(len(condition_situations)),
                "affiliation_precision": affiliation_p,
                "affiliation_recall": affiliation_r,
                "jitter_affiliation_precision": jittered_p,
                "jitter_affiliation_recall": jittered_r,
                "delay_p50_s": summary["p50_s"],
                "delay_p95_s": summary["p95_s"],
                "detected_rate": summary["detected_rate"],
                "jitter_delay_p50_s": jittered_summary["p50_s"],
                "jitter_delay_p95_s": jittered_summary["p95_s"],
                "jitter_detected_rate": jittered_summary["detected_rate"],
            }
        )
    return (
        pd.concat(delay_parts, ignore_index=True) if delay_parts else pd.DataFrame(),
        pd.DataFrame(rows),
    )


def _score_chunk(frame: pd.DataFrame) -> pd.DataFrame:
    serializable = frame.drop(columns="population").copy()
    serializable["rule_negative"] = ~serializable["rule_flag"].astype(bool)
    serializable["features_fired"] = serializable["features_fired"].map(
        lambda value: value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    )
    return serializable


def _write_scores(frame: pd.DataFrame, path: Path) -> bool:
    # Avoid materializing JSON strings and Arrow buffers for millions of rows
    # at once. The frame itself already stores only the full population.
    full = frame if frame["population"].eq("full").all() else frame[frame["population"] == "full"]
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        payload = _score_chunk(full).to_json(orient="table", date_format="iso")
        assert payload is not None
        path.write_text(payload, encoding="utf-8")
        path.with_suffix(".parquet.format.json").write_text(
            json.dumps({"format": "pandas-table-json", "reason": "no parquet engine"}) + "\n",
            encoding="utf-8",
        )
        return False
    writer = None
    try:
        for start in range(0, max(len(full), 1), _SCORE_ROW_GROUP_SIZE):
            chunk = _score_chunk(full.iloc[start:start + _SCORE_ROW_GROUP_SIZE])
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=_SCORE_ROW_GROUP_SIZE)
    finally:
        if writer is not None:
            writer.close()
    return True


def _features_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except ValueError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return value if isinstance(value, dict) else {}


def _write_attribution(scores: pd.DataFrame, path: Path) -> None:
    lines = ["# E1 feature attribution", ""]
    for policy in ("R5", "R2", "R10"):
        selected = scores[
            (scores["policy"] == policy)
            & (scores["budget_pct"] == 2.0)
            & (scores["population"] == "full")
            & scores["admitted"]
            & (~scores["rule_flag"].astype(bool))
            & (scores["label"] >= 0.5)
        ].drop_duplicates("event_id")
        totals: dict[str, float] = {}
        for features in selected["features_fired"].map(_features_dict):
            for name, value in features.get("hbos_terms", {}).items():
                totals[name] = totals.get(name, 0.0) + float(value)
        lines.extend([f"## {policy}: HBOS contributions on true-positive deviation admissions", ""])
        lines.extend(f"- {name}: {value:.6f}" for name, value in sorted(totals.items(), key=lambda item: -item[1]))
        lines.extend(["", f"### {policy}: audited cases", ""])
        for _, row in selected.head(10).iterrows():
            terms = _features_dict(row["features_fired"]).get("hbos_terms", {})
            leading = sorted(terms.items(), key=lambda item: -float(item[1]))[:3]
            lines.append(f"- `{row['event_id']}` ({row['subject']}): {leading}")
        if selected.empty:
            lines.append(f"- No true-positive {policy} deviation admissions at 2%.")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_human_sanity_sample(
    scores: pd.DataFrame, events: Sequence[EvalEvent], out_dir: Path, *, size: int = 100
) -> Path:
    """Write the spec's human sanity sample: top-scored rule-negative events.

    Half comes from the per-source design condition (R10), half from the global
    scorer (R2); labels and scores go to a separate key file so the annotator
    sheet is blind.
    """

    from harnext_eval.e1.labels import _text

    by_id = {event.id: event for event in events}
    picks: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    for policy, order in (("R10", ["score"]), ("R2", ["score"])):
        subset = scores[
            (scores["policy"] == policy)
            & (scores["budget_pct"] == 2.0)
            & (scores["population"] == "full")
            & (~scores["rule_flag"].astype(bool))
            & scores["label"].notna()
        ].drop_duplicates("event_id")
        if subset.empty:
            continue
        ranked = subset.sort_values(order, ascending=[False] * len(order))
        for _, row in ranked.iterrows():
            if len([pick for pick in picks if pick[0] == policy]) >= size // 2:
                break
            event_id = str(row["event_id"])
            if event_id in seen:
                continue
            seen.add(event_id)
            picks.append((policy, event_id, float(row["score"])))
    rows: list[dict[str, Any]] = []
    keys: list[dict[str, Any]] = []
    labels = scores.drop_duplicates("event_id").set_index("event_id")["label"]
    for item, (policy, event_id, score) in enumerate(picks, start=1):
        event = by_id.get(event_id)
        text = _text(event) if event is not None else ""
        rows.append(
            {
                "item": item,
                "event_id": event_id,
                "source": event.source if event is not None else "",
                "subject": event.subject if event is not None else "",
                "type": event.type if event is not None else "",
                "time": event.time.isoformat() if event is not None else "",
                "excerpt": text[:600],
                "would_interrupt_annotator_1": "",
                "would_interrupt_annotator_2": "",
            }
        )
        keys.append(
            {
                "item": item,
                "event_id": event_id,
                "policy": policy,
                "score": score,
                "p_urgent": float(labels.get(event_id, float("nan"))),
            }
        )
    path = out_dir / "human_sanity_sample.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    pd.DataFrame(keys).to_csv(out_dir / "human_sanity_key.csv", index=False)
    return path


def _metric_remediation(metrics: pd.DataFrame, *, design: str = "R10", floor: str = "R0") -> pd.DataFrame:
    """Prospective kept/dropped record for every secondary metric (review finding 6).

    A metric is kept when the design condition is separated from the random
    floor by more than twice the paired standard error over evaluation months
    (2% budget, rule-negative population, all sources). The always-flag ceiling
    is reported next to it. The decision is mechanical and recorded before any
    interpretation.
    """

    rows: list[dict[str, Any]] = []
    selected = metrics[
        (metrics["budget_pct"] == 2.0)
        & (metrics["population"] == "rule_negative")
        & (metrics["source"] == "all")
    ]
    for metric in _ROWINDEX_SECONDARY:
        if metric not in selected.columns:
            continue
        pivot = selected.pivot_table(index="month", columns="policy", values=metric, aggfunc="first", observed=True)
        row: dict[str, Any] = {"metric": metric, "design": design, "floor": floor}
        for policy in (floor, "R5", "R2", design, "R7"):
            row[f"mean_{policy}"] = float(pivot[policy].mean()) if policy in pivot.columns else float("nan")
        if design in pivot.columns and floor in pivot.columns:
            paired = (pivot[design] - pivot[floor]).dropna()
            n = int(len(paired))
            mean = float(paired.mean()) if n else float("nan")
            se = float(paired.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
            kept = bool(n >= 10 and math.isfinite(se) and abs(mean) > 2.0 * se)
            row.update(paired_months=n, design_minus_floor=mean, standard_error=se, decision="kept" if kept else "dropped")
            row["reason"] = (
                "design separated from random floor by > 2 paired SE"
                if kept
                else "design not separated from the random floor (|mean| <= 2 SE or < 10 months); reported, not interpreted"
            )
        else:
            row.update(paired_months=0, design_minus_floor=float("nan"), standard_error=float("nan"), decision="dropped", reason="policy rows missing")
        rows.append(row)
    return pd.DataFrame(rows)


def _rule_exempt_capacity_respected(scores: pd.DataFrame) -> bool:
    """R8/R9 deviation (non-mandatory) admissions never exceed the month capacity."""

    subset = scores[scores["policy"].isin(sorted(RULE_EXEMPT_POLICIES)) & (scores["population"] == "full")]
    if subset.empty:
        return True
    deviation = (subset["admitted"].astype(bool) & ~subset["mandatory"].astype(bool)).astype(int)
    grouped = subset.assign(_deviation=deviation).groupby(["month", "policy", "budget_pct"], observed=True)
    return bool((grouped["_deviation"].sum() <= grouped["capacity"].first()).all())


def _text_provenance(events: Sequence[EvalEvent]) -> dict[str, Any]:
    """Count events whose payload text was edited after the event time (review finding 7)."""

    edited = 0
    dated = 0
    for event in events:
        data = event.data or {}
        stamp = data.get("updated") or data.get("updated_at")
        if not isinstance(stamp, str) or not stamp:
            continue
        dated += 1
        try:
            parsed = pd.Timestamp(stamp)
            parsed = parsed.tz_convert("UTC") if parsed.tzinfo else parsed.tz_localize("UTC")
        except (ValueError, TypeError):
            continue
        if parsed > pd.Timestamp(event.time):
            edited += 1
    return {
        "events": len(events),
        "with_edit_timestamp": dated,
        "edited_after_event_time": edited,
        "note": (
            "summary/description/comment/PR text is export-time snapshot text; edits after t "
            "cannot be reconstructed from the sources and are declared as a limitation"
        ),
    }


def _write_charts(calibration: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path) -> list[Path]:
    calibration_path = out_dir / "calibration.png"
    operating_path = out_dir / "operating_curves.png"
    fig, axis = plt.subplots(figsize=(6, 4))
    if not calibration.empty:
        curve = calibration.groupby("decile", as_index=False)["urgency_rate"].mean()
        axis.plot(curve["decile"], curve["urgency_rate"], marker="o")
    axis.set(xlabel="score decile", ylabel="revealed urgency rate", title="E1 calibration")
    fig.tight_layout()
    fig.savefig(calibration_path)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(6, 4))
    selected = metrics[(metrics["source"] == "all") & (metrics["population"] == "rule_negative")]
    for policy, group in selected.groupby("policy", sort=True, observed=True):
        curve = group.groupby("budget_pct", as_index=False)["recall_at_b"].mean()
        axis.plot(curve["budget_pct"], curve["recall_at_b"], marker="o", label=policy)
    axis.set(xlabel="admission budget (%)", ylabel="recall", title="E1 operating curves")
    if not selected.empty:
        axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(operating_path)
    plt.close(fig)
    return [calibration_path, operating_path]


def _preflight(corpus: CorpusHandle, situations: pd.DataFrame) -> pd.DataFrame:
    name = corpus.name.casefold()
    smoke = bool(corpus.meta.get("smoke", name == "synthetic"))
    real = "kafka" in name or "flink" in name
    rows = [
        {"requirement": "rolling_month_ahead", "status": "run"},
        {"requirement": "multi_corpus_cli", "status": "supported-not-run"},
        {
            "requirement": "R-long_2022-01_to_2026-06",
            "status": "supported-not-run" if smoke or not real else "run",
        },
        {
            "requirement": "flink_replication",
            "status": "run" if "flink" in name else "supported-not-run",
        },
        {
            "requirement": "corpus_s_200_situations_3_seeds",
            "status": "run" if len(situations) >= 200 else "supported-not-run",
        },
    ]
    return pd.DataFrame(rows)


_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "assignee_candidates": {"type": "array", "items": {"type": "string"}},
        "reviewer_candidates": {"type": "array", "items": {"type": "string"}},
        "component": {"type": ["string", "null"]},
        "duplicate_of": {"type": ["string", "null"]},
        "priority_change": {"type": ["string", "null"]},
        "suspected_locations": {"type": "array", "items": {"type": "string"}},
        "draft_reply": {"type": "string"},
        "cited_ids": {"type": "array", "items": {"type": "string"}},
        "action": {"type": "string"},
    },
    "required": [
        "assignee_candidates",
        "reviewer_candidates",
        "component",
        "duplicate_of",
        "priority_change",
        "suspected_locations",
        "draft_reply",
        "cited_ids",
        "action",
    ],
    "additionalProperties": False,
}


def _next_entity_window_close(
    event: EvalEvent, events: Sequence[EvalEvent], cfg: EngineConfig
) -> datetime | None:
    """Return the close of the first entity window beginning after admission."""

    future = [
        candidate
        for candidate in events
        if candidate.subject == event.subject and candidate.time > event.time
    ]
    if not future:
        return None
    first = future[0].time
    last = first
    count = 1
    if count >= cfg.window.max_events:
        return last
    for candidate in future[1:]:
        due = min(
            last + timedelta(seconds=cfg.window.gap_s),
            first + timedelta(seconds=cfg.window.max_age_s),
        )
        if candidate.time >= due:
            return due
        last = candidate.time
        count += 1
        if count >= cfg.window.max_events:
            return last
    return min(
        last + timedelta(seconds=cfg.window.gap_s),
        first + timedelta(seconds=cfg.window.max_age_s),
    )


def _harm_tasks(
    corpus: CorpusHandle, events: Sequence[EvalEvent]
) -> dict[str, Task]:
    raw = corpus.meta.get("harm_tasks")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        tasks = [item if isinstance(item, Task) else Task.model_validate(item) for item in raw]
    else:
        try:
            from harnext_eval.e4.tasks import build_constructed_tasks

            tasks = build_constructed_tasks(
                events,
                corpus=corpus.name,
                corpus_meta=corpus.meta,
                limit=None,
            )
        except ValueError:
            tasks = []
    return {task.trigger_event_id: task for task in tasks}


def _gold_action_time(task: Task) -> datetime | None:
    values = [
        raw
        for group in ("people", "category", "place", "text")
        if isinstance((payload := task.gold.get(group)), Mapping)
        for raw in payload.get("decision_times", [])
    ]
    parsed = [pd.Timestamp(value).to_pydatetime() for value in values]
    return min(parsed) if parsed else None


def _post_t_action_gold(task: Task) -> dict[str, Any]:
    """Return only post-cutoff human decisions for leakage inspection.

    Constructed tasks also carry the trigger event and state facts required to
    make the decision. Those inputs belong in the envelope and are not the
    future action whose absence the leakage gate must prove.
    """

    people = task.gold.get("people", {})
    category = task.gold.get("category", {})
    text = task.gold.get("text", {})
    return {
        "assignees": people.get("assignees", []) if isinstance(people, Mapping) else [],
        "reviewers": people.get("reviewers", []) if isinstance(people, Mapping) else [],
        "duplicate_of": (
            category.get("duplicate_of", []) if isinstance(category, Mapping) else []
        ),
        "priority_changes": (
            category.get("priority_changes", []) if isinstance(category, Mapping) else []
        ),
        "replies": text.get("replies", []) if isinstance(text, Mapping) else [],
    }


def _run_action(
    task: Task,
    cutoff: datetime,
    *,
    store: StoreHandle,
    events: Sequence[EvalEvent],
    provider: LLMProvider,
    gate_path: Path,
) -> dict[str, Any]:
    timed_task = task.model_copy(update={"T": cutoff})
    snapshot = store.snapshot(cutoff)
    envelope = build_envelope(
        timed_task,
        snapshot,
        "V3",
        {"store_handle": store, "events": events},
    )
    leakage_pass = leakage_gate(
        timed_task,
        store=store,
        T=cutoff,
        all_events=events,
        envelope=envelope.text,
        gold_action=_post_t_action_gold(task),
        gold_action_time=_gold_action_time(task),
        out_csv=gate_path,
    )
    started = time.perf_counter()
    result = provider.complete(
        envelope.prefix,
        "\n\n".join(
            f"## {name}\n{body}" for name, body in envelope.sections.items()
        ),
        json_schema=_ACTION_SCHEMA,
        max_tokens=1_000,
    )
    latency = time.perf_counter() - started
    payload: Any = result.json
    if payload is None:
        payload = json.loads(result.text)
    if not isinstance(payload, Mapping):
        raise ValueError("harm action provider returned a non-object prediction")
    grade = grade_action(
        task.task_id,
        payload,
        task.gold,
        gold_coverage=task.gold_coverage,
    )
    return {
        "quality": grade.value,
        "snapshot_sha": snapshot.sha,
        "leakage_pass": leakage_pass,
        "tokens": int(result.usage.get("input_tokens", envelope.token_count))
        + int(result.usage.get("output_tokens", 0)),
        "dollars": 0.0,
        "latency_s": latency,
    }


def _run_harm_check(
    corpus: CorpusHandle,
    cfg: EngineConfig,
    events: Sequence[EvalEvent],
    scores: pd.DataFrame,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute paired S3/E4 action tasks for every R5 2%-budget promotion."""

    promoted_rows = scores[
        (scores["policy"] == "R5")
        & (scores["budget_pct"] == 2.0)
        & (scores["population"] == "full")
        & scores["admitted"].astype(bool)
    ].drop_duplicates("event_id")
    promoted_ids = promoted_rows["event_id"].astype(str).tolist()
    store = corpus.meta.get("store_handle")
    if not isinstance(store, StoreHandle):
        return pd.DataFrame(
            columns=[
                "event_id",
                "entity",
                "quality_now",
                "quality_window_close",
                "harm_delta",
                "status",
            ]
        ), {
            "store_provided": False,
            "promoted": len(promoted_ids),
            "paired": 0,
            "leakage_pass": False,
            "non_vacuous": False,
            "real_provider": False,
        }
    raw_provider = corpus.meta.get("harm_provider")
    provider = raw_provider if isinstance(raw_provider, LLMProvider) else make_llm(cfg)
    task_by_event = _harm_tasks(corpus, events)
    event_by_id = {event.id: event for event in events}
    rows: list[dict[str, Any]] = []
    gate_path = out_dir / "harm-gate.csv"
    for event_id in promoted_ids:
        event = event_by_id[event_id]
        task = task_by_event.get(event_id)
        close = _next_entity_window_close(event, events, cfg)
        if task is None or close is None:
            rows.append(
                {
                    "event_id": event_id,
                    "entity": event.subject,
                    "status": "missing_task" if task is None else "missing_next_window",
                }
            )
            continue
        try:
            now = _run_action(
                task,
                event.time,
                store=store,
                events=events,
                provider=provider,
                gate_path=gate_path,
            )
            at_close = _run_action(
                task,
                close,
                store=store,
                events=events,
                provider=provider,
                gate_path=gate_path,
            )
            rows.append(
                {
                    "event_id": event_id,
                    "entity": event.subject,
                    "admission_ts": event.time,
                    "window_close_ts": close,
                    "quality_now": now["quality"],
                    "quality_window_close": at_close["quality"],
                    "harm_delta": now["quality"] - at_close["quality"],
                    "snapshot_now": now["snapshot_sha"],
                    "snapshot_window_close": at_close["snapshot_sha"],
                    "leakage_now": now["leakage_pass"],
                    "leakage_window_close": at_close["leakage_pass"],
                    "tokens_now": now["tokens"],
                    "tokens_window_close": at_close["tokens"],
                    "dollars_now": now["dollars"],
                    "dollars_window_close": at_close["dollars"],
                    "latency_now_s": now["latency_s"],
                    "latency_window_close_s": at_close["latency_s"],
                    "provider": str(
                        getattr(provider, "model", getattr(provider, "model_id", type(provider).__name__))
                    ),
                    "status": "paired",
                }
            )
        except (LookupError, ValueError) as exc:
            rows.append(
                {
                    "event_id": event_id,
                    "entity": event.subject,
                    "status": "error",
                    "reason": str(exc),
                }
            )
    harm = pd.DataFrame(rows)
    paired = harm[harm.get("status", pd.Series(dtype=str)) == "paired"]
    leakage_pass = bool(
        len(paired) == len(promoted_ids)
        and not paired.empty
        and paired["leakage_now"].astype(bool).all()
        and paired["leakage_window_close"].astype(bool).all()
    )
    non_vacuous = bool(
        not paired.empty
        and (
            paired["quality_now"].nunique() > 1
            or paired["quality_window_close"].nunique() > 1
            or not np.allclose(paired["harm_delta"], 0.0)
        )
    )
    return harm, {
        "store_provided": True,
        "store_s3": store.layout == "S3",
        "promoted": len(promoted_ids),
        "paired": len(paired),
        "leakage_pass": leakage_pass,
        "non_vacuous": non_vacuous,
        "real_provider": not isinstance(provider, FakeLLM),
    }


def _reference_metric_checks() -> tuple[bool, bool]:
    """Execute the three frozen hand-computed fixtures for each reference metric."""

    vus_cases = (
        ([0, 1, 1, 0, 0, 1, 0], [0.1, 0.9, 0.8, 0.2, 0.0, 0.7, 0.3], 1.0),
        ([0, 1, 0, 0], [0.1, 0.8, 0.9, 0.2], 0.5923495156295323),
        ([0, 1, 1, 0], [1.0, 1.0, 1.0, 1.0], 0.617851130197758),
    )
    vus_ok = all(
        np.isclose(vus_pr(labels, values, max_buffer=2), expected, atol=1e-12)
        for labels, values, expected in vus_cases
    )
    affiliation_cases = (
        ([(2.0, 4.0)], (1.0, 1.0)),
        ([(3.0, 4.0)], (1.0, 11.0 / 12.0)),
        ([(4.0, 5.0)], (0.5, 2.0 / 3.0)),
    )
    from harnext_eval.e1.score import affiliation_pr_from_events

    affiliation_ok = all(
        np.allclose(
            affiliation_pr_from_events(predicted, [(2.0, 4.0)], (0.0, 6.0)),
            expected,
            atol=1e-10,
        )
        for predicted, expected in affiliation_cases
    )
    return vus_ok, affiliation_ok


def _add_gate(
    metrics: dict[str, float],
    details: dict[str, dict[str, Any]],
    required_results: list[bool],
    name: str,
    *,
    passed: bool | None,
    value: Any,
    reason: str,
    required: bool = True,
) -> None:
    """Record one tri-state gate and make its requirement feed final validity."""

    status = "pass" if passed is True else "fail" if passed is False else "not_applicable"
    metrics[f"check.{name}"] = (
        1.0 if passed is True else 0.0 if passed is False else float("nan")
    )
    details[name] = {
        "status": status,
        "passed": passed,
        "required": required,
        "value": value,
        "reason": reason,
    }
    if required:
        required_results.append(passed is True)


class E1Experiment:
    """Offline rolling router evaluation registered as experiment ``e1``."""

    name = "e1"

    def run(
        self, cfg: EngineConfig, corpus: CorpusHandle, out_dir: Path, seed: int
    ) -> ExperimentResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        original_events = sorted(corpus.events(), key=lambda event: (event.time, event.id))
        window = corpus.meta.get("e1_window")
        if window:
            original_events = [event for event in original_events if window[0] <= event.time < window[1]]
        from harnext_eval.corpus.committers import stamp_events

        original_events, committer_counts = stamp_events(original_events)
        if not original_events:
            raise ValueError("E1 requires a non-empty replay")
        exact_labels, situations = _situation_gold(corpus, original_events)
        run_weak_diagnostics = bool(corpus.meta.get("run_weak_label_diagnostics", False))
        excluded_functions = list(corpus.meta.get("e1_exclude_label_functions", []) or [])
        known_functions = {function.name for function in DEFAULT_LABELING_FUNCTIONS}
        unknown = sorted(set(excluded_functions) - known_functions)
        if unknown:
            raise ValueError(f"unknown labeling functions in e1.exclude_label_functions: {unknown}")
        active_functions = [
            function for function in DEFAULT_LABELING_FUNCTIONS if function.name not in excluded_functions
        ]
        label_result = (
            build_labels(
                original_events,
                active_functions,
                observation_end=window[1] if window else original_events[-1].time,
            )
            if exact_labels is None or run_weak_diagnostics
            else _constructed_label_result(exact_labels)
        )
        event_labels = (
            exact_labels if exact_labels is not None else label_result.probabilities.to_dict()
        )
        events = _router_events(original_events)
        months = sorted({_month(event) for event in events})
        if len(months) < 2:
            raise ValueError("rolling month-ahead E1 requires at least two event months")
        evaluated_months = months[2:] if len(months) > 2 else months[1:]
        score_pieces: list[pd.DataFrame] = []
        chronology: list[bool] = []
        _FEATURE_CACHE.clear()
        extractors = {kind: CausalFeatureExtractor(global_only=kind) for kind in (False, True)}
        for kind in extractors:
            _FEATURE_CACHE[kind] = {}
        folded_months: set[str] = set()
        for month_index, month in enumerate(evaluated_months):
            tuning = [event for event in events if _tuning_start(month) <= _month(event) < month]
            evaluation = [event for event in events if _month(event) == month]
            if not tuning or not evaluation:
                continue
            chronology.append(max(event.time for event in tuning) < min(event.time for event in evaluation))
            # Feature state follows the causal stream once, independently of
            # model refits. Cache only the trailing fit window and current month.
            retained_ids = {event.id for event in [*tuning, *evaluation]}
            new_events = [event for event in [*tuning, *evaluation] if _month(event) not in folded_months]
            for kind, extractor in extractors.items():
                cache = _FEATURE_CACHE[kind]
                for event_id in list(cache):
                    if event_id not in retained_ids:
                        del cache[event_id]
                cache.update((event.id, extractor.update(event)) for event in new_events)
            folded_months.update(_month(event) for event in new_events)
            # R5 refits independently at each budget; distribute those jobs
            # too, so one worker does not serialize all four expensive fits.
            selections = [
                (name, budgets)
                for name in _POLICIES
                for budgets in ([(budget,) for budget in _BUDGETS] if name in GUARDED_POLICIES else [_BUDGETS])
            ]
            workers = int(corpus.meta.get("e1_workers", os.environ.get("HARNEXT_E1_WORKERS", "4")))
            if len(events) >= 5_000 and workers > 1:
                # Spawned workers receive this month's inputs once through the
                # initializer and never inherit the parent's heap. Each policy
                # fits its own model and guard state. Bound BLAS threads externally.
                context = multiprocessing.get_context(
                    "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
                )
                month_labels = {event.id: event_labels[event.id] for event in [*tuning, *evaluation] if event.id in event_labels}
                with ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=context,
                    initializer=_init_worker,
                    initargs=(tuning, evaluation, cfg, seed + month_index, month_labels,
                              exact_labels is not None, _FEATURE_CACHE),
                ) as pool:
                    score_pieces.extend(pool.map(_policy_month, selections))
                gc.collect()
            else:
                score_pieces.extend(
                    _policy_month((name, tuning, evaluation, cfg, seed + month_index, event_labels,
                                   exact_labels is not None, budgets))
                    for name, budgets in selections
                )
            print(f"E1 scored {month} ({len(evaluation)} events; fit {len(tuning)})", flush=True)
        _FEATURE_CACHE.clear()
        if not score_pieces:
            raise ValueError("E1 produced no evaluable rolling months")
        scores = pd.concat(score_pieces, ignore_index=True)
        del score_pieces
        for column in ("event_id", "policy", "lane", "source", "subject", "rule", "month", "population", "baseline_key_used"):
            scores[column] = scores[column].astype("category")
        metrics = pd.DataFrame(_metric_rows(scores))

        primary_scores = scores[
            (scores["policy"] == "R5")
            & (scores["budget_pct"] == 2.0)
            & (~scores["rule_flag"].astype(bool))
            & scores["label"].notna()
        ]
        calibration_parts: list[pd.DataFrame] = []
        for month, group in primary_scores.groupby("month", sort=True, observed=True):
            finite_scores = group["score"].replace([np.inf, -np.inf], np.nan).fillna(-1e30)
            curve = decile_rates(finite_scores, group["label"])
            curve["month"] = month
            curve["source"] = "all"
            curve["spearman_rho"] = calibration_spearman(curve)
            curve["lift_over_rules"] = lift_over_rules(
                group["label"], group["admitted"], group["rule_flag"]
            )
            calibration_parts.append(curve)
        calibration = pd.concat(calibration_parts, ignore_index=True) if calibration_parts else pd.DataFrame()

        robustness_rows: list[dict[str, Any]] = []
        for identifiers, group in _population_groups(scores):
            known = group[group["label"].notna()]
            flipped = flip_labels(known["label"], seed=seed)
            swapped = swap_labels(known["label"], seed=seed)
            robustness_rows.append(
                {
                    "month": identifiers[0],
                    "policy": identifiers[1],
                    "budget_pct": identifiers[2],
                    "population": identifiers[3],
                    "recall_label_flip": recall_at_budget(flipped, known["admitted"]),
                    "precision_label_flip": precision_at_budget(flipped, known["admitted"]),
                    # Prevalence-preserving perturbation (review finding 10).
                    "recall_label_swap": recall_at_budget(swapped, known["admitted"]),
                    "precision_label_swap": precision_at_budget(swapped, known["admitted"]),
                }
            )
        robustness = pd.DataFrame(robustness_rows)
        situation_sets: list[tuple[str, pd.DataFrame]] = []
        if not situations.empty:
            situation_sets.append(("constructed", situations))
        elif exact_labels is None:
            # Real corpus: derive timestamped, subject-separated situations from the
            # labels (review finding 8) for the full and rule-negative populations.
            reference = scores[
                (scores["policy"] == "R1")
                & (scores["budget_pct"] == 2.0)
                & (scores["population"] == "full")
                & scores["label"].notna()
            ].drop_duplicates("event_id")
            gap_hours = float(corpus.meta.get("e1_label_situation_gap_hours", 24.0))
            situation_sets.append(("full", label_situations(reference, gap_hours=gap_hours)))
            situation_sets.append(
                (
                    "rule_negative",
                    label_situations(reference[~reference["rule_flag"].astype(bool)], gap_hours=gap_hours),
                )
            )
        delay_parts: list[pd.DataFrame] = []
        for population, situation_frame in situation_sets:
            delays_part, situation_robustness = _situation_metrics(
                scores, situation_frame, seed=seed, population=population
            )
            if not delays_part.empty:
                delay_parts.append(delays_part)
            if not situation_robustness.empty:
                robustness = pd.concat([robustness, situation_robustness], ignore_index=True)
        delays = pd.concat(delay_parts, ignore_index=True) if delay_parts else pd.DataFrame()
        label_situation_frame = next(
            (frame for name, frame in situation_sets if name == "rule_negative"), pd.DataFrame()
        )

        evaluated = scores[
            (scores["policy"] == "R0")
            & (scores["budget_pct"] == 2.0)
            & (scores["population"] == "full")
            & scores["label"].notna()
        ].drop_duplicates("event_id")
        evaluated = evaluated.sort_values(["t", "event_id"])
        relative_tolerance = corpus.meta.get("e1_sanity_relative_tolerance")
        random_check = random_sanity_scorer(
            evaluated["label"],
            budget_pct=2.0,
            seed=seed,
            max_buffer=_VUS_MAX_BUFFER,
            # Matched geometry with the reported monthly VUS (review finding 6).
            timestamps=evaluated["t"] if relative_tolerance is not None else None,
            vus_repeats=20 if len(evaluated) >= 50_000 else None,
        )
        always_check = always_flag_sanity_scorer(
            evaluated["label"], max_buffer=_VUS_MAX_BUFFER
        )

        def _near(value: float, target: float) -> bool:
            if relative_tolerance is None:
                return abs(value - target) <= 0.05
            if not (math.isfinite(value) and math.isfinite(target)) or target <= 0:
                return False
            return abs(value / target - 1.0) <= float(relative_tolerance)

        sanity_reason = (
            f"uniform random value must be within {float(relative_tolerance):.0%} (relative) of prevalence"
            if relative_tolerance is not None
            else "uniform random value must be within 0.05 of prevalence"
        )
        random_vus_applicable = bool(
            len(evaluated) >= 200 and (evaluated["label"] >= 0.5).sum() >= 5
        )
        r1_constructed = scores[
            (scores["policy"] == "R1")
            & (scores["budget_pct"] == 2.0)
            & (scores["population"] == "full")
            & scores["constructed_label"]
        ]
        injected_recall = recall_at_budget(
            r1_constructed["label"], r1_constructed["admitted"]
        ) if not r1_constructed.empty else float("nan")
        smoke_profile = bool(corpus.meta.get("smoke", corpus.name.casefold() == "synthetic"))

        scores_path = out_dir / "scores.parquet"
        metrics_path = out_dir / "metrics.csv"
        diagnostics_path = out_dir / "label_diagnostics.csv"
        robustness_path = out_dir / "robustness.csv"
        delays_path = out_dir / "delays.csv"
        preflight_path = out_dir / "preflight.csv"
        parquet_complete = _write_scores(scores, scores_path)
        metrics.to_csv(metrics_path, index=False)
        calibration.to_csv(out_dir / "calibration.csv", index=False)
        label_result.diagnostics.to_csv(diagnostics_path)
        robustness.to_csv(robustness_path, index=False)
        delays.to_csv(delays_path, index=False)
        if not label_situation_frame.empty:
            label_situation_frame.to_csv(out_dir / "label_situations.csv", index=False)
        preflight = _preflight(corpus, situations)
        preflight.to_csv(preflight_path, index=False)
        attribution_path = out_dir / "attribution.md"
        _write_attribution(scores, attribution_path)
        remediation_frame = _metric_remediation(metrics)
        remediation_path = out_dir / "metric_remediation.csv"
        remediation_frame.to_csv(remediation_path, index=False)
        human_sample_path = _write_human_sanity_sample(scores, original_events, out_dir)
        if exact_labels is None:
            # Votes make the registered post-hoc label-definition sensitivity
            # computable from the run outputs alone (review finding 2).
            votes_path = out_dir / "label_votes.parquet"
            try:
                label_result.votes.astype("int8").to_parquet(votes_path)
            except (ImportError, ValueError):
                label_result.votes.to_csv(out_dir / "label_votes.csv")
        provenance = _text_provenance(original_events)
        harm, harm_evidence = _run_harm_check(corpus, cfg, events, scores, out_dir)
        harm_path = out_dir / "harm.csv"
        harm.to_csv(harm_path, index=False)

        check_metrics: dict[str, float] = {
            "random_precision": random_check.precision,
            "random_vus_pr": random_check.vus_pr,
            "prevalence": random_check.prevalence,
            "always_flag_recall": always_check.recall,
            "declared_outcome_agreement": label_result.declared_outcome_agreement,
            "declared_outcome_comparable": float(label_result.declared_outcome_comparable),
            "label_unknown_count": float(label_result.probabilities.isna().sum()),
        }
        check_details: dict[str, dict[str, Any]] = {}
        required_results: list[bool] = []
        check_metrics["random_precision_sd"] = random_check.precision_sd
        check_metrics["random_recall_sd"] = random_check.recall_sd
        _add_gate(
            check_metrics,
            check_details,
            required_results,
            "random_precision_at_prevalence",
            passed=_near(random_check.precision, random_check.prevalence),
            value=random_check.precision,
            reason=sanity_reason.replace("value", "precision"),
        )
        _add_gate(
            check_metrics,
            check_details,
            required_results,
            "random_vus_at_prevalence",
            passed=(
                _near(random_check.vus_pr, random_check.prevalence)
                if random_vus_applicable
                else None
            ),
            value=random_check.vus_pr if random_vus_applicable else None,
            reason=(
                sanity_reason.replace("value", "same-geometry VUS-PR")
                if random_vus_applicable
                else "requires at least 200 labelled events and five positives"
            ),
        )
        for name, passed, value, reason in (
            (
                "always_flag_recall_one",
                bool(np.isclose(always_check.recall, 1.0)),
                always_check.recall,
                "always-fast recall must equal one",
            ),
            (
                "always_flag_precision_prevalence",
                bool(np.isclose(always_check.precision, always_check.prevalence)),
                always_check.precision,
                "always-fast precision must equal prevalence",
            ),
            (
                "tuning_precedes_evaluation",
                bool(chronology) and all(chronology),
                chronology,
                "every evaluation month uses only prior-month tuning events",
            ),
            (
                "r5_ineligible_never_admitted",
                not bool(
                    scores[
                        scores["policy"].isin(sorted(GUARDED_POLICIES))
                        & ~scores["eligible"]
                        & ~scores["mandatory"]
                        & scores["admitted"]
                    ].shape[0]
                ),
                None,
                "R5 deviation admissions must carry both published guards",
            ),
            (
                "r7_always_fast",
                bool(
                    scores[scores["policy"] == "R7"]["admitted"].astype(bool).all()
                ),
                int(scores[scores["policy"] == "R7"]["admitted"].sum()),
                "R7 is the unbudgeted always-fast cost ceiling",
            ),
        ):
            _add_gate(
                check_metrics,
                check_details,
                required_results,
                name,
                passed=passed,
                value=value,
                reason=reason,
            )

        compared = scores[
            scores["policy"].isin(list(_SHARED_BUDGET_POLICIES))
            & (scores["population"] == "full")
        ]
        capacity_respected = bool(
            (
                compared.groupby(["month", "policy", "budget_pct"], observed=True)["admitted"].sum()
                <= compared.groupby(["month", "policy", "budget_pct"], observed=True)["capacity"].first()
            ).all()
        )
        _add_gate(
            check_metrics,
            check_details,
            required_results,
            "equal_total_budget",
            passed=capacity_respected,
            value=int(compared["rules_over_budget"].max()),
            reason="every R0-R6 arm is capped by the same monthly total budget",
        )
        _add_gate(
            check_metrics,
            check_details,
            required_results,
            "rule_floor_budget_feasible",
            passed=bool(compared["budget_feasible"].all()),
            value=int(compared["rules_over_budget"].max()),
            reason="months whose rule floor exceeds capacity are invalid, never over-admitted",
        )

        label_required = exact_labels is None
        support_min = int(corpus.meta.get("e1_label_positive_support_min", 0) or 0)
        for function, row in label_result.diagnostics.iterrows():
            accuracy = float(row["accuracy"])
            coverage = float(row["coverage"])
            if support_min > 0:
                support = int(row["positive_votes"])
                _add_gate(
                    check_metrics,
                    check_details,
                    required_results,
                    f"lf.{function}.positive_support",
                    passed=(support >= support_min if label_required else None),
                    value=support,
                    reason=(
                        f"LF must cast at least {support_min} positive votes to be informative (review finding 3)"
                        if label_required
                        else "constructed exact gold does not use weak-label diagnostics"
                    ),
                    required=label_required,
                )
            _add_gate(
                check_metrics,
                check_details,
                required_results,
                f"lf.{function}.accuracy",
                passed=(math.isfinite(accuracy) and accuracy >= 0.6 if label_required else None),
                value=accuracy if math.isfinite(accuracy) else None,
                reason=(
                    "estimated LF accuracy must be at least 0.6"
                    if label_required
                    else "constructed exact gold does not use weak-label diagnostics"
                ),
                required=label_required,
            )
            _add_gate(
                check_metrics,
                check_details,
                required_results,
                f"lf.{function}.coverage",
                passed=(coverage >= 0.01 if label_required else None),
                value=coverage,
                reason=(
                    "LF coverage, including zero coverage, must be at least 1%"
                    if label_required
                    else "constructed exact gold does not use weak-label diagnostics"
                ),
                required=label_required,
            )
        _add_gate(
            check_metrics,
            check_details,
            required_results,
            "declared_outcome_agreement_reported",
            passed=(
                math.isfinite(label_result.declared_outcome_agreement)
                and label_result.declared_outcome_comparable > 0
                if label_required
                else None
            ),
            value=label_result.declared_outcome_agreement,
            reason="declared-priority/outcome LF agreement must be measurable",
            required=label_required,
        )
        _add_gate(
            check_metrics,
            check_details,
            required_results,
            "injected_non_trivial",
            passed=(
                bool(np.isfinite(injected_recall) and injected_recall <= 0.9)
                if exact_labels is not None
                else None
            ),
            value=injected_recall,
            reason="Corpus-S R1 recall@2% must not exceed 0.9",
            required=exact_labels is not None,
        )
        vus_reference, affiliation_reference = _reference_metric_checks()
        _add_gate(
            check_metrics,
            check_details,
            required_results,
            "vus_three_reference_series",
            passed=vus_reference,
            value=3,
            reason="three Paparrizos-reference numeric fixtures execute in-process",
        )
        _add_gate(
            check_metrics,
            check_details,
            required_results,
            "affiliation_three_reference_series",
            passed=affiliation_reference,
            value=3,
            reason="three Huet-reference numeric fixtures execute in-process",
        )

        prereg_ok = bool(
            corpus.meta.get("prereg_ref")
            and corpus.meta.get("prereg_predates_evaluation") is True
        )
        _add_gate(
            check_metrics,
            check_details,
            required_results,
            "prereg_chronology",
            passed=prereg_ok if not smoke_profile else None,
            value=corpus.meta.get("prereg_ref"),
            reason=(
                str(corpus.meta.get("prereg_reason", "verified preregistration must predate evaluation"))
                if not smoke_profile
                else "offline smoke has no evidentiary preregistration chronology"
            ),
        )
        human = corpus.meta.get("human_sanity")
        human_ok = bool(
            isinstance(human, Mapping)
            and int(human.get("items", 0)) >= 100
            and int(human.get("annotators", 0)) >= 2
            and math.isfinite(float(human.get("kappa", float("nan"))))
        )
        _add_gate(
            check_metrics,
            check_details,
            required_results,
            "human_sanity_100_two_annotators",
            passed=human_ok if human is not None else None,
            value=human,
            reason="100 top rule-negative items require two annotators and reported kappa",
        )
        remediation = corpus.meta.get("metric_remediation")
        if remediation is None and not remediation_frame.empty:
            remediation = {
                "path": str(remediation_path),
                "kept": remediation_frame.loc[remediation_frame["decision"] == "kept", "metric"].tolist(),
                "dropped": remediation_frame.loc[remediation_frame["decision"] == "dropped", "metric"].tolist(),
                "criterion": "design (R10) minus random (R0), paired over months at 2%/rule-negative, |mean| > 2 SE",
            }
        _add_gate(
            check_metrics,
            check_details,
            required_results,
            "metric_remediation_recorded",
            passed=bool(remediation) if remediation is not None else None,
            value=remediation,
            reason="floor-near-design metrics require an explicit kept/dropped action record",
        )
        _add_gate(
            check_metrics,
            check_details,
            required_results,
            "historical_text_provenance",
            passed=None,
            value=provenance,
            reason=(
                "rule/feature/label text is export-time snapshot text (review finding 7); "
                "components are replayed as-of-event, text edits cannot be; declared limitation"
            ),
            required=False,
        )
        _add_gate(
            check_metrics,
            check_details,
            required_results,
            "human_sanity_sample_written",
            passed=None,
            value=str(human_sample_path),
            reason="blind annotator sheet for the 100-item human sanity check; annotate then rerun with human_sanity meta",
            required=False,
        )
        corpus_ok = bool((preflight["status"] == "run").all())
        _add_gate(
            check_metrics,
            check_details,
            required_results,
            "corpus_preflight",
            passed=corpus_ok if not smoke_profile else None,
            value=preflight.to_dict(orient="records"),
            reason=(
                "every declared corpus profile requirement must run"
                if not smoke_profile
                else "synthetic smoke is explicitly non-evidentiary"
            ),
        )
        _add_gate(
            check_metrics,
            check_details,
            required_results,
            "real_parquet",
            passed=parquet_complete,
            value=parquet_complete,
            reason="scores.parquet must be a real Parquet artifact",
        )
        for name, key, reason in (
            ("harm_store_s3", "store_s3", "harm execution requires an S3 store handle"),
            (
                "harm_minimum_five_promotions",
                "promoted",
                "smoke harm requires at least five R5 2%-budget promotions",
            ),
            ("harm_paired_coverage", "paired", "every R5 promotion requires now/close scores"),
            ("harm_leakage", "leakage_pass", "both action envelopes must pass §4.2"),
            ("harm_non_vacuous", "non_vacuous", "paired action scores must vary"),
            ("harm_real_provider", "real_provider", "evidentiary harm requires a pinned non-fake provider"),
        ):
            if name == "harm_minimum_five_promotions":
                passed = int(harm_evidence.get("promoted", 0)) >= 5
                value = int(harm_evidence.get("promoted", 0))
            elif name == "harm_paired_coverage":
                passed = bool(
                    harm_evidence.get("promoted", 0) > 0
                    and harm_evidence.get("paired") == harm_evidence.get("promoted")
                )
                value = {
                    "paired": harm_evidence.get("paired", 0),
                    "promoted": harm_evidence.get("promoted", 0),
                }
            else:
                passed = bool(harm_evidence.get(key, False))
                value = harm_evidence.get(key)
            applicable = bool(harm_evidence.get("store_provided"))
            required = True
            detail_reason = reason
            if corpus.meta.get("e1_only"):
                applicable = False
                required = False
                detail_reason = "no real action provider / no S3 store in this profile"
            if name == "harm_real_provider" and smoke_profile:
                applicable = False
                required = False
                detail_reason = (
                    "offline smoke uses FakeLLM; real-provider harm is evidentiary-only"
                )
            _add_gate(
                check_metrics,
                check_details,
                required_results,
                name,
                passed=passed if applicable else None,
                value=value,
                reason=detail_reason,
                required=required,
            )

        valid = all(required_results)
        failed_required = [
            f"{name} ({detail.get('status')}): {detail.get('reason')}"
            for name, detail in check_details.items()
            if detail.get("required") and detail.get("passed") is not True
        ]
        check_metrics["check.valid"] = float(valid)
        check_details["excluded_label_functions"] = {
            "passed": None,
            "required": False,
            "value": excluded_functions,
            "reason": "labeling functions dropped by registered amendment (config e1.exclude_label_functions)",
        }
        check_details["valid"] = {
            "status": "pass" if valid else "fail",
            "passed": valid,
            "required": True,
            "value": valid,
            "reason": (
                "all required E1 validity items passed"
                if valid
                else f"failed required E1 validity items: {failed_required}"
            ),
        }
        validity = pd.DataFrame(
            [{"gate": name, **detail} for name, detail in check_details.items()]
        )
        validity.to_csv(out_dir / "validity.csv", index=False)
        chart_paths = _write_charts(calibration, metrics, out_dir)
        primary = _paired_primary(scores, seed)
        primary["prereg_hash"] = corpus.meta.get("prereg_hash")
        rule_reference = scores[
            (scores["policy"] == "R1") & (scores["budget_pct"] == 2.0) & (scores["population"] == "full")
        ]
        primary["rule_hit_rate"] = (
            float(rule_reference["rule_flag"].astype(bool).mean()) if not rule_reference.empty else float("nan")
        )
        primary["rule_hits"] = int(rule_reference["rule_flag"].astype(bool).sum())
        primary["metric_remediation"] = remediation
        primary["committer_matches"] = committer_counts
        primary["fit_window_months"] = 12
        primary["window"] = [str(value) for value in window] if window else None
        primary["valid"] = valid
        primary["evidence_status"] = "valid" if valid else "non-evidentiary"

        artifacts = [
            metrics_path,
            diagnostics_path,
            robustness_path,
            delays_path,
            preflight_path,
            attribution_path,
            harm_path,
            remediation_path,
            human_sample_path,
            out_dir / "validity.csv",
            *chart_paths,
        ]
        if parquet_complete:
            artifacts.append(scores_path)
        if not smoke_profile and not valid and not corpus.meta.get("e1_only"):
            failed = [
                name
                for name, detail in check_details.items()
                if detail.get("required") and detail.get("passed") is not True
            ]
            raise ValueError(f"E1 validity gates failed: {', '.join(failed)}")
        # Small callers retain the historical in-memory population view. Large
        # runs expose only compact scores and never inline them into results.json.
        if len(original_events) < 5_000:
            negative = scores[~scores["rule_flag"].astype(bool)].copy()
            negative["population"] = "rule_negative"
            scores = pd.concat([scores, negative], ignore_index=True)
        return ExperimentResult(
            name=self.name,
            metrics=check_metrics,
            tables={
                "scores": scores,
                "metrics": metrics,
                "calibration": calibration,
                "label_diagnostics": label_result.diagnostics.reset_index(),
                "robustness": robustness,
                "delays": delays,
                "preflight": preflight,
                "harm": harm,
                "validity": validity,
                "metric_remediation": remediation_frame,
            },
            artifacts=artifacts,
            primary=primary,
            check_details=check_details,
        )

    def chart(self, result: ExperimentResult, out_dir: Path) -> list[Path]:
        """Write the named E1 PNGs using the already computed tables."""

        out_dir.mkdir(parents=True, exist_ok=True)
        return _write_charts(result.tables["calibration"], result.tables["metrics"], out_dir)


register_experiment(E1Experiment())
