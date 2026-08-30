"""Rolling month-ahead E1 experiment from docs/evaluation-spec.md §7 E1."""

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportGeneralTypeIssues=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportReturnType=false

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
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
from harnext_eval.e1.labels import (
    ABSTAIN,
    DEFAULT_LABELING_FUNCTIONS,
    LabelModelResult,
    build_labels,
)
from harnext_eval.e1.policies import budgeted_decisions, make_policy
from harnext_eval.e1.score import (
    affiliation_precision_recall,
    always_flag_sanity_scorer,
    delay_summary,
    detection_delays,
    flip_labels,
    jitter_onsets,
    nab_low_fn_score,
    precision_at_budget,
    random_sanity_scorer,
    recall_at_budget,
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
_POLICIES = tuple(f"R{index}" for index in range(8))
_VUS_MAX_BUFFER = 5
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
    policy = make_policy(name, cfg.router, seed=seed, budget_pct=budget).fit(fit_events)
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
    policy = make_policy(name, cfg.router, seed=seed, budget_pct=budget).fit(tuning)
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
            }
        )
    elif name == "R1":
        eligible = scored["rule_flag"].astype(bool).to_numpy()
    elif name == "R5":
        eligible = np.asarray(
            [
                bool(features.get("eligible", False)) or bool(rule)
                for features, rule in zip(scored["features_fired"], scored["rule_flag"], strict=True)
            ]
        )
    if name != "R7":
        mandatory = (
            scored["rule_flag"].astype(bool).to_numpy()
            if name in {"R1", "R5"}
            else None
        )
        decisions = budgeted_decisions(
            scored["event_id"].tolist(),
            scored["score"].tolist(),
            budget_pct=budget,
            tuning_scores=tuning_scores,
            eligible=eligible,
            mandatory=mandatory,
        ).drop(columns="score")
    admitted = scored.merge(decisions, on="event_id", how="left", validate="one_to_one")
    admitted["lane"] = np.where(admitted["admitted"], "fast", "batch")
    full = admitted.assign(population="full")
    rule_negative = admitted[~admitted["rule_flag"].astype(bool)].copy()
    rule_negative["population"] = "rule_negative"
    return pd.concat([full, rule_negative], ignore_index=True)


def _metric_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = ["month", "policy", "budget_pct", "population"]
    for identifiers, month_group in frame.groupby(keys, sort=True):
        month, policy, budget, population = identifiers
        source_groups = [("all", month_group), *month_group.groupby("source", sort=True)]
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
                    "affiliation_precision": affiliation_p,
                    "affiliation_recall": affiliation_r,
                    "nab_low_fn": nab_low_fn_score(labels, admitted),
                    "decision_latency_ms": float(group["decision_latency_ms"].mean()),
                    "tokens": int(group["routing_tokens"].sum()),
                    "dollars": float(group["routing_dollars"].sum()),
                    "unused_capacity": int(group["unused_capacity"].iloc[0]),
                    "rules_over_budget": int(group["rules_over_budget"].iloc[0]),
                    "budget_feasible": bool(group["budget_feasible"].iloc[0]),
                }
            )
    return rows


def _paired_primary(scores: pd.DataFrame, seed: int) -> dict[str, Any]:
    subset = scores[
        (scores["budget_pct"] == 2.0)
        & (scores["population"] == "rule_negative")
        & scores["label"].notna()
        & (scores["label"] >= 0.5)
    ]
    pivot = subset.pivot_table(
        index=["event_id", "subject"], columns="policy", values="admitted", aggfunc="first"
    ).reset_index()
    by_policy = {
        str(name): float(pivot[name].mean()) for name in _POLICIES if name in pivot.columns
    }
    result: dict[str, Any] = {
        "metric": "recall_at_2pct_rule_negative",
        "by_policy": by_policy,
    }
    for baseline in ("R1", "R2"):
        key = f"r5_minus_{baseline.casefold()}"
        if {"R5", baseline} <= set(pivot.columns) and pivot["subject"].nunique() >= 2:
            contrast = paired_difference_bca(
                pivot["R5"].astype(float),
                pivot[baseline].astype(float),
                pivot["subject"],
                n_resamples=10_000,
                random_state=seed,
            )
            result[key] = contrast.effect
            result[f"{key}_ci_low"] = contrast.ci_low
            result[f"{key}_ci_high"] = contrast.ci_high
            result[f"{key}_n_entities"] = contrast.n_clusters
        else:
            result[key] = float("nan")
            result[f"{key}_ci_low"] = float("nan")
            result[f"{key}_ci_high"] = float("nan")
            result[f"{key}_n_entities"] = int(pivot["subject"].nunique()) if not pivot.empty else 0
    return result


def _situation_metrics(
    scores: pd.DataFrame, situations: pd.DataFrame, *, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if situations.empty:
        return pd.DataFrame(), pd.DataFrame()
    delay_parts: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    full = scores[scores["population"] == "full"]
    for (policy, budget), relevant in full.groupby(["policy", "budget_pct"], sort=True):
        relevant = relevant.drop_duplicates("event_id")
        condition_situations = situations[
            situations["event_id"].isin(relevant["event_id"])
        ].copy()
        if condition_situations.empty:
            continue
        admissions = relevant.rename(columns={"subject": "entity", "t": "time"})
        delays = detection_delays(condition_situations, admissions)
        delays["policy"] = policy
        delays["budget_pct"] = budget
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


def _write_scores(frame: pd.DataFrame, path: Path) -> bool:
    serializable = frame.copy()
    serializable["features_fired"] = serializable["features_fired"].map(
        lambda value: json.dumps(value, sort_keys=True, default=str)
    )
    try:
        serializable.to_parquet(path, index=False)
    except ImportError:
        payload = serializable.to_json(orient="table", date_format="iso")
        assert payload is not None
        path.write_text(payload, encoding="utf-8")
        path.with_suffix(".parquet.format.json").write_text(
            json.dumps({"format": "pandas-table-json", "reason": "no parquet engine"}) + "\n",
            encoding="utf-8",
        )
        return False
    return True


def _write_attribution(scores: pd.DataFrame, path: Path) -> None:
    selected = scores[
        (scores["policy"] == "R5")
        & (scores["budget_pct"] == 2.0)
        & (scores["population"] == "full")
        & scores["admitted"]
        & (scores["label"] >= 0.5)
    ].drop_duplicates("event_id")
    totals: dict[str, float] = {}
    for features in selected["features_fired"]:
        for name, value in features.get("hbos_terms", {}).items():
            totals[name] = totals.get(name, 0.0) + float(value)
    lines = ["# E1 feature attribution", "", "## HBOS contributions on true positives", ""]
    lines.extend(f"- {name}: {value:.6f}" for name, value in sorted(totals.items(), key=lambda item: -item[1]))
    lines.extend(["", "## Audited cases", ""])
    for _, row in selected.head(10).iterrows():
        terms = row["features_fired"].get("hbos_terms", {})
        leading = sorted(terms.items(), key=lambda item: -float(item[1]))[:3]
        lines.append(f"- `{row['event_id']}` ({row['subject']}): {leading}")
    if selected.empty:
        lines.append("- No true-positive R5 admissions in this smoke replay.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    for policy, group in selected.groupby("policy", sort=True):
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
        gold_action=task.gold,
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
        if not original_events:
            raise ValueError("E1 requires a non-empty replay")
        exact_labels, situations = _situation_gold(corpus, original_events)
        run_weak_diagnostics = bool(corpus.meta.get("run_weak_label_diagnostics", False))
        label_result = (
            build_labels(original_events, observation_end=original_events[-1].time)
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
        for month_index, month in enumerate(evaluated_months):
            tuning = [event for event in events if _month(event) < month]
            evaluation = [event for event in events if _month(event) == month]
            if not tuning or not evaluation:
                continue
            chronology.append(max(event.time for event in tuning) < min(event.time for event in evaluation))
            for policy_name in _POLICIES:
                budget_scores: dict[float, tuple[pd.DataFrame, list[float]]] = {}
                for budget in (_BUDGETS if policy_name == "R5" else (_BUDGETS[0],)):
                    budget_scores[budget] = _score_month(
                        policy_name, tuning, evaluation, cfg, seed + month_index, budget
                    )
                for budget in _BUDGETS:
                    scored, tuning_scores = budget_scores.get(budget, budget_scores[_BUDGETS[0]])
                    scored = scored.copy()
                    scored["budget_pct"] = budget
                    scored["p_urgent"] = scored["event_id"].map(event_labels).astype(float)
                    scored["label"] = scored["p_urgent"]
                    scored["constructed_label"] = exact_labels is not None
                    score_pieces.append(
                        _admit_month(
                            scored,
                            name=policy_name,
                            budget=budget,
                            tuning_scores=tuning_scores,
                        )
                    )
        if not score_pieces:
            raise ValueError("E1 produced no evaluable rolling months")
        scores = pd.concat(score_pieces, ignore_index=True)
        metrics = pd.DataFrame(_metric_rows(scores))

        primary_scores = scores[
            (scores["policy"] == "R5")
            & (scores["budget_pct"] == 2.0)
            & (scores["population"] == "rule_negative")
            & scores["label"].notna()
        ]
        calibration_parts: list[pd.DataFrame] = []
        for month, group in primary_scores.groupby("month", sort=True):
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
        for identifiers, group in scores.groupby(
            ["month", "policy", "budget_pct", "population"], sort=True
        ):
            known = group[group["label"].notna()]
            flipped = flip_labels(known["label"], seed=seed)
            robustness_rows.append(
                {
                    "month": identifiers[0],
                    "policy": identifiers[1],
                    "budget_pct": identifiers[2],
                    "population": identifiers[3],
                    "recall_label_flip": recall_at_budget(flipped, known["admitted"]),
                    "precision_label_flip": precision_at_budget(flipped, known["admitted"]),
                }
            )
        robustness = pd.DataFrame(robustness_rows)
        delays, situation_robustness = _situation_metrics(scores, situations, seed=seed)
        if not situation_robustness.empty:
            robustness = pd.concat([robustness, situation_robustness], ignore_index=True)

        evaluated = scores[
            (scores["policy"] == "R0")
            & (scores["budget_pct"] == 2.0)
            & (scores["population"] == "full")
            & scores["label"].notna()
        ].drop_duplicates("event_id")
        random_check = random_sanity_scorer(
            evaluated["label"],
            budget_pct=2.0,
            seed=seed,
            max_buffer=_VUS_MAX_BUFFER,
        )
        always_check = always_flag_sanity_scorer(
            evaluated["label"], max_buffer=_VUS_MAX_BUFFER
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
        preflight = _preflight(corpus, situations)
        preflight.to_csv(preflight_path, index=False)
        attribution_path = out_dir / "attribution.md"
        _write_attribution(scores, attribution_path)
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
        _add_gate(
            check_metrics,
            check_details,
            required_results,
            "random_precision_at_prevalence",
            passed=abs(random_check.precision - random_check.prevalence) <= 0.05,
            value=random_check.precision,
            reason="uniform random precision must be within 0.05 of prevalence",
        )
        _add_gate(
            check_metrics,
            check_details,
            required_results,
            "random_vus_at_prevalence",
            passed=(
                abs(random_check.vus_pr - random_check.prevalence) <= 0.05
                if random_vus_applicable
                else None
            ),
            value=random_check.vus_pr if random_vus_applicable else None,
            reason=(
                "same-buffer random VUS-PR must be within 0.05 of prevalence"
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
                        (scores["policy"] == "R5")
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
            scores["policy"].isin([f"R{index}" for index in range(7)])
            & (scores["population"] == "full")
        ]
        capacity_respected = bool(
            (
                compared.groupby(["month", "policy", "budget_pct"])["admitted"].sum()
                <= compared.groupby(["month", "policy", "budget_pct"])["capacity"].first()
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
        for function, row in label_result.diagnostics.iterrows():
            accuracy = float(row["accuracy"])
            coverage = float(row["coverage"])
            _add_gate(
                check_metrics,
                check_details,
                required_results,
                f"lf.{function}.accuracy",
                passed=(accuracy >= 0.6 if label_required and math.isfinite(accuracy) else None),
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
                "verified preregistration must predate evaluation"
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
        _add_gate(
            check_metrics,
            check_details,
            required_results,
            "metric_remediation_recorded",
            passed=bool(remediation) if remediation is not None else None,
            value=remediation,
            reason="floor-near-R5 metrics require an explicit kept/dropped action record",
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
            ("harm_paired_coverage", "paired", "every R5 promotion requires now/close scores"),
            ("harm_leakage", "leakage_pass", "both action envelopes must pass §4.2"),
            ("harm_non_vacuous", "non_vacuous", "paired action scores must vary"),
            ("harm_real_provider", "real_provider", "evidentiary harm requires a pinned non-fake provider"),
        ):
            if name == "harm_paired_coverage":
                passed = bool(
                    harm_evidence.get("promoted", 0) > 0
                    and harm_evidence.get("paired") == harm_evidence.get("promoted")
                )
                value: Any = {
                    "paired": harm_evidence.get("paired", 0),
                    "promoted": harm_evidence.get("promoted", 0),
                }
            else:
                passed = bool(harm_evidence.get(key, False))
                value = harm_evidence.get(key)
            _add_gate(
                check_metrics,
                check_details,
                required_results,
                name,
                passed=passed if harm_evidence.get("store_provided") else None,
                value=value,
                reason=reason,
            )

        valid = all(required_results)
        check_metrics["check.valid"] = float(valid)
        check_details["valid"] = {
            "status": "pass" if valid else "fail",
            "passed": valid,
            "required": True,
            "value": valid,
            "reason": "single conjunction of every required E1 validity item",
        }
        validity = pd.DataFrame(
            [{"gate": name, **detail} for name, detail in check_details.items()]
        )
        validity.to_csv(out_dir / "validity.csv", index=False)
        chart_paths = _write_charts(calibration, metrics, out_dir)
        primary = _paired_primary(scores, seed)
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
            out_dir / "validity.csv",
            *chart_paths,
        ]
        if parquet_complete:
            artifacts.append(scores_path)
        if not smoke_profile and not valid:
            failed = [
                name
                for name, detail in check_details.items()
                if detail.get("required") and detail.get("passed") is not True
            ]
            raise ValueError(f"E1 validity gates failed: {', '.join(failed)}")
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
