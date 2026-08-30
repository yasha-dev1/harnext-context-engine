"""Rolling month-ahead E1 experiment from docs/evaluation-spec.md §7 E1."""

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportGeneralTypeIssues=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportReturnType=false

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from harnext_eval.config import EngineConfig
from harnext_eval.corpus import CorpusHandle
from harnext_eval.e1.calibration import calibration_spearman, decile_rates, lift_over_rules
from harnext_eval.e1.labels import build_labels
from harnext_eval.e1.policies import budgeted_decisions, make_policy
from harnext_eval.e1.score import (
    always_flag_sanity_scorer,
    delay_summary,
    detection_delays,
    flip_labels,
    jitter_onsets,
    precision_at_budget,
    random_sanity_scorer,
    recall_at_budget,
    timestamped_affiliation_precision_recall,
    vus_pr,
)
from harnext_eval.registry import ExperimentResult, register_experiment
from harnext_eval.stats.stats import paired_difference_bca
from harnext_eval.types import EvalEvent, RouterRecord

_BUDGETS = (1.0, 2.0, 5.0, 10.0)
_POLICIES = tuple(f"R{index}" for index in range(8))
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

    if name == "R1":
        eligible = scored["rule_flag"].astype(bool).to_numpy()
    elif name == "R5":
        eligible = np.asarray(
            [
                bool(features.get("eligible", False)) or bool(rule)
                for features, rule in zip(scored["features_fired"], scored["rule_flag"], strict=True)
            ]
        )
    else:
        eligible = np.ones(len(scored), dtype=bool)
    mandatory = scored["rule_flag"].astype(bool).to_numpy() if name in {"R1", "R5"} else None
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
            labels = known["label"].to_numpy(dtype=float)
            admitted = known["admitted"].to_numpy(dtype=bool)
            scores = known["score"].replace([np.inf, -np.inf], np.nan).fillna(-1e30).to_numpy()
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
                    "vus_pr": vus_pr(labels, scores, max_buffer=5) if len(labels) else float("nan"),
                    "decision_latency_ms": float(group["decision_latency_ms"].mean()),
                    "tokens": int(group["routing_tokens"].sum()),
                    "dollars": float(group["routing_dollars"].sum()),
                    "unused_capacity": int(group["unused_capacity"].iloc[0]),
                    "rules_over_budget": int(group["rules_over_budget"].iloc[0]),
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
    relevant = scores[(scores["policy"] == "R5") & (scores["budget_pct"] == 2.0) & (scores["population"] == "full")]
    admissions = relevant.rename(columns={"subject": "entity", "t": "time"})
    delays = detection_delays(situations, admissions)
    summary = delay_summary(delays["delay_s"])
    affiliation_p, affiliation_r = timestamped_affiliation_precision_recall(situations, admissions)
    jittered = situations.copy()
    jittered["onset"] = jitter_onsets(jittered["onset"], seed=seed)
    jittered_delays = detection_delays(jittered, admissions)
    jittered_summary = delay_summary(jittered_delays["delay_s"])
    robustness = pd.DataFrame(
        [
            {
                "policy": "R5",
                "budget_pct": 2.0,
                "affiliation_precision": affiliation_p,
                "affiliation_recall": affiliation_r,
                "delay_p50_s": summary["p50_s"],
                "delay_p95_s": summary["p95_s"],
                "detected_rate": summary["detected_rate"],
                "jitter_delay_p50_s": jittered_summary["p50_s"],
                "jitter_delay_p95_s": jittered_summary["p95_s"],
                "jitter_detected_rate": jittered_summary["detected_rate"],
            }
        ]
    )
    return delays, robustness


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
        label_result = build_labels(original_events, observation_end=original_events[-1].time)
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
        random_check = random_sanity_scorer(evaluated["label"], budget_pct=2.0, seed=seed)
        always_check = always_flag_sanity_scorer(evaluated["label"])
        r1_constructed = scores[
            (scores["policy"] == "R1")
            & (scores["budget_pct"] == 2.0)
            & (scores["population"] == "full")
            & scores["constructed_label"]
        ]
        injected_recall = recall_at_budget(
            r1_constructed["label"], r1_constructed["admitted"]
        ) if not r1_constructed.empty else float("nan")
        diagnostics_applicable = label_result.diagnostics[
            label_result.diagnostics["coverage"] > 0
        ]
        check_metrics = {
            "check.random_precision_at_prevalence": float(abs(random_check.precision - random_check.prevalence) <= 0.05),
            "check.random_vus_at_prevalence": float(abs(random_check.vus_pr - random_check.prevalence) <= 0.05),
            "check.always_flag_recall_one": float(np.isclose(always_check.recall, 1.0)),
            "check.always_flag_precision_prevalence": float(np.isclose(always_check.precision, always_check.prevalence)),
            "check.injected_non_trivial": float(exact_labels is None or (np.isfinite(injected_recall) and injected_recall <= 0.9)),
            "check.label_accuracy_min_0_6": float(exact_labels is not None or (not diagnostics_applicable.empty and (diagnostics_applicable["accuracy"] >= 0.6).all())),
            "check.label_coverage_min_0_01": float(exact_labels is not None or (not diagnostics_applicable.empty and (diagnostics_applicable["coverage"] >= 0.01).all())),
            "check.tuning_precedes_evaluation": float(bool(chronology) and all(chronology)),
            "check.r5_rules_floor_preserved": float(not bool(scores[(scores["policy"] == "R5") & scores["rule_flag"] & ~scores["admitted"]].shape[0])),
            "check.r5_ineligible_never_admitted": float(not bool(scores[(scores["policy"] == "R5") & ~scores["eligible"] & ~scores["mandatory"] & scores["admitted"]].shape[0])),
            "random_precision": random_check.precision,
            "prevalence": random_check.prevalence,
            "always_flag_recall": always_check.recall,
            "declared_outcome_agreement": label_result.declared_outcome_agreement,
            "declared_outcome_comparable": float(label_result.declared_outcome_comparable),
            "label_unknown_count": float(label_result.probabilities.isna().sum()),
        }
        smoke_profile = bool(corpus.meta.get("smoke", corpus.name.casefold() == "synthetic"))
        check_metrics["check.prereg_present"] = float(
            smoke_profile or bool(corpus.meta.get("prereg_ref"))
        )
        required_gates = [
            "check.random_precision_at_prevalence",
            "check.random_vus_at_prevalence",
            "check.always_flag_recall_one",
            "check.always_flag_precision_prevalence",
            "check.injected_non_trivial",
            "check.label_accuracy_min_0_6",
            "check.label_coverage_min_0_01",
            "check.tuning_precedes_evaluation",
            "check.r5_rules_floor_preserved",
            "check.r5_ineligible_never_admitted",
            "check.prereg_present",
        ]
        check_metrics["check.valid"] = float(
            all(bool(check_metrics[name]) for name in required_gates)
        )
        if not smoke_profile and not check_metrics["check.valid"]:
            failed = [name for name in required_gates if not check_metrics[name]]
            raise ValueError(f"E1 validity gates failed: {', '.join(failed)}")

        scores_path = out_dir / "scores.parquet"
        metrics_path = out_dir / "metrics.csv"
        diagnostics_path = out_dir / "label_diagnostics.csv"
        robustness_path = out_dir / "robustness.csv"
        delays_path = out_dir / "delays.csv"
        preflight_path = out_dir / "preflight.csv"
        parquet_complete = _write_scores(scores, scores_path)
        check_metrics["check.real_parquet"] = float(parquet_complete)
        metrics.to_csv(metrics_path, index=False)
        calibration.to_csv(out_dir / "calibration.csv", index=False)
        label_result.diagnostics.to_csv(diagnostics_path)
        robustness.to_csv(robustness_path, index=False)
        delays.to_csv(delays_path, index=False)
        preflight = _preflight(corpus, situations)
        preflight.to_csv(preflight_path, index=False)
        attribution_path = out_dir / "attribution.md"
        _write_attribution(scores, attribution_path)
        harm_rows = corpus.meta.get("harm_results", [])
        harm = pd.DataFrame(harm_rows)
        harm_path = out_dir / "harm.csv"
        harm.to_csv(harm_path, index=False)
        check_metrics["check.harm_supported"] = float(bool(harm_rows))
        chart_paths = _write_charts(calibration, metrics, out_dir)
        primary = _paired_primary(scores, seed)

        artifacts = [
            metrics_path,
            diagnostics_path,
            robustness_path,
            delays_path,
            preflight_path,
            attribution_path,
            *chart_paths,
        ]
        if parquet_complete:
            artifacts.append(scores_path)
        if harm_rows:
            artifacts.append(harm_path)
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
            },
            artifacts=artifacts,
            primary=primary,
        )

    def chart(self, result: ExperimentResult, out_dir: Path) -> list[Path]:
        """Write the named E1 PNGs using the already computed tables."""

        out_dir.mkdir(parents=True, exist_ok=True)
        return _write_charts(result.tables["calibration"], result.tables["metrics"], out_dir)


register_experiment(E1Experiment())
