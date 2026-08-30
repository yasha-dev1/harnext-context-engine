"""Rolling month-ahead E1 experiment from docs/evaluation-spec.md §7 E1."""

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportGeneralTypeIssues=false, reportReturnType=false

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from harnext_eval.config import EngineConfig
from harnext_eval.corpus import CorpusHandle
from harnext_eval.e1.calibration import calibration_spearman, decile_rates, lift_over_rules
from harnext_eval.e1.labels import build_labels
from harnext_eval.e1.policies import budgeted_decisions, make_policy
from harnext_eval.e1.score import (
    affiliation_precision_recall,
    always_flag_sanity_scorer,
    flip_labels,
    nab_low_fn_score,
    precision_at_budget,
    random_sanity_scorer,
    recall_at_budget,
    vus_pr,
)
from harnext_eval.registry import ExperimentResult, register_experiment
from harnext_eval.types import EvalEvent, RouterRecord

_BUDGETS = (1.0, 2.0, 5.0, 10.0)
_POLICIES = tuple(f"R{index}" for index in range(8))


def _month(event: EvalEvent) -> str:
    return f"{event.time.year:04d}-{event.time.month:02d}"


def _source(event: EvalEvent) -> str:
    return event.source.split(":", 1)[0]


def _explicit_label(event: EvalEvent) -> float | None:
    data = event.data or {}
    for key in ("is_urgent", "urgent", "injected_positive", "situation_label"):
        if key in data:
            value = data[key]
            if isinstance(value, str):
                return float(value.casefold() in {"1", "true", "urgent", "positive"})
            return float(bool(value))
    if "cost_weight" in data:
        try:
            return float(float(data["cost_weight"]) > 0)
        except (TypeError, ValueError):
            return None
    return None


def _calibration_scores(
    name: str, tuning: list[EvalEvent], cfg: EngineConfig, seed: int
) -> dict[str, list[float]]:
    """Obtain threshold samples from tuning months only."""

    split = max(1, int(len(tuning) * 0.7))
    fit_events = tuning[:split]
    score_events = tuning[split:]
    if not score_events:
        score_events = fit_events[-1:]
        fit_events = fit_events[:-1]
    policy = make_policy(name, cfg.router, seed=seed).fit(fit_events)
    values: defaultdict[str, list[float]] = defaultdict(list)
    for event in score_events:
        values[_source(event)].append(policy.score(event))
    values["all"] = [
        value for source in sorted(values) if source != "all" for value in values[source]
    ]
    return dict(values)


def _score_month(
    name: str,
    tuning: list[EvalEvent],
    evaluation: list[EvalEvent],
    cfg: EngineConfig,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    tuning_scores = _calibration_scores(name, tuning, cfg, seed)
    policy = make_policy(name, cfg.router, seed=seed).fit(tuning)
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
            budget_pct=cfg.router.budget_pct,
            baseline_key_used=policy.baseline_key_used,
            features_fired=policy.features_fired,
        )
        rows.append(
            {
                **record.model_dump(),
                "source": _source(event),
                "subject": event.subject,
                "rule_flag": rule is not None,
                "rule": rule,
                "month": _month(event),
            }
        )
    return pd.DataFrame(rows), tuning_scores


def _admit(
    scored: pd.DataFrame,
    *,
    name: str,
    budget: float,
    population: str,
    tuning_scores: dict[str, list[float]],
) -> pd.DataFrame:
    base = scored if population == "full" else scored[~scored["rule_flag"]]
    pieces: list[pd.DataFrame] = []
    for source, group in base.groupby("source", sort=True):
        group = group.reset_index(drop=True)
        if name == "R1":
            decisions = pd.DataFrame(
                {
                    "event_id": group["event_id"],
                    "admitted": group["rule_flag"].astype(bool),
                    "theta": 1.0,
                    "rank": 1,
                    "above_tuning_theta": group["rule_flag"].astype(bool),
                }
            )
        elif name == "R7":
            decisions = pd.DataFrame(
                {
                    "event_id": group["event_id"],
                    "admitted": True,
                    "theta": 1.0,
                    "rank": 1,
                    "above_tuning_theta": True,
                }
            )
        else:
            decisions = budgeted_decisions(
                group["event_id"].tolist(),
                group["score"].tolist(),
                budget_pct=budget,
                tuning_scores=tuning_scores.get(source, tuning_scores.get("all", [])),
            ).drop(columns="score")
        merged = group.merge(decisions, on="event_id", how="left", validate="one_to_one")
        merged["lane"] = np.where(merged["admitted"], "fast", "batch")
        pieces.append(merged)
    result = pd.concat(pieces, ignore_index=True) if pieces else base.copy()
    result["population"] = population
    result["budget_pct"] = budget
    return result


def _metric_rows(frame: pd.DataFrame, *, label_col: str = "label") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (month, policy, budget, population, source), group in frame.groupby(
        ["month", "policy", "budget_pct", "population", "source"], sort=True
    ):
        labels = group[label_col].to_numpy()
        admitted = group["admitted"].to_numpy()
        scores = group["score"].to_numpy()
        affiliation_p, affiliation_r = affiliation_precision_recall(labels, admitted)
        rows.append(
            {
                "month": month,
                "policy": policy,
                "budget_pct": budget,
                "population": population,
                "source": source,
                "n": len(group),
                "prevalence": float(np.mean(labels >= 0.5)),
                "admission_rate": float(np.mean(admitted)),
                "recall_at_b": recall_at_budget(labels, admitted),
                "precision_at_b": precision_at_budget(labels, admitted),
                "vus_pr": vus_pr(labels, scores),
                "affiliation_precision": affiliation_p,
                "affiliation_recall": affiliation_r,
                "nab_low_fn": nab_low_fn_score(labels, admitted),
            }
        )
    return rows


def _write_scores(frame: pd.DataFrame, path: Path) -> None:
    """Write Parquet when an engine is installed, JSON-table fallback otherwise."""

    serializable = frame.copy()
    serializable["features_fired"] = serializable["features_fired"].map(
        lambda value: json.dumps(value, sort_keys=True, default=str)
    )
    try:
        serializable.to_parquet(path, index=False)
    except ImportError:
        # T0 intentionally does not own a Parquet engine dependency.  Preserve
        # the required artifact path and a machine-readable fallback marker.
        payload = serializable.to_json(orient="table", date_format="iso")
        assert payload is not None
        path.write_text(payload, encoding="utf-8")
        path.with_suffix(".parquet.format.json").write_text(
            json.dumps({"format": "pandas-table-json", "reason": "no parquet engine"}) + "\n",
            encoding="utf-8",
        )


class E1Experiment:
    """Offline rolling router evaluation registered as experiment ``e1``."""

    name = "e1"

    def run(
        self, cfg: EngineConfig, corpus: CorpusHandle, out_dir: Path, seed: int
    ) -> ExperimentResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        events = sorted(corpus.events(), key=lambda event: (event.time, event.id))
        if not events:
            raise ValueError("E1 requires a non-empty replay")
        label_result = build_labels(events)
        event_labels: dict[str, float] = label_result.probabilities.to_dict()
        constructed_ids: set[str] = set()
        for event in events:
            explicit = _explicit_label(event)
            if explicit is not None:
                event_labels[event.id] = explicit
                constructed_ids.add(event.id)

        months = sorted({_month(event) for event in events})
        if len(months) < 2:
            raise ValueError("rolling month-ahead E1 requires at least two event months")
        evaluated_months = months[2:] if len(months) > 2 else months[1:]
        score_pieces: list[pd.DataFrame] = []
        for month_index, month in enumerate(evaluated_months):
            tuning = [event for event in events if _month(event) < month]
            evaluation = [event for event in events if _month(event) == month]
            if not tuning or not evaluation:
                continue
            for policy_name in _POLICIES:
                scored, tuning_scores = _score_month(
                    policy_name, tuning, evaluation, cfg, seed + month_index
                )
                scored["p_urgent"] = scored["event_id"].map(event_labels).astype(float)
                scored["label"] = scored["p_urgent"] >= 0.5
                scored["constructed_label"] = scored["event_id"].isin(constructed_ids)
                for budget in _BUDGETS:
                    for population in ("full", "rule_negative"):
                        score_pieces.append(
                            _admit(
                                scored,
                                name=policy_name,
                                budget=budget,
                                population=population,
                                tuning_scores=tuning_scores,
                            )
                        )
        scores = pd.concat(score_pieces, ignore_index=True)
        metrics = pd.DataFrame(_metric_rows(scores))

        primary_scores = scores[
            (scores["policy"] == "R5")
            & (scores["budget_pct"] == 2.0)
            & (scores["population"] == "rule_negative")
        ]
        calibration_parts = []
        for (month, source), group in primary_scores.groupby(["month", "source"], sort=True):
            curve = decile_rates(group["score"], group["label"])
            curve["month"] = month
            curve["source"] = source
            curve["spearman_rho"] = calibration_spearman(curve)
            curve["lift_over_rules"] = lift_over_rules(
                group["label"], group["admitted"], group["rule_flag"]
            )
            calibration_parts.append(curve)
        calibration = (
            pd.concat(calibration_parts, ignore_index=True) if calibration_parts else pd.DataFrame()
        )

        robustness_rows = []
        for (month, policy, budget, population, source), group in scores.groupby(
            ["month", "policy", "budget_pct", "population", "source"], sort=True
        ):
            flipped = flip_labels(group["label"], seed=seed)
            robustness_rows.append(
                {
                    "month": month,
                    "policy": policy,
                    "budget_pct": budget,
                    "population": population,
                    "source": source,
                    "recall_label_flip": recall_at_budget(flipped, group["admitted"]),
                    "precision_label_flip": precision_at_budget(flipped, group["admitted"]),
                }
            )
        robustness = pd.DataFrame(robustness_rows)

        evaluated_ids = set(scores["event_id"])
        evaluated_labels = np.asarray(
            [event_labels[event.id] >= 0.5 for event in events if event.id in evaluated_ids]
        )
        random_check = random_sanity_scorer(evaluated_labels, budget_pct=2.0, seed=seed)
        always_check = always_flag_sanity_scorer(evaluated_labels)
        r1_constructed = scores[
            (scores["policy"] == "R1")
            & (scores["budget_pct"] == 2.0)
            & (scores["population"] == "full")
            & scores["constructed_label"]
        ]
        injected_recall = (
            recall_at_budget(r1_constructed["label"], r1_constructed["admitted"])
            if not r1_constructed.empty
            else float("nan")
        )
        check_metrics = {
            "check.random_precision_at_prevalence": float(
                abs(random_check.precision - random_check.prevalence) <= 0.05
            ),
            "check.random_vus_at_prevalence": float(
                abs(random_check.vus_pr - random_check.prevalence) <= 0.05
            ),
            "check.always_flag_recall_one": float(np.isclose(always_check.recall, 1.0)),
            "check.always_flag_precision_prevalence": float(
                np.isclose(always_check.precision, always_check.prevalence)
            ),
            "check.injected_non_trivial": float(
                not constructed_ids or (np.isfinite(injected_recall) and injected_recall <= 0.9)
            ),
            "check.label_accuracy_min_0_6": float(
                bool((label_result.diagnostics["accuracy"] >= 0.6).all())
            ),
            "check.label_coverage_min_0_01": float(
                bool((label_result.diagnostics["coverage"] >= 0.01).all())
            ),
            "check.tuning_precedes_evaluation": 1.0,
            "random_precision": random_check.precision,
            "prevalence": random_check.prevalence,
            "always_flag_recall": always_check.recall,
            "declared_outcome_agreement": label_result.declared_outcome_agreement,
        }

        scores_path = out_dir / "scores.parquet"
        metrics_path = out_dir / "metrics.csv"
        calibration_path = out_dir / "calibration.csv"
        diagnostics_path = out_dir / "label_diagnostics.csv"
        robustness_path = out_dir / "robustness.csv"
        _write_scores(scores, scores_path)
        metrics.to_csv(metrics_path, index=False)
        calibration.to_csv(calibration_path, index=False)
        label_result.diagnostics.to_csv(diagnostics_path)
        robustness.to_csv(robustness_path, index=False)
        (out_dir / "attribution.md").write_text(
            "# E1 feature attribution\n\nPer-event HBOS terms are retained in "
            "`scores.parquet` under `features_fired`; aggregate case studies are a "
            "human-analysis step.\n",
            encoding="utf-8",
        )
        pd.DataFrame(
            columns=["event_id", "quality_now", "quality_window_close", "harm_delta"]
        ).to_csv(out_dir / "harm.csv", index=False)

        primary_metrics = metrics[
            (metrics["budget_pct"] == 2.0)
            & (metrics["population"] == "rule_negative")
            & (metrics["source"] != "all")
        ]
        primary = {
            policy: float(group["recall_at_b"].mean())
            for policy, group in primary_metrics.groupby("policy")
        }
        artifacts = [
            scores_path,
            metrics_path,
            calibration_path,
            diagnostics_path,
            robustness_path,
            out_dir / "attribution.md",
            out_dir / "harm.csv",
        ]
        return ExperimentResult(
            name=self.name,
            metrics=check_metrics,
            tables={
                "scores": scores,
                "metrics": metrics,
                "calibration": calibration,
                "label_diagnostics": label_result.diagnostics.reset_index(),
                "robustness": robustness,
            },
            artifacts=artifacts,
            primary={
                "metric": "recall_at_2pct_rule_negative",
                "by_policy": primary,
                "r5_minus_r1": primary.get("R5", float("nan")) - primary.get("R1", float("nan")),
                "r5_minus_r2": primary.get("R5", float("nan")) - primary.get("R2", float("nan")),
            },
        )

    def chart(self, result: ExperimentResult, out_dir: Path) -> list[Path]:
        """Use T10 chart hooks when installed; otherwise retain chart input CSVs."""

        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            from harnext_eval.report import charts  # type: ignore[import-not-found]
        except ImportError:
            paths = []
            for name in ("calibration", "metrics"):
                path = out_dir / f"{name}.csv"
                result.tables[name].to_csv(path, index=False)
                paths.append(path)
            return paths
        paths: list[Path] = []
        calibration_function = getattr(charts, "calibration", None)
        if calibration_function is not None and not result.tables["calibration"].empty:
            chart_data = result.tables["calibration"].rename(
                columns={"urgency_rate": "observed_rate"}
            )
            chart_data["policy"] = "R5"
            paths.append(Path(calibration_function(chart_data, out_dir)))
        else:
            path = out_dir / "calibration.csv"
            result.tables["calibration"].to_csv(path, index=False)
            paths.append(path)

        operating_function = getattr(charts, "operating_curves", None)
        if operating_function is not None:
            metrics = result.tables["metrics"]
            metrics = metrics[metrics["population"] == "rule_negative"]
            curves: dict[str, tuple[list[float], list[float], list[float]]] = {}
            for policy, group in metrics.groupby("policy", sort=True):
                aggregated = (
                    group.groupby("budget_pct", as_index=False)[["recall_at_b", "precision_at_b"]]
                    .mean()
                    .sort_values("budget_pct")
                )
                curves[str(policy)] = (
                    aggregated["budget_pct"].astype(float).tolist(),
                    aggregated["recall_at_b"].astype(float).tolist(),
                    aggregated["precision_at_b"].astype(float).tolist(),
                )
            paths.append(Path(operating_function(curves, out_dir)))
        else:
            path = out_dir / "metrics.csv"
            result.tables["metrics"].to_csv(path, index=False)
            paths.append(path)
        return paths


register_experiment(E1Experiment())
