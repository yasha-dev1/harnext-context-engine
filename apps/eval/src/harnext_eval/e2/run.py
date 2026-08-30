"""State-fidelity run loop from docs/evaluation-spec.md §7 E2."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from harnext_eval.agents.reader import Material, answer
from harnext_eval.config import EngineConfig
from harnext_eval.corpus import CorpusHandle
from harnext_eval.e2.arms import build_arm
from harnext_eval.grade.exact import grade_exact, normalize_exact
from harnext_eval.grade.links import grade_links
from harnext_eval.grade.localisation import grade_localisation
from harnext_eval.providers.embeddings import EmbeddingsProvider
from harnext_eval.providers.llm import LLMProvider
from harnext_eval.registry import ExperimentResult, register_experiment
from harnext_eval.replay.gate import leakage_gate
from harnext_eval.report.charts import e2_family_bars
from harnext_eval.stats.stats import mcnemar_test, paired_difference_bca
from harnext_eval.stores.base import StoreHandle
from harnext_eval.types import Answer, EvalEvent, GradeResult, Probe, SnapshotRef

DEFAULT_ARMS = (
    "A0",
    "A1",
    "A2",
    "A3",
    "A4",
    "retrieve_everything",
    "retrieve_nothing",
)
_PATH_RE = re.compile(r"(?<![\w/.-])(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]+")


@dataclass(frozen=True)
class ProbeOutcome:
    probe: Probe
    answer: Answer
    grade: GradeResult
    original_tokens: int
    supersession_error: bool


def load_probes(path: Path | None) -> list[Probe]:
    """Read frozen probe JSONL through T0's shared model."""

    if path is None or not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return [Probe.model_validate_json(line) for line in source if line.strip()]


def _passes_leakage_gate(probe: Probe, events: Sequence[EvalEvent], snapshot: Any) -> bool:
    return leakage_gate(
        probe,
        snapshot,
        [event for event in events if event.time <= probe.T],
        all_events=events,
        gold_action_time=probe.T + timedelta(microseconds=1),
        out_csv=Path(os.devnull),
    )


def _as_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        for key in ("links", "files", "values", "gold"):
            if key in value:
                return _as_values(value[key])
        return [str(item) for item in value.values()]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def grade_answer(probe: Probe, prediction: str) -> GradeResult:
    """Dispatch E2's deterministic grader from the frozen gold type."""

    if probe.gold_type == "links" or probe.family == "multisource":
        return grade_links(probe.probe_id, prediction, _as_values(probe.gold))
    if probe.gold_type == "files" or probe.family == "code_location":
        predicted = _PATH_RE.findall(prediction)
        result = grade_localisation(probe.probe_id, predicted, _as_values(probe.gold))
        details = result.details
        precision = float(details.get("file_precision", 0.0))
        recall = float(details.get("file_recall", 0.0))
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return result.model_copy(
            update={"metric": "file_f1", "value": f1, "details": {**details, "file_f1": f1}}
        )
    return grade_exact(probe.probe_id, prediction, probe.gold)


def _macro_family(family: str) -> str:
    return "multisource" if family == "code_location" else family


def macro_accuracy(outcomes: Sequence[ProbeOutcome]) -> float:
    """Mean family accuracy, treating code localisation as multi-source."""

    by_family: dict[str, list[float]] = defaultdict(list)
    for outcome in outcomes:
        by_family[_macro_family(outcome.probe.family)].append(outcome.grade.value)
    return float(np.mean([np.mean(values) for values in by_family.values()])) if by_family else 0.0


def _metric_rows(outcomes: Sequence[ProbeOutcome], arms: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in arms:
        arm_rows = [outcome for outcome in outcomes if outcome.answer.arm == arm]
        families = sorted({outcome.probe.family for outcome in arm_rows})
        for family in families:
            selected = [outcome for outcome in arm_rows if outcome.probe.family == family]
            rows.append(_summarise(arm, family, selected))
        if arm_rows:
            summary = _summarise(arm, "macro", arm_rows)
            summary["accuracy"] = macro_accuracy(arm_rows)
            rows.append(summary)
    return rows


def _summarise(arm: str, family: str, rows: Sequence[ProbeOutcome]) -> dict[str, Any]:
    updates = [row for row in rows if row.probe.family == "update"]
    abstentions = [row for row in rows if row.probe.family == "abstention"]
    return {
        "arm": arm,
        "family": family,
        "n": len(rows),
        "accuracy": float(np.mean([row.grade.value for row in rows])) if rows else math.nan,
        "supersession_error": (
            float(np.mean([row.supersession_error for row in updates])) if updates else math.nan
        ),
        "abstention_precision": (
            float(np.mean([normalize_exact(row.answer.text) == "unknown" for row in abstentions]))
            if abstentions
            else math.nan
        ),
        "tokens_read": float(np.mean([row.answer.tokens_read for row in rows])) if rows else 0.0,
        "tool_calls": float(np.mean([row.answer.tool_calls for row in rows])) if rows else 0.0,
        "latency_s": float(np.mean([row.answer.latency_s for row in rows])) if rows else 0.0,
    }


def _paired_contrast(
    outcomes: Sequence[ProbeOutcome], left: str, right: str, seed: int
) -> pd.DataFrame:
    indexed = {(row.probe.probe_id, row.answer.arm): row for row in outcomes}
    pairs = [
        (indexed[(probe_id, left)], indexed[(probe_id, right)])
        for probe_id in sorted({key[0] for key in indexed})
        if (probe_id, left) in indexed and (probe_id, right) in indexed
    ]
    columns = [
        "contrast",
        "delta",
        "ci_low",
        "ci_high",
        "mcnemar_p",
        "discordant_left",
        "discordant_right",
        "n",
        "entities",
    ]
    if not pairs:
        return pd.DataFrame(columns=columns)
    left_scores = [pair[0].grade.value for pair in pairs]
    right_scores = [pair[1].grade.value for pair in pairs]
    entities = [pair[0].probe.entity for pair in pairs]
    delta = float(np.mean(np.asarray(left_scores) - np.asarray(right_scores)))
    unique_entities = set(entities)
    if len(unique_entities) >= 2:
        bootstrap = paired_difference_bca(
            left_scores,
            right_scores,
            entities,
            n_resamples=1_000,
            random_state=seed,
        )
        ci_low, ci_high = bootstrap.ci_low, bootstrap.ci_high
    else:
        ci_low = ci_high = delta
    mcnemar = mcnemar_test(
        [value >= 1.0 for value in left_scores],
        [value >= 1.0 for value in right_scores],
    )
    return pd.DataFrame(
        [
            {
                "contrast": f"{left}-{right}",
                "delta": delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "mcnemar_p": mcnemar.p_value,
                "discordant_left": mcnemar.b,
                "discordant_right": mcnemar.c,
                "n": len(pairs),
                "entities": len(unique_entities),
            }
        ],
        columns=columns,
    )


def _answer_record(outcome: ProbeOutcome) -> dict[str, Any]:
    return {
        **outcome.answer.model_dump(mode="json"),
        "family": outcome.probe.family,
        "entity": outcome.probe.entity,
        "grade_metric": outcome.grade.metric,
        "grade": outcome.grade.value,
        "grade_details": outcome.grade.details,
        "supersession_error": outcome.supersession_error,
    }


def _write_gate(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=["probe_id", "T", "sha", "last_event_id", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)


def evaluate_e2(
    *,
    cfg: EngineConfig,
    probes: Sequence[Probe],
    events: Sequence[EvalEvent],
    out_dir: Path,
    seed: int,
    store: StoreHandle | None = None,
    llm: LLMProvider | None = None,
    embeddings: EmbeddingsProvider | None = None,
    arms: Sequence[str] = DEFAULT_ARMS,
) -> tuple[ExperimentResult, list[ProbeOutcome]]:
    """Run the complete probe × arm matrix and persist E2 outputs."""

    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes: list[ProbeOutcome] = []
    gate_rows: list[dict[str, Any]] = []
    budget = cfg.reader.budget_tokens
    budget_checks: list[bool] = []
    for probe in probes:
        sha = ""
        last_event_id = ""
        snapshot: Any = None
        if store is not None:
            try:
                ref = store.snapshot(probe.T)
                sha, last_event_id = ref.sha, ref.last_event_id
                snapshot = ref
            except LookupError:
                pass
        if snapshot is None:
            visible = [event for event in events if event.time <= probe.T]
            if visible:
                last = max(visible, key=lambda event: (event.time, event.id))
                snapshot = SnapshotRef(
                    sha="raw-events",
                    T_last_event=last.time,
                    last_event_id=last.id,
                    lane="raw",
                )
            else:
                snapshot = SnapshotRef(
                    sha="empty",
                    T_last_event=probe.T,
                    last_event_id="",
                    lane="raw",
                )
        passed = _passes_leakage_gate(probe, events, snapshot)
        gate_rows.append(
            {
                "probe_id": probe.probe_id,
                "T": probe.T.isoformat(),
                "sha": sha,
                "last_event_id": last_event_id,
                "status": "PASS" if passed else "FAIL",
            }
        )
        if not passed:
            continue
        for arm in arms:
            material: Material = build_arm(
                arm,
                probe,
                events,
                cfg,
                store=store,
                embeddings=embeddings,
            )
            response = answer(probe, material, cfg, provider=llm)
            grade = grade_answer(probe, response.text)
            superseded = any(
                normalize_exact(value) in normalize_exact(response.text)
                for value in probe.superseded_values
                if normalize_exact(value)
            )
            outcomes.append(
                ProbeOutcome(
                    probe=probe,
                    answer=response,
                    grade=grade,
                    original_tokens=material.original_tokens or 0,
                    supersession_error=superseded,
                )
            )
            if material.enforce_budget and (material.original_tokens or 0) >= budget:
                budget_checks.append(0.9 * budget <= response.tokens_read <= 1.1 * budget)

    answer_path = out_dir / "answers.jsonl"
    answer_path.write_text(
        "".join(json.dumps(_answer_record(row), sort_keys=True) + "\n" for row in outcomes),
        encoding="utf-8",
    )
    _write_gate(out_dir / "gate.csv", gate_rows)
    metric_rows = _metric_rows(outcomes, arms)
    metrics_table = pd.DataFrame(metric_rows)
    metrics_path = out_dir / "metrics.csv"
    metrics_table.to_csv(metrics_path, index=False)
    contrasts = _paired_contrast(outcomes, "A4", "A3", seed)
    contrasts_path = out_dir / "contrasts.csv"
    contrasts.to_csv(contrasts_path, index=False)

    arm_outcomes = {arm: [row for row in outcomes if row.answer.arm == arm] for arm in arms}
    everything = macro_accuracy(arm_outcomes.get("retrieve_everything", []))
    prior_target = [
        row
        for row in arm_outcomes.get("A0", [])
        if _macro_family(row.probe.family) in {"temporal", "update", "multisource"}
    ]
    prior = float(np.mean([row.grade.value for row in prior_target])) if prior_target else 0.0
    passed_count = sum(row["status"] == "PASS" for row in gate_rows)
    checks: dict[str, float] = {
        "checks.floor_retrieve_everything_ge_0_9": float(everything >= 0.9),
        "checks.prior_leq_0_3": float(prior <= 0.3),
        "checks.leakage_gate_100_pct": float(passed_count == len(gate_rows)),
        "checks.budget_within_10_pct": float(all(budget_checks)),
        "floor_retrieve_everything": everything,
        "prior_target_accuracy": prior,
        "gate_pass_count": float(passed_count),
        "gate_exclusion_count": float(len(gate_rows) - passed_count),
    }
    for arm, rows in arm_outcomes.items():
        checks[f"macro_acc.{arm}"] = macro_accuracy(rows)
    primary_row = contrasts.iloc[0].to_dict() if not contrasts.empty else {}
    result = ExperimentResult(
        name="e2",
        metrics=checks,
        tables={"metrics": metrics_table, "contrasts": contrasts},
        artifacts=[answer_path, metrics_path, contrasts_path, out_dir / "gate.csv"],
        primary=primary_row,
    )
    return result, outcomes


class E2Experiment:
    """Registry adapter for the shared Experiment protocol."""

    name = "e2"

    def run(
        self, cfg: EngineConfig, corpus: CorpusHandle, out_dir: Path, seed: int
    ) -> ExperimentResult:
        probes = load_probes(corpus.probes_path)
        events = list(corpus.events())
        store = corpus.meta.get("store")
        return evaluate_e2(
            cfg=cfg,
            probes=probes,
            events=events,
            out_dir=out_dir,
            seed=seed,
            store=store if isinstance(store, StoreHandle) else None,
        )[0]

    def chart(self, result: ExperimentResult, out_dir: Path) -> list[Path]:
        table = result.tables["metrics"]
        if table.empty:
            return []
        chart_data = pd.DataFrame(table[table["family"] != "macro"])
        return [e2_family_bars(chart_data, out_dir)]


EXPERIMENT = register_experiment(E2Experiment())

__all__ = [
    "DEFAULT_ARMS",
    "E2Experiment",
    "ProbeOutcome",
    "evaluate_e2",
    "grade_answer",
    "load_probes",
    "macro_accuracy",
]
