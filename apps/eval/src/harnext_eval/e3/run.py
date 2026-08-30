"""Store-organisation ablation from docs/evaluation-spec.md §7 E3 and D8/D9."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from harnext_eval.config import EngineConfig
from harnext_eval.corpus import CorpusHandle
from harnext_eval.e2.arms import build_arm
from harnext_eval.e2.run import ProbeOutcome, evaluate_e2, load_probes, macro_accuracy
from harnext_eval.providers.embeddings import EmbeddingsProvider
from harnext_eval.providers.llm import LLMProvider
from harnext_eval.registry import ExperimentResult, register_experiment
from harnext_eval.stats.stats import mcnemar_test, paired_difference_bca
from harnext_eval.stores.base import StoreHandle
from harnext_eval.types import EvalEvent, Probe, SnapshotRef

READ_BUDGETS = (2_000, 8_000, 32_000)
_CHECKPOINT_WEEKS: tuple[int | str, ...] = (1, 2, 4, 8, "end")
_LINK_RE = re.compile(r"\[[^]]*]\(([^)#]+)(?:#[^)]+)?\)")


def compute_erosion_slope(
    checkpoints: Sequence[float | int | datetime], accuracies: Sequence[float]
) -> float:
    """Return the OLS change in accuracy per replay week."""

    if len(checkpoints) != len(accuracies):
        raise ValueError("checkpoints and accuracies must have equal lengths")
    if len(checkpoints) < 2:
        return 0.0
    first = checkpoints[0]
    if isinstance(first, datetime):
        origin = first
        x = np.asarray(
            [
                (checkpoint - origin).total_seconds() / (7 * 24 * 60 * 60)
                for checkpoint in checkpoints
                if isinstance(checkpoint, datetime)
            ],
            dtype=float,
        )
        if len(x) != len(checkpoints):
            raise TypeError("checkpoint types must not be mixed")
    else:
        x = np.asarray(checkpoints, dtype=float)
    y = np.asarray(accuracies, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2 or np.ptp(x[finite]) == 0:
        return 0.0
    design = np.column_stack((np.ones(np.count_nonzero(finite)), x[finite]))
    return float(np.linalg.lstsq(design, y[finite], rcond=None)[0][1])


def _with_budget(cfg: EngineConfig, budget: int) -> EngineConfig:
    return cfg.model_copy(
        update={"reader": cfg.reader.model_copy(update={"budget_tokens": budget})}
    )


def _store_labels(stores: Sequence[StoreHandle]) -> list[tuple[str, StoreHandle]]:
    counts = Counter(str(store.layout).upper() for store in stores)
    seen: defaultdict[str, int] = defaultdict(int)
    labelled: list[tuple[str, StoreHandle]] = []
    for store in stores:
        layout = str(store.layout).upper()
        seen[layout] += 1
        seed = getattr(store, "seed", None)
        if counts[layout] == 1:
            label = layout
        else:
            suffix = seed if seed is not None else seen[layout]
            label = f"{layout}-seed-{suffix}"
        labelled.append((label, store))
    return labelled


def _checkpoint_times(events: Sequence[EvalEvent]) -> list[tuple[str, float, datetime]]:
    if not events:
        return []
    start = min(event.time for event in events)
    end = max(event.time for event in events)
    checkpoints: list[tuple[str, float, datetime]] = []
    for marker in _CHECKPOINT_WEEKS:
        if marker == "end":
            week = (end - start).total_seconds() / (7 * 24 * 60 * 60)
            checkpoints.append(("end", week, end))
        elif isinstance(marker, int):
            checkpoints.append(
                (
                    f"week-{marker}",
                    float(marker),
                    min(start + timedelta(weeks=marker), end),
                )
            )
    return checkpoints


def _external_health(store: StoreHandle, ref: SnapshotRef) -> dict[str, Any] | None:
    try:
        module = importlib.import_module("harnext_eval.health.store_health")
    except ModuleNotFoundError:
        return None
    for name in ("store_health", "measure_store_health", "compute_store_health"):
        function = getattr(module, name, None)
        if function is None:
            continue
        if name == "compute_store_health":
            checkout = store.materialise(ref)
            try:
                result = function(checkout)
            finally:
                shutil.rmtree(checkout, ignore_errors=True)
        else:
            try:
                result = function(store, ref)
            except TypeError:
                continue
        if isinstance(result, dict):
            return result
        if hasattr(result, "model_dump"):
            return dict(result.model_dump())
    return None


def _local_health(store: StoreHandle, ref: SnapshotRef, probes: Sequence[Probe]) -> dict[str, Any]:
    # TODO(integration): retain only as a fallback for deployments without T4 health.
    files = store.list_files(ref)
    contents = {path: store.read(ref, path) or "" for path in files}
    markdown = {path: text for path, text in contents.items() if path.casefold().endswith(".md")}
    refs = 0
    dangling = 0
    for source, text in markdown.items():
        for target in _LINK_RE.findall(text):
            refs += 1
            resolved = (Path(source).parent / target).as_posix()
            if resolved not in contents:
                dangling += 1
    index_entries = 0
    index_resolving = 0
    for source, text in markdown.items():
        if Path(source).name.casefold() != "index.md":
            continue
        for target in _LINK_RE.findall(text):
            index_entries += 1
            resolved = (Path(source).parent / target).as_posix()
            index_resolving += int(resolved in contents)
    fact_lines = [
        line.strip().casefold()
        for path, text in markdown.items()
        if Path(path).name.casefold() == "facts.md"
        for line in text.splitlines()
        if line.strip()
    ]
    duplicates = len(fact_lines) - len(set(fact_lines))
    overview = "\n".join(
        text for path, text in markdown.items() if Path(path).name.casefold() == "overview.md"
    ).casefold()
    superseded_entities = {
        probe.entity: [value.casefold() for value in probe.superseded_values]
        for probe in probes
        if probe.superseded_values
    }
    leaking = sum(
        any(value in overview for value in values) for values in superseded_entities.values()
    )
    return {
        "file_count": len(files),
        "bytes": sum(len(text.encode()) for text in contents.values()),
        "over_cap_share": (
            sum(len(text.splitlines()) > 200 for text in contents.values()) / len(files)
            if files
            else 0.0
        ),
        "index_accuracy": index_resolving / index_entries if index_entries else 1.0,
        "dangling_refs": dangling / refs if refs else 0.0,
        "duplicate_fact_rate": duplicates / len(fact_lines) if fact_lines else 0.0,
        "supersession_leakage": (
            leaking / len(superseded_entities) if superseded_entities else 0.0
        ),
    }


def _health(store: StoreHandle, ref: SnapshotRef, probes: Sequence[Probe]) -> dict[str, Any]:
    return _external_health(store, ref) or _local_health(store, ref, probes)


def _usage_files(store: StoreHandle) -> list[Path]:
    candidates = [Path(store.root) / "usage.jsonl"]
    try:
        candidates.append(Path(store.worktree) / "usage.jsonl")
    except (AttributeError, TypeError):
        pass
    return list(dict.fromkeys(path for path in candidates if path.exists()))


def _usage_cost(store: StoreHandle) -> dict[str, float]:
    records: list[dict[str, Any]] = []
    for path in _usage_files(store):
        with path.open(encoding="utf-8") as source:
            records.extend(json.loads(line) for line in source if line.strip())
    tokens_in = sum(float(row.get("tokens_in", row.get("input_tokens", 0))) for row in records)
    tokens_out = sum(float(row.get("tokens_out", row.get("output_tokens", 0))) for row in records)
    events = sum(float(row.get("event_count", row.get("events", 0))) for row in records)
    failures = sum(
        str(row.get("status", "")).casefold() in {"failed", "rolled_back", "dlq"}
        or row.get("success") is False
        for row in records
    )
    folds = len(records)
    files_touched = sum(
        len(value) if isinstance(value, list) else float(value or 0)
        for value in (row.get("files_touched", 0) for row in records)
    )
    total_tokens = tokens_in + tokens_out
    return {
        "records": float(folds),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "build_tokens": total_tokens,
        "tokens_per_1000_events": total_tokens * 1_000 / events if events else 0.0,
        "cost_usd": sum(float(row.get("cost_usd", row.get("cost", 0))) for row in records),
        "wall_time_s": sum(
            float(row.get("wall_time_s", row.get("latency_s", 0))) for row in records
        ),
        "files_touched": float(files_touched),
        "builder_failure_rate": failures / folds if folds else 0.0,
    }


def _ledger_hash(store: StoreHandle) -> str | None:
    path = Path(store.snapshots_csv)
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as source:
        delivered = [row.get("last_event_id", "") for row in csv.DictReader(source)]
    return hashlib.sha256("\n".join(delivered).encode()).hexdigest()


def _s4_recall_at_10(
    cfg: EngineConfig,
    probes: Sequence[Probe],
    events: Sequence[EvalEvent],
    embeddings: EmbeddingsProvider | None,
) -> float:
    relevant = [probe for probe in probes if probe.source_event_ids]
    if not relevant:
        return 0.0
    hits = []
    for probe in relevant:
        material = build_arm("A3", probe, events, cfg, embeddings=embeddings)
        hits.append(bool(set(probe.source_event_ids) & set(material.source_ids)))
    return float(np.mean(hits))


def _layout(label: str) -> str:
    return label.split("-seed-", maxsplit=1)[0]


def _curve_macro(curve: pd.DataFrame, layout: str, budget: int) -> float | None:
    selected = curve[(curve["layout"] == layout) & (curve["budget"] == budget)]
    return (
        float(np.asarray(selected["macro_acc"], dtype=float).mean()) if not selected.empty else None
    )


def _primary_contrasts(
    curve: pd.DataFrame,
    outcomes: dict[tuple[str, int], list[ProbeOutcome]],
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    s3_labels = sorted({label for label, _ in outcomes if _layout(label) == "S3"})
    s3_values = [
        macro_accuracy(outcomes[(label, 8_000)])
        for label in s3_labels
        if (label, 8_000) in outcomes
    ]
    seed_spread = float(np.std(s3_values, ddof=1)) if len(s3_values) > 1 else 0.0
    s3_macro = _curve_macro(curve, "S3", 8_000)
    for comparator in ("S1", "S4"):
        other_macro = _curve_macro(curve, comparator, 8_000)
        if s3_macro is None or other_macro is None:
            continue
        comparator_labels = sorted(
            {
                label
                for label, budget in outcomes
                if _layout(label) == comparator and budget == 8_000
            }
        )
        by_probe: dict[str, dict[str, list[ProbeOutcome]]] = defaultdict(lambda: defaultdict(list))
        for label in s3_labels:
            for outcome in outcomes.get((label, 8_000), []):
                by_probe[outcome.probe.probe_id]["left"].append(outcome)
        for label in comparator_labels:
            for outcome in outcomes.get((label, 8_000), []):
                by_probe[outcome.probe.probe_id]["right"].append(outcome)
        paired = [value for value in by_probe.values() if value["left"] and value["right"]]
        left_scores = [
            float(np.mean([row.grade.value for row in value["left"]])) for value in paired
        ]
        right_scores = [
            float(np.mean([row.grade.value for row in value["right"]])) for value in paired
        ]
        entities = [value["left"][0].probe.entity for value in paired]
        delta = s3_macro - other_macro
        if len(set(entities)) >= 2:
            stats = paired_difference_bca(
                left_scores,
                right_scores,
                entities,
                n_resamples=1_000,
                random_state=seed,
            )
            ci_low, ci_high = stats.ci_low, stats.ci_high
        else:
            ci_low = ci_high = delta
        mcnemar = mcnemar_test(
            [value >= 1.0 for value in left_scores],
            [value >= 1.0 for value in right_scores],
        )
        reliable = abs(delta) > seed_spread
        rows.append(
            {
                "contrast": f"S3-{comparator}@8000",
                "delta": delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "mcnemar_p": mcnemar.p_value,
                "seed_spread": seed_spread,
                "reliable_difference": reliable,
                "verdict": "reliable difference" if reliable else "no reliable difference",
                "n": len(paired),
            }
        )
    return pd.DataFrame(rows)


def evaluate_e3(
    *,
    stores: Sequence[StoreHandle],
    cfg: EngineConfig,
    probes: Sequence[Probe],
    events: Sequence[EvalEvent],
    out_dir: Path,
    seed: int,
    llm: LLMProvider | None = None,
    embeddings: EmbeddingsProvider | None = None,
    budgets: Sequence[int] = READ_BUDGETS,
) -> ExperimentResult:
    """Run E2 over store layouts, budgets, health checkpoints and usage logs."""

    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_e2"
    labelled = _store_labels(stores)
    curve_rows: list[dict[str, Any]] = []
    outcome_map: dict[tuple[str, int], list[ProbeOutcome]] = {}
    for label, store in labelled:
        for budget in budgets:
            budget_cfg = _with_budget(cfg, int(budget))
            _, outcomes = evaluate_e2(
                cfg=budget_cfg,
                probes=probes,
                events=events,
                out_dir=work / label / str(budget),
                seed=seed,
                store=store,
                llm=llm,
                embeddings=embeddings,
                arms=("A4",),
            )
            outcome_map[(label, int(budget))] = outcomes
            curve_rows.append(
                {
                    "store": label,
                    "layout": _layout(label),
                    "budget": int(budget),
                    "macro_acc": macro_accuracy(outcomes),
                    "n": len(outcomes),
                    "tokens_read": (
                        float(np.mean([row.answer.tokens_read for row in outcomes]))
                        if outcomes
                        else 0.0
                    ),
                }
            )
    curve = pd.DataFrame(curve_rows)

    health_rows: list[dict[str, Any]] = []
    erosion_rows: list[dict[str, Any]] = []
    fixed_subset = sorted(probes, key=lambda probe: probe.probe_id)[:60]
    checkpoints = _checkpoint_times(events)
    for label, store in labelled:
        checkpoint_accuracies: list[float] = []
        checkpoint_weeks: list[float] = []
        for checkpoint, week, at in checkpoints:
            try:
                ref = store.snapshot(at)
            except LookupError:
                continue
            health_rows.append(
                {
                    "store": label,
                    "layout": _layout(label),
                    "checkpoint": checkpoint,
                    "replay_week": week,
                    "snapshot_sha": ref.sha,
                    **_health(store, ref, fixed_subset),
                }
            )
            checkpoint_probes = [probe.model_copy(update={"T": at}) for probe in fixed_subset]
            _, checkpoint_outcomes = evaluate_e2(
                cfg=_with_budget(cfg, 8_000),
                probes=checkpoint_probes,
                events=events,
                out_dir=work / label / "erosion" / checkpoint,
                seed=seed,
                store=store,
                llm=llm,
                embeddings=embeddings,
                arms=("A4",),
            )
            accuracy = macro_accuracy(checkpoint_outcomes)
            checkpoint_weeks.append(week)
            checkpoint_accuracies.append(accuracy)
            erosion_rows.append(
                {
                    "store": label,
                    "layout": _layout(label),
                    "checkpoint": checkpoint,
                    "replay_week": week,
                    "accuracy": accuracy,
                    "n": len(checkpoint_outcomes),
                    "slope_per_week": math.nan,
                }
            )
        slope = compute_erosion_slope(checkpoint_weeks, checkpoint_accuracies)
        for row in erosion_rows:
            if row["store"] == label:
                row["slope_per_week"] = slope

    health = pd.DataFrame(health_rows)
    erosion = pd.DataFrame(erosion_rows)
    cost_rows = []
    for label, store in labelled:
        usage = _usage_cost(store)
        cost_rows.append({"store": label, "layout": _layout(label), **usage})
    cost = pd.DataFrame(cost_rows)
    contrasts = _primary_contrasts(curve, outcome_map, seed)

    paths = {
        "curve": out_dir / "curve.csv",
        "health": out_dir / "health.csv",
        "erosion": out_dir / "erosion.csv",
        "cost": out_dir / "cost.csv",
        "contrasts": out_dir / "contrasts.csv",
    }
    for name, table in (
        ("curve", curve),
        ("health", health),
        ("erosion", erosion),
        ("cost", cost),
        ("contrasts", contrasts),
    ):
        table.to_csv(paths[name], index=False)

    ledgers = [_ledger_hash(store) for _, store in labelled]
    same_input = (
        bool(ledgers) and all(value is not None for value in ledgers) and len(set(ledgers)) == 1
    )
    failure_rates = cost["builder_failure_rate"].tolist() if not cost.empty else []
    s4_recall = (
        _s4_recall_at_10(_with_budget(cfg, 8_000), probes, events, embeddings)
        if any(_layout(label) == "S4" for label, _ in labelled)
        else 0.0
    )
    metrics: dict[str, float] = {
        "checks.same_input_hash": float(same_input),
        "checks.builder_failure_rate_le_0_05": float(all(value <= 0.05 for value in failure_rates)),
        "s4_recall_at_10": s4_recall,
    }
    layouts = sorted(set(curve["layout"].astype(str).tolist())) if "layout" in curve else []
    for layout in layouts:
        for budget in budgets:
            value = _curve_macro(curve, str(layout), int(budget))
            if value is not None:
                metrics[f"acc.{layout}.{int(budget)}"] = value
    primary = {
        row["contrast"]: {
            "delta": row["delta"],
            "seed_spread": row["seed_spread"],
            "reliable_difference": bool(row["reliable_difference"]),
        }
        for row in contrasts.to_dict(orient="records")
    }
    return ExperimentResult(
        name="e3",
        metrics=metrics,
        tables={
            "curve": curve,
            "health": health,
            "erosion": erosion,
            "cost": cost,
            "contrasts": contrasts,
        },
        artifacts=list(paths.values()),
        primary=primary,
    )


class E3Experiment:
    """Registry adapter for store handles supplied in ``CorpusHandle.meta``."""

    name = "e3"

    def run(
        self, cfg: EngineConfig, corpus: CorpusHandle, out_dir: Path, seed: int
    ) -> ExperimentResult:
        raw_stores = corpus.meta.get("stores", [])
        stores = list(raw_stores) if isinstance(raw_stores, (list, tuple)) else []
        return evaluate_e3(
            stores=stores,
            cfg=cfg,
            probes=load_probes(corpus.probes_path),
            events=list(corpus.events()),
            out_dir=out_dir,
            seed=seed,
        )

    def chart(self, result: ExperimentResult, out_dir: Path) -> list[Path]:
        del result, out_dir
        return []


EXPERIMENT = register_experiment(E3Experiment())

__all__ = ["E3Experiment", "READ_BUDGETS", "compute_erosion_slope", "evaluate_e3"]
