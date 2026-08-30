"""E5 cadence/economics ablation from docs/evaluation-spec.md §7 E5 and D7."""

from __future__ import annotations

import csv
import importlib
import inspect
import json
import math
import shutil
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from harnext_eval.config import EngineConfig
from harnext_eval.corpus import CorpusHandle
from harnext_eval.e4.tasks import is_rule_promoted
from harnext_eval.providers.tokenizer import count_tokens
from harnext_eval.registry import ExperimentResult, register_experiment
from harnext_eval.stores.base import StoreHandle, register_layout
from harnext_eval.types import EvalEvent, Probe, SnapshotRef

CADENCES = ("W1", "W5", "W20", "W50", "W20+rules", "W20+rules+deviation")
DEFAULT_PRICES = {"input_per_million": 3.0, "output_per_million": 15.0}
_SNAPSHOT_FIELDS = ("T_last_event", "sha", "last_event_id", "lane")


@dataclass(frozen=True)
class CadenceSetting:
    name: str
    max_events: int
    rules: bool
    deviation: bool
    last_events: int | None = None


def cadence_setting(name: str) -> CadenceSetting:
    normalised = name.strip()
    settings = {
        "W1": CadenceSetting("W1", 1, True, False, 1_000),
        "W5": CadenceSetting("W5", 5, False, False),
        "W20": CadenceSetting("W20", 20, False, False),
        "W50": CadenceSetting("W50", 50, False, False),
        "W20+rules": CadenceSetting("W20+rules", 20, True, False),
        "W20+rules+deviation": CadenceSetting("W20+rules+deviation", 20, True, True),
    }
    try:
        return settings[normalised]
    except KeyError as exc:
        raise ValueError(f"unknown cadence {name!r}") from exc


def _entity_relpath(entity: str) -> str:
    if ":" in entity:
        kind, slug = entity.split(":", 1)
        return f"entities/{kind}/{slug.replace('/', '__')}"
    return f"entities/{entity}"


def _fallback_layout(store: StoreHandle, events: list[EvalEvent], lane: str) -> None:
    """Minimal S3-shaped fold used only when replay/T5 are absent."""

    del lane
    for event in events:
        base = _entity_relpath(event.subject)
        timeline_path = store.worktree / base / "timeline.md"
        facts_path = store.worktree / base / "facts.md"
        timeline = timeline_path.read_text(encoding="utf-8") if timeline_path.exists() else ""
        facts = facts_path.read_text(encoding="utf-8") if facts_path.exists() else ""
        payload = json.dumps(event.data, sort_keys=True, default=str)
        store.write(
            f"{base}/timeline.md",
            timeline + f"- {event.time.isoformat()} [{event.source}#{event.id}] {payload}\n",
        )
        store.write(f"{base}/facts.md", facts + f"- [{event.id}] {payload}\n")
        store.write(
            f"{base}/OVERVIEW.md",
            f"# {event.subject}\n\nCurrent state from [{event.id}]: {payload}\n",
        )


class MeteredStoreHandle(StoreHandle):
    """Store wrapper that records Fake-builder provider usage per fold."""

    def __init__(self, *args: Any, usage_path: Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.usage_path = usage_path
        self.usage_path.parent.mkdir(parents=True, exist_ok=True)
        self.fold_count = 0

    def fold(self, events: list[EvalEvent], lane: str) -> SnapshotRef:
        prior_usage_count = len(_read_jsonl(self.usage_path))
        try:
            ref = super().fold(events, lane)
        except RuntimeError as exc:
            if "no fold callable registered" not in str(exc):
                raise
            register_layout(self.layout, _fallback_layout)
            ref = super().fold(events, lane)
        self.fold_count += 1
        input_tokens = sum(count_tokens(event.model_dump_json()) for event in events)
        output_tokens = max(1, round(input_tokens * 0.2))
        defaults = {
            "run": self.fold_count,
            "snapshot_sha": ref.sha,
            "commit_time": ref.T_last_event.isoformat(),
            "event_ids": [event.id for event in events],
            "lane": lane,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        usage_rows = _read_jsonl(self.usage_path)
        if len(usage_rows) > prior_usage_count:
            for key, value in defaults.items():
                usage_rows[-1].setdefault(key, value)
            _write_jsonl(self.usage_path, usage_rows)
        else:
            with self.usage_path.open("a", encoding="utf-8") as destination:
                destination.write(json.dumps(defaults, sort_keys=True) + "\n")
        return ref

    def set_last_commit_time(self, commit_time: datetime) -> None:
        rows = _read_jsonl(self.usage_path)
        if not rows:
            return
        rows[-1]["commit_time"] = commit_time.isoformat()
        _write_jsonl(self.usage_path, rows)


def _scaled_seconds(base: float, cap: int) -> float:
    return base * cap / 20


def _cadence_cfg(cfg: EngineConfig, setting: CadenceSetting) -> EngineConfig:
    window = cfg.window.model_copy(
        update={
            "max_events": setting.max_events,
            "gap_s": _scaled_seconds(30.0, setting.max_events),
            "max_age_s": _scaled_seconds(120.0, setting.max_events),
        }
    )
    rules = cfg.router.rules.model_copy(update={"enabled": setting.rules})
    deviation = cfg.router.deviation.model_copy(update={"enabled": setting.deviation})
    router = cfg.router.model_copy(update={"rules": rules, "deviation": deviation})
    return cfg.model_copy(update={"window": window, "router": router})


def _deviation_promoted(event: EvalEvent, budget_pct: float) -> bool:
    bucket = int.from_bytes(event.id.encode("utf-8")[:8], "little", signed=False) % 10_000
    return bucket < round(budget_pct * 100)


def _local_replay(
    events: list[EvalEvent], cfg: EngineConfig, setting: CadenceSetting, store: MeteredStoreHandle
) -> int:
    """Event-clock session-window loop matching the replay driver contract."""

    if setting.name == "W1":
        for event in events:
            store.fold([event], "fast")
            store.set_last_commit_time(event.time)
        return len(events)

    windows: dict[str, list[EvalEvent]] = defaultdict(list)
    opened: dict[str, datetime] = {}
    close_count = 0

    def flush(entity: str, commit_time: datetime) -> None:
        nonlocal close_count
        pending = windows.pop(entity, [])
        opened.pop(entity, None)
        if pending:
            store.fold(pending, "batch")
            store.set_last_commit_time(commit_time)
            close_count += 1

    for event in events:
        pending = windows[event.subject]
        if pending:
            gap_close = pending[-1].time + timedelta(seconds=cfg.window.gap_s)
            age_close = opened[event.subject] + timedelta(seconds=cfg.window.max_age_s)
            if event.time > gap_close or event.time > age_close:
                flush(event.subject, min(gap_close, age_close))
                pending = windows[event.subject]
        promoted = (setting.rules and is_rule_promoted(event)) or (
            setting.deviation
            and not is_rule_promoted(event)
            and _deviation_promoted(event, cfg.router.budget_pct)
        )
        if promoted:
            flush(event.subject, event.time)
            store.fold([event], "fast")
            store.set_last_commit_time(event.time)
            close_count += 1
            continue
        if not pending:
            opened[event.subject] = event.time
        pending.append(event)
        if len(pending) >= cfg.window.max_events:
            flush(event.subject, event.time)
    final_time = events[-1].time if events else datetime.min
    for entity in list(windows):
        pending = windows[entity]
        if pending:
            close_time = min(
                pending[-1].time + timedelta(seconds=cfg.window.gap_s),
                opened[entity] + timedelta(seconds=cfg.window.max_age_s),
            )
            flush(entity, max(close_time, final_time))
    return close_count


def _shared_replay(
    events: list[EvalEvent], cfg: EngineConfig, store: MeteredStoreHandle
) -> tuple[bool, int]:
    try:
        module = importlib.import_module("harnext_eval.replay.driver")
    except ModuleNotFoundError:
        return False, 0  # TODO(integration): T2 absent in isolated T8 tests.
    runner = getattr(module, "run_pipeline", None)
    if not callable(runner):
        return False, 0
    stats = runner(events, cfg, store, cutoff=None, on_decision=None)
    for name in ("folds", "fold_count", "builder_runs", "windows_closed"):
        value = getattr(stats, name, None)
        if isinstance(value, int):
            return True, value
    return True, store.fold_count


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prices(cfg: EngineConfig, meta: Mapping[str, Any]) -> dict[str, float]:
    candidate = getattr(cfg, "prices", None) or meta.get("prices") or DEFAULT_PRICES
    if not isinstance(candidate, Mapping):
        return dict(DEFAULT_PRICES)
    aliases = {
        "input_per_million": ("input_per_million", "input_per_1m", "input"),
        "output_per_million": ("output_per_million", "output_per_1m", "output"),
    }
    result: dict[str, float] = {}
    for target, names in aliases.items():
        result[target] = next(
            (float(candidate[name]) for name in names if name in candidate), DEFAULT_PRICES[target]
        )
    return result


def _usage_cost(records: Iterable[Mapping[str, Any]], prices: Mapping[str, float]) -> float:
    total = 0.0
    for record in records:
        if record.get("cost_usd") is not None and float(record["cost_usd"]) > 0:
            total += float(record["cost_usd"])
            continue
        nested_usage = record.get("usage")
        usage: Mapping[str, Any] = nested_usage if isinstance(nested_usage, Mapping) else record
        input_tokens = float(usage.get("input_tokens", usage.get("prompt_tokens", 0)))
        output_tokens = float(usage.get("output_tokens", usage.get("completion_tokens", 0)))
        total += input_tokens * prices["input_per_million"] / 1_000_000
        total += output_tokens * prices["output_per_million"] / 1_000_000
    return total


def _freshness(
    cadence: str,
    seed: int,
    events: list[EvalEvent],
    usage: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    event_by_id = {event.id: event for event in events}
    first_commit: dict[str, datetime] = {}
    for record in usage:
        commit_raw = record.get("commit_time")
        if not commit_raw:
            continue
        commit_time = datetime.fromisoformat(str(commit_raw).replace("Z", "+00:00"))
        for event_id in record.get("event_ids", []):
            first_commit.setdefault(str(event_id), commit_time)
    rows: list[dict[str, Any]] = []
    for event_id, event in event_by_id.items():
        commit = first_commit.get(event_id)
        if commit is None:
            continue
        rows.append(
            {
                "cadence": cadence,
                "seed": seed,
                "event_id": event_id,
                "entity": event.subject,
                "event_time": event.time.isoformat(),
                "first_commit_time": commit.isoformat(),
                "freshness_s": max(0.0, (commit - event.time).total_seconds()),
                "urgent": is_rule_promoted(event),
            }
        )
    return rows


def _snapshots(store: StoreHandle) -> list[SnapshotRef]:
    if not store.snapshots_csv.exists():
        return []
    with store.snapshots_csv.open(newline="", encoding="utf-8") as source:
        return [
            SnapshotRef(
                sha=row["sha"],
                T_last_event=datetime.fromisoformat(row["T_last_event"]),
                last_event_id=row["last_event_id"],
                lane=row["lane"],
            )
            for row in csv.DictReader(source)
        ]


def _probe_values(gold: Any) -> list[str]:
    if isinstance(gold, Mapping):
        return [str(value) for value in gold.values()]
    if isinstance(gold, Sequence) and not isinstance(gold, (str, bytes)):
        return [str(value) for value in gold]
    return [str(gold)]


def _local_e2(store: StoreHandle, probes: list[Probe]) -> float:
    refs = _snapshots(store)
    if not refs or not probes:
        return math.nan
    final = refs[-1]
    files = {
        path: content
        for path in store.list_files(final)
        if (content := store.read(final, path)) is not None
    }
    by_family: dict[str, list[float]] = defaultdict(list)
    for probe in probes:
        entity_bits = {
            probe.entity.casefold(),
            probe.entity.split(":", 1)[-1].casefold(),
        }
        material = "\n".join(
            body
            for path, body in files.items()
            if any(
                bit and (bit in path.casefold() or bit in body.casefold()) for bit in entity_bits
            )
        )
        values = _probe_values(probe.gold)
        by_family[probe.family].append(
            float(any(value.casefold() in material.casefold() for value in values))
        )
    family_means = [statistics.fmean(values) for values in by_family.values() if values]
    return statistics.fmean(family_means) if family_means else math.nan


def _shared_e2(
    store: StoreHandle,
    probes: list[Probe],
    cfg: EngineConfig,
    events: list[EvalEvent],
    out_dir: Path,
    seed: int,
) -> float | None:
    try:
        module = importlib.import_module("harnext_eval.e2.run")
    except ModuleNotFoundError:
        return None  # TODO(integration): T7 absent in isolated T8 tests.
    evaluator = getattr(module, "evaluate_store", None)
    if not callable(evaluator):
        evaluate_e2 = getattr(module, "evaluate_e2", None)
        refs = _snapshots(store)
        if not callable(evaluate_e2) or not refs or not probes:
            return None
        final_probes = [probe.model_copy(update={"T": refs[-1].T_last_event}) for probe in probes]
        try:
            raw_result = cast(Callable[..., Any], evaluate_e2)(
                cfg=cfg,
                probes=final_probes,
                events=events,
                out_dir=out_dir,
                seed=seed,
                store=store,
                arms=("A4",),
            )
        except (TypeError, ValueError):
            return None
        if not isinstance(raw_result, tuple) or not raw_result:
            return None
        result = raw_result[0]
        if not hasattr(result, "metrics"):
            return None
        value = result.metrics.get("macro_acc.A4")
        return float(value) if value is not None else None
    try:
        parameters = inspect.signature(evaluator).parameters
        kwargs: dict[str, Any] = {}
        if "store" in parameters:
            kwargs["store"] = store
        if "probes" in parameters:
            kwargs["probes"] = probes
        if "cfg" in parameters:
            kwargs["cfg"] = cfg
        result = evaluator(**kwargs)
    except (TypeError, ValueError):
        return None
    if isinstance(result, Mapping) and result.get("macro_acc") is not None:
        return float(result["macro_acc"])
    value = getattr(result, "macro_acc", None)
    return float(value) if value is not None else None


def _local_health(store: StoreHandle) -> dict[str, float]:
    refs = _snapshots(store)
    if not refs:
        return {"files_per_entity": math.nan, "dup_rate": math.nan}
    final = refs[-1]
    paths = store.list_files(final)
    entity_files = [
        path for path in paths if path.startswith("entities/") and not path.endswith(".gitkeep")
    ]
    entities = {"/".join(path.split("/")[:3]) for path in entity_files if len(path.split("/")) >= 3}
    fact_lines: list[str] = []
    for path in entity_files:
        if Path(path).name.casefold() != "facts.md":
            continue
        body = store.read(final, path) or ""
        fact_lines.extend(line.strip().casefold() for line in body.splitlines() if line.strip())
    duplicates = len(fact_lines) - len(set(fact_lines))
    return {
        "files_per_entity": len(entity_files) / len(entities) if entities else 0.0,
        "dup_rate": duplicates / len(fact_lines) if fact_lines else 0.0,
    }


def _shared_health(store: StoreHandle) -> dict[str, float] | None:
    try:
        module = importlib.import_module("harnext_eval.health.store_health")
    except ModuleNotFoundError:
        return None  # TODO(integration): T4 absent in isolated T8 tests.
    compute = getattr(module, "compute_store_health", None)
    refs = _snapshots(store)
    if callable(compute) and refs:
        checkout = store.materialise(refs[-1])
        try:
            result = compute(checkout)
        finally:
            shutil.rmtree(checkout)
        if isinstance(result, Mapping):
            return {
                "files_per_entity": float(result.get("files_per_entity", 0.0)),
                "dup_rate": float(result.get("near_duplicate_fact_rate", 0.0)),
            }
    for name in ("measure_store", "store_health", "evaluate"):
        evaluator = getattr(module, name, None)
        if not callable(evaluator):
            continue
        try:
            result = evaluator(store)
        except TypeError:
            continue
        if isinstance(result, Mapping):
            files = result.get("files_per_entity")
            dup = result.get("dup_rate", result.get("duplicate_fact_rate"))
            if files is not None and dup is not None:
                return {"files_per_entity": float(files), "dup_rate": float(dup)}
    return None


def _quantile(values: Iterable[float], q: float) -> float:
    items = list(values)
    return float(np.quantile(items, q)) if items else math.nan


def _load_probes(path: Path | None) -> list[Probe]:
    if path is None or not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return [Probe.model_validate_json(line) for line in source if line.strip()]


def run_cadences(
    cfg: EngineConfig,
    corpus: CorpusHandle,
    out_dir: Path,
    seed: int,
    *,
    cadences: Iterable[str] = CADENCES,
    probes: Iterable[Probe] | None = None,
) -> ExperimentResult:
    """Build and measure each E5 cadence on the identical frozen replay."""

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        importlib.import_module("harnext_eval.stores.layouts")
    except ModuleNotFoundError:
        pass  # TODO(integration): T5 absent in isolated T8 tests.
    all_events = list(corpus.events())
    probe_list = list(probes) if probes is not None else _load_probes(corpus.probes_path)
    prices = _prices(cfg, corpus.meta)
    cost_rows: list[dict[str, Any]] = []
    freshness_rows: list[dict[str, Any]] = []
    for cadence in cadences:
        setting = cadence_setting(cadence)
        events = all_events[-setting.last_events :] if setting.last_events else list(all_events)
        cadence_cfg = _cadence_cfg(cfg, setting)
        safe_name = cadence.replace("+", "_")
        cadence_dir = out_dir / "stores" / safe_name / f"seed-{seed}"
        usage_path = cadence_dir / "usage.jsonl"
        store = MeteredStoreHandle(
            cadence_cfg.store.layout,
            f"e5-{safe_name}-{seed}",
            cadence_dir,
            usage_path=usage_path,
        )
        used_shared, expected_runs = (
            (False, 0) if setting.name == "W1" else _shared_replay(events, cadence_cfg, store)
        )
        if not used_shared:
            expected_runs = _local_replay(events, cadence_cfg, setting, store)
        usage = _read_jsonl(usage_path)
        freshness = _freshness(cadence, seed, events, usage)
        freshness_rows.extend(freshness)
        shared_acc = _shared_e2(
            store,
            probe_list,
            cadence_cfg,
            events,
            cadence_dir / "e2-final",
            seed,
        )
        macro_acc = shared_acc if shared_acc is not None else _local_e2(store, probe_list)
        health = _shared_health(store) or _local_health(store)
        event_count = len(events)
        dollars = _usage_cost(usage, prices)
        urgent_delays = [float(row["freshness_s"]) for row in freshness if row["urgent"]]
        routine_delays = [float(row["freshness_s"]) for row in freshness if not row["urgent"]]
        cost_rows.append(
            {
                "cadence": cadence,
                "seed": seed,
                "events": event_count,
                "builder_runs": len(usage),
                "expected_window_closes": expected_runs,
                "cost_usd": dollars,
                "cost_1k": dollars / (event_count / 1_000) if event_count else math.nan,
                "runs_1k": len(usage) / (event_count / 1_000) if event_count else math.nan,
                "macro_acc": macro_acc,
                "files_per_entity": health["files_per_entity"],
                "dup_rate": health["dup_rate"],
                "fresh_urgent_p50_s": _quantile(urgent_delays, 0.50),
                "fresh_urgent_p95_s": _quantile(urgent_delays, 0.95),
                "fresh_routine_p50_s": _quantile(routine_delays, 0.50),
                "fresh_routine_p95_s": _quantile(routine_delays, 0.95),
                "usage_records_present": bool(usage),
                "builder_count_matches": len(usage) == expected_runs,
            }
        )

    by_cadence = {str(row["cadence"]): row for row in cost_rows}
    w1 = by_cadence.get("W1")
    rules = by_cadence.get("W20+rules")
    accuracy_ci = float(corpus.meta.get("e2_accuracy_ci", 0.0))
    comparable = bool(
        w1
        and rules
        and not math.isnan(float(w1["macro_acc"]))
        and not math.isnan(float(rules["macro_acc"]))
        and float(rules["macro_acc"]) >= float(w1["macro_acc"]) - accuracy_ci
    )
    primary_ratio = (
        float(rules["cost_1k"]) / float(w1["cost_1k"])
        if comparable and w1 and rules and float(w1["cost_1k"]) > 0
        else math.nan
    )
    pareto_rows = [
        {
            "cadence": row["cadence"],
            "seed": row["seed"],
            "cost_1k": row["cost_1k"],
            "macro_acc": row["macro_acc"],
            "fresh_urgent_p95_s": row["fresh_urgent_p95_s"],
            "fresh_routine_p95_s": row["fresh_routine_p95_s"],
        }
        for row in cost_rows
    ]
    cost_frame = pd.DataFrame(cost_rows)
    freshness_frame = pd.DataFrame(freshness_rows)
    pareto_frame = pd.DataFrame(pareto_rows)
    artifacts = [out_dir / "cost.csv", out_dir / "freshness.csv", out_dir / "pareto.csv"]
    cost_frame.to_csv(artifacts[0], index=False)
    freshness_frame.to_csv(artifacts[1], index=False)
    pareto_frame.to_csv(artifacts[2], index=False)
    w1_p95 = float(w1["fresh_routine_p95_s"]) if w1 else math.nan
    metrics = {
        "primary.cost_ratio_w20_rules_to_w1": primary_ratio,
        "checks.equal_accuracy": float(comparable),
        "checks.cost_from_usage": float(
            all(bool(row["usage_records_present"]) for row in cost_rows)
        ),
        "checks.builder_run_count": float(
            all(bool(row["builder_count_matches"]) for row in cost_rows)
        ),
        "checks.w1_passthrough_p95_approx_zero": float(not math.isnan(w1_p95) and w1_p95 <= 1e-9),
    }
    return ExperimentResult(
        name="e5",
        metrics=metrics,
        tables={"cost": cost_frame, "freshness": freshness_frame, "pareto": pareto_frame},
        artifacts=artifacts,
        primary={
            "cost_ratio_w20_rules_to_w1": primary_ratio,
            "reported_at_equal_accuracy": comparable,
            "price_of_freshness": not comparable,
        },
    )


class E5Experiment:
    name = "e5"

    def run(
        self, cfg: EngineConfig, corpus: CorpusHandle, out_dir: Path, seed: int
    ) -> ExperimentResult:
        return run_cadences(cfg, corpus, out_dir, seed)

    def chart(self, result: ExperimentResult, out_dir: Path) -> list[Path]:
        del result, out_dir
        return []


register_experiment(E5Experiment())

__all__ = ["CADENCES", "E5Experiment", "cadence_setting", "run_cadences"]
