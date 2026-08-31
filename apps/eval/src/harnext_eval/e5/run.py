"""E5 cadence/economics ablation from docs/evaluation-spec.md §7 E5 and §8."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

from harnext_eval.config import EngineConfig
from harnext_eval.corpus import CorpusHandle
from harnext_eval.e1.labels import build_labels
from harnext_eval.e1.policies import RouterPolicy, make_policy
from harnext_eval.e2.run import (
    BOOTSTRAP_RESAMPLES,
    ProbeOutcome,
    _paired_contrast,
    evaluate_e2,
    macro_accuracy,
)
from harnext_eval.health.store_health import compute_store_health
from harnext_eval.probes.common import string_value
from harnext_eval.probes.gen_code_location import code_location_gold
from harnext_eval.probes.gen_multisource import _links_as_of
from harnext_eval.probes.gold import (
    GoldAuditTrail,
    GoldRequest,
    PythonGold,
    SqlGold,
    write_gold_report,
)
from harnext_eval.registry import ExperimentResult, register_experiment
from harnext_eval.report.charts import e5_pareto
from harnext_eval.stores.base import StoreHandle
from harnext_eval.stores.fake_usage import ensure_fake_fold_usage
from harnext_eval.stores.layouts import configure_store, runtime_for
from harnext_eval.types import EvalEvent, Probe, RouterRecord, SnapshotRef

CADENCES = ("W1", "W5", "W20", "W50", "W20+rules", "W20+rules+deviation")
MACRO_FAMILIES = frozenset({"extraction", "temporal", "update", "multisource", "abstention"})
PROVIDER_LAYOUTS = frozenset({"S2", "S3", "S5"})
_ACCOUNTING_PATHS = frozenset({"_meta/delivered_event_ids.jsonl", "_meta/input.json"})
_FIELD_PATTERNS = (
    re.compile(r"current\s+(.+?)\s+of\s+", re.IGNORECASE),
    re.compile(r"latest\s+(.+?)\s+of\s+", re.IGNORECASE),
    re.compile(r"what was the\s+(.+?)\s+of\s+", re.IGNORECASE),
    re.compile(r"what is the\s+(.+?)\s+of\s+", re.IGNORECASE),
)


@dataclass(frozen=True)
class CadenceSetting:
    name: str
    max_events: int
    rules: bool
    deviation: bool
    last_events: int | None = None


@dataclass(frozen=True)
class PriceTable:
    input_per_million: float
    output_per_million: float
    cache_creation_input_per_million: float | None
    cache_read_input_per_million: float | None
    model: str | None
    effective_date: str | None


@dataclass(frozen=True)
class ReplayResult:
    folds: tuple[dict[str, Any], ...]
    decisions: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class E2Result:
    outcomes: tuple[ProbeOutcome, ...]
    checks: Mapping[str, float]
    gate: pd.DataFrame
    store_arm: str


def cadence_setting(name: str) -> CadenceSetting:
    settings = {
        # W1 is a cadence, not a history-retention policy. Truncating it to the
        # last 1,000 events made frozen pre-window probes unanswerable and gave
        # its retrieve-everything floor a different population from every other
        # cadence.
        "W1": CadenceSetting("W1", 1, True, False),
        "W5": CadenceSetting("W5", 5, False, False),
        "W20": CadenceSetting("W20", 20, False, False),
        "W50": CadenceSetting("W50", 50, False, False),
        "W20+rules": CadenceSetting("W20+rules", 20, True, False),
        "W20+rules+deviation": CadenceSetting("W20+rules+deviation", 20, True, True),
    }
    try:
        return settings[name.strip()]
    except KeyError as exc:
        raise ValueError(f"unknown cadence {name!r}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )


class MeteredStoreHandle(StoreHandle):
    """Add E5 fold metadata without inventing provider usage."""

    def __init__(
        self,
        *args: Any,
        usage_path: Path,
        prices: PriceTable | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.usage_path = usage_path
        self.prices = prices or PriceTable(0.0, 0.0, None, None, "fake", "test-default")
        self.provider_required = self.layout in PROVIDER_LAYOUTS
        self.fold_rows: list[dict[str, Any]] = []

    def fold_at(
        self,
        events: list[EvalEvent],
        lane: str,
        *,
        commit_time: datetime,
        close_reason: str,
    ) -> SnapshotRef:
        prior_usage = _read_jsonl(self.usage_path)
        ref = super().fold(events, lane)
        if self.layout in {"S0", "S1", "S4"}:
            self.provider_required = ensure_fake_fold_usage(
                self.usage_path,
                len(prior_usage),
                events,
                lane,
                self.layout,
                input_per_million=self.prices.input_per_million,
                output_per_million=self.prices.output_per_million,
                price_effective_date=self.prices.effective_date or "unspecified",
            )
        usage = _read_jsonl(self.usage_path)
        added = len(usage) - len(prior_usage)
        if self.provider_required and added != 1:
            raise ValueError(
                f"{self.layout} fold {ref.sha} emitted {added} provider usage records; expected 1"
            )
        if added < 0:
            raise ValueError("provider usage log shrank during a fold")
        for row in usage[len(prior_usage) :]:
            row.update(
                {
                    "snapshot_sha": ref.sha,
                    "commit_time": commit_time.isoformat(),
                    "close_reason": close_reason,
                    "seed": runtime_for(self).seed,
                }
            )
        if added:
            _write_jsonl(self.usage_path, usage)
        self.fold_rows.append(
            {
                "fold_index": len(self.fold_rows) + 1,
                "snapshot_sha": ref.sha,
                "T_last_event": ref.T_last_event.isoformat(),
                "commit_time": commit_time.isoformat(),
                "event_ids": [event.id for event in events],
                "entities": sorted({event.subject for event in events}),
                "lane": lane,
                "close_reason": close_reason,
                "provider_usage_records": added,
            }
        )
        return ref


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
    guards = cfg.router.guards
    if setting.deviation:
        guards = guards.model_copy(
            update={"absolute_floor": max(1.0, guards.absolute_floor), "multi_window": True}
        )
    router = cfg.router.model_copy(
        update={"rules": rules, "deviation": deviation, "guards": guards}
    )
    return cfg.model_copy(update={"window": window, "router": router})


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _warmup_cutoff(corpus: CorpusHandle, probes: Sequence[Probe]) -> datetime | None:
    for key in ("router_warmup_end", "warmup_end", "probe_start"):
        parsed = _parse_datetime(corpus.meta.get(key))
        if parsed is not None:
            return parsed
    return min((probe.T for probe in probes), default=None)


def _policy_for(
    cfg: EngineConfig,
    setting: CadenceSetting,
    events: Sequence[EvalEvent],
    warmup_cutoff: datetime | None,
    seed: int,
) -> RouterPolicy:
    name = "R5" if setting.deviation else "R1"
    policy = make_policy(name, cfg.router, seed=seed)
    warmup = [event for event in events if warmup_cutoff is not None and event.time < warmup_cutoff]
    return policy.fit(warmup)


def _close_deadline(batch: Sequence[EvalEvent], cfg: EngineConfig) -> datetime:
    return min(
        batch[-1].time + timedelta(seconds=cfg.window.gap_s),
        batch[0].time + timedelta(seconds=cfg.window.max_age_s),
    )


def _event_clock_replay(
    events: Sequence[EvalEvent],
    cfg: EngineConfig,
    setting: CadenceSetting,
    store: MeteredStoreHandle,
    *,
    warmup_cutoff: datetime | None,
    seed: int,
) -> ReplayResult:
    """Causal replay using the shared R1/R5 ``RouterPolicy`` seam."""

    ordered = sorted(events, key=lambda event: (event.time, event.id))
    policy = _policy_for(cfg, setting, ordered, warmup_cutoff, seed)
    windows: dict[tuple[str, str], list[EvalEvent]] = {}
    decisions: list[dict[str, Any]] = []
    rule_negative_seen = 0
    deviation_admitted = 0

    def flush(key: tuple[str, str], commit_time: datetime, reason: str) -> None:
        pending = windows.pop(key, [])
        if pending:
            store.fold_at(pending, "batch", commit_time=commit_time, close_reason=reason)

    def flush_due(now: datetime) -> None:
        due = sorted(
            (
                (_close_deadline(batch, cfg), key)
                for key, batch in windows.items()
                if _close_deadline(batch, cfg) <= now
            ),
            key=lambda item: (item[0], item[1]),
        )
        for deadline, key in due:
            if key in windows:
                batch = windows[key]
                gap = batch[-1].time + timedelta(seconds=cfg.window.gap_s)
                reason = (
                    "gap"
                    if gap <= batch[0].time + timedelta(seconds=cfg.window.max_age_s)
                    else "max_age"
                )
                flush(key, deadline, reason)

    for event in ordered:
        flush_due(event.time)
        in_warmup = warmup_cutoff is not None and event.time < warmup_cutoff
        rule = policy.rules(event) if setting.rules else None
        score = float(policy.score(event)) if setting.deviation and not in_warmup else 0.0
        policy_features = dict(getattr(policy, "features_fired", {}))
        policy_threshold = float(getattr(policy, "threshold", math.inf))
        policy_eligible = bool(policy_features.get("eligible", False))
        deviation_candidate = False
        budget_capacity = math.floor(rule_negative_seen * cfg.router.budget_pct / 100.0)
        if rule is None:
            rule_negative_seen += 1
            budget_capacity = math.floor(rule_negative_seen * cfg.router.budget_pct / 100.0)
            deviation_candidate = (
                setting.deviation
                and not in_warmup
                and policy_eligible
                and score >= policy_threshold
                and deviation_admitted < budget_capacity
            )
            if deviation_candidate:
                deviation_admitted += 1
        lane = (
            "fast"
            if rule is not None or deviation_candidate or setting.name == "W1"
            else "batch"
        )
        record = RouterRecord(
            event_id=event.id,
            t=event.time,
            score=score,
            lane=lane,
            policy=str(getattr(policy, "name", type(policy).__name__)),
            budget_pct=cfg.router.budget_pct,
            baseline_key_used=getattr(policy, "baseline_key_used", None),
            features_fired=policy_features,
        ).model_dump(mode="json")
        record.update(
            {
                "rule": rule,
                "in_warmup": in_warmup,
                "rule_negative_seen": rule_negative_seen,
                "deviation_budget_capacity": budget_capacity,
                "deviation_admitted_so_far": deviation_admitted,
                "r5_threshold": policy_threshold if setting.deviation else None,
                "r5_eligible": policy_eligible if setting.deviation else None,
                "admission_scope": "rules-floor-plus-causal-rule-negative-budget",
            }
        )
        decisions.append(record)
        key = (event.mgtenant, event.subject)
        if lane == "fast":
            flush(key, event.time, "pre_fast")
            store.fold_at([event], "fast", commit_time=event.time, close_reason="fast")
            continue
        batch = windows.setdefault(key, [])
        batch.append(event)
        if len(batch) >= cfg.window.max_events:
            flush(key, event.time, "cap")

    while windows:
        deadline, key = min(
            ((_close_deadline(batch, cfg), key) for key, batch in windows.items()),
            key=lambda item: (item[0], item[1]),
        )
        batch = windows[key]
        gap = batch[-1].time + timedelta(seconds=cfg.window.gap_s)
        reason = (
            "gap" if gap <= batch[0].time + timedelta(seconds=cfg.window.max_age_s) else "max_age"
        )
        flush(key, deadline, reason)
    return ReplayResult(tuple(store.fold_rows), tuple(decisions))


def _changed_added_text(store: StoreHandle, sha: str) -> str:
    names = subprocess.run(
        [
            "git",
            "-C",
            str(store.worktree),
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            sha,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    paths = [path for path in names if path and path not in _ACCOUNTING_PATHS]
    if not paths:
        return ""
    diff = subprocess.run(
        [
            "git",
            "-C",
            str(store.worktree),
            "show",
            "--format=",
            "--unified=0",
            sha,
            "--",
            *paths,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return "\n".join(
        line[1:]
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def _first_mentions(
    store: StoreHandle, folds: Sequence[Mapping[str, Any]], event_ids: Iterable[str]
) -> dict[str, datetime]:
    remaining = set(event_ids)
    first: dict[str, datetime] = {}
    prior_commit: datetime | None = None
    for fold in folds:
        commit = _parse_datetime(fold["commit_time"])
        assert commit is not None
        if prior_commit is not None and commit < prior_commit:
            raise ValueError("fold commit clock regressed")
        prior_commit = commit
        added = _changed_added_text(store, str(fold["snapshot_sha"]))
        mentioned = {event_id for event_id in remaining if event_id in added}
        for event_id in mentioned:
            first[event_id] = commit
        remaining.difference_update(mentioned)
    if remaining:
        sample = ", ".join(sorted(remaining)[:5])
        raise ValueError(f"no durable snapshot diff mentions {len(remaining)} event(s): {sample}")
    return first


def _urgency_labels(
    events: Sequence[EvalEvent], corpus: CorpusHandle
) -> tuple[dict[str, bool], str, bool]:
    situations = corpus.meta.get("injected_situations")
    if isinstance(situations, list):
        positive = {
            str(row["event_id"])
            for row in situations
            if isinstance(row, Mapping) and row.get("event_id") is not None
        }
        return ({event.id: event.id in positive for event in events}, "constructed-injected", True)
    labels_path = corpus.meta.get("e1_labels_path")
    if labels_path:
        path = Path(str(labels_path))
        rows = (
            _read_jsonl(path)
            if path.suffix == ".jsonl"
            else pd.read_csv(path).to_dict("records")
        )
        labels = {
            str(row["event_id"]): bool(float(row.get("p_urgent", row.get("label", 0))) >= 0.5)
            for row in rows
        }
        missing = {event.id for event in events}.difference(labels)
        if missing:
            raise ValueError(f"frozen E1 labels omit {len(missing)} replay events")
        return labels, "revealed-e1-frozen", True
    result = build_labels(list(events))
    return (
        {event.id: bool(result.probabilities.loc[event.id] >= 0.5) for event in events},
        "revealed-e1-derived",
        False,
    )


def _freshness_rows(
    cadence: str,
    seed: int,
    events: Sequence[EvalEvent],
    first_mentions: Mapping[str, datetime],
    labels: Mapping[str, bool],
    label_provenance: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        commit = first_mentions[event.id]
        delay = (commit - event.time).total_seconds()
        if delay < 0:
            raise ValueError(f"negative freshness for {event.id}: {delay}")
        rows.append(
            {
                "cadence": cadence,
                "seed": seed,
                "event_id": event.id,
                "entity": event.subject,
                "event_time": event.time.isoformat(),
                "first_commit_time": commit.isoformat(),
                "freshness_s": delay,
                "urgent": bool(labels[event.id]),
                "urgency_provenance": label_provenance,
            }
        )
    if len(rows) != len(events):
        raise ValueError("freshness row count does not equal replay event count")
    return rows


def _field_for_probe(probe: Probe) -> str | None:
    for pattern in _FIELD_PATTERNS:
        match = pattern.search(probe.question)
        if match:
            return match.group(1).strip()
    return None


def _temporal_target(probe: Probe) -> datetime:
    if probe.family != "temporal":
        return probe.T
    raw = probe.question.rsplit(" as of ", 1)[-1].rstrip("?. ")
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"cannot parse temporal cutoff for probe {probe.probe_id}") from exc


def _rederive_probes(
    probes: Sequence[Probe],
    events: Sequence[EvalEvent],
    *,
    report_path: Path | None = None,
) -> list[Probe]:
    """Keep frozen cutoffs and independently rederive every probe's gold at that cutoff."""

    python = PythonGold(events)
    derived: list[Probe] = []
    audit = GoldAuditTrail(source="normalised-smoke-adapter")
    with SqlGold(events) as sql:
        for probe in probes:
            if probe.family in {"extraction", "temporal", "update"}:
                field = _field_for_probe(probe)
                if field is None:
                    raise ValueError(f"cannot identify gold field for probe {probe.probe_id}")
                target = _temporal_target(probe)
                py_value = python.field_value(probe.entity, field, target)
                history = python.transitions(probe.entity, field, target)
                if py_value is None or not history:
                    raise ValueError(f"probe {probe.probe_id} has no gold at its frozen cutoff")
                if all(item.source_kind == "jira" for item in history):
                    sql_value = sql.field_value(probe.entity, field, target)
                    audit.compare(GoldRequest(probe.entity, field, target), py_value, sql_value)
                superseded = probe.superseded_values
                if probe.family == "update":
                    values = [item.old_value for item in history[:1]] + [
                        item.new_value for item in history[:-1]
                    ]
                    superseded = sorted(
                        {str(value) for value in values if value is not None and value != py_value}
                    )
                derived.append(
                    probe.model_copy(
                        update={
                            "gold": string_value(py_value),
                            "superseded_values": superseded,
                            "source_event_ids": [item.event_id for item in history],
                        }
                    )
                )
            elif probe.family == "multisource":
                links, source_ids = _links_as_of(list(events), probe.entity, probe.T)
                derived.append(
                    probe.model_copy(update={"gold": links, "source_event_ids": source_ids})
                )
            elif probe.family == "code_location":
                gold, source_ids = code_location_gold(list(events), probe.entity, probe.T)
                derived.append(
                    probe.model_copy(update={"gold": gold, "source_event_ids": source_ids})
                )
            else:
                derived.append(probe.model_copy(update={"gold": "UNKNOWN"}))
    if report_path is not None:
        write_gold_report(audit, report_path)
    audit.require_valid(evidentiary=False)
    return derived


def _load_probes(path: Path | None) -> list[Probe]:
    if path is None or not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return [Probe.model_validate_json(line) for line in source if line.strip()]


def _run_e2(
    store: StoreHandle,
    probes: Sequence[Probe],
    cfg: EngineConfig,
    events: Sequence[EvalEvent],
    out_dir: Path,
    seed: int,
    validation_audit: Mapping[str, float],
) -> E2Result:
    store_arm = "A4" if store.layout == "S3" else store.layout
    result, outcomes = evaluate_e2(
        cfg=cfg,
        probes=probes,
        events=events,
        out_dir=out_dir,
        seed=seed,
        store=store,
        arms=("A0", store_arm, "retrieve_everything"),
        validation_audit=validation_audit,
    )
    gate = pd.read_csv(out_dir / "gate.csv")
    checks = {
        key: float(value)
        for key, value in result.metrics.items()
        if key.startswith("checks.")
        or key.startswith("gate_")
        or key.startswith("floor_")
        or key.startswith("prior_")
    }
    return E2Result(tuple(outcomes), checks, gate, store_arm)


def _fixed_macro(outcomes: Sequence[ProbeOutcome]) -> float:
    present = {
        "multisource" if row.probe.family == "code_location" else row.probe.family
        for row in outcomes
    }
    if not MACRO_FAMILIES.issubset(present):
        return math.nan
    return macro_accuracy(outcomes)


def _shared_health(store: StoreHandle) -> dict[str, float]:
    refs = store._snapshots()  # noqa: SLF001 - StoreHandle has no public snapshot iterator
    if not refs:
        return {"files_per_entity": math.nan, "dup_rate": math.nan}
    checkout = store.materialise(refs[-1])
    try:
        result = compute_store_health(checkout)
    finally:
        shutil.rmtree(checkout, ignore_errors=True)
    return {
        "files_per_entity": float(result.get("files_per_entity", 0.0)),
        "dup_rate": float(result.get("near_duplicate_fact_rate", 0.0)),
    }


def _prices(cfg: EngineConfig, meta: Mapping[str, Any]) -> PriceTable:
    del meta
    configured = cfg.prices
    if configured is None:
        raise ValueError("E5 requires a frozen 'prices' section in ExperimentConfig")
    candidate = (
        dict(configured)
        if isinstance(configured, Mapping)
        else configured.model_dump(mode="python")
    )

    def required(*names: str) -> float:
        for name in names:
            if name in candidate:
                value = float(candidate[name])
                if value < 0:
                    raise ValueError(f"price {name} must be non-negative")
                return value
        raise ValueError(f"E5 prices are missing {names[0]}")

    def optional(*names: str) -> float | None:
        for name in names:
            if name in candidate and candidate[name] is not None:
                value = float(candidate[name])
                if value < 0:
                    raise ValueError(f"price {name} must be non-negative")
                return value
        return None

    return PriceTable(
        input_per_million=required("input_per_million", "input_per_1m"),
        output_per_million=required("output_per_million", "output_per_1m"),
        cache_creation_input_per_million=optional(
            "cache_creation_input_per_million", "cache_write_per_million"
        ),
        cache_read_input_per_million=optional(
            "cache_read_input_per_million", "cache_read_per_million"
        ),
        model=str(candidate["model"]) if candidate.get("model") is not None else None,
        effective_date=(
            str(candidate["effective_date"])
            if candidate.get("effective_date") is not None
            else None
        ),
    )


def _tokens(record: Mapping[str, Any]) -> dict[str, int]:
    chain: list[Mapping[str, Any]] = [record]
    current = record
    while True:
        nested = current.get("usage")
        if not isinstance(nested, Mapping):
            break
        current = nested
        chain.insert(0, current)

    def value(label: str, *names: str, required: bool = False) -> int:
        for usage in chain:
            for name in names:
                if name in usage:
                    number = int(usage[name])
                    if number < 0:
                        raise ValueError(f"usage token count {name} must be non-negative")
                    return number
        if required:
            raise ValueError(f"provider usage record is missing {label}")
        return 0

    return {
        "input_tokens": value(
            "input_tokens", "input_tokens", "prompt_tokens", "tokens_in", required=True
        ),
        "output_tokens": value(
            "output_tokens", "output_tokens", "completion_tokens", "tokens_out", required=True
        ),
        "cache_creation_input_tokens": value(
            "cache_creation_input_tokens", "cache_creation_input_tokens"
        ),
        "cache_read_input_tokens": value(
            "cache_read_input_tokens", "cache_read_input_tokens"
        ),
    }


def _record_cost(record: Mapping[str, Any], prices: PriceTable) -> tuple[float, dict[str, int]]:
    if str(record.get("status", "success")) != "success":
        raise ValueError("failed provider usage record cannot be included in E5 cost")
    if not isinstance(record.get("model"), str) or not str(record["model"]).strip():
        raise ValueError("provider usage record is missing its model")
    if prices.model and record.get("model") != prices.model:
        raise ValueError(
            f"usage model {record.get('model')} does not match price model {prices.model}"
        )
    raw_cache_creation = int(record.get("cache_creation_input_tokens", 0))
    raw_cache_read = int(record.get("cache_read_input_tokens", 0))
    if raw_cache_creation and prices.cache_creation_input_per_million is None:
        raise ValueError("cache-creation tokens require a frozen cache-creation price")
    if raw_cache_read and prices.cache_read_input_per_million is None:
        raise ValueError("cache-read tokens require a frozen cache-read price")
    tokens = _tokens(record)
    if tokens["cache_creation_input_tokens"] and prices.cache_creation_input_per_million is None:
        raise ValueError("cache-creation tokens require a frozen cache-creation price")
    if tokens["cache_read_input_tokens"] and prices.cache_read_input_per_million is None:
        raise ValueError("cache-read tokens require a frozen cache-read price")
    cost = tokens["input_tokens"] * prices.input_per_million
    cost += tokens["output_tokens"] * prices.output_per_million
    cost += tokens["cache_creation_input_tokens"] * (
        prices.cache_creation_input_per_million or 0.0
    )
    cost += tokens["cache_read_input_tokens"] * (prices.cache_read_input_per_million or 0.0)
    return cost / 1_000_000, tokens


def _usage_summary(
    records: Sequence[Mapping[str, Any]], prices: PriceTable
) -> tuple[dict[str, float], dict[str, float]]:
    totals: defaultdict[str, float] = defaultdict(float)
    event_cost: defaultdict[str, float] = defaultdict(float)
    for record in records:
        event_ids = record.get("event_ids")
        if not isinstance(event_ids, list) or not event_ids:
            raise ValueError("provider usage record must list the fold event_ids")
        cost, tokens = _record_cost(record, prices)
        totals["cost_usd"] += cost
        for key, token_count in tokens.items():
            totals[key] += token_count
        latency = record.get("latency_s", record.get("duration_s"))
        if latency is not None:
            totals["builder_latency_s"] += float(latency)
        share = cost / len(event_ids)
        for event_id in event_ids:
            event_cost[str(event_id)] += share
    return dict(totals), dict(event_cost)


def _quantile(values: Iterable[float], q: float) -> float:
    items = list(values)
    return float(np.quantile(items, q)) if items else math.nan


def _ratio_stat(
    rows: NDArray[np.integer[Any]],
    numerator_cost: np.ndarray,
    numerator_events: np.ndarray,
    denominator_cost: np.ndarray,
    denominator_events: np.ndarray,
) -> float:
    num = numerator_cost[list(rows)].sum() / numerator_events[list(rows)].sum()
    den = denominator_cost[list(rows)].sum() / denominator_events[list(rows)].sum()
    return float(num / den) if den > 0 else math.nan


def _paired_ratio_bca(
    numerator_cost: Sequence[float],
    numerator_events: Sequence[int],
    denominator_cost: Sequence[float],
    denominator_events: Sequence[int],
    *,
    n_resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    arrays = [
        np.asarray(numerator_cost, dtype=float),
        np.asarray(numerator_events, dtype=float),
        np.asarray(denominator_cost, dtype=float),
        np.asarray(denominator_events, dtype=float),
    ]
    if len(arrays[0]) < 2 or any(len(array) != len(arrays[0]) for array in arrays):
        return math.nan, math.nan, math.nan
    rows = np.arange(len(arrays[0]))
    effect = _ratio_stat(rows, *arrays)
    if not math.isfinite(effect):
        return effect, math.nan, math.nan
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [_ratio_stat(rng.integers(0, len(rows), len(rows)), *arrays) for _ in range(n_resamples)]
    )
    bootstrap = bootstrap[np.isfinite(bootstrap)]
    if len(bootstrap) < 100:
        return effect, math.nan, math.nan
    less = (
        np.count_nonzero(bootstrap < effect) + 0.5 * np.count_nonzero(bootstrap == effect)
    ) / len(bootstrap)
    z0 = float(norm.ppf(np.clip(less, 0.5 / len(bootstrap), 1 - 0.5 / len(bootstrap))))
    jackknife = np.asarray(
        [_ratio_stat(rows[rows != index], *arrays) for index in rows], dtype=float
    )
    centre = float(np.mean(jackknife)) - jackknife
    denominator = 6 * float(np.sum(centre**2)) ** 1.5
    acceleration = 0.0 if denominator == 0 else float(np.sum(centre**3)) / denominator
    adjusted: list[float] = []
    for z_alpha in norm.ppf([0.025, 0.975]):
        shifted = z0 + z_alpha
        divisor = 1 - acceleration * shifted
        adjusted.append(float(norm.cdf(z0 + shifted / divisor)))
    low, high = np.quantile(bootstrap, np.clip(np.sort(adjusted), 0, 1))
    return effect, float(low), float(high)


def _fresh_summary(rows: Sequence[Mapping[str, Any]], urgent: bool) -> tuple[float, float]:
    values = [float(row["freshness_s"]) for row in rows if bool(row["urgent"]) is urgent]
    return _quantile(values, 0.50), _quantile(values, 0.95)


def _equal_accuracy(ci_high: float) -> bool:
    """W1 is not significantly better exactly when the cadence CI reaches zero."""

    return math.isfinite(ci_high) and ci_high >= 0


def _nontrivial_accuracy(values: Iterable[float]) -> bool:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return len(finite) >= 2 and len({round(value, 12) for value in finite}) > 1


def _claim_profile(corpus: CorpusHandle, cfg: EngineConfig, labels_frozen: bool) -> bool:
    return bool(
        not corpus.meta.get("smoke", False)
        and cfg.builder.harness != "fake"
        and labels_frozen
        and corpus.meta.get("prereg_ref")
        and corpus.meta.get("g4_passed")
        and corpus.meta.get("replay_hash")
        and corpus.meta.get("probe_hash")
    )


def run_cadences(
    cfg: EngineConfig,
    corpus: CorpusHandle,
    out_dir: Path,
    seed: int,
    *,
    cadences: Iterable[str] = CADENCES,
    probes: Iterable[Probe] | None = None,
) -> ExperimentResult:
    """Build and measure each E5 cadence on an identical frozen replay."""

    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"E5 output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_events = sorted(corpus.events(), key=lambda event: (event.time, event.id))
    if not all_events:
        raise ValueError("E5 requires a non-empty replay")
    frozen_probes = list(probes) if probes is not None else _load_probes(corpus.probes_path)
    probe_list = (
        _rederive_probes(
            frozen_probes,
            all_events,
            report_path=out_dir / "gold-agreement.json",
        )
        if frozen_probes
        else []
    )
    prices = _prices(cfg, corpus.meta)
    validation_audit = {
        "dual_gold_agreement": float(corpus.meta.get("dual_gold_agreement", 0.0)),
        "pilot_kappa": float(corpus.meta.get("pilot_kappa", 0.0)),
        "claim_disagreement": float(corpus.meta.get("claim_disagreement", 1.0)),
    }
    labels, label_provenance, labels_frozen = _urgency_labels(all_events, corpus)
    warmup_cutoff = _warmup_cutoff(corpus, probe_list)
    cadence_names = tuple(cadences)
    cost_rows: list[dict[str, Any]] = []
    freshness_rows: list[dict[str, Any]] = []
    gate_frames: list[pd.DataFrame] = []
    e2_by_cadence: dict[str, E2Result] = {}
    e2_check_rows: list[dict[str, Any]] = []
    entity_economics: dict[str, dict[str, dict[str, float]]] = {}

    for cadence in cadence_names:
        setting = cadence_setting(cadence)
        events = all_events[-setting.last_events :] if setting.last_events else list(all_events)
        cadence_cfg = _cadence_cfg(cfg, setting)
        safe_name = cadence.replace("+", "_")
        cadence_dir = out_dir / "stores" / safe_name / f"seed-{seed}"
        if cadence_dir.exists():
            raise FileExistsError(f"cadence store already exists: {cadence_dir}")
        usage_path = cadence_dir / "usage.jsonl"
        store = MeteredStoreHandle(
            cadence_cfg.store.layout,
            f"e5-{safe_name}-{seed}",
            cadence_dir,
            usage_path=usage_path,
            prices=prices,
        )
        configure_store(
            store,
            harness=cadence_cfg.builder.harness,
            model=cadence_cfg.builder.model,
            seed=seed,
        )
        replay = _event_clock_replay(
            events,
            cadence_cfg,
            setting,
            store,
            warmup_cutoff=warmup_cutoff,
            seed=seed,
        )
        _write_jsonl(cadence_dir / "folds.jsonl", replay.folds)
        _write_jsonl(cadence_dir / "decisions.jsonl", replay.decisions)
        usage = _read_jsonl(usage_path)
        usage_totals, event_cost = _usage_summary(usage, prices) if usage else ({}, {})
        first_mentions = _first_mentions(store, replay.folds, (event.id for event in events))
        freshness = _freshness_rows(
            cadence,
            seed,
            events,
            first_mentions,
            labels,
            label_provenance,
        )
        freshness_rows.extend(freshness)
        if probe_list:
            e2_result = _run_e2(
                store,
                probe_list,
                cadence_cfg,
                events,
                cadence_dir / "e2-frozen",
                seed,
                validation_audit,
            )
        else:
            e2_result = E2Result((), {}, pd.DataFrame(), "")
        e2_by_cadence[cadence] = e2_result
        e2_check_rows.extend(
            {
                "cadence": cadence,
                "check": key.removeprefix("checks."),
                "value": value,
                        "status": (
                    "not_applicable_in_smoke"
                    if corpus.meta.get("smoke")
                    and key
                    in {
                        "checks.budget_within_10_pct",
                        "checks.budget_fill_applicable",
                        "checks.pilot_kappa_ge_0_8",
                        "checks.claim_disagreement_le_0_02",
                        "checks.non_evidentiary_smoke",
                    }
                    else "pass"
                    if value == 1.0
                    else "fail"
                        ),
                        "reason": (
                            None
                            if value == 1.0
                            else (
                                "not applicable to the reduced deterministic smoke profile"
                                if corpus.meta.get("smoke")
                                and key
                                in {
                                    "checks.budget_within_10_pct",
                                    "checks.budget_fill_applicable",
                                    "checks.pilot_kappa_ge_0_8",
                                    "checks.claim_disagreement_le_0_02",
                                    "checks.non_evidentiary_smoke",
                                }
                                else f"{key.removeprefix('checks.')} observed {value!r}"
                            )
                        ),
            }
            for key, value in sorted(e2_result.checks.items())
            if key.startswith("checks.")
        )
        if not e2_result.gate.empty:
            gate = e2_result.gate.copy()
            gate.insert(0, "cadence", cadence)
            gate_frames.append(gate)
        health = _shared_health(store)
        expected_provider_runs = len(replay.folds) if store.provider_required else 0
        successful_usage = sum(str(row.get("status", "success")) == "success" for row in usage)
        authentic_usage = bool(usage) and all(
            row.get("usage_kind") != "deterministic_projection" for row in usage
        )
        rule_count = sum(row.get("rule") is not None for row in replay.decisions)
        deviation_count = sum(
            row["lane"] == "fast" and row.get("rule") is None and setting.deviation
            for row in replay.decisions
        )
        entity_rows: defaultdict[str, dict[str, float]] = defaultdict(
            lambda: {"cost": 0.0, "events": 0.0}
        )
        for event in events:
            entity_rows[event.subject]["events"] += 1
            entity_rows[event.subject]["cost"] += event_cost.get(event.id, 0.0)
        entity_economics[cadence] = dict(entity_rows)
        urgent_p50, urgent_p95 = _fresh_summary(freshness, True)
        routine_p50, routine_p95 = _fresh_summary(freshness, False)
        event_count = len(events)
        cost = float(usage_totals.get("cost_usd", 0.0))
        store_rows = [
            row for row in e2_result.outcomes if row.answer.arm == e2_result.store_arm
        ]
        cost_rows.append(
            {
                "cadence": cadence,
                "seed": seed,
                "events": event_count,
                "classifier_folds": len(replay.folds),
                "classifier_batch_closes": sum(row["lane"] == "batch" for row in replay.folds),
                "classifier_fast_runs": sum(row["lane"] == "fast" for row in replay.folds),
                "builder_runs": successful_usage,
                "expected_builder_runs": expected_provider_runs,
                "input_tokens": int(usage_totals.get("input_tokens", 0)),
                "output_tokens": int(usage_totals.get("output_tokens", 0)),
                "cache_creation_input_tokens": int(
                    usage_totals.get("cache_creation_input_tokens", 0)
                ),
                "cache_read_input_tokens": int(usage_totals.get("cache_read_input_tokens", 0)),
                "cost_usd": cost,
                "cost_1k": cost / (event_count / 1_000),
                "runs_1k": len(replay.folds) / (event_count / 1_000),
                "macro_acc": _fixed_macro(store_rows),
                "reader_tokens": (
                    float(np.mean([row.answer.tokens_read for row in store_rows]))
                    if store_rows
                    else math.nan
                ),
                "reader_latency_s": (
                    float(np.mean([row.answer.latency_s for row in store_rows]))
                    if store_rows
                    else math.nan
                ),
                "builder_latency_s": float(usage_totals.get("builder_latency_s", 0.0)),
                "files_per_entity": health["files_per_entity"],
                "dup_rate": health["dup_rate"],
                "fresh_urgent_p50_s": urgent_p50,
                "fresh_urgent_p95_s": urgent_p95,
                "fresh_routine_p50_s": routine_p50,
                "fresh_routine_p95_s": routine_p95,
                "freshness_rows_complete": len(freshness) == event_count,
                "provider_usage_required": store.provider_required,
                "cost_from_usage": successful_usage == expected_provider_runs,
                "authentic_provider_usage": authentic_usage,
                "builder_count_matches": successful_usage == expected_provider_runs,
                "rule_admissions": rule_count,
                "deviation_admissions": deviation_count,
                "realized_fast_share": float(
                    np.mean([row["lane"] == "fast" for row in replay.decisions])
                ),
                "price_model": prices.model,
                "price_effective_date": prices.effective_date,
            }
        )

    common_probe_ids: set[str] = {probe.probe_id for probe in probe_list}
    for e2_result in e2_by_cadence.values():
        common_probe_ids.intersection_update(
            row.probe.probe_id
            for row in e2_result.outcomes
            if row.answer.arm == e2_result.store_arm
        )
    by_cadence = {str(row["cadence"]): row for row in cost_rows}
    per_probe: dict[str, dict[str, ProbeOutcome]] = {}
    for cadence, e2_result in e2_by_cadence.items():
        rows = {
            row.probe.probe_id: row
            for row in e2_result.outcomes
            if row.answer.arm == e2_result.store_arm
            and row.probe.probe_id in common_probe_ids
        }
        per_probe[cadence] = rows
        by_cadence[cadence]["macro_acc"] = _fixed_macro(list(rows.values()))

    w1 = by_cadence.get("W1")
    rules = by_cadence.get("W20+rules")
    accuracy_delta = accuracy_low = accuracy_high = math.nan
    if w1 and rules and common_probe_ids:
        ordered_ids = sorted(common_probe_ids)
        left = [per_probe["W20+rules"][probe_id] for probe_id in ordered_ids]
        right = [per_probe["W1"][probe_id] for probe_id in ordered_ids]
        paired_outcomes = [
            replace(row, answer=row.answer.model_copy(update={"arm": "W20+rules"}))
            for row in left
        ] + [
            replace(row, answer=row.answer.model_copy(update={"arm": "W1"}))
            for row in right
        ]
        accuracy_contrast = _paired_contrast(
            paired_outcomes, "W20+rules", "W1", seed
        )
        if not accuracy_contrast.empty:
            accuracy_row = accuracy_contrast.iloc[0]
            accuracy_delta = float(accuracy_row["delta"])
            accuracy_low = float(accuracy_row["ci_low"])
            accuracy_high = float(accuracy_row["ci_high"])
    equal_accuracy = _equal_accuracy(accuracy_high)

    ratio = ratio_low = ratio_high = math.nan
    if w1 and rules:
        w1_cost_1k = float(w1["cost_1k"])
        rules_cost_1k = float(rules["cost_1k"])
        ratio = rules_cost_1k / w1_cost_1k if w1_cost_1k > 0 else math.nan
        entities = sorted(
            set(entity_economics["W1"]).intersection(entity_economics["W20+rules"])
        )
        _, ratio_low, ratio_high = _paired_ratio_bca(
            [entity_economics["W20+rules"][entity]["cost"] for entity in entities],
            [int(entity_economics["W20+rules"][entity]["events"]) for entity in entities],
            [entity_economics["W1"][entity]["cost"] for entity in entities],
            [int(entity_economics["W1"][entity]["events"]) for entity in entities],
            n_resamples=BOOTSTRAP_RESAMPLES,
            seed=seed,
        )

    gate_frame = pd.concat(gate_frames, ignore_index=True) if gate_frames else pd.DataFrame()
    leakage_ok = bool(gate_frame.empty or (gate_frame["result"] == "PASS").all())
    smoke = bool(corpus.meta.get("smoke"))
    common_e2_requirements = (
        "checks.floor_retrieve_everything_ge_0_9",
        "checks.prior_leq_0_3",
        "checks.leakage_gate_100_pct",
        "checks.primary_budget_is_8000",
        "checks.exact_rerun_identical",
        "checks.dual_gold_agreement_ge_0_98",
    )
    evidentiary_e2_requirements = (
        "checks.budget_within_10_pct",
        "checks.budget_fill_applicable",
        "checks.pilot_kappa_ge_0_8",
        "checks.claim_disagreement_le_0_02",
    )
    e2_checks_ok = bool(
        e2_by_cadence
        and all(
            all(e2_result.checks.get(key) == 1.0 for key in common_e2_requirements)
            and (
                smoke
                or all(
                    e2_result.checks.get(key) == 1.0
                    for key in evidentiary_e2_requirements
                )
            )
            for e2_result in e2_by_cadence.values()
        )
    )
    population_families = {
        "multisource" if probe.family == "code_location" else probe.family
        for probe in probe_list
    }
    family_population_ok = MACRO_FAMILIES.issubset(population_families)
    nontrivial = _nontrivial_accuracy(float(row["macro_acc"]) for row in cost_rows)
    cost_accounting_ok = all(bool(row["cost_from_usage"]) for row in cost_rows)
    usage_ok = cost_accounting_ok and all(
        bool(row["authentic_provider_usage"]) for row in cost_rows
    )
    counts_ok = all(bool(row["builder_count_matches"]) for row in cost_rows)
    freshness_ok = all(bool(row["freshness_rows_complete"]) for row in cost_rows)
    w1_p95 = float(w1["fresh_routine_p95_s"]) if w1 else math.nan
    passthrough_ok = math.isfinite(w1_p95) and w1_p95 <= 1e-9
    claim_profile = _claim_profile(corpus, cfg, labels_frozen)
    valid_primary = bool(
        claim_profile
        and equal_accuracy
        and math.isfinite(ratio)
        and math.isfinite(ratio_low)
        and math.isfinite(ratio_high)
        and usage_ok
        and counts_ok
        and freshness_ok
        and leakage_ok
        and e2_checks_ok
        and family_population_ok
        and nontrivial
        and passthrough_ok
    )
    invalid_reasons = [
        name
        for name, passed in {
            "claim_profile": claim_profile,
            "equal_accuracy_ci": equal_accuracy,
            "ratio_ci": all(math.isfinite(value) for value in (ratio, ratio_low, ratio_high)),
            "usage_provenance": usage_ok,
            "builder_counts": counts_ok,
            "freshness_complete": freshness_ok,
            "leakage": leakage_ok,
            "e2_checks": e2_checks_ok,
            "probe_families": family_population_ok,
            "nontrivial_fake_path": nontrivial,
            "w1_passthrough": passthrough_ok,
        }.items()
        if not passed
    ]
    reported_ratio = ratio if valid_primary else math.nan
    price_of_freshness = bool(
        claim_profile and math.isfinite(accuracy_high) and accuracy_high < 0
    )
    price_detail = {
        "w1_minus_rules_accuracy": -accuracy_delta,
        "w1_minus_rules_cost_1k": (
            float(w1["cost_1k"]) - float(rules["cost_1k"]) if w1 and rules else math.nan
        ),
        "w1_minus_rules_urgent_p95_s": (
            float(w1["fresh_urgent_p95_s"]) - float(rules["fresh_urgent_p95_s"])
            if w1 and rules
            else math.nan
        ),
    }

    cost_frame = pd.DataFrame(cost_rows)
    freshness_frame = pd.DataFrame(freshness_rows)
    pareto_frame = cost_frame[
        [
            "cadence",
            "seed",
            "cost_1k",
            "macro_acc",
            "fresh_urgent_p95_s",
            "fresh_routine_p95_s",
            "input_tokens",
            "output_tokens",
            "reader_latency_s",
            "builder_latency_s",
        ]
    ].copy()
    cost_path = out_dir / "cost.csv"
    freshness_path = out_dir / "freshness.csv"
    pareto_path = out_dir / "pareto.csv"
    gate_path = out_dir / "gate.csv"
    e2_checks_path = out_dir / "e2_checks.csv"
    cost_frame.to_csv(cost_path, index=False)
    freshness_frame.to_csv(freshness_path, index=False)
    pareto_frame.to_csv(pareto_path, index=False)
    gate_frame.to_csv(gate_path, index=False)
    e2_checks_frame = pd.DataFrame(e2_check_rows)
    e2_checks_frame.to_csv(e2_checks_path, index=False)
    chart_data = pd.DataFrame(pareto_frame).rename(
        columns={
            "cost_1k": "cost",
            "macro_acc": "acc",
            "fresh_urgent_p95_s": "freshness",
        }
    )
    pareto_png = e5_pareto(chart_data, out_dir)
    metrics = {
        "primary.cost_ratio_w20_rules_to_w1": reported_ratio,
        "primary.cost_ratio_ci_low": ratio_low if valid_primary else math.nan,
        "primary.cost_ratio_ci_high": ratio_high if valid_primary else math.nan,
        "diagnostic.cost_ratio_w20_rules_to_w1": ratio,
        "diagnostic.cost_ratio_ci_low": ratio_low,
        "diagnostic.cost_ratio_ci_high": ratio_high,
        "accuracy_delta_w20_rules_minus_w1": accuracy_delta,
        "accuracy_delta_ci_low": accuracy_low,
        "accuracy_delta_ci_high": accuracy_high,
        "checks.valid_primary": float(valid_primary),
        "checks.equal_accuracy": float(equal_accuracy),
        "checks.cost_from_usage": float(cost_accounting_ok),
        "checks.authentic_provider_usage": float(usage_ok),
        "checks.builder_run_count": float(counts_ok),
        "checks.freshness_complete": float(freshness_ok),
        "checks.leakage_gate": float(leakage_ok),
        "checks.shared_e2": float(e2_checks_ok),
        "checks.fixed_probe_families": float(family_population_ok),
        "checks.nontrivial_cadence_scores": float(nontrivial),
        "checks.w1_passthrough_p95_approx_zero": float(passthrough_ok),
        "checks.labels_frozen": float(labels_frozen),
        "checks.claim_profile": float(claim_profile),
    }
    check_details: dict[str, dict[str, Any]] = {}
    failed_shared_e2 = [
        {
            "cadence": str(row["cadence"]),
            "check": str(row["check"]),
            "value": float(row["value"]),
        }
        for row in e2_check_rows
        if row["status"] == "fail"
    ]
    check_details["shared_e2"] = {
        "passed": e2_checks_ok,
        "value": {
            "cadences": sorted(e2_by_cadence),
            "failed_checks": failed_shared_e2,
        },
        "reason": (
            "all applicable shared E2 cadence checks passed"
            if e2_checks_ok
            else "shared E2 failed: "
            + "; ".join(
                f"{row['cadence']}.{row['check']}={row['value']}"
                for row in failed_shared_e2
            )
        ),
    }
    if corpus.meta.get("smoke"):
        smoke_reasons = {
            "valid_primary": "the tiny deterministic smoke validates cadence plumbing, not the preregistered equal-accuracy cost claim",
            "equal_accuracy": "the smoke probe population is too small for the paired equal-accuracy confidence interval",
            "nontrivial_cadence_scores": "FakeLLM smoke is deterministic and is not intended to establish cadence accuracy differences",
            "claim_profile": "the E5 primary claim requires the non-smoke Corpus R-H1/S evidentiary profile",
            "authentic_provider_usage": "offline projections are priced from config but are not authentic provider calls",
        }
        check_details.update(
            {
                name: {
                    "passed": None,
                    "value": "not-applicable-in-smoke",
                    "reason": reason,
                }
                for name, reason in smoke_reasons.items()
            }
        )
    return ExperimentResult(
        name="e5",
        metrics=metrics,
        tables={
            "cost": cost_frame,
            "freshness": freshness_frame,
            "pareto": pd.DataFrame(pareto_frame),
            "gate": gate_frame,
            "e2_checks": e2_checks_frame,
        },
        artifacts=[
            cost_path,
            freshness_path,
            pareto_path,
            gate_path,
            e2_checks_path,
            pareto_png,
        ],
        primary={
            "cost_ratio_w20_rules_to_w1": ratio if valid_primary else None,
            "cost_ratio_bca_ci": [ratio_low, ratio_high] if valid_primary else None,
            "accuracy_delta_bca_ci": (
                [accuracy_low, accuracy_high] if math.isfinite(accuracy_low) else None
            ),
            "reported_at_equal_accuracy": valid_primary,
            "valid_primary": valid_primary,
            "invalid_reasons": invalid_reasons,
            "unavailable_reason": (
                None
                if valid_primary
                else "primary suppressed until equal-accuracy and all CI/validity gates pass: "
                + ", ".join(invalid_reasons)
            ),
            "diagnostic_ungated_cost_ratio": ratio,
            "price_of_freshness": price_of_freshness,
            "price_of_freshness_detail": price_detail if price_of_freshness else None,
            "evidence_status": "claim-eligible" if claim_profile else "plumbing-only",
        },
        check_details=check_details,
    )


class E5Experiment:
    name = "e5"

    def run(
        self, cfg: EngineConfig, corpus: CorpusHandle, out_dir: Path, seed: int
    ) -> ExperimentResult:
        cadences = (
            ("W1", "W20+rules", "W20+rules+deviation")
            if corpus.meta.get("smoke")
            else CADENCES
        )
        return run_cadences(cfg, corpus, out_dir, seed, cadences=cadences)

    def chart(self, result: ExperimentResult, out_dir: Path) -> list[Path]:
        table = result.tables["pareto"]
        if table.empty:
            return []
        chart_data = table.rename(
            columns={
                "cost_1k": "cost",
                "macro_acc": "acc",
                "fresh_urgent_p95_s": "freshness",
            }
        )
        return [e5_pareto(chart_data, out_dir)]


register_experiment(E5Experiment())

__all__ = [
    "CADENCES",
    "E5Experiment",
    "MeteredStoreHandle",
    "cadence_setting",
    "run_cadences",
]
