"""Store-organisation ablation from docs/evaluation-spec.md §7 E3 and §8."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from harnext_eval.agents.reader import Material, answer, truncate_to_tokens
from harnext_eval.config import EngineConfig
from harnext_eval.corpus import CorpusHandle
from harnext_eval.e2.arms import build_arm
from harnext_eval.e2.run import ProbeOutcome, evaluate_e2, grade_answer, load_probes
from harnext_eval.health.store_health import compute_store_health
from harnext_eval.probes.common import (
    changed_files,
    is_formatting_only,
    is_merged_pr,
    issue_keys_for_pr,
    module_for_file,
    string_value,
)
from harnext_eval.probes.gold import PythonGold
from harnext_eval.providers.embeddings import EmbeddingsProvider
from harnext_eval.providers.llm import LLMProvider
from harnext_eval.providers.tokenizer import count_tokens
from harnext_eval.registry import ExperimentResult, register_experiment
from harnext_eval.report.charts import e3_curve, erosion
from harnext_eval.stats.stats import mcnemar_test
from harnext_eval.stores.base import StoreHandle
from harnext_eval.stores.vector_index import VectorIndex
from harnext_eval.types import EvalEvent, Probe, SnapshotRef

READ_BUDGETS = (2_000, 8_000, 32_000)
BOOTSTRAP_RESAMPLES = 10_000
EROSION_PANEL_SIZE = 60
_CHECKPOINT_WEEKS = (1, 2, 4, 8)
_MACRO_FAMILIES = ("extraction", "temporal", "update", "multisource", "abstention")
_DELIVERED_PATH = "_meta/delivered_event_ids.jsonl"
_INPUT_META_PATH = "_meta/input.json"
_WORD_RE = re.compile(r"[\w.-]+", re.UNICODE)
_ISSUE_RE = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")


@dataclass(frozen=True, slots=True)
class StoreCondition:
    """Immutable E3 identity layered over the shared ``StoreHandle`` contract."""

    store: StoreHandle
    seed: int | None = None
    tier: str = "baseline"
    replay_hash: str | None = None
    model: str | None = None
    label: str | None = None

    @property
    def layout(self) -> str:
        return str(self.store.layout).upper()

    @property
    def stable_label(self) -> str:
        if self.label:
            return self.label
        if self.layout in {"S2", "S3", "S5"}:
            if self.seed is None:
                raise ValueError(f"{self.layout} requires an explicit seed in E3 metadata")
            return f"{self.layout}-{self.tier}-seed-{self.seed}"
        return self.layout


def compute_erosion_slope(
    checkpoints: Sequence[float | int | datetime], accuracies: Sequence[float]
) -> float:
    """Return the OLS change in accuracy per replay week."""

    if len(checkpoints) != len(accuracies):
        raise ValueError("checkpoints and accuracies must have equal lengths")
    if len(checkpoints) < 2:
        return math.nan
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
        return math.nan
    design = np.column_stack((np.ones(np.count_nonzero(finite)), x[finite]))
    return float(np.linalg.lstsq(design, y[finite], rcond=None)[0][1])


def _with_budget(cfg: EngineConfig, budget: int) -> EngineConfig:
    return cfg.model_copy(
        update={"reader": cfg.reader.model_copy(update={"budget_tokens": budget})}
    )


def _normalise_conditions(
    stores: Sequence[StoreHandle | StoreCondition],
) -> list[StoreCondition]:
    conditions: list[StoreCondition] = []
    for value in stores:
        if isinstance(value, StoreCondition):
            conditions.append(value)
            continue
        layout = str(value.layout).upper()
        tier = str(getattr(value, "builder_tier", "sonnet" if layout == "S3" else "baseline"))
        conditions.append(
            StoreCondition(
                store=value,
                seed=getattr(value, "seed", None),
                tier=tier,
                replay_hash=getattr(value, "replay_hash", None),
                model=getattr(value, "builder_model", None),
            )
        )
    labels = [condition.stable_label for condition in conditions]
    if len(labels) != len(set(labels)):
        raise ValueError("E3 condition labels must be unique and independent of caller order")
    return conditions


def _validate_condition_matrix(conditions: Sequence[StoreCondition]) -> None:
    layouts = {condition.layout for condition in conditions}
    missing = sorted({"S0", "S1", "S3", "S4"} - layouts)
    if missing:
        raise ValueError("E3 requires S0, S1, S3, and S4; missing " + ", ".join(missing))
    sonnet = [
        condition
        for condition in conditions
        if condition.layout == "S3" and condition.tier.casefold() == "sonnet"
    ]
    if not sonnet:
        raise ValueError("E3 requires at least one explicitly labelled S3 Sonnet condition")
    seeds = [condition.seed for condition in sonnet]
    if any(seed is None for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("S3 Sonnet conditions require unique explicit seeds")


def _checkpoint_times(events: Sequence[EvalEvent]) -> list[tuple[str, float, datetime]]:
    if not events:
        return []
    start = min(event.time for event in events)
    end = max(event.time for event in events)
    checkpoints = [
        (f"week-{week}", float(week), start + timedelta(weeks=week))
        for week in _CHECKPOINT_WEEKS
        if start + timedelta(weeks=week) < end
    ]
    end_week = (end - start).total_seconds() / (7 * 24 * 60 * 60)
    checkpoints.append(("end", end_week, end))
    timestamps = [checkpoint[2] for checkpoint in checkpoints]
    if len(timestamps) != len(set(timestamps)):
        raise AssertionError("E3 checkpoints must have unique replay timestamps")
    return checkpoints


def _health(store: StoreHandle, ref: SnapshotRef) -> dict[str, Any]:
    checkout = store.materialise(ref)
    try:
        return compute_store_health(checkout)
    finally:
        shutil.rmtree(checkout, ignore_errors=True)


def _usage_files(store: StoreHandle) -> list[Path]:
    candidates = [Path(store.root) / "usage.jsonl"]
    try:
        candidates.append(Path(store.worktree) / "usage.jsonl")
    except (AttributeError, TypeError):
        pass
    return list(dict.fromkeys(path for path in candidates if path.exists()))


def _usage_cost(condition: StoreCondition) -> dict[str, float | bool]:
    records: list[dict[str, Any]] = []
    for path in _usage_files(condition.store):
        with path.open(encoding="utf-8") as source:
            records.extend(json.loads(line) for line in source if line.strip())
    tokens_in = sum(float(row.get("tokens_in", row.get("input_tokens", 0))) for row in records)
    tokens_out = sum(float(row.get("tokens_out", row.get("output_tokens", 0))) for row in records)
    event_ids = {
        str(event_id)
        for row in records
        if isinstance(row.get("event_ids", []), list)
        for event_id in row["event_ids"]
    }
    attempted_events = sum(
        float(row.get("event_count", row.get("events", 0))) for row in records
    )
    distinct_events = float(len(event_ids)) if event_ids else attempted_events
    failures = sum(
        str(row.get("status", "")).casefold() in {"failed", "rolled_back", "dlq"}
        or row.get("success") is False
        for row in records
    )
    files_touched = sum(
        len(value) if isinstance(value, list) else float(value or 0)
        for value in (row.get("files_touched", 0) for row in records)
    )
    total_tokens = tokens_in + tokens_out
    cost_usd = sum(
        float(row.get("cost_usd", row.get("cost", row.get("total_cost_usd", 0))) or 0)
        for row in records
    )
    builder_applicable = condition.layout in {"S2", "S3", "S5"}
    return {
        "records": float(len(records)),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "build_tokens": total_tokens,
        "distinct_events": distinct_events,
        "tokens_per_1000_events": (
            total_tokens * 1_000 / distinct_events if distinct_events else 0.0
        ),
        "cost_usd": cost_usd,
        "dollars_per_1000_events": (
            cost_usd * 1_000 / distinct_events if distinct_events else 0.0
        ),
        "wall_time_s": sum(
            float(row.get("wall_time_s", row.get("latency_s", 0))) for row in records
        ),
        "files_touched": float(files_touched),
        "builder_failure_rate": failures / len(records) if records else math.nan,
        "builder_usage_applicable": builder_applicable,
        "builder_usage_present": bool(records) if builder_applicable else True,
    }


def _current_ledger(store: StoreHandle) -> list[str] | None:
    try:
        return [str(event_id) for event_id in store.delivered_event_ids()]
    except (OSError, ValueError):
        return None
    try:
        path = Path(store.worktree) / _DELIVERED_PATH
    except (AttributeError, TypeError):
        return None
    if not path.exists():
        return None
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _input_metadata(store: StoreHandle) -> dict[str, Any]:
    try:
        path = Path(store.worktree) / _INPUT_META_PATH
    except (AttributeError, TypeError):
        return {}
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _ledger_sha(ids: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{event_id}\n" for event_id in ids).encode()).hexdigest()


def _same_input_proof(conditions: Sequence[StoreCondition]) -> tuple[bool, dict[str, Any]]:
    replay_hashes = {condition.replay_hash for condition in conditions}
    replay_ok = None not in replay_hashes and len(replay_hashes) == 1
    ledgers = {condition.stable_label: _current_ledger(condition.store) for condition in conditions}
    available = all(ledger is not None for ledger in ledgers.values())
    first_label = conditions[0].stable_label
    reference = ledgers[first_label] or []
    mismatches: dict[str, Any] = {}
    metadata_ok = True
    for condition in conditions:
        label = condition.stable_label
        ledger = ledgers[label]
        metadata = _input_metadata(condition.store)
        if ledger is None:
            mismatches[label] = {"reason": "missing delivered-event ledger"}
            metadata_ok = False
            continue
        expected_hash = _ledger_sha(ledger)
        if metadata.get("event_ids_sha256") != expected_hash:
            metadata_ok = False
            mismatches.setdefault(label, {})["metadata_event_ids_sha256"] = {
                "expected": expected_hash,
                "actual": metadata.get("event_ids_sha256"),
            }
        if ledger != reference:
            first_difference = next(
                (
                    index
                    for index, (left, right) in enumerate(
                        zip(reference, ledger, strict=False)
                    )
                    if left != right
                ),
                min(len(reference), len(ledger)),
            )
            mismatches.setdefault(label, {})["ledger"] = {
                "first_difference": first_difference,
                "expected_count": len(reference),
                "actual_count": len(ledger),
                "expected": reference[first_difference : first_difference + 5],
                "actual": ledger[first_difference : first_difference + 5],
            }
    ledger_ok = available and all(ledger == reference for ledger in ledgers.values())
    proof = {
        "replay_hashes": {
            condition.stable_label: condition.replay_hash for condition in conditions
        },
        "replay_hash_identical": replay_ok,
        "ledger_hashes": {
            label: _ledger_sha(ledger) if ledger is not None else None
            for label, ledger in ledgers.items()
        },
        "ledger_identical": ledger_ok,
        "ledger_metadata_valid": metadata_ok,
        "mismatches": mismatches,
    }
    return replay_ok and ledger_ok and metadata_ok, proof


def _snapshot_ledger(store: StoreHandle, ref: SnapshotRef) -> list[str] | None:
    try:
        return [str(event_id) for event_id in store.delivered_event_ids(ref)]
    except (OSError, ValueError):
        return None
    content = store.read(ref, _DELIVERED_PATH)
    if content is None:
        return None
    return [line.strip() for line in content.splitlines() if line.strip()]


def _gate_probe(
    probe: Probe,
    condition: StoreCondition,
    events: Sequence[EvalEvent],
    ref: SnapshotRef,
) -> tuple[bool, str]:
    ledger = _snapshot_ledger(condition.store, ref)
    if ledger is None:
        return False, "missing_snapshot_delivery_ledger"
    by_id = {event.id: event for event in events}
    missing = [event_id for event_id in ledger if event_id not in by_id]
    reasons: list[str] = []
    if missing:
        reasons.append("ledger_id_absent_from_replay:" + "|".join(missing[:5]))
    if any(by_id[event_id].time > probe.T for event_id in ledger if event_id in by_id):
        reasons.append("delivered_event_after_T")
    if ref.T_last_event > probe.T:
        reasons.append("snapshot_after_T")
    for source_id in probe.source_event_ids:
        source = by_id.get(source_id)
        if source is None:
            reasons.append(f"source_event_unresolved:{source_id}")
        elif source.time > probe.T:
            reasons.append(f"source_event_after_T:{source_id}")
    before_tokens: set[str] = set()
    after_tokens: set[str] = set()
    for event in events:
        target = after_tokens if event.time > probe.T else before_tokens
        target.update(token.casefold() for token in _WORD_RE.findall(event.model_dump_json()))
    question_tokens = {token.casefold() for token in _WORD_RE.findall(probe.question)}
    post_only = sorted(question_tokens & (after_tokens - before_tokens))
    if post_only:
        reasons.append("question_token_only_post_T:" + "|".join(post_only))
    return not reasons, ";".join(reasons)


def _text_documents(
    condition: StoreCondition,
    ref: SnapshotRef,
    probe: Probe,
) -> tuple[list[tuple[str, str]], int]:
    store = condition.store
    files = [
        str(Path(raw)).replace("\\", "/")
        for raw in store.list_files(ref)
        if ".git" not in Path(raw).parts and not str(raw).startswith("_vector/")
    ]
    opened = 0
    cached: dict[str, str] = {}

    def read(relpath: str) -> str:
        nonlocal opened
        if relpath not in cached:
            cached[relpath] = store.read(ref, relpath) or ""
            opened += 1
        return cached[relpath]

    if condition.layout == "S0":
        candidates = [relpath for relpath in files if relpath.startswith("events/")]
        question_terms = {
            term.casefold()
            for term in _WORD_RE.findall(probe.question)
            if len(term) > 2
        }
        ranked: list[tuple[int, int, str, str]] = []
        for relpath in candidates:
            content = read(relpath)
            folded = content.casefold()
            entity_hit = int(probe.entity.casefold() in folded)
            overlap = sum(term in folded for term in question_terms)
            ranked.append((entity_hit, overlap, relpath, content))
        ranked.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
        documents = [
            (relpath, f"[file:{relpath}]\n{content}")
            for entity_hit, _, relpath, content in ranked
            if entity_hit
        ]
        if not documents:
            documents = [
                (relpath, f"[file:{relpath}]\n{content}")
                for _, _, relpath, content in ranked
            ]
        return documents, opened

    entity_folded = probe.entity.casefold()
    index_files = [relpath for relpath in files if relpath.casefold() == "index.md"]
    entity_files = [relpath for relpath in files if entity_folded in relpath.casefold()]
    if not entity_files:
        overview_files = [relpath for relpath in files if Path(relpath).name == "OVERVIEW.md"]
        entity_files = [
            relpath for relpath in overview_files if entity_folded in read(relpath).casefold()
        ]
    entity_dirs = {str(Path(relpath).parent) for relpath in entity_files}
    related = [
        relpath
        for relpath in files
        if str(Path(relpath).parent) in entity_dirs
        and Path(relpath).suffix.casefold() in {".md", ".txt", ".json", ".jsonl"}
    ]
    ordered = list(dict.fromkeys([*index_files, *sorted(related), *sorted(entity_files)]))
    if condition.layout == "S2":
        ordered = list(dict.fromkeys([*sorted(related), *sorted(entity_files)]))
    return [(relpath, f"[file:{relpath}]\n{read(relpath)}") for relpath in ordered], opened


def _vector_documents(
    condition: StoreCondition,
    ref: SnapshotRef,
    probe: Probe,
    embeddings: EmbeddingsProvider | None,
    *,
    top_k: int = 10,
) -> tuple[list[tuple[str, str]], int]:
    index = VectorIndex.from_store(condition.store, provider=embeddings, ref=ref)
    hits = index.search_hits(probe.question, top_k=top_k)
    raw_ids = condition.store.read(ref, "_vector/ids.json")
    raw_documents = condition.store.read(ref, "_vector/documents.json")
    if raw_ids is None or raw_documents is None:
        return [], 1
    ids = json.loads(raw_ids)
    documents = json.loads(raw_documents)
    by_id = dict(zip(ids, documents, strict=True))
    return [
        (hit.item_id, f"[source:{hit.item_id}]\n{by_id.get(hit.item_id, '')}")
        for hit in hits
    ], 1


def _store_material(
    condition: StoreCondition,
    ref: SnapshotRef,
    probe: Probe,
    cfg: EngineConfig,
    embeddings: EmbeddingsProvider | None,
) -> Material:
    if condition.layout in {"S4", "S5"}:
        documents, opened = _vector_documents(condition, ref, probe, embeddings)
    else:
        documents, opened = _text_documents(condition, ref, probe)
    full_text = "\n".join(text for _, text in documents)
    selected = truncate_to_tokens(full_text, cfg.reader.budget_tokens)
    source_ids = [item_id for item_id, _ in documents if item_id in selected]
    return Material(
        arm=condition.stable_label,
        text=selected,
        source_ids=source_ids,
        tool_calls=opened + 1,
        original_tokens=count_tokens(full_text),
    )


def _macro_family(family: str) -> str:
    return "multisource" if family == "code_location" else family


def _macro_accuracy(outcomes: Sequence[ProbeOutcome], *, require_all: bool = True) -> float:
    by_family: dict[str, list[float]] = defaultdict(list)
    for outcome in outcomes:
        by_family[_macro_family(outcome.probe.family)].append(outcome.grade.value)
    if require_all and any(not by_family[family] for family in _MACRO_FAMILIES):
        return math.nan
    values = [float(np.mean(by_family[family])) for family in _MACRO_FAMILIES if by_family[family]]
    return float(np.mean(values)) if values else math.nan


def _evaluate_condition(
    condition: StoreCondition,
    cfg: EngineConfig,
    probes: Sequence[Probe],
    events: Sequence[EvalEvent],
    *,
    llm: LLMProvider | None,
    embeddings: EmbeddingsProvider | None,
    gate_rows: list[dict[str, Any]],
    phase: str,
    shared_e2_dir: Path,
    seed: int,
) -> list[ProbeOutcome]:
    arm = "A4" if condition.layout == "S3" else condition.layout
    cell = shared_e2_dir / phase.replace(":", "-") / condition.stable_label
    result, outcomes = evaluate_e2(
        cfg=cfg,
        probes=probes,
        events=events,
        out_dir=cell,
        seed=seed,
        store=condition.store,
        llm=llm,
        embeddings=embeddings,
        arms=(arm,),
    )
    del result
    gate = pd.read_csv(cell / "gate.csv")
    for row in gate.to_dict(orient="records"):
        gate_rows.append(
            {
                "phase": phase,
                "store": condition.stable_label,
                "probe_id": row.get("probe_id", row.get("item_id", "")),
                "T": row.get("T", ""),
                "sha": row.get("sha", ""),
                "status": row.get("result", "FAIL"),
                "reasons": row.get("reasons", ""),
            }
        )
    return outcomes


def _resource_summary(outcomes: Sequence[ProbeOutcome], budget: int) -> dict[str, Any]:
    fill = [outcome for outcome in outcomes if outcome.original_tokens >= budget]
    within = (
        all(0.9 * budget <= row.answer.tokens_read <= 1.1 * budget for row in fill)
        if fill
        else None
    )
    return {
        "n": len(outcomes),
        "tokens_read": (
            float(np.mean([outcome.answer.tokens_read for outcome in outcomes]))
            if outcomes
            else 0.0
        ),
        "tool_calls": (
            float(np.mean([outcome.answer.tool_calls for outcome in outcomes]))
            if outcomes
            else 0.0
        ),
        "latency_s": (
            float(np.mean([outcome.answer.latency_s for outcome in outcomes]))
            if outcomes
            else 0.0
        ),
        "reader_cost_usd": (
            float(np.mean([outcome.reader_cost_usd for outcome in outcomes]))
            if outcomes
            else 0.0
        ),
        "budget_fill_observations": len(fill),
        "budget_within_10_pct": within,
        "budget_check_status": "measured" if fill else "not_applicable_no_fill",
    }


def _floor_checks(
    cfg: EngineConfig,
    probes: Sequence[Probe],
    events: Sequence[EvalEvent],
    *,
    llm: LLMProvider | None,
) -> dict[str, float]:
    by_arm: dict[str, list[ProbeOutcome]] = defaultdict(list)
    for probe in probes:
        for arm in ("A0", "retrieve_everything", "retrieve_nothing"):
            material = build_arm(arm, probe, events, cfg)
            response = answer(probe, material, cfg, provider=llm)
            by_arm[arm].append(
                ProbeOutcome(
                    probe=probe,
                    answer=response,
                    grade=grade_answer(probe, response.text),
                    original_tokens=material.original_tokens or 0,
                    supersession_error=False,
                )
            )
    everything = _macro_accuracy(by_arm["retrieve_everything"])
    prior_rows = [
        row
        for row in by_arm["A0"]
        if _macro_family(row.probe.family) in {"temporal", "update", "multisource"}
    ]
    prior = float(np.mean([row.grade.value for row in prior_rows])) if prior_rows else math.nan
    return {
        "floor_retrieve_everything": everything,
        "prior_target_accuracy": prior,
        "checks.floor_retrieve_everything_ge_0_9": float(
            np.isfinite(everything) and everything >= 0.9
        ),
        "checks.prior_leq_0_3": float(np.isfinite(prior) and prior <= 0.3),
    }


def _condition_frame(
    outcomes: dict[tuple[str, int], list[ProbeOutcome]], label: str, budget: int
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "probe_id": row.probe.probe_id,
                "entity": row.probe.entity,
                "family": _macro_family(row.probe.family),
                "score": row.grade.value,
            }
            for row in outcomes.get((label, budget), [])
        ],
        columns=["probe_id", "entity", "family", "score"],
    )


def _macro_from_frame(frame: pd.DataFrame, score_column: str) -> float:
    if frame.empty:
        return math.nan
    scores: dict[str, list[float]] = defaultdict(list)
    for row in frame.to_dict(orient="records"):
        value = row.get(score_column)
        if isinstance(value, (int, float, np.integer, np.floating)):
            scores[str(row.get("family"))].append(float(value))
    if any(not scores.get(family) for family in _MACRO_FAMILIES):
        return math.nan
    return float(np.mean([np.mean(scores[family]) for family in _MACRO_FAMILIES]))


def _macro_effect(frame: pd.DataFrame, left_columns: Sequence[str], right: str) -> float:
    left = [_macro_from_frame(frame, column) for column in left_columns]
    right_value = _macro_from_frame(frame, right)
    if not all(np.isfinite(value) for value in [*left, right_value]):
        return math.nan
    return float(np.mean(left) - right_value)


def _macro_clustered_bca(
    frame: pd.DataFrame,
    left_columns: Sequence[str],
    right_column: str,
    *,
    n_resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    effect = _macro_effect(frame, left_columns, right_column)
    entities = frame["entity"].drop_duplicates().tolist()
    if not np.isfinite(effect) or len(entities) < 2:
        return effect, effect, effect
    families = list(_MACRO_FAMILIES)
    condition_columns = [*left_columns, right_column]
    entity_index = {entity: index for index, entity in enumerate(entities)}
    family_index = {family: index for index, family in enumerate(families)}
    sums = np.zeros((len(entities), len(families), len(condition_columns)), dtype=float)
    counts = np.zeros((len(entities), len(families)), dtype=float)
    for row in frame.to_dict(orient="records"):
        entity_position = entity_index[row["entity"]]
        family_position = family_index[row["family"]]
        counts[entity_position, family_position] += 1
        for condition_position, column in enumerate(condition_columns):
            sums[entity_position, family_position, condition_position] += float(row[column])
    rng = np.random.default_rng(seed)
    cluster_weights = rng.multinomial(
        len(entities),
        np.full(len(entities), 1.0 / len(entities)),
        size=n_resamples,
    )
    draw_sums = np.einsum("re,efc->rfc", cluster_weights, sums)
    draw_counts = np.einsum("re,ef->rf", cluster_weights, counts)
    with np.errstate(divide="ignore", invalid="ignore"):
        family_means = draw_sums / draw_counts[:, :, np.newaxis]
    condition_macros = np.mean(family_means, axis=1)
    bootstrap = np.mean(condition_macros[:, : len(left_columns)], axis=1) - condition_macros[
        :, -1
    ]
    bootstrap = bootstrap[np.isfinite(bootstrap)]
    if len(bootstrap) < max(100, n_resamples // 2):
        return effect, effect, effect
    proportion_less = (
        np.count_nonzero(bootstrap < effect) + 0.5 * np.count_nonzero(bootstrap == effect)
    ) / len(bootstrap)
    epsilon = 0.5 / len(bootstrap)
    bias = float(norm.ppf(np.clip(proportion_less, epsilon, 1 - epsilon)))
    jackknife = np.asarray(
        [
            _macro_effect(
                pd.DataFrame(frame.loc[frame["entity"] != entity]),
                left_columns,
                right_column,
            )
            for entity in entities
        ],
        dtype=float,
    )
    jackknife = jackknife[np.isfinite(jackknife)]
    acceleration = 0.0
    if len(jackknife) >= 2:
        centred = float(np.mean(jackknife)) - jackknife
        denominator = 6.0 * float(np.sum(centred**2)) ** 1.5
        if denominator:
            acceleration = float(np.sum(centred**3)) / denominator
    adjusted: list[float] = []
    for quantile in norm.ppf([0.025, 0.975]):
        shifted = bias + quantile
        divisor = 1.0 - acceleration * shifted
        if math.isclose(divisor, 0.0, abs_tol=1e-15):
            adjusted.append(0.0 if shifted < 0 else 1.0)
        else:
            adjusted.append(float(norm.cdf(bias + shifted / divisor)))
    low, high = np.quantile(bootstrap, np.clip(np.sort(adjusted), 0.0, 1.0))
    return effect, float(low), float(high)


def _primary_contrasts(
    conditions: Sequence[StoreCondition],
    outcomes: dict[tuple[str, int], list[ProbeOutcome]],
    *,
    seed: int,
    same_input: bool,
) -> pd.DataFrame:
    s3 = sorted(
        (
            condition
            for condition in conditions
            if condition.layout == "S3" and condition.tier.casefold() == "sonnet"
        ),
        key=lambda condition: int(condition.seed or 0),
    )
    s3_labels = [condition.stable_label for condition in s3]
    seed_macros = [_macro_accuracy(outcomes[(label, 8_000)]) for label in s3_labels]
    seed_spread = (
        float(np.std(seed_macros, ddof=1))
        if len(seed_macros) >= 3 and all(np.isfinite(seed_macros))
        else math.nan
    )
    rows: list[dict[str, Any]] = []
    for comparator_layout in ("S1", "S4"):
        comparator = next(
            condition for condition in conditions if condition.layout == comparator_layout
        )
        frames = []
        for index, label in enumerate(s3_labels):
            frame = _condition_frame(outcomes, label, 8_000).rename(
                columns={"score": f"left_{index}"}
            )
            frames.append(frame)
        right = _condition_frame(outcomes, comparator.stable_label, 8_000).rename(
            columns={"score": "right"}
        )
        keys = ["probe_id", "entity", "family"]
        paired = right
        for frame in frames:
            paired = paired.merge(frame, on=keys, how="inner", validate="one_to_one")
        left_columns = [f"left_{index}" for index in range(len(frames))]
        delta, ci_low, ci_high = _macro_clustered_bca(
            paired,
            left_columns,
            "right",
            n_resamples=BOOTSTRAP_RESAMPLES,
            seed=seed,
        )
        mcnemar = mcnemar_test(
            paired[left_columns[0]].to_numpy(float) >= 1.0,
            paired["right"].to_numpy(float) >= 1.0,
        )
        spread_supported = np.isfinite(seed_spread)
        spread_qualified = bool(spread_supported and abs(delta) > seed_spread)
        valid = bool(
            same_input
            and np.isfinite(delta)
            and spread_supported
            and all(family in set(paired["family"]) for family in _MACRO_FAMILIES)
        )
        significantly_better = bool(valid and spread_qualified and delta > 0 and ci_low > 0)
        if not same_input:
            verdict = "invalid: same-input proof failed"
        elif not spread_supported:
            verdict = "supported-not-run: reliability requires >=3 configured Sonnet seeds"
        elif not np.isfinite(delta):
            verdict = "invalid: incomplete five-family probe matrix"
        elif significantly_better:
            verdict = "significantly better"
        elif spread_qualified:
            verdict = "spread-qualified, not significantly better"
        else:
            verdict = "no reliable difference"
        rows.append(
            {
                "contrast": f"S3-{comparator_layout}@8000",
                "delta": delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "mcnemar_seed": s3[0].seed,
                "mcnemar_p": mcnemar.p_value,
                "discordant_s3": mcnemar.b,
                "discordant_comparator": mcnemar.c,
                "seed_spread": seed_spread,
                "seed_count": len(s3),
                "spread_qualified": spread_qualified,
                "significantly_better": significantly_better,
                "reliable_difference": valid and spread_qualified,
                "valid": valid,
                "verdict": verdict,
                "n": len(paired),
                "entities": paired["entity"].nunique(),
                "n_resamples": BOOTSTRAP_RESAMPLES,
            }
        )
    return pd.DataFrame(rows)


def _stable_probe_key(probe: Probe) -> str:
    return hashlib.sha256(probe.probe_id.encode()).hexdigest()


def _fixed_erosion_panel(probes: Sequence[Probe], limit: int) -> list[Probe]:
    if limit <= 0:
        raise ValueError("erosion probe limit must be positive")
    grouped: dict[str, list[Probe]] = defaultdict(list)
    for probe in probes:
        grouped[_macro_family(probe.family)].append(probe)
    for family in grouped:
        grouped[family].sort(key=lambda probe: (_stable_probe_key(probe), probe.probe_id))
    panel: list[Probe] = []
    cursor = 0
    while len(panel) < min(limit, len(probes)):
        progressed = False
        for family in _MACRO_FAMILIES:
            candidates = grouped.get(family, [])
            if cursor < len(candidates) and len(panel) < limit:
                panel.append(candidates[cursor])
                progressed = True
        if not progressed:
            break
        cursor += 1
    return panel


def _question_safe_at_checkpoint(
    probe: Probe, checkpoint: datetime, events: Sequence[EvalEvent], gold: PythonGold
) -> bool:
    """Apply E2's question-token firewall before freezing the erosion panel."""

    derived = _rederive_probe(probe, checkpoint, events, gold)
    if derived is None:
        return False
    before_tokens: set[str] = set()
    after_tokens: set[str] = set()
    for event in events:
        target = after_tokens if event.time > checkpoint else before_tokens
        target.update(token.casefold() for token in _WORD_RE.findall(event.model_dump_json()))
    question_tokens = {
        token.casefold() for token in _WORD_RE.findall(derived.question)
    }
    return not question_tokens.intersection(after_tokens - before_tokens)


def _field_from_question(probe: Probe) -> str | None:
    patterns = (
        r"current\s+([\w.-]+)\s+of",
        r"latest\s+([\w.-]+)\s+of",
        r"was the\s+([\w.-]+)\s+of",
        r"what is the\s+([\w.-]+)\s+of",
    )
    for pattern in patterns:
        if match := re.search(pattern, probe.question, re.IGNORECASE):
            return match.group(1)
    return None


def _links_at(events: Sequence[EvalEvent], entity: str, at: datetime) -> tuple[list[str], list[str]]:
    links: list[str] = []
    source_ids: list[str] = []
    for event in events:
        if event.time > at:
            continue
        data = event.data or {}
        text = " ".join(
            str(value)
            for value in (
                event.subject,
                data.get("issue_key", ""),
                data.get("title", ""),
                data.get("subject", ""),
                data.get("body", ""),
            )
        )
        if entity.casefold() not in {key.casefold() for key in _ISSUE_RE.findall(text)}:
            continue
        event_links: list[str] = []
        if "pull_request" in event.type.casefold():
            number = data.get("number") or data.get("pr_number")
            if number is not None:
                event_links.append(f"pr:{number}")
        if "mail" in event.type.casefold():
            thread = data.get("thread_id") or data.get("thread_root")
            if isinstance(thread, str) and thread:
                event_links.append(thread if thread.startswith("thread:") else f"thread:{thread}")
        for link in event_links:
            if link not in links:
                links.append(link)
                source_ids.append(event.id)
    return sorted(links), source_ids


def _code_at(
    events: Sequence[EvalEvent], entity: str, at: datetime
) -> tuple[dict[str, list[str]], list[str]]:
    files: set[str] = set()
    source_ids: list[str] = []
    for event in events:
        if event.time > at or not is_merged_pr(event) or is_formatting_only(event):
            continue
        if entity.casefold() not in {key.casefold() for key in issue_keys_for_pr(event)}:
            continue
        paths = changed_files(event)
        if paths:
            files.update(paths)
            source_ids.append(event.id)
    ordered = sorted(files)
    return {"files": ordered, "modules": sorted({module_for_file(path) for path in ordered})}, source_ids


def _rederive_probe(
    probe: Probe,
    checkpoint: datetime,
    events: Sequence[EvalEvent],
    gold: PythonGold,
) -> Probe | None:
    if probe.family in {"extraction", "temporal", "update"}:
        field = _field_from_question(probe)
        if field is None:
            return None
        target = checkpoint
        question = probe.question
        if probe.family == "temporal" and " as of " in probe.question.casefold():
            raw_target = probe.question.rsplit(" as of ", 1)[-1].rstrip("?. ")
            try:
                frozen_target = datetime.fromisoformat(raw_target.replace("Z", "+00:00"))
            except ValueError:
                frozen_target = checkpoint
            target = min(frozen_target, checkpoint)
            question = re.sub(
                r"(?i)\s+as of\s+.+?[?.]*$",
                f" as of {target.isoformat()}?",
                probe.question,
            )
        transitions = gold.transitions(probe.entity, field, target)
        current = (
            string_value(transitions[-1].new_value)
            if transitions and transitions[-1].new_value is not None
            else "UNKNOWN"
        )
        superseded = []
        if probe.family == "update" and transitions:
            raw = [transitions[0].old_value, *(item.new_value for item in transitions[:-1])]
            superseded = list(
                dict.fromkeys(
                    string_value(value)
                    for value in raw
                    if value is not None and string_value(value) != current
                )
            )
        return probe.model_copy(
            update={
                "T": checkpoint,
                "question": question,
                "gold": current,
                "superseded_values": superseded,
                "source_event_ids": [item.event_id for item in transitions],
            }
        )
    if probe.family == "multisource":
        links, source_ids = _links_at(events, probe.entity, checkpoint)
        return probe.model_copy(
            update={"T": checkpoint, "gold": links, "source_event_ids": source_ids}
        )
    if probe.family == "code_location":
        code_gold, source_ids = _code_at(events, probe.entity, checkpoint)
        return probe.model_copy(
            update={"T": checkpoint, "gold": code_gold, "source_event_ids": source_ids}
        )
    if probe.family == "abstention":
        return probe.model_copy(
            update={
                "T": checkpoint,
                "gold": "UNKNOWN",
                "superseded_values": [],
                "source_event_ids": [],
            }
        )
    return probe.model_copy(update={"T": checkpoint})


def _health_seed_spread(health: pd.DataFrame, sonnet_labels: set[str]) -> pd.DataFrame:
    selected = pd.DataFrame(health.loc[health["store"].isin(sorted(sonnet_labels))])
    id_columns = {
        "corpus",
        "store",
        "layout",
        "tier",
        "seed",
        "model",
        "checkpoint",
        "replay_week",
        "snapshot_sha",
    }
    numeric = [
        column
        for column in selected.columns
        if column not in id_columns and pd.api.types.is_numeric_dtype(selected[column])
    ]
    rows: list[dict[str, Any]] = []
    for checkpoint, frame in selected.groupby("checkpoint", sort=False):
        records = frame.to_dict(orient="records")
        for metric in numeric:
            values = [
                float(value)
                for row in records
                if isinstance((value := row.get(metric)), (int, float, np.integer, np.floating))
                and not math.isnan(float(value))
            ]
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "metric": metric,
                    "seed_count": len(values),
                    "seed_spread": float(np.std(values, ddof=1)) if len(values) >= 3 else math.nan,
                    "status": "measured" if len(values) >= 3 else "supported-not-run",
                }
            )
    return pd.DataFrame(rows)


def _s4_recall(
    condition: StoreCondition,
    probes: Sequence[Probe],
    events: Sequence[EvalEvent],
    embeddings: EmbeddingsProvider | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for probe in probes:
        gold_ids = set(probe.source_event_ids)
        if not gold_ids:
            continue
        try:
            ref = condition.store.snapshot(probe.T)
        except LookupError:
            continue
        passed, _ = _gate_probe(probe, condition, events, ref)
        if not passed:
            continue
        hits = VectorIndex.from_store(
            condition.store, provider=embeddings, ref=ref
        ).search_hits(probe.question, top_k=10)
        retrieved = {
            str(hit.metadata.get("event_id", hit.item_id))
            for hit in hits
            if hit.metadata.get("event_id", hit.item_id)
        }
        rows.append(
            {
                "probe_id": probe.probe_id,
                "family": _macro_family(probe.family),
                "entity": probe.entity,
                "gold_count": len(gold_ids),
                "retrieved_gold_count": len(gold_ids & retrieved),
                "recall_at_10": len(gold_ids & retrieved) / len(gold_ids),
            }
        )
    return pd.DataFrame(rows)


def evaluate_e3(
    *,
    stores: Sequence[StoreHandle | StoreCondition],
    cfg: EngineConfig,
    probes: Sequence[Probe],
    events: Sequence[EvalEvent],
    out_dir: Path,
    seed: int,
    llm: LLMProvider | None = None,
    embeddings: EmbeddingsProvider | None = None,
    budgets: Sequence[int] = READ_BUDGETS,
    erosion_probe_limit: int = EROSION_PANEL_SIZE,
    corpus_name: str = "unknown",
    smoke: bool = False,
    validation_audit: Mapping[str, float | bool] | None = None,
) -> ExperimentResult:
    """Run every E3 store condition once and aggregate all configured seeds."""

    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = _normalise_conditions(stores)
    _validate_condition_matrix(conditions)
    budgets = tuple(dict.fromkeys(int(value) for value in budgets))
    if 8_000 not in budgets:
        raise ValueError("E3 configured budgets must include the 8,000-token primary")

    same_input, same_input_details = _same_input_proof(conditions)
    ledger_diff_path = out_dir / "ledger_diff.json"
    ledger_diff_path.write_text(
        json.dumps(same_input_details, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    curve_rows: list[dict[str, Any]] = []
    outcome_map: dict[tuple[str, int], list[ProbeOutcome]] = {}
    gate_rows: list[dict[str, Any]] = []
    for condition in conditions:
        for budget in budgets:
            outcomes = _evaluate_condition(
                condition,
                _with_budget(cfg, budget),
                probes,
                events,
                llm=llm,
                embeddings=embeddings,
                gate_rows=gate_rows,
                phase="accuracy",
                shared_e2_dir=out_dir / "shared-e2" / f"budget-{budget}",
                seed=seed,
            )
            outcome_map[(condition.stable_label, budget)] = outcomes
            curve_rows.append(
                {
                    "corpus": corpus_name,
                    "store": condition.stable_label,
                    "layout": condition.layout,
                    "tier": condition.tier,
                    "seed": condition.seed,
                    "model": condition.model,
                    "budget": budget,
                    "macro_acc": _macro_accuracy(outcomes),
                    "family_complete": all(
                        family
                        in {_macro_family(outcome.probe.family) for outcome in outcomes}
                        for family in _MACRO_FAMILIES
                    ),
                    **_resource_summary(outcomes, budget),
                }
            )
    curve = pd.DataFrame(curve_rows)

    checkpoints = _checkpoint_times(events)
    gold = PythonGold(events)
    earliest_checkpoint = checkpoints[0][2]
    eligible_templates = [
        probe
        for probe in probes
        if _question_safe_at_checkpoint(probe, earliest_checkpoint, events, gold)
    ]
    panel = _fixed_erosion_panel(eligible_templates, erosion_probe_limit)
    panel_payload = {
        "requested_size": erosion_probe_limit,
        "actual_size": len(panel),
        "checkpoint_safe_template_count": len(eligible_templates),
        "probe_ids": [probe.probe_id for probe in panel],
        "sha256": hashlib.sha256(
            "".join(f"{probe.probe_id}\n" for probe in panel).encode()
        ).hexdigest(),
        "status": (
            "measured-60-probe-panel"
            if len(panel) == EROSION_PANEL_SIZE and erosion_probe_limit == EROSION_PANEL_SIZE
            else "non-evidentiary-configured-reduced-panel"
            if smoke and len(panel) == erosion_probe_limit
            else "invalid-incomplete-panel"
        ),
    }
    panel_path = out_dir / "erosion_panel.json"
    panel_path.write_text(
        json.dumps(panel_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    health_rows: list[dict[str, Any]] = []
    erosion_rows: list[dict[str, Any]] = []
    for condition in conditions:
        condition_erosion: list[dict[str, Any]] = []
        for checkpoint, week, at in checkpoints:
            try:
                ref = condition.store.snapshot(at)
            except LookupError:
                continue
            health_rows.append(
                {
                    "corpus": corpus_name,
                    "store": condition.stable_label,
                    "layout": condition.layout,
                    "tier": condition.tier,
                    "seed": condition.seed,
                    "model": condition.model,
                    "checkpoint": checkpoint,
                    "replay_week": week,
                    "snapshot_sha": ref.sha,
                    **_health(condition.store, ref),
                }
            )
            checkpoint_probes = [
                derived
                for probe in panel
                if (derived := _rederive_probe(probe, at, events, gold)) is not None
            ]
            outcomes = _evaluate_condition(
                condition,
                _with_budget(cfg, 8_000),
                checkpoint_probes,
                events,
                llm=llm,
                embeddings=embeddings,
                gate_rows=gate_rows,
                phase=f"erosion:{checkpoint}",
                shared_e2_dir=out_dir / "shared-e2" / "erosion",
                seed=seed,
            )
            panel_complete = len(checkpoint_probes) == len(panel) and len(outcomes) == len(panel)
            family_complete = all(
                family in {_macro_family(outcome.probe.family) for outcome in outcomes}
                for family in _MACRO_FAMILIES
            )
            condition_erosion.append(
                {
                    "corpus": corpus_name,
                    "store": condition.stable_label,
                    "layout": condition.layout,
                    "tier": condition.tier,
                    "seed": condition.seed,
                    "checkpoint": checkpoint,
                    "replay_week": week,
                    "accuracy": (
                        _macro_accuracy(outcomes) if panel_complete and family_complete else math.nan
                    ),
                    "panel_size": len(panel),
                    "eligible_at_checkpoint": len(checkpoint_probes),
                    "n": len(outcomes),
                    "gate_exclusions": len(checkpoint_probes) - len(outcomes),
                    "panel_complete": panel_complete,
                    "family_complete": family_complete,
                }
            )
        slope = (
            compute_erosion_slope(
                [row["replay_week"] for row in condition_erosion],
                [row["accuracy"] for row in condition_erosion],
            )
            if condition_erosion
            and all(
                bool(row["panel_complete"]) and bool(row["family_complete"])
                for row in condition_erosion
            )
            else math.nan
        )
        for row in condition_erosion:
            row["slope_per_week"] = slope
        erosion_rows.extend(condition_erosion)

    health = pd.DataFrame(health_rows)
    erosion_table = pd.DataFrame(erosion_rows)
    sonnet_labels = {
        condition.stable_label
        for condition in conditions
        if condition.layout == "S3" and condition.tier.casefold() == "sonnet"
    }
    health_spread = _health_seed_spread(health, sonnet_labels)

    cost = pd.DataFrame(
        [
            {
                "corpus": corpus_name,
                "store": condition.stable_label,
                "layout": condition.layout,
                "tier": condition.tier,
                "seed": condition.seed,
                "model": condition.model,
                **_usage_cost(condition),
            }
            for condition in conditions
        ]
    )
    contrasts = _primary_contrasts(conditions, outcome_map, seed=seed, same_input=same_input)
    s4 = next(condition for condition in conditions if condition.layout == "S4")
    s4_recall = _s4_recall(s4, probes, events, embeddings)

    gate = pd.DataFrame(gate_rows)
    paths = {
        "curve": out_dir / "curve.csv",
        "health": out_dir / "health.csv",
        "health_seed_spread": out_dir / "health_seed_spread.csv",
        "erosion": out_dir / "erosion.csv",
        "cost": out_dir / "cost.csv",
        "contrasts": out_dir / "contrasts.csv",
        "gate": out_dir / "gate.csv",
        "s4_recall": out_dir / "s4_recall.csv",
    }
    for name, table in (
        ("curve", curve),
        ("health", health),
        ("health_seed_spread", health_spread),
        ("erosion", erosion_table),
        ("cost", cost),
        ("contrasts", contrasts),
        ("gate", gate),
        ("s4_recall", s4_recall),
    ):
        table.to_csv(paths[name], index=False)

    curve_chart = e3_curve(curve.rename(columns={"macro_acc": "acc"}), out_dir)
    erosion_chart = erosion(
        erosion_table.rename(columns={"accuracy": "acc"}), out_dir
    )

    floor_metrics = _floor_checks(
        _with_budget(cfg, 8_000), probes, events, llm=llm
    )
    audit = dict(validation_audit or {})
    cost_records = cost.to_dict(orient="records")
    failure_records = [row for row in cost_records if bool(row["builder_usage_applicable"])]
    failure_ok = bool(failure_records) and all(
        bool(row["builder_usage_present"])
        and isinstance(row["builder_failure_rate"], (int, float))
        and not math.isnan(float(row["builder_failure_rate"]))
        and float(row["builder_failure_rate"]) <= 0.05
        for row in failure_records
    )
    accuracy_gate_records = [
        row for row in gate.to_dict(orient="records") if row.get("phase") == "accuracy"
    ]
    leakage_ok = bool(accuracy_gate_records) and all(
        row.get("status") == "PASS" for row in accuracy_gate_records
    )
    budget_records = [
        row
        for row in curve.to_dict(orient="records")
        if float(row.get("budget_fill_observations", 0)) > 0
    ]
    budget_ok = bool(budget_records) and all(
        bool(row.get("budget_within_10_pct")) for row in budget_records
    )
    erosion_complete = bool(panel) and len(erosion_rows) == len(conditions) * len(checkpoints) and all(
        bool(row.get("panel_complete")) and bool(row.get("family_complete"))
        for row in erosion_rows
    )
    recall_values = [
        float(row["recall_at_10"])
        for row in s4_recall.to_dict(orient="records")
        if isinstance(row.get("recall_at_10"), (int, float))
    ]
    recall_value = float(np.mean(recall_values)) if recall_values else math.nan
    seed_count = len(sonnet_labels)
    exact_rerun_identical = all(
        grade_answer(row.probe, row.answer.text) == row.grade
        for rows in outcome_map.values()
        for row in rows
    )
    gold_agreement = float(audit.get("dual_gold_agreement", 0.0)) >= 0.98
    pilot_ready = float(audit.get("pilot_kappa", 0.0)) >= 0.8
    claims_ready = float(audit.get("claim_disagreement", 1.0)) <= 0.02
    s1_review_ready = bool(audit.get("s1_review_passed", False))
    consolidated_gates = {
        "same_input": same_input,
        "answerability_floor": floor_metrics["checks.floor_retrieve_everything_ge_0_9"] == 1,
        "prior_floor": floor_metrics["checks.prior_leq_0_3"] == 1,
        "exact_grader_rerun": exact_rerun_identical,
        "dual_gold_agreement": gold_agreement,
        "pilot_kappa": pilot_ready,
        "claims_disagreement": claims_ready,
        "leakage": leakage_ok,
        "equal_budget": budget_ok,
        "s1_review": s1_review_ready,
        "s4_recall": np.isfinite(recall_value) and recall_value >= 0.7,
        "seed_reliability": seed_count >= 3,
        "builder_failures": failure_ok,
        "erosion_panel": (
            len(panel) == EROSION_PANEL_SIZE
            and erosion_complete
            and len(checkpoints) >= 2
        ),
        "evidentiary_profile": not smoke,
    }
    failed_gates = [name for name, passed in consolidated_gates.items() if not passed]
    if not contrasts.empty:
        contrasts["statistical_valid"] = contrasts["valid"]
        contrasts["invalid_reasons"] = contrasts.apply(
            lambda row: ";".join(
                [
                    *([] if bool(row["statistical_valid"]) else ["statistical_inference"]),
                    *failed_gates,
                ]
            ),
            axis=1,
        )
        contrasts["valid"] = contrasts["statistical_valid"] & (not failed_gates)
        contrasts.loc[~contrasts["valid"], "reliable_difference"] = False
        contrasts.loc[~contrasts["valid"], "significantly_better"] = False
        contrasts.loc[~contrasts["valid"], "verdict"] = contrasts.loc[
            ~contrasts["valid"], "invalid_reasons"
        ].map(lambda reasons: f"invalid: {reasons}")
        contrasts.to_csv(paths["contrasts"], index=False)
    metrics: dict[str, float] = {
        "checks.same_input_hash": float(same_input_details["replay_hash_identical"]),
        "checks.same_input_ledger": float(
            same_input_details["ledger_identical"]
            and same_input_details["ledger_metadata_valid"]
        ),
        "checks.required_condition_matrix": 1.0,
        "checks.seed_reliability_measured": float(seed_count >= 3),
        "checks.builder_failure_rate_le_0_05": float(failure_ok),
        "checks.leakage_gate_100_pct": float(leakage_ok),
        "checks.budget_within_10_pct": float(budget_ok),
        "checks.erosion_panel_60": float(len(panel) == EROSION_PANEL_SIZE),
        "checks.erosion_panel_complete": float(erosion_complete),
        "checks.exact_rerun_identical": float(exact_rerun_identical),
        "checks.dual_gold_agreement_ge_0_98": float(gold_agreement),
        "checks.pilot_kappa_ge_0_8": float(pilot_ready),
        "checks.claim_disagreement_le_0_02": float(claims_ready),
        "checks.s1_review_passed": float(s1_review_ready),
        "checks.s4_recall_at_10_ge_0_7": float(
            np.isfinite(recall_value) and recall_value >= 0.7
        ),
        "s4_recall_at_10": recall_value,
        "gate_pass_count": float((gate["status"] == "PASS").sum()),
        "gate_exclusion_count": float((gate["status"] == "FAIL").sum()),
        **floor_metrics,
    }
    accuracy_groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in curve.to_dict(orient="records"):
        accuracy_groups[(str(row["layout"]), int(row["budget"]))].append(
            float(row["macro_acc"])
        )
    for (layout, budget), values in accuracy_groups.items():
        metrics[f"acc.{layout}.{budget}"] = float(np.mean(values))
    primary = {
        str(row["contrast"]): {
            "delta": row["delta"],
            "ci_low": row["ci_low"],
            "ci_high": row["ci_high"],
            "seed_spread": row["seed_spread"],
            "reliable_difference": bool(row["reliable_difference"]),
            "valid": bool(row["valid"]),
            "verdict": row["verdict"],
        }
        for row in contrasts.to_dict(orient="records")
    }
    check_details: dict[str, dict[str, Any]] = {}
    if smoke:
        check_details.update(
            {
                "erosion_panel_60": {
                    "passed": None,
                    "value": "not-applicable-in-smoke",
                    "reason": f"smoke config selects a fixed {erosion_probe_limit}-probe panel; the evidentiary profile fixes 60",
                },
                "budget_within_10_pct": {
                    "passed": None,
                    "value": "not-applicable-in-smoke",
                    "reason": "the reduced smoke stores do not fill the registered token budget",
                },
                "dual_gold_agreement_ge_0_98": {
                    "passed": None,
                    "value": "not-applicable-in-smoke",
                    "reason": "the independent frozen evidentiary gold audit is outside smoke",
                },
                "pilot_kappa_ge_0_8": {
                    "passed": None,
                    "value": "not-applicable-in-smoke",
                    "reason": "the preregistered two-human pilot is outside smoke",
                },
                "claim_disagreement_le_0_02": {
                    "passed": None,
                    "value": "not-applicable-in-smoke",
                    "reason": "the repeated claim-grader audit is outside smoke",
                },
                "s1_review_passed": {
                    "passed": None,
                    "value": "not-applicable-in-smoke",
                    "reason": "the 20-folder S1 human review is required only for evidentiary comparison",
                },
                "seed_reliability_measured": {
                    "passed": None,
                    "value": "not-applicable-in-smoke",
                    "reason": "smoke builds one configured S3 seed; reliability inference requires three Sonnet seeds",
                },
                "s4_recall_at_10_ge_0_7": {
                    "passed": None,
                    "value": "not-applicable-in-smoke",
                    "reason": "smoke uses FakeEmbeddings; the S4 fairness gate applies to the pinned evidentiary embedding model",
                },
            }
        )
    return ExperimentResult(
        name="e3",
        metrics=metrics,
        tables={
            "curve": curve,
            "health": health,
            "health_seed_spread": health_spread,
            "erosion": erosion_table,
            "cost": cost,
            "contrasts": contrasts,
            "gate": gate,
            "s4_recall": s4_recall,
        },
        artifacts=[
            *paths.values(),
            ledger_diff_path,
            panel_path,
            curve_chart,
            erosion_chart,
        ],
        primary=primary,
        check_details=check_details,
    )


class E3Experiment:
    """Registry adapter for store conditions supplied in ``CorpusHandle.meta``."""

    name = "e3"

    def run(
        self, cfg: EngineConfig, corpus: CorpusHandle, out_dir: Path, seed: int
    ) -> ExperimentResult:
        raw_stores = corpus.meta.get("stores", [])
        stores = list(raw_stores) if isinstance(raw_stores, (list, tuple)) else []
        raw_budgets = corpus.meta.get("read_budgets", READ_BUDGETS)
        budgets = tuple(int(value) for value in raw_budgets)
        return evaluate_e3(
            stores=stores,
            cfg=cfg,
            probes=load_probes(corpus.probes_path),
            events=list(corpus.events()),
            out_dir=out_dir,
            seed=seed,
            budgets=budgets,
            erosion_probe_limit=int(
                corpus.meta.get("erosion_panel_size", EROSION_PANEL_SIZE)
            ),
            corpus_name=corpus.name,
            smoke=bool(corpus.meta.get("smoke")),
            validation_audit={
                "dual_gold_agreement": float(corpus.meta.get("dual_gold_agreement", 0.0)),
                "pilot_kappa": float(corpus.meta.get("pilot_kappa", 0.0)),
                "claim_disagreement": float(corpus.meta.get("claim_disagreement", 1.0)),
                "s1_review_passed": bool(corpus.meta.get("s1_review_passed", False)),
            },
        )

    def chart(self, result: ExperimentResult, out_dir: Path) -> list[Path]:
        del result, out_dir
        return []


EXPERIMENT = register_experiment(E3Experiment())

__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "E3Experiment",
    "EROSION_PANEL_SIZE",
    "READ_BUDGETS",
    "StoreCondition",
    "compute_erosion_slope",
    "evaluate_e3",
]
