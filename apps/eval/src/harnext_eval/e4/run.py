"""E4 envelope ablation runner from docs/evaluation-spec.md §7 E4 and D14."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import cohen_kappa_score

from harnext_eval.agents.envelope import Envelope, build, execute_tools
from harnext_eval.config import EngineConfig
from harnext_eval.corpus import CorpusHandle
from harnext_eval.e4.tasks import (
    DEFAULT_BOT_ACCOUNTS,
    GOLD_GROUPS,
    build_tasks,
    select_fast_tasks,
)
from harnext_eval.grade.action import grade_rouge_l, judge_pairwise_stable
from harnext_eval.grade.localisation import localisation_scores
from harnext_eval.providers.factory import make_harness_name, make_llm
from harnext_eval.providers.llm import FakeLLM, LLMProvider, LLMResult
from harnext_eval.providers.tokenizer import tokenizer_for
from harnext_eval.registry import ExperimentResult, register_experiment
from harnext_eval.replay.driver import run_pipeline
from harnext_eval.replay.gate import leakage_gate as replay_leakage_gate
from harnext_eval.report.charts import e4_envelopes
from harnext_eval.stats.stats import holm_bonferroni, mcnemar_test, paired_difference_bca
from harnext_eval.stores.base import StoreHandle
from harnext_eval.stores.layouts import configure_store
from harnext_eval.types import EvalEvent, Probe, SnapshotRef, Task

VARIANTS = ("V0", "V1-N20", "V1-N100", "V2", "V3", "V4", "V5", "V6", "V7", "V8")
CONTRASTS = (
    ("V3", "V1-N20", "primary"),
    ("V3", "V1-N100", "primary"),
    ("V3", "V6", "primary"),
    ("V7", "V3", "secondary"),
    ("V8", "V3", "secondary"),
)
_EVENT_ID_FIELD_RE = re.compile(r"evidence_event_ids[^\n]*", re.I)


class ActionPrediction(BaseModel):
    """Typed E4 action schema mandated by the experiment card."""

    model_config = ConfigDict(extra="ignore")

    assignee_candidates: list[str] = Field(default_factory=list, max_length=3)
    reviewer_candidates: list[str] = Field(default_factory=list, max_length=3)
    component: str | None = None
    duplicate_of: str | None = None
    priority_change: str | None = None
    suspected_locations: list[str] = Field(default_factory=list, max_length=5)
    draft_reply: str = ""
    cited_ids: list[str] = Field(default_factory=list)
    action: str = ""


def _normalise(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def _complete(provider: LLMProvider, envelope: Envelope) -> tuple[ActionPrediction, LLMResult]:
    """Use the shared provider request shape, including for the deterministic fake."""

    result = provider.complete(
        envelope.prefix,
        "\n\n".join(f"## {name}\n{body}" for name, body in envelope.sections.items()),
        json_schema=ActionPrediction.model_json_schema(),
        max_tokens=1_000,
    )
    payload: Any = result.json if result.json is not None else json.loads(result.text)
    return ActionPrediction.model_validate(payload), result


def _hit_at(predicted: Iterable[str], gold: Iterable[str]) -> float:
    targets = {_normalise(value) for value in gold}
    return float(any(_normalise(value) in targets for value in predicted)) if targets else math.nan


def _exact_alternative(predicted: str | None, gold: Iterable[str]) -> float:
    targets = {_normalise(value) for value in gold}
    return float(_normalise(predicted) in targets) if targets else math.nan


def _numeric_metric(metrics: Mapping[str, float | list[str]], name: str) -> float:
    value = metrics[name]
    if not isinstance(value, (int, float)):
        raise TypeError(f"localisation metric {name} was not numeric")
    return float(value)


def _score_action(task: Task, prediction: ActionPrediction) -> dict[str, float]:
    """Compute literal Q; malformed primary gold is rejected rather than reweighted."""

    people = task.gold.get("people", {})
    category = task.gold.get("category", {})
    place = task.gold.get("place", {})
    text = task.gold.get("text", {})
    field_scores = {
        "assignee_hit_at_3": _hit_at(prediction.assignee_candidates, people.get("assignees", [])),
        "reviewer_hit_at_3": _hit_at(prediction.reviewer_candidates, people.get("reviewers", [])),
        "component_em": _exact_alternative(prediction.component, category.get("components", [])),
        "duplicate_em": _exact_alternative(prediction.duplicate_of, category.get("duplicate_of", [])),
        "priority_em": _exact_alternative(prediction.priority_change, category.get("priority_changes", [])),
    }
    available = [value for value in field_scores.values() if not math.isnan(value)]
    required = {_normalise(value) for value in category.get("required_ids", []) if value}
    if not available or not required:
        raise ValueError(f"task {task.task_id} lacks a Q component")
    field_em = statistics.fmean(available)
    cited = {_normalise(value) for value in prediction.cited_ids}
    id_cov = len(required & cited) / len(required)
    local_raw = localisation_scores(
        prediction.suspected_locations,
        _list(place.get("files")),
        k=5,
        agentless_superset=True,
    )
    local = {
        "file_hit_at_5": _numeric_metric(local_raw, "file_hit@5"),
        "file_recall": _numeric_metric(local_raw, "file_recall"),
        "file_precision": _numeric_metric(local_raw, "file_precision"),
        "module_hit": _numeric_metric(local_raw, "module_hit"),
    }
    replies = _list(text.get("replies"))
    rouge = max(
        (grade_rouge_l(task.task_id, prediction.draft_reply, reply).value for reply in replies),
        default=math.nan,
    )
    return {
        **field_scores,
        "field_em": field_em,
        "id_cov": id_cov,
        "Q": (field_em + id_cov) / 2,
        **local,
        "rouge_l": rouge,
    }


def _gold_action(task: Task) -> dict[str, Any]:
    """Strip known-at-T IDs and audit metadata from future action values."""

    result: dict[str, Any] = {}
    for group in GOLD_GROUPS:
        payload = task.gold.get(group)
        if not isinstance(payload, Mapping):
            continue
        excluded = {"decision_times", "event_ids", "required_ids", "modules"}
        result[group] = {key: value for key, value in payload.items() if key not in excluded}
    if task.gold.get("scripted_action"):
        result["action"] = task.gold["scripted_action"]
    return result


def _gold_action_time(task: Task) -> Any:
    """Return the earliest action time only when every declared time is valid."""

    raw_times = [
        value
        for group in GOLD_GROUPS
        if isinstance((payload := task.gold.get(group)), Mapping)
        for value in _list(payload.get("decision_times"))
    ]
    if not raw_times:
        return None
    parsed = []
    for value in raw_times:
        try:
            parsed.append(pd.Timestamp(value).to_pydatetime())
        except (TypeError, ValueError):
            return None
    return min(parsed)


def _task_gold_is_bot_free(task: Task) -> bool:
    people = task.gold.get("people", {})
    values = _list(people.get("assignees")) + _list(people.get("reviewers"))
    bots = {account.casefold() for account in DEFAULT_BOT_ACCOUNTS}
    return all(
        value.casefold() not in bots
        and not value.casefold().endswith("[bot]")
        and not value.casefold().endswith("-bot")
        for value in values
    )


def _task_archetype(task: Task) -> tuple[str, str]:
    trigger = task.gold.get("_trigger_event", {})
    source = str(trigger.get("source", "unknown")) if isinstance(trigger, Mapping) else "unknown"
    return str(task.gold.get("_archetype", task.kind)), source


def _mean(values: Iterable[Any]) -> float:
    numeric = [float(value) for value in values if value is not None and not pd.isna(value)]
    return statistics.fmean(numeric) if numeric else math.nan


def _price(cfg: EngineConfig, input_tokens: int, output_tokens: int) -> float:
    prices = cfg.prices or {}
    input_rate = float(prices.get("input_per_million", 0.0))
    output_rate = float(prices.get("output_per_million", 0.0))
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


def _normalise_variants(variants: Iterable[str]) -> tuple[str, ...]:
    expanded: list[str] = []
    for raw in variants:
        value = raw.upper()
        cells = ("V1-N20", "V1-N100") if value == "V1" else (value,)
        for cell in cells:
            if cell not in VARIANTS:
                raise ValueError(f"unknown E4 variant {raw!r}")
            if cell not in expanded:
                expanded.append(cell)
    return tuple(expanded)


def _event_index(events: Sequence[EvalEvent]) -> dict[str, EvalEvent]:
    return {event.id: event for event in events}


def _validate_evidence(ids: Sequence[str], events: Mapping[str, EvalEvent], cutoff: Any) -> dict[str, Any]:
    missing = sorted({value for value in ids if value not in events})
    post_t = sorted({value for value in ids if value in events and events[value].time > cutoff})
    malformed = sorted({value for value in ids if not value.strip()})
    return {
        "evidence_valid": float(bool(ids) and not missing and not post_t and not malformed),
        "evidence_missing": missing,
        "evidence_post_T": post_t,
        "evidence_malformed": malformed,
    }


def _pr_join_audit(tasks: Sequence[Task]) -> tuple[float, float, bool]:
    """Measure PR-key joins against optional independently audited expected IDs."""

    true_positive = false_positive = false_negative = 0
    audited = 0
    for task in tasks:
        raw = task.gold.get("_join_audit")
        if not isinstance(raw, Mapping):
            continue
        expected = {_normalise(value) for value in _list(raw.get("expected_pr_ids"))}
        observed = {_normalise(value) for value in _list(raw.get("observed_pr_ids"))}
        if not expected and not observed:
            continue
        audited += 1
        true_positive += len(expected & observed)
        false_positive += len(observed - expected)
        false_negative += len(expected - observed)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else math.nan
    recall = true_positive / recall_denominator if recall_denominator else math.nan
    return precision, recall, audited > 0


def _scratch_batch_fold(
    task: Task,
    envelope: Envelope,
    snapshot: SnapshotRef,
    source: StoreHandle,
    cfg: EngineConfig,
    events: Sequence[EvalEvent],
    reader: LLMProvider,
    repetition: int,
    variant: str,
) -> dict[str, Any]:
    """Fold Vx on an isolated S3 copy, then grade the post-delta state with E2."""

    from harnext_eval.agents.reader import Material
    from harnext_eval.agents.reader import answer as read_answer
    from harnext_eval.e2.arms import a4
    from harnext_eval.e2.run import grade_answer

    source_files = {path: source.read(snapshot, path) or "" for path in source.list_files(snapshot)}
    with tempfile.TemporaryDirectory(prefix="harnext-e4-batch-") as temp:
        scratch = StoreHandle("S3", "scratch", Path(temp) / "store")
        configure_store(scratch, harness=make_harness_name(cfg), model=cfg.builder.model)
        for path, content in source_files.items():
            scratch.write(path, content)
        visible_ids = sorted(
            event.id for event in events if event.time <= task.T and event.id in envelope.text
        )
        envelope_digest = hashlib.sha256(envelope.text.encode()).hexdigest()[:12]
        builder_event = EvalEvent(
            id=f"e4-builder-{task.trigger_event_id}-{envelope_digest}-{repetition}",
            source="harnext-eval:e4",
            type="dev.harnext.eval.envelope_fold",
            subject=task.entity,
            time=task.T,
            mgtenant="e4-scratch",
            data={
                "envelope_variant": variant,
                "context_envelope": envelope.text,
                "evidence_event_ids": visible_ids,
            },
        )
        ref = scratch.fold([builder_event], "batch")
        after_files = {path: scratch.read(ref, path) or "" for path in scratch.list_files(ref)}
        changed = sorted(
            path for path in set(source_files) | set(after_files)
            if source_files.get(path) != after_files.get(path)
        )
        delta_text = "\n".join(after_files.get(path, "") for path in changed)
        emitted = sorted({event_id for event_id in _event_index(events) if event_id in delta_text})
        evidence = _validate_evidence(emitted, _event_index(events), task.T)
        grades: list[float] = []
        probe_grades: list[dict[str, Any]] = []
        reader_tokens = 0
        reader_latency = 0.0
        for raw_probe in task.gold.get("probes", []):
            probe = Probe.model_validate(raw_probe).model_copy(update={"T": task.T})
            material = a4(probe, scratch, cfg)
            kind, _, slug = probe.entity.partition(":")
            canonical_prefix = f"entities/{kind}/{slug}/" if slug else f"entities/{kind}/"
            if canonical_prefix.casefold() not in material.text.casefold():
                entity_paths = [
                    path for path in scratch.list_files(ref)
                    if path.casefold().startswith(canonical_prefix.casefold())
                ]
                entity_text = "\n".join(
                    f"[file:{path}]\n{scratch.read(ref, path) or ''}" for path in entity_paths
                )
                material = Material(
                    arm="A4",
                    text=entity_text,
                    source_ids=emitted,
                    tool_calls=len(entity_paths),
                )
            response = read_answer(probe, material, cfg, provider=reader)
            grade = grade_answer(probe, response.text)
            grades.append(grade.value)
            probe_grades.append(
                {"probe_id": probe.probe_id, "answer": response.text, "grade": grade.value}
            )
            reader_tokens += response.tokens_read
            reader_latency += response.latency_s
        usage_path = scratch.root / "usage.jsonl"
        usage = json.loads(usage_path.read_text(encoding="utf-8").splitlines()[-1]) if usage_path.exists() else {}
        input_tokens = int(usage.get("input_tokens", usage.get("tokens_in", 0))) + reader_tokens
        output_tokens = int(usage.get("output_tokens", usage.get("tokens_out", 0)))
        return {
            "batch_e2_acc": _mean(grades),
            "batch_result_sha": ref.sha,
            "batch_delta_files": changed,
            "batch_probe_count": len(grades),
            "batch_probe_grades": probe_grades,
            "tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_s": reader_latency,
            **evidence,
        }


def _calibrate_judge(
    provider: LLMProvider | None,
    records: Sequence[Mapping[str, Any]],
) -> tuple[float, bool, list[dict[str, Any]]]:
    if provider is None or len(records) < 200:
        return math.nan, False, []
    human: list[bool] = []
    predicted: list[bool] = []
    details: list[dict[str, Any]] = []
    for index, record in enumerate(records[:200]):
        left = bool(record.get("human_a"))
        right = bool(record.get("human_b"))
        if left != right:
            continue
        grade = judge_pairwise_stable(
            f"calibration-{index}",
            str(record.get("candidate", "")),
            str(record.get("baseline", "")),
            provider,
            criterion=str(record.get("criterion", "quality, correctness, and usefulness")),
        )
        human.append(left)
        predicted.append(bool(grade.value))
        details.append(dict(grade.details))
    if len(human) < 2 or len(set(human)) < 2:
        return math.nan, False, details
    kappa = float(cohen_kappa_score(human, predicted))
    return kappa, kappa >= 0.6, details


def _contrast_table(
    run_rows: Sequence[Mapping[str, Any]],
    task_by_id: Mapping[str, Task],
    *,
    seed: int,
    practical_threshold: float,
) -> pd.DataFrame:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    resources: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for record in run_rows:
        task_id, variant = str(record["task_id"]), str(record["variant"])
        quality = record.get("Q")
        if isinstance(quality, (int, float)) and not math.isnan(float(quality)):
            values[(task_id, variant)].append(float(quality))
        for metric in ("tokens", "output_tokens", "cost_usd", "latency_s", "tool_calls"):
            resources[(task_id, variant, metric)].append(float(record.get(metric, 0.0)))
    means = {key: statistics.fmean(items) for key, items in values.items()}
    rows: list[dict[str, Any]] = []
    for left, right, family in CONTRASTS:
        task_ids = sorted(
            task_id for task_id, variant in means
            if variant == left and (task_id, right) in means
        )
        left_values = [means[(task_id, left)] for task_id in task_ids]
        right_values = [means[(task_id, right)] for task_id in task_ids]
        entities = [task_by_id[task_id].entity for task_id in task_ids]
        effect = _mean(a - b for a, b in zip(left_values, right_values, strict=True))
        inference_valid = len(set(entities)) >= 2 and bool(task_ids)
        ci_low = ci_high = math.nan
        if inference_valid:
            bca = paired_difference_bca(
                left_values, right_values, entities,
                n_resamples=10_000, random_state=seed,
            )
            ci_low, ci_high = bca.ci_low, bca.ci_high
        mcnemar = mcnemar_test(
            [value >= 1.0 for value in left_values],
            [value >= 1.0 for value in right_values],
        ) if task_ids else None
        row: dict[str, Any] = {
            "contrast": f"{left}-{right}", "family": family, "left": left, "right": right,
            "n": len(task_ids), "entities": len(set(entities)), "mean_delta_Q": effect,
            "ci_low": ci_low, "ci_high": ci_high, "confidence_level": 0.95,
            "bca_resamples": 10_000, "inference_valid": inference_valid,
            "mcnemar_p": mcnemar.p_value if mcnemar else math.nan,
            "discordant_left": mcnemar.b if mcnemar else 0,
            "discordant_right": mcnemar.c if mcnemar else 0,
            "practical_threshold": practical_threshold,
            "practically_meaningful": bool(not math.isnan(effect) and abs(effect) >= practical_threshold),
        }
        for metric in ("tokens", "output_tokens", "cost_usd", "latency_s", "tool_calls"):
            row[f"mean_delta_{metric}"] = _mean(
                _mean(resources[(task_id, left, metric)]) - _mean(resources[(task_id, right, metric)])
                for task_id in task_ids
            )
        rows.append(row)
    secondary = {
        row["contrast"]: float(row["mcnemar_p"])
        for row in rows
        if row["family"] == "secondary" and not pd.isna(row["mcnemar_p"])
    }
    adjusted = holm_bonferroni(secondary)
    holm = adjusted.set_index("hypothesis").to_dict("index") if not adjusted.empty else {}
    for row in rows:
        correction = holm.get(row["contrast"], {})
        row["holm_adjusted_p"] = correction.get("adjusted_p", math.nan)
        row["holm_reject"] = correction.get("reject", False)
    return pd.DataFrame(rows)


def aggregate_seed_outputs(e4_dir: Path, *, required_seeds: int = 3) -> pd.DataFrame:
    """Aggregate paired E4 effects across completed S3 seed directories."""

    by_contrast: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for path in sorted(e4_dir.glob("seed-*/contrasts.csv")):
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            value = row.get("mean_delta_Q")
            numeric = float(value) if value is not None else math.nan
            if not math.isnan(numeric):
                by_contrast[str(row["contrast"])].append((path.parent.name, numeric))
    rows = []
    for contrast, values in sorted(by_contrast.items()):
        effects = [value for _, value in values]
        rows.append(
            {
                "contrast": contrast,
                "seeds": len(values),
                "seed_ids": "|".join(seed for seed, _ in values),
                "mean_effect": statistics.fmean(effects),
                "seed_spread_sd": statistics.stdev(effects) if len(effects) >= 2 else math.nan,
                "status": "complete" if len(values) >= required_seeds else "supported-not-run",
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=["contrast", "seeds", "seed_ids", "mean_effect", "seed_spread_sd", "status"],
    )
    e4_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(e4_dir / "seed_spread.csv", index=False)
    return frame


def run_e4(
    tasks: Iterable[Task],
    store: StoreHandle,
    cfg: EngineConfig,
    out_dir: Path,
    *,
    provider: LLMProvider | None = None,
    variants: Iterable[str] = VARIANTS,
    runs: int = 3,
    events: Iterable[EvalEvent] = (),
    seed: int = 0,
    expected_fast_tasks: int = 150,
    expected_batch_tasks: int = 150,
    practical_threshold: float = 0.10,
    judge_provider: LLMProvider | None = None,
    judge_calibration: Sequence[Mapping[str, Any]] = (),
    judge_model_family: str | None = None,
) -> ExperimentResult:
    """Run the paired task × envelope × repetition matrix on one fixed S3."""

    if store.layout != "S3":
        raise ValueError(f"E4 requires a fixed S3 store, got {store.layout}")
    if runs <= 0:
        raise ValueError("runs must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_variants = _normalise_variants(variants)
    provider = provider or make_llm(cfg)
    token_counter = tokenizer_for(provider)
    provider_model = str(
        getattr(provider, "model", getattr(provider, "model_id", type(provider).__name__))
    )
    judge_kappa, judge_usable, _ = _calibrate_judge(judge_provider, judge_calibration)
    action_family = str(getattr(provider, "model", getattr(provider, "model_id", type(provider).__name__)))
    model_family_ok = (
        judge_model_family is not None
        and judge_model_family.casefold() not in action_family.casefold()
    )
    judge_used = judge_usable and model_family_ok
    event_list = list(events)
    task_list = [task for task in tasks if _task_gold_is_bot_free(task)]
    task_by_id = {task.task_id: task for task in task_list}
    run_rows: list[dict[str, Any]] = []
    size_rows: list[dict[str, Any]] = []
    descriptors: list[dict[str, str]] = []
    manual_gate_rows: list[dict[str, str]] = []
    gate_api = out_dir / "gate-api.csv"
    if gate_api.exists():
        raise FileExistsError(f"E4 output is not empty: {gate_api}")
    excluded_tasks: set[str] = set()

    for task in sorted(task_list, key=lambda item: item.task_id):
        try:
            snapshot = store.snapshot(task.T)
        except LookupError:
            excluded_tasks.add(task.task_id)
            manual_gate_rows.append(
                {
                    "task_id": task.task_id, "variant": "", "probe_id": "",
                    "item_id": task.task_id, "T": task.T.isoformat(), "sha": "",
                    "last_event_id": "", "result": "FAIL", "reasons": "snapshot_unavailable",
                }
            )
            continue
        built: dict[str, Envelope] = {}
        task_passes = True
        for variant in selected_variants:
            envelope = build(
                task,
                snapshot,
                variant,
                {
                    "store_handle": store,
                    "events": event_list,
                    "token_counter": token_counter,
                },
            )
            if variant == "V5":
                envelope = execute_tools(
                    envelope,
                    queries={"read_state": task.entity, "search_facts": task.entity, "recent_events": "20"},
                    budget_tokens=12_000,
                )
            built[variant] = envelope
            size_rows.append(
                {
                    "task_id": task.task_id, "entity": task.entity, "variant": variant,
                    "snapshot_sha": snapshot.sha, "tokens": envelope.token_count,
                    "tokenizer_id": token_counter.tokenizer_id,
                    "tokenizer_revision": token_counter.tokenizer_revision,
                    **{f"tokens_{name}": count for name, count in envelope.tokens_by_section.items()},
                }
            )
            if task.kind == "fast":
                passed = replay_leakage_gate(
                    task, store=store, T=task.T, all_events=event_list,
                    envelope=envelope.text, gold_action=_gold_action(task),
                    gold_action_time=_gold_action_time(task), out_csv=gate_api,
                )
                descriptors.append({"task_id": task.task_id, "variant": variant, "probe_id": ""})
                task_passes &= passed
            else:
                probes = [Probe.model_validate(raw) for raw in task.gold.get("probes", [])]
                if not probes:
                    task_passes = False
                for probe in probes:
                    passed = replay_leakage_gate(
                        probe, store=store, T=task.T, question=probe.question,
                        all_events=event_list, material=envelope.text, out_csv=gate_api,
                    )
                    descriptors.append(
                        {"task_id": task.task_id, "variant": variant, "probe_id": probe.probe_id}
                    )
                    task_passes &= passed
        if not task_passes:
            excluded_tasks.add(task.task_id)
            continue
        for variant in selected_variants:
            envelope = built[variant]
            for repetition in range(1, runs + 1):
                if task.kind == "batch":
                    outcome = _scratch_batch_fold(
                        task, envelope, snapshot, store, cfg, event_list, provider, repetition,
                        variant,
                    )
                    input_tokens = int(outcome["tokens"])
                    output_tokens = int(outcome["output_tokens"])
                    run_rows.append(
                        {
                            "task_id": task.task_id, "entity": task.entity, "kind": task.kind,
                            "variant": variant, "run": repetition, "snapshot_sha": snapshot.sha,
                            "provider_model": provider_model,
                            "tool_calls": envelope.observed_tool_calls,
                            "cost_usd": _price(cfg, input_tokens, output_tokens),
                            "prediction": None, "Q": math.nan, **outcome,
                        }
                    )
                    continue
                started = time.perf_counter()
                prediction, result = _complete(provider, envelope)
                latency = time.perf_counter() - started
                input_tokens = int(result.usage.get("input_tokens", envelope.token_count))
                output_tokens = int(result.usage.get("output_tokens", 0))
                scores = _score_action(task, prediction)
                judge_win = math.nan
                if judge_used:
                    assert judge_provider is not None
                    replies = _list(task.gold.get("text", {}).get("replies"))
                    criterion = (
                        "Quality and correctness relative to this reference reply: "
                        + (replies[0] if replies else "no reference")
                    )
                    judge_win = judge_pairwise_stable(
                        f"{task.task_id}:{variant}:{repetition}",
                        prediction.draft_reply,
                        str(task.gold.get("judge_baseline", "")),
                        judge_provider,
                        criterion=criterion,
                    ).value
                run_rows.append(
                    {
                        "task_id": task.task_id, "entity": task.entity, "kind": task.kind,
                        "variant": variant, "run": repetition, "snapshot_sha": snapshot.sha,
                        "provider_model": provider_model,
                        "tokens": input_tokens, "output_tokens": output_tokens,
                        "cost_usd": _price(cfg, input_tokens, output_tokens), "latency_s": latency,
                        "tool_calls": envelope.observed_tool_calls,
                        "prediction": prediction.model_dump(mode="json"),
                        "batch_e2_acc": math.nan,
                        "evidence_valid": math.nan,
                        "evidence_missing": [],
                        "evidence_post_T": [],
                        "evidence_malformed": [],
                        "judge_win": judge_win,
                        **scores,
                    }
                )

    gate_columns = ["probe_id", "item_id", "T", "sha", "last_event_id", "result", "reasons"]
    gate_frame = pd.read_csv(gate_api) if gate_api.exists() else pd.DataFrame(columns=gate_columns)
    if len(gate_frame) != len(descriptors):
        if descriptors and gate_frame.empty:
            gate_frame = pd.DataFrame(
                [{**descriptor, "item_id": descriptor["task_id"], "T": "", "sha": "", "last_event_id": "", "result": "FAIL", "reasons": "snapshot_unavailable"} for descriptor in descriptors]
            )
        else:
            raise AssertionError("shared leakage gate rows do not match E4 calls")
    else:
        gate_frame.insert(0, "task_id", [item["task_id"] for item in descriptors])
        gate_frame.insert(1, "variant", [item["variant"] for item in descriptors])
        gate_frame["probe_id"] = [item["probe_id"] or value for item, value in zip(descriptors, gate_frame["probe_id"], strict=True)]
    if manual_gate_rows:
        gate_frame = pd.concat([gate_frame, pd.DataFrame(manual_gate_rows)], ignore_index=True)
    gate_path = out_dir / "gate.csv"
    gate_frame.to_csv(gate_path, index=False)

    accepted_ids = sorted(set(task_by_id) - excluded_tasks)
    for variant in selected_variants:
        variant_ids = sorted({str(row["task_id"]) for row in run_rows if row["variant"] == variant})
        if variant_ids != accepted_ids:
            raise AssertionError(f"unpaired E4 population for {variant}: {variant_ids} != {accepted_ids}")

    metric_names = (
        "Q", "field_em", "id_cov", "assignee_hit_at_3", "reviewer_hit_at_3",
        "component_em", "duplicate_em", "priority_em", "file_hit_at_5", "file_recall",
        "file_precision", "module_hit", "rouge_l", "judge_win", "batch_e2_acc",
        "evidence_valid", "tokens", "output_tokens", "cost_usd", "latency_s", "tool_calls",
    )
    metric_rows: list[dict[str, Any]] = []
    fast_tasks = [task for task in task_list if task.kind == "fast"]
    for variant in selected_variants:
        subset = [row for row in run_rows if row["variant"] == variant]
        row: dict[str, Any] = {"variant": variant, "tasks": len({item["task_id"] for item in subset})}
        row.update({metric: _mean(item.get(metric) for item in subset) for metric in metric_names})
        correctness: dict[str, list[bool]] = defaultdict(list)
        for item in subset:
            quality = item.get("Q")
            if isinstance(quality, (int, float)) and not math.isnan(float(quality)):
                correctness[str(item["task_id"])].append(float(quality) >= 1.0)
        observed_pass3 = _mean(
            float(len(values) == runs and all(values)) for values in correctness.values()
        )
        row["pass3_observed"] = observed_pass3
        row["pass3"] = math.nan if isinstance(provider, FakeLLM) or runs != 3 else observed_pass3
        row["median_tokens"] = statistics.median(
            [item["tokens"] for item in size_rows if item["variant"] == variant]
        ) if any(item["variant"] == variant for item in size_rows) else math.nan
        for group in GOLD_GROUPS:
            row[f"coverage_{group}"] = _mean(
                float(task.gold_coverage.get(group, False)) for task in fast_tasks
            )
        probe_families = [
            Probe.model_validate(raw).family
            for task in task_list if task.kind == "batch"
            for raw in task.gold.get("probes", [])
        ]
        for family in sorted(set(probe_families)):
            row[f"batch_coverage_{family}"] = probe_families.count(family)
        metric_rows.append(row)
    metrics_frame = pd.DataFrame(metric_rows)
    contrasts_frame = _contrast_table(
        run_rows, task_by_id, seed=seed, practical_threshold=practical_threshold
    )
    sizes_frame = pd.DataFrame(size_rows)

    balance_counts: dict[tuple[str, str], int] = defaultdict(int)
    for task in fast_tasks:
        balance_counts[_task_archetype(task)] += 1
    balance_frame = pd.DataFrame(
        [
            {"archetype": archetype, "source": source, "count": count}
            for (archetype, source), count in sorted(balance_counts.items())
        ],
        columns=["archetype", "source", "count"],
    )
    archetype_totals: dict[str, int] = defaultdict(int)
    for (archetype, _), count in balance_counts.items():
        archetype_totals[archetype] += count
    max_share = max(archetype_totals.values()) / sum(archetype_totals.values()) if archetype_totals else math.nan
    medians = {str(row["variant"]): float(row["median_tokens"]) for row in metric_rows}
    v3, v6 = medians.get("V3", math.nan), medians.get("V6", math.nan)
    fast_count = sum(task_by_id[task_id].kind == "fast" for task_id in accepted_ids)
    batch_count = sum(task_by_id[task_id].kind == "batch" for task_id in accepted_ids)
    fake = isinstance(provider, FakeLLM)
    arm_q = [
        float(quality)
        for row in metric_rows
        if isinstance((quality := row.get("Q")), (int, float))
        and not math.isnan(float(quality))
    ]
    arm_spread = max(arm_q) - min(arm_q) if arm_q else math.nan
    q_by_variant = {
        str(row["variant"]): float(row["Q"])
        for row in metric_rows
        if isinstance(row.get("Q"), (int, float)) and not math.isnan(float(row["Q"]))
    }
    non_vacuous_arms = bool(
        q_by_variant.get("V0", math.nan) < q_by_variant.get("V3", math.nan)
        and not math.isclose(
            q_by_variant.get("V6", math.nan),
            q_by_variant.get("V3", math.nan),
        )
    )
    repeated_predictions: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in run_rows:
        if row.get("prediction") is not None:
            repeated_predictions[(str(row["task_id"]), str(row["variant"]))].add(
                json.dumps(row["prediction"], sort_keys=True)
            )
    repetition_variation = any(len(values) > 1 for values in repeated_predictions.values())
    pr_precision, pr_recall, pr_audited = _pr_join_audit(fast_tasks)

    mandatory = {
        "s3_fixed_store": store.layout == "S3",
        "leakage_gate_100_pct": bool(task_list) and not excluded_tasks,
        "v3_median_le_12k": not math.isnan(v3) and v3 <= 12_000,
        "v6_median_ge_3x_v3": not math.isnan(v3) and not math.isnan(v6) and v6 >= 3 * v3,
        "task_balance": not math.isnan(max_share) and max_share <= 0.40,
        "sample_cells": fast_count >= expected_fast_tasks and batch_count >= expected_batch_tasks,
        "three_runs": runs == 3,
        "real_action_provider": not fake,
        "non_vacuous_arms": non_vacuous_arms,
        "primary_inference": bool(not contrasts_frame.empty and contrasts_frame.loc[contrasts_frame["family"] == "primary", "inference_valid"].all()),
        "pr_join_audited": pr_audited or not any(task.gold_coverage.get("place") for task in fast_tasks),
    }
    invalid_reasons = sorted(name for name, passed in mandatory.items() if not passed)
    checks: dict[str, bool | float] = {
        **{f"checks.{name}": passed for name, passed in mandatory.items()},
        "checks.gold_actions_after_T": all(
            (action_time := _gold_action_time(task)) is not None and action_time > task.T
            for task in fast_tasks
        ),
        "checks.bot_accounts_excluded": all(_task_gold_is_bot_free(task) for task in task_list),
        "checks.judge_kappa_ge_0_6": judge_usable,
        "checks.position_swapped": judge_used,
        "checks.repetition_variation_observed": repetition_variation,
        "checks.pr_join_precision": pr_precision,
        "checks.pr_join_recall": pr_recall,
        "max_archetype_share": max_share,
        "gate_pass_count": float((gate_frame.get("result") == "PASS").sum()) if not gate_frame.empty else 0.0,
        "gate_exclusion_count": float(len(excluded_tasks)),
        "tasks_accepted": float(len(accepted_ids)),
        "fast_tasks": float(fast_count),
        "batch_tasks": float(batch_count),
        "arm_Q_spread": arm_spread,
    }
    judge_frame = pd.DataFrame(
        [{
            "corpus": task_list[0].corpus if task_list else "", "n_calibration": len(judge_calibration),
            "judge_kappa": judge_kappa, "used": judge_used,
            "reason": "used" if judge_used else "calibration/different-family requirement unavailable",
        }]
    )
    artifacts = [
        out_dir / "runs.jsonl", out_dir / "metrics.csv", out_dir / "contrasts.csv",
        out_dir / "sizes.csv", gate_path, out_dir / "judge_kappa.csv", out_dir / "balance.csv",
    ]
    artifacts[0].write_text(
        "".join(json.dumps(dict(row), sort_keys=True, default=str) + "\n" for row in run_rows),
        encoding="utf-8",
    )
    metrics_frame.to_csv(artifacts[1], index=False)
    contrasts_frame.to_csv(artifacts[2], index=False)
    sizes_frame.to_csv(artifacts[3], index=False)
    judge_frame.to_csv(artifacts[5], index=False)
    balance_frame.to_csv(artifacts[6], index=False)
    seed_spread = (
        aggregate_seed_outputs(out_dir.parent)
        if out_dir.name.startswith("seed-")
        else pd.DataFrame()
    )
    if out_dir.name.startswith("seed-"):
        artifacts.append(out_dir.parent / "seed_spread.csv")
    primary = {
        "evidence_status": "plumbing-only" if fake else "substantive",
        "valid_primary": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "contrasts": {
            str(record["contrast"]): float(record["mean_delta_Q"])
            for record in contrasts_frame.to_dict("records")
            if record["family"] == "primary"
        }
        if not invalid_reasons
        else {},
    }
    check_details: dict[str, dict[str, Any]] = {}
    if fake:
        smoke_reasons = {
            "judge_kappa_ge_0_6": "the 200-pair human judge calibration is outside offline smoke",
            "position_swapped": "pairwise judging is disabled until the calibrated different-family judge is supplied",
            "real_action_provider": "offline smoke intentionally uses the deterministic FakeLLM action provider",
            "repetition_variation_observed": "deterministic FakeLLM smoke runs once per task/arm by design",
            "primary_inference": "the one-run tiny smoke matrix validates plumbing, not the preregistered inference",
            "sample_cells": "smoke uses the available injected scenarios, not 150 fast plus 150 batch tasks",
            "three_runs": "smoke uses one deterministic repetition; evidentiary pass^3 uses three runs",
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
        name="e4", metrics=checks,
        tables={
            "metrics": metrics_frame, "contrasts": contrasts_frame, "sizes": sizes_frame,
            "gate": gate_frame, "balance": balance_frame, "judge_kappa": judge_frame,
            "seed_spread": seed_spread,
        },
        artifacts=artifacts,
        primary=primary,
        check_details=check_details,
    )


def _build_experiment_store(
    events: list[EvalEvent], tasks: list[Task], cfg: EngineConfig, out_dir: Path, seed: int
) -> StoreHandle:
    """Build the fixed S3 store through the shared replay/layout modules."""

    store = StoreHandle("S3", f"e4-{seed}", out_dir / "store")
    configure_store(store, harness=make_harness_name(cfg), model=cfg.builder.model)
    cutoff = max((task.T for task in tasks), default=None)
    run_pipeline(events, cfg, store, cutoff=cutoff, on_decision=None)
    return store


def _load_probes(path: Path | None) -> list[Probe]:
    if path is None or not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return [Probe.model_validate_json(line) for line in source if line.strip()]


def _load_tasks(path: Path | None) -> list[Task]:
    if path is None or not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return [Task.model_validate_json(line) for line in source if line.strip()]


def _attach_join_audits(tasks: Sequence[Task], meta: Mapping[str, Any]) -> list[Task]:
    raw = meta.get("e4_pr_join_audit")
    if not isinstance(raw, Mapping):
        return list(tasks)
    result: list[Task] = []
    for task in tasks:
        audit = raw.get(task.task_id, raw.get(task.trigger_event_id))
        if isinstance(audit, Mapping):
            result.append(task.model_copy(update={"gold": {**task.gold, "_join_audit": dict(audit)}}))
        else:
            result.append(task)
    return result


class E4Experiment:
    name = "e4"

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider

    def run(self, cfg: EngineConfig, corpus: CorpusHandle, out_dir: Path, seed: int) -> ExperimentResult:
        events = list(corpus.events())
        tasks = _load_tasks(corpus.tasks_path)
        if not tasks:
            catalog = corpus.meta.get("actor_catalog", {})
            committers = catalog.get("committers", []) if isinstance(catalog, Mapping) else []
            try:
                tasks = build_tasks(
                    events, corpus=corpus.name, probes=_load_probes(corpus.probes_path),
                    window=cfg.window, committer_accounts=committers,
                    corpus_meta=corpus.meta, seed=seed,
                    fast_limit=5 if corpus.meta.get("smoke") else 150,
                    batch_limit=5 if corpus.meta.get("smoke") else 150,
                )
            except ValueError:
                if not corpus.meta.get("smoke"):
                    raise
                tasks = select_fast_tasks(
                    events, corpus=corpus.name, committer_accounts=committers,
                    limit=5, seed=seed,
                )
        tasks = _attach_join_audits(tasks, corpus.meta)
        meta_store = corpus.meta.get("store_handle") or corpus.meta.get("store")
        store = meta_store if isinstance(meta_store, StoreHandle) and meta_store.layout == "S3" else _build_experiment_store(events, tasks, cfg, out_dir, seed)
        smoke = bool(corpus.meta.get("smoke"))
        return run_e4(
            tasks, store, cfg, out_dir, provider=self.provider,
            variants=("V0", "V1", "V3", "V5", "V6", "V7", "V8") if smoke else VARIANTS,
            runs=1 if smoke else 3, events=events, seed=seed,
            expected_fast_tasks=5 if smoke else 150,
            expected_batch_tasks=0 if smoke else 150,
            judge_calibration=corpus.meta.get("e4_judge_calibration", ()),
            judge_model_family=corpus.meta.get("e4_judge_model_family"),
        )

    def chart(self, result: ExperimentResult, out_dir: Path) -> list[Path]:
        table = result.tables["metrics"]
        if table.empty:
            return []
        return [e4_envelopes(table.rename(columns={"variant": "envelope", "median_tokens": "tokens"}), out_dir)]


register_experiment(E4Experiment())

__all__ = ["ActionPrediction", "E4Experiment", "aggregate_seed_outputs", "run_e4"]
