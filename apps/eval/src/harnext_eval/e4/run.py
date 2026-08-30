"""E4 envelope ablation runner from docs/evaluation-spec.md §7 E4 and D14."""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from harnext_eval.agents.envelope import Envelope, build
from harnext_eval.config import EngineConfig
from harnext_eval.corpus import CorpusHandle
from harnext_eval.e4.tasks import (
    DEFAULT_BOT_ACCOUNTS,
    GOLD_GROUPS,
    build_tasks,
)
from harnext_eval.grade.action import grade_action, grade_rouge_l
from harnext_eval.grade.localisation import localisation_scores
from harnext_eval.providers.factory import make_harness_name, make_llm
from harnext_eval.providers.llm import FakeLLM, LLMProvider, LLMResult
from harnext_eval.providers.tokenizer import count_tokens
from harnext_eval.registry import ExperimentResult, register_experiment
from harnext_eval.replay.driver import run_pipeline
from harnext_eval.report.charts import e4_envelopes
from harnext_eval.stores.base import StoreHandle
from harnext_eval.stores.layouts import configure_store
from harnext_eval.types import EvalEvent, Probe, SnapshotRef, Task

VARIANTS = tuple(f"V{index}" for index in range(9))
CONTRASTS = (("V3", "V1"), ("V3", "V6"), ("V7", "V3"), ("V8", "V3"))
_ID_RE = re.compile(
    r"\b(?:[A-Z][A-Z0-9]+-\d+|KIP-\d+|CVE-\d{4}-\d+|[a-f0-9]{16,64})\b",
    re.IGNORECASE,
)
_PATH_RE = re.compile(r"(?<![\w.-])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+")


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


def _field_values(text: str, field: str) -> list[str]:
    pattern = re.compile(
        rf"[\"']?{re.escape(field)}[\"']?\s*(?::|=|\bis\b)\s*(?:\[)?[\"']?([^\]\n,;}}]+)",
        re.IGNORECASE,
    )
    values: list[str] = []
    for match in pattern.finditer(text):
        value = match.group(1).strip().strip("\"'.")
        if value and _normalise(value) not in {_normalise(item) for item in values}:
            values.append(value)
    return values


def _fake_prediction(envelope: Envelope) -> ActionPrediction:
    """Produce a deterministic, evidence-derived action for foundation FakeLLM."""

    text = envelope.text
    assignees = _field_values(text, "assignee")
    reviewers = _field_values(text, "reviewer")
    components = _field_values(text, "component") + _field_values(text, "components")
    duplicates = _field_values(text, "duplicate_of") + _field_values(text, "duplicate-of")
    priorities = _field_values(text, "priority_change") + _field_values(text, "priority")
    paths = []
    for path in _PATH_RE.findall(text):
        if path.startswith(("entities/", "_meta/")):
            continue
        if path not in paths:
            paths.append(path)
    cited_ids: list[str] = []
    for match in _ID_RE.finditer(text):
        value = match.group(0)
        if _normalise(value) not in {_normalise(item) for item in cited_ids}:
            cited_ids.append(value)
    significant = re.search(r"(?:\bcritical\b|\bblocker\b|\[vote\]|\bcve-)", text, re.I)
    summary_line = next(
        (
            line.lstrip("#- ").strip()
            for line in envelope.sections.get("overview", "").splitlines()
            if line.strip() and not line.lstrip().startswith("[")
        ),
        "I am reviewing the triggering event and its cited state.",
    )
    return ActionPrediction(
        assignee_candidates=assignees[-3:],
        reviewer_candidates=reviewers[-3:],
        component=components[-1] if components else None,
        duplicate_of=duplicates[-1] if duplicates else None,
        priority_change=priorities[-1] if priorities else None,
        suspected_locations=paths[:5],
        draft_reply=summary_line,
        cited_ids=cited_ids,
        action="escalate_and_route" if significant else "route_and_reply",
    )


def _complete(provider: LLMProvider, envelope: Envelope) -> tuple[ActionPrediction, LLMResult]:
    if isinstance(provider, FakeLLM):
        prediction = _fake_prediction(envelope)
        payload = prediction.model_dump(mode="json")
        rendered = json.dumps(payload, sort_keys=True)
        result = LLMResult(
            text=rendered,
            json=payload,
            usage={
                "input_tokens": envelope.token_count,
                "output_tokens": count_tokens(rendered),
            },
        )
        return prediction, result
    result = provider.complete(
        envelope.prefix,
        "\n\n".join(f"## {name}\n{body}" for name, body in envelope.sections.items()),
        json_schema=ActionPrediction.model_json_schema(),
        max_tokens=1_000,
    )
    payload: Any = result.json
    if payload is None:
        payload = json.loads(result.text)
    return ActionPrediction.model_validate(payload), result


def _hit_at(predicted: Iterable[str], gold: Iterable[str]) -> float:
    gold_values = {_normalise(value) for value in gold}
    return (
        float(any(_normalise(value) in gold_values for value in predicted))
        if gold_values
        else math.nan
    )


def _exact_one(predicted: str | None, gold: Iterable[str]) -> float:
    gold_values = {_normalise(value) for value in gold}
    return float(_normalise(predicted) in gold_values) if gold_values else math.nan


def _module(path: str) -> str:
    parts = [part for part in path.replace("\\", "/").split("/") if part]
    return "/".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "")


def _location_match(predicted: str, gold: str) -> bool:
    pred = predicted.replace("\\", "/").strip("/ ").casefold()
    target = gold.replace("\\", "/").strip("/ ").casefold()
    return pred == target or target.startswith(f"{pred}/")


def _local_localisation(predicted: list[str], gold: list[str]) -> dict[str, float]:
    if not gold:
        return {
            "file_hit_at_5": math.nan,
            "file_recall": math.nan,
            "file_precision": math.nan,
            "module_hit": math.nan,
        }
    top = predicted[:5]
    matched_gold = {target for target in gold if any(_location_match(item, target) for item in top)}
    matched_pred = {item for item in top if any(_location_match(item, target) for target in gold)}
    gold_modules = {_module(path).casefold() for path in gold}
    pred_modules = {_module(path).casefold() for path in top}
    return {
        "file_hit_at_5": float(bool(matched_gold)),
        "file_recall": len(matched_gold) / len(set(gold)),
        "file_precision": len(matched_pred) / len(set(top)) if top else 0.0,
        "module_hit": float(bool(gold_modules.intersection(pred_modules))),
    }


def _numeric_metric(metrics: Mapping[str, float | list[str]], name: str) -> float:
    value = metrics[name]
    if not isinstance(value, (int, float)):
        raise TypeError(f"localisation metric {name} was not numeric")
    return float(value)


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1
                if left_token == right_token
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def rouge_l(prediction: str, references: Iterable[str]) -> float:
    """Return maximum token-level ROUGE-L F1 over the supplied references."""

    predicted = _normalise(prediction).split()
    scores: list[float] = []
    for reference in references:
        target = _normalise(reference).split()
        if not predicted or not target:
            scores.append(0.0)
            continue
        common = _lcs_length(predicted, target)
        precision = common / len(predicted)
        recall = common / len(target)
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return max(scores, default=math.nan)


def _score_action(task: Task, prediction: ActionPrediction) -> dict[str, float]:
    people = task.gold.get("people", {})
    category = task.gold.get("category", {})
    place = task.gold.get("place", {})
    text = task.gold.get("text", {})
    field_scores = {
        "assignee_hit_at_3": _hit_at(prediction.assignee_candidates, people.get("assignees", [])),
        "reviewer_hit_at_3": _hit_at(prediction.reviewer_candidates, people.get("reviewers", [])),
        "component_em": _exact_one(prediction.component, category.get("components", [])),
        "duplicate_em": _exact_one(prediction.duplicate_of, category.get("duplicate_of", [])),
        "priority_em": _exact_one(prediction.priority_change, category.get("priority_changes", [])),
    }
    available = [value for value in field_scores.values() if not math.isnan(value)]
    field_em = statistics.fmean(available) if available else 0.0
    required = {_normalise(value) for value in category.get("required_ids", []) if value}
    cited = {_normalise(value) for value in prediction.cited_ids}
    id_cov = len(required.intersection(cited)) / len(required) if required else 1.0
    shared_action = grade_action(
        task.task_id,
        prediction.model_dump(),
        task.gold,
        gold_coverage=task.gold_coverage,
    )
    field_em = float(shared_action.details.get("field_em") or 0.0)
    id_cov = float(shared_action.details.get("id_cov") or 0.0)
    field_scores.update(
        {
            "assignee_hit_at_3": float(
                shared_action.details.get("assignee_hit@3", field_scores["assignee_hit_at_3"])
            ),
            "reviewer_hit_at_3": float(
                shared_action.details.get("reviewer_hit@3", field_scores["reviewer_hit_at_3"])
            ),
            "component_em": float(
                shared_action.details.get("component_exact", field_scores["component_em"])
            ),
            "duplicate_em": float(
                shared_action.details.get("duplicate_of_exact", field_scores["duplicate_em"])
            ),
            "priority_em": float(
                shared_action.details.get("priority_change_exact", field_scores["priority_em"])
            ),
        }
    )
    quality = shared_action.value
    shared_local = localisation_scores(
        prediction.suspected_locations,
        _list(place.get("files")),
        k=5,
        agentless_superset=True,
    )
    local = {
        "file_hit_at_5": _numeric_metric(shared_local, "file_hit@5"),
        "file_recall": _numeric_metric(shared_local, "file_recall"),
        "file_precision": _numeric_metric(shared_local, "file_precision"),
        "module_hit": _numeric_metric(shared_local, "module_hit"),
    }
    replies = _list(text.get("replies"))
    rouge_score = max(
        (grade_rouge_l(task.task_id, prediction.draft_reply, reply).value for reply in replies),
        default=math.nan,
    )
    return {
        **field_scores,
        "field_em": field_em,
        "id_cov": id_cov,
        "Q": quality,
        **local,
        "rouge_l": rouge_score,
    }


def _gold_times(task: Task) -> list[str]:
    times: list[str] = []
    for group in GOLD_GROUPS:
        payload = task.gold.get(group)
        if isinstance(payload, Mapping):
            times.extend(_list(payload.get("decision_times")))
    return times


def _gold_event_ids(task: Task) -> list[str]:
    event_ids: list[str] = []
    for group in GOLD_GROUPS:
        payload = task.gold.get(group)
        if isinstance(payload, Mapping):
            event_ids.extend(_list(payload.get("event_ids")))
    return event_ids


def _task_archetype(task: Task) -> tuple[str, str]:
    trigger = task.gold.get("_trigger_event", {})
    rendered = json.dumps(trigger, sort_keys=True, default=str).casefold()
    if "[vote]" in rendered:
        archetype = "vote"
    elif re.search(r"\bcve(?:-\d{4}-\d+)?\b", rendered):
        archetype = "cve"
    elif re.search(r"\b(?:blocker|critical)\b", rendered):
        archetype = "declared_priority"
    else:
        archetype = task.kind
    source = str(trigger.get("source", "unknown")) if isinstance(trigger, Mapping) else "unknown"
    return archetype, source


def _task_gold_is_bot_free(task: Task) -> bool:
    people = task.gold.get("people", {})
    if not isinstance(people, Mapping):
        return True
    bots = {account.casefold() for account in DEFAULT_BOT_ACCOUNTS}
    values = _list(people.get("assignees")) + _list(people.get("reviewers"))
    return all(
        value.casefold() not in bots
        and not value.casefold().endswith("[bot]")
        and not value.casefold().endswith("-bot")
        for value in values
    )


def leakage_gate(
    task: Task, snapshot: SnapshotRef, envelope: Envelope
) -> tuple[bool, dict[str, bool]]:
    """Apply the E4-computable parts of §4.2, including gold-after-T."""

    snapshot_ok = snapshot.T_last_event <= task.T
    gold_after_t = True
    for value in _gold_times(task):
        try:
            gold_after_t = gold_after_t and datetime_from_iso(value) > task.T
        except ValueError:
            gold_after_t = False
    no_gold_events = not any(
        event_id and event_id in envelope.text for event_id in _gold_event_ids(task)
    )
    checks = {
        "snapshot_at_or_before_T": snapshot_ok,
        "gold_after_T": gold_after_t,
        "gold_events_absent": no_gold_events,
    }
    return all(checks.values()), checks


def datetime_from_iso(value: str) -> datetime:
    """Parse an ISO timestamp without weakening timezone-aware comparisons."""

    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _batch_accuracy(task: Task, envelope: Envelope) -> float:
    probes = task.gold.get("probes", [])
    values = []
    for raw in probes:
        probe = Probe.model_validate(raw)
        gold_values = _list(probe.gold)
        values.append(
            float(any(_normalise(value) in _normalise(envelope.text) for value in gold_values))
        )
    return statistics.fmean(values) if values else math.nan


def _mean(values: Iterable[Any]) -> float:
    numeric = [float(value) for value in values if value is not None and not pd.isna(value)]
    return statistics.fmean(numeric) if numeric else math.nan


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )


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
) -> ExperimentResult:
    """Run paired task × envelope × repetition E4 evaluation and write outputs."""

    if runs <= 0:
        raise ValueError("runs must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    task_list = list(tasks)
    provider = provider or make_llm(cfg)
    event_list = list(events)
    selected_variants = tuple(variant.upper() for variant in variants)
    run_rows: list[dict[str, Any]] = []
    size_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    accepted_tasks: set[str] = set()
    for task in task_list:
        try:
            snapshot = store.snapshot(task.T)
        except LookupError:
            gate_rows.append(
                {"task_id": task.task_id, "variant": "", "PASS": False, "reason": "no_snapshot"}
            )
            continue
        for variant in selected_variants:
            envelope = build(
                task,
                snapshot,
                variant,
                {"store_handle": store, "events": event_list},
            )
            gate_pass, checks = leakage_gate(task, snapshot, envelope)
            gate_rows.append(
                {"task_id": task.task_id, "variant": variant, "PASS": gate_pass, **checks}
            )
            size_rows.append(
                {
                    "task_id": task.task_id,
                    "entity": task.entity,
                    "variant": variant,
                    "tokens": envelope.token_count,
                    **{
                        f"tokens_{name}": count
                        for name, count in envelope.tokens_by_section.items()
                    },
                }
            )
            if not gate_pass:
                continue
            accepted_tasks.add(task.task_id)
            for repetition in range(1, runs + 1):
                started = time.perf_counter()
                prediction, result = _complete(provider, envelope)
                latency = time.perf_counter() - started
                scores = _score_action(task, prediction) if task.kind == "fast" else {}
                run_rows.append(
                    {
                        "task_id": task.task_id,
                        "entity": task.entity,
                        "kind": task.kind,
                        "variant": variant,
                        "run": repetition,
                        "snapshot_sha": snapshot.sha,
                        "tokens": result.usage.get("input_tokens", envelope.token_count),
                        "output_tokens": result.usage.get("output_tokens", 0),
                        "latency_s": latency,
                        "tool_calls": len(envelope.tools),
                        "prediction": prediction.model_dump(mode="json"),
                        "batch_e2_acc": _batch_accuracy(task, envelope)
                        if task.kind == "batch"
                        else math.nan,
                        "evidence_valid": float(
                            all(citation in envelope.text for citation in prediction.cited_ids)
                        ),
                        **scores,
                    }
                )

    metric_rows: list[dict[str, Any]] = []
    metric_names = (
        "Q",
        "field_em",
        "id_cov",
        "assignee_hit_at_3",
        "reviewer_hit_at_3",
        "component_em",
        "duplicate_em",
        "priority_em",
        "file_hit_at_5",
        "file_recall",
        "file_precision",
        "module_hit",
        "rouge_l",
        "batch_e2_acc",
        "evidence_valid",
        "latency_s",
        "tool_calls",
    )
    for variant in selected_variants:
        subset = [item for item in run_rows if item["variant"] == variant]
        row: dict[str, Any] = {
            "variant": variant,
            "tasks": len({str(item["task_id"]) for item in subset}),
        }
        for metric in metric_names:
            row[metric] = _mean(item.get(metric) for item in subset)
        if subset:
            correctness: dict[str, list[bool]] = defaultdict(list)
            for item in subset:
                value = item.get("Q")
                if isinstance(value, (int, float)) and not math.isnan(float(value)):
                    correctness[str(item["task_id"])].append(float(value) >= 1.0)
            row["pass3"] = _mean(
                float(len(values) == runs and all(values)) for values in correctness.values()
            )
        else:
            row["pass3"] = math.nan
        row["median_tokens"] = (
            statistics.median([item["tokens"] for item in size_rows if item["variant"] == variant])
            if any(item["variant"] == variant for item in size_rows)
            else math.nan
        )
        for group in GOLD_GROUPS:
            row[f"coverage_{group}"] = _mean(
                float(task.gold_coverage.get(group, False)) for task in task_list
            )
        metric_rows.append(row)
    metrics_frame = pd.DataFrame(metric_rows)

    task_variant_q: dict[tuple[str, str], list[float]] = defaultdict(list)
    for item in run_rows:
        value = item.get("Q")
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            task_variant_q[(str(item["task_id"]), str(item["variant"]))].append(float(value))
    task_means = {key: statistics.fmean(values) for key, values in task_variant_q.items()}
    contrast_rows = []
    for left, right in CONTRASTS:
        task_ids = {
            task_id
            for task_id, variant in task_means
            if variant == left and (task_id, right) in task_means
        }
        deltas = [
            task_means[(task_id, left)] - task_means[(task_id, right)] for task_id in task_ids
        ]
        contrast_rows.append(
            {
                "contrast": f"{left}-{right}",
                "left": left,
                "right": right,
                "n": len(deltas),
                "mean_delta_Q": _mean(deltas),
            }
        )
    contrasts_frame = pd.DataFrame(contrast_rows)
    sizes_frame = pd.DataFrame(size_rows)
    gate_frame = pd.DataFrame(gate_rows)
    balance_counts: dict[tuple[str, str], int] = defaultdict(int)
    for task in task_list:
        balance_counts[_task_archetype(task)] += 1
    balance_rows = [
        {"archetype": archetype, "source": source, "count": count}
        for (archetype, source), count in sorted(balance_counts.items())
    ]
    balance_frame = pd.DataFrame(balance_rows, columns=["archetype", "source", "count"])
    archetype_totals: dict[str, int] = defaultdict(int)
    for (archetype, _), count in balance_counts.items():
        archetype_totals[archetype] += count
    max_archetype_share = (
        max(archetype_totals.values()) / sum(archetype_totals.values())
        if archetype_totals
        else math.nan
    )
    medians = {str(row["variant"]): float(row["median_tokens"]) for row in metric_rows}
    v3_median = medians.get("V3", math.nan)
    v6_median = medians.get("V6", math.nan)
    gate_passes = sum(bool(row.get("PASS")) for row in gate_rows)
    checks = {
        "checks.leakage_gate_100_pct": float(gate_passes == len(gate_rows)),
        "checks.leakage_gate_passed": float(gate_passes),
        "checks.leakage_gate_failed": float(len(gate_rows) - gate_passes),
        "checks.v3_median_le_12k": float(not math.isnan(v3_median) and v3_median <= 12_000),
        "checks.v6_median_ge_3x_v3": float(
            not math.isnan(v3_median) and not math.isnan(v6_median) and v6_median >= 3 * v3_median
        ),
        "checks.tasks_accepted_gt_0": float(bool(accepted_tasks)),
        "checks.gold_actions_after_T": float(
            all(
                datetime_from_iso(value) > task.T
                for task in task_list
                for value in _gold_times(task)
            )
        ),
        "checks.bot_accounts_excluded": float(
            all(_task_gold_is_bot_free(task) for task in task_list)
        ),
        "checks.pr_join_precision": float(
            all(
                bool(task.gold.get("place", {}).get("event_ids"))
                for task in task_list
                if task.gold_coverage.get("place", False)
            )
        ),
        "checks.pr_join_recall": float(
            all(
                bool(task.gold.get("place", {}).get("files"))
                for task in task_list
                if task.gold_coverage.get("place", False)
            )
        ),
        "checks.judge_kappa_ge_0_6": 0.0,
        "checks.position_swapped": 0.0,
        "checks.max_archetype_le_40pct": float(
            not math.isnan(max_archetype_share) and max_archetype_share <= 0.40
        ),
        "max_archetype_share": max_archetype_share,
        "gate_pass_count": float(gate_passes),
        "gate_exclusion_count": float(len(gate_rows) - gate_passes),
        "tasks_accepted": float(len(accepted_tasks)),
    }
    artifacts = [
        out_dir / "runs.jsonl",
        out_dir / "metrics.csv",
        out_dir / "contrasts.csv",
        out_dir / "sizes.csv",
        out_dir / "gate.csv",
        out_dir / "judge_kappa.csv",
        out_dir / "balance.csv",
    ]
    _write_jsonl(artifacts[0], run_rows)
    metrics_frame.to_csv(artifacts[1], index=False)
    contrasts_frame.to_csv(artifacts[2], index=False)
    sizes_frame.to_csv(artifacts[3], index=False)
    gate_frame.to_csv(artifacts[4], index=False)
    pd.DataFrame(
        [
            {
                "corpus": task_list[0].corpus if task_list else "",
                "judge_kappa": math.nan,
                "used": False,
                "reason": "human calibration unavailable; ROUGE-L only",
            }
        ]
    ).to_csv(artifacts[5], index=False)
    balance_frame.to_csv(artifacts[6], index=False)
    primary = {
        row["contrast"]: row["mean_delta_Q"]
        for row in contrast_rows
        if row["contrast"] in {"V3-V1", "V3-V6"}
    }
    return ExperimentResult(
        name="e4",
        metrics=checks,
        tables={
            "metrics": metrics_frame,
            "contrasts": contrasts_frame,
            "sizes": sizes_frame,
            "gate": gate_frame,
            "balance": balance_frame,
        },
        artifacts=artifacts,
        primary=primary,
    )


def _build_experiment_store(
    events: list[EvalEvent],
    tasks: list[Task],
    cfg: EngineConfig,
    out_dir: Path,
    seed: int,
) -> StoreHandle:
    """Build the fixed S3 store through the shared replay and layout modules."""

    store = StoreHandle("S3", f"e4-{seed}", out_dir / "store")
    configure_store(
        store,
        harness=make_harness_name(cfg),
        model=cfg.builder.model,
    )
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


class E4Experiment:
    name = "e4"

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider

    def run(
        self, cfg: EngineConfig, corpus: CorpusHandle, out_dir: Path, seed: int
    ) -> ExperimentResult:
        events = list(corpus.events())
        tasks = _load_tasks(corpus.tasks_path)
        if not tasks:
            tasks = build_tasks(
                events,
                corpus=corpus.name,
                probes=_load_probes(corpus.probes_path),
                window=cfg.window,
            )
        meta_store = corpus.meta.get("store_handle") or corpus.meta.get("store")
        store = (
            meta_store
            if isinstance(meta_store, StoreHandle)
            else _build_experiment_store(events, tasks, cfg, out_dir, seed)
        )
        return run_e4(
            tasks,
            store,
            cfg,
            out_dir,
            provider=self.provider,
            variants=("V1", "V3", "V6", "V7", "V8")
            if corpus.meta.get("smoke")
            else VARIANTS,
            runs=1 if corpus.meta.get("smoke") else 3,
            events=events,
        )

    def chart(self, result: ExperimentResult, out_dir: Path) -> list[Path]:
        table = result.tables["metrics"]
        if table.empty:
            return []
        return [
            e4_envelopes(
                table.rename(columns={"variant": "envelope", "median_tokens": "tokens"}),
                out_dir,
            )
        ]


register_experiment(E4Experiment())

__all__ = ["ActionPrediction", "E4Experiment", "leakage_gate", "rouge_l", "run_e4"]
