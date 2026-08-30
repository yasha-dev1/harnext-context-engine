"""State-fidelity run loop from docs/evaluation-spec.md §7 E2."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from harnext_eval.agents.reader import SYSTEM_PROMPT, answer
from harnext_eval.config import EngineConfig
from harnext_eval.corpus import CorpusHandle
from harnext_eval.e2.arms import build_arm
from harnext_eval.grade.exact import grade_exact, normalize_exact
from harnext_eval.grade.links import grade_links
from harnext_eval.grade.localisation import grade_localisation, module_for, normalize_path
from harnext_eval.providers.embeddings import EmbeddingsProvider, FakeEmbeddings
from harnext_eval.providers.factory import make_embeddings, make_llm
from harnext_eval.providers.llm import FakeLLM, LLMProvider
from harnext_eval.providers.tokenizer import tokenizer_for
from harnext_eval.registry import ExperimentResult, register_experiment
from harnext_eval.replay.gate import leakage_gate
from harnext_eval.report.charts import e2_family_bars
from harnext_eval.stats.stats import (
    holm_bonferroni,
    mcnemar_test,
    paired_difference_bca,
)
from harnext_eval.stores.base import StoreHandle
from harnext_eval.types import Answer, EvalEvent, GradeResult, Probe

DEFAULT_ARMS = (
    "A0",
    "A1-N20",
    "A1-N100",
    "A2",
    "A3",
    "A4",
    "retrieve_everything",
    "retrieve_nothing",
)
MACRO_FAMILIES = ("extraction", "temporal", "update", "multisource", "abstention")
BOOTSTRAP_RESAMPLES = 10_000
_PATH_RE = re.compile(
    r"(?<![\w/.-])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+|"
    r"(?<![\w.-])(?:Dockerfile|BUILD|WORKSPACE|Makefile)(?![\w.-])"
)


@dataclass(frozen=True)
class ProbeOutcome:
    probe: Probe
    answer: Answer
    grade: GradeResult
    original_tokens: int
    supersession_error: bool
    provider_input_tokens: int = 0
    reported_provider_input_tokens: int = 0
    tokenizer_id: str = ""
    tokenizer_revision: str = ""
    reader_cost_usd: float = 0.0


def load_probes(path: Path | None) -> list[Probe]:
    """Read frozen probe JSONL through T0's shared model."""

    if path is None or not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return [Probe.model_validate_json(line) for line in source if line.strip()]


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

    if probe.gold_type == "links" or (probe.family == "multisource" and probe.gold_type != "files"):
        return grade_links(probe.probe_id, prediction, _as_values(probe.gold))
    if probe.gold_type == "files" or probe.family == "code_location":
        predicted, predicted_modules = _code_prediction(prediction)
        gold_files = _as_values(probe.gold)
        result = grade_localisation(probe.probe_id, predicted, gold_files)
        details = dict(result.details)
        if predicted_modules:
            gold_modules = {module_for(path) for path in gold_files}
            details["predicted_modules"] = sorted(predicted_modules)
            details["module_hit"] = float(gold_modules <= predicted_modules)
        precision = float(details.get("file_precision", 0.0))
        recall = float(details.get("file_recall", 0.0))
        file_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        details["file_f1"] = file_f1
        return result.model_copy(
            update={"metric": "file_f1", "value": file_f1, "details": details}
        )
    return grade_exact(probe.probe_id, prediction, probe.gold)


def _code_prediction(prediction: str) -> tuple[list[str], set[str]]:
    """Parse typed or line-oriented file/module answers without extension assumptions."""

    files: list[str] = []
    modules: set[str] = set()
    try:
        value = json.loads(prediction)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        raw_files = value.get("files", [])
        raw_modules = value.get("modules", [])
        if isinstance(raw_files, list):
            files.extend(str(item) for item in raw_files)
        if isinstance(raw_modules, list):
            modules.update(normalize_path(str(item)) for item in raw_modules)
    if not files:
        files.extend(_PATH_RE.findall(prediction))
    return list(dict.fromkeys(normalize_path(path) for path in files)), modules


def _contains_complete_value(answer_text: str, value: str) -> bool:
    """Match a retired value as a complete canonical answer value, not a substring."""

    answer = normalize_exact(answer_text)
    retired = normalize_exact(value)
    if not retired:
        return False
    return re.search(rf"(?<![\w]){re.escape(retired)}(?![\w])", answer) is not None


def _macro_family(family: str) -> str:
    return "multisource" if family == "code_location" else family


def macro_accuracy(
    outcomes: Sequence[ProbeOutcome], *, require_all_families: bool = True
) -> float:
    """Literal equal-weight mean of the five preregistered family scores."""

    by_family: dict[str, list[float]] = defaultdict(list)
    for outcome in outcomes:
        by_family[_macro_family(outcome.probe.family)].append(outcome.grade.value)
    if require_all_families and any(not by_family[family] for family in MACRO_FAMILIES):
        return math.nan
    values = [np.mean(by_family[family]) for family in MACRO_FAMILIES if by_family[family]]
    return float(np.mean(values)) if values else math.nan


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
        "N": int(arm.removeprefix("A1-N")) if arm.startswith("A1-N") else math.nan,
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
        "provider_input_tokens": (
            float(np.mean([row.provider_input_tokens for row in rows])) if rows else 0.0
        ),
        "tool_calls": float(np.mean([row.answer.tool_calls for row in rows])) if rows else 0.0,
        "latency_s": float(np.mean([row.answer.latency_s for row in rows])) if rows else 0.0,
        "reader_cost_usd": (
            float(np.mean([row.reader_cost_usd for row in rows])) if rows else 0.0
        ),
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
        "n_resamples",
        "valid",
        "invalid_reason",
        "binary_definition",
    ]
    if not pairs:
        return pd.DataFrame(columns=columns)
    left_scores = np.asarray([pair[0].grade.value for pair in pairs], dtype=float)
    right_scores = np.asarray([pair[1].grade.value for pair in pairs], dtype=float)
    entities = np.asarray([pair[0].probe.entity for pair in pairs], dtype=object)
    families = np.asarray(
        [_macro_family(pair[0].probe.family) for pair in pairs], dtype=object
    )
    unique_entities = pd.unique(entities)
    present = set(str(value) for value in families)
    missing = sorted(set(MACRO_FAMILIES) - present)
    invalid_reasons: list[str] = []
    if missing:
        invalid_reasons.append("missing_families:" + "|".join(missing))
    if len(unique_entities) < 2:
        invalid_reasons.append("fewer_than_two_entity_clusters")
    delta = _macro_delta(left_scores, right_scores, families)
    ci_low = ci_high = math.nan
    if not invalid_reasons:
        try:
            ci_low, ci_high = _macro_bca_interval(
                left_scores,
                right_scores,
                families,
                entities,
                n_resamples=BOOTSTRAP_RESAMPLES,
                seed=seed,
            )
        except ValueError as exc:
            invalid_reasons.append(f"bootstrap_incomplete:{exc}")
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
                "n_resamples": BOOTSTRAP_RESAMPLES,
                "valid": not invalid_reasons,
                "invalid_reason": ";".join(invalid_reasons),
                "binary_definition": "score_eq_1_exact_correctness",
            }
        ],
        columns=columns,
    )


def _macro_delta(
    left: np.ndarray,
    right: np.ndarray,
    families: np.ndarray,
) -> float:
    """Recompute the equal-family-weight estimand for one paired sample."""

    values: list[float] = []
    for family in MACRO_FAMILIES:
        selected = families == family
        if not np.any(selected):
            return math.nan
        values.append(float(np.mean(left[selected]) - np.mean(right[selected])))
    return float(np.mean(values))


def _macro_bca_interval(
    left: np.ndarray,
    right: np.ndarray,
    families: np.ndarray,
    entities: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Entity-clustered BCa interval, recomputing family means in every draw."""

    unique_entities = pd.unique(entities)
    cluster_rows = [np.flatnonzero(entities == entity) for entity in unique_entities]
    effect = _macro_delta(left, right, families)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_resamples, dtype=float)
    for draw_index in range(n_resamples):
        selected = rng.integers(0, len(cluster_rows), size=len(cluster_rows))
        rows = np.concatenate([cluster_rows[index] for index in selected])
        bootstrap[draw_index] = _macro_delta(left[rows], right[rows], families[rows])
    bootstrap = bootstrap[np.isfinite(bootstrap)]
    if len(bootstrap) < max(100, n_resamples // 2):
        raise ValueError("entity bootstrap cannot retain all five E2 families")

    proportion_less = (
        np.count_nonzero(bootstrap < effect)
        + 0.5 * np.count_nonzero(bootstrap == effect)
    ) / len(bootstrap)
    epsilon = 0.5 / len(bootstrap)
    bias_correction = float(norm.ppf(np.clip(proportion_less, epsilon, 1 - epsilon)))
    jackknife = []
    for rows in cluster_rows:
        keep = np.ones(len(left), dtype=bool)
        keep[rows] = False
        value = _macro_delta(left[keep], right[keep], families[keep])
        if np.isfinite(value):
            jackknife.append(value)
    if len(jackknife) < 2:
        raise ValueError("entity jackknife cannot retain all five E2 families")
    jackknife_values = np.asarray(jackknife)
    centred = float(np.mean(jackknife_values)) - jackknife_values
    denominator = 6.0 * float(np.sum(centred**2)) ** 1.5
    acceleration = 0.0 if denominator == 0 else float(np.sum(centred**3)) / denominator
    adjusted: list[float] = []
    for quantile in norm.ppf([0.025, 0.975]):
        shifted = bias_correction + quantile
        divisor = 1.0 - acceleration * shifted
        adjusted.append(
            0.0
            if math.isclose(divisor, 0.0, abs_tol=1e-15) and shifted < 0
            else 1.0
            if math.isclose(divisor, 0.0, abs_tol=1e-15)
            else float(norm.cdf(bias_correction + shifted / divisor))
        )
    low_q, high_q = np.clip(np.sort(adjusted), 0.0, 1.0)
    ci_low, ci_high = np.quantile(bootstrap, [low_q, high_q])
    return float(ci_low), float(ci_high)


def _secondary_contrasts(
    outcomes: Sequence[ProbeOutcome], left: str, right: str, seed: int
) -> pd.DataFrame:
    """Per-family paired CIs and Holm-adjusted exact-correctness tests."""

    indexed = {(row.probe.probe_id, row.answer.arm): row for row in outcomes}
    pairs = [
        (indexed[(probe_id, left)], indexed[(probe_id, right)])
        for probe_id in sorted({key[0] for key in indexed})
        if (probe_id, left) in indexed and (probe_id, right) in indexed
    ]
    rows: list[dict[str, Any]] = []
    p_values: dict[str, float] = {}
    for family in MACRO_FAMILIES:
        selected = [pair for pair in pairs if _macro_family(pair[0].probe.family) == family]
        entities = [pair[0].probe.entity for pair in selected]
        left_scores = [pair[0].grade.value for pair in selected]
        right_scores = [pair[1].grade.value for pair in selected]
        valid = len(set(entities)) >= 2 and bool(selected)
        ci_low = ci_high = math.nan
        delta = (
            float(np.mean(np.asarray(left_scores) - np.asarray(right_scores)))
            if selected
            else math.nan
        )
        if valid:
            interval = paired_difference_bca(
                left_scores,
                right_scores,
                entities,
                n_resamples=BOOTSTRAP_RESAMPLES,
                random_state=seed,
            )
            ci_low, ci_high = interval.ci_low, interval.ci_high
        mcnemar = mcnemar_test(
            [value == 1.0 for value in left_scores],
            [value == 1.0 for value in right_scores],
        )
        p_values[family] = mcnemar.p_value
        rows.append(
            {
                "family": family,
                "contrast": f"{left}-{right}",
                "delta": delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "mcnemar_p": mcnemar.p_value,
                "n": len(selected),
                "entities": len(set(entities)),
                "n_resamples": BOOTSTRAP_RESAMPLES,
                "valid": valid,
            }
        )
    adjusted = holm_bonferroni(p_values).set_index("hypothesis")
    for row in rows:
        row["holm_adjusted_p"] = float(adjusted.loc[row["family"], "adjusted_p"])
        row["holm_reject"] = bool(adjusted.loc[row["family"], "reject"])
    return pd.DataFrame(rows)


def _answer_record(outcome: ProbeOutcome) -> dict[str, Any]:
    return {
        **outcome.answer.model_dump(mode="json"),
        "N": (
            int(outcome.answer.arm.removeprefix("A1-N"))
            if outcome.answer.arm.startswith("A1-N")
            else None
        ),
        "family": outcome.probe.family,
        "entity": outcome.probe.entity,
        "grade_metric": outcome.grade.metric,
        "grade": outcome.grade.value,
        "grade_details": outcome.grade.details,
        "supersession_error": outcome.supersession_error,
        "provider_input_tokens": outcome.provider_input_tokens,
        "reported_provider_input_tokens": outcome.reported_provider_input_tokens,
        "tokenizer_id": outcome.tokenizer_id,
        "tokenizer_revision": outcome.tokenizer_revision,
        "reader_cost_usd": outcome.reader_cost_usd,
    }


def _provider_identity(provider: Any) -> dict[str, str]:
    """Record a stable provider/model/revision triple without changing shared types."""

    return {
        "provider": str(
            getattr(provider, "provider_id", None)
            or getattr(provider, "provider", None)
            or f"{type(provider).__module__}.{type(provider).__qualname__}"
        ),
        "model": str(
            getattr(provider, "model_id", None)
            or getattr(provider, "model", None)
            or type(provider).__qualname__
        ),
        "revision": str(
            getattr(provider, "model_revision", None)
            or getattr(provider, "revision", None)
            or "unspecified"
        ),
    }


def _reader_cost(cfg: EngineConfig, accounting: Mapping[str, int | str | bool]) -> float:
    prices = cfg.prices
    if prices is None:
        return 0.0
    return (
        int(accounting["reported_provider_input_tokens"]) * prices.input_per_million
        + int(accounting["reported_provider_output_tokens"]) * prices.output_per_million
    ) / 1_000_000


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
    validation_audit: Mapping[str, float] | None = None,
) -> tuple[ExperimentResult, list[ProbeOutcome]]:
    """Run the complete probe × arm matrix and persist E2 outputs."""

    out_dir.mkdir(parents=True, exist_ok=True)
    if "A4" in arms and store is None:
        raise ValueError("E2 A4 requires an explicit S3 StoreHandle, received none")
    effective_arms = tuple(
        store.layout if arm == "A4" and store is not None and store.layout != "S3" else arm
        for arm in arms
    )
    selected_llm = llm or make_llm(cfg)
    selected_embeddings = embeddings or make_embeddings(cfg)
    token_counter = tokenizer_for(selected_llm)
    non_evidentiary_smoke = (
        isinstance(selected_llm, FakeLLM)
        or isinstance(selected_embeddings, FakeEmbeddings)
        or token_counter.smoke_only
    )
    outcomes: list[ProbeOutcome] = []
    gate_results: list[bool] = []
    budget = cfg.reader.budget_tokens
    budget_checks: list[bool] = []
    budget_rows: list[dict[str, Any]] = []
    gate_path = out_dir / "gate.csv"
    gate_path.unlink(missing_ok=True)
    cache_path = out_dir / "reader_cache.jsonl"
    response_cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                response_cache[str(row["key"])] = row
    for probe in probes:
        materials = [
            build_arm(
                arm,
                probe,
                events,
                cfg,
                store=store,
                embeddings=selected_embeddings,
                tokenizer=token_counter,
            )
            for arm in effective_arms
        ]
        passed = leakage_gate(
            probe,
            store=store,
            T=probe.T,
            all_events=events,
            material=[material.text for material in materials],
            out_csv=gate_path,
        )
        gate_results.append(passed)
        if not passed:
            continue
        for material in materials:
            accounting: dict[str, int | str | bool] = {}
            cache_key = hashlib.sha256(
                json.dumps(
                    {
                        "probe": probe.model_dump(mode="json"),
                        "arm": material.arm,
                        "material_sha256": hashlib.sha256(material.text.encode()).hexdigest(),
                        "provider": _provider_identity(selected_llm),
                        "tokenizer": [
                            token_counter.tokenizer_id,
                            token_counter.tokenizer_revision,
                        ],
                        "prompt": SYSTEM_PROMPT,
                        "budget": budget,
                    },
                    sort_keys=True,
                    default=str,
                ).encode()
            ).hexdigest()
            cached = response_cache.get(cache_key)
            if cached is not None:
                response = Answer.model_validate(cached["answer"])
                accounting.update(cached["accounting"])
                accounting["cache_hit"] = True
            else:
                response = answer(
                    probe,
                    material,
                    cfg,
                    provider=selected_llm,
                    tokenizer=token_counter,
                    accounting=accounting,
                )
                accounting["cache_hit"] = False
                response_cache[cache_key] = {
                    "key": cache_key,
                    "answer": response.model_dump(mode="json"),
                    "accounting": dict(accounting),
                }
            grade = grade_answer(probe, response.text)
            superseded = any(
                _contains_complete_value(response.text, value)
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
                    provider_input_tokens=int(accounting["provider_input_tokens"]),
                    reported_provider_input_tokens=int(
                        accounting["reported_provider_input_tokens"]
                    ),
                    tokenizer_id=str(accounting["tokenizer_id"]),
                    tokenizer_revision=str(accounting["tokenizer_revision"]),
                    reader_cost_usd=_reader_cost(cfg, accounting),
                )
            )
            eligible = material.enforce_budget and (material.original_tokens or 0) >= budget
            within = 0.9 * budget <= response.tokens_read <= 1.1 * budget if eligible else None
            if within is not None:
                budget_checks.append(within)
            budget_rows.append(
                {
                    "probe_id": probe.probe_id,
                    "arm": material.arm,
                    "budget": budget,
                    "eligible_to_fill": eligible,
                    "selected_material_tokens": response.tokens_read,
                    "provider_input_tokens": accounting["provider_input_tokens"],
                    "within_10_pct": within,
                    "tokenizer_id": accounting["tokenizer_id"],
                    "tokenizer_revision": accounting["tokenizer_revision"],
                }
            )

    cache_path.write_text(
        "".join(
            json.dumps(response_cache[key], sort_keys=True) + "\n"
            for key in sorted(response_cache)
        ),
        encoding="utf-8",
    )
    answer_path = out_dir / "answers.jsonl"
    answer_path.write_text(
        "".join(json.dumps(_answer_record(row), sort_keys=True) + "\n" for row in outcomes),
        encoding="utf-8",
    )
    metric_rows = _metric_rows(outcomes, effective_arms)
    metrics_table = pd.DataFrame(metric_rows)
    metrics_path = out_dir / "metrics.csv"
    metrics_table.to_csv(metrics_path, index=False)
    contrasts = _paired_contrast(outcomes, "A4", "A3", seed)
    contrasts_path = out_dir / "contrasts.csv"
    contrasts.to_csv(contrasts_path, index=False)
    secondaries = _secondary_contrasts(outcomes, "A4", "A3", seed)
    secondaries_path = out_dir / "secondary_contrasts.csv"
    secondaries.to_csv(secondaries_path, index=False)
    budget_table = pd.DataFrame(budget_rows)
    budget_path = out_dir / "budget_checks.csv"
    budget_table.to_csv(budget_path, index=False)
    provider_path = out_dir / "providers.json"
    provider_path.write_text(
        json.dumps(
            {
                "reader": _provider_identity(selected_llm),
                "embeddings": _provider_identity(selected_embeddings),
                "tokenizer": {
                    "id": token_counter.tokenizer_id,
                    "revision": token_counter.tokenizer_revision,
                },
                "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                "non_evidentiary_smoke": non_evidentiary_smoke,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiment": "e2",
                "seed": seed,
                "budget_tokens": budget,
                "arms": list(effective_arms),
                "replay_sha256": hashlib.sha256(
                    "".join(event.model_dump_json() + "\n" for event in events).encode()
                ).hexdigest(),
                "probe_sha256": hashlib.sha256(
                    "".join(probe.model_dump_json() + "\n" for probe in probes).encode()
                ).hexdigest(),
                "provider_metadata": json.loads(provider_path.read_text(encoding="utf-8")),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    arm_outcomes = {
        arm: [row for row in outcomes if row.answer.arm == arm] for arm in effective_arms
    }
    everything = macro_accuracy(arm_outcomes.get("retrieve_everything", []))
    prior_target = [
        row
        for row in arm_outcomes.get("A0", [])
        if _macro_family(row.probe.family) in {"temporal", "update", "multisource"}
    ]
    prior = float(np.mean([row.grade.value for row in prior_target])) if prior_target else 0.0
    a0_correct = [
        {
            "probe_id": row.probe.probe_id,
            "family": row.probe.family,
            "entity": row.probe.entity,
            "grade": row.grade.value,
        }
        for row in arm_outcomes.get("A0", [])
        if row.grade.value == 1.0
    ]
    a0_correct_path = out_dir / "a0_correct.jsonl"
    a0_correct_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in a0_correct),
        encoding="utf-8",
    )
    passed_count = sum(gate_results)
    budget_applicable = bool(budget_checks)
    exact_rerun_identical = all(
        grade_answer(row.probe, row.answer.text) == row.grade for row in outcomes
    )
    audit = dict(validation_audit or {})
    dual_gold_ready = audit.get("dual_gold_agreement", 0.0) >= 0.98
    pilot_ready = audit.get("pilot_kappa", 0.0) >= 0.8
    claim_ready = audit.get("claim_disagreement", 1.0) <= 0.02
    checks: dict[str, float] = {
        "checks.floor_retrieve_everything_ge_0_9": float(
            np.isfinite(everything) and everything >= 0.9
        ),
        "checks.prior_leq_0_3": float(prior <= 0.3),
        "checks.leakage_gate_100_pct": float(bool(probes) and passed_count == len(probes)),
        "checks.budget_within_10_pct": float(budget_applicable and all(budget_checks)),
        "checks.budget_fill_applicable": float(budget_applicable),
        "checks.primary_budget_is_8000": float(budget == 8_000),
        "checks.non_evidentiary_smoke": float(non_evidentiary_smoke),
        "checks.exact_rerun_identical": float(exact_rerun_identical),
        "checks.dual_gold_agreement_ge_0_98": float(dual_gold_ready),
        "checks.pilot_kappa_ge_0_8": float(pilot_ready),
        "checks.claim_disagreement_le_0_02": float(claim_ready),
        "floor_retrieve_everything": everything,
        "prior_target_accuracy": prior,
        "gate_pass_count": float(passed_count),
        "gate_exclusion_count": float(len(probes) - passed_count),
    }
    for arm, rows in arm_outcomes.items():
        checks[f"macro_acc.{arm}"] = macro_accuracy(rows)
    validity_path = out_dir / "validity.json"
    validity_path.write_text(
        json.dumps(
            {
                "exact_rerun_identical": exact_rerun_identical,
                "dual_gold": "pass" if dual_gold_ready else "supported-not-run",
                "human_pilot": "pass" if pilot_ready else "supported-not-run",
                "claim_grader": "pass" if claim_ready else "supported-not-run",
                "a0_correct_probe_count": len(a0_correct),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    primary_row = contrasts.iloc[0].to_dict() if not contrasts.empty else {}
    validity_checks = (
        checks["checks.floor_retrieve_everything_ge_0_9"] == 1
        and checks["checks.prior_leq_0_3"] == 1
        and checks["checks.leakage_gate_100_pct"] == 1
        and checks["checks.primary_budget_is_8000"] == 1
        and checks["checks.exact_rerun_identical"] == 1
        and checks["checks.dual_gold_agreement_ge_0_98"] == 1
        and checks["checks.pilot_kappa_ge_0_8"] == 1
        and checks["checks.claim_disagreement_le_0_02"] == 1
        and not non_evidentiary_smoke
        and bool(primary_row.get("valid", False))
    )
    primary_row.update(
        {
            "valid_primary": validity_checks,
            "evidence_status": "non-evidentiary-smoke"
            if non_evidentiary_smoke
            else "evidentiary",
            "budget": budget,
        }
    )
    result = ExperimentResult(
        name="e2",
        metrics=checks,
        tables={
            "metrics": metrics_table,
            "contrasts": contrasts,
            "secondary_contrasts": secondaries,
            "budget_checks": budget_table,
        },
        artifacts=[
            answer_path,
            metrics_path,
            contrasts_path,
            secondaries_path,
            gate_path,
            budget_path,
            provider_path,
            manifest_path,
            cache_path,
            a0_correct_path,
            validity_path,
        ],
        primary=primary_row,
        check_details=(
            {
                "pilot_kappa_ge_0_8": {
                    "passed": None,
                    "value": "not-applicable-in-smoke",
                    "reason": "the preregistered 30-probe two-human pilot is not part of offline smoke",
                },
                "claim_disagreement_le_0_02": {
                    "passed": None,
                    "value": "not-applicable-in-smoke",
                    "reason": "the repeated model claim-grader audit is not used by deterministic smoke grading",
                },
            }
            if non_evidentiary_smoke
            else {}
        ),
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
        store = corpus.meta.get("store_handle")
        if not isinstance(store, StoreHandle):
            raise ValueError("E2 registry run requires the built S3 store_handle for A4")
        if store.layout != "S3":
            raise ValueError(f"E2 A4 requires store_handle layout S3, received {store.layout}")
        result = evaluate_e2(
            cfg=cfg,
            probes=probes,
            events=events,
            out_dir=out_dir,
            seed=seed,
            store=store,
            llm=make_llm(cfg),
            embeddings=make_embeddings(cfg),
            validation_audit={
                "dual_gold_agreement": float(corpus.meta.get("dual_gold_agreement", 0.0)),
                "pilot_kappa": float(corpus.meta.get("pilot_kappa", 0.0)),
                "claim_disagreement": float(corpus.meta.get("claim_disagreement", 1.0)),
            },
        )[0]
        if result.tables["contrasts"].empty or not (
            result.tables["contrasts"]["contrast"] == "A4-A3"
        ).any():
            raise RuntimeError("E2 registry run did not produce the mandatory A4-A3 contrast row")
        return result

    def chart(self, result: ExperimentResult, out_dir: Path) -> list[Path]:
        table = result.tables["metrics"]
        if table.empty:
            return []
        chart_data = pd.DataFrame(table[table["family"] != "macro"])
        return [e2_family_bars(chart_data, out_dir)]


EXPERIMENT = register_experiment(E2Experiment())

__all__ = [
    "DEFAULT_ARMS",
    "BOOTSTRAP_RESAMPLES",
    "E2Experiment",
    "ProbeOutcome",
    "evaluate_e2",
    "grade_answer",
    "load_probes",
    "macro_accuracy",
]
