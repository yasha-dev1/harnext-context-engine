"""Offline rolling E1 runner regressions for docs/evaluation-spec.md §7 E1."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from harnext_eval.config import load_config
from harnext_eval.corpus.synthetic import generate_synthetic_corpus
from harnext_eval.e1.run import E1Experiment, _run_harm_check
from harnext_eval.providers.llm import LLMResult
from harnext_eval.registry import get_experiment
from harnext_eval.stores.base import DeliveryRecord, StoreHandle
from harnext_eval.types import EvalEvent, SnapshotRef, Task


def test_e1_uses_sidecar_gold_global_monthly_admission_and_required_outputs(
    tmp_path, monkeypatch
) -> None:
    corpus = generate_synthetic_corpus(
        tmp_path / "replay.jsonl", seed=5, event_count=300, days=70, entity_count=10
    )
    corpus = replace(corpus, meta={**corpus.meta, "smoke": True})
    cfg = load_config("apps/eval/configs/baseline-minimal.yaml").engine
    experiment = get_experiment("e1")
    assert isinstance(experiment, E1Experiment)
    monkeypatch.setattr(
        "harnext_eval.e1.run.build_labels",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("exact sidecar runs must skip weak-label construction")
        ),
    )
    out = tmp_path / "e1"
    result = experiment.run(cfg, corpus, out, seed=5)

    assert result.primary["metric"] == "recall_at_2pct_rule_negative"
    assert {"r5_minus_r1_ci_low", "r5_minus_r1_ci_high", "r5_minus_r2_ci_low", "r5_minus_r2_ci_high"} <= set(result.primary)
    assert set(result.tables["scores"]["policy"]) == {f"R{index}" for index in range(8)}

    exact_positive_ids = {
        item["event_id"] for item in corpus.meta["injected_situations"]
    }
    evaluated = result.tables["scores"]
    labelled_positive = set(evaluated.loc[evaluated["label"] >= 0.5, "event_id"])
    assert labelled_positive == exact_positive_ids & set(evaluated["event_id"])
    assert evaluated["constructed_label"].all()

    r0 = evaluated[
        (evaluated["policy"] == "R0") & (evaluated["population"] == "full")
    ]
    for (_, budget), group in r0.groupby(["month", "budget_pct"]):
        expected = max(1, int(round(len(group) * float(budget) / 100.0)))
        assert group["admitted"].sum() == expected
        assert group["event_id"].nunique() == len(group)

    compared = evaluated[
        evaluated["policy"].isin({f"R{index}" for index in range(7)})
        & (evaluated["population"] == "full")
    ]
    for _, group in compared.groupby(["month", "policy", "budget_pct"]):
        assert group["admitted"].sum() <= group["capacity"].iloc[0]

    r7 = evaluated[(evaluated["policy"] == "R7") & (evaluated["population"] == "full")]
    assert r7["admitted"].all()

    r5 = evaluated[(evaluated["policy"] == "R5") & (evaluated["population"] == "full")]
    assert not r5.loc[~r5["eligible"] & ~r5["mandatory"], "admitted"].any()
    assert result.metrics["check.r5_ineligible_never_admitted"] == 1.0
    assert result.metrics["check.tuning_precedes_evaluation"] == 1.0
    assert result.metrics["check.r7_always_fast"] == 1.0
    assert result.metrics["check.valid"] == 0.0
    assert result.primary["valid"] is False
    assert result.check_details["human_sanity_100_two_annotators"]["status"] == "not_applicable"
    assert result.check_details["harm_paired_coverage"]["status"] == "not_applicable"
    assert {
        f"lf.{name}.coverage" for name in result.tables["label_diagnostics"]["function"]
    } <= set(result.check_details)

    metrics = result.tables["metrics"]
    assert "all" in set(metrics["source"])
    assert {
        "tokens",
        "dollars",
        "decision_latency_ms",
        "unused_capacity",
        "vus_pr",
        "affiliation_precision",
        "affiliation_recall",
        "nab_low_fn",
    } <= set(metrics)
    assert (metrics["tokens"] == 0).all()
    assert np.allclose(metrics["dollars"], 0.0)
    r7_metrics = metrics[
        (metrics["policy"] == "R7")
        & (metrics["population"] == "full")
        & (metrics["source"] == "all")
    ]
    assert np.allclose(r7_metrics["admission_rate"], 1.0)
    finite_recall = r7_metrics["recall_at_b"].dropna()
    assert not finite_recall.empty and np.allclose(finite_recall, 1.0)
    finite_precision = r7_metrics.dropna(subset=["precision_at_b", "prevalence"])
    assert np.allclose(
        finite_precision["precision_at_b"], finite_precision["prevalence"]
    )
    assert (r7_metrics["tokens"] == 0).all()
    assert not result.tables["delays"].empty
    assert {"delay_p50_s", "jitter_delay_p50_s", "affiliation_recall"} <= set(
        result.tables["robustness"].columns
    )
    situation_rows = result.tables["robustness"].dropna(
        subset=["affiliation_recall"]
    )
    assert set(situation_rows["policy"]) == {f"R{index}" for index in range(8)}
    assert set(situation_rows["budget_pct"]) == {1.0, 2.0, 5.0, 10.0}

    required = {
        "metrics.csv",
        "calibration.png",
        "operating_curves.png",
        "attribution.md",
        "harm.csv",
        "label_diagnostics.csv",
        "robustness.csv",
        "delays.csv",
        "preflight.csv",
        "validity.csv",
    }
    assert required <= {path.name for path in out.iterdir()}
    assert "future human-analysis step" not in (out / "attribution.md").read_text()
    assert pd.read_csv(out / "harm.csv").empty
    assert set(pd.read_csv(out / "preflight.csv")["status"]) <= {
        "run",
        "supported-not-run",
    }
    assert not result.tables["validity"].empty

    chart_paths = experiment.chart(result, out / "charts")
    assert {path.name for path in chart_paths} == {"calibration.png", "operating_curves.png"}


class _TinyS3Store(StoreHandle):
    def __init__(self, start: datetime) -> None:
        self.layout = "S3"
        self.refs = (
            SnapshotRef(sha="now", T_last_event=start, last_event_id="promoted", lane="fast"),
            SnapshotRef(
                sha="close",
                T_last_event=start + timedelta(seconds=10),
                last_event_id="next",
                lane="batch",
            ),
        )
        self.rows = (
            DeliveryRecord(
                sequence=0,
                event_id="promoted",
                event_time=start,
                sha="now",
                fold_index=0,
                fold_event_index=0,
                fold_max_event_time=start,
                snapshot_T_last_event=start,
                lane="fast",
            ),
            DeliveryRecord(
                sequence=1,
                event_id="next",
                event_time=start + timedelta(seconds=10),
                sha="close",
                fold_index=1,
                fold_event_index=0,
                fold_max_event_time=start + timedelta(seconds=10),
                snapshot_T_last_event=start + timedelta(seconds=10),
                lane="batch",
            ),
        )

    def snapshot(self, T: datetime) -> SnapshotRef:  # noqa: N803
        return [ref for ref in self.refs if ref.T_last_event <= T][-1]

    def delivery_records(
        self, ref: SnapshotRef | None = None
    ) -> tuple[DeliveryRecord, ...]:
        return self.rows[:1] if ref is not None and ref.sha == "now" else self.rows

    def list_files(self, ref: SnapshotRef) -> list[str]:
        del ref
        return ["entities/issue/HNX-1/facts.md"]

    def read(self, ref: SnapshotRef, relpath: str) -> str | None:
        del relpath
        return "phase: close" if ref.sha == "close" else "phase: now"


class _PhaseProvider:
    model_id = "fixture-action-v1"

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict[str, object] | None = None,
        max_tokens: int,
    ) -> LLMResult:
        del system, json_schema, max_tokens
        close = "phase: close" in user
        return LLMResult(
            text="",
            json={
                "assignee_candidates": ["alice" if close else "bob"],
                "reviewer_candidates": [],
                "component": None,
                "duplicate_of": None,
                "priority_change": None,
                "suspected_locations": [],
                "draft_reply": "",
                "cited_ids": ["ID-1"] if close else [],
                "action": "route",
            },
            usage={"input_tokens": 10, "output_tokens": 5},
        )


def test_harm_check_executes_each_r5_promotion_now_and_at_next_window_close(
    tmp_path,
) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)

    def event(event_id: str, seconds: int) -> EvalEvent:
        return EvalEvent(
            id=event_id,
            source="jira:test",
            type="jira.issue.updated",
            subject="issue:HNX-1",
            time=start + timedelta(seconds=seconds),
            mgtenant="test",
            baseline_keys=["component:runtime"],
            data={"body": event_id},
        )

    events = [event("promoted", 0), event("next", 10), event("gold-outcome-99", 100)]
    task = Task(
        task_id="harm:promoted",
        corpus="synthetic",
        T=start,
        trigger_event_id="promoted",
        entity="issue:HNX-1",
        kind="fast",
        gold={
            "people": {
                "assignees": ["alice"],
                "decision_times": [(start + timedelta(seconds=100)).isoformat()],
            },
            "category": {"required_ids": ["ID-1"], "decision_times": []},
            "place": {},
            "text": {},
        },
        gold_coverage={"people": True, "category": False, "place": False, "text": False},
    )
    corpus = generate_synthetic_corpus(
        tmp_path / "unused.jsonl", seed=1, event_count=20, days=2, entity_count=2
    )
    corpus = replace(
        corpus,
        replay_path=tmp_path / "harm.jsonl",
        meta={
            **corpus.meta,
            "store_handle": _TinyS3Store(start),
            "harm_provider": _PhaseProvider(),
            "harm_tasks": [task],
        },
    )
    corpus.replay_path.write_text(
        "".join(item.model_dump_json() + "\n" for item in events), encoding="utf-8"
    )
    scores = pd.DataFrame(
        [
            {
                "event_id": "promoted",
                "policy": "R5",
                "budget_pct": 2.0,
                "population": "full",
                "admitted": True,
            }
        ]
    )
    cfg = load_config("apps/eval/configs/baseline-minimal.yaml").engine

    harm, evidence = _run_harm_check(corpus, cfg, events, scores, tmp_path)

    assert len(harm) == 1
    assert harm.iloc[0]["quality_now"] == 0.0
    assert harm.iloc[0]["quality_window_close"] == 1.0
    assert harm.iloc[0]["harm_delta"] == -1.0
    assert harm.iloc[0]["snapshot_now"] == "now"
    assert harm.iloc[0]["snapshot_window_close"] == "close"
    assert evidence == {
        "store_provided": True,
        "store_s3": True,
        "promoted": 1,
        "paired": 1,
        "leakage_pass": True,
        "non_vacuous": True,
        "real_provider": True,
    }
    assert len((tmp_path / "harm-gate.csv").read_text().splitlines()) == 3
