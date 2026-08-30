"""Adversarial E5 regressions for docs/evaluation-spec.md §4 and §7 E5."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from harnext_eval.config import load_config
from harnext_eval.corpus import CorpusHandle
from harnext_eval.e5.run import (
    MeteredStoreHandle,
    PriceTable,
    _cadence_cfg,
    _equal_accuracy,
    _event_clock_replay,
    _first_mentions,
    _fixed_macro,
    _freshness_rows,
    _nontrivial_accuracy,
    _paired_ratio_bca,
    _prices,
    _record_cost,
    _rederive_probes,
    _urgency_labels,
    cadence_setting,
)
from harnext_eval.stores.base import register_layout
from harnext_eval.types import EvalEvent, Probe

BASE = datetime(2026, 5, 1, tzinfo=UTC)


def _event(
    event_id: str,
    second: int,
    *,
    subject: str = "issue:HNX-1",
    data: dict[str, object] | None = None,
) -> EvalEvent:
    return EvalEvent(
        id=event_id,
        source="jira:test",
        type="jira.event",
        subject=subject,
        time=BASE + timedelta(seconds=second),
        mgtenant="test",
        baseline_keys=["component:test"],
        data=data or {"field": "status", "to": event_id},
    )


def _cfg():
    return load_config("apps/eval/configs/baseline-minimal.yaml").engine


def _store(tmp_path: Path, name: str = "S0") -> MeteredStoreHandle:
    root = tmp_path / name.casefold()
    return MeteredStoreHandle(name, f"test-{name}", root, usage_path=root / "usage.jsonl")


def test_ours_constructs_r5_fits_only_warmup_and_audits_causal_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fitted: list[list[str]] = []
    requested: list[str] = []

    class SpyPolicy:
        name = "R5"
        baseline_key_used = "component:test"
        features_fired = {"scorer": "spy-hbos", "eligible": True}
        threshold = 5.0

        def fit(self, events: list[EvalEvent]):
            fitted.append([event.id for event in events])
            return self

        def rules(self, event: EvalEvent) -> str | None:
            return "declared" if event.id == "rule" else None

        def score(self, event: EvalEvent) -> float:
            del event
            return 10.0

    def fake_make_policy(name, cfg, *, seed=0):
        del cfg, seed
        requested.append(name)
        return SpyPolicy()

    monkeypatch.setattr("harnext_eval.e5.run.make_policy", fake_make_policy)
    warmup = [_event("warm-0", 0), _event("warm-1", 1)]
    evaluation = [_event(f"eval-{index}", 10 + index) for index in range(100)]
    events = warmup + evaluation + [_event("rule", 200, data={"priority": "Critical"})]
    setting = cadence_setting("W20+rules+deviation")
    cfg = _cadence_cfg(_cfg(), setting)
    assert cfg.router.guards.absolute_floor >= 1
    assert cfg.router.guards.multi_window is True

    result = _event_clock_replay(
        events,
        cfg,
        setting,
        _store(tmp_path),
        warmup_cutoff=BASE + timedelta(seconds=2),
        seed=7,
    )

    assert requested == ["R5"]
    assert fitted == [["warm-0", "warm-1"]]
    deviations = [
        row for row in result.decisions if row["lane"] == "fast" and row["rule"] is None
    ]
    assert deviations
    assert all(
        row["deviation_admitted_so_far"] <= row["deviation_budget_capacity"]
        for row in result.decisions
    )
    assert result.decisions[-1]["rule"] == "declared"
    assert result.decisions[-1]["lane"] == "fast"
    assert {row["policy"] for row in result.decisions} == {"R5"}


def test_ours_never_admits_an_r5_guard_ineligible_deviation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class IneligiblePolicy:
        name = "R5"
        baseline_key_used = "component:test"
        features_fired = {"eligible": False, "multi_window_confirmed": False}
        threshold = 1.0

        def fit(self, events: list[EvalEvent]):
            del events
            return self

        def rules(self, event: EvalEvent) -> str | None:
            del event
            return None

        def score(self, event: EvalEvent) -> float:
            del event
            return 100.0

    monkeypatch.setattr(
        "harnext_eval.e5.run.make_policy",
        lambda name, cfg, seed=0: IneligiblePolicy(),
    )
    events = [_event(f"event-{index}", index) for index in range(100)]
    setting = cadence_setting("W20+rules+deviation")
    result = _event_clock_replay(
        events,
        _cadence_cfg(_cfg(), setting),
        setting,
        _store(tmp_path),
        warmup_cutoff=None,
        seed=1,
    )

    assert not any(row["lane"] == "fast" for row in result.decisions)
    assert all(row["r5_eligible"] is False for row in result.decisions)


def test_fast_event_flushes_older_entity_state_before_fast_commit(tmp_path: Path) -> None:
    events = [
        _event("older", 0),
        _event("urgent", 1, data={"priority": "Critical"}),
    ]
    setting = cadence_setting("W20+rules")
    result = _event_clock_replay(
        events,
        _cadence_cfg(_cfg(), setting),
        setting,
        _store(tmp_path),
        warmup_cutoff=None,
        seed=1,
    )

    assert [row["close_reason"] for row in result.folds] == ["pre_fast", "fast"]
    assert [row["event_ids"] for row in result.folds] == [["older"], ["urgent"]]
    assert [row["commit_time"] for row in result.folds] == [
        (BASE + timedelta(seconds=1)).isoformat(),
        (BASE + timedelta(seconds=1)).isoformat(),
    ]


def test_durable_diff_not_delivery_ledger_defines_first_mention(tmp_path: Path) -> None:
    def no_mention(store, events, lane):
        del events, lane
        store.write("state.md", "updated without provenance\n")

    register_layout("E5NOMENTION", no_mention)
    store = _store(tmp_path, "E5NOMENTION")
    event = _event("must-appear", 0)
    ref = store.fold_at([event], "batch", commit_time=event.time, close_reason="cap")

    with pytest.raises(ValueError, match="no durable snapshot diff mentions"):
        _first_mentions(store, store.fold_rows, [event.id])
    assert ref.sha


def test_provider_layout_fails_when_fold_emits_no_usage_record(tmp_path: Path) -> None:
    def mentions_event(store, events, lane):
        del lane
        store.write("state.md", "\n".join(event.id for event in events))

    register_layout("E5NOUSAGE", mentions_event)
    store = _store(tmp_path, "E5NOUSAGE")
    store.provider_required = True

    with pytest.raises(ValueError, match="emitted 0 provider usage records"):
        store.fold_at([_event("e-1", 0)], "batch", commit_time=BASE, close_reason="cap")


def test_usage_cost_ignores_provider_dollars_and_requires_cache_prices() -> None:
    prices = PriceTable(2.0, 15.0, None, None, "model-a", "2026-08-01")
    cost, tokens = _record_cost(
        {
            "status": "success",
            "model": "model-a",
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "cost_usd": 999,
        },
        prices,
    )
    assert cost == 17.0
    assert tokens["input_tokens"] == 1_000_000

    with pytest.raises(ValueError, match="cache-read tokens"):
        _record_cost(
            {
                "status": "success",
                "model": "model-a",
                "cache_read_input_tokens": 1,
            },
            prices,
        )
    with pytest.raises(ValueError, match="frozen 'prices'"):
        _prices(_cfg().model_copy(update={"prices": None}), {})
    configured = _cfg().model_copy(
        update={"prices": {"input_per_million": 3.0, "output_per_million": 12.0}}
    )
    assert _prices(configured, {}).input_per_million == 3.0


def test_usage_cost_reads_authentic_nested_sdk_usage_and_frozen_fake_prices() -> None:
    prices = PriceTable(2.0, 10.0, 3.0, 0.5, "model-a", "2026-08-01")
    cost, tokens = _record_cost(
        {
            "status": "success",
            "model": "model-a",
            "usage": {
                "usage": {
                    "input_tokens": 1_000_000,
                    "output_tokens": 500_000,
                    "cache_creation_input_tokens": 100_000,
                    "cache_read_input_tokens": 200_000,
                }
            },
        },
        prices,
    )

    assert tokens == {
        "input_tokens": 1_000_000,
        "output_tokens": 500_000,
        "cache_creation_input_tokens": 100_000,
        "cache_read_input_tokens": 200_000,
    }
    assert cost == pytest.approx(7.4)
    fake_cost, _ = _record_cost(
        {
            "status": "success",
            "model": "fake",
            "input_tokens": 10_000,
            "output_tokens": 10_000,
            "cost_usd": 999.0,
        },
        PriceTable(0.0, 0.0, None, None, "fake", "offline-zero"),
    )
    assert fake_cost == 0.0


def test_frozen_probe_cutoff_is_preserved_and_gold_is_rederived() -> None:
    events = [
        _event("opened", 0, data={"field": "status", "to": "Open"}),
        _event("closed", 10, data={"field": "status", "to": "Closed"}),
    ]
    cutoff = BASE + timedelta(seconds=5)
    probe = Probe(
        probe_id="p-status",
        family="extraction",
        entity="issue:HNX-1",
        T=cutoff,
        question="What is the current status of issue:HNX-1 at the snapshot time?",
        gold="Closed",
        gold_type="exact",
        source_event_ids=["closed"],
    )

    derived = _rederive_probes([probe], events)[0]

    assert derived.T == cutoff
    assert derived.gold == "Open"
    assert derived.source_event_ids == ["opened"]


def test_urgency_uses_frozen_e1_artifact_and_rejects_negative_freshness(
    tmp_path: Path,
) -> None:
    events = [_event("urgent", 0), _event("routine", 1)]
    replay = tmp_path / "replay.jsonl"
    replay.write_text("".join(event.model_dump_json() + "\n" for event in events))
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps({"event_id": "urgent", "p_urgent": 0.9})
        + "\n"
        + json.dumps({"event_id": "routine", "p_urgent": 0.1})
        + "\n"
    )
    corpus = CorpusHandle("R", replay, None, None, "R-H1", {"e1_labels_path": str(labels)})

    values, provenance, frozen = _urgency_labels(events, corpus)

    assert values == {"urgent": True, "routine": False}
    assert provenance == "revealed-e1-frozen"
    assert frozen is True
    with pytest.raises(ValueError, match="negative freshness"):
        _freshness_rows(
            "W20",
            1,
            events,
            {event.id: event.time - timedelta(seconds=1) for event in events},
            values,
            provenance,
        )


def test_ratio_has_paired_entity_bca_ci_and_missing_families_invalidate_macro() -> None:
    ratio, low, high = _paired_ratio_bca(
        [1.0, 2.0, 3.0],
        [10, 10, 10],
        [2.0, 4.0, 6.0],
        [10, 10, 10],
        n_resamples=500,
        seed=3,
    )
    assert ratio == pytest.approx(0.5)
    assert low == pytest.approx(0.5)
    assert high == pytest.approx(0.5)
    assert _equal_accuracy(0.0) is True
    assert _equal_accuracy(-0.001) is False
    assert _nontrivial_accuracy([0.4, 0.4]) is False
    assert _nontrivial_accuracy([0.4, 0.5]) is True

    extraction_only = [SimpleNamespace(probe=SimpleNamespace(family="extraction"))]
    assert math.isnan(_fixed_macro(extraction_only))
