"""Offline synthetic tests for docs/evaluation-spec.md §7 E2 and E3."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest
from harnext_eval.agents.reader import Material, answer, truncate_to_tokens
from harnext_eval.cli import _build_e3_conditions
from harnext_eval.config import EngineConfig, load_config
from harnext_eval.e2.arms import a1, a2, a3, a4, retrieve_everything, store_read
from harnext_eval.e2.run import (
    BOOTSTRAP_RESAMPLES as E2_BOOTSTRAP_RESAMPLES,
)
from harnext_eval.e2.run import (
    ProbeOutcome,
    _contains_complete_value,
    _paired_contrast,
    evaluate_e2,
    grade_answer,
)
from harnext_eval.e3.run import (
    BOOTSTRAP_RESAMPLES,
    READ_BUDGETS,
    StoreCondition,
    _checkpoint_times,
    _rederive_probe,
    _same_input_proof,
    compute_erosion_slope,
    evaluate_e3,
)
from harnext_eval.probes.gold import PythonGold
from harnext_eval.providers.embeddings import FakeEmbeddings
from harnext_eval.providers.llm import FakeLLM
from harnext_eval.providers.tokenizer import CallableTokenCounter
from harnext_eval.stores.base import StoreHandle
from harnext_eval.stores.layouts import configure_store
from harnext_eval.types import Answer, EvalEvent, GradeResult, Probe, SnapshotRef

NOW = datetime(2026, 5, 1, tzinfo=UTC)
CONFIG_PATH = Path(__file__).parents[2] / "configs" / "baseline-minimal.yaml"


def _cfg(budget: int = 8_000) -> EngineConfig:
    base = load_config(CONFIG_PATH).engine
    return base.model_copy(
        update={"reader": base.reader.model_copy(update={"budget_tokens": budget})}
    )


def _event(event_id: str, at: datetime, status: str, *, padding: str = "") -> EvalEvent:
    return EvalEvent(
        id=event_id,
        source="jira:test",
        type="issue.transition",
        subject="issue:HNX-1",
        time=at,
        mgtenant="test",
        data={"issue_key": "HNX-1", "field": "status", "to": status, "padding": padding},
    )


def _related_pr(at: datetime) -> EvalEvent:
    return EvalEvent(
        id="pr7",
        source="github:test",
        type="pull_request.merged",
        subject="pr:7",
        baseline_keys=["issue:HNX-1"],
        time=at,
        mgtenant="test",
        data={"number": 7, "title": "HNX-1 fix", "state": "merged"},
    )


def _probes() -> list[Probe]:
    common = {
        "entity": "issue:HNX-1",
        "T": NOW + timedelta(days=1),
        "gold_type": "exact",
    }
    return [
        Probe(
            probe_id="temporal",
            family="temporal",
            question=f"What was the status of HNX-1 as of {(NOW - timedelta(hours=2)).isoformat()}?",
            gold="Open",
            source_event_ids=["e1"],
            **common,
        ),
        Probe(
            probe_id="extract",
            family="extraction",
            question="What is the status of HNX-1?",
            gold="Done",
            source_event_ids=["e2"],
            **common,
        ),
        Probe(
            probe_id="links",
            family="multisource",
            question="Which pull requests or mail threads are related to HNX-1?",
            gold=["pr:7"],
            gold_type="links",
            source_event_ids=["pr7"],
            entity="issue:HNX-1",
            T=NOW + timedelta(days=1),
        ),
        Probe(
            probe_id="update",
            family="update",
            question="What is the latest status of HNX-1?",
            gold="Done",
            superseded_values=["Open"],
            source_event_ids=["e2"],
            **common,
        ),
        Probe(
            probe_id="absent",
            family="abstention",
            question="What is the assignee of HNX-1?",
            gold="UNKNOWN",
            **common,
        ),
    ]


class ToyStore:
    """Small immutable StoreHandle-shaped object for read-agent tests."""

    def __init__(self, root: Path, layout: str, status: str) -> None:
        self.layout = layout
        self.root = root
        self.worktree = root / "worktree"
        self.worktree.mkdir(parents=True)
        self.snapshots_csv = root / "snapshots.csv"
        self._files = {
            "INDEX.md": "# Index\n- [issue:HNX-1](entities/issue/HNX-1/OVERVIEW.md)\n",
            "entities/issue/HNX-1/OVERVIEW.md": (
                f"# issue:HNX-1\n- status: {status}\n"
                "- [facts](facts.md)\n- [shared](../../../shared/status.md)\n"
            ),
            "entities/issue/HNX-1/facts.md": f"issue:HNX-1 status={status}\n",
            "shared/status.md": f"issue:HNX-1 status: {status}\n",
        }
        with self.snapshots_csv.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(
                destination, fieldnames=["T_last_event", "sha", "last_event_id", "lane"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "T_last_event": NOW.isoformat(),
                    "sha": f"{layout.lower()}-sha",
                    "last_event_id": "e2",
                    "lane": "batch",
                }
            )
        (root / "usage.jsonl").write_text(
            json.dumps(
                {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "event_count": 2,
                    "cost_usd": 0.01,
                    "status": "ok",
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def snapshot(self, at: datetime) -> SnapshotRef:
        if at < NOW:
            raise LookupError
        return SnapshotRef(
            sha=f"{self.layout.lower()}-sha",
            T_last_event=NOW,
            last_event_id="e2",
            lane="batch",
        )

    def materialise(self, ref: SnapshotRef) -> Path:
        del ref
        checkout = Path(tempfile.mkdtemp(prefix="e2-toy-"))
        for relpath, content in self._files.items():
            target = checkout / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return checkout

    def list_files(self, ref: SnapshotRef) -> list[str]:
        del ref
        return sorted(self._files)

    def read(self, ref: SnapshotRef, relpath: str) -> str | None:
        del ref
        return self._files.get(relpath)


def _as_store(store: ToyStore) -> StoreHandle:
    return cast(StoreHandle, store)


def _real_store(
    root: Path,
    layout: str,
    events: list[EvalEvent],
    *,
    seed: int | None = None,
) -> StoreHandle:
    store = StoreHandle(layout, f"test-{layout.casefold()}-{seed or 0}", root)
    configure_store(store, embeddings=FakeEmbeddings())
    store.fold(events, "batch")
    return store


def _e3_events() -> list[EvalEvent]:
    return [
        _event("e1", NOW, "Open"),
        EvalEvent(
            id="e2",
            source="jira:test",
            type="issue.transition",
            subject="issue:HNX-2",
            time=NOW + timedelta(hours=1),
            mgtenant="test",
            data={"issue_key": "HNX-2", "field": "status", "to": "Open"},
        ),
        EvalEvent(
            id="e3",
            source="github:test",
            type="pull_request.merged",
            subject="pr:7",
            time=NOW + timedelta(hours=2),
            mgtenant="test",
            data={
                "number": 7,
                "title": "HNX-1 implement fix",
                "state": "merged",
                "changed_files": ["src/core/fix.py"],
            },
        ),
        _event("e4", NOW + timedelta(hours=3), "Done"),
        EvalEvent(
            id="e5",
            source="jira:test",
            type="issue.transition",
            subject="issue:HNX-2",
            time=NOW + timedelta(hours=4),
            mgtenant="test",
            data={"issue_key": "HNX-2", "field": "status", "to": "Done"},
        ),
    ]


def _e3_probes(events: list[EvalEvent]) -> list[Probe]:
    at = events[-1].time
    return [
        Probe(
            probe_id="extract-1",
            family="extraction",
            entity="HNX-1",
            T=at,
            question="What is the current status of HNX-1 at the snapshot time?",
            gold="Done",
            gold_type="exact",
            source_event_ids=["e4"],
        ),
        Probe(
            probe_id="extract-2",
            family="extraction",
            entity="HNX-2",
            T=at,
            question="What is the current status of HNX-2 at the snapshot time?",
            gold="Done",
            gold_type="exact",
            source_event_ids=["e5"],
        ),
        Probe(
            probe_id="temporal",
            family="temporal",
            entity="HNX-1",
            T=at,
            question=f"What was the status of HNX-1 as of {NOW.isoformat()}?",
            gold="Open",
            gold_type="exact",
            source_event_ids=["e1"],
        ),
        Probe(
            probe_id="update",
            family="update",
            entity="HNX-1",
            T=at,
            question="After all updates through the snapshot, what is the latest status of HNX-1?",
            gold="Done",
            gold_type="exact",
            superseded_values=["Open"],
            source_event_ids=["e1", "e4"],
        ),
        Probe(
            probe_id="links",
            family="multisource",
            entity="HNX-1",
            T=at,
            question="Which pull requests or mail threads are related to HNX-1?",
            gold=["pr:7"],
            gold_type="links",
            source_event_ids=["e3"],
        ),
        Probe(
            probe_id="absent",
            family="abstention",
            entity="MISSING-1",
            T=at,
            question="What is the status of MISSING-1?",
            gold="UNKNOWN",
            gold_type="exact",
        ),
    ]


def test_arms_enforce_budget(tmp_path: Path) -> None:
    cfg = _cfg(30)
    events = [
        _event("e1", NOW - timedelta(hours=2), "Open", padding="word " * 100),
        _event("e2", NOW - timedelta(hours=1), "Done", padding="word " * 100),
    ]
    probe = _probes()[0]
    store = _as_store(ToyStore(tmp_path / "store", "S3", "Done"))

    materials = [
        a1(probe, events, cfg, n=100),
        a2(probe, events, cfg, k=10),
        a3(probe, events, cfg, embeddings=FakeEmbeddings(), k=10),
        a4(probe, store, cfg),
    ]

    for material in materials:
        response = answer(probe, material, cfg, provider=FakeLLM())
        assert response.tokens_read <= 30
    assert (retrieve_everything(probe, events, cfg).original_tokens or 0) > 30


def test_a1_runs_both_n_conditions_includes_cutoff_and_matches_exact_keys() -> None:
    probe = Probe(
        probe_id="boundary",
        family="extraction",
        entity="issue:KAFKA-1",
        T=NOW,
        question="What is the status of KAFKA-1?",
        gold="Done",
        gold_type="exact",
    )
    events = [
        EvalEvent(
            id="owned",
            source="jira:test",
            type="transition",
            subject="issue:KAFKA-1",
            time=NOW,
            mgtenant="test",
            data={"field": "status", "to": "Done"},
        ),
        EvalEvent(
            id="prefix-collision",
            source="jira:test",
            type="transition",
            subject="issue:KAFKA-10",
            time=NOW,
            mgtenant="test",
            data={"field": "status", "to": "Wrong"},
        ),
        EvalEvent(
            id="prose-mention",
            source="mail:test",
            type="message",
            subject="thread:other",
            time=NOW,
            mgtenant="test",
            data={"body": "KAFKA-1 was mentioned, but this event belongs elsewhere"},
        ),
        EvalEvent(
            id="relation",
            source="github:test",
            type="pull_request.merged",
            subject="pr:9",
            baseline_keys=["issue:KAFKA-1"],
            time=NOW,
            mgtenant="test",
            data={"number": 9},
        ),
    ]

    n20 = a1(probe, events, _cfg(), n=20)
    n100 = a1(probe, events, _cfg(), n=100)

    assert n20.arm == "A1-N20"
    assert n100.arm == "A1-N100"
    assert '"id":"owned"' in n20.text
    assert '"id":"relation"' in n20.text
    assert "prefix-collision" not in n20.text
    assert "prose-mention" not in n20.text


def test_a4_starts_at_index_and_recursively_opens_canonical_entity_links(
    tmp_path: Path,
) -> None:
    probe = _probes()[0]
    material = a4(probe, _as_store(ToyStore(tmp_path / "store", "S3", "Done")), _cfg())

    assert material.text.startswith("[file:INDEX.md]")
    assert "[file:entities/issue/HNX-1/OVERVIEW.md]" in material.text
    assert "[file:entities/issue/HNX-1/facts.md]" in material.text
    assert "[file:shared/status.md]" in material.text
    assert material.tool_calls == 5  # list_files plus four actual reads


@pytest.mark.parametrize("layout", ["S4", "S5"])
def test_vector_layouts_are_read_only_through_snapshot_top_k(
    tmp_path: Path, layout: str
) -> None:
    events = [_event("e1", NOW, "Done")]
    store = _real_store(tmp_path / layout.casefold(), layout, events)
    material = store_read(
        _probes()[0],
        store,
        _cfg(),
        embeddings=FakeEmbeddings(),
    )

    assert material.arm == layout
    assert material.tool_calls == 1
    assert "Done" in material.text
    assert "[source:" in material.text


def test_provider_tokenizer_controls_truncation_and_logged_input_count() -> None:
    tokenizer = CallableTokenCounter(
        len,
        tokenizer_id="fixture-character-tokenizer",
        tokenizer_revision="2026-08-30",
    )
    accounting: dict[str, int | str | bool] = {}
    material = Material(arm="A2", text="abcdef", original_tokens=6)

    assert truncate_to_tokens("abcdef", 3, tokenizer=tokenizer) == "abc"
    response = answer(
        _probes()[0],
        material,
        _cfg(3),
        provider=FakeLLM(),
        tokenizer=tokenizer,
        accounting=accounting,
    )

    assert response.tokens_read == 3
    assert accounting["selected_material_tokens"] == 3
    assert accounting["provider_input_tokens"] > response.tokens_read
    assert accounting["tokenizer_id"] == "fixture-character-tokenizer"


def _outcome(
    probe_id: str,
    family: str,
    entity: str,
    arm: str,
    score: float,
) -> ProbeOutcome:
    probe = Probe(
        probe_id=probe_id,
        family=family,  # type: ignore[arg-type]
        entity=entity,
        T=NOW,
        question="fixture",
        gold="fixture",
        gold_type="exact",
    )
    return ProbeOutcome(
        probe=probe,
        answer=Answer(
            probe_id=probe_id,
            arm=arm,
            text="fixture",
            cited_ids=[],
            tokens_read=1,
            tool_calls=0,
            latency_s=0,
        ),
        grade=GradeResult(item_id=probe_id, metric="fixture", value=score, details={}),
        original_tokens=1,
        supersession_error=False,
    )


def test_primary_contrast_is_literal_equal_weight_family_macro() -> None:
    outcomes: list[ProbeOutcome] = []
    for entity in ("issue:A-1", "issue:A-2"):
        for arm, right_score in (("A4", 1.0), ("A3", 0.0)):
            outcomes.append(
                _outcome(f"{entity}-extract-1", "extraction", entity, arm, right_score)
            )
            outcomes.append(
                _outcome(f"{entity}-extract-2", "extraction", entity, arm, 0.0)
            )
            for family in ("temporal", "update", "multisource", "abstention"):
                outcomes.append(
                    _outcome(
                        f"{entity}-{family}",
                        family,
                        entity,
                        arm,
                        right_score,
                    )
                )

    contrast = _paired_contrast(outcomes, "A4", "A3", seed=7).iloc[0]

    assert contrast["delta"] == pytest.approx(0.9)
    assert contrast["n_resamples"] == E2_BOOTSTRAP_RESAMPLES
    assert contrast["valid"]


def test_code_location_primary_score_is_exact_set_with_named_secondaries() -> None:
    probe = Probe(
        probe_id="code",
        family="code_location",
        entity="issue:HNX-1",
        T=NOW,
        question="Which files changed?",
        gold=["src/core/a.py", "src/core/b.py"],
        gold_type="files",
    )

    partial = grade_answer(probe, "src/core/a.py")
    exact = grade_answer(probe, "src/core/a.py\nsrc/core/b.py")

    assert partial.metric == "exact_file_set"
    assert partial.value == 0
    assert partial.details["file_recall"] == 0.5
    assert "module_hit" in partial.details
    assert exact.value == 1

    typed_probe = probe.model_copy(update={"gold": ["Dockerfile", "scripts/release"]})
    typed = grade_answer(
        typed_probe,
        json.dumps({"files": ["Dockerfile", "scripts/release"], "modules": ["scripts/release"]}),
    )
    assert typed.value == 1


def test_supersession_detection_uses_complete_values_not_substrings() -> None:
    assert _contains_complete_value("Open", "Open")
    assert not _contains_complete_value("Reopened", "Open")


def test_leakage_gate_excludes_post_cutoff_source_and_preserves_reason(tmp_path: Path) -> None:
    before = _event("before", NOW, "Open")
    after = _event("after", NOW + timedelta(hours=1), "Done")
    store = _real_store(tmp_path / "store", "S3", [before])
    probe = Probe(
        probe_id="leaky",
        family="extraction",
        entity="issue:HNX-1",
        T=NOW,
        question="What is the status of HNX-1?",
        gold="Done",
        gold_type="exact",
        source_event_ids=["after"],
    )

    result, outcomes = evaluate_e2(
        cfg=_cfg(),
        probes=[probe],
        events=[before, after],
        out_dir=tmp_path / "e2",
        seed=1,
        store=store,
        llm=FakeLLM(),
        embeddings=FakeEmbeddings(),
        arms=("A0",),
    )

    assert not outcomes
    assert result.metrics["gate_exclusion_count"] == 1
    assert "source_event_after_T:after" in (tmp_path / "e2" / "gate.csv").read_text()


def test_reader_returns_unknown_without_evidence() -> None:
    response = answer(
        _probes()[0],
        Material(arm="A0", text="HNX-1 priority: Major"),
        _cfg(100),
        provider=FakeLLM(),
    )

    assert response.text == "UNKNOWN"
    assert response.cited_ids == []


def test_e2_tiny_end_to_end_writes_metrics_and_checks(tmp_path: Path) -> None:
    events = [
        _event("e1", NOW - timedelta(hours=2), "Open"),
        _related_pr(NOW - timedelta(hours=1)),
        _event("e2", NOW, "Done"),
    ]
    store = _real_store(tmp_path / "store", "S3", events, seed=1)

    result, outcomes = evaluate_e2(
        cfg=_cfg(80),
        probes=_probes(),
        events=events,
        out_dir=tmp_path / "e2",
        seed=7,
        store=store,
        llm=FakeLLM(),
        embeddings=FakeEmbeddings(),
    )

    assert outcomes
    assert result.metrics["checks.floor_retrieve_everything_ge_0_9"] == 1
    assert result.metrics["checks.leakage_gate_100_pct"] == 1
    assert {"answers.jsonl", "metrics.csv", "contrasts.csv", "gate.csv"} <= {
        path.name for path in result.artifacts
    }
    assert not result.tables["metrics"].empty
    assert result.primary["contrast"] == "A4-A3"
    assert result.primary["evidence_status"] == "non-evidentiary-smoke"
    assert result.primary["valid_primary"] is False
    assert {"A1-N20", "A1-N100"} <= set(result.tables["metrics"]["arm"])


def test_e3_curves_contrast_cost_and_erosion(tmp_path: Path) -> None:
    events = _e3_events()
    probes = _e3_probes(events)
    replay_hash = hashlib.sha256(
        "".join(event.model_dump_json() + "\n" for event in events).encode()
    ).hexdigest()
    conditions = [
        StoreCondition(
            _real_store(tmp_path / layout.casefold(), layout, events),
            tier="baseline",
            replay_hash=replay_hash,
        )
        for layout in ("S0", "S1", "S4")
    ]
    conditions.extend(
        StoreCondition(
            _real_store(tmp_path / f"s3-{seed}", "S3", events, seed=seed),
            seed=seed,
            tier="sonnet",
            replay_hash=replay_hash,
            model="fake",
        )
        for seed in (1, 2, 3)
    )

    result = evaluate_e3(
        stores=conditions,
        cfg=_cfg(),
        probes=probes,
        events=events,
        out_dir=tmp_path / "e3",
        seed=3,
        llm=FakeLLM(),
        embeddings=FakeEmbeddings(),
        erosion_probe_limit=len(probes),
        corpus_name="fixture",
    )

    assert set(result.tables["curve"]["budget"]) == set(READ_BUDGETS)
    assert set(result.tables["curve"]["layout"]) == {"S0", "S1", "S3", "S4"}
    assert set(result.tables["contrasts"]["contrast"]) == {
        "S3-S1@8000",
        "S3-S4@8000",
    }
    assert set(result.tables["contrasts"]["seed_count"]) == {3}
    assert set(result.tables["contrasts"]["n_resamples"]) == {BOOTSTRAP_RESAMPLES}
    assert result.tables["contrasts"]["valid"].all()
    assert result.tables["health_seed_spread"]["status"].eq("measured").all()
    s4_curve = result.tables["curve"].query("layout == 'S4' and budget == 8000").iloc[0]
    assert s4_curve["n"] == len(probes)
    assert s4_curve["macro_acc"] > 0
    assert result.metrics["s4_recall_at_10"] == pytest.approx(1.0)
    assert result.metrics["checks.same_input_hash"] == 1
    assert result.metrics["checks.same_input_ledger"] == 1
    assert {
        "curve.csv",
        "curve.png",
        "health.csv",
        "erosion.csv",
        "erosion.png",
        "cost.csv",
        "contrasts.csv",
    } <= {path.name for path in result.artifacts}


def test_e3_same_input_detects_changed_middle_event_with_same_last_id(tmp_path: Path) -> None:
    events = _e3_events()
    replay_hash = "frozen-replay"
    stores = [
        _real_store(tmp_path / layout.casefold(), layout, events)
        for layout in ("S0", "S1", "S4")
    ]
    stores.extend(
        _real_store(tmp_path / f"s3-{seed}", "S3", events, seed=seed)
        for seed in (1, 2, 3)
    )
    ledger = stores[-1].delivered_jsonl
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    rows[1]["event_id"] = "different-middle-event"
    ledger.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    conditions = [
        StoreCondition(
            store,
            seed=index - 2 if index >= 3 else None,
            tier="sonnet" if index >= 3 else "baseline",
            replay_hash=replay_hash,
        )
        for index, store in enumerate(stores)
    ]

    same_input, details = _same_input_proof(conditions)

    assert not same_input
    assert not details["ledger_identical"]
    assert details["mismatches"]["S3-sonnet-seed-3"]["ledger"]["first_difference"] == 1


def test_e3_erosion_rederives_gold_and_skips_future_probes() -> None:
    events = [
        _event("e1", NOW, "Open"),
        _event("e2", NOW + timedelta(days=1), "Done"),
        _event("e3", NOW + timedelta(days=2), "Closed"),
    ]
    probe = Probe(
        probe_id="update-late",
        family="update",
        entity="HNX-1",
        T=events[1].time,
        question="After all updates through the snapshot, what is the latest status of HNX-1?",
        gold="Done",
        gold_type="exact",
        superseded_values=["Open"],
        source_event_ids=["e1", "e2"],
    )
    gold = PythonGold(events)

    assert _rederive_probe(probe, NOW, events, gold) is None
    derived = _rederive_probe(probe, events[-1].time, events, gold)

    assert derived is not None
    assert derived.T == events[-1].time
    assert derived.gold == "Closed"
    assert derived.superseded_values == ["Open", "Done"]
    assert derived.source_event_ids == ["e1", "e2", "e3"]
    assert _checkpoint_times(events) == [("end", pytest.approx(2 / 7), events[-1].time)]


def test_e3_fails_closed_when_mandatory_s4_is_missing(tmp_path: Path) -> None:
    events = _e3_events()
    conditions = [
        StoreCondition(
            _real_store(tmp_path / layout.casefold(), layout, events),
            tier="baseline",
            replay_hash="same",
        )
        for layout in ("S0", "S1")
    ]
    conditions.append(
        StoreCondition(
            _real_store(tmp_path / "s3", "S3", events, seed=1),
            seed=1,
            tier="sonnet",
            replay_hash="same",
        )
    )

    with pytest.raises(ValueError, match="missing S4"):
        evaluate_e3(
            stores=conditions,
            cfg=_cfg(),
            probes=_e3_probes(events),
            events=events,
            out_dir=tmp_path / "e3",
            seed=1,
            llm=FakeLLM(),
            embeddings=FakeEmbeddings(),
        )


def test_cli_builds_one_aggregate_e3_matrix_and_skips_opus_in_smoke(
    tmp_path: Path,
) -> None:
    conditions = _build_e3_conditions(
        cfg=load_config(CONFIG_PATH),
        events=_e3_events(),
        root=tmp_path / "stores",
        replay_hash="replay-hash",
        smoke=True,
        optional_stores=set(),
        opus_model="claude-opus-5",
    )

    typed = [condition for condition in conditions if isinstance(condition, StoreCondition)]
    assert {condition.stable_label for condition in typed} == {
        "S0",
        "S1",
        "S4",
        "S3-sonnet-seed-1",
    }
    registry = json.loads((tmp_path / "stores" / "e3-registry.json").read_text())
    opus = next(row for row in registry if row["tier"] == "opus")
    assert opus["status"] == "supported-not-run"


def test_erosion_slope_uses_ols_per_week() -> None:
    assert compute_erosion_slope([1, 2, 4, 8], [0.90, 0.85, 0.75, 0.55]) == pytest.approx(-0.05)


@pytest.fixture(autouse=True)
def _clean_materialised_checkouts() -> Iterator[None]:
    """Document that ToyStore checkouts are removed by A4 after every call."""

    yield
    for path in Path(tempfile.gettempdir()).glob("e2-toy-*"):
        shutil.rmtree(path, ignore_errors=True)
