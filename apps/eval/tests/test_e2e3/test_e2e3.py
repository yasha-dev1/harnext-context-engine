"""Offline synthetic tests for docs/evaluation-spec.md §7 E2 and E3."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest
from harnext_eval.agents.reader import Material, answer
from harnext_eval.config import EngineConfig, load_config
from harnext_eval.e2.arms import a1, a2, a3, a4, retrieve_everything
from harnext_eval.e2.run import evaluate_e2
from harnext_eval.e3.run import READ_BUDGETS, compute_erosion_slope, evaluate_e3
from harnext_eval.providers.embeddings import FakeEmbeddings
from harnext_eval.providers.llm import FakeLLM
from harnext_eval.stores.base import StoreHandle
from harnext_eval.types import EvalEvent, Probe, SnapshotRef

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


def _probes() -> list[Probe]:
    common = {"entity": "HNX-1", "T": NOW + timedelta(days=1), "gold_type": "exact"}
    return [
        Probe(
            probe_id="extract",
            family="extraction",
            question="What is the status of HNX-1?",
            gold="Done",
            source_event_ids=["e2"],
            **common,
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
            "INDEX.md": "# Index\n- [HNX-1](entities/HNX-1/OVERVIEW.md)\n",
            "entities/HNX-1/OVERVIEW.md": f"HNX-1 status: {status}\n",
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
    events = [_event("e1", NOW - timedelta(hours=2), "Open"), _event("e2", NOW, "Done")]
    store = _as_store(ToyStore(tmp_path / "store", "S3", "Done"))

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


def test_e3_curves_contrast_cost_and_erosion(tmp_path: Path) -> None:
    events = [_event("e1", NOW - timedelta(hours=2), "Open"), _event("e2", NOW, "Done")]
    s3 = _as_store(ToyStore(tmp_path / "s3", "S3", "Done"))
    s1 = _as_store(ToyStore(tmp_path / "s1", "S1", "Open"))

    result = evaluate_e3(
        stores=[s3, s1],
        cfg=_cfg(),
        probes=_probes(),
        events=events,
        out_dir=tmp_path / "e3",
        seed=3,
        llm=FakeLLM(),
        embeddings=FakeEmbeddings(),
    )

    assert set(result.tables["curve"]["budget"]) == set(READ_BUDGETS)
    assert len(result.tables["curve"]) == 6
    assert result.tables["contrasts"].iloc[0]["contrast"] == "S3-S1@8000"
    assert result.tables["contrasts"].iloc[0]["delta"] > 0
    assert result.tables["cost"]["build_tokens"].tolist() == [120, 120]
    assert result.metrics["checks.same_input_hash"] == 1
    assert {path.name for path in result.artifacts} == {
        "curve.csv",
        "health.csv",
        "erosion.csv",
        "cost.csv",
        "contrasts.csv",
    }


def test_erosion_slope_uses_ols_per_week() -> None:
    assert compute_erosion_slope([1, 2, 4, 8], [0.90, 0.85, 0.75, 0.55]) == pytest.approx(-0.05)


@pytest.fixture(autouse=True)
def _clean_materialised_checkouts() -> Iterator[None]:
    """Document that ToyStore checkouts are removed by A4 after every call."""

    yield
    for path in Path(tempfile.gettempdir()).glob("e2-toy-*"):
        shutil.rmtree(path, ignore_errors=True)
