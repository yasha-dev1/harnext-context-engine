"""Scale invariants for docs/evaluation-spec.md §7 E1."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pyarrow.parquet as pq
import pytest
from harnext_eval.config import load_config
from harnext_eval.corpus.synthetic import generate_synthetic_events
from harnext_eval.e1.features import CausalFeatureExtractor
from harnext_eval.e1.labels import _OutcomeIndex
from harnext_eval.e1.run import _FEATURE_CACHE, _policy_month, _tuning_start, _write_scores


def test_identity_index_omits_prose_and_keeps_issue_pr_and_message_ids() -> None:
    events = generate_synthetic_events(1, event_count=10, days=2, entity_count=2)
    events = [event.model_copy(update={"data": {
        "body": "ordinary prose Kafka-123 KIP-42 #17 pr:18 <message@example.org>",
    }}) for event in events]
    index = _OutcomeIndex(events)
    assert {"kafka-123", "kip-42", "#17", "pr:18", "message@example.org"} <= set(index.by_token)
    assert not {"ordinary", "prose", "body"} & set(index.by_token)
    assert all(len(indices) == 10 for indices in index.by_token.values())


def test_fit_window_is_twelve_prior_calendar_months() -> None:
    assert _tuning_start("2026-01") == "2025-01"
    assert _tuning_start("2024-02") == "2023-02"
    months = [f"{year}-{month:02d}" for year in (2024, 2025, 2026) for month in range(1, 13)]
    selected = [month for month in months if _tuning_start("2026-03") <= month < "2026-03"]
    assert len(selected) == 12
    assert selected[0] == "2025-03" and selected[-1] == "2026-02"


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="fork unavailable")
def test_parallel_scoring_matches_serial_and_compact_parquet(tmp_path, monkeypatch) -> None:
    events = generate_synthetic_events(9, event_count=20, days=140, entity_count=2)
    cfg = load_config("apps/eval/configs/e1-kafka.yaml").engine
    extractor = CausalFeatureExtractor()
    _FEATURE_CACHE[False] = {event.id: extractor.update(event) for event in events}
    labels = {event.id: float(event.data["injected_positive"]) for event in events}
    jobs = [(name, events[:12], events[12:], cfg, 1, labels, True) for name in ("R3", "R5")]
    try:
        serial = [_policy_month(job) for job in jobs]
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("fork")) as pool:
            parallel = list(pool.map(_policy_month, jobs))
        for left, right in zip(serial, parallel, strict=True):
            pd.testing.assert_frame_equal(left, right)
        split_r5 = pd.concat([_policy_month((*jobs[1], (budget,))) for budget in (1.0, 2.0, 5.0, 10.0)], ignore_index=True)
        pd.testing.assert_frame_equal(serial[1], split_r5)
        monkeypatch.setattr("harnext_eval.e1.run._SCORE_ROW_GROUP_SIZE", 7)
        frame = pd.concat(parallel, ignore_index=True)
        path = tmp_path / "scores.parquet"
        assert _write_scores(frame, path)
        stored = pd.read_parquet(path)
        assert len(stored) == 2 * 4 * 8
        assert not stored.duplicated(["event_id", "policy", "budget_pct"]).any()
        assert "population" not in stored and "rule_negative" in stored
        assert stored.features_fired.map(lambda value: isinstance(value, str)).all()
        metadata = pq.ParquetFile(path).metadata
        assert metadata.num_row_groups > 1
        assert all(metadata.row_group(i).num_rows <= 7 for i in range(metadata.num_row_groups))
        assert metadata.row_group(0).column(0).compression == "ZSTD"
    finally:
        _FEATURE_CACHE.clear()


def test_rolling_run_folds_features_once_and_caps_actual_fits(tmp_path, monkeypatch) -> None:
    from dataclasses import replace
    from datetime import UTC, datetime

    import harnext_eval.e1.run as runner
    from harnext_eval.cli import _handle_for_replay
    from harnext_eval.corpus.build_replay import write_replay

    events = generate_synthetic_events(2, event_count=30, days=900, entity_count=2)
    events = [event.model_copy(update={"time": datetime(2026 + i // 12, i % 12 + 1, 1, tzinfo=UTC)})
              for i, event in enumerate(events)]
    path = tmp_path / "replay.jsonl"
    write_replay(events, path)
    corpus = _handle_for_replay(path, "synthetic")
    corpus = replace(corpus, meta={**corpus.meta, "e1_only": True, "smoke": True})
    calls = []
    fits = []
    original_fit = runner._fitted_policy

    class CountedExtractor(CausalFeatureExtractor):
        def update(self, event):
            calls.append((self.global_only, event.id))
            return super().update(event)

    def fitted(name, tuning, cfg, seed, budget):
        fits.append(sorted({runner._month(event) for event in tuning}))
        return original_fit(name, tuning, cfg, seed, budget)

    monkeypatch.setattr(runner, "CausalFeatureExtractor", CountedExtractor)
    monkeypatch.setattr(runner, "_fitted_policy", fitted)
    runner.E1Experiment().run(load_config("apps/eval/configs/e1-kafka.yaml").engine,
                              corpus, tmp_path / "run", 1)
    assert len(calls) == len(set(calls)) == 2 * len(events)
    assert all(len(months) <= 12 for months in fits)
    assert not runner._FEATURE_CACHE


def test_vus_scale_optimizations_match_dense_reference_with_ties() -> None:
    import numpy as np
    from harnext_eval.e1.score import _buffered_index_ranges, _intervals, _range_pr_area

    rng = np.random.default_rng(31)
    for length in (20, 113):
        binary = rng.integers(0, 2, length)
        scores = rng.integers(0, 8, length).astype(float)
        coordinates = np.cumsum(rng.integers(0, 4, length)).astype(float)
        ranges = _intervals(binary)
        for window in (0.0, 1.0, 5.0):
            soft = binary.astype(float).copy()
            if window:
                for start, end in ranges:
                    distance = np.where(coordinates < coordinates[start],
                                        coordinates[start] - coordinates, coordinates - coordinates[end])
                    outside = (coordinates < coordinates[start]) | (coordinates > coordinates[end])
                    mask = outside & (distance > 0) & (distance <= window / 2)
                    soft[mask] += np.sqrt(np.maximum(0, 1 - distance[mask] / window))
            soft = np.minimum(soft, 1)
            buffered = _buffered_index_ranges(ranges, coordinates, window)
            thresholds = np.sort(scores)[::-1][np.linspace(0, length - 1, 250).astype(int)]
            recalls = []
            precisions = []
            for threshold in thresholds:
                predicted = scores >= threshold
                tp = np.dot(soft, predicted)
                credit = np.dot(soft - binary, predicted)
                existence = np.mean([np.max(scores[start:end + 1]) >= threshold for start, end in buffered])
                recalls.append(min(tp / (binary.sum() + credit / 2), 1) * existence)
                precisions.append(tp / predicted.sum())
            expected = np.dot(np.diff(np.r_[0.0, recalls]), precisions)
            actual = _range_pr_area(binary, scores, ranges, coordinates, window, thresholds=250)
            assert actual == pytest.approx(expected, abs=1e-12)
