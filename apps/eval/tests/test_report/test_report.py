"""Integration test for self-contained report rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from harnext_eval.report.charts import e2_family_bars
from harnext_eval.report.report import build_report


def test_build_report_discovers_outputs_and_embeds_charts(tmp_path: Path) -> None:
    run_dir = tmp_path / "fake-run"
    experiment_dir = run_dir / "e2" / "seed-1"
    chart_dir = experiment_dir / "charts"
    chart_dir.mkdir(parents=True)
    manifest = {
        "run_id": "fake-eval-run",
        "git_sha": "abc123",
        "replay_hash": "replay-hash",
        "seeds": [1, 2, 3],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = {
        "engine": {
            "router": {
                "rules": {"enabled": True},
                "deviation": {"enabled": True},
                "budget_pct": 2.0,
                "guards": {
                    "absolute_floor": 10,
                    "multi_window": True,
                    "situation_dedup": True,
                },
            },
            "window": {"gap_s": 30, "max_events": 20, "max_age_s": 120},
            "store": {"layout": "S3", "backend": "git"},
            "builder": {"harness": "fake", "model": None, "prompt_version": "v1"},
            "reader": {"provider": "fake", "budget_tokens": 8_000},
            "envelope": "V3",
        },
        "budgets": {"read_tokens": [2_000, 8_000, 32_000]},
        "seeds": [1, 2, 3],
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    results = {
        "name": "e2",
        "metrics": {"macro_acc": 0.82},
        "checks": {
            "leakage_gate": {"passed": True, "value": 1.0},
            "answerability": {
                "passed": False,
                "value": 0.88,
                "reason": "retrieve-everything macro accuracy was 0.88",
            },
        },
        "primary": {"A4_minus_A3": 0.12, "ci_low": 0.04, "ci_high": 0.2},
    }
    (experiment_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")
    pd.DataFrame(
        {
            "contrast": ["A4 - A3"],
            "delta": [0.12],
            "ci_low": [0.04],
            "ci_high": [0.2],
        }
    ).to_csv(experiment_dir / "contrasts.csv", index=False)
    pd.DataFrame({"arm": ["A3", "A4"], "macro_acc": [0.7, 0.82]}).to_csv(
        experiment_dir / "metrics.csv", index=False
    )
    chart = e2_family_bars(
        pd.DataFrame(
            {
                "arm": ["A3", "A4"],
                "family": ["update", "update"],
                "accuracy": [0.68, 0.84],
            }
        ),
        chart_dir,
    )

    report = build_report(run_dir)

    assert report == run_dir / "report.html"
    html = report.read_text(encoding="utf-8")
    assert "fake-eval-run" in html
    assert "S3" in html
    assert "Nudges summary" in html
    assert "A4_minus_A3" in html
    assert "A4 - A3" in html
    assert "leakage_gate" in html
    assert "Reason" in html
    assert "retrieve-everything macro accuracy was 0.88" in html
    assert "PASS" in html
    assert "data:image/png;base64," in html
    assert chart.name in html
    assert "e2/seed-1/metrics.csv" in html
    assert "http://" not in html and "https://" not in html
