"""Offline rolling E1 runner test for docs/evaluation-spec.md §7 E1."""

from harnext_eval.config import load_config
from harnext_eval.corpus.synthetic import generate_synthetic_corpus
from harnext_eval.e1.run import E1Experiment
from harnext_eval.registry import get_experiment


def test_e1_registered_runner_writes_required_outputs(tmp_path) -> None:
    corpus = generate_synthetic_corpus(
        tmp_path / "replay.jsonl", seed=5, event_count=300, days=70, entity_count=10
    )
    cfg = load_config("apps/eval/configs/baseline-minimal.yaml").engine
    experiment = get_experiment("e1")
    assert isinstance(experiment, E1Experiment)
    out = tmp_path / "e1"
    result = experiment.run(cfg, corpus, out, seed=5)
    assert result.primary["metric"] == "recall_at_2pct_rule_negative"
    assert (out / "scores.parquet").is_file()
    assert (out / "metrics.csv").is_file()
    assert (out / "calibration.csv").is_file()
    assert result.metrics["check.always_flag_recall_one"] == 1.0
    assert set(result.tables["scores"]["policy"]) == {f"R{index}" for index in range(8)}
    chart_paths = experiment.chart(result, out / "charts")
    assert {path.name for path in chart_paths} == {"calibration.png", "operating_curves.png"}
