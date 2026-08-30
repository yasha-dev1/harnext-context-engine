"""Core configuration tests for docs/evaluation-spec.md §12."""

from pathlib import Path

import pytest
import yaml
from harnext_eval.config import load_config
from pydantic import ValidationError

CONFIGS = Path(__file__).parents[2] / "configs"


@pytest.mark.parametrize(
    "profile",
    ["baseline-minimal.yaml", "s1-templated.yaml", "s3-curated.yaml", "e6-twolane.yaml", "e6-single.yaml"],
)
def test_profiles_load(profile: str) -> None:
    cfg = load_config(CONFIGS / profile)
    assert cfg.engine.window.max_events == 20
    assert cfg.budgets.read_tokens == [2_000, 8_000, 32_000]


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load((CONFIGS / "baseline-minimal.yaml").read_text())
    raw["engine"]["router"]["budegt_pct"] = 3
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValidationError, match="budegt_pct"):
        load_config(path)
