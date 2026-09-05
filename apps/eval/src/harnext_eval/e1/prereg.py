"""Frozen E1 registration and read-only chronology verification (§7 E1/§6)."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harnext_eval.config import ExperimentConfig
from harnext_eval.manifest import current_git_sha, sha256_file, sha256_json

PRIMARY = "recall_at_2pct_rule_negative"
EXCLUSIONS = [
    "First two observed months are tuning only (one for a two-month pilot).",
    "Window is half-open [START, END); no events outside it reach labels or tuning.",
    "Censored or inapplicable labeling functions abstain; unknown labels excluded from quality metrics.",
    "Rule-negative is the primary population; full-population results are secondary.",
    "Failed LF accuracy/coverage gates invalidate evidence; functions are never silently retained as valid.",
    "Rules exceeding monthly capacity invalidate that month; no hidden capacity is added.",
    "Harm is N/A: no real action provider / no S3 store in this profile.",
]


def parse_window(values: tuple[str, str] | None) -> tuple[datetime, datetime] | None:
    if values is None:
        return None
    parsed = tuple(datetime.fromisoformat(value.replace("Z", "+00:00")) for value in values)
    start, end = (value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC) for value in parsed)
    if start >= end:
        raise ValueError("window START must precede END")
    return start, end


def registration(
    replay: Path, config: ExperimentConfig, window: tuple[datetime, datetime] | None,
) -> dict[str, Any]:
    return {
        "schema": "e1-prereg-v1",
        "replay_sha256": sha256_file(replay),
        "config_sha256": sha256_json(config),
        "policies": [f"R{index}" for index in range(8)],
        "budgets_pct": [1.0, 2.0, 5.0, 10.0],
        "primary_metric": PRIMARY,
        "primary_contrasts": ["R5-R1", "R5-R2"],
        "window": [value.isoformat() for value in window] if window else None,
        "fit_window_months": 12,
        "feature_history": "single chronological fold within requested window; four-week rolling baselines",
        "label_accuracy_min": 0.6,
        "label_coverage_min": 0.01,
        "excluded_label_functions": list(config.e1.exclude_label_functions),
        "exclusion_rules": EXCLUSIONS + (
            ["Labeling functions listed in excluded_label_functions are dropped before fitting; "
             "they are uncomputable from the registered sources and are not gated."]
            if config.e1.exclude_label_functions else []
        ),
        "git_head": current_git_sha(),
        "created_at": datetime.now(UTC).isoformat(),
    }


def write_prereg(
    replay: Path, config: ExperimentConfig, out: Path,
    window: tuple[datetime, datetime] | None,
) -> Path:
    """Create a reviewable registration; never overwrite an existing registration."""
    payload = registration(replay, config, window)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as target:
        target.write(
            "# E1 preregistration\n\n"
            "Commit this document before the first evidentiary run. Amendments must be appended with dates.\n\n"
            "```json\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n```\n"
        )
    return out


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=False)


def verify_prereg(
    path: Path, replay: Path, config: ExperimentConfig,
    window: tuple[datetime, datetime] | None, run_started: datetime,
) -> dict[str, Any]:
    """Require a matching committed blob on HEAD's ancestry, dated before run."""
    result: dict[str, Any] = {
        "prereg_ref": None, "prereg_hash": None,
        "prereg_predates_evaluation": False, "prereg_reason": "registration file is missing",
    }
    if not path.is_file():
        return result
    result.update(prereg_ref=str(path), prereg_hash=sha256_file(path))
    try:
        content = path.read_text(encoding="utf-8")
        payload = json.loads(content.split("```json\n", 1)[1].split("\n```", 1)[0])
        expected = registration(replay, config, window)
        for key in expected.keys() - {"git_head", "created_at"}:
            if payload.get(key) != expected[key]:
                raise ValueError(f"registration differs from run: {key}")
        root_result = _git("rev-parse", "--show-toplevel", cwd=path.resolve().parent)
        if root_result.returncode:
            raise ValueError("registration is outside a git repository")
        root = Path(root_result.stdout.strip())
        relative = path.resolve().relative_to(root).as_posix()
        log = _git("log", "-1", "--format=%H %ct", "HEAD", "--", relative, cwd=root)
        if not log.stdout.strip():
            raise ValueError("registration has no commit on HEAD ancestry")
        commit, timestamp = log.stdout.strip().split()
        blob = _git("show", f"{commit}:{relative}", cwd=root)
        if blob.returncode or blob.stdout != content:
            raise ValueError("registration has uncommitted changes")
        ancestor = _git("merge-base", "--is-ancestor", str(payload["git_head"]), commit, cwd=root)
        if ancestor.returncode:
            raise ValueError("registration generation HEAD is not an ancestor of its commit")
        if datetime.fromtimestamp(int(timestamp), UTC) >= run_started:
            raise ValueError("registration commit does not predate run start")
        result.update(
            prereg_predates_evaluation=True, prereg_commit=commit,
            prereg_reason="matching committed registration predates run start",
        )
    except (ValueError, KeyError, IndexError, OSError) as exc:
        result["prereg_reason"] = str(exc)
    return result
