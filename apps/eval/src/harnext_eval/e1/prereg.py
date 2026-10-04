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
# Registered contrasts. R5-R1 and R5-R2 are the original design contrasts; run 3 added
# the scorer bake-off against the random floor; run 4 (2026-09-06) registers the
# per-source global HBOS (R10) as the design condition and three global detector
# variants of R2 (R11 IsolationForest, R12 ECOD, R13 LOF). R8/R9 were removed after
# run 3 showed them identical to R5.
PRIMARY_CONTRASTS = (
    "R5-R1", "R5-R2", "R2-R0", "R4-R0", "R10-R0", "R10-R2", "R11-R2", "R12-R2", "R13-R2",
)
POLICIES = (*(f"R{index}" for index in range(8)), "R10", "R11", "R12", "R13")
# Run 5 (2026-10-04): Jev routing arms, active only when a decision cache is configured
# (config e1.jev_scores). R14 = software-engineering category router, R15 = direct urgency.
JEV_POLICIES = ("R14", "R15")
JEV_CONTRASTS = ("R14-R0", "R14-R2", "R15-R0", "R15-R2", "R14-R15", "R14-R11")


def policies_for(jev: bool) -> tuple[str, ...]:
    return (*POLICIES, *JEV_POLICIES) if jev else POLICIES


def contrasts_for(jev: bool, policies: tuple[str, ...] | None = None) -> tuple[str, ...]:
    contrasts = (*PRIMARY_CONTRASTS, *JEV_CONTRASTS) if jev else PRIMARY_CONTRASTS
    if policies is None:
        return contrasts
    return tuple(c for c in contrasts if set(c.split("-")) <= set(policies))


def active_policies(jev: bool, subset: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Registered arms of a run: all of ``policies_for(jev)``, or the configured subset in that order."""
    known = policies_for(jev)
    if not subset:
        return known
    unknown = sorted(set(subset) - set(known))
    if unknown:
        raise ValueError(f"e1.policies names unknown or inactive arms: {unknown}")
    return tuple(name for name in known if name in set(subset))
# Post-hoc label-definition sensitivity, registered before the run so the fusion
# rule cannot be chosen on the outcome (review finding 2). The registered label
# stays the weighted model at p >= 0.5; the others are reported alongside it.
LABEL_DEFINITIONS = (
    "weighted_model_p50",
    "any_outcome_lf",
    "two_outcome_lfs",
    "half_of_votes",
)
AMENDED_EXCLUSIONS = [
    "R10 fits one global HBOS per source and ranks by within-source percentile; R11-R13 are "
    "R2 with IsolationForest, ECOD and LOF; all share the R0-R6 monthly capacity.",
    "An LF whose positive support is below label_positive_support_min fails its gate.",
    "Secondary metrics are kept only if the design condition (R10) is separated from the "
    "random floor by more than twice the paired monthly standard error; the decision is "
    "recorded in metric_remediation.csv before any interpretation.",
    "Lateness metrics use timestamped, subject-separated situations derived from consecutive "
    "positives (label_situation_gap_hours); row-index affiliation/NAB are reported as *_rowindex only.",
]


def _hashed_config(config: ExperimentConfig) -> dict[str, Any]:
    """The registered config; run-5 Jev keys are left out when unset so earlier registrations
    of configs without Jev keep their hash."""
    payload = config.model_dump(mode="json")
    for key in ("jev_scores", "jev_prompt_sha256", "jev_registration", "policies"):
        if payload.get("e1", {}).get(key) is None:
            payload.get("e1", {}).pop(key, None)
    return payload


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
    jev = config.e1.jev_scores is not None
    jev_registration: dict[str, Any] = {}
    if jev:
        from harnext_eval.e1 import jev as jev_module

        jev_registration = {"jev": {
            "model": jev_module.JEV_MODEL,
            "prompt_version": jev_module.PROMPT_VERSION,
            "prompt_sha256": jev_module.prompt_sha256(),
            "categories": list(jev_module.CATEGORIES),
            "fast_categories": list(jev_module.FAST_CATEGORIES),
            "scores": {
                "R14": "sum of category probabilities over fast_categories + 0.001 * urgency",
                "R15": "urgency (noul)",
                "failed_decision": -1.0,
            },
            "admission": "R2-style monthly top-b% over all events, shared R0-R6 capacity, no guards",
            "dev_set": "2021-10..2021-12 (before the window); prompt never scored on window months before registration",
            "view": "JIRA summary/description/labels rebuilt as of the event from the changelog; issue keys, "
                    "PR numbers, hashes and links removed; author shown only as person/automation",
            "limitations": [
                "comment bodies and PR titles/bodies are export-time text (never versioned by the sources)",
                "the Jev build (2026-09-17) postdates the window: memorised project history cannot be excluded",
                "fast-lane categories name security problems, broken builds and release risks, which the "
                "outcome labeling functions also reflect; per-source results are reported",
            ],
            "required_gates": ["jev_decisions_complete (failed <= 0.1 %)", "jev_single_model",
                               "jev_decided_after_registration"],
            **({"prompt_registered_in": config.e1.jev_registration} if config.e1.jev_registration else {}),
            **({"arm_subset": "only the listed policies run; R3-R6 and R10-R13 keep their run-4 results "
                              "(same replay and configuration, deterministic); metric-remediation design = R14"}
               if config.e1.policies else {}),
        }}
    return {
        **jev_registration,
        "schema": "e1-prereg-v1",
        "replay_sha256": sha256_file(replay),
        "config_sha256": sha256_json(_hashed_config(config)),
        "policies": list(active_policies(jev, config.e1.policies)),
        "budgets_pct": [1.0, 2.0, 5.0, 10.0],
        "primary_metric": PRIMARY,
        "primary_contrasts": list(contrasts_for(jev, active_policies(jev, config.e1.policies))),
        "window": [value.isoformat() for value in window] if window else None,
        "fit_window_months": 12,
        "feature_history": "single chronological fold within requested window; four-week rolling baselines",
        "label_accuracy_min": 0.6,
        "label_coverage_min": 0.01,
        "label_positive_support_min": int(config.e1.label_positive_support_min),
        "sanity_relative_tolerance": config.e1.sanity_relative_tolerance,
        "label_situation_gap_hours": float(config.e1.label_situation_gap_hours),
        "rules": {
            "vote_thread_start_only": bool(config.engine.router.rules.vote_thread_start_only),
            "dedup_per_subject": bool(config.engine.router.rules.dedup_per_subject),
        },
        "label_definition_sensitivity": list(LABEL_DEFINITIONS),
        "excluded_label_functions": list(config.e1.exclude_label_functions),
        "exclusion_rules": EXCLUSIONS + (
            ["Labeling functions listed in excluded_label_functions are dropped before fitting; "
             "they are uncomputable from the registered sources and are not gated."]
            if config.e1.exclude_label_functions else []
        ) + list(AMENDED_EXCLUSIONS),
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
