"""Probe-set CLI for docs/evaluation-spec.md §5 and §7 E2."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from harnext_eval.probes.common import load_replay, parse_time, validate_period
from harnext_eval.probes.gen_abstention import generate_abstention_probes
from harnext_eval.probes.gen_code_location import generate_code_location_probes
from harnext_eval.probes.gen_extraction import generate_extraction_probes
from harnext_eval.probes.gen_multisource import (
    JoinAuditTrail,
    generate_multisource_probes,
    write_join_report,
)
from harnext_eval.probes.gen_temporal import generate_temporal_probes
from harnext_eval.probes.gen_update import generate_update_probes
from harnext_eval.probes.gold import GoldAuditTrail, RawJiraInput, write_gold_report
from harnext_eval.types import EvalEvent, Probe


def generate_probe_set(
    events: list[EvalEvent],
    *,
    per_family: int,
    seed: int,
    probe_start: datetime,
    probe_end: datetime,
    multisource_code_count: int | None = None,
    evidentiary: bool = False,
    minimum_entities: int = 150,
    raw_jira: RawJiraInput | None = None,
    gold_resolutions: dict[str, object] | None = None,
    join_expected: dict[str, list[str]] | None = None,
    a0_answers: dict[str, str] | None = None,
    gold_audit: GoldAuditTrail | None = None,
    join_audit: JoinAuditTrail | None = None,
) -> list[Probe]:
    """Generate five equally weighted families with code inside multi-source."""

    validate_period(probe_start, probe_end)
    if per_family < 0:
        raise ValueError("per-family must be non-negative")
    code_count = (
        per_family // 2
        if multisource_code_count is None and evidentiary
        else (multisource_code_count or 0)
    )
    if not 0 <= code_count <= per_family:
        raise ValueError("multisource-code-count must be between zero and per-family")
    if evidentiary and per_family != 60:
        raise ValueError("evidentiary E2 requires exactly 60 probes per macro family")
    common = {
        "count": per_family,
        "seed": seed,
        "probe_start": probe_start,
        "probe_end": probe_end,
    }
    audit = gold_audit or GoldAuditTrail(
        source="raw-jira-export" if raw_jira is not None else "normalised-smoke-adapter",
        resolutions=gold_resolutions or {},
    )
    links_audit = join_audit or JoinAuditTrail(expected=join_expected or {})
    state_common = {**common, "raw_jira": raw_jira, "gold_audit": audit}
    probes: list[Probe] = []
    probes.extend(generate_extraction_probes(events, **state_common))
    probes.extend(generate_temporal_probes(events, **state_common))
    probes.extend(generate_update_probes(events, **state_common))
    probes.extend(
        generate_multisource_probes(
            events,
            **{**common, "count": per_family - code_count},
            join_audit=links_audit,
        )
    )
    probes.extend(
        generate_code_location_probes(
            events,
            **{**common, "count": code_count},
        )
    )
    probes.extend(generate_abstention_probes(events, **common))
    if len({probe.probe_id for probe in probes}) != len(probes):
        raise ValueError("generated duplicate probe ids")
    family_counts = Counter(probe.family for probe in probes)
    expected_counts = {
        family: per_family
        for family in ("extraction", "temporal", "update", "multisource", "abstention")
    }
    if family_counts != expected_counts:
        raise ValueError(f"probe families are not equally weighted: {dict(family_counts)}")
    audit.require_valid(evidentiary=evidentiary)
    links_audit.require_valid(evidentiary=evidentiary)
    if evidentiary:
        validate_evidentiary_population(probes, minimum_entities=minimum_entities)
        validate_abstention_prior(probes, a0_answers)
    return probes


def validate_evidentiary_population(
    probes: Sequence[Probe], *, minimum_entities: int = 150
) -> None:
    """Reject frozen E2 sets that violate D9's exact population constraints."""

    counts = Counter(probe.family for probe in probes)
    expected = {
        family: 60
        for family in ("extraction", "temporal", "update", "multisource", "abstention")
    }
    if len(probes) != 300 or counts != expected:
        raise ValueError("evidentiary E2 requires 300 probes: 60 in each of five families")
    entity_count = len({probe.entity.casefold() for probe in probes})
    if entity_count < minimum_entities:
        raise ValueError(
            f"evidentiary E2 requires at least {minimum_entities} entities; got {entity_count}"
        )


def validate_abstention_prior(
    probes: Sequence[Probe],
    answers: dict[str, str] | None,
    *,
    maximum_accuracy: float = 0.3,
) -> float:
    """Reject a frozen abstention set answerable by the no-material A0 prior."""

    abstentions = [probe for probe in probes if probe.family == "abstention"]
    if not abstentions:
        raise ValueError("A0 audit requires abstention probes")
    if answers is None:
        raise ValueError("evidentiary abstention generation requires --a0-audit")
    missing = [probe.probe_id for probe in abstentions if probe.probe_id not in answers]
    if missing:
        raise ValueError(f"A0 audit is missing {len(missing)} abstention answers")
    correct = sum(
        answers[probe.probe_id].strip().casefold().rstrip(".!?") == "unknown"
        for probe in abstentions
    )
    accuracy = correct / len(abstentions)
    if accuracy > maximum_accuracy:
        raise ValueError(
            f"abstention A0 accuracy {accuracy:.3%} exceeds {maximum_accuracy:.0%}"
        )
    return accuracy


def write_probe_set(probes: Sequence[Probe], output: str | Path) -> tuple[Path, Path, str]:
    """Write canonical JSONL and a conventional SHA-256 sidecar."""

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(probe.model_dump_json() + "\n" for probe in probes).encode()
    output_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = Path(f"{output_path}.sha256")
    sidecar.write_text(f"{digest}  {output_path.name}\n", encoding="utf-8")
    return output_path, sidecar, digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate frozen harnext evaluation probes")
    parser.add_argument("--replay", required=True, type=Path, help="EvalEvent JSONL replay")
    parser.add_argument("--out", required=True, type=Path, help="output probes JSONL")
    parser.add_argument("--per-family", default=60, type=int)
    parser.add_argument("--multisource-code-count", type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--probe-start", required=True)
    parser.add_argument("--probe-end", required=True)
    parser.add_argument("--evidentiary", action="store_true")
    parser.add_argument("--minimum-entities", default=150, type=int)
    parser.add_argument("--raw-jira", type=Path)
    parser.add_argument("--gold-resolutions", type=Path)
    parser.add_argument("--join-audit", type=Path)
    parser.add_argument("--a0-audit", type=Path)
    parser.add_argument("--gold-report", type=Path)
    parser.add_argument("--join-report", type=Path)
    return parser


def _load_mapping(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    gold_audit: GoldAuditTrail | None = None
    join_audit: JoinAuditTrail | None = None
    try:
        start = parse_time(args.probe_start)
        end = parse_time(args.probe_end)
        events = load_replay(args.replay)
        raw_jira = json.loads(args.raw_jira.read_text(encoding="utf-8")) if args.raw_jira else None
        raw_resolutions = _load_mapping(args.gold_resolutions)
        resolutions = {
            key: value.get("resolution") if isinstance(value, dict) else value
            for key, value in raw_resolutions.items()
        }
        raw_join_expected = _load_mapping(args.join_audit)
        join_expected = {
            key: [str(item) for item in value]
            for key, value in raw_join_expected.items()
            if isinstance(value, list)
        }
        raw_a0_answers = _load_mapping(args.a0_audit)
        a0_answers = {key: str(value) for key, value in raw_a0_answers.items()}
        gold_audit = GoldAuditTrail(
            source="raw-jira-export" if raw_jira is not None else "normalised-smoke-adapter",
            resolutions=resolutions,
        )
        join_audit = JoinAuditTrail(expected=join_expected)
        probes = generate_probe_set(
            events,
            per_family=args.per_family,
            seed=args.seed,
            probe_start=start,
            probe_end=end,
            multisource_code_count=args.multisource_code_count,
            evidentiary=args.evidentiary,
            minimum_entities=args.minimum_entities,
            raw_jira=raw_jira,
            gold_audit=gold_audit,
            join_audit=join_audit,
            a0_answers=a0_answers or None,
        )
        output, sidecar, digest = write_probe_set(probes, args.out)
        write_gold_report(gold_audit, args.gold_report or Path(f"{args.out}.gold.json"))
        write_join_report(join_audit, args.join_report or Path(f"{args.out}.joins.json"))
    except ValueError as exc:
        if gold_audit is not None:
            write_gold_report(
                gold_audit,
                args.gold_report or Path(f"{args.out}.gold.json"),
            )
        if join_audit is not None:
            write_join_report(
                join_audit,
                args.join_report or Path(f"{args.out}.joins.json"),
            )
        _parser().error(str(exc))
    print(f"wrote {len(probes)} probes to {output}")
    print(f"sha256 {digest} ({sidecar})")
    if not args.evidentiary:
        print("profile non-evidentiary-smoke; 300-probe/150-entity profile supported-not-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
