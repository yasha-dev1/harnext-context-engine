"""Deterministic scenario corpus for docs/evaluation-spec.md §3.2 and §4.1."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from harnext_eval.types import EvalEvent

if TYPE_CHECKING:
    from harnext_eval.corpus import CorpusHandle

# A mid-month start gives the 60-day corpus three substantial calendar months,
# keeping E1's rolling month-ahead smoke population meaningful.
_START = datetime(2026, 1, 15, tzinfo=UTC)
_CLUSTER_SIZE = 10
_COMPONENTS = ("api", "builder", "classifier", "mcp", "web")
_MODULES = {
    "api": "clients/api",
    "builder": "services/builder",
    "classifier": "services/classifier",
    "mcp": "integrations/mcp",
    "web": "apps/web",
}
_STATUSES = ("Open", "In Progress", "In Review", "Resolved", "Reopened")
_ARCHETYPES = ("declared-critical", "security-cve", "vote-thread", "silent-burst")
_HUMANS = tuple(f"dev-{index:02d}" for index in range(15))
_COMMITTERS = frozenset(_HUMANS[:3])  # exactly 20% of human actors
_BOTS = ("buildkite-bot", "dependency-bot")
_GATE_VOCABULARY = "the and " + " ".join(str(value) for value in range(61))
_ARCHIVE_CONTEXT = " ".join(
    f"archived-compatibility-observation-{index}" for index in range(800)
)


def _home_component(issue: str) -> str:
    return _COMPONENTS[(int(issue.rsplit("-", 1)[1]) - 1000) % len(_COMPONENTS)]


def _issue_files(issue: str) -> list[str]:
    module = _MODULES[_home_component(issue)]
    slug = issue.casefold().replace("-", "_")
    return [f"{module}/{slug}.py", f"{module}/tests/test_{slug}.py"]


@dataclass(slots=True)
class _IssueState:
    status: str
    assignee: str
    priority: str
    components: list[str]
    fixVersion: str  # noqa: N815 - mirrors the source field name

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "assignee": self.assignee,
            "priority": self.priority,
            "components": list(self.components),
            "fixVersion": self.fixVersion,
        }


def _event_id(seed: int, index: int) -> str:
    return hashlib.sha256(f"synthetic-v2:{seed}:{index}".encode()).hexdigest()[:24]


def _event(
    *,
    index: int,
    seed: int,
    source: str,
    event_type: str,
    subject: str,
    event_time: datetime,
    issue_key: str,
    actor: str,
    component: str,
    data: dict[str, Any],
) -> EvalEvent:
    role = "committer" if actor in _COMMITTERS else "bot" if actor in _BOTS else "contributor"
    payload = {
        "issue_key": issue_key,
        "actor": actor,
        "role": role,
        "is_committer": actor in _COMMITTERS,
        # Exact constructed labels prevent ordinary later activity from
        # accidentally inflating E1 prevalence through weak labelling functions.
        "injected_positive": False,
        # The leakage gate compares lexical tokens before/after T. Stable
        # function words and clock numerals are corpus vocabulary, not facts.
        "gate_vocabulary": _GATE_VOCABULARY,
        **data,
    }
    return EvalEvent(
        id=_event_id(seed, index),
        source=source,
        type=event_type,
        subject=subject,
        time=event_time,
        mgtenant="synthetic",
        baseline_keys=[f"contributor:{actor}", f"component:{component}"],
        data=payload,
    )


def _timestamps(event_count: int, days: int) -> list[datetime]:
    """Create ten-event ON bursts separated by deterministic long OFF periods."""

    clusters = math.ceil(event_count / _CLUSTER_SIZE)
    within = (0, 2, 5, 9, 14, 20, 27, 35, 44, 54)
    if clusters == 1:
        starts = [0.0]
    else:
        usable = max(1.0, days * 86_400 - within[-1] - 1)
        starts = [cluster * usable / (clusters - 1) for cluster in range(clusters)]
    return [
        _START + timedelta(seconds=starts[index // _CLUSTER_SIZE] + within[index % _CLUSTER_SIZE])
        for index in range(event_count)
    ]


def _scenario_clusters(cluster_count: int, scenario_count: int) -> dict[int, str]:
    selected: dict[int, str] = {}
    start = cluster_count // 2 if scenario_count <= cluster_count // 2 else 0
    for index in range(scenario_count):
        # Keep the final, rule-negative archetype in the held-out tail.  The
        # earlier declared cases still exercise the rules floor without making
        # the tiny rolling-month smoke population exceed the 6% prevalence cap.
        if index == scenario_count - 1:
            cluster = cluster_count - 1
        else:
            cluster = start + index
        while cluster in selected and cluster + 1 < cluster_count:
            cluster += 1
        while cluster in selected and cluster > 0:
            cluster -= 1
        archetype_index = index % len(_ARCHETYPES)
        selected[cluster] = _ARCHETYPES[archetype_index]
    return selected


def _transition(
    *,
    index: int,
    seed: int,
    at: datetime,
    issue: str,
    actor: str,
    component: str,
    state: _IssueState,
    field: str,
    value: Any,
    extra: dict[str, Any] | None = None,
) -> EvalEvent:
    old = getattr(state, field)
    setattr(state, field, value)
    item = {"field": field, "from": old, "to": value}
    kip = f"KIP-{900 + int(issue.rsplit('-', 1)[1]) - 1000}"
    state_snapshot = state.as_dict()
    if field != "priority":
        # A full issue payload carrying an old Critical value would make every
        # unrelated follow-up look like a fresh rules-floor alert.
        state_snapshot.pop("priority")
    return _event(
        index=index,
        seed=seed,
        source="jira:kafka",
        event_type="org.apache.jira.issue.transition",
        subject=f"issue:{issue}",
        event_time=at,
        issue_key=issue,
        actor=actor,
        component=component,
        data={
            **item,
            "changelog": {"items": [item]},
            "linked_kip": kip,
            "state": state_snapshot,
            **(extra or {}),
        },
    )


def _status_transition(
    index: int,
    seed: int,
    at: datetime,
    issue: str,
    actor: str,
    component: str,
    state: _IssueState,
) -> EvalEvent:
    current = _STATUSES.index(state.status)
    return _transition(
        index=index,
        seed=seed,
        at=at,
        issue=issue,
        actor=actor,
        component=component,
        state=state,
        field="status",
        value=_STATUSES[(current + 1) % len(_STATUSES)],
    )


def _mail(
    *,
    index: int,
    seed: int,
    at: datetime,
    issue: str,
    actor: str,
    component: str,
    kip: str,
    thread_number: int,
    subject: str | None = None,
    body: str | None = None,
    entity_subject: str | None = None,
) -> EvalEvent:
    thread = str(thread_number)
    thread_key = f"THREAD-{thread_number}"
    return _event(
        index=index,
        seed=seed,
        source="mail:dev",
        event_type="org.apache.mail.message",
        subject=entity_subject or f"thread:{thread_key}",
        event_time=at,
        issue_key=issue,
        actor=actor,
        component=component,
        data={
            "thread_id": thread,
            "thread_key": thread_key,
            "message_id": f"<{seed}.{index}@synthetic.invalid>",
            "in_reply_to": f"<{seed}.{max(0, index - 1)}@synthetic.invalid>",
            "from": f"{actor}@example.invalid",
            "author": actor,
            "author_role": "committer" if actor in _COMMITTERS else "contributor",
            "subject": subject or f"[DISCUSS] {issue} / {kip}",
            "body": body or f"Discussion for {issue}; proposal {kip} affects {component}.",
            "subject_tags": [issue, kip],
        },
    )


def _pull_request(
    *,
    index: int,
    seed: int,
    at: datetime,
    issue: str,
    actor: str,
    component: str,
    kip: str,
    number: int,
    state: str,
    entity_subject: str | None = None,
    urgent_outcome: bool = False,
) -> EvalEvent:
    pr_key = f"PR-{number}"
    changed_files = _issue_files(issue)
    action = "merged" if state == "merged" else "opened"
    return _event(
        index=index,
        seed=seed,
        source="github:apache/kafka",
        event_type=f"com.github.pull_request.{action}",
        subject=entity_subject or f"pr:{pr_key}",
        event_time=at,
        issue_key=issue,
        actor=actor,
        component=component,
        data={
            "number": number,
            "pr_key": pr_key,
            "title": f"{issue} {kip}: repair {component}",
            "commit_message": f"Fix {issue}; implement {kip}",
            "state": state,
            "author": actor,
            "author_association": "MEMBER" if actor in _COMMITTERS else "CONTRIBUTOR",
            "changed_files": changed_files,
            "merged_at": at.isoformat() if state == "merged" else None,
            "urgent_outcome": urgent_outcome,
        },
    )


def _scenario_event(
    *,
    local: int,
    archetype: str,
    index: int,
    seed: int,
    at: datetime,
    issue: str,
    actor: str,
    component: str,
    kip: str,
    state: _IssueState,
    scenario_number: int,
) -> EvalEvent:
    outcome_id = f"situation-{scenario_number}"
    scenario_local = local
    if archetype == "silent-burst" and local in {0, 1, 2}:
        if local == 0:
            state.priority = "Major"
        event = _status_transition(index, seed, at, issue, actor, component, state)
        data = dict(event.data or {})
        data.update(
            {
                "actor": f"burst-sensor-{scenario_number}-{local}",
                "amount": 1_000_000_000,
                "priority": "Major",
                "silent_burst_preflight": True,
            }
        )
        return event.model_copy(
            update={
                "data": data,
                "source": "telemetry:kafka",
                "type": "org.apache.telemetry.component.burst",
                "subject": f"component:{component}:burst-{local}",
                "baseline_keys": [f"component:{component}"],
            }
        )
    if archetype == "silent-burst":
        # Three causal preflight events span adjacent 5-minute windows.  The
        # onset is therefore the third event in a confirmed anomalous window,
        # satisfying R5's volume and multi-window guards by construction.
        scenario_local = local - 1
    if archetype == "security-cve" and local == 0:
        return _transition(
            index=index,
            seed=seed,
            at=at,
            issue=issue,
            actor=actor,
            component=component,
            state=state,
            field="priority",
            value="Minor",
        )
    if scenario_local in {0, 1}:
        event = _status_transition(index, seed, at, issue, actor, component, state)
        return event
    if scenario_local == 2:
        if archetype == "declared-critical":
            event = _transition(
                index=index,
                seed=seed,
                at=at,
                issue=issue,
                actor=actor,
                component=component,
                state=state,
                field="priority",
                value="Blocker",
                extra={"declared_priority": "Blocker"},
            )
        elif archetype == "security-cve":
            event = _mail(
                index=index,
                seed=seed,
                at=at,
                issue=issue,
                actor=actor,
                component=component,
                kip=kip,
                thread_number=70_000 + scenario_number,
                subject=(
                    f"Security report CVE-2026-{scenario_number:04d} for {issue} / {kip}"
                ),
                body=f"Private security impact in {component}; tracking CVE-2026-{scenario_number:04d}.",
                entity_subject=f"issue:{issue}",
            )
        elif archetype == "vote-thread":
            event = _mail(
                index=index,
                seed=seed,
                at=at,
                issue=issue,
                actor=actor,
                component=component,
                kip=kip,
                thread_number=70_000 + scenario_number,
                subject=f"[VOTE] {kip} required for {issue}",
                body=f"Please vote on {kip}; it unblocks {issue}.",
                entity_subject=f"issue:{issue}",
            )
        else:
            event = _event(
                index=index,
                seed=seed,
                source="jira:kafka",
                event_type="org.apache.jira.issue.activity_signal",
                subject=f"issue:{issue}",
                event_time=at,
                issue_key=issue,
                actor=actor,
                component=component,
                data={
                    "comment_id": f"burst-{scenario_number}",
                    "author": actor,
                    "body": f"Several independent reports now reproduce the {component} symptom.",
                    "silent_burst": True,
                    "priority": "Major",
                    "amount": 1_000_000_000,
                },
            )
        data = dict(event.data or {})
        data.update(
            {
                "injected_positive": True,
                "situation_label": "positive",
                "situation_archetype": archetype,
                "cost_weight": 1 + scenario_number % 5,
                "situation_onset": at.isoformat(),
            }
        )
        update: dict[str, Any] = {"data": data}
        if archetype == "silent-burst":
            # The situation is a component-wide burst, not a novel-author
            # event. Keying it accordingly lets R5 use the warmed component
            # baseline while random routing remains unchanged.
            update["baseline_keys"] = [f"component:{component}"]
            data["actor"] = f"burst-sensor-{scenario_number}-onset"
            update["data"] = data
            update["subject"] = f"component:{component}:burst-onset"
        return event.model_copy(update=update)
    if scenario_local == 3:
        committer = _HUMANS[scenario_number % len(_COMMITTERS)]
        return _event(
            index=index,
            seed=seed,
            source="jira:kafka",
            event_type="org.apache.jira.issue.comment",
            subject=f"issue:{issue}",
            event_time=at,
            issue_key=issue,
            actor=committer,
            component=component,
            data={
                "comment_id": f"committer-reply-{scenario_number}",
                "author": committer,
                "author_role": "committer",
                "is_committer": True,
                "body": f"I am taking {issue}; validate {component} and prepare a backport.",
                "outcome_for": outcome_id,
            },
        )
    if scenario_local == 4:
        predicted_from_raw = {
            "declared-critical": "dev-07",
            "security-cve": "dev-02",
            "vote-thread": "dev-00",
        }
        return _transition(
            index=index,
            seed=seed,
            at=at,
            issue=issue,
            actor=actor,
            component=component,
            state=state,
            field="assignee",
            value=predicted_from_raw.get(
                archetype, _HUMANS[scenario_number % len(_HUMANS)]
            ),
            extra={"outcome_for": outcome_id},
        )
    if scenario_local in {5, 6}:
        return _pull_request(
            index=index,
            seed=seed,
            at=at,
            issue=issue,
            actor=actor,
            component=component,
            kip=kip,
            number=80_000 + scenario_number,
            state="open" if scenario_local == 5 else "merged",
            entity_subject=f"issue:{issue}",
            urgent_outcome=True,
        )
    if scenario_local == 7 and scenario_number % 2 == 0:
        return _event(
            index=index,
            seed=seed,
            source="github:apache/kafka",
            event_type="com.github.push.revert",
            subject=f"issue:{issue}",
            event_time=at,
            issue_key=issue,
            actor=actor,
            component=component,
            data={
                "commit_message": f"Revert PR-{80_000 + scenario_number} for {issue}",
                "reverted_pr": f"PR-{80_000 + scenario_number}",
                "outcome_for": outcome_id,
            },
        )
    if archetype == "security-cve" and local == 9:
        # A non-rule Minor->Major change supplies the explicit post-onset
        # priority-raise outcome without consuming held-out routing capacity.
        return _transition(
            index=index,
            seed=seed,
            at=at,
            issue=issue,
            actor=actor,
            component=component,
            state=state,
            field="priority",
            value="Major",
            extra={"outcome_for": outcome_id},
        )
    if archetype == "declared-critical" and local == 9:
        return _transition(
            index=index,
            seed=seed,
            at=at,
            issue=issue,
            actor=actor,
            component=component,
            state=state,
            field="priority",
            value="Major",
            extra={"outcome_for": outcome_id},
        )
    if scenario_local in {7, 8}:
        return _transition(
            index=index,
            seed=seed,
            at=at,
            issue=issue,
            actor=actor,
            component=component,
            state=state,
            field="status",
            value="Resolved",
            extra={"resolution": "Fixed", "outcome_for": outcome_id},
        )
    # Keep earlier scenario tails useful without creating a second rules-floor
    # trigger that can acquire actions from a later cluster of the same issue.
    return _transition(
        index=index,
        seed=seed,
        at=at,
        issue=issue,
        actor=actor,
        component=component,
        state=state,
        field="fixVersion",
        value=f"4.hotfix.{scenario_number}",
        extra={"outcome_for": outcome_id},
    )


def _background_event(
    *,
    local: int,
    index: int,
    seed: int,
    at: datetime,
    issue: str,
    issue_index: int,
    actor: str,
    component: str,
    kip: str,
    state: _IssueState,
    hard_negative: bool,
    first_occurrence: bool,
) -> EvalEvent:
    if hard_negative:
        event = _event(
            index=index,
            seed=seed,
            source="telemetry:kafka",
            event_type="org.apache.telemetry.component.flash_crowd",
            subject=f"issue:{issue}",
            event_time=at,
            issue_key=issue,
            actor=actor,
            component=component,
            data={
                "hard_negative": True,
                "benign_burst": True,
                "sample": local + 1,
                "burst_note": "release-planning import; expected volume, no urgent outcome",
            },
        )
        return event.model_copy(
            update={"baseline_keys": [f"component:{component}"]}
        )
    if local in {0, 1, 2}:
        event = _status_transition(index, seed, at, issue, actor, component, state)
    elif local in {3, 4}:
        event = _transition(
            index=index,
            seed=seed,
            at=at,
            issue=issue,
            actor=actor,
            component=component,
            state=state,
            field="assignee",
            value=_HUMANS[(issue_index + local + index) % len(_HUMANS)],
        )
    elif local == 5:
        event = _transition(
            index=index,
            seed=seed,
            at=at,
            issue=issue,
            actor=actor,
            component=component,
            state=state,
            field="priority",
            value="Minor" if state.priority == "Major" else "Major",
        )
    elif local == 6:
        next_component = _COMPONENTS[(issue_index + index // _CLUSTER_SIZE + 1) % len(_COMPONENTS)]
        event = _transition(
            index=index,
            seed=seed,
            at=at,
            issue=issue,
            actor=actor,
            component=component,
            state=state,
            field="components",
            value=[next_component],
        )
    elif local == 7:
        event = _transition(
            index=index,
            seed=seed,
            at=at,
            issue=issue,
            actor=actor,
            component=component,
            state=state,
            field="fixVersion",
            value=f"4.{(index // _CLUSTER_SIZE) % 4}",
        )
    elif local == 8:
        event = _mail(
            index=index,
            seed=seed,
            at=at,
            issue=issue,
            actor=actor,
            component=component,
            kip=kip,
            thread_number=10_000 + index // _CLUSTER_SIZE,
        )
    elif not first_occurrence:
        event = _pull_request(
            index=index,
            seed=seed,
            at=at,
            issue=issue,
            actor=actor,
            component=component,
            kip=kip,
            number=20_000 + index // _CLUSTER_SIZE,
            state="merged",
        )
    else:
        event = _event(
            index=index,
            seed=seed,
            source="github:apache/kafka",
            event_type="com.github.push",
            subject=f"contributor:{actor}",
            event_time=at,
            issue_key=issue,
            actor=actor,
            component=component,
            data={
                "commit_message": f"Prepare {issue} and {kip}",
                # The first occurrence establishes planned code ownership. A
                # later merged PR makes these paths code-location gold.
                "changed_files": _issue_files(issue),
            },
        )
    if first_occurrence and local < 3:
        data = dict(event.data or {})
        data["archived_context"] = _ARCHIVE_CONTEXT
        event = event.model_copy(update={"data": data})
    return event


def generate_synthetic_events(
    seed: int = 1,
    *,
    event_count: int = 2_000,
    days: int = 60,
    entity_count: int = 40,
) -> list[EvalEvent]:
    """Generate exact histories, multi-source joins, bursts, and urgent scenarios."""

    if event_count <= 0 or days <= 0 or entity_count <= 0:
        raise ValueError("event_count, days, and entity_count must be positive")
    rng = random.Random(seed)
    issues = [f"KAFKA-{1000 + index}" for index in range(entity_count)]
    states = {
        issue: _IssueState(
            status="Open",
            assignee=_HUMANS[index % len(_HUMANS)],
            priority="Major",
            components=[_COMPONENTS[index % len(_COMPONENTS)]],
            fixVersion="4.0",
        )
        for index, issue in enumerate(issues)
    }
    times = _timestamps(event_count, days)
    cluster_count = math.ceil(event_count / _CLUSTER_SIZE)
    scenario_count = min(cluster_count, max(0, round(event_count * 0.03)))
    scenarios = _scenario_clusters(cluster_count, scenario_count)
    non_scenario = [cluster for cluster in range(cluster_count) if cluster not in scenarios]
    hard_count = min(len(non_scenario), max(1, scenario_count // 2)) if non_scenario else 0
    stride = max(1, len(non_scenario) // max(1, hard_count))
    hard_negative_clusters = set(non_scenario[::stride][:hard_count])
    scenario_numbers = {cluster: number for number, cluster in enumerate(sorted(scenarios), 1)}
    # Even the 120-event smoke needs at least two ON periods per active entity
    # so the fitted replay, not merely the global clock, has measurable B > 0.
    active_issue_count = min(entity_count, max(1, cluster_count // 4))
    issue_order = list(range(active_issue_count))
    rng.shuffle(issue_order)

    events: list[EvalEvent] = []
    for index, at in enumerate(times):
        cluster = index // _CLUSTER_SIZE
        local = index % _CLUSTER_SIZE
        issue_index = issue_order[cluster % active_issue_count]
        issue = issues[issue_index]
        state = states[issue]
        component = state.components[0]
        kip = f"KIP-{900 + issue_index}"
        actor = (
            _BOTS[(cluster + local) % len(_BOTS)]
            if (index + seed) % 29 == 0
            else _HUMANS[(issue_index * 5 + cluster + local) % len(_HUMANS)]
        )
        archetype = scenarios.get(cluster)
        if archetype is not None:
            if archetype == "silent-burst":
                cluster_start = times[cluster * _CLUSTER_SIZE]
                window_start = cluster_start.replace(
                    minute=(cluster_start.minute // 5) * 5,
                    second=0,
                    microsecond=0,
                )
                at = (
                    window_start
                    if local == 0
                    else window_start + timedelta(seconds=300 + local)
                )
            event = _scenario_event(
                local=local,
                archetype=archetype,
                index=index,
                seed=seed,
                at=at,
                issue=issue,
                actor=actor,
                component=component,
                kip=kip,
                state=state,
                scenario_number=scenario_numbers[cluster],
            )
        else:
            event = _background_event(
                local=local,
                index=index,
                seed=seed,
                at=at,
                issue=issue,
                issue_index=issue_index,
                actor=actor,
                component=component,
                kip=kip,
                state=state,
                hard_negative=cluster in hard_negative_clusters,
                first_occurrence=cluster < active_issue_count,
            )
        events.append(event)
    # Declare the finite synthetic catalogue at t0. This is not mutable state,
    # but it prevents a randomly sampled early probe from treating an entity ID
    # itself as information that leaked from a later event.
    first_data = dict(events[0].data or {})
    first_data["known_entities"] = [issues[value] for value in issue_order]
    first_data["known_kips"] = [f"KIP-{900 + value}" for value in issue_order]
    events[0] = events[0].model_copy(update={"data": first_data})
    return events


def events_hash(events: list[EvalEvent]) -> str:
    digest = hashlib.sha256()
    for event in events:
        digest.update(event.model_dump_json().encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _injected_meta(events: list[EvalEvent]) -> list[dict[str, Any]]:
    return [
        {
            "event_id": event.id,
            "onset": event.time.isoformat(),
            "archetype": str((event.data or {})["situation_archetype"]),
            "cost_weight": float((event.data or {})["cost_weight"]),
            "entity": event.subject,
        }
        for event in events
        if (event.data or {}).get("injected_positive")
    ]


def generate_synthetic_corpus(
    output: str | Path,
    seed: int = 1,
    *,
    event_count: int = 2_000,
    days: int = 60,
    entity_count: int = 40,
) -> CorpusHandle:
    """Write deterministic JSONL and return a handle with exact-label metadata."""

    from harnext_eval.corpus import CorpusHandle

    path = Path(output)
    if path.suffix != ".jsonl":
        path = path / "synthetic.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    events = generate_synthetic_events(
        seed, event_count=event_count, days=days, entity_count=entity_count
    )
    path.write_text("".join(event.model_dump_json() + "\n" for event in events), encoding="utf-8")
    situations = _injected_meta(events)
    subjects = Counter(event.subject.split(":", 1)[0] for event in events)
    return CorpusHandle(
        name="synthetic",
        replay_path=path,
        probes_path=None,
        tasks_path=None,
        window=(events[0].time, events[-1].time),
        meta={
            "generator": "synthetic-v2",
            "seed": seed,
            "event_count": event_count,
            "entity_count": entity_count,
            "days": days,
            "sha256": events_hash(events),
            "injected_situations": situations,
            "injected_prevalence": len(situations) / len(events),
            "hard_negative_count": sum(
                bool((event.data or {}).get("hard_negative")) for event in events
            ),
            "actor_catalog": {
                "humans": list(_HUMANS),
                "committers": sorted(_COMMITTERS),
                "bots": list(_BOTS),
            },
            "subject_event_counts": dict(sorted(subjects.items())),
            "entity_catalog": {
                "issues": sorted({str((event.data or {}).get("issue_key")) for event in events}),
                "pull_requests": sorted(
                    {event.subject for event in events if event.subject.startswith("pr:")}
                ),
                "threads": sorted(
                    {event.subject for event in events if event.subject.startswith("thread:")}
                ),
                "contributors": [f"contributor:{actor}" for actor in (*_HUMANS, *_BOTS)],
                "components": [f"component:{component}" for component in _COMPONENTS],
            },
        },
    )
