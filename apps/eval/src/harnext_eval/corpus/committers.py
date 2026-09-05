"""Frozen Kafka committer attribution for docs/evaluation-spec.md §4.1.

The roster is a fetch-date snapshot, not evidence of historical membership.
Matching is exact after case/space normalization; no substring name matches.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from harnext_eval.types import EvalEvent

DEFAULT_ROSTER = Path(__file__).resolve().parents[3] / "configs/kafka-committers.yaml"


def _normalized(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


class CommitterRoster:
    """Match source identities against the frozen roster, entirely offline."""

    def __init__(self, people: Iterable[dict[str, Any]]) -> None:
        self.names: set[str] = set()
        self.ids: set[str] = set()
        self.logins: set[str] = set()
        for person in people:
            self.names.add(_normalized(person.get("name")))
            self.ids.update(_normalized(item) for item in person.get("apache_ids", []))
            self.logins.update(_normalized(item) for item in person.get("github", []))
        self.names.discard("")
        self.ids.discard("")
        self.logins.discard("")

    def matches(self, source: str, data: dict[str, Any]) -> bool:
        if source.startswith("github:"):
            return (
                _normalized(data.get("author_association")) in {"member", "owner"}
                or _normalized(data.get("actor_login")) in self.logins
                or _normalized(data.get("author_login")) in self.logins
            )
        names = (data.get(key) for key in ("actor_name", "author_name", "creator_name"))
        accounts = (data.get(key) for key in ("actor_account_id", "author_account_id", "creator_account_id"))
        email = _normalized(data.get("author_email"))
        local = email.split("@", 1)[0] if "@" in email else ""
        return (
            any(_normalized(value) in self.names for value in names)
            or any(_normalized(value) in self.ids for value in accounts)
            or bool(local and local in self.ids)
        )

    def stamp(self, event: EvalEvent) -> EvalEvent:
        data = dict(event.data or {})
        data["is_committer"] = data.get("is_committer") is True or self.matches(event.source, data)
        return event.model_copy(update={"data": data})


@lru_cache(maxsize=4)
def load_roster(path: Path = DEFAULT_ROSTER) -> CommitterRoster:
    """Load once per process; importing the module never fetches anything."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return CommitterRoster(payload["committers"])


def stamp_events(
    events: Iterable[EvalEvent], roster: CommitterRoster | None = None,
) -> tuple[list[EvalEvent], dict[str, int]]:
    """Stamp Jira/mail/GitHub events and return matched event counts by source."""
    roster = roster or load_roster()
    stamped: list[EvalEvent] = []
    counts: Counter[str] = Counter()
    for event in events:
        kind = event.source.split(":", 1)[0]
        if kind in {"jira", "mail", "github"}:
            event = roster.stamp(event)
            counts[f"{kind}.total"] += 1
            counts[f"{kind}.matched"] += int(bool((event.data or {}).get("is_committer")))
        stamped.append(event)
    return stamped, dict(counts)
