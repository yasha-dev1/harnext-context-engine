"""S1 no-LLM templated store from docs/evaluation-spec.md §7 E3."""

from __future__ import annotations

from harnext_eval.stores.base import StoreHandle
from harnext_eval.stores.layouts import record_input_metadata, unseen_events
from harnext_eval.stores.templated import apply_event, rebuild_index
from harnext_eval.types import EvalEvent


def fold_s1(store: StoreHandle, events: list[EvalEvent], lane: str) -> None:
    """Apply the deterministic entity projection and commit metadata for one fold."""

    accepted = unseen_events(store, events)
    for event in accepted:
        apply_event(store, event, lane)
    record_input_metadata(store, accepted)
    rebuild_index(store)


fold = fold_s1

