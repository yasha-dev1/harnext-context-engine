"""S2 curated-flat builder store from docs/evaluation-spec.md §7 E3."""

from __future__ import annotations

import shutil

from harnext_eval.stores.base import StoreHandle
from harnext_eval.stores.build_s3 import run_builder_harness
from harnext_eval.stores.fake_curator import curate_events
from harnext_eval.stores.layouts import record_input_metadata, runtime_for, unseen_events
from harnext_eval.types import EvalEvent

_FLAT_CLAUDE = """# Flat Context Filesystem

Maintain one folder per event subject under `entities/<type>/<slug>/`.
Each entity may contain `OVERVIEW.md`, `facts.md`, and `timeline.md`.
Do not create or maintain `INDEX.md` or `topics/`; this condition intentionally
isolates curation without global organisation. Keep provenance by event ID.
"""

_FLAT_SCHEMA = """# Flat Store Conventions

- `entities/<type>/<slug>/` is the only durable context layout.
- Each entity can have `OVERVIEW.md`, `facts.md`, and `timeline.md`.
- There is deliberately no global INDEX and no topics hierarchy.
"""

_FLAT_PROMPT = """You maintain a flat per-entity context filesystem.
Read `CLAUDE.md` and incorporate every supplied event into its entity folder.
Keep current state in OVERVIEW.md, dated atomic facts in facts.md, and append-only
provenance in timeline.md. Never create INDEX.md or any topics directory. Cite
event IDs, do not invent facts, and edit only inside the working directory.
"""


def _apply_flat_seed(store: StoreHandle) -> None:
    store.write("CLAUDE.md", _FLAT_CLAUDE)
    store.write("_meta/schema.md", _FLAT_SCHEMA)
    topics = store.worktree / "topics"
    if topics.exists():
        shutil.rmtree(topics)


def fold_s2(store: StoreHandle, events: list[EvalEvent], lane: str) -> None:
    """Run the harness with the no-INDEX/no-topics seed and prompt variant."""

    accepted = unseen_events(store, events)
    if not accepted:
        return
    _apply_flat_seed(store)
    runtime = runtime_for(store)
    record_input_metadata(store, accepted)
    if runtime.harness == "fake":
        curate_events(store, accepted, lane, global_organisation=False)
    else:
        run_builder_harness(store, accepted, lane, system_prompt=_FLAT_PROMPT)
    index = store.worktree / "INDEX.md"
    if index.exists():
        index.unlink()
    topics = store.worktree / "topics"
    if topics.exists():
        shutil.rmtree(topics)


fold = fold_s2
