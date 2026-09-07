"""S3 curated-and-indexed builder store from docs/evaluation-spec.md §7 E3."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import cast

from harnext_builder.build_runner import RUNNER_CMD
from harnext_builder.event_fs import event_files
from harnext_builder.harness.base import ConversationTranscript, HarnessRequest
from harnext_builder.prompts import SYSTEM_PROMPT, render_instruction
from harnext_builder.work_item import WorkItem
from harnext_shared import CloudEvent

from harnext_eval.stores.base import StoreHandle
from harnext_eval.types import EvalEvent


def _head(store: StoreHandle) -> str:
    return subprocess.run(
        ["git", "-C", str(store.worktree), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _instruction(store: StoreHandle, events: list[EvalEvent], lane: str) -> str:
    effective_lane = "fast" if lane == "fast" and len(events) == 1 else "batch"
    work_item = WorkItem(
        org_id=store.org_id,
        lane=effective_lane,
        dedupe_key="eval:" + ":".join(event.id for event in events),
        subjects=sorted({event.subject for event in events}),
        events=list(events),
    )
    return render_instruction(work_item)


def run_builder_harness(
    store: StoreHandle,
    events: list[EvalEvent],
    lane: str,
    *,
    system_prompt: str = SYSTEM_PROMPT,
) -> ConversationTranscript:
    """Run the production harness protocol locally against the git worktree."""

    from harnext_eval.stores.layouts import append_usage, runtime_for

    runtime = runtime_for(store)
    request = HarnessRequest(
        harness=runtime.harness,
        working_dir=".",
        instruction=_instruction(store, events, lane),
        system_prompt=system_prompt,
        event_files=event_files(cast(list[CloudEvent], events)),
        model=runtime.model,
        seed=runtime.seed,
        max_turns=runtime.max_turns,
        timeout_s=runtime.timeout_s,
    )
    previous = _head(store)
    transcript: ConversationTranscript | None = None
    try:
        with tempfile.TemporaryDirectory(prefix=f"eval-build-{store.org_id}-") as temp:
            request_path = Path(temp) / "request.json"
            result_path = Path(temp) / "result.json"
            request_path.write_text(request.model_dump_json(), encoding="utf-8")
            result = store.backend.run_build(
                store.org_id,
                RUNNER_CMD,
                {"REQUEST_PATH": str(request_path), "RESULT_PATH": str(result_path)},
                runtime.timeout_s + 30,
            )
            if result_path.exists():
                try:
                    transcript = ConversationTranscript.model_validate_json(
                        result_path.read_text(encoding="utf-8")
                    )
                except ValueError:
                    transcript = None
            if transcript is None:
                detail = result.stderr.strip() or result.stdout.strip() or "no transcript"
                transcript = ConversationTranscript(
                    harness=runtime.harness,
                    model=runtime.model,
                    stop_reason="error",
                    error=f"builder runner failed ({result.returncode}): {detail[:1000]}",
                )
    except Exception as exc:
        transcript = ConversationTranscript(
            harness=runtime.harness,
            model=runtime.model,
            stop_reason="error",
            error=f"{type(exc).__name__}: {exc}",
        )

    append_usage(store, transcript, events, lane)
    if not transcript.ok:
        store.backend.restore(store.org_id, previous)
        raise RuntimeError(transcript.error or "builder harness failed")
    return transcript


def fold_s3(store: StoreHandle, events: list[EvalEvent], lane: str) -> None:
    """Curate a fold with the standard harnext seed and builder prompt."""

    from harnext_eval.stores.fake_curator import curate_events
    from harnext_eval.stores.layouts import (
        record_input_metadata,
        runtime_for,
        unseen_events,
    )

    accepted = unseen_events(store, events)
    if not accepted:
        return
    record_input_metadata(store, accepted)
    if runtime_for(store).harness == "fake":
        curate_events(store, accepted, lane, global_organisation=True)
    else:
        run_builder_harness(store, accepted, lane)


fold = fold_s3
