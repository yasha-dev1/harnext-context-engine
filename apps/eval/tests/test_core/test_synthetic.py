"""Synthetic corpus tests for docs/evaluation-spec.md §3.2."""

from harnext_eval.corpus.synthetic import events_hash, generate_synthetic_events


def test_same_seed_produces_same_hash() -> None:
    first = generate_synthetic_events(seed=19)
    second = generate_synthetic_events(seed=19)

    assert len(first) == 2_000
    assert events_hash(first) == events_hash(second)
    assert first[0].time < first[-1].time


def test_stream_has_derivable_cross_source_gold() -> None:
    events = generate_synthetic_events(seed=2, event_count=200)
    sources = {event.source.split(":", 1)[0] for event in events}

    assert sources == {"jira", "mail", "github"}
    assert any(event.data and event.data.get("changelog") for event in events)
    assert any(event.data and event.data.get("in_reply_to") for event in events)
    assert any(event.data and event.data.get("changed_files") for event in events)
    assert all(event.baseline_keys for event in events)
