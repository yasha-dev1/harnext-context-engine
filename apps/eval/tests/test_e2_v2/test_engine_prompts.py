"""The engine's model-facing texts (prompts, Jev instructions and option texts, tool
descriptions) share no word 3-gram with the benchmark's question templates and public prompts
(w2026 and smoke), using the raw store's own neutrality check (stop words, ids, dates removed)."""

import json

from harnext_eval.e2.v2.engine import branch, memory, prompts, rawtools
from harnext_eval.e2.v2.stores.base import ROOT

from .test_raw_store import benchmark_phrases, trigrams


def texts() -> list[str]:
    found = []

    def walk(value):
        if isinstance(value, str):
            found.append(value)
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(key)
                walk(item)
        elif isinstance(value, list | tuple):
            for item in value:
                walk(item)

    walk(list(prompts.texts(budget=True).values()))
    for tool in (rawtools.FIND, rawtools.READ, rawtools.READ_THREAD, rawtools.COMPARE_TIMES, branch.DONE, memory.SEARCH, memory.READ_FILE):  # fmt: skip
        walk(tool["function"]["description"])
        walk(tool["function"]["parameters"])
    return found


def phrases() -> set[str]:
    found = benchmark_phrases()
    smoke = ROOT / "benchmarks/kafka-500-v2-smoke/tasks.public.jsonl"
    for line in smoke.read_text().splitlines():
        task = json.loads(line)
        found |= trigrams(task["prompt"]) | trigrams(task["family"].replace("_", " "))
    return found


def test_model_facing_texts_do_not_echo_the_benchmark():
    known = phrases()
    assert len(known) > 1000
    shared = {g: t[:120] for t in texts() for g in trigrams(t) & known}
    for phrase, text in sorted(shared.items()):
        print(f"{phrase!r} <- {text}")
    assert not shared, sorted(shared)
