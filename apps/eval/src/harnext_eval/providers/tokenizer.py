"""Deterministic approximate token counting for docs/evaluation-spec.md D8."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def count_tokens(text: str) -> int:
    """Approximate provider tokenisation without loading a model tokenizer."""

    return len(_TOKEN_RE.findall(text))
