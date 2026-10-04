"""Deterministic checks of findings and reports against what a research call showed to a model.

A checkable value is any token that is not ordinary prose: it contains a digit, an internal
punctuation character (- _ . / : # @), or upper case beyond a capitalised first letter; plus
quoted or backticked strings and proper names (two or more capitalised words in a row). Tokens
made only of letters are compared case-insensitively, all others exactly (after normalising
typographic hyphens and spaces to ASCII, so a value copied from a row still matches).

- `values_in(text)`: the checkable values of a text.
- `unverified(text, corpus, known)`: the values found neither in the corpus (rows shown to a
  model, plus findings) nor in `known` (the question and the instructions).
- `citations(text, ids)`: only identifiers shown to a model in this call stay inside brackets,
  one per bracket; returns the cleaned text and what was removed. `cited(text, ids)` lists them.
- `strip(text, values)`: drops the lines stating values that stayed unverified and says so.
"""

from __future__ import annotations

import re

DASHES = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]")
SPACES = re.compile("[\u00a0\u2007\u2009\u202f]")
ZERO = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
BRACKETS = str.maketrans({"【": "[", "】": "]"})
TOKEN = re.compile(r"[\w@#][\w\-./:#@]*[\w]|\w")
INNER = re.compile(r"\w[-_./:#@]\w")
NAME = re.compile(r"\b[A-Z][a-z]+(?:[ ][A-Z][a-z]+)+\b")
QUOTED = re.compile(r"`([^`\n]{2,100})`|\"([^\"\n]{2,100})\"")
CITATION = re.compile(r"\[([^\[\]\n]{1,400})\]")
FENCE = re.compile(r"```.*?```", re.DOTALL)


def normalise(text: str) -> str:
    return ZERO.sub("", SPACES.sub(" ", DASHES.sub("-", str(text or "")))).translate(BRACKETS)


def checkable(token: str) -> bool:
    return any(ch.isdigit() for ch in token) or bool(INNER.search(token)) or any(ch.isupper() for ch in token[1:])  # fmt: skip


def values_in(text: str) -> list[str]:
    text = normalise(text)
    found = [t for t in TOKEN.findall(text) if checkable(t)]
    found += [a or b for a, b in QUOTED.findall(text)]
    found += NAME.findall(text)
    return list(dict.fromkeys(v.strip() for v in found if v.strip()))


def contained(value: str, corpus: str, folded: str) -> bool:
    return value.casefold() in folded if value.replace(" ", "").isalpha() else value in corpus


def unverified(text: str, corpus: str, known: str = "") -> list[str]:
    corpus, known = normalise(corpus), normalise(known)
    folded, known_folded = corpus.casefold(), known.casefold()
    return [v for v in values_in(text) if not contained(v, corpus, folded) and not contained(v, known, known_folded)]  # fmt: skip


def outside_fences(text: str) -> list[tuple[int, int]]:
    spans, last = [], 0
    for fence in FENCE.finditer(text):
        spans.append((last, fence.start()))
        last = fence.end()
    return [*spans, (last, len(text))]


def citations(text: str, ids) -> tuple[str, list[str]]:
    """Keep only citation tokens that are identifiers shown to a model in this call (`ids`),
    one per bracket; remove everything else inside brackets (JSON arrays and code blocks are
    left alone). Returns the cleaned text and the removed tokens."""
    text, ids = normalise(text), set(ids)
    removed: list[str] = []

    def keep(match: re.Match) -> str:
        body = match.group(1)
        if body.lstrip().startswith(("{", '"', "'")) or re.fullmatch(r"[\d.,\s-]+", body):
            return match.group(0)  # a JSON array or a list of numbers, not a citation
        tokens = [t.strip("`'\" ").removeprefix("event_id:").strip() for t in re.split(r"[,;\s|]+", body)]  # fmt: skip
        good = [t for t in dict.fromkeys(tokens) if t in ids]
        removed.extend(t for t in tokens if t and t not in ids)
        return "".join(f"[{t}]" for t in good)

    while True:  # repeat: removing an inner bracket may expose an outer non-citation bracket
        parts, last = [], 0
        for start, end in outside_fences(text):
            parts.append(text[last:start])
            parts.append(CITATION.sub(keep, text[start:end]))
            last = end
        parts.append(text[last:])
        if "".join(parts) == text:
            return text, removed
        text = "".join(parts)


def cited(text: str, ids) -> list[str]:
    """The identifiers a text cites in brackets, in order."""
    return list(dict.fromkeys(t for body in CITATION.findall(normalise(text)) for t in re.split(r"[,;\s|]+", body) if t in set(ids)))  # fmt: skip


def strip(text: str, values: list[str]) -> str:
    """Drop the lines that state unverified values; say what could not be determined."""
    kept = [line for line in text.splitlines() if not any(v in normalise(line) for v in values)]
    note = "Could not be determined from the records: the values " + ", ".join(values) + " (removed: not found in the retrieved records)."  # fmt: skip
    return "\n".join([*kept, "", note]).strip()
