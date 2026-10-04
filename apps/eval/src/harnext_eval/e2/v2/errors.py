"""Errors that separate a model's failure from the infrastructure's."""

from __future__ import annotations

import re


class PauseRun(RuntimeError):  # noqa: N818
    """Quota, credits, auth, rate limit or infrastructure failure: stop cleanly, record
    nothing for the unit in flight, retry it on resume."""


INFRA = re.compile(
    r"rate.?limit|\b429\b|quota|insufficient|credit|billing|usage limit|exceeded your|"
    r"session limit|weekly limit|hit your [\w ]*limit|limit (·|-)? ?resets|api_error|"
    r"unauthori[sz]ed|\b401\b|\b403\b|authenticat|login|token (has )?expired|"
    r"overloaded|\b5\d\d\b|server error|connection (reset|refused|error)|timed? ?out connecting|"
    r"network|dns|tls|econn|enotfound|stream disconnected|"
    # Appended from the shipped CLIs' own strings (Codex 0.154.0, Claude Code 2.1.278).
    r"could not be refreshed|sign in again|out of credits|at capacity|try again (later|in )|"
    r"invalid api key|/login|exceeded retry limit|service unavailable|bad gateway",
    re.IGNORECASE,
)


def is_infra(text: str | None) -> bool:
    """True when an agent/CLI error message describes the provider or the machine,
    not the model's attempt at the task."""
    return bool(text and INFRA.search(text))


RESET_AT = re.compile(r"resets?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", re.IGNORECASE)
RESET_IN = re.compile(
    r"(?:resets?|retry|try again)\s+in\s+(\d+)\s*(second|minute|hour|min|sec|s|m|h)", re.IGNORECASE
)


def reset_delay(message: str | None, default: int = 900, now=None) -> int:
    """Seconds to wait before retrying after a usage-limit refusal. Parses "resets 2:20pm"
    and "retry in 5 minutes"; otherwise `default`. Never less than a minute or more than 6 h."""
    import datetime as dt

    now = now or dt.datetime.now()
    seconds = default
    if message and (m := RESET_IN.search(message)):
        unit = m.group(2).lower()[0]
        seconds = int(m.group(1)) * {"s": 1, "m": 60, "h": 3600}[unit]
    elif message and (m := RESET_AT.search(message)):
        hour, minute, ampm = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower()
        if ampm == "pm" and hour < 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
        if "utc" in message.lower() and now.tzinfo is None:  # "resets 9:20pm (UTC)"
            now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
        target = now.replace(hour=hour % 24, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += dt.timedelta(days=1)
        seconds = int((target - now).total_seconds()) + 60  # a minute past the reset
    return max(60, min(seconds, 6 * 3600))
