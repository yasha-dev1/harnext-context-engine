"""Operational driver: fetch KAFKA JIRA (changelog+comments) and dev@kafka mail, raw to disk, resumable.

Run:  UV_NO_SYNC=1 uv run python apps/eval/scripts/fetch_kafka_jira_mail.py  (writes under apps/eval/out/corpus/kafka/)
Outputs under apps/eval/out/corpus/kafka/raw/{jira,mail}/ ; parsed EvalEvent JSONL under parsed/.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1] / "out" / "corpus" / "kafka"
RAW_JIRA = ROOT / "raw" / "jira"
RAW_MAIL = ROOT / "raw" / "mail"
PARSED = ROOT / "parsed"
for d in (RAW_JIRA, RAW_MAIL, PARSED):
    d.mkdir(parents=True, exist_ok=True)

START = date(2019, 1, 1)
END = date(2026, 7, 1)  # exclusive
JIRA = "https://issues.apache.org/jira"
FIELDS = "summary,description,created,creator,reporter,assignee,status,priority,fixVersions,components,labels,comment"


def months(start: date, end: date):
    y, m = start.year, start.month
    while date(y, m, 1) < end:
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m == 13:
            y, m = y + 1, 1


def get(url: str, *, tries: int = 6, timeout: float = 120) -> bytes:
    delay = 5.0
    for attempt in range(tries):
        try:
            with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=timeout) as r:
                return r.read()
        except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
            code = getattr(exc, "code", None)
            if code in (400, 404):
                raise
            print(f"  retry {attempt+1}/{tries} after {exc} ({url[:90]}...)", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 120)
    raise RuntimeError(f"gave up on {url}")


def fetch_jira():
    """One JQL window per month of `updated`; every page saved as raw JSON. Skips complete months."""
    for month in months(START, END):
        marker = RAW_JIRA / f"{month}.done"
        if marker.exists():
            continue
        y, m = map(int, month.split("-"))
        nxt = f"{y + (m == 12):04d}-{(m % 12) + 1:02d}-01"
        jql = f'project = KAFKA AND updated >= "{month}-01" AND updated < "{nxt}" ORDER BY key ASC'
        start_at, page = 0, 0
        total = None
        while True:
            q = urlencode({"jql": jql, "startAt": start_at, "maxResults": 100, "expand": "changelog", "fields": FIELDS})
            body = get(f"{JIRA}/rest/api/2/search?{q}")
            payload = json.loads(body)
            issues = payload.get("issues", [])
            (RAW_JIRA / f"{month}-p{page:03d}.json").write_bytes(body)
            total = payload.get("total", total)
            if not issues:
                break
            start_at += len(issues)
            page += 1
            if total is not None and start_at >= total:
                break
            time.sleep(0.3)
        marker.write_text(f"{total}\n")
        print(f"jira {month}: {total} issues, {page + 1} pages", flush=True)


def fetch_mail():
    for month in months(START, END):
        mbox = RAW_MAIL / f"dev-{month}.mbox"
        if mbox.exists() and mbox.stat().st_size > 0:
            continue
        q = urlencode({"list": "dev@kafka.apache.org", "date": month})
        body = get(f"https://lists.apache.org/api/mbox.lua?{q}", timeout=300)
        mbox.write_bytes(body)
        print(f"mail {month}: {len(body)/1e6:.1f} MB", flush=True)
        time.sleep(0.5)


def parse():
    """Sweep every cached page/month, deduplicate, and report before publishing."""
    from collections import Counter, defaultdict

    from harnext_eval.corpus.build_replay import write_replay
    from harnext_eval.corpus.jira import parse_search_page
    from harnext_eval.corpus.pony_mail import parse_mbox

    report = ["# K2 full JIRA/mail parse report", "", f"Parsed on {date.today()}.", "",
              "Counts below are deduplicated by event ID; sorted raw paths use last occurrence.",
              "Inconsistencies count distinct issues per field across all snapshots.",
              "Roster matches use a current snapshot retrospectively; Apache IDs were not published on the source page.", ""]
    failures = []
    for source, paths in (("jira", sorted(RAW_JIRA.glob("*.json"))),
                          ("mail", sorted(RAW_MAIL.glob("dev-*.mbox")))):
        seen = {}
        inconsistencies = defaultdict(set)
        raw_count = 0
        conflicts = 0
        failed = []
        expected = set(months(START, END))
        present = {path.name[:7] if source == "jira" else path.name[4:11] for path in paths}
        missing = sorted(expected - present)
        markers = {path.stem for path in RAW_JIRA.glob("*.done")} if source == "jira" else expected
        for path in paths:
            try:
                events = (parse_search_page(path.read_bytes()) if source == "jira" else
                          parse_mbox(path, list_name="dev", domain="kafka.apache.org",
                                     month=path.stem.removeprefix("dev-")))
                for event in events:
                    raw_count += 1
                    if event.id in seen and seen[event.id] != event:
                        conflicts += 1
                    seen[event.id] = event
                    for field in event.data.get("state_inconsistencies", []):
                        inconsistencies[field].add(event.subject)
            except Exception as exc:  # noqa: BLE001 - finish sweep, fail publication
                failed.append(f"{path.name}: {exc}")
        counts = Counter(event.type for event in seen.values())
        matches = Counter(event.type for event in seen.values() if event.data.get("is_committer"))
        report.extend([f"## {source}", "", f"Files: {len(paths)}; parsed occurrences: {raw_count:,}; "
                       f"unique events: {len(seen):,}; duplicates removed: {raw_count - len(seen):,}.", "",
                       f"Months: {len(present)} ({min(present)} through {max(present)}); "
                       f"missing months: {missing or 'none'}; "
                       f"missing completion markers: {sorted(expected - markers) or 'none'}.",
                       f"Repeated IDs with differing payloads: {conflicts} (last sorted occurrence retained).", "",
                       "| Event type | Count | Roster matches |", "|---|---:|---:|"])
        report.extend(f"| {kind} | {count:,} | {matches[kind]:,} |" for kind, count in sorted(counts.items()))
        affected = set().union(*inconsistencies.values()) if inconsistencies else set()
        report.extend(["", f"Distinct inconsistent issues: {len(affected):,}."])
        report.extend(f"- {field}: {len(issues):,}" for field, issues in sorted(inconsistencies.items()))
        report.extend(["", "Failed files/months: " + ("; ".join(failed) if failed else "none"), ""])
        failures.extend(failed)
        if not failed:
            art = write_replay(sorted(seen.values(), key=lambda event: (event.time, event.id)),
                               PARSED / f"{source}.jsonl")
            report.extend([f"Output: `{art.path.relative_to(ROOT)}`; SHA-256: `{art.sha256}`.", ""])
            print(f"parsed {source}: {art.event_count} events -> {art.path}", flush=True)
        del seen
    report_path = Path(__file__).resolve().parents[1] / "STATUS" / "K2-parse-report.md"
    report_path.write_text("\n".join(report) + "\n")
    if failures:
        raise RuntimeError(f"{len(failures)} files failed; see {report_path}")


if __name__ == "__main__":
    what = sys.argv[1:] or ["jira", "mail", "parse"]
    if "jira" in what:
        fetch_jira()
    if "mail" in what:
        fetch_mail()
    if "parse" in what:
        parse()
    print("DONE", flush=True)
