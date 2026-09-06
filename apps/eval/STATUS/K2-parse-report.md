# K2 full JIRA/mail parse report

Parsed on 2026-09-06.

Counts below are deduplicated by event ID; sorted raw paths use last occurrence.
Inconsistencies count distinct issues per field across all snapshots.
Roster matches use a current snapshot retrospectively; Apache IDs were not published on the source page.

## jira

Files: 167; parsed occurrences: 235,407; unique events: 235,407; duplicates removed: 0.

Months: 90 (2019-01 through 2026-06); missing months: none; missing completion markers: none.
Repeated IDs with differing payloads: 0 (last sorted occurrence retained).

| Event type | Count | Roster matches |
|---|---:|---:|
| org.apache.jira.issue.comment | 35,484 | 16,123 |
| org.apache.jira.issue.created | 12,670 | 6,328 |
| org.apache.jira.issue.transition | 187,253 | 51,919 |

Distinct inconsistent issues: 1,472.
- fixVersion: 1,472

Failed files/months: none

Output: `parsed/jira.jsonl`; SHA-256: `d940615fdc4a8f0feed3d2e5243a7366c9d8257d6023f040dc3cb8c314dd5424`.

## mail

Files: 90; parsed occurrences: 63,598; unique events: 63,572; duplicates removed: 26.

Months: 90 (2019-01 through 2026-06); missing months: none; missing completion markers: none.
Repeated IDs with differing payloads: 13 (last sorted occurrence retained).

| Event type | Count | Roster matches |
|---|---:|---:|
| org.apache.mail.message | 63,572 | 16,603 |

Distinct inconsistent issues: 0.

Failed files/months: none

Output: `parsed/mail.jsonl`; SHA-256: `defb77937a0fd68ca42917dc8ca8eb2f07f74dee822d0910d7ac144c5ad5df24`.

