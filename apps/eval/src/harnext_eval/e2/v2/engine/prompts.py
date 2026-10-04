"""Every prompt of the engine (hashed into the config hash). Generic: no project, benchmark or
family names. Jev instructions reference state fields in backticks; the question id is never
seen by the model, so each instruction carries its whole meaning."""

from __future__ import annotations

# ---- Jev: routing and aggregation -------------------------------------------------------------
ROUTE = "Would records of the kind described in `sources.{name}` contain information needed to answer `question`?"  # fmt: skip
MEMORY_SOURCE = (
    "Curated notes: markdown files an assistant wrote while reading the project's history "
    "(one file per issue, pull request, discussion thread or topic, plus an index). Summaries, "
    "decisions and rationale in prose; may be incomplete; exact field values and timestamps "
    "are not guaranteed."
)
SUPPORT = (
    "Is every fact that `finding` states (names, values, dates, decisions, reasons) shown in "
    "`records`? Answer yes as well when the finding only says that nothing relevant was found."
)
FACTUAL = "Does `findings.{name}` state concrete facts or values, rather than only that nothing relevant was found or that it could not be determined?"  # fmt: skip
TIMING = "Does answering `question` require relating the recorded instants of two events to each other?"  # fmt: skip
EVENT = "`question` relates the instants of two events. Which record is the {which} of those two events, in the order `question` names them?"  # fmt: skip
UNIT = "In which unit does `question` ask for the time between the two events? Choose seconds if it asks for no amount."  # fmt: skip
UNITS = {"milliseconds": "Milliseconds.", "seconds": "Seconds.", "minutes": "Minutes.", "hours": "Hours.", "days": "Days.", "weeks": "Weeks."}  # fmt: skip
ROUNDING = "How does the wording of `question` ask for that amount to be rounded?"
ROUNDINGS = {"down": "Rounded down to whole units (e.g. complete units only).", "nearest": "Rounded to the nearest whole unit.",  # fmt: skip
             "up": "Rounded up to whole units.", "exact": "Not rounded (or no rounding stated)."}  # fmt: skip
VERIFY = "Judge `report` as an answer to `question`, using only `findings` as the truth."
VERDICTS = {
    "supported": "Every fact, name and value the report states is stated in the findings.",
    "unsupported": "The report states at least one fact, name or value that the findings do not state, or it contradicts them.",  # fmt: skip
    "declined": "The report only says that the answer was not found or could not be determined.",
}
FIX_VALUES = (
    "These values in your report occur neither in the records nor in the findings: {values}. "
    "Rewrite the report: copy every value exactly from the findings, and where the findings do "
    "not give a value, say that it could not be determined. Do not guess."
)
FIX_UNSUPPORTED = (
    "A check found statements in your report that the findings and their records may not "
    "support. Keep every fact the findings or their records state, copied exactly; remove or "
    "mark as undetermined only the statements they do not support."
)
FINDING = "Does `findings.{name}` contain information that answers `question` or bears directly on it?"  # fmt: skip

# ---- Jev: slot filling ------------------------------------------------------------------------
TABLE = (
    "Which kind of record holds what `need` asks for? `values` are literal values that the "
    "wanted records (or records linked to them) must match."
)
SHAPE = "What kind of result does `need` ask for?"
SHAPES = {
    "lookup": "Specific records identified by the given values (an id, key, number, name or exact title).",  # fmt: skip
    "list": "Every record that matches the values, e.g. all items of some kind that belong to an entity.",  # fmt: skip
    "count": "A number: how many records match, or one count per group (per person, file, value, ...).",  # fmt: skip
    "timeline": "The records in chronological order: a history of changes, or the state at a moment (the latest change before it).",  # fmt: skip
    "search": "Records whose free text mentions some words or a phrase (a topic, a discussion, a keyword).",  # fmt: skip
    "values": "The distinct values of one column among the matching records.",
    "latest": "The single most recent matching record (e.g. the state at a moment = the latest change before it).",  # fmt: skip
    "earliest": "The single oldest matching record (e.g. the first message, the first change).",
}
LINK = (
    "Besides matching `values` directly, must the wanted records be restricted through a "
    "different kind of record they are connected to (for example: items that belong to "
    "entities which only that other kind of record describes)? Choose that other kind of "
    "record, or none."
)
NO_LINK = "No other kind of record is needed."
BIND = (
    "The literal value `values.{name}` was given for `need`. Which column of the records holds "
    "this value in the role `need` gives it? Read the column meanings: who performed an event "
    "is a different column from whose item it is (its author, reporter or owner). Choose none "
    "if the value is not a filter."
)
NOT_A_FILTER = "The value is context only, not a filter on any column."
DATE_OP = "How does the date or time `values.{name}` restrict the records wanted by `need`?"
DATE_OPS = {
    "before": "Strictly before the start of this date or instant.",
    "through": "Up to and including this date or instant (through the end of the named day, month or second).",  # fmt: skip
    "from": "From the start of this date or instant on, inclusive.",
    "after": "Strictly after this date or instant (after the end of the named day, month or second).",  # fmt: skip
    "on": "Within this day, month or instant.",
}
VALUE_OP = "How does `values.{name}` restrict the records wanted by `need`?"
TEXT_OPS = {
    "eq": "The records must have this value.",
    "ne": "The records must NOT have this value (exclude records that have it).",
}
NUMBER_OPS = {
    "eq": "The records must have exactly this number.",
    "from": "The number must be at least this value.",
    "after": "The number must be greater than this value.",
    "through": "The number must be at most this value.",
    "before": "The number must be less than this value.",
    "ne": "The records must NOT have this number.",
}
COMPARES = "Does `need` compare two fields of the same record with each other (for example: exclude records where two people or values are the same)?"  # fmt: skip
COMPARE = "Which two fields does `need` compare, and must they be equal or differ?"
NO_COMPARE = "No comparison of two fields."
EMPTIES = "Does `need` explicitly ask for records in which some field has no value at all, or explicitly only records in which that field has a value?"  # fmt: skip
EMPTY = "Which field must be empty or set for the records wanted by `need`?"
NO_EMPTY = "No field must be empty or set."
DISTINCT = "`need` asks for a count. What is counted: every matching record, or the distinct values of one field (e.g. distinct items or people)?"  # fmt: skip
ALL_ROWS = "Every matching record."
VALUES_OF = "`need` asks for the set of values of one field. Which field?"
OUTPUT = "Is the column described in `columns.{name}` needed to answer `need`?"
GROUP = "`need` asks for a count. One count per distinct value of which column? Choose none for one total."  # fmt: skip
NO_GROUP = "One total count, no grouping."
ORDER = "By which column should the records for `need` be ordered?"
DIRECTION = "In which direction should the records for `need` be ordered?"
DIRECTIONS = {
    "asc": "Oldest / lowest first (e.g. how something began, or a full history in order).",
    "desc": "Newest / highest first (e.g. the latest state or the most recent items).",
}
VALUES_MODE = "When `need` gives several values for the same field, must a record match any of them or all of them?"  # fmt: skip
VALUES_MODES = {"any": "Any of them (alternatives).", "all": "All of them at once."}
ROW = "Does the record `rows.{name}` contain information that answers `need` or helps directly with it?"  # fmt: skip
HIT = "Is the line `hits.{name}` from a notes file worth reading in full to answer `need`?"

# ---- chat: branch agents ----------------------------------------------------------------------
RAW_BRANCH = """You research one question in the recorded {label} history of a software project. \
Every record is from before {boundary}. Available kinds of record: {tables}.

Tools:
- find(need, values): say in ONE full sentence which records you need and what you want to \
know from them, and pass as values the literal data the records must contain: keys, numbers, \
logins, names, exact titles, dates as YYYY-MM-DD, a state, a changed field or a relation as \
written in the records. Values are data, never column names. The tool builds the query and \
returns rows with their event ids. One need per call.
- read(event_id): the complete record of one event.
- read_thread(event_id, need): every record of the thread or item the event belongs to (a \
discussion, the comments on an item), relevant ones in full: use it to learn what a \
discussion said or decided.
- compare_times(first_event_id, second_event_id, unit, rounding): the order of two events and \
the time between them, computed from the records.
- done(finding, evidence): finish.

Values are matched exactly (case-sensitive) and spellings differ between kinds of record; the \
same person may appear under different names in different systems (a display name in one, a \
login in another). Some discussions and items began before the recorded window, so their \
first record may be missing; lists of referenced ids may come from titles or subjects only, \
so also search the texts.

Answer exactly what the question asks and keep every qualifier it states (which event, which \
state, which role, which period); never widen or narrow it. Let the tools compute counts, \
distinct values, the latest or earliest record, and the order of and time between two events \
(compare_times, with the unit and rounding the question's wording asks for); do not compute \
these yourself. Pass dates and instants exactly as written in the question.

Work step by step: first turn names or titles into identifiers, then fetch exactly what the \
question asks. Rows show long texts cut to snippets: read(event_id) the records that matter \
(e.g. the messages of a discussion) before you conclude what they say. If a find returns nothing or the wrong records, restate the need differently \
(another kind of record, other or fewer values). Do not repeat an identical call.

When the question asks for every item or change, list every row that matches, none omitted. \
Other sources are researched in parallel: report every fact you found that bears on the \
question even if it answers only part of it. Finish with done: a compact statement of the \
facts that bear on the question, in full \
sentences that name their subject (e.g. the key or number and what the value is), each with \
the event ids it comes from. Copy identifiers, names, titles and timestamps exactly as they appear \
in the rows, never reformatted or rounded. Give the rows' facts, not guesses; if nothing \
relevant was found, say so plainly."""
SQL_TOOL = (
    "Fallback: run ONE read-only ClickHouse SELECT (or WITH) over the tables below when find "
    "cannot express the need. Only these tables; no table functions, SETTINGS or comments. "
    "Compare times with plain literals such as '2024-01-31 00:00:00'. Returns rows."
)
SQL_UNLOCKED = "The sql tool is now available as a fallback. Tables and columns:\n{schema}"
MEMORY_BRANCH = """You research one question in a folder of curated markdown notes about a \
software project (one file per issue, pull request, discussion thread or topic, plus an index).

Tools:
- search(need, keywords): keyword search over all notes (case-insensitive; each keyword is a \
word, identifier or short phrase). Returns file paths with matching lines.
- read_file(path, from_line): read a notes file from a line on.
- done(finding, evidence): finish.

Search with the identifiers and distinctive words of the question, read the most promising \
files, then call done with a compact statement of the facts that bear on the question, each \
with the file path it comes from. Copy identifiers, names and timestamps exactly; if nothing \
relevant was found, say so plainly."""
FINISH_NOW = (
    "The research budget is used up. Call no more tools: state now what you found that bears "
    "on the question, with the event ids or notes file names it comes from, and what is still unknown."
)
UNREADABLE = "Your last reply could not be read as a tool call. Reply again with exactly one valid tool call (JSON arguments), or finish with done."  # fmt: skip
TEXT_REPLY = "Call one of the tools; finish with done(finding, evidence)."

# ---- chat: final report -----------------------------------------------------------------------
REPORT = """You write the final research report for a question about a software project's \
recorded history, from findings that research agents gathered in the records.

Write:
1. The direct answer first, in ONE sentence that names the instant or condition it refers to \
(the moment, period or state the question asks about).
2. The supporting facts in time order, each with its timestamp and its citation. When a fact \
could be read as contradicting the answer (an earlier or later change), say how it relates to \
the instant asked about (before it or after it).
3. What could not be determined, if anything. No speculation.

Cite only record identifiers (event ids, notes file names) exactly as the findings give them, in \
square brackets, one identifier per bracket; never put other text inside the brackets. If the findings found nothing relevant, say so plainly: an honest \
"not found" is better than a guess. Keep timestamps with all their digits.

Each finding comes with the records it cites (or the last records its agent retrieved): use \
them to check the finding and to copy exact values; when the finding is vague but the records \
answer the question, answer from the records.

Rules: use only the findings and their records. The findings come from different sources and each may lack what \
another has: combine them (one may give one date, another the other). Copy exact values (identifiers, names, titles, timestamps, \
numbers) verbatim from the findings, never reformatted or rounded. If findings conflict, say \
so and prefer the one backed by records. If the findings do not answer the question, say so."""


# ---- engine v2: tool-call budget feedback (used, and hashed, only when budget_feedback is on) --
BUDGET_SYSTEM = (
    "You have {budget} tool calls (done does not count); every tool result ends with how many you "
    "have used. Plan them: when they are used up, or the time runs out, further calls are not "
    "run and you must answer with done from what you already have."
)
BUDGET_LINE = "Tool call budget: {used}/{budget}"
BUDGET_EXHAUSTED = (
    "Not executed: the tool call budget is used up. Answer now from the results you already "
    "have: call done with your finding and the evidence ids."
)
BUDGET_TIME = (
    "Not executed: the time for this research is almost up. Answer now from the results you "
    "already have: call done with your finding and the evidence ids."
)

BUDGET_TIME_FINAL = (
    "The time for this research is up: no more tool calls. Answer now from the results you "
    "already have, in plain text: your finding, with the evidence ids it comes from."
)


def texts(budget: bool = False) -> dict[str, object]:
    """Every prompt text; the v2 budget texts only when budget feedback is on (so the v1
    configuration hashes exactly as before)."""
    return {k: v for k, v in globals().items() if k.isupper() and (budget or not k.startswith("BUDGET_"))}  # fmt: skip
