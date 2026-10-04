"""Slot filling: Jev turns (need, values) into a query plan in two decisions calls.

  call 1: table (choice over the branch's tables), shape (choice), link (choice: none or a
          table the catalog joins to one of the branch's tables, in any source group), and
          whether the need compares two fields or needs a field empty/set (noul)
  call 2: per value its column (choice over the columns of the table and the linked table, or
          "none"; column meanings in the options separate who acted from whose item it is) and
          its operator (dates: after / before / on; numbers: = / >= / < / !=; text: = / !=);
          per column whether it is needed in the output (noul, row shapes only); the group /
          distinct column of a count, the value column of a values list, the order column and
          direction of a row list, and the compared pair / empty field when call 1 said so

Jev never writes text: every slot is a choice among catalog identifiers or fixed options, and
the literals are the agent's own values. A choice below the confidence threshold is reported
as an uncertain slot.
"""

from __future__ import annotations

import re

from harnext_eval.e2.v2.engine import prompts
from harnext_eval.e2.v2.engine.plan import (  # fmt: skip
    TIME,
    Compare,
    Filter,
    Link,
    Plan,
    inner,
    is_number,
    is_text,
    is_time,
)
from harnext_eval.e2.v2.engine.session import Session

NOT_BOUND = {"raw"}
NOT_SHOWN = {"raw", "text", "event_id", "time", "event_type", "entities"}
MAX_VALUES = 8
MAX_OPTIONS = 40  # options of one generated choice (column pairs, empty checks)
STRICT = 0.8  # optional slots (field comparison, empty/set check) need a clear signal
QUOTED = re.compile(r"'([^'\n]{1,200})'|\"([^\"\n]{1,200})\"")


def is_date(value: str) -> bool:
    return bool(re.match(r"\d{4}-\d\d", value) and TIME.fullmatch(value.strip()))


def confidence(answer: dict) -> float:
    if "confidence" in answer:
        return float(answer["confidence"])
    probabilities = answer.get("probabilities") or {}
    return max(probabilities.values(), default=1.0)


def picked(answer: dict, options) -> str:
    """The chosen option; an unknown choice falls back to the most probable known option."""
    found = answer.get("choice")
    if found in options:
        return found
    probabilities = {k: v for k, v in (answer.get("probabilities") or {}).items() if k in options}
    return max(probabilities, key=probabilities.get) if probabilities else next(iter(options))


def alternatives(answer: dict, n: int = 3) -> dict:
    probabilities = answer.get("probabilities") or {}
    return dict(sorted(probabilities.items(), key=lambda kv: -kv[1])[:n])


def choice_q(instructions: str, criteria: dict) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def clean(values, need: str = "") -> list[str]:
    """Distinct non-empty values; literals quoted in the need are values too."""
    if not isinstance(values, list):  # schemas are not enforced: a bare string or number
        values = [] if values is None else [values]
    values = [*values, *(a or b for a, b in QUOTED.findall(str(need or "")))]
    found = []
    for value in values or []:
        text = " ".join(str(value).split()).strip().strip("\"'`")
        if text and text not in found:
            found.append(text)
    return found[:MAX_VALUES]


def uncertain_slot(slot: str, answer: dict, chosen: str, threshold: float) -> list[dict]:
    if (c := confidence(answer)) < threshold:
        return [{"slot": slot, "choice": chosen, "confidence": round(c, 3), "alternatives": alternatives(answer)}]  # fmt: skip
    return []


def bindable(cat, table: str, prefix: str) -> dict[str, str]:
    """Meanings of the bind options of a table: every column, plus "either of" a column and its
    siblings (the meanings go into the state once; the options only point at them)."""
    found = {f"{prefix}{c}": cat.describe(table, c, examples=1) for c in cat.types[table] if c not in NOT_BOUND}  # fmt: skip
    for c in cat.types[table]:
        group = sorted([c, *cat.siblings(table, c)], key=lambda x: (len(x), x))  # base first
        if len(group) > 1 and group[0] == c:
            found[prefix + "|".join(group)] = f"Either of the columns {', '.join(group)} (a value found in any of them): " + " / ".join(cat.describe(table, x, examples=1) for x in group)  # fmt: skip
    return found


async def translate(session: Session, need: str, values: list[str], scope: list[str]) -> tuple[Plan, dict]:  # fmt: skip
    """(plan, info): info holds the uncertain slots and the Jev answers for the trace. When the
    chosen table leaves a value unused, the runner-up table is tried and the plan that uses
    more of the given values wins."""
    cat, config = session.catalog, session.config
    vals = {f"v{i}": v for i, v in enumerate(values, 1)}
    links = sorted({o for (t, o) in cat.edges if t in scope})
    first = {"shape": choice_q(prompts.SHAPE, prompts.SHAPES),
             "compares": {"type": "noul", "instructions": prompts.COMPARES},
             "empties": {"type": "noul", "instructions": prompts.EMPTIES}}  # fmt: skip
    if len(scope) > 1:
        first["table"] = choice_q(prompts.TABLE, {t: cat.summary(t) for t in scope})
    if links:
        first["link"] = choice_q(prompts.LINK, {"none": prompts.NO_LINK, **{t: cat.tables[t]["about"] for t in links}})  # fmt: skip
    one = await session.decide({"need": need, "values": vals}, first, "plan:table")
    table = picked(one["table"], scope) if "table" in first else scope[0]
    shape = picked(one["shape"], prompts.SHAPES)
    link = picked(one["link"], ["none", *links]) if links else "none"
    table_note = uncertain_slot("table", one["table"], table, config.confidence) if "table" in first else []  # fmt: skip
    table_note += uncertain_slot("shape", one["shape"], shape, config.confidence)
    plan, uncertain, unused = await fill(session, need, vals, table, shape, link, one)
    switched = None
    if unused and "table" in first:
        ranked = sorted((one["table"].get("probabilities") or {}).items(), key=lambda kv: -kv[1])
        runner = next((t for t, p in ranked if t in scope and t != table and p >= 0.02), None)
        if runner:
            other = await fill(session, need, vals, runner, shape, link, one)
            if len(other[2]) < len(unused):
                switched = {"from": table, "to": runner, "unused_before": unused, "unused_after": other[2]}  # fmt: skip
                plan, uncertain, unused = other
                table_note = []  # the values decided the table
    trace = {"table": plan.table, "shape": shape, "link": plan.link.table if plan.link else None,
             "uncertain": table_note + uncertain, "unused_values": unused, "switched": switched,
             "confidence": {"table": confidence(one["table"]) if "table" in one else 1.0,
                            "shape": confidence(one["shape"])}}  # fmt: skip
    return plan, trace


def state_key(option: str) -> str:
    """A state key for a column option: no dots or bars (Jev reads `a.b` as a nested path)."""
    return re.sub(r"\W", "__", option)


def pointer(option: str) -> str:
    return f"The column described in `columns.{state_key(option)}`."


def numeric(value: str) -> bool:
    return bool(re.fullmatch(r"#?-?\d+(?:\.\d+)?", value.strip()))


def short(cat, table: str, column: str) -> bool:
    """A scalar text column of short values (names, states, keys), not prose."""
    kind = cat.types[table][column]
    examples = cat.info[table][column].get("examples", [])
    return is_text(kind) and not kind.startswith("Array(") and column not in NOT_SHOWN and not any(str(e).endswith("…") or len(str(e)) >= 60 for e in examples)  # fmt: skip


def pairs(cat, table: str, link: str | None, edge) -> dict[str, str]:
    """Column comparisons Jev may pick: two short columns of the table, or a table column
    against a column of the linked row (scalar join keys only)."""
    mine = [c for c in cat.types[table] if short(cat, table, c) or c == "actor"]
    found = {}
    for i, a in enumerate(mine):
        for b in mine[i + 1 :]:
            if cat.types[table][a] == cat.types[table][b]:
                found[f"{a}!={b}"] = f"{a} differs from {b}"
                found[f"{a}=={b}"] = f"{a} equals {b}"
    if link and edge and not any(k.startswith("Array(") for k in (cat.types[table][edge[0]], cat.types[link][edge[1]])):  # fmt: skip
        for a in mine:
            for b in (c for c in cat.types[link] if short(cat, link, c) or c == "actor"):
                if inner(cat.types[table][a]) == inner(cat.types[link][b]):
                    found[f"{a}!={link}.{b}"] = f"{a} differs from {b} of the linked {link} record"
    return dict(list(found.items())[:MAX_OPTIONS])


async def fill(session: Session, need: str, vals: dict, table: str, shape: str, link: str, one: dict) -> tuple[Plan, list[dict], list[str]]:  # fmt: skip
    """Call 2 for one table: (plan, uncertain slots, values not used as filters)."""
    cat, config = session.catalog, session.config
    uncertain: list[dict] = []
    edge = cat.edges.get((table, link)) if link not in ("none", table) else None
    link = link if edge else None
    types = cat.types[table]
    meanings = bindable(cat, table, "")
    if link:
        meanings |= bindable(cat, link, f"{link}.")
    columns = {c: pointer(c) for c in meanings}

    def kind(option: str) -> str:
        t, _, c = option.split("|")[0].rpartition(".")
        return cat.types[t or table][c]

    second: dict[str, dict] = {}
    for key, value in vals.items():
        date = is_date(value)
        options = {c: d for c, d in columns.items() if is_time(kind(c))} if date else columns
        if not options:
            continue
        second[f"bind_{key}"] = choice_q(prompts.BIND.format(name=key), {**options, "none": prompts.NOT_A_FILTER})  # fmt: skip
        ops = (
            prompts.DATE_OPS if date else prompts.NUMBER_OPS if numeric(value) else prompts.TEXT_OPS
        )
        second[f"op_{key}"] = choice_q((prompts.DATE_OP if date else prompts.VALUE_OP).format(name=key), ops)  # fmt: skip
    rows = shape not in ("count", "values")
    shown = [c for c in types if c not in NOT_SHOWN] if rows else []
    for c in shown:
        second[f"out_{c}"] = {
            "type": "noul",
            "instructions": prompts.OUTPUT.format(name=state_key(c)),
        }
    fields = {c: pointer(c) for c in types if c not in NOT_SHOWN}
    if shape == "count":
        second["group"] = choice_q(prompts.GROUP, {"none": prompts.NO_GROUP, **fields})
        second["distinct"] = choice_q(prompts.DISTINCT, {"rows": prompts.ALL_ROWS, **fields})
    elif shape == "values":
        second["group"] = choice_q(prompts.VALUES_OF, fields)
    else:
        sortable = is_time if shape in ("timeline", "latest", "earliest") else lambda k: is_time(k) or is_number(k)  # fmt: skip
        orders = {c: pointer(c) for c in types if sortable(types[c]) and c not in NOT_BOUND}
        if len(orders) > 1:
            second["order"] = choice_q(prompts.ORDER, orders)
        if shape in ("lookup", "list", "search", "timeline"):
            second["direction"] = choice_q(prompts.DIRECTION, prompts.DIRECTIONS)
    if len(vals) > 1:
        second["values_mode"] = choice_q(prompts.VALUES_MODE, prompts.VALUES_MODES)
    if one.get("compares", {}).get("noul", 0) > STRICT and (
        options := pairs(cat, table, link, edge)
    ):
        second["compare"] = choice_q(prompts.COMPARE, {"none": prompts.NO_COMPARE, **options})
    if one.get("empties", {}).get("noul", 0) > STRICT:
        optional = [c for c in types if c not in NOT_SHOWN and types[c].startswith(("Nullable(", "Array("))]  # fmt: skip
        empties = {f"{c}:{st}": f"{c} is {'empty' if st == 'null' else 'set'}" for c in optional for st in ("null", "notnull")}  # fmt: skip
        second["empty"] = choice_q(prompts.EMPTY, {"none": prompts.NO_EMPTY, **dict(list(empties.items())[:MAX_OPTIONS])})  # fmt: skip
    state = {"need": need, "values": vals, "records": f"{table}: {cat.tables[table]['about']}", "columns": {state_key(c): m for c, m in meanings.items()}}  # fmt: skip
    two = await session.decide(state, second, "plan:slots")

    filters, bound, unused = [], [], []
    for key, value in vals.items():
        if f"bind_{key}" not in second:
            unused.append(value)
            continue
        answer = two[f"bind_{key}"]
        column = picked(answer, second[f"bind_{key}"]["criteria"])
        if column == "none":
            unused.append(value)
            continue
        uncertain += uncertain_slot(
            f"column for value {value!r}", answer, column, config.confidence
        )
        owner, _, names = column.rpartition(".")
        owner = owner or table
        name, *also = names.split("|")
        op = picked(two[f"op_{key}"], second[f"op_{key}"]["criteria"])
        if is_date(value):
            uncertain += uncertain_slot(f"comparison for {value!r}", two[f"op_{key}"], op, config.confidence)  # fmt: skip
        elif op == "eq" and shape == "search" and is_text(kind(column)):
            op = "contains"
        filters.append(Filter(owner, name, op, value, tuple(also)))
        if owner == table:
            bound.append(name)
    compares, empty = [], "none"
    if "compare" in second and (pair := picked(two["compare"], second["compare"]["criteria"])) != "none":  # fmt: skip
        a, op, b = re.fullmatch(r"(\w+)(!=|==)([\w.]+)", pair).groups()
        compares.append(Compare(a, "ne" if op == "!=" else "eq", b.rpartition(".")[2], "." in b))
    if "empty" in second and (empty := picked(two["empty"], second["empty"]["criteria"])) != "none":  # fmt: skip
        column, _, state_ = empty.rpartition(":")
        filters.append(Filter(table, column, state_, ""))
    outputs = [c for c in shown if two[f"out_{c}"].get("noul", 0) >= 0.5]
    outputs = [*dict.fromkeys(bound + list(cat.tables[table].get("key_columns") or []) + outputs)]
    group = order = distinct = None
    if "group" in second and (g := picked(two["group"], second["group"]["criteria"])) != "none":
        group = g
    if "distinct" in second and (d := picked(two["distinct"], second["distinct"]["criteria"])) != "rows":  # fmt: skip
        distinct = d
    if "order" in second:
        order = picked(two["order"], second["order"]["criteria"])
    desc = "direction" in second and picked(two["direction"], prompts.DIRECTIONS) == "desc"
    all_values = "values_mode" in second and picked(two["values_mode"], prompts.VALUES_MODES) == "all"  # fmt: skip
    if link and not any(f.table == link for f in filters) and not any(c.linked for c in compares):
        link = None
    if link:  # the shape barely changes the rows; an unused link does not matter
        uncertain += uncertain_slot("link", one["link"], link, config.confidence)
    plan = Plan(table, shape, filters, [c for c in outputs if c in types], group, order, desc,
                Link(link, edge[1], edge[0]) if link else None, config.fetch_rows, compares, distinct,
                all_values=all_values)  # fmt: skip
    return plan, uncertain, unused
