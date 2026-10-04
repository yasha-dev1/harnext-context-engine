"""research(question): route (Jev) -> one agent branch per chosen source (gpt-oss, concurrent)
-> judge the findings (Jev) -> one report call (gpt-oss) with a direct answer and citations.

    engine = Engine(store=<RAW export dir>, memory=<notes dir or None>, config=EngineConfig())
    result = await engine.research("...")   # answer, evidence, partial, usage, seconds, trace
    await engine.aclose()

Routing is per source group (the table-name prefix: jira / github / mail, plus the notes), not
per table: tables of one system share identifiers and join to each other, a group's paragraph
in the catalog is what separates questions, and the table is chosen inside `find` anyway.
Budgets (time, chat tokens, Jev calls, turns) end branches gracefully; the report then says
what is missing and the result is marked partial. InfraError propagates.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

from harnext_eval.e2.v2.engine import prompts, verify
from harnext_eval.e2.v2.engine.branch import Branch, conclude, run
from harnext_eval.e2.v2.engine.clients import ChatClient, InfraError, JevClient, ProviderError
from harnext_eval.e2.v2.engine.config import EngineConfig, code_hash, config_hash
from harnext_eval.e2.v2.engine.memory import MemoryTools, load
from harnext_eval.e2.v2.engine.plan import times_sql
from harnext_eval.e2.v2.engine.rawtools import RawTools, describe_times
from harnext_eval.e2.v2.engine.session import BudgetExceeded, Session, SharedDB
from harnext_eval.e2.v2.engine.translate import picked

MEMORY = "memory"


def infra(error: Exception) -> InfraError:
    """A provider failure as the runner's pause signal (kept as is when it already is one)."""
    return error if isinstance(error, InfraError) else InfraError(f"provider rejected a request: {error}")  # fmt: skip


def open_store(store: Path, scratch: Path | None = None):
    """Open a RAW export; one retry (an export may be swapped in place while we open it)."""
    from harnext_eval.e2.v2.stores.raw.query import open as open_raw

    try:
        return open_raw(Path(store), scratch)
    except Exception:  # noqa: BLE001 - retried once, then raised
        time.sleep(1.0)
        return open_raw(Path(store), scratch)


class Engine:
    def __init__(self, store: Path | None = None, memory: Path | None = None,
                 config: EngineConfig | None = None, *, jev=None, chat=None, db=None,
                 scratch: Path | None = None):  # fmt: skip
        self.config = config or EngineConfig()
        if db is None and store is not None:
            db = open_store(store, scratch)
        self.db = SharedDB(db) if db is not None else None
        self.memory = load(Path(memory)) if memory else None
        if self.db is None and not self.memory:
            raise ValueError("the engine needs a RAW export, a memory directory, or both")
        self.jev = jev or JevClient(self.config)
        self.chat = chat or ChatClient(self.config)
        self.hash = config_hash(self.config)
        catalog = self.db.catalog.doc if self.db else {}
        self.identity = {"config_sha256": self.hash, "code_sha256": code_hash(), "checkpoint": catalog.get("checkpoint"),
                         "boundary": catalog.get("boundary"), "memory_files": len(self.memory or {}),
                         "chat_model": self.config.chat_model, "chat_base_url": self.config.chat_base_url,
                         "jev_model": self.config.jev_model, "sql_fallback": self.config.sql_fallback,
                         "budget_feedback": self.config.budget_feedback, "tool_budget": self.config.tool_budget,
                         "deadline_s": self.config.deadline_s}  # fmt: skip

    async def aclose(self) -> None:
        await self.jev.close()
        await self.chat.close()
        if self.db:
            self.db.close()
            self.db = None

    # ---- routing -------------------------------------------------------------------------
    def sources(self) -> dict[str, str]:
        found = {}
        if self.db:
            cat = self.db.catalog
            for group, tables in cat.groups().items():
                system = cat.tables[tables[0]].get("system", group)
                found[group] = f"{system}. " + " ".join(f"{t}: {cat.summary(t)}" for t in tables)
        if self.memory:
            found[MEMORY] = prompts.MEMORY_SOURCE
        return found

    async def route(self, session: Session, question: str) -> dict:
        sources = self.sources()
        if len(sources) == 1:
            return {"probabilities": dict.fromkeys(sources, 1.0), "chosen": list(sources)}
        questions = {g: {"type": "noul", "instructions": prompts.ROUTE.format(name=g)} for g in sources}  # fmt: skip
        answers = await session.decide({"question": question, "sources": sources}, questions, "route")  # fmt: skip
        p = {g: round(float(answers[g].get("noul", 0)), 4) for g in sources}
        ranked = sorted(p, key=lambda g: -p[g])
        chosen = [g for g in ranked if p[g] >= self.config.route_threshold][: self.config.max_branches]  # fmt: skip
        return {"probabilities": p, "chosen": chosen or ranked[:1]}

    # ---- branches ------------------------------------------------------------------------
    def make(self, session: Session, source: str) -> tuple[Branch, str]:
        """One branch: the notes, or one raw source group, or (`+`-joined names) several raw
        groups served by one agent."""
        if source == MEMORY:
            branch = Branch(source, "memory", tools=MemoryTools(session, self.memory))
            return branch, prompts.MEMORY_BRANCH
        cat = self.db.catalog
        groups = source.split("+")
        tables = [t for g in groups for t in cat.groups()[g]]
        branch = Branch(source, "raw", tools=RawTools(session, tables))
        notes = [f"{t}: {n}" for t in tables for n in cat.notes(t)] + ([cat.doc["people"]] if cat.doc.get("people") else [])  # fmt: skip
        label = " and ".join(dict.fromkeys(cat.tables[cat.groups()[g][0]].get("system", g) for g in groups))  # fmt: skip
        system = prompts.RAW_BRANCH.format(
            label=label, boundary=cat.doc.get("boundary"),
            tables="; ".join(f"{t} ({cat.tables[t]['about']})" for t in tables),
        ) + ("\n\nNotes on these records:\n" + "\n".join(f"- {n}" for n in notes) if notes else "")  # fmt: skip
        return branch, system

    def plan_branches(self, chosen: list[str]) -> list[str]:
        """The branches to run for the routed sources (`cross_source` = single: all raw groups
        in one agent)."""
        raw = [s for s in chosen if s != MEMORY]
        if self.config.cross_source == "single" and len(raw) > 1:
            return ["+".join(raw), *(s for s in chosen if s == MEMORY)]
        return chosen

    async def branches(self, session: Session, question: str, chosen: list[str]) -> list[Branch]:
        """Run the branches; returned in routing order, or with `parallel_branches` (several chat
        calls at once, completion order varies) by source name: whatever follows (judging,
        combine, report, evidence, rows) sees findings in an order independent of timing."""
        made = [self.make(session, s) for s in chosen]
        if self.config.parallel_branches > 0:
            made.sort(key=lambda m: m[0].source)
            session.parallel(min(len(made), self.config.parallel_branches))
        tasks = {asyncio.create_task(run(session, b, system, question)): b for b, system in made}
        pending = set(tasks)
        try:
            while pending:
                wait = max(0.0, session.left() - self.config.reserve_s)
                done, pending = await asyncio.wait(pending, timeout=wait, return_when=asyncio.FIRST_EXCEPTION)  # fmt: skip
                for task in done:
                    error = task.exception()
                    if isinstance(error, InfraError | ProviderError):
                        raise infra(error)
                    if error is not None:
                        branch = tasks[task]
                        branch.status, branch.error = "failed", f"{type(error).__name__}: {error}"
                        branch.salvage(branch.error)
                if not done:
                    break
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        cut = [tasks[task] for task in pending]
        if cut and self.config.budget_feedback:  # every cut branch still writes its own answer
            await asyncio.gather(*(conclude(session, b) for b in cut))
        for branch in cut:
            branch.salvage(session.exhausted() or "time budget")
        return [b for b, _ in made]

    # ---- aggregation ---------------------------------------------------------------------
    async def judge(self, session: Session, question: str, branches: list[Branch]) -> dict:
        """Per finding: relevance to the question and whether it states facts (one Jev call, when
        there are several findings, which also asks whether the question compares the times of
        two events), and support by the rows the finding cites. Irrelevant or unsupported
        findings are dropped; the best one is kept, flagged low-confidence, when nothing passes;
        "nothing found" findings are dropped when another kept finding states facts."""
        usable = [b for b in branches if b.finding]
        config = self.config
        try:
            (relevance, factual, timing), support = await asyncio.gather(self.relevance(session, question, usable),
                                                                         asyncio.gather(*(self.support(session, question, b) for b in usable)))  # fmt: skip
        except BudgetExceeded:
            return {"relevance": {}, "factual": {}, "support": {}, "timing": None, "kept": [b.source for b in usable], "flagged": []}  # fmt: skip
        how = {b.source: h for b, (_, h) in zip(usable, support, strict=True)}
        support = {b.source: v for b, (v, _) in zip(usable, support, strict=True)}
        score = {s: relevance.get(s, 1.0) * support[s] for s in support}
        kept = [s for s in score if relevance.get(s, 1.0) >= config.finding_threshold and support[s] >= config.support_threshold]  # fmt: skip
        if any(factual.get(s, 1.0) >= 0.5 for s in kept):
            kept = [s for s in kept if factual.get(s, 1.0) >= config.factual_threshold]
        flagged = [] if kept or not score else [max(score, key=score.get)]
        return {"relevance": relevance, "factual": factual, "support": support, "support_by": how,
                "timing": timing, "kept": kept or flagged, "flagged": flagged}  # fmt: skip

    async def relevance(self, session: Session, question: str, usable: list[Branch]) -> tuple[dict, dict, float | None]:  # fmt: skip
        if len(usable) < 2:
            return {}, {}, None
        state = {"question": question, "findings": {b.source: b.finding[:3000] for b in usable}}
        questions = {f"{b.source}:relevant": {"type": "noul", "instructions": prompts.FINDING.format(name=b.source)} for b in usable}  # fmt: skip
        questions |= {f"{b.source}:factual": {"type": "noul", "instructions": prompts.FACTUAL.format(name=b.source)} for b in usable}  # fmt: skip
        questions["timing"] = {"type": "noul", "instructions": prompts.TIMING}
        answers = await session.decide(state, questions, "judge")
        score = lambda key: round(float(answers[key].get("noul", 0)), 4)  # noqa: E731
        return ({b.source: score(f"{b.source}:relevant") for b in usable},
                {b.source: score(f"{b.source}:factual") for b in usable}, score("timing"))  # fmt: skip

    def cited(self, branch: Branch) -> list[str]:
        records = getattr(branch.tools, "records", {})
        ids = [*branch.evidence, *(i for i in records if i in branch.finding)]
        return [i for i in dict.fromkeys(ids) if i in records]

    def candidates(self, branch: Branch) -> list[str]:
        """Events a branch offers the combine step: the rows it cites, else those its last
        successful find showed."""
        records = getattr(branch.tools, "records", {})
        shown = next((c.get("shown") for c in reversed(getattr(branch.tools, "log", [])) if c.get("shown")), [])  # fmt: skip
        return self.cited(branch) or [i for i in shown if i in records]

    async def support(self, session: Session, question: str, branch: Branch) -> tuple[float, str]:
        """(score, how). Deterministic first: every value the finding states occurs in the rows
        it cites -> 1.0; the values it adds to the question occur in anything its branch
        retrieved -> the share found (kept from `support_share` on: the values are real data).
        Otherwise Jev judges the finding
        against the rows it cites (or, citing none, the latest results)."""
        records = getattr(branch.tools, "records", {})
        cited = "\n".join(records[i] for i in self.cited(branch))[: self.config.support_chars]
        stated = verify.values_in(branch.finding)
        if stated and cited and not verify.unverified(branch.finding, cited):
            return 1.0, "contained in cited rows"
        retrieved = "\n".join(branch.results)
        new = [
            v for v in stated if v not in verify.normalise(question)
        ]  # values the question lacks
        if new:
            share = 1 - len(verify.unverified(branch.finding, retrieved, question)) / len(new)
            if share >= self.config.support_share:
                return round(share, 4), "contained in retrieved rows"
        seen = cited or retrieved[-self.config.support_chars :]
        if not seen:
            return 0.0, "nothing retrieved"
        state = {"finding": branch.finding[:3000], "records": seen}
        answers = await session.decide(state, {"support": {"type": "noul", "instructions": prompts.SUPPORT}}, "support")  # fmt: skip
        return round(float(answers["support"].get("noul", 0)), 4), "cited rows" if cited else "latest results"  # fmt: skip

    # ---- combine: a query over facts of two branches -------------------------------------
    async def combine(self, session: Session, question: str, kept: list[Branch]) -> Branch | None:
        """When the question relates the instants of two events found by different branches:
        Jev picks the two cited events and the unit and rounding the question asks for; SQL
        orders them and measures the time between them."""
        events = {i: getattr(b.tools, "records", {})[i] for b in kept for i in self.candidates(b)[: self.config.combine_events]}  # fmt: skip
        if len({i.split(":")[0] for i in events}) < 1 or len(events) < 2:
            return None
        keys = {f"e{n}": i for n, i in enumerate(events, 1)}
        options = {k: " ".join(events[i].split())[:240] for k, i in keys.items()}
        questions = {"first": {"type": "choice", "instructions": prompts.EVENT.format(which="first"), "criteria": options},
                     "second": {"type": "choice", "instructions": prompts.EVENT.format(which="second"), "criteria": options},
                     "unit": {"type": "choice", "instructions": prompts.UNIT, "criteria": prompts.UNITS},
                     "rounding": {"type": "choice", "instructions": prompts.ROUNDING, "criteria": prompts.ROUNDINGS}}  # fmt: skip
        answers = await session.decide({"question": question}, questions, "combine")
        a, b = keys[picked(answers["first"], options)], keys[picked(answers["second"], options)]
        unit, rounding = picked(answers["unit"], prompts.UNITS), picked(answers["rounding"], prompts.ROUNDINGS)  # fmt: skip
        if a == b:
            return None
        places = await session.db.sources([a, b])
        if a not in places or b not in places:
            return None
        sql = times_sql(session.catalog, (places[a], a), (places[b], b), unit, rounding)
        result = await session.db.sql(sql, max_rows=1)
        record = {"tool": "combine", "events": [a, b], "unit": unit, "rounding": rounding, "sql": sql, "error": result["error"]}  # fmt: skip
        if result["error"] or not result["rows"]:
            return None
        text = describe_times([a, b], result["rows"][0], unit, rounding)
        record["result"] = text
        tools = SimpleNamespace(
            log=[record], records={a: text, b: text}, seen={a, b}, unlocked=None
        )
        return Branch("combined", "raw", status="done", finding=f"Computed from the records: {text}",
                      evidence=[a, b], results=[text], tools=tools)  # fmt: skip

    # ---- report with verification --------------------------------------------------------
    async def write(self, session: Session, messages: list[dict]) -> str | None:
        try:
            message = await asyncio.wait_for(session.complete(messages, "report", max_tokens=self.config.report_tokens), timeout=max(1.0, session.left()))  # fmt: skip
        except (TimeoutError, BudgetExceeded):
            return None
        text = verify.normalise((message.get("content") or "").strip())
        if not text and self.config.budget_feedback:
            session.empty_report = True  # v2: say why no report was written
        return text or None

    async def report(self, session: Session, question: str, kept: list[Branch], flagged=(), corpus: str = "", ids=()) -> tuple[str, bool, dict]:  # fmt: skip
        """Write the report, then: citations not found in the retrieved rows are removed;
        values found neither in rows nor findings -> one regeneration naming them, then the
        lines stating them are dropped; Jev verifies the report against the findings
        (supported / unsupported / declined) -> unsupported: one regeneration, then partial."""
        check = {"removed_citations": [], "unverified_values": [], "regenerated": [], "stripped": [], "verdict": None}  # fmt: skip
        parts = [f"## Findings from the {b.source} records{' (partial)' if b.status != 'done' else ''}{' (low confidence: NOT confirmed by the retrieved records)' if b.source in flagged else ''}\n{b.finding[: self.config.finding_chars]}\nEvidence: {', '.join(b.evidence) or '-'}\n{self.grounds(b)}" for b in kept]  # fmt: skip
        if not parts:
            check["verdict"] = "declined"
            return "Nothing relevant was found in the records; the question could not be answered.", True, check  # fmt: skip
        findings = "\n\n".join(parts)
        everything = f"{corpus}\n{findings}"
        base = [{"role": "system", "content": prompts.REPORT}, {"role": "user", "content": f"Question: {question}\n\n{findings}"}]  # fmt: skip
        text = await self.write(session, base)
        if text is None:
            reason = "empty reply" if getattr(session, "empty_report", False) else "time budget"
            return f"(report not written: {reason}) " + findings, True, check

        known = (
            question + "\n" + "\n".join(map(str, prompts.texts().values()))
        )  # the question and instructions

        def clean(text: str) -> tuple[str, list[str]]:
            text, removed = verify.citations(text, ids)
            check["removed_citations"] += removed
            return text, verify.unverified(text, everything, known)

        text, bad = clean(text)
        if not self.config.verify_report:
            return text, False, check
        if bad and session.left() > self.config.reserve_s / 4:
            check["unverified_values"] += bad
            check["regenerated"].append("values")
            again = await self.write(session, [*base, {"role": "assistant", "content": text}, {"role": "user", "content": prompts.FIX_VALUES.format(values=", ".join(bad))}])  # fmt: skip
            text, bad = clean(again) if again else (text, bad)
            if bad:
                check["stripped"] += bad
                text = verify.strip(text, bad)
        verdict = await self.verdict(session, question, findings, text)
        check["verdict"] = verdict
        if verdict == "unsupported" and session.left() > self.config.reserve_s / 4:
            check["regenerated"].append("unsupported")
            check["drafts"] = [text]
            again = await self.write(session, [*base, {"role": "assistant", "content": text}, {"role": "user", "content": prompts.FIX_UNSUPPORTED}])  # fmt: skip
            if again:
                candidate, bad = clean(again)
                if bad:
                    candidate = verify.strip(candidate, bad)
                second = await self.verdict(session, question, findings, candidate)
                check["drafts"].append(candidate)
                if second in ("supported", "declined"):  # a rewrite is kept only when it verifies
                    text, verdict = candidate, second
                    check["stripped"] += bad
                check["verdict"] = verdict
        return text, verdict == "unsupported" or bool(check["stripped"]), check

    def grounds(self, branch: Branch) -> str:
        """The rows a finding cites (else the branch's last result), compact: the report writer
        can check and complete the finding against them."""
        records = getattr(branch.tools, "records", {})
        cited = "\n".join(records[i] for i in self.cited(branch))
        if cited:
            return "Records it cites:\n" + cited[: self.config.grounds_chars]
        return "Last records retrieved:\n" + branch.results[-1][: self.config.grounds_chars] if branch.results else ""  # fmt: skip

    async def verdict(
        self, session: Session, question: str, findings: str, text: str
    ) -> str | None:
        state = {"question": question, "findings": findings[:12000], "report": text[:6000]}
        try:
            answers = await session.decide(state, {"verdict": {"type": "choice", "instructions": prompts.VERIFY, "criteria": prompts.VERDICTS}}, "verify")  # fmt: skip
        except BudgetExceeded:
            return None
        return picked(answers["verdict"], prompts.VERDICTS)

    # ---- entry point ---------------------------------------------------------------------
    async def research(self, question: str) -> dict:
        """See the module docstring. Provider failures raise InfraError; every exception leaving
        here carries `.usage` (the metered cost so far) and `.calls`."""
        session = Session(self.config, self.jev, self.chat, self.db, self.memory)
        try:
            return await self.investigate(session, question)
        except BaseException as error:
            failure = infra(error) if isinstance(error, ProviderError) else error
            failure.usage, failure.calls = session.meter.summary(), session.meter.records()
            if failure is not error:
                raise failure from error
            raise

    async def investigate(self, session: Session, question: str) -> dict:
        routing = await self.route(session, question)
        branches = await self.branches(session, question, self.plan_branches(routing["chosen"]))
        judged = await self.judge(session, question, branches)
        kept = [b for b in branches if b.source in judged["kept"]]
        if len(kept) > 1 and (judged.get("timing") or 0) > 0.5:
            try:
                if combined := await self.combine(session, question, kept):
                    kept.append(combined)
                    branches.append(combined)
            except BudgetExceeded:
                pass
        corpus = "\n".join(r for b in branches for r in b.results)
        shown = {i for b in branches for i in getattr(b.tools, "seen", set())}
        answer, unsure, check = await self.report(session, question, kept, judged["flagged"], corpus, shown)  # fmt: skip
        evidence = list(dict.fromkeys([*verify.cited(answer, shown), *(e for b in kept for e in b.evidence)]))  # fmt: skip
        partial = unsure or bool(judged["flagged"]) or any(b.status != "done" for b in kept)
        calls = [c for b in branches for c in b.trace()["calls"]]
        finds = [c for c in calls if c["tool"] == "find"]
        stats = {"find": len(finds), "find_rows": sum(c.get("outcome") == "rows" for c in finds),
                 "find_empty": sum(c.get("outcome") == "empty" for c in finds),
                 "find_invalid": sum(c.get("outcome") in ("invalid", "error") for c in finds),
                 "find_restate": sum(c.get("outcome") == "restate" for c in finds),
                 "sql": sum(c["tool"] == "sql" for c in calls),
                 "sql_ok": sum(c["tool"] == "sql" and not c.get("error") for c in calls),
                 "read": sum(c["tool"] in ("read", "read_file") for c in calls),
                 "search": sum(c["tool"] == "search" for c in calls),
                 "compare_times": sum(c["tool"] in ("compare_times", "combine") for c in calls),
                 "rejected": [{"sql": c.get("sql") or c.get("query"), "reason": c["rejected"]} for c in calls if c.get("rejected")]}  # fmt: skip
        usage = session.meter.summary()
        seconds = round(session.elapsed(), 3)
        trace = {"question": question, "identity": self.identity, "routing": routing,
                 "branches": [b.trace() for b in branches], "aggregation": judged, "verification": check,
                 "stats": stats, "answer": answer, "evidence": evidence, "partial": partial,
                 "usage": usage, "calls": session.meter.records(), "seconds": seconds}  # fmt: skip
        return {"answer": answer, "evidence": evidence, "partial": partial, "usage": usage,
                "seconds": seconds, "turns": sum(b.turns for b in branches), "trace": trace}  # fmt: skip
