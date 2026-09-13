"""Constructed development tasks; never label these as real, unseen PRs.

Public task inputs and private gold are exported separately by development.py.
Each coding task starts independently at its own pre-change source snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from harnext_eval.e2.retrieval_metrics import EvidenceSpan
from harnext_eval.types import Probe


@dataclass
class CodingTask:
    task_id: str
    prompt: str
    base: dict[str, str]
    reference: dict[str, str]
    visible_tests: str
    hidden_tests: str
    required_groups: list[set[str]]


def suite(body: str, module: str = "service") -> str:
    return (
        f"import unittest\nfrom {module} import *\n\nclass ContractTests(unittest.TestCase):\n"
        + body
    )


def dataset():
    facts = {
        "pagination-order": "DEC-11 accepted: tenant feed ordering is ascending by (created_at, id); IDs break equal-time ties. Pagination uses exclusive keyset cursors, never numeric offsets.",
        "pagination-shape": "API-4 implemented: feed responses have items and next_cursor; next_cursor is [created_at, id] only when another tenant-visible row remains. Empty and final pages return null. Input rows need not be sorted.",
        "pagination-tenant": "SEC-8 accepted: filter by tenant before applying the cursor and limit. A cursor is a position, not a record lookup. limit must be a positive integer and must reject booleans.",
        "pagination-old": "DEC-02 superseded by DEC-11: an earlier feed proposal used insertion order and numeric offsets. That proposal was never the shipped cursor API.",
        "idempotency-scope": "SEC-17 accepted: idempotency identity is the pair (tenant, key). Reusing a key in a different tenant is an independent request, even if the body matches.",
        "idempotency-fingerprint": "DEC-19 accepted: request fingerprints are SHA-256 of JSON encoded with sort_keys=True and separators=(',', ':'). Dictionary insertion order is irrelevant; list order remains significant.",
        "idempotency-conflict": "API-9 implemented: repeating the same identity and fingerprint returns its original result without calling the factory. A different fingerprint raises ValueError without changing the stored result.",
        "idempotency-old": "SEC-03 superseded by SEC-17: the initial cache keyed only on the client key. Reports of cross-tenant reuse refer to this old implementation, not the accepted contract.",
        "retry-status": "DEC-23 accepted: retry only HTTP 408, 425, 429, 500, 502, 503, and 504. In particular 501, 401, 403, and 404 are permanent. Attempt is a zero-based count of retries already made.",
        "retry-delay": "OPS-6 accepted: attempts 0, 1, and 2 may retry; attempt 3 or higher returns null. Default delay is min(100 * 2**attempt, 1000) milliseconds. Retry-After is integer milliseconds, including zero, capped at 5000; malformed or negative values use the default.",
        "retry-implemented": "PR-18 implemented the HTTP response adapter and retry_delay(status, attempt, retry_after_ms=None) interface. PR-22 proposed random jitter but was rejected; deterministic delay is required by the current worker contract.",
        "retry-old": "OPS-1 superseded by OPS-6: an older note used exponential seconds and treated zero Retry-After as absent. Do not apply it to the current millisecond interface.",
        "retry-batch": "REQ-40 accepted, not implemented: add planner.plan_retries(jobs). Each job has id, status, attempt and optional retry_after_ms. Use the current worker retry contract. Return a list of {'id': job id, 'delay_ms': delay} only for retryable jobs, in input order. Preserve duplicate IDs as separate jobs, retain zero delays, and do not mutate input jobs. The existing service.retry_delay helper is available at this base revision.",
        "snapshot-eligibility": "ADR-31 accepted: a snapshot is eligible only when completed is true, event_time <= event_cutoff, and published_at <= observed_at. Event time and publication time are independent clocks.",
        "snapshot-selection": "ADR-32 accepted: choose the eligible snapshot with maximum (event_time, sequence), returning its sha. Return null if none exists. Never select the newest publication merely because it was published last.",
        "snapshot-implemented": "PR-28 implemented snapshot manifests with sha, event_time, published_at, sequence, and completed. The select_snapshot API returns a sha or null; no wall-clock lookup is implicit.",
        "snapshot-old": "ADR-12 superseded by ADR-31: the old selector filtered event_time only. A publication delay incident showed that this could expose context before readers could actually observe it.",
    }
    for index in range(12):
        facts[f"distractor-{index:02}"] = (
            f"ARCHIVE-{index:02}: the unrelated metrics dashboard paginates display rows by offset, "
            "retries chart refreshes after 30 seconds, and caches anonymous chart requests globally. "
            "These are dashboard UI policies, not tenant feed, worker, or snapshot service contracts."
        )
    files = {f"events/{key}.md": text + "\n" for key, text in facts.items()}
    files["INDEX.md"] = "# Development context\n" + "\n".join(sorted(files)) + "\n"
    annotations = [
        EvidenceSpan(
            unit_id=key, path=f"events/{key}.md", start=0, end=len(text.encode()), quote=text
        )
        for key, text in facts.items()
    ]
    rows = [
        (
            "pagination",
            "Which ordering fields are accepted for the tenant feed, in contrast to the superseded offset proposal? Return the set of field names.",
            ["created_at", "id"],
            ["pagination-order", "pagination-old"],
        ),
        (
            "pagination",
            "Which response fields are already implemented for the feed? Return only the field names.",
            ["items", "next_cursor"],
            ["pagination-shape"],
        ),
        (
            "pagination",
            "After the final tenant-visible page, what is next_cursor? Return the literal string null.",
            "null",
            ["pagination-shape", "pagination-tenant"],
        ),
        (
            "pagination",
            "Which decision replaced the numeric-offset proposal? Return its identifier.",
            "DEC-11",
            ["pagination-old", "pagination-order"],
        ),
        (
            "idempotency",
            "What fields jointly identify an idempotent request under the current accepted contract? Return field names.",
            ["tenant", "key"],
            ["idempotency-scope", "idempotency-old"],
        ),
        (
            "idempotency",
            "Which cryptographic algorithm is specified for the request fingerprint?",
            "SHA-256",
            ["idempotency-fingerprint"],
        ),
        (
            "idempotency",
            "A repeated tenant/key has a different canonical body. Which exception is required by the implemented API?",
            "ValueError",
            ["idempotency-scope", "idempotency-fingerprint", "idempotency-conflict"],
        ),
        ("idempotency", "Who signed off the idempotency security review?", None, []),
        (
            "retry",
            "Which HTTP status codes are retryable under DEC-23? Return a set of strings.",
            ["408", "425", "429", "500", "502", "503", "504"],
            ["retry-status"],
        ),
        (
            "retry",
            "At attempt 2 with malformed Retry-After, what delay in milliseconds follows the current contract? Return a numeric string.",
            "400",
            ["retry-status", "retry-delay", "retry-old"],
        ),
        (
            "retry",
            "Which PR implemented the retry adapter and interface, excluding the rejected jitter proposal?",
            "PR-18",
            ["retry-implemented"],
        ),
        (
            "retry",
            "For HTTP 503 at attempt 3 with Retry-After 0, what does the current contract require? Return the literal string null.",
            "null",
            ["retry-status", "retry-delay"],
        ),
        (
            "snapshot",
            "Which manifest fields are already implemented? Return their exact names.",
            ["sha", "event_time", "published_at", "sequence", "completed"],
            ["snapshot-implemented"],
        ),
        (
            "snapshot",
            "What two fields rank eligible snapshots, rather than newest publication? Return field names.",
            ["event_time", "sequence"],
            ["snapshot-selection", "snapshot-eligibility"],
        ),
        (
            "snapshot",
            "Which additional timestamp prevents the early exposure diagnosed after ADR-12? Return the manifest field name.",
            "published_at",
            ["snapshot-old", "snapshot-eligibility"],
        ),
        (
            "snapshot",
            "What was the measured production p99 publication delay during the incident?",
            None,
            [],
        ),
    ]
    # Scenario probes require applying several retrieved contracts, rather than
    # copying one field. Their inputs are in the question; policy is in context.
    rows.extend(
        [
            (
                "pagination",
                "Apply the accepted feed contract to rows (tenant, created_at, id): "
                "(a,10,7), (b,9,1), (a,10,5), (a,11,2), (a,9,8). For tenant a, limit 2, "
                "cursor [9,8], return the set of IDs in the next page as strings.",
                ["5", "7"],
                ["pagination-order", "pagination-tenant"],
            ),
            (
                "pagination",
                "Rows are (a,10,7), (b,10,6), (a,10,5), (a,11,2), with tuple fields "
                "(tenant,created_at,id). For tenant a, limit 2, no cursor, what next_cursor is required? "
                "Encode the answer as created_at:id, with no spaces.",
                "10:7",
                ["pagination-order", "pagination-tenant", "pagination-shape"],
            ),
            (
                "pagination",
                "Rows are (a,1,1), (b,2,2), (a,3,3). With tuple fields "
                "(tenant,created_at,id), tenant a, limit 2, and cursor [1,9], how many items are returned "
                "and is there a next cursor? Return count:null or count:present.",
                "1:null",
                ["pagination-order", "pagination-tenant", "pagination-shape"],
            ),
            (
                "pagination",
                "For the current feed, classify these requests by their limit value: "
                "A=0, B=true, C=2, D=-3, E=1.5, F=1. Return the set of request labels rejected by "
                "the accepted security contract, ignoring the unrelated dashboard policy.",
                ["A", "B", "D", "E"],
                ["pagination-tenant"],
            ),
            (
                "idempotency",
                "An empty store receives (tenant,key,body) requests in order: "
                "(a,k,{x:1,y:2}), (a,k,{y:2,x:1}), (b,k,{x:1,y:2}), (a,k,{x:9,y:2}), "
                "(a,k,{x:1,y:2}). Factories always succeed and conflicts are caught. How many "
                "factory invocations occur under the accepted contracts? Return a numeric string.",
                "2",
                ["idempotency-scope", "idempotency-fingerprint", "idempotency-conflict"],
            ),
            (
                "idempotency",
                "A successful request (a,k,[1,2]) is followed by A=(a,k,[2,1]), "
                "B=(b,k,[2,1]), C=(a,k,[1,2]), D=(a,j,[2,1]), in that order. Which requests "
                "raise ValueError? Return their labels as a set.",
                ["A"],
                ["idempotency-scope", "idempotency-fingerprint", "idempotency-conflict"],
            ),
            (
                "idempotency",
                "An empty store executes (a,k,{x:1}) whose factory returns R1, then "
                "(a,k,{x:2}) whose factory would return R2, then (a,k,{x:1}) whose factory would "
                "return R3. Catch conflicts. What does the third request return?",
                "R1",
                ["idempotency-fingerprint", "idempotency-conflict"],
            ),
            (
                "idempotency",
                "The old global-key implementation and SEC-17 disagree about "
                "requests (a,k,{x:1}) and (b,k,{x:1}). Which accepted decision and implemented API "
                "contract jointly establish independent execution and replay behavior? Return identifiers.",
                ["SEC-17", "API-9"],
                ["idempotency-old", "idempotency-scope", "idempotency-conflict"],
            ),
            (
                "retry",
                "Jobs have (status,attempt,Retry-After): A=(503,2,bad), B=(429,0,0), "
                "C=(501,0,17), D=(408,3,17), E=(504,1,9000). Which jobs retry under the current "
                "worker policy? Return labels as a set.",
                ["A", "B", "E"],
                ["retry-status", "retry-delay"],
            ),
            (
                "retry",
                "For jobs (503,2,bad), (429,0,0), (501,0,17), (408,3,17), (504,1,9000), "
                "where tuples are (status,attempt,Retry-After), sum the delays of jobs that retry. "
                "Use the accepted millisecond policy, excluding permanent failures. Return a numeric string.",
                "5400",
                ["retry-status", "retry-delay", "retry-old"],
            ),
            (
                "retry",
                "A worker gets HTTP 503 on attempts 0, 1, 2, and 3, always without "
                "Retry-After. How much total waiting is scheduled before the retry budget is exhausted? "
                "Return milliseconds as a numeric string, applying the accepted policy rather than OPS-1.",
                "700",
                ["retry-status", "retry-delay", "retry-old"],
            ),
            (
                "retry",
                "Which of the following capabilities are implemented according to this "
                "snapshot: HTTP response adapter, retry_delay interface, random jitter? Return the "
                "set of those exact capability names that are implemented.",
                ["HTTP response adapter", "retry_delay interface"],
                ["retry-implemented"],
            ),
            (
                "snapshot",
                "Snapshots (sha,event_time,published_at,sequence,completed) are "
                "(A,10,12,1,true), (B,11,20,2,true), (C,10,13,3,true), (D,12,14,4,false), "
                "(E,13,14,5,true). At event_cutoff=12 and observed_at=14, which sha is selected?",
                "C",
                ["snapshot-eligibility", "snapshot-selection"],
            ),
            (
                "snapshot",
                "Snapshots (sha,event_time,published_at,sequence,completed) are "
                "(A,10,12,1,true), (B,11,20,2,true), (C,10,13,3,true), (D,12,14,4,false), "
                "(E,13,14,5,true). With event_cutoff fixed at 12, which sha is selected when "
                "observed_at advances from 14 to 20? Return the new sha.",
                "B",
                ["snapshot-eligibility", "snapshot-selection"],
            ),
            (
                "snapshot",
                "Snapshots (sha,event_time,published_at,sequence,completed) are "
                "(A,10,12,1,true), (B,11,20,2,true), (C,10,13,3,true). At event_cutoff=10 "
                "and observed_at=12, which sha is selected? Respect inclusive boundaries.",
                "A",
                ["snapshot-eligibility", "snapshot-selection"],
            ),
            (
                "snapshot",
                "Snapshots (sha,event_time,published_at,sequence,completed) are "
                "(A,10,20,1,true), (B,11,12,2,false), (C,13,12,3,true). At event_cutoff=12 "
                "and observed_at=12, what should the implemented selector return? Use the literal string null.",
                "null",
                ["snapshot-eligibility", "snapshot-selection", "snapshot-implemented"],
            ),
        ]
    )
    probes = []
    requirements = {}
    for index, (family, question, gold, required) in enumerate(rows, 1):
        probe_id = f"dev-{index:02}"
        probes.append(
            Probe(
                probe_id=probe_id,
                family="abstention" if gold is None else "multisource",
                entity=family,
                T=datetime(2026, 1, 2, tzinfo=UTC),
                question=question,
                gold=gold,
                gold_type="links" if isinstance(gold, list) else "exact",
                source_event_ids=required,
            )
        )
        requirements[probe_id] = [{value} for value in required]
    tasks = [
        CodingTask(
            "dev-code-pagination",
            "Fix the tenant feed pagination bug in paginate(rows, tenant, limit, cursor=None). "
            "Match the accepted feed/API/security contracts at this snapshot, preserve the response "
            "interface, and handle ties, final pages, interleaved tenants, and invalid limits.",
            {
                "service.py": "def paginate(rows, tenant, limit, cursor=None):\n    start = 0\n    items = rows[start:start + limit]\n    return {'items': items, 'next_cursor': None}\n"
            },
            {
                "service.py": "def paginate(rows, tenant, limit, cursor=None):\n    if type(limit) is not int or limit <= 0:\n        raise ValueError('positive integer limit required')\n    selected = sorted((r for r in rows if r['tenant'] == tenant), key=lambda r: (r['created_at'], r['id']))\n    if cursor is not None:\n        selected = [r for r in selected if (r['created_at'], r['id']) > tuple(cursor)]\n    items = selected[:limit]\n    next_cursor = [items[-1]['created_at'], items[-1]['id']] if len(selected) > limit else None\n    return {'items': items, 'next_cursor': next_cursor}\n"
            },
            suite(
                "    def test_regression_empty(self):\n        self.assertEqual(paginate([], 'a', 2), {'items': [], 'next_cursor': None})\n    def test_acceptance_tenant(self):\n        rows = [{'tenant': 'b', 'created_at': 1, 'id': 1}, {'tenant': 'a', 'created_at': 2, 'id': 2}]\n        self.assertEqual([r['id'] for r in paginate(rows, 'a', 1)['items']], [2])\n"
            ),
            suite(
                "    def test_acceptance_ties_and_pages(self):\n        rows = [{'tenant': t, 'created_at': c, 'id': i} for t,c,i in [('a',2,3),('a',1,2),('b',1,9),('a',1,1)]]\n        first = paginate(rows, 'a', 2)\n        self.assertEqual([r['id'] for r in first['items']], [1,2])\n        self.assertEqual(first['next_cursor'], [1,2])\n        last = paginate(rows, 'a', 2, first['next_cursor'])\n        self.assertEqual([r['id'] for r in last['items']], [3])\n        self.assertIsNone(last['next_cursor'])\n    def test_acceptance_missing_cursor_row(self):\n        row = {'tenant':'a','created_at':2,'id':4}\n        self.assertEqual(paginate([row], 'a', 1, [1,99])['items'], [row])\n    def test_acceptance_limits(self):\n        for value in (0,-1,True,1.5):\n            with self.assertRaises(ValueError):\n                paginate([], 'a', value)\n"
            ),
            [{"pagination-order"}, {"pagination-shape"}, {"pagination-tenant"}],
        ),
        CodingTask(
            "dev-code-idempotency",
            "Fix execute_once(store, tenant, key, body, factory) so request replay and conflicts obey "
            "the current tenant isolation and fingerprint contracts. Preserve stored results on a "
            "conflict and avoid invoking factory for a valid replay.",
            {
                "service.py": "def execute_once(store, tenant, key, body, factory):\n    if key not in store:\n        store[key] = factory()\n    return store[key]\n"
            },
            {
                "service.py": "import hashlib\nimport json\n\ndef execute_once(store, tenant, key, body, factory):\n    identity = (tenant, key)\n    fingerprint = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()\n    if identity in store:\n        saved = store[identity]\n        if saved['fingerprint'] != fingerprint:\n            raise ValueError('idempotency conflict')\n        return saved['result']\n    result = factory()\n    store[identity] = {'fingerprint': fingerprint, 'result': result}\n    return result\n"
            },
            suite(
                "    def test_regression_repeat(self):\n        store = {}\n        self.assertEqual(execute_once(store, 'a', 'x', {}, lambda: 1), 1)\n        self.assertEqual(execute_once(store, 'a', 'x', {}, lambda: 2), 1)\n    def test_acceptance_scope(self):\n        store = {}\n        execute_once(store, 'a', 'x', {}, lambda: 1)\n        self.assertEqual(execute_once(store, 'b', 'x', {}, lambda: 2), 2)\n"
            ),
            suite(
                "    def test_acceptance_canonical_conflict(self):\n        store = {}\n        calls = []\n        def factory():\n            calls.append(1)\n            return {'id':7}\n        first = execute_once(store, 'a', 'x', {'b':2,'a':1}, factory)\n        self.assertEqual(execute_once(store, 'a', 'x', {'a':1,'b':2}, factory), first)\n        with self.assertRaises(ValueError):\n            execute_once(store, 'a', 'x', {'a':9,'b':2}, factory)\n        self.assertEqual(execute_once(store, 'a', 'x', {'a':1,'b':2}, factory), first)\n        self.assertEqual(len(calls), 1)\n    def test_acceptance_list_order(self):\n        store = {}\n        execute_once(store, 'a', 'k', [1,2], lambda: 3)\n        with self.assertRaises(ValueError):\n            execute_once(store, 'a', 'k', [2,1], lambda: 4)\n"
            ),
            [{"idempotency-scope"}, {"idempotency-fingerprint"}, {"idempotency-conflict"}],
        ),
        CodingTask(
            "dev-code-retry",
            "Implement the accepted worker retry policy in retry_delay(status, attempt, "
            "retry_after_ms=None). Return delay milliseconds or None. Reconcile the current "
            "operations decision with the older note and rejected jitter proposal.",
            {
                "service.py": "def retry_delay(status, attempt, retry_after_ms=None):\n    if status < 500:\n        return None\n    return retry_after_ms or 1000\n"
            },
            {
                "service.py": "def retry_delay(status, attempt, retry_after_ms=None):\n    if status not in {408,425,429,500,502,503,504} or attempt >= 3:\n        return None\n    default = min(100 * 2**attempt, 1000)\n    try:\n        if isinstance(retry_after_ms, bool) or not isinstance(retry_after_ms, (str,int)):\n            return default\n        value = int(retry_after_ms)\n        return min(value, 5000) if value >= 0 else default\n    except ValueError:\n        return default\n"
            },
            suite(
                "    def test_regression_permanent(self):\n        self.assertIsNone(retry_delay(404, 0))\n    def test_acceptance_rate_limit(self):\n        self.assertEqual(retry_delay(429, 0), 100)\n"
            ),
            suite(
                "    def test_acceptance_statuses(self):\n        for status in (408,425,429,500,502,503,504):\n            self.assertEqual(retry_delay(status, 1), 200)\n        for status in (200,401,403,404,501):\n            self.assertIsNone(retry_delay(status, 0))\n    def test_acceptance_boundary(self):\n        self.assertIsNone(retry_delay(503, 3, 0))\n        self.assertEqual(retry_delay(503, 2, 'bad'), 400)\n        self.assertEqual(retry_delay(503, 2, '-1'), 400)\n    def test_acceptance_header(self):\n        self.assertEqual(retry_delay(503, 0, 0), 0)\n        self.assertEqual(retry_delay(503, 0, '17'), 17)\n        self.assertEqual(retry_delay(503, 0, '99999'), 5000)\n"
            ),
            [{"retry-status"}, {"retry-delay"}, {"retry-implemented"}],
        ),
        CodingTask(
            "dev-code-snapshot",
            "Fix select_snapshot(snapshots, event_cutoff, observed_at) to obey the current "
            "two-clock visibility contract and tie-breaking rule. Return the selected sha or "
            "None. Input manifests may arrive in arbitrary order and include incomplete builds.",
            {
                "service.py": "def select_snapshot(snapshots, event_cutoff, observed_at):\n    eligible = [s for s in snapshots if s['event_time'] <= event_cutoff]\n    return max(eligible, key=lambda s:s['event_time'])['sha'] if eligible else None\n"
            },
            {
                "service.py": "def select_snapshot(snapshots, event_cutoff, observed_at):\n    eligible = [s for s in snapshots if s['completed'] is True and s['event_time'] <= event_cutoff and s['published_at'] <= observed_at]\n    return max(eligible, key=lambda s:(s['event_time'], s['sequence']))['sha'] if eligible else None\n"
            },
            suite(
                "    def test_regression_empty(self):\n        self.assertIsNone(select_snapshot([], 10, 10))\n    def test_acceptance_publication(self):\n        rows = [{'sha':'future','event_time':2,'published_at':9,'sequence':1,'completed':True}]\n        self.assertIsNone(select_snapshot(rows, 5, 5))\n"
            ),
            suite(
                "    def test_acceptance_clocks_and_completion(self):\n        rows = [{'sha':s,'event_time':e,'published_at':p,'sequence':q,'completed':c} for s,e,p,q,c in [('ok',3,4,1,True),('future-event',8,4,2,True),('future-publish',4,9,3,True),('incomplete',5,5,4,False)]]\n        self.assertEqual(select_snapshot(rows, 5, 5), 'ok')\n    def test_acceptance_tie(self):\n        rows = [{'sha':s,'event_time':3,'published_at':p,'sequence':q,'completed':True} for s,p,q in [('older-sequence',5,1),('newer-sequence',4,2)]]\n        self.assertEqual(select_snapshot(rows, 3, 5), 'newer-sequence')\n    def test_acceptance_inclusive(self):\n        row = {'sha':'edge','event_time':5,'published_at':5,'sequence':1,'completed':True}\n        self.assertEqual(select_snapshot([row], 5, 5), 'edge')\n"
            ),
            [{"snapshot-eligibility"}, {"snapshot-selection"}, {"snapshot-implemented"}],
        ),
    ]
    retry_source = tasks[2].reference["service.py"]
    tasks.append(
        CodingTask(
            "dev-code-batch-feature",
            "Implement the accepted batch retry planning feature in planner.plan_retries(jobs), "
            "using the existing worker service and the requirements available at this snapshot. "
            "Preserve worker behavior and support mixed batches without mutating their inputs.",
            {"service.py": retry_source, "planner.py": "def plan_retries(jobs):\n    return []\n"},
            {
                "service.py": retry_source,
                "planner.py": "from service import retry_delay\n\ndef plan_retries(jobs):\n    result = []\n    for job in jobs:\n        delay = retry_delay(job['status'], job['attempt'], job.get('retry_after_ms'))\n        if delay is not None:\n            result.append({'id': job['id'], 'delay_ms': delay})\n    return result\n",
            },
            suite(
                "    def test_regression_empty(self):\n        self.assertEqual(plan_retries([]), [])\n    def test_acceptance_rate_limit(self):\n        self.assertEqual(plan_retries([{'id':'a','status':429,'attempt':0}]), [{'id':'a','delay_ms':100}])\n",
                "planner",
            ),
            suite(
                "    def test_acceptance_mixed_order_and_zero(self):\n        jobs = [{'id':i,'status':s,'attempt':a,'retry_after_ms':h} for i,s,a,h in [('x',503,2,'bad'),('x',429,0,0),('p',501,0,3),('q',408,3,17),('z',504,1,9000)]]\n        import copy\n        before = copy.deepcopy(jobs)\n        self.assertEqual(plan_retries(jobs), [{'id':'x','delay_ms':400},{'id':'x','delay_ms':0},{'id':'z','delay_ms':5000}])\n        self.assertEqual(jobs, before)\n    def test_regression_worker(self):\n        from service import retry_delay\n        self.assertIsNone(retry_delay(501,0))\n        self.assertEqual(retry_delay(503,2),400)\n",
                "planner",
            ),
            [{"retry-batch"}, {"retry-status"}, {"retry-delay"}],
        )
    )
    return files, annotations, probes, requirements, tasks
