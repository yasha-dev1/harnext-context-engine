"""Dependency signals of the w2026 coding candidates: how much a PR builds on earlier work inside
the window. Written to benchmarks/kafka-50-w2026/dependencies.json (a side file: the task records
the run has pinned are never touched) and used by the charts to split coding results into
dependent vs standalone tasks.

    .venv/bin/python apps/eval/scripts/e2_dependencies.py

Signals, all as of the PR's context cutoff: JIRA Link items and issue links (2 each), other
KAFKA keys and PR numbers named in the ticket/PR text (2 each), a KIP (1), a parent ticket (1),
earlier merged window PRs sharing changed files (1 each), and other PRs of the same ticket
(a multi-PR series). A task is "dependent" at score >= 4 or when it is part of a series.
"""
import ast
import collections
import glob
import json
import re

from harnext_eval.e2.benchmark import inventory
from harnext_eval.e2.v2 import families

issues, prs, commits = inventory()
C = [json.loads(l) for l in open("apps/eval/benchmarks/kafka-500-v2/coding-candidates.jsonl")]
W = [r for r in C if (r.get("checkpoint") or "") >= "2026-01"]
adm = {}
for f in glob.glob("apps/eval/out/kafka-admission-v2/results/**/*.json", recursive=True):
    try: d = json.load(open(f))
    except Exception: continue
    if isinstance(d, dict) and d.get("status"):
        tid = d.get("id") or next((c["id"] for c in C if c["id"] in f), None)
        if tid: adm[tid] = (d["status"], d.get("base_failure"))
def files(r):
    v = r.get("changed_files"); return set(ast.literal_eval(v) if isinstance(v, str) else v or [])
# earlier merged PRs in the window with their files (from candidates; plus inventory prs if it has files)
merged = [(r["pr_merged_at"], r["pr"], r["issue"], files(r)) for r in C if (r.get("pr_merged_at") or "") >= "2026-01-01"]
KEY = re.compile(r"KAFKA-\d+"); PRN = re.compile(r"(?:#|pull/)(\d{5})"); KIP = re.compile(r"KIP-\d+")
out = []
for r in W:
    me, cut = r["issue"], r["context_cutoff"]
    iss = issues.get(me, {})
    links = []
    for h in (iss.get("changelog") or {}).get("histories", []):
        if h["created"][:19] > cut[:19]: continue
        for it in h.get("items", []):
            if it.get("field") == "Link":
                links.append((it.get("toString") or it.get("fromString") or "")[:90])
    fl = (iss.get("fields") or {})
    for l in fl.get("issuelinks") or []:
        o = l.get("outwardIssue") or l.get("inwardIssue") or {}
        links.append(f"{(l.get('type') or {}).get('name')}: {o.get('key')}")
    parent = ((fl.get("parent") or {}).get("key"))
    text = (r.get("description") or "") + " " + (r.get("title") or "") + " " + (fl.get("description") or "" if isinstance(fl.get("description"), str) else "")
    refs = sorted(set(KEY.findall(text)) - {me}); prrefs = sorted(set(PRN.findall(text)) - {r["pr"]}); kips = sorted(set(KIP.findall(text)))
    mine = files(r)
    overlap = [(p, i, sorted(mine & fs)) for (t, p, i, fs) in merged if "2026-01-01" <= t < cut and p != r["pr"] and mine & fs]
    score = 2*len(set(links)) + 2*len(refs) + 2*len(prrefs) + len(overlap) + (1 if parent else 0) + (1 if kips else 0)
    out.append(dict(id=r["id"], issue=me, title=r["title"][:80], ckpt=r["checkpoint"], adm=adm.get(r["id"]), score=score,
                    links=sorted(set(links))[:4], parent=parent, refs=refs[:5], prrefs=prrefs[:5], kips=kips[:3],
                    overlap=[(p, i, len(f)) for p, i, f in overlap][:4]))
out.sort(key=lambda x: -x["score"])
series = collections.Counter(o["issue"] for o in out)
for o in out:
    o["series"] = series[o["issue"]] > 1
    o["dependent"] = o["score"] >= 4 or o["series"]
    o.pop("adm", None)  # admission status is not a dependency property
json.dump({o["id"]: o for o in out}, open("apps/eval/benchmarks/kafka-50-w2026/dependencies.json", "w"), indent=1, sort_keys=True)
print(sum(o["dependent"] for o in out), "of", len(out), "window candidates are dependent")
