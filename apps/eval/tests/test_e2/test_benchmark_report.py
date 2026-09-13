import base64
import gzip
import json
import re

import pytest
from harnext_eval.e2.benchmark_report import build_report, read_run


def saved_run(root, access="none"):
    folder = root / "01-status_at_snapshot"
    (folder / "reader").mkdir(parents=True)
    (folder / "reader/prompt.txt").write_text('Question <script>alert("unsafe")</script>')
    (folder / "condition.yaml").write_text("trial:\n  budget_bytes: 65536\n")
    report = {
        "finished": True,
        "selection": {
            "task_ids": ["task-1"],
            "benchmark_sha256": "fixed",
            "context_access": access,
            "pilot": {"model": "test", "effort": "medium"},
        },
        "results": [
            {
                "task_id": "task-1",
                "family": "status_at_snapshot",
                "cutoff": "2020-01-01",
                "expected": {"value": "Open"},
                "retrieval": None,
            }
        ],
    }
    (root / "report.json").write_text(json.dumps(report))
    return report


def test_no_mcp_report_needs_no_snapshot_and_embeds_source_as_data(tmp_path):
    root = tmp_path / "run"
    saved_run(root)
    out = build_report([root], tmp_path / "report.html")
    html = out.read_text()
    assert "__PILOT_DATA__" not in html
    assert '<script>alert("unsafe")</script>' not in html
    encoded = re.search(r'<script id="payload"[^>]*>([^<]+)</script>', html).group(1)
    data = json.loads(gzip.decompress(base64.b64decode(encoded)))
    assert data["runs"][0]["details"][0]["calls"] == []
    assert data["runs"][0]["details"][0]["snapshot_bytes"] is None
    with pytest.raises(ValueError, match="immutable"):
        build_report([root], out)


def test_report_refuses_incomplete_missing_receipts_and_mismatched_pairs(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    saved_run(first)
    report = saved_run(second)
    report["selection"]["benchmark_sha256"] = "other"
    (second / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="benchmark hashes"):
        build_report([first, second], tmp_path / "bad.html")
    report["results"][0]["retrieval"] = {"context_calls": 1}
    (second / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="receipt count"):
        read_run(second)
    report["finished"] = False
    (second / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="incomplete"):
        read_run(second)
