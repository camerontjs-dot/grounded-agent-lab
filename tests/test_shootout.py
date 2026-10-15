"""Shootout metrics are computed from a live run, not hand-written."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grounded_agent.embedder import NeuralEmbedderUnavailable, encode_neural
from grounded_agent.paths import REPORTS_DIR
from grounded_agent.shootout import METHODS, _prf, run_shootout, write_shootout


def test_prf_helpers() -> None:
    assert _prf(["a"], ["a", "b"]) == (0.5, 1.0)
    assert _prf([], []) == (1.0, 1.0)
    assert _prf(["a"], []) == (0.0, 0.0)


def test_graph_expand_recovers_the_designed_lexical_miss() -> None:
    payload = run_shootout()
    lexical = {row["id"]: row for row in payload["rows"]["lexical"]}
    expanded = {row["id"]: row for row in payload["rows"]["graph_expand"]}
    assert lexical["g-graph-hop"]["miss"] is True
    assert "knowledge/neighbor-only.md" not in lexical["g-graph-hop"]["retrieved"]
    assert expanded["g-graph-hop"]["miss"] is False
    assert "knowledge/neighbor-only.md" in expanded["g-graph-hop"]["retrieved"]


def test_metrics_recompute_from_rows() -> None:
    payload = run_shootout()
    for method in METHODS:
        rows = payload["rows"][method]
        n = len(rows)
        recall = round(sum(row["recall"] for row in rows) / n, 4)
        precision = round(sum(row["precision"] for row in rows) / n, 4)
        assert payload["metrics"][method]["macro_recall"] == recall
        assert payload["metrics"][method]["macro_precision"] == precision
        assert payload["metrics"][method]["n"] == 5


def test_write_shootout_receipt(tmp_path: Path) -> None:
    payload = write_shootout(tmp_path)
    json_path = tmp_path / "retrieval-shootout.json"
    md_path = tmp_path / "retrieval-shootout.md"
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["metrics"] == payload["metrics"]
    text = md_path.read_text(encoding="utf-8")
    assert "Do not generalize" in text
    assert "hashed_vector" in text
    assert "Inspected failures" in text
    assert "n=5 Harbor notes" in text


def _ranking(payload: dict) -> dict:
    metrics = {
        method: {key: value for key, value in row.items() if key != "mean_latency_ms"}
        for method, row in payload["metrics"].items()
    }
    rows = {
        method: [
            {key: value for key, value in item.items() if key != "latency_ms"}
            for item in method_rows
        ]
        for method, method_rows in payload["rows"].items()
    }
    return {"metrics": metrics, "rows": rows}


def test_committed_report_matches_fresh_run() -> None:
    committed_path = REPORTS_DIR / "retrieval-shootout.json"
    assert committed_path.is_file(), "run grounded-agent shootout and commit reports/"
    fresh = run_shootout()
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    assert _ranking(committed) == _ranking(fresh)


def test_neural_embedder_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise ImportError("blocked in test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(NeuralEmbedderUnavailable):
        encode_neural(["Harbor trust labels"])


def test_pipeline_still_green_with_new_note() -> None:
    from grounded_agent.models import ResearchRequest
    from grounded_agent.pipeline import run_research

    result = run_research(
        ResearchRequest(
            request_id="post-hop",
            question="What must Harbor do when combining retrieval results from different indexes?",
        )
    )
    assert result.answer.outcome == "answered"
    salary = run_research(
        ResearchRequest(request_id="post-hop-salary", question="What is Harbor's CEO salary?")
    )
    assert salary.answer.outcome == "abstained"


def test_previous_table_is_kept_as_measured() -> None:
    # The 2026-08-23 receipt stays as it was measured, with its old gates.
    history = REPORTS_DIR / "history" / "2026-08-23-retrieval-shootout.json"
    old = json.loads(history.read_text(encoding="utf-8"))
    assert old["schema"] == "grounded-agent-shootout-v1"
    assert old["metrics"]["hashed_vector"]["abstain_ok"] == 0
    assert old["metrics"]["graph_expand"]["abstain_ok"] == 0
    current = json.loads((REPORTS_DIR / "retrieval-shootout.json").read_text(encoding="utf-8"))
    assert current["previous_report"]["path"] == f"reports/history/{history.name}"


def test_every_method_declares_a_gate_and_separate_build_time() -> None:
    payload = run_shootout(repeats=1)
    assert set(payload["gates"]) == set(METHODS)
    assert set(payload["timing"]["mean_index_build_ms"]) == set(METHODS)
    for row in payload["rows"]["hashed_vector"]:
        assert row["cutoff"] is not None
        assert row["null_scores"] >= 20


def test_graph_expand_abstains_without_a_seed_that_passes_the_lexical_gate() -> None:
    payload = run_shootout(repeats=1)
    insufficient = next(r for r in payload["rows"]["graph_expand"] if r["id"] == "g-insufficient")
    lexical = next(r for r in payload["rows"]["lexical"] if r["id"] == "g-insufficient")
    assert lexical["retrieved"] == []
    assert insufficient["retrieved"] == []
