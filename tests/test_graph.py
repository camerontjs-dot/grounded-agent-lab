"""LangGraph wrap must match the Phase 1 fixture decisions."""

from __future__ import annotations

import json
from pathlib import Path

from grounded_agent.cli import main
from grounded_agent.graph import GRAPH_NODES, run_research_graph
from grounded_agent.models import ResearchRequest
from grounded_agent.paths import QUESTIONS_PATH
from grounded_agent.pipeline import run_research


def _cases() -> list[dict[str, object]]:
    return json.loads(Path(QUESTIONS_PATH).read_text(encoding="utf-8"))


def _decision(result) -> tuple:
    return (
        result.route.intent,
        tuple(result.route.scopes),
        result.answer.outcome,
        result.answer.abstain_reason,
        tuple((c.trust_profile, c.source_path) for c in result.answer.citations),
        result.receipt.outcome,
        result.receipt.citation_paths,
        result.receipt.abstain_reason,
        result.receipt.receipt_hash,
        result.receipt.content_redacted,
    )


def test_graph_nodes_are_explicit() -> None:
    assert GRAPH_NODES == ("route", "retrieve", "draft", "review", "emit_receipt")


def test_graph_matches_baseline_fixture_table() -> None:
    for case in _cases():
        request = ResearchRequest(request_id=str(case["id"]), question=str(case["question"]))
        baseline = run_research(request)
        graphed = run_research_graph(request)
        assert _decision(baseline) == _decision(graphed), case["id"]


def test_cli_graph_runtime_matches_baseline_shape(capsys) -> None:
    question = "What is the next action for lantern-agent?"
    assert main(["ask", question]) == 0
    baseline = capsys.readouterr().out
    assert main(["ask", "--runtime", "graph", question]) == 0
    graphed = capsys.readouterr().out
    assert "outcome: answered" in baseline
    assert "outcome: answered" in graphed
    assert "project_status" in graphed
