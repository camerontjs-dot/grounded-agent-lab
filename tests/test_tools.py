"""Allowlist, schema, timeout, no-write, and redacted tool receipts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from grounded_agent.models import ResearchRequest
from grounded_agent.pipeline import run_research
from grounded_agent.router import route_intent
from grounded_agent.tools import (
    ALLOWED_TOOLS,
    DENIED_WRITE_TOOLS,
    FixtureToolClient,
    LiveMindgraphAdapter,
    ToolDenied,
    ToolSchemaError,
    ToolTimeout,
    ToolUnavailable,
    retrieve_traced,
)


def test_allowlist_is_exactly_the_two_read_queries() -> None:
    assert ALLOWED_TOOLS == frozenset({"query_knowledge", "query_projects"})
    assert DENIED_WRITE_TOOLS.isdisjoint(ALLOWED_TOOLS)


def test_write_and_unknown_tools_are_denied() -> None:
    client = FixtureToolClient()
    with pytest.raises(ToolDenied, match="write_index"):
        client.invoke("write_index", {"question": "x"})
    with pytest.raises(ToolDenied, match="grant_write"):
        client.invoke("grant_write", {"question": "x"})
    with pytest.raises(ToolDenied, match="not_a_tool"):
        client.invoke("not_a_tool", {"question": "x"})


def test_extra_arguments_fail_schema() -> None:
    client = FixtureToolClient()
    with pytest.raises(ToolSchemaError):
        client.invoke("query_knowledge", {"question": "Harbor indexes", "mode": "write"})
    with pytest.raises(ToolSchemaError):
        client.invoke("query_knowledge", {})


def test_timeout_fails_closed() -> None:
    client = FixtureToolClient(min_delay_s=0.3)
    with pytest.raises(ToolTimeout, match="query_knowledge"):
        client.invoke(
            "query_knowledge",
            {"question": "Harbor trust labels"},
            timeout_s=0.05,
        )


def test_federated_route_calls_both_labelled_tools() -> None:
    request = ResearchRequest(
        request_id="t-fed",
        question="How does the lantern-agent next action relate to Harbor trust labels?",
    )
    trace = retrieve_traced(request, route_intent(request))
    assert trace.tools_used == ("query_knowledge", "query_projects")
    assert trace.tool_errors == ()
    assert trace.bundle.knowledge
    assert trace.bundle.project
    assert all(item.trust_profile == "durable_knowledge" for item in trace.bundle.knowledge)
    assert all(item.trust_profile == "project_status" for item in trace.bundle.project)


def test_receipt_records_tools_but_not_snippets() -> None:
    request = ResearchRequest(
        request_id="t-receipt",
        question="What must Harbor do when combining retrieval results from different indexes?",
    )
    result = run_research(request)
    assert result.receipt.tools_used == ("query_knowledge",)
    serialized = result.receipt.model_dump_json()
    assert "query_knowledge" in serialized
    for item in result.evidence.knowledge:
        assert item.snippet not in serialized
    assert request.question not in serialized


def test_tool_timeout_is_traced_and_yields_empty_evidence() -> None:
    request = ResearchRequest(request_id="t-timeout", question="Harbor trust labels")
    client = FixtureToolClient(min_delay_s=0.3, timeout_s=0.05)
    trace = retrieve_traced(request, route_intent(request), client=client)
    assert trace.tools_used == ("query_knowledge",)
    assert trace.tool_errors == ("query_knowledge:ToolTimeout",)
    assert trace.bundle.citable() == ()


def test_live_adapter_fail_closed_without_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROUNDED_AGENT_MINDGRAPH_URL", raising=False)
    live = LiveMindgraphAdapter(url=None)
    with pytest.raises(ToolUnavailable, match="not set"):
        live.invoke("query_knowledge", {"question": "Harbor"})
    with pytest.raises(ToolDenied):
        live.invoke("write_index", {"question": "Harbor"})
    with pytest.raises(ValidationError):
        live.invoke("query_knowledge", {"question": "Harbor", "scope": "both"})
