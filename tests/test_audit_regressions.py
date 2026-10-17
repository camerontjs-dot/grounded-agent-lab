"""Audit regression gates for the bounded Harbor fixture.

Known failures have strict xfail markers, not silent skips. A repaired behavior
becomes XPASS (and fails CI) until the corresponding mark is removed. Run
pytest --runxfail on this file to see the original defects as ordinary failures.

These tests exercise only the public synthetic corpus and in-process functions.
No hosted model, private notes, real MCP write capability, or live daemon.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path

import pytest

import grounded_agent.pipeline as pipeline_module
import grounded_agent.retrieve as retrieve_module
import grounded_agent.tools as tools_module
from grounded_agent.draft import draft_or_abstain
from grounded_agent.mcp_server import build_harbor_mcp
from grounded_agent.models import EvidenceBundle, EvidenceItem, ResearchRequest
from grounded_agent.provider import ModelDraft, answer_from_model_draft
from grounded_agent.router import route_intent
from grounded_agent.shootout import run_shootout
from grounded_agent.tools import ALLOWED_TOOLS, FixtureToolClient, ToolTimeout, ToolUnavailable

TRUST_QUESTION = "What must Harbor do when combining retrieval results from different indexes?"
FEDERATED_QUESTION = "How does the lantern-agent next action relate to Harbor trust labels?"
COUNT_QUESTION = "How many Harbor notes are in the durable knowledge index?"
TRUST_PATH = "knowledge/trust-labels.md"


def _request(key: str, question: str = TRUST_QUESTION) -> ResearchRequest:
    return ResearchRequest(request_id=key, question=question)


def _trust_evidence() -> EvidenceBundle:
    request = _request("model-evidence")
    return retrieve_module.retrieve(request, route_intent(request))


def _citable_note(snippet: str) -> EvidenceBundle:
    return EvidenceBundle(
        knowledge=(
            EvidenceItem(
                source_path="knowledge/count-note.md",
                trust_profile="durable_knowledge",
                title="Harbor durable knowledge index",
                snippet=snippet,
                score=0.9,
                weak_fit=False,
                citation_class="citable",
            ),
        )
    )


# Positive/negative controls that already pass. They constrain the repair
# against simply changing all answers into refusals.


def test_positive_literal_extraction_remains_answerable() -> None:
    request = _request("positive-extract", COUNT_QUESTION)
    evidence = _citable_note("The Harbor durable knowledge index contains six notes.")
    answer = draft_or_abstain(request, evidence)
    assert answer.outcome == "answered"
    assert "six" in answer.text.lower()
    assert tuple(c.source_path for c in answer.citations) == ("knowledge/count-note.md",)


def test_positive_verbatim_model_claim_keeps_valid_citation() -> None:
    draft = ModelDraft(
        outcome="answered",
        text="Harbor stores durable notes and live project status in two separate indexes.",
        citation_paths=(TRUST_PATH,),
    )
    answer = answer_from_model_draft(draft, _trust_evidence())
    assert answer.outcome == "answered"
    assert tuple(c.source_path for c in answer.citations) == (TRUST_PATH,)


def test_unmodified_receipt_is_reproducible_and_redacted() -> None:
    request = _request("stable-receipt")
    first = pipeline_module.run_research(request)
    second = pipeline_module.run_research(request)
    assert first.receipt.receipt_hash == second.receipt.receipt_hash
    serialized = first.receipt.model_dump_json()
    assert request.question not in serialized
    assert "Harbor stores durable notes" not in serialized


def test_count_question_without_count_evidence_abstains_in_both_runtimes() -> None:
    request = _request("missing-count", COUNT_QUESTION)
    baseline = pipeline_module.run_research(request)
    from grounded_agent.graph import run_research_graph

    graphed = run_research_graph(request)
    assert baseline.answer.outcome == "abstained"
    assert graphed.answer.outcome == "abstained"
    assert baseline.answer.citations == ()
    assert graphed.answer.citations == ()
    assert baseline.receipt.outcome == "abstained"
    assert graphed.receipt.outcome == "abstained"


def test_citable_near_miss_without_number_abstains() -> None:
    request = _request("near-miss", COUNT_QUESTION)
    evidence = _citable_note(
        "Harbor durable knowledge notes live in a separate index from project status."
    )
    answer = draft_or_abstain(request, evidence)
    assert answer.outcome == "abstained"
    assert answer.citations == ()


def test_model_valid_path_does_not_warrant_unrelated_salary_claim() -> None:
    draft = ModelDraft(
        outcome="answered",
        text="Harbor's CEO salary is $10 million.",
        citation_paths=(TRUST_PATH,),
    )
    answer = answer_from_model_draft(draft, _trust_evidence())
    assert answer.outcome == "abstained"
    assert answer.citations == ()


def test_model_valid_path_does_not_warrant_changed_number() -> None:
    draft = ModelDraft(
        outcome="answered",
        text="Harbor stores durable notes and live project status in seven separate indexes.",
        citation_paths=(TRUST_PATH,),
    )
    answer = answer_from_model_draft(draft, _trust_evidence())
    assert answer.outcome == "abstained"
    assert answer.citations == ()


def test_same_path_changed_source_bytes_change_receipt_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    source = retrieve_module.KNOWLEDGE_CORPUS / "trust-labels.md"
    target = knowledge / "trust-labels.md"
    shutil.copyfile(source, target)
    monkeypatch.setattr(retrieve_module, "KNOWLEDGE_CORPUS", knowledge)

    request = _request("same-source-path")
    before = pipeline_module.run_research(request)
    original = target.read_text(encoding="utf-8")
    target.write_text(original + "\nAn unrelated fixture-only source revision.\n", encoding="utf-8")
    after = pipeline_module.run_research(request)

    assert tuple(c.source_path for c in before.answer.citations) == (TRUST_PATH,)
    assert before.answer.citations == after.answer.citations
    assert before.answer.text == after.answer.text
    assert before.receipt.receipt_hash != after.receipt.receipt_hash
    for result in (before, after):
        receipt_text = result.receipt.model_dump_json()
        assert request.question not in receipt_text
        assert "An unrelated fixture-only source revision" not in receipt_text


@pytest.mark.parametrize("failed_tool", ["query_knowledge", "query_projects"])
@pytest.mark.xfail(strict=True, reason="#19: missing required index still answers")
def test_federated_outage_cannot_produce_complete_answer(
    monkeypatch: pytest.MonkeyPatch, failed_tool: str
) -> None:
    class FailedIndexClient(FixtureToolClient):
        def invoke(self, name: str, arguments: dict, *, timeout_s: float | None = None):
            if name == failed_tool:
                raise ToolUnavailable("injected fixture index outage")
            return super().invoke(name, arguments, timeout_s=timeout_s)

    original = tools_module.retrieve_traced

    def with_injected_outage(request: ResearchRequest, route):
        return original(request, route, client=FailedIndexClient())

    monkeypatch.setattr(pipeline_module, "retrieve_traced", with_injected_outage)
    result = pipeline_module.run_research(_request("partial-" + failed_tool, FEDERATED_QUESTION))

    assert f"{failed_tool}:ToolUnavailable" in result.receipt.tool_errors
    assert result.answer.outcome == "abstained"
    assert result.answer.citations == ()
    assert result.receipt.outcome == "abstained"


@pytest.mark.xfail(strict=True, reason="#12: timeout still waits for worker")
def test_timeout_restores_caller_within_bounded_wall_clock() -> None:
    client = FixtureToolClient(min_delay_s=0.4, timeout_s=0.05)
    started = time.monotonic()
    with pytest.raises(ToolTimeout):
        client.invoke("query_knowledge", {"question": "Harbor trust labels"})
    elapsed = time.monotonic() - started
    # Slack permits ordinary CI scheduling jitter but not waiting for 0.4s worker.
    assert elapsed < 0.25, f"timeout=0.05s returned after {elapsed:.3f}s"


@pytest.mark.xfail(strict=True, reason="#16: the FastMCP manager can still register a write tool")
def test_mcp_registry_cannot_gain_write_capability() -> None:
    server = build_harbor_mcp()

    def write_index() -> str:
        return "test-only write marker"

    manager = getattr(server, "_tool_manager", None)
    if manager is not None and hasattr(manager, "add_tool"):
        try:
            manager.add_tool(write_index)
        except Exception:
            # Denial at the manager boundary is acceptable. Listing a new tool is not.
            pass

    listed = asyncio.run(server.list_tools())
    assert {tool.name for tool in listed} == set(ALLOWED_TOOLS)


@pytest.mark.parametrize("method", ["hashed_vector", "graph_expand"])
@pytest.mark.xfail(strict=True, reason="#15: retrieval method needs a meaningful no-evidence gate")
def test_shootout_insufficient_case_abstains_for_every_method(method: str) -> None:
    payload = run_shootout()
    case = next(row for row in payload["rows"][method] if row["id"] == "g-insufficient")
    assert case["expected_abstain"] is True
    assert case["retrieved"] == []
    assert case["abstained"] is True
