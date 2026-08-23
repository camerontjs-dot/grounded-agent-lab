"""Schema construction and trust-split invariants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from grounded_agent.models import (
    Answer,
    Citation,
    EvidenceBundle,
    EvidenceItem,
    Receipt,
    ResearchRequest,
    RouteDecision,
)


def test_request_strips_and_rejects_blank() -> None:
    request = ResearchRequest(request_id=" r1 ", question=" What is a trust label? ")
    assert request.request_id == "r1"
    assert request.question == "What is a trust label?"
    with pytest.raises(ValidationError):
        ResearchRequest(request_id="r1", question="   ")


def test_route_scopes_must_match_intent() -> None:
    with pytest.raises(ValidationError):
        RouteDecision(
            intent="knowledge",
            scopes=("durable_knowledge", "project_status"),
            reason="bad blend",
        )


def test_evidence_bundle_rejects_cross_index_blend() -> None:
    project_item = EvidenceItem(
        source_path="projects/lantern-agent.md",
        trust_profile="project_status",
        title="Lantern",
        snippet="next action is the router",
        score=0.9,
        weak_fit=False,
        citation_class="citable",
    )
    with pytest.raises(ValidationError, match="knowledge group"):
        EvidenceBundle(knowledge=(project_item,))


def test_weak_fit_cannot_be_citable() -> None:
    with pytest.raises(ValidationError, match="weak-fit"):
        EvidenceItem(
            source_path="knowledge/trust-labels.md",
            trust_profile="durable_knowledge",
            title="Trust",
            snippet="labels required",
            score=0.1,
            weak_fit=True,
            citation_class="citable",
        )


def test_answered_requires_citation_and_abstain_forbids_it() -> None:
    citation = Citation(
        source_path="knowledge/trust-labels.md",
        trust_profile="durable_knowledge",
        title="Trust labels",
    )
    with pytest.raises(ValidationError, match="require at least one citation"):
        Answer(outcome="answered", text="Harbor requires labels.")
    with pytest.raises(ValidationError, match="must not carry citations"):
        Answer(
            outcome="abstained",
            text="no",
            abstain_reason="insufficient_evidence",
            citations=(citation,),
        )


def test_receipt_serialization_has_no_snippet_field() -> None:
    receipt = Receipt(
        request_id="r1",
        intent="knowledge",
        scopes=("durable_knowledge",),
        outcome="answered",
        citation_paths=("knowledge/trust-labels.md",),
        trust_profiles_used=("durable_knowledge",),
        content_redacted=True,
        receipt_hash="a" * 64,
    )
    dumped = receipt.model_dump()
    assert "snippet" not in dumped
    assert dumped["content_redacted"] is True
    assert receipt.model_dump_json().count("snippet") == 0
