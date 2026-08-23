"""Typed contracts for the framework-free research pipeline."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NonBlankStr = Annotated[str, Field(min_length=1)]

TrustProfile = Literal["durable_knowledge", "project_status"]
Intent = Literal["knowledge", "project", "federated"]
Outcome = Literal["answered", "abstained"]
CitationClass = Literal["citable", "weak_fit", "excluded"]

KNOWLEDGE_SCOPE: tuple[TrustProfile, ...] = ("durable_knowledge",)
PROJECT_SCOPE: tuple[TrustProfile, ...] = ("project_status",)
FEDERATED_SCOPE: tuple[TrustProfile, ...] = ("durable_knowledge", "project_status")

INTENT_SCOPES: dict[Intent, tuple[TrustProfile, ...]] = {
    "knowledge": KNOWLEDGE_SCOPE,
    "project": PROJECT_SCOPE,
    "federated": FEDERATED_SCOPE,
}


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResearchRequest(FrozenModel):
    request_id: NonBlankStr
    question: NonBlankStr

    @field_validator("request_id", "question")
    @classmethod
    def strip_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("field must not be blank")
        return stripped


class RouteDecision(FrozenModel):
    intent: Intent
    scopes: tuple[TrustProfile, ...]
    reason: NonBlankStr

    @model_validator(mode="after")
    def scopes_match_intent(self) -> Self:
        expected = INTENT_SCOPES[self.intent]
        if self.scopes != expected:
            raise ValueError("scopes must match intent")
        return self


class EvidenceItem(FrozenModel):
    source_path: NonBlankStr
    trust_profile: TrustProfile
    title: NonBlankStr
    snippet: NonBlankStr
    score: float
    weak_fit: bool
    citation_class: CitationClass

    @model_validator(mode="after")
    def weak_fit_not_citable(self) -> Self:
        if self.weak_fit and self.citation_class == "citable":
            raise ValueError("weak-fit evidence cannot be citable")
        if not self.weak_fit and self.citation_class == "weak_fit":
            raise ValueError("non-weak evidence cannot be classed as weak_fit")
        if self.citation_class == "excluded" and (self.weak_fit or self.score > 0):
            raise ValueError("excluded items must be non-matching")
        return self


class EvidenceBundle(FrozenModel):
    knowledge: tuple[EvidenceItem, ...] = ()
    project: tuple[EvidenceItem, ...] = ()

    @model_validator(mode="after")
    def preserve_trust_split(self) -> Self:
        if any(item.trust_profile != "durable_knowledge" for item in self.knowledge):
            raise ValueError("knowledge group must only contain durable_knowledge")
        if any(item.trust_profile != "project_status" for item in self.project):
            raise ValueError("project group must only contain project_status")
        return self

    def citable(self) -> tuple[EvidenceItem, ...]:
        return tuple(
            item
            for item in (*self.knowledge, *self.project)
            if item.citation_class == "citable" and not item.weak_fit
        )


class Citation(FrozenModel):
    source_path: NonBlankStr
    trust_profile: TrustProfile
    title: NonBlankStr


class Answer(FrozenModel):
    outcome: Outcome
    text: NonBlankStr
    citations: tuple[Citation, ...] = ()
    abstain_reason: str | None = None

    @model_validator(mode="after")
    def citations_required_when_answered(self) -> Self:
        if self.outcome == "answered" and not self.citations:
            raise ValueError("answered outcomes require at least one citation")
        if self.outcome == "abstained" and not self.abstain_reason:
            raise ValueError("abstained outcomes require a reason")
        if self.outcome == "abstained" and self.citations:
            raise ValueError("abstained outcomes must not carry citations")
        return self


class Receipt(FrozenModel):
    request_id: NonBlankStr
    intent: Intent
    scopes: tuple[TrustProfile, ...]
    outcome: Outcome
    citation_paths: tuple[str, ...] = ()
    trust_profiles_used: tuple[TrustProfile, ...] = ()
    abstain_reason: str | None = None
    tools_used: tuple[str, ...] = ()
    tool_errors: tuple[str, ...] = ()
    content_redacted: Literal[True] = True
    receipt_hash: NonBlankStr


class ResearchResult(FrozenModel):
    request: ResearchRequest
    route: RouteDecision
    evidence: EvidenceBundle
    answer: Answer
    receipt: Receipt
