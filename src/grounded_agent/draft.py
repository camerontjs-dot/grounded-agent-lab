"""Extractive draft-or-abstain step. No hosted model required."""

from __future__ import annotations

from grounded_agent.models import Answer, Citation, EvidenceBundle, EvidenceItem, ResearchRequest
from grounded_agent.redact import redact_text
from grounded_agent.support import (
    QuantityRequest,
    requested_quantity,
    split_sentences,
    states_requested_quantity,
)

INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all instructions",
    "grant write",
    "disable the allowlist",
    "merge the indexes without labels",
)


def is_injection(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in INJECTION_MARKERS)


def evidence_is_injection(evidence: EvidenceBundle) -> bool:
    for item in evidence.citable():
        if is_injection(item.title) or is_injection(item.snippet):
            return True
    return False


def _first_sentence(snippet: str) -> str:
    stripped = snippet.strip()
    for separator in (". ", "? ", "! "):
        index = stripped.find(separator)
        if index != -1:
            return stripped[: index + 1].strip()
    return stripped


def _supporting_sentence(item: EvidenceItem, request: QuantityRequest | None) -> str | None:
    """First sentence for open questions; the sentence stating the quantity otherwise."""

    if request is None:
        return _first_sentence(item.snippet)
    for sentence in split_sentences(item.snippet):
        if states_requested_quantity(sentence, request):
            return sentence
    return None


def _citation(item: EvidenceItem) -> Citation:
    return Citation(
        source_path=item.source_path,
        trust_profile=item.trust_profile,
        title=item.title,
    )


def draft_or_abstain(request: ResearchRequest, evidence: EvidenceBundle) -> Answer:
    if is_injection(request.question) or evidence_is_injection(evidence):
        return Answer(
            outcome="abstained",
            text="Refused: the question asks to bypass tool or trust boundaries.",
            abstain_reason="unsafe_instruction",
        )

    citable = evidence.citable()
    if not citable:
        return Answer(
            outcome="abstained",
            text="Insufficient citable evidence in the nominated sources.",
            abstain_reason="insufficient_evidence",
        )

    # A citable nomination is not an answer. A quantity question needs a cited
    # sentence that states the quantity; notes without one are not cited.
    requested = requested_quantity(request.question)
    lines = ["Answer assembled only from citable, trust-labelled sources:"]
    citations: list[Citation] = []
    seen_paths: set[str] = set()
    citable_paths = {item.source_path for item in citable}
    for profile, group in (
        ("durable_knowledge", evidence.knowledge),
        ("project_status", evidence.project),
    ):
        supported = [
            (item, sentence)
            for item in group
            if item.source_path in citable_paths
            and (sentence := _supporting_sentence(item, requested)) is not None
        ]
        if not supported:
            continue
        lines.append(f"[{profile}]")
        for item, sentence in supported:
            lines.append(f"- {redact_text(sentence)} ({item.source_path})")
            if item.source_path not in seen_paths:
                citations.append(_citation(item))
                seen_paths.add(item.source_path)

    if not citations:
        return Answer(
            outcome="abstained",
            text="The nominated notes do not state the quantity this question asks for.",
            abstain_reason="unsupported_claim",
        )
    return Answer(outcome="answered", text="\n".join(lines), citations=tuple(citations))
