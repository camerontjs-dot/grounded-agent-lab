"""Extractive draft-or-abstain step. No hosted model required."""

from __future__ import annotations

from grounded_agent.models import Answer, Citation, EvidenceBundle, EvidenceItem, ResearchRequest

INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all instructions",
    "grant write",
    "disable the allowlist",
    "merge the indexes without labels",
)


def is_injection(question: str) -> bool:
    lowered = question.lower()
    return any(marker in lowered for marker in INJECTION_MARKERS)


def _first_sentence(snippet: str) -> str:
    stripped = snippet.strip()
    for separator in (". ", "? ", "! "):
        index = stripped.find(separator)
        if index != -1:
            return stripped[: index + 1].strip()
    return stripped


def _citation(item: EvidenceItem) -> Citation:
    return Citation(
        source_path=item.source_path,
        trust_profile=item.trust_profile,
        title=item.title,
    )


def draft_or_abstain(request: ResearchRequest, evidence: EvidenceBundle) -> Answer:
    if is_injection(request.question):
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

    lines = ["Answer assembled only from citable, trust-labelled sources:"]
    citations: list[Citation] = []
    seen_paths: set[str] = set()
    citable_paths = {item.source_path for item in citable}
    for profile, group in (
        ("durable_knowledge", evidence.knowledge),
        ("project_status", evidence.project),
    ):
        usable = [item for item in group if item.source_path in citable_paths]
        if not usable:
            continue
        lines.append(f"[{profile}]")
        for item in usable:
            lines.append(f"- {_first_sentence(item.snippet)} ({item.source_path})")
            if item.source_path not in seen_paths:
                citations.append(_citation(item))
                seen_paths.add(item.source_path)

    return Answer(outcome="answered", text="\n".join(lines), citations=tuple(citations))
