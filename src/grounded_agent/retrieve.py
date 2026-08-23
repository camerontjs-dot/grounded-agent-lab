"""Fixture retrievers that keep durable knowledge and project status apart."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from grounded_agent.models import (
    EvidenceBundle,
    EvidenceItem,
    ResearchRequest,
    RouteDecision,
    TrustProfile,
)
from grounded_agent.paths import KNOWLEDGE_CORPUS, PROJECT_CORPUS

CITABLE_THRESHOLD = 0.2
MIN_CITABLE_TOKENS = 2
STOPWORDS = {
    "the",
    "and",
    "for",
    "what",
    "when",
    "must",
    "does",
    "how",
    "from",
    "with",
    "that",
    "this",
    "are",
    "is",
    "its",
    "do",
}


@dataclass(frozen=True)
class FixtureNote:
    source_path: str
    trust_profile: TrustProfile
    title: str
    body: str


def _tokens(text: str) -> set[str]:
    buffered: list[str] = []
    for char in text.lower():
        buffered.append(char if char.isalnum() else " ")
    return {token for token in "".join(buffered).split() if len(token) > 2}


def overlap(question: str, body: str) -> tuple[float, int]:
    query_tokens = _tokens(question) - STOPWORDS
    if not query_tokens:
        return 0.0, 0
    shared = query_tokens & (_tokens(body) - STOPWORDS)
    return len(shared) / len(query_tokens), len(shared)


def overlap_score(question: str, body: str) -> float:
    score, _shared = overlap(question, body)
    return score


def parse_note(path: Path, expected_trust: TrustProfile) -> FixtureNote:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"fixture note missing frontmatter: {path.name}")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"fixture note frontmatter is incomplete: {path.name}")
    meta: dict[str, str] = {}
    for line in parts[1].strip().splitlines():
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    trust = meta.get("trust_profile")
    if trust != expected_trust:
        raise ValueError(f"{path.name} trust_profile {trust!r} != {expected_trust!r}")
    title = meta.get("title")
    if not title:
        raise ValueError(f"{path.name} missing title")
    relative = f"{path.parent.name}/{path.name}"
    return FixtureNote(
        source_path=relative,
        trust_profile=expected_trust,
        title=title,
        body=parts[2].strip(),
    )


def load_corpus(directory: Path, expected_trust: TrustProfile) -> tuple[FixtureNote, ...]:
    notes = tuple(
        parse_note(path, expected_trust)
        for path in sorted(directory.glob("*.md"))
    )
    if not notes:
        raise ValueError(f"no fixture notes in {directory.name}")
    return notes


def _to_item(note: FixtureNote, score: float, shared_tokens: int) -> EvidenceItem:
    if score >= CITABLE_THRESHOLD and shared_tokens >= MIN_CITABLE_TOKENS:
        citation_class = "citable"
        weak_fit = False
    elif shared_tokens > 0:
        citation_class = "weak_fit"
        weak_fit = True
    else:
        citation_class = "excluded"
        weak_fit = False
    snippet = " ".join(note.body.split())
    return EvidenceItem(
        source_path=note.source_path,
        trust_profile=note.trust_profile,
        title=note.title,
        snippet=snippet,
        score=round(score, 4),
        weak_fit=weak_fit,
        citation_class=citation_class,
    )


def _search(notes: tuple[FixtureNote, ...], question: str) -> tuple[EvidenceItem, ...]:
    scored: list[EvidenceItem] = []
    for note in notes:
        score, shared = overlap(question, f"{note.title}\n{note.body}")
        scored.append(_to_item(note, score, shared))
    ranked = sorted(scored, key=lambda item: item.score, reverse=True)
    return tuple(item for item in ranked if item.citation_class != "excluded")


def retrieve(request: ResearchRequest, route: RouteDecision) -> EvidenceBundle:
    knowledge_items: tuple[EvidenceItem, ...] = ()
    project_items: tuple[EvidenceItem, ...] = ()
    if "durable_knowledge" in route.scopes:
        knowledge_items = _search(
            load_corpus(KNOWLEDGE_CORPUS, "durable_knowledge"),
            request.question,
        )
    if "project_status" in route.scopes:
        project_items = _search(
            load_corpus(PROJECT_CORPUS, "project_status"),
            request.question,
        )
    return EvidenceBundle(knowledge=knowledge_items, project=project_items)
