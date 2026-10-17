"""Fixture retrievers that keep durable knowledge and project status apart."""

from __future__ import annotations

import hashlib
import json
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
    # Set by parse_note from the file's bytes. Notes built in memory have none and
    # cannot enter a corpus digest or a receipt citation.
    source_sha256: str | None = None


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
    raw = path.read_bytes()
    text = raw.decode("utf-8")
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
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


def load_all_notes() -> tuple[FixtureNote, ...]:
    return load_corpus(KNOWLEDGE_CORPUS, "durable_knowledge") + load_corpus(
        PROJECT_CORPUS, "project_status"
    )


def notes_for_scopes(scopes: tuple[TrustProfile, ...]) -> tuple[FixtureNote, ...]:
    return tuple(note for note in load_all_notes() if note.trust_profile in scopes)


def load_corpus(directory: Path, expected_trust: TrustProfile) -> tuple[FixtureNote, ...]:
    notes = tuple(
        parse_note(path, expected_trust)
        for path in sorted(directory.glob("*.md"))
    )
    if not notes:
        raise ValueError(f"no fixture notes in {directory.name}")
    return notes


def corpus_digest(notes: tuple[FixtureNote, ...]) -> str:
    """Manifest digest: which notes were searched, and the bytes behind each one."""

    missing = [note.source_path for note in notes if note.source_sha256 is None]
    if missing:
        raise ValueError(f"cannot digest a corpus with unhashed notes: {missing}")
    manifest = sorted([note.source_path, note.source_sha256] for note in notes)
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
        source_sha256=note.source_sha256,
    )


def _search(notes: tuple[FixtureNote, ...], question: str) -> tuple[EvidenceItem, ...]:
    scored: list[EvidenceItem] = []
    for note in notes:
        score, shared = overlap(question, f"{note.title}\n{note.body}")
        scored.append(_to_item(note, score, shared))
    ranked = sorted(scored, key=lambda item: item.score, reverse=True)
    return tuple(item for item in ranked if item.citation_class != "excluded")


def search_corpus_with_digest(
    question: str, trust_profile: TrustProfile
) -> tuple[tuple[EvidenceItem, ...], str]:
    """Search one corpus and return the manifest digest of the notes it searched."""

    directory = KNOWLEDGE_CORPUS if trust_profile == "durable_knowledge" else PROJECT_CORPUS
    notes = load_corpus(directory, trust_profile)
    return _search(notes, question), corpus_digest(notes)


def search_corpus(question: str, trust_profile: TrustProfile) -> tuple[EvidenceItem, ...]:
    items, _digest = search_corpus_with_digest(question, trust_profile)
    return items


def retrieve(request: ResearchRequest, route: RouteDecision) -> EvidenceBundle:
    knowledge_items: tuple[EvidenceItem, ...] = ()
    project_items: tuple[EvidenceItem, ...] = ()
    if "durable_knowledge" in route.scopes:
        knowledge_items = search_corpus(request.question, "durable_knowledge")
    if "project_status" in route.scopes:
        project_items = search_corpus(request.question, "project_status")
    return EvidenceBundle(knowledge=knowledge_items, project=project_items)
