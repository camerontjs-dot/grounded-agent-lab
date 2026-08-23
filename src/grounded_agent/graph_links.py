"""One-hop expansion over [[wiki]] links in the Harbor fixture corpus."""

from __future__ import annotations

import re

from grounded_agent.retrieve import FixtureNote, overlap_score

LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def parse_links(body: str) -> tuple[str, ...]:
    return tuple(LINK_RE.findall(body))


def strip_wiki_links(text: str) -> str:
    return LINK_RE.sub(" ", text)


def resolve_link(slug: str, notes: tuple[FixtureNote, ...]) -> str | None:
    needle = slug.strip().lower()
    for note in notes:
        stem = note.source_path.rsplit("/", 1)[-1].removesuffix(".md")
        if stem.lower() == needle:
            return note.source_path
    return None


def neighbors(note: FixtureNote, notes: tuple[FixtureNote, ...]) -> tuple[str, ...]:
    found: list[str] = []
    for slug in parse_links(note.body):
        path = resolve_link(slug, notes)
        if path and path not in found:
            found.append(path)
    return tuple(found)


def graph_expand_rank(
    question: str,
    notes: tuple[FixtureNote, ...],
    *,
    k: int,
    seed_k: int | None = None,
) -> tuple[tuple[str, float], ...]:
    """Lexical seeds plus one-hop neighbors, re-ranked by score."""

    by_path = {note.source_path: note for note in notes}
    lexical = sorted(
        (
            (
                note.source_path,
                overlap_score(question, strip_wiki_links(f"{note.title}\n{note.body}")),
            )
            for note in notes
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    scores: dict[str, float] = {}
    seeds = lexical[: seed_k if seed_k is not None else k]
    for path, score in seeds:
        scores[path] = max(scores.get(path, 0.0), score)
        seed = by_path[path]
        for neighbor in neighbors(seed, notes):
            if neighbor not in by_path:
                continue
            hop = score * 0.85 if score > 0 else 0.15
            scores[neighbor] = max(scores.get(neighbor, 0.0), hop)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return tuple(ranked[:k])
