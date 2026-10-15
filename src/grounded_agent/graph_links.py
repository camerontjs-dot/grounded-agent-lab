"""One-hop expansion over [[wiki]] links in the Harbor fixture corpus."""

from __future__ import annotations

import re

from grounded_agent.retrieve import FixtureNote

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


def neighbour_map(notes: tuple[FixtureNote, ...]) -> dict[str, tuple[str, ...]]:
    """One-hop [[wiki]] neighbours for every note, resolved within `notes`."""

    return {note.source_path: neighbors(note, notes) for note in notes}


def expand_from_seeds(
    seeds: tuple[tuple[str, float], ...],
    neighbours: dict[str, tuple[str, ...]],
    *,
    k: int,
) -> tuple[tuple[str, float], ...]:
    """Seeds plus their one-hop neighbours, re-ranked by score.

    A neighbour scores 0.85 of the seed that reached it. Callers decide which
    seeds count; with no seeds the result is empty.
    """

    scores: dict[str, float] = {}
    for path, score in seeds:
        scores[path] = max(scores.get(path, 0.0), score)
        for neighbour in neighbours.get(path, ()):
            hop = score * 0.85 if score > 0 else 0.15
            scores[neighbour] = max(scores.get(neighbour, 0.0), hop)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return tuple(ranked[:k])
