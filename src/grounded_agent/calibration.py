"""Null calibration for the hashed-token vector gate (#15).

`hashed_vector` scores notes by cosine on a 64-bucket hashed bag of tokens. That
score is not on the lexical scale (query-token recall), so the lexical 0.2 cutoff
means nothing there. This module derives a cosine cutoff from the corpus itself,
without reading gold labels:

- The lexical rule treats one shared query token as weak fit, not evidence
  (`MIN_CITABLE_TOKENS = 2`). The null here is the vector version of that case: a
  pseudo-query with the same token count as the real query that shares exactly
  one token with the target note. Its other tokens come from the other notes in
  the same scope, minus every token of the target note, so any extra similarity
  is hash collision.
- The cutoff is the 95th percentile (nearest rank) of those null cosines. A note
  is admissible only if its cosine is strictly above the cutoff.
- With fewer than `MIN_NULL_SCORES` null scores the cutoff is undefined and the
  method abstains.

The protocol was fixed on 2026-10-15 before the republished shootout was run. It
uses note text and the query's token count. It never reads the gold labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from grounded_agent.graph_links import strip_wiki_links
from grounded_agent.retrieve import FixtureNote, _tokens
from grounded_agent.vector_store import cosine, embed_text

PROTOCOL = "hashed-vector-single-token-null-v1"
NULL_PERCENTILE = 0.95
MIN_NULL_SCORES = 20


@dataclass(frozen=True)
class NullCutoff:
    """Cosine cutoff for one scope's notes and one query length."""

    query_tokens: int
    null_scores: int
    cutoff: float | None


def _visible_tokens(note: FixtureNote) -> list[str]:
    return sorted(_tokens(strip_wiki_links(f"{note.title}\n{note.body}")))


def null_scores(notes: tuple[FixtureNote, ...], query_tokens: int) -> list[float]:
    """Cosines of single-shared-token pseudo-queries against each note."""

    if query_tokens < 1:
        return []
    width = query_tokens - 1
    tokens = {note.source_path: _visible_tokens(note) for note in notes}
    scores: list[float] = []
    for target in notes:
        target_tokens = tokens[target.source_path]
        excluded = set(target_tokens)
        target_vector = embed_text(" ".join(target_tokens))
        chunks: list[list[str]] = [[]] if width == 0 else []
        for donor in notes:
            if width == 0 or donor.source_path == target.source_path:
                continue
            disjoint = [token for token in tokens[donor.source_path] if token not in excluded]
            chunks.extend(
                disjoint[start : start + width]
                for start in range(0, len(disjoint) - width + 1, width)
            )
        for chunk in chunks:
            for shared in target_tokens:
                pseudo_query = embed_text(" ".join([*chunk, shared]))
                scores.append(cosine(pseudo_query, target_vector))
    return scores


def null_cutoff(notes: tuple[FixtureNote, ...], query_tokens: int) -> NullCutoff:
    scores = sorted(null_scores(notes, query_tokens))
    if len(scores) < MIN_NULL_SCORES:
        return NullCutoff(query_tokens=query_tokens, null_scores=len(scores), cutoff=None)
    rank = math.ceil(NULL_PERCENTILE * len(scores))
    return NullCutoff(
        query_tokens=query_tokens,
        null_scores=len(scores),
        cutoff=round(scores[rank - 1], 6),
    )


def query_token_count(question: str) -> int:
    """Token count on the vector index's own tokenizer (no stopword removal)."""

    return len(_tokens(question))
