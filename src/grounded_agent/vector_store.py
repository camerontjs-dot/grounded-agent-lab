"""In-process hashed-token vector store.

This is a local vector index, not a neural embedding model and not Pinecone.
Hashes are sha256-derived so ranking is stable across processes.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from grounded_agent.graph_links import strip_wiki_links
from grounded_agent.retrieve import FixtureNote, _tokens

VECTOR_DIM = 64


def _stable_bucket(token: str) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % VECTOR_DIM


def embed_text(text: str) -> tuple[float, ...]:
    counts = [0.0] * VECTOR_DIM
    for token in _tokens(text):
        counts[_stable_bucket(token)] += 1.0
    norm = math.sqrt(sum(value * value for value in counts))
    if norm == 0:
        return tuple(counts)
    return tuple(value / norm for value in counts)


def cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


@dataclass(frozen=True)
class HashVectorStore:
    vectors: tuple[tuple[str, tuple[float, ...]], ...]

    @classmethod
    def build(cls, notes: tuple[FixtureNote, ...]) -> HashVectorStore:
        indexed = tuple(
            (
                note.source_path,
                embed_text(strip_wiki_links(f"{note.title}\n{note.body}")),
            )
            for note in notes
        )
        return cls(vectors=indexed)

    def query(self, question: str, k: int) -> tuple[tuple[str, float], ...]:
        query_vec = embed_text(question)
        ranked = sorted(
            ((path, cosine(query_vec, vector)) for path, vector in self.vectors),
            key=lambda item: item[1],
            reverse=True,
        )
        return tuple(ranked[:k])
