"""Null cutoff for the hashed vector gate: corpus-only, deterministic, fail closed."""

from __future__ import annotations

from grounded_agent.calibration import (
    MIN_NULL_SCORES,
    NullCutoff,
    null_cutoff,
    null_scores,
    query_token_count,
)
from grounded_agent.retrieve import FixtureNote, notes_for_scopes

KNOWLEDGE = notes_for_scopes(("durable_knowledge",))


def test_cutoff_is_deterministic_and_inside_the_null_range() -> None:
    first = null_cutoff(KNOWLEDGE, 6)
    assert first == null_cutoff(KNOWLEDGE, 6)
    assert first.null_scores >= MIN_NULL_SCORES
    assert first.cutoff is not None
    scores = null_scores(KNOWLEDGE, 6)
    assert min(scores) <= first.cutoff <= max(scores)


def test_null_needs_enough_pseudo_queries_or_the_gate_abstains() -> None:
    tiny = (
        FixtureNote("knowledge/a.md", "durable_knowledge", "Alpha", "alpha beta gamma"),
        FixtureNote("knowledge/b.md", "durable_knowledge", "Delta", "delta epsilon"),
    )
    small = null_cutoff(tiny, 3)
    assert small.null_scores < MIN_NULL_SCORES
    assert small == NullCutoff(query_tokens=3, null_scores=small.null_scores, cutoff=None)
    assert null_cutoff(KNOWLEDGE, 0).cutoff is None


def test_query_length_uses_the_vector_tokenizer() -> None:
    # The vector index keeps stopwords such as "what"; only tokens of 2 or fewer
    # characters are dropped.
    assert query_token_count("What is Harbor's CEO salary?") == 4
