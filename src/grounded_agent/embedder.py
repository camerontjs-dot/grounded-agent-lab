"""Optional neural embedder. Fail closed when sentence-transformers is missing."""

from __future__ import annotations


class NeuralEmbedderUnavailable(Exception):
    """sentence-transformers is not installed; neural method must not be faked."""


def encode_neural(texts: list[str]) -> list[list[float]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise NeuralEmbedderUnavailable(
            "sentence-transformers is not installed; pass only if extras [embeddings] are present"
        ) from exc
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(texts, convert_to_numpy=True)
    return [vector.tolist() for vector in vectors]
