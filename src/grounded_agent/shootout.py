"""Retrieval shootout runner. Writes JSON and Markdown from a live run.

Each method has an index step and a query step. The index step runs outside
the timer, and `latency_ms` is the median query time over `REPEATS` runs. Each
method has its own evidence gate on its own scale:

- lexical: query-token recall >= CITABLE_THRESHOLD and at least
  MIN_CITABLE_TOKENS shared tokens, the agent's own citable rule;
- hashed_vector: cosine strictly above a null cutoff from
  `grounded_agent.calibration`, derived without gold labels;
- graph_expand: lexical seeds that pass the lexical gate, plus their one-hop
  [[wiki]] neighbours. No passing seed means abstain.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grounded_agent.calibration import (
    NULL_PERCENTILE,
    PROTOCOL,
    NullCutoff,
    null_cutoff,
    query_token_count,
)
from grounded_agent.graph_links import expand_from_seeds, neighbour_map, strip_wiki_links
from grounded_agent.paths import GOLD_PATH
from grounded_agent.retrieve import (
    CITABLE_THRESHOLD,
    MIN_CITABLE_TOKENS,
    STOPWORDS,
    FixtureNote,
    _tokens,
    notes_for_scopes,
)
from grounded_agent.vector_store import HashVectorStore

SCHEMA = "grounded-agent-shootout-v2"
TOP_K = 3
METHODS = ("lexical", "hashed_vector", "graph_expand")
REPEATS = 25
PREVIOUS_REPORT = "reports/history/2026-08-23-retrieval-shootout.json"
GATES = {
    "lexical": (
        f"query-token recall >= {CITABLE_THRESHOLD} and >= {MIN_CITABLE_TOKENS} shared tokens"
    ),
    "hashed_vector": (
        f"cosine above the {round(NULL_PERCENTILE * 100)}th percentile of a single-shared-token "
        f"null ({PROTOCOL})"
    ),
    "graph_expand": "lexical seeds that pass the lexical gate, plus one-hop [[wiki]] neighbours",
}
PREVIOUS_TABLE_NOTES = (
    "hashed_vector was gated by the lexical recall cutoff (0.2) applied to raw cosine, with no "
    "two-token minimum. Its 0/1 abstain measured that scale mismatch, not retrieval quality.",
    "graph_expand had no gate and always returned up to top_k notes, so its 0/1 abstain could "
    "not have been anything else.",
    "No latency was query-only. Every method redid per-note work for every question: "
    "lexical and graph_expand tokenized each note, and hashed_vector rebuilt its index, the "
    "most expensive of the three. Query-only, hashed_vector is still the slowest here.",
)

Ranked = tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class _TokenIndex:
    """Visible note tokens minus stopwords, as `retrieve.overlap` sees them."""

    docs: tuple[tuple[str, frozenset[str]], ...]


@dataclass(frozen=True)
class _GraphIndex:
    tokens: _TokenIndex
    neighbours: dict[str, tuple[str, ...]]


def load_gold(path: Path = GOLD_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _visible(note: FixtureNote) -> str:
    return strip_wiki_links(f"{note.title}\n{note.body}")


def _token_index(notes: tuple[FixtureNote, ...]) -> _TokenIndex:
    return _TokenIndex(
        docs=tuple(
            (note.source_path, frozenset(_tokens(_visible(note)) - STOPWORDS)) for note in notes
        )
    )


def _lexical_scores(question: str, index: _TokenIndex) -> list[tuple[str, float, int]]:
    query = _tokens(question) - STOPWORDS
    scored: list[tuple[str, float, int]] = []
    for path, tokens in index.docs:
        shared = len(query & tokens) if query else 0
        scored.append((path, shared / len(query) if query else 0.0, shared))
    return sorted(scored, key=lambda item: item[1], reverse=True)


def _passes_lexical_gate(score: float, shared: int) -> bool:
    return score >= CITABLE_THRESHOLD and shared >= MIN_CITABLE_TOKENS


def _lexical_rank(question: str, index: _TokenIndex, k: int, _cutoff: NullCutoff) -> Ranked:
    passing = [
        (path, score)
        for path, score, shared in _lexical_scores(question, index)
        if _passes_lexical_gate(score, shared)
    ]
    return tuple(passing[:k])


def _vector_rank(question: str, index: HashVectorStore, k: int, cutoff: NullCutoff) -> Ranked:
    if cutoff.cutoff is None:
        return ()
    above = [
        (path, score)
        for path, score in index.query(question, len(index.vectors))
        if score > cutoff.cutoff
    ]
    return tuple(above[:k])


def _graph_rank(question: str, index: _GraphIndex, k: int, _cutoff: NullCutoff) -> Ranked:
    seeds = tuple(
        (path, score)
        for path, score, shared in _lexical_scores(question, index.tokens)[:k]
        if _passes_lexical_gate(score, shared)
    )
    return expand_from_seeds(seeds, index.neighbours, k=k)


def _graph_index(notes: tuple[FixtureNote, ...]) -> _GraphIndex:
    return _GraphIndex(tokens=_token_index(notes), neighbours=neighbour_map(notes))


INDEXERS: dict[str, Callable[[tuple[FixtureNote, ...]], Any]] = {
    "lexical": _token_index,
    "hashed_vector": HashVectorStore.build,
    "graph_expand": _graph_index,
}
RANKERS: dict[str, Callable[[str, Any, int, NullCutoff], Ranked]] = {
    "lexical": _lexical_rank,
    "hashed_vector": _vector_rank,
    "graph_expand": _graph_rank,
}


def _median_ms(action: Callable[..., object], repeats: int, *args: Any) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        action(*args)
        samples.append((time.perf_counter() - started) * 1000)
    return round(statistics.median(samples), 4)


def _prf(retrieved: list[str], relevant: list[str]) -> tuple[float, float]:
    if not relevant:
        return (1.0, 1.0) if not retrieved else (0.0, 0.0)
    hit = set(retrieved) & set(relevant)
    recall = len(hit) / len(relevant)
    precision = len(hit) / len(retrieved) if retrieved else 0.0
    return recall, precision


def run_shootout(
    *, gold_path: Path = GOLD_PATH, top_k: int = TOP_K, repeats: int = REPEATS
) -> dict[str, Any]:
    gold = load_gold(gold_path)
    method_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in METHODS}
    build_ms: dict[str, list[float]] = {name: [] for name in METHODS}
    calibration_ms: list[float] = []
    for case in gold:
        question = str(case["question"])
        notes = notes_for_scopes(tuple(case["scopes"]))
        relevant = list(case["relevant_paths"])
        started = time.perf_counter()
        cutoff = null_cutoff(notes, query_token_count(question))
        calibration_ms.append(round((time.perf_counter() - started) * 1000, 4))
        for method in METHODS:
            build = INDEXERS[method]
            index = build(notes)
            build_ms[method].append(_median_ms(build, repeats, notes))
            rank = RANKERS[method]
            ranked = rank(question, index, top_k, cutoff)
            latency = _median_ms(rank, repeats, question, index, top_k, cutoff)
            retrieved = [path for path, _score in ranked]
            recall, precision = _prf(retrieved, relevant)
            row: dict[str, Any] = {
                "id": case["id"],
                "retrieved": retrieved,
                "relevant": relevant,
                "recall": recall,
                "precision": precision,
                "abstained": len(retrieved) == 0,
                "expected_abstain": case["expected_abstain"],
                "latency_ms": latency,
                "miss": recall < 1.0 and not case["expected_abstain"],
                "false_hits_on_abstain": case["expected_abstain"] and len(retrieved) > 0,
            }
            if method == "hashed_vector":
                row["query_tokens"] = cutoff.query_tokens
                row["null_scores"] = cutoff.null_scores
                row["cutoff"] = cutoff.cutoff
            method_rows[method].append(row)
    metrics = {}
    inspected: list[dict[str, Any]] = []
    for method, rows in method_rows.items():
        n = len(rows)
        metrics[method] = {
            "n": n,
            "macro_recall": round(sum(row["recall"] for row in rows) / n, 4),
            "macro_precision": round(sum(row["precision"] for row in rows) / n, 4),
            "abstain_ok": sum(
                1 for row in rows if row["expected_abstain"] and row["abstained"]
            ),
            "abstain_n": sum(1 for row in rows if row["expected_abstain"]),
            "mean_latency_ms": round(sum(row["latency_ms"] for row in rows) / n, 4),
        }
        for row in rows:
            if row["miss"] or row["false_hits_on_abstain"]:
                inspected.append({"method": method, **row})
    return {
        "schema": SCHEMA,
        "corpus": "grounded_agent/fixtures/corpus",
        "gold": "grounded_agent/fixtures/gold.json",
        "top_k": top_k,
        "methods": list(METHODS),
        "gates": dict(GATES),
        "timing": {
            "repeats": repeats,
            "latency_ms": "median query time per question; the index is built outside the timer",
            "mean_index_build_ms": {
                method: round(sum(values) / len(values), 4) for method, values in build_ms.items()
            },
            "mean_calibration_ms": round(sum(calibration_ms) / len(calibration_ms), 4),
        },
        "metrics": metrics,
        "rows": method_rows,
        "inspected_failures": inspected,
        "previous_report": {"path": PREVIOUS_REPORT, "notes": list(PREVIOUS_TABLE_NOTES)},
        "limitations": [
            "n is the Harbor fixture set, not a public IR benchmark.",
            "hashed_vector is a sha256 bag-of-tokens index, not a neural embedding API.",
            "graph_expand is one-hop [[wiki]] following, not Microsoft GraphRAG.",
            "Each gate is defined on its own method's scale; abstain counts compare gates, "
            "not one shared threshold.",
            "Latencies are microsecond-scale medians on one machine; they order methods "
            "loosely at best.",
            "Do not generalize these numbers to production corpora.",
        ],
    }


def _paths(values: list[str]) -> str:
    if not values:
        return "(none)"
    return ", ".join(f"`{path}`" for path in values)


def render_report(payload: dict[str, Any]) -> str:
    timing = payload["timing"]
    lines = [
        "# Retrieval shootout on the Harbor fixture",
        "",
        "This file is generated by `grounded-agent shootout`.",
        "The numbers come from the JSON receipt beside it.",
        "n=5 Harbor notes. That is the whole experiment.",
        "The five are labelled questions; the corpus behind them has eight notes.",
        "",
        f"top_k={payload['top_k']}. Methods: {', '.join(payload['methods'])}.",
        "",
        "| method | n | macro recall | macro precision | abstain ok "
        "| mean query ms | mean index build ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in payload["metrics"].items():
        lines.append(
            f"| {method} | {row['n']} | {row['macro_recall']} | {row['macro_precision']} | "
            f"{row['abstain_ok']}/{row['abstain_n']} | {row['mean_latency_ms']} | "
            f"{timing['mean_index_build_ms'][method]} |"
        )
    lines.extend(["", "## Gates", ""])
    for method, gate in payload["gates"].items():
        lines.append(f"- `{method}`: {gate}.")
    lines.extend(
        [
            "",
            "Query times are medians over "
            f"{timing['repeats']} runs with the index already built. The hashed_vector cutoff "
            "is computed once per question before timing "
            f"(mean {timing['mean_calibration_ms']} ms) and is not in either column.",
            "",
            "| question | query tokens | null scores | hashed_vector cutoff |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]["hashed_vector"]:
        lines.append(
            f"| `{row['id']}` | {row['query_tokens']} | {row['null_scores']} | {row['cutoff']} |"
        )
    lines.extend(["", "## Inspected failures", ""])
    if not payload["inspected_failures"]:
        lines.append("None recorded.")
    for failure in payload["inspected_failures"]:
        lines.append(
            f"- `{failure['method']}` / `{failure['id']}`: got {_paths(failure['retrieved'])}; "
            f"gold was {_paths(failure['relevant'])}"
        )
    previous = payload["previous_report"]
    lines.extend(
        [
            "",
            "## What the 2026-08-23 table measured",
            "",
            f"The earlier table is kept unchanged at `{previous['path']}` and its Markdown twin.",
            "",
        ]
    )
    for note in previous["notes"]:
        lines.append(f"- {note}")
    lines.extend(["", "## Limitations", ""])
    for item in payload["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_shootout(out_dir: Path | str = "reports") -> dict[str, Any]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload = run_shootout()
    json_path = target / "retrieval-shootout.json"
    md_path = target / "retrieval-shootout.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_report(payload), encoding="utf-8")
    return payload
