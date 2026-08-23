"""Retrieval shootout runner. Writes JSON and Markdown from a live run."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from grounded_agent.graph_links import graph_expand_rank, strip_wiki_links
from grounded_agent.paths import GOLD_PATH, REPORTS_DIR
from grounded_agent.retrieve import (
    CITABLE_THRESHOLD,
    MIN_CITABLE_TOKENS,
    FixtureNote,
    notes_for_scopes,
    overlap,
)
from grounded_agent.vector_store import HashVectorStore

TOP_K = 3
METHODS = ("lexical", "hashed_vector", "graph_expand")

RankFn = Callable[[str, tuple[FixtureNote, ...], int], tuple[tuple[str, float], ...]]


def load_gold(path: Path = GOLD_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _visible(note: FixtureNote) -> str:
    return strip_wiki_links(f"{note.title}\n{note.body}")


def _lexical_rank(
    question: str, notes: tuple[FixtureNote, ...], k: int
) -> tuple[tuple[str, float], ...]:
    scored: list[tuple[str, float, int]] = []
    for note in notes:
        score, shared = overlap(question, _visible(note))
        scored.append((note.source_path, score, shared))
    ranked = sorted(scored, key=lambda item: item[1], reverse=True)
    citable = [
        (path, score)
        for path, score, shared in ranked
        if score >= CITABLE_THRESHOLD and shared >= MIN_CITABLE_TOKENS
    ]
    return tuple(citable[:k])


def _vector_rank(
    question: str, notes: tuple[FixtureNote, ...], k: int
) -> tuple[tuple[str, float], ...]:
    store = HashVectorStore.build(notes)
    return tuple(
        (path, score)
        for path, score in store.query(question, len(notes))
        if score >= CITABLE_THRESHOLD
    )[:k]


def _graph_rank(
    question: str, notes: tuple[FixtureNote, ...], k: int
) -> tuple[tuple[str, float], ...]:
    return graph_expand_rank(question, notes, k=k)


RANKERS: dict[str, RankFn] = {
    "lexical": _lexical_rank,
    "hashed_vector": _vector_rank,
    "graph_expand": _graph_rank,
}


def _prf(retrieved: list[str], relevant: list[str]) -> tuple[float, float]:
    if not relevant:
        return (1.0, 1.0) if not retrieved else (0.0, 0.0)
    hit = set(retrieved) & set(relevant)
    recall = len(hit) / len(relevant)
    precision = len(hit) / len(retrieved) if retrieved else 0.0
    return recall, precision


def run_shootout(*, gold_path: Path = GOLD_PATH, top_k: int = TOP_K) -> dict[str, Any]:
    gold = load_gold(gold_path)
    method_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in METHODS}
    for case in gold:
        scopes = tuple(case["scopes"])
        notes = notes_for_scopes(scopes)
        relevant = list(case["relevant_paths"])
        for method in METHODS:
            started = time.perf_counter()
            ranked = RANKERS[method](str(case["question"]), notes, top_k)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            retrieved = [path for path, _score in ranked]
            recall, precision = _prf(retrieved, relevant)
            abstained = len(retrieved) == 0
            method_rows[method].append(
                {
                    "id": case["id"],
                    "retrieved": retrieved,
                    "relevant": relevant,
                    "recall": recall,
                    "precision": precision,
                    "abstained": abstained,
                    "expected_abstain": case["expected_abstain"],
                    "latency_ms": elapsed_ms,
                    "miss": recall < 1.0 and not case["expected_abstain"],
                    "false_hits_on_abstain": case["expected_abstain"] and len(retrieved) > 0,
                }
            )
    metrics = {}
    inspected: list[dict[str, Any]] = []
    for method, rows in method_rows.items():
        n = len(rows)
        metrics[method] = {
            "n": n,
            "macro_recall": round(sum(row["recall"] for row in rows) / n, 4),
            "macro_precision": round(sum(row["precision"] for row in rows) / n, 4),
            "abstain_ok": sum(
                1
                for row in rows
                if row["expected_abstain"] and row["abstained"]
            ),
            "abstain_n": sum(1 for row in rows if row["expected_abstain"]),
            "mean_latency_ms": round(sum(row["latency_ms"] for row in rows) / n, 3),
        }
        for row in rows:
            if row["miss"] or row["false_hits_on_abstain"]:
                inspected.append({"method": method, **row})
    return {
        "schema": "grounded-agent-shootout-v1",
        "corpus": "fixtures/corpus",
        "gold": "fixtures/gold.json",
        "top_k": top_k,
        "methods": list(METHODS),
        "metrics": metrics,
        "rows": method_rows,
        "inspected_failures": inspected,
        "limitations": [
            "n is the Harbor fixture set, not a public IR benchmark.",
            "hashed_vector is a sha256 bag-of-tokens index, not a neural embedding API.",
            "graph_expand is one-hop [[wiki]] following, not Microsoft GraphRAG.",
            "Do not generalize these numbers to production corpora.",
        ],
    }


def _paths(values: list[str]) -> str:
    if not values:
        return "(none)"
    return ", ".join(f"`{path}`" for path in values)


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Retrieval shootout on the Harbor fixture",
        "",
        "This file is generated by `grounded-agent shootout`.",
        "The numbers come from the JSON receipt beside it.",
        "n=5 Harbor notes. That is the whole experiment.",
        "",
        f"top_k={payload['top_k']}. Methods: {', '.join(payload['methods'])}.",
        "",
        "| method | n | macro recall | macro precision | abstain ok | mean latency ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, row in payload["metrics"].items():
        lines.append(
            f"| {method} | {row['n']} | {row['macro_recall']} | {row['macro_precision']} | "
            f"{row['abstain_ok']}/{row['abstain_n']} | {row['mean_latency_ms']} |"
        )
    lines.extend(["", "## Inspected failures", ""])
    if not payload["inspected_failures"]:
        lines.append("None recorded.")
    for failure in payload["inspected_failures"]:
        lines.append(
            f"- `{failure['method']}` / `{failure['id']}`: got {_paths(failure['retrieved'])}; "
            f"gold was {_paths(failure['relevant'])}"
        )
    lines.extend(["", "## Limitations", ""])
    for item in payload["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_shootout(out_dir: Path = REPORTS_DIR) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = run_shootout()
    json_path = out_dir / "retrieval-shootout.json"
    md_path = out_dir / "retrieval-shootout.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_report(payload), encoding="utf-8")
    return payload
