"""Packaging contract: the default Harbor corpus must ship inside the wheel."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from grounded_agent import paths

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _declared_globs() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return list(data["tool"]["setuptools"]["package-data"]["grounded_agent"])


def _fixture_files() -> list[Path]:
    # Hidden files and __pycache__ are not fixture content, so they are skipped.
    files: list[Path] = []
    for path in paths.FIXTURE_ROOT.rglob("*"):
        parts = path.relative_to(paths.FIXTURE_ROOT).parts
        if path.is_file() and not any(p.startswith(".") or p == "__pycache__" for p in parts):
            files.append(path)
    return sorted(files)


def _unmatched(globs: list[str]) -> list[str]:
    matched: set[Path] = set()
    for pattern in globs:
        matched.update(path for path in paths.PACKAGE_ROOT.glob(pattern) if path.is_file())
    return [
        path.relative_to(paths.PACKAGE_ROOT).as_posix()
        for path in _fixture_files()
        if path not in matched
    ]


def test_every_fixture_file_is_matched_by_a_declared_glob() -> None:
    assert _fixture_files(), "fixture directory is missing or empty"
    assert _unmatched(_declared_globs()) == []


def test_glob_check_flags_fixtures_without_a_declared_glob() -> None:
    # Guard the guard: with only the JSON globs, the Markdown notes must be reported.
    json_only = [glob for glob in _declared_globs() if glob.endswith(".json")]
    unmatched = _unmatched(json_only)
    assert "fixtures/corpus/knowledge/trust-labels.md" in unmatched
    assert "fixtures/gold.json" not in unmatched


def test_default_corpus_is_inside_the_package() -> None:
    assert paths.KNOWLEDGE_CORPUS.is_relative_to(paths.PACKAGE_ROOT)
    assert paths.PROJECT_CORPUS.is_relative_to(paths.PACKAGE_ROOT)
    assert len(list(paths.KNOWLEDGE_CORPUS.glob("*.md"))) == 6
    assert len(list(paths.PROJECT_CORPUS.glob("*.md"))) == 2


def test_questions_and_gold_load_as_json_lists() -> None:
    for path in (paths.QUESTIONS_PATH, paths.GOLD_PATH):
        assert path.is_relative_to(paths.PACKAGE_ROOT)
        assert isinstance(json.loads(path.read_text(encoding="utf-8")), list)
