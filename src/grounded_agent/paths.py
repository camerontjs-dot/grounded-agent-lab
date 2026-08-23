"""Workbench-relative paths. Never hardcode machine-local directories."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
WORKBENCH_ROOT = PACKAGE_ROOT.parent.parent
FIXTURE_ROOT = WORKBENCH_ROOT / "fixtures"
KNOWLEDGE_CORPUS = FIXTURE_ROOT / "corpus" / "knowledge"
PROJECT_CORPUS = FIXTURE_ROOT / "corpus" / "projects"
QUESTIONS_PATH = FIXTURE_ROOT / "questions.json"
