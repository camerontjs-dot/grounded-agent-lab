"""Package-relative resource paths. Never hardcode machine-local directories.

The default Harbor corpus ships inside the package at ``grounded_agent/fixtures``, so a
source checkout and an installed wheel read the same files.

``WORKBENCH_ROOT`` and ``REPORTS_DIR`` only make sense in a source checkout. They name the
repository root and its committed ``docs/`` and ``reports/``. In an installed wheel they
resolve inside the Python installation, so runtime code must not use them for input or
output.
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
FIXTURE_ROOT = PACKAGE_ROOT / "fixtures"
KNOWLEDGE_CORPUS = FIXTURE_ROOT / "corpus" / "knowledge"
PROJECT_CORPUS = FIXTURE_ROOT / "corpus" / "projects"
QUESTIONS_PATH = FIXTURE_ROOT / "questions.json"
GOLD_PATH = FIXTURE_ROOT / "gold.json"
# Source-checkout only: the repository root and its committed reports directory.
WORKBENCH_ROOT = PACKAGE_ROOT.parent.parent
REPORTS_DIR = WORKBENCH_ROOT / "reports"
