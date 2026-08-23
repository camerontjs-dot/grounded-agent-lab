"""Keep LangGraph tests offline and quiet about vendor tracing."""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")

warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects` will change",
)
warnings.filterwarnings(
    "ignore",
    message="Field 'lifespan' has an incomplete definition",
)
