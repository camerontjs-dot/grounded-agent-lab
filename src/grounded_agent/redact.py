"""Fixture secrets that must never appear in receipts or traces."""

from __future__ import annotations

SENSITIVE_MARKERS = (
    "HARBOR_SECRET_PLACEHOLDER",
    "harbor-operator@example.invalid",
)
REDACTED = "[redacted]"


def contains_sensitive(text: str) -> bool:
    return any(marker in text for marker in SENSITIVE_MARKERS)


def redact_text(text: str) -> str:
    redacted = text
    for marker in SENSITIVE_MARKERS:
        redacted = redacted.replace(marker, REDACTED)
    return redacted


def assert_no_sensitive(blob: str, *, label: str) -> None:
    for marker in SENSITIVE_MARKERS:
        if marker in blob:
            raise ValueError(f"{label} leaked sensitive fixture content")
