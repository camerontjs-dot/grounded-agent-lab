"""Draft providers. Extractive is default; Ollama and hosted fail closed."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Protocol
from urllib.parse import urljoin, urlparse

from pydantic import BaseModel, ConfigDict, ValidationError

from grounded_agent.draft import _citation, draft_or_abstain, evidence_is_injection, is_injection
from grounded_agent.models import Answer, EvidenceBundle, NonBlankStr, Outcome, ResearchRequest
from grounded_agent.redact import redact_text

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"
TAGS_TIMEOUT_S = 2.0
GENERATE_TIMEOUT_S = 60.0
HOSTED_TIMEOUT_S = 30.0


def _generate_timeout(*, default: float) -> float:
    raw = os.environ.get("GROUNDED_AGENT_PROVIDER_TIMEOUT_S")
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ProviderUnavailable("GROUNDED_AGENT_PROVIDER_TIMEOUT_S must be a number") from exc
    if value <= 0:
        raise ProviderUnavailable("GROUNDED_AGENT_PROVIDER_TIMEOUT_S must be positive")
    return value

SYSTEM_PROMPT = (
    "Answer only from the supplied Harbor notes. If they do not support the "
    "question, abstain. Return JSON with keys outcome, text, citation_paths, "
    "abstain_reason. citation_paths must be a subset of the supplied "
    "source_path values. Do not invent paths."
)


class ProviderUnavailable(Exception):
    """Local or hosted model backend is missing, unreachable, or misconfigured."""


class DraftProvider(Protocol):
    name: str

    def ensure(self) -> None: ...

    def draft(self, request: ResearchRequest, evidence: EvidenceBundle) -> Answer: ...


class ModelDraft(BaseModel):
    """Untrusted model JSON. Extra fields are ignored; required shape is not."""

    model_config = ConfigDict(extra="ignore")
    outcome: Outcome
    text: NonBlankStr
    citation_paths: tuple[str, ...] = ()
    abstain_reason: str | None = None


class UrlLibTransport:
    """Stdlib HTTP. Tests inject a fake transport; CI never needs a live daemon."""

    def request(
        self,
        method: str,
        url: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> str:
        request = urllib.request.Request(url, data=body, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            exc.read()
            raise ProviderUnavailable(f"HTTP {exc.code} from provider") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderUnavailable("provider request failed") from exc


def guarded_draft(
    provider: DraftProvider, request: ResearchRequest, evidence: EvidenceBundle
) -> Answer:
    """Refuse injection and empty evidence before calling a live model."""

    provider.ensure()
    if is_injection(request.question) or evidence_is_injection(evidence) or not evidence.citable():
        return draft_or_abstain(request, evidence)
    return provider.draft(request, evidence)


def _citable_map(evidence: EvidenceBundle) -> dict[str, Any]:
    return {item.source_path: item for item in evidence.citable()}


def _user_payload(request: ResearchRequest, evidence: EvidenceBundle) -> str:
    notes = [
        {
            "source_path": item.source_path,
            "trust_profile": item.trust_profile,
            "title": item.title,
            "snippet": item.snippet,
        }
        for item in evidence.citable()
    ]
    return json.dumps({"question": request.question, "notes": notes}, sort_keys=True)


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def parse_model_draft(raw: object) -> ModelDraft:
    if isinstance(raw, dict):
        payload = raw
    elif isinstance(raw, str):
        try:
            payload = json.loads(_strip_fence(raw))
        except json.JSONDecodeError as exc:
            raise ValueError("model output was not JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("model JSON must be an object")
    else:
        raise ValueError("model output must be JSON object or string")
    try:
        return ModelDraft.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("model JSON failed the draft schema") from exc


def answer_from_model_draft(draft: ModelDraft, evidence: EvidenceBundle) -> Answer:
    if draft.outcome == "abstained":
        reason = draft.abstain_reason or "insufficient_evidence"
        return Answer(outcome="abstained", text=redact_text(draft.text), abstain_reason=reason)

    citable = _citable_map(evidence)
    unknown = [path for path in draft.citation_paths if path not in citable]
    if unknown or not draft.citation_paths:
        return Answer(
            outcome="abstained",
            text="Model cited a path that is not in the citable evidence.",
            abstain_reason="unsupported_citation",
        )
    citations = tuple(_citation(citable[path]) for path in dict.fromkeys(draft.citation_paths))
    return Answer(outcome="answered", text=redact_text(draft.text), citations=citations)


def _safe_json(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderUnavailable("provider returned non-JSON") from exc
    if not isinstance(payload, dict):
        raise ProviderUnavailable("provider JSON must be an object")
    return payload


def _model_listed(tags: dict[str, Any], name: str) -> bool:
    wanted = name.lower()
    for model in tags.get("models", []):
        if not isinstance(model, dict):
            continue
        candidate = str(model.get("name") or model.get("model") or "").lower()
        if not candidate:
            continue
        if candidate == wanted:
            return True
        if candidate.startswith(f"{wanted}:"):
            return True
        stem = candidate.split(":", 1)[0]
        if wanted == stem:
            return True
    return False


class ExtractiveProvider:
    """Deterministic Harbor drafter. No network, no model."""

    name = "extractive"

    def ensure(self) -> None:
        return None

    def draft(self, request: ResearchRequest, evidence: EvidenceBundle) -> Answer:
        return draft_or_abstain(request, evidence)


class OllamaProvider:
    """Local Ollama chat. Fail closed if the daemon or model is missing."""

    name = "ollama"

    def __init__(
        self,
        *,
        url: str | None = None,
        model: str | None = None,
        transport: UrlLibTransport | None = None,
    ) -> None:
        configured = url or os.environ.get("GROUNDED_AGENT_OLLAMA_URL") or DEFAULT_OLLAMA_URL
        self.url = configured.rstrip("/")
        self.model = model or os.environ.get("GROUNDED_AGENT_OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
        self.transport = transport or UrlLibTransport()

    def ensure(self) -> None:
        parsed = urlparse(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ProviderUnavailable("GROUNDED_AGENT_OLLAMA_URL is not a valid HTTP URL")
        body = self.transport.request(
            "GET",
            urljoin(self.url + "/", "api/tags"),
            timeout=TAGS_TIMEOUT_S,
        )
        tags = _safe_json(body)
        if not _model_listed(tags, self.model):
            raise ProviderUnavailable(f"Ollama model {self.model!r} is not installed")

    def draft(self, request: ResearchRequest, evidence: EvidenceBundle) -> Answer:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_payload(request, evidence)},
            ],
            "stream": False,
            "format": "json",
            "options": {"num_predict": 256},
        }
        raw = self.transport.request(
            "POST",
            urljoin(self.url + "/", "api/chat"),
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            timeout=_generate_timeout(default=GENERATE_TIMEOUT_S),
        )
        message = _safe_json(raw).get("message")
        if not isinstance(message, dict):
            return Answer(
                outcome="abstained",
                text="Model output was not valid structured JSON.",
                abstain_reason="provider_unparseable",
            )
        try:
            draft = parse_model_draft(message.get("content"))
        except ValueError:
            return Answer(
                outcome="abstained",
                text="Model output was not valid structured JSON.",
                abstain_reason="provider_unparseable",
            )
        return answer_from_model_draft(draft, evidence)


class HostedProvider:
    """OpenAI-compatible /v1/chat/completions. Fail closed without credentials."""

    name = "hosted"

    def __init__(
        self,
        *,
        url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        transport: UrlLibTransport | None = None,
    ) -> None:
        env_url = os.environ.get("GROUNDED_AGENT_HOSTED_URL")
        env_key = os.environ.get("GROUNDED_AGENT_HOSTED_API_KEY")
        env_model = os.environ.get("GROUNDED_AGENT_HOSTED_MODEL")
        self.url = (url if url is not None else env_url or "").rstrip("/")
        self.api_key = api_key if api_key is not None else env_key or ""
        self.model = model or env_model or ""
        self.transport = transport or UrlLibTransport()

    def ensure(self) -> None:
        if not self.url:
            raise ProviderUnavailable("GROUNDED_AGENT_HOSTED_URL is not set")
        if not self.api_key:
            raise ProviderUnavailable("GROUNDED_AGENT_HOSTED_API_KEY is not set")
        if not self.model:
            raise ProviderUnavailable("GROUNDED_AGENT_HOSTED_MODEL is not set")
        parsed = urlparse(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ProviderUnavailable("GROUNDED_AGENT_HOSTED_URL is not a valid HTTP URL")

    def draft(self, request: ResearchRequest, evidence: EvidenceBundle) -> Answer:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_payload(request, evidence)},
            ],
            "temperature": 0,
        }
        raw = self.transport.request(
            "POST",
            urljoin(self.url + "/", "chat/completions"),
            body=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            timeout=_generate_timeout(default=HOSTED_TIMEOUT_S),
        )
        choices = _safe_json(raw).get("choices")
        if not isinstance(choices, list) or not choices:
            return Answer(
                outcome="abstained",
                text="Model output was not valid structured JSON.",
                abstain_reason="provider_unparseable",
            )
        first = choices[0]
        if not isinstance(first, dict):
            return Answer(
                outcome="abstained",
                text="Model output was not valid structured JSON.",
                abstain_reason="provider_unparseable",
            )
        message = first.get("message")
        if not isinstance(message, dict):
            return Answer(
                outcome="abstained",
                text="Model output was not valid structured JSON.",
                abstain_reason="provider_unparseable",
            )
        try:
            draft = parse_model_draft(message.get("content"))
        except ValueError:
            return Answer(
                outcome="abstained",
                text="Model output was not valid structured JSON.",
                abstain_reason="provider_unparseable",
            )
        return answer_from_model_draft(draft, evidence)


PROVIDER_NAMES = ("extractive", "ollama", "hosted")


def build_provider(
    name: str,
    *,
    transport: UrlLibTransport | None = None,
) -> ExtractiveProvider | OllamaProvider | HostedProvider:
    if name == "extractive":
        return ExtractiveProvider()
    if name == "ollama":
        return OllamaProvider(transport=transport)
    if name == "hosted":
        return HostedProvider(transport=transport)
    raise ValueError(f"unknown provider {name!r}")
