# Decisions — Grounded Agent Lab

Newest first. These are public architecture tradeoffs for this repository.

## ADR-003 — Redacted receipts are the public trace (2026-08-23)

**Decision:** Every run emits a receipt with route, outcome, citation paths, trust profiles, and a stable hash. Receipts omit the question text, prompts, and retrieved snippets.

**Reason:** Observability without leaking the working corpus. A reviewer can still see *what was decided* and *which files were cited*.

**Consequence:** Debugging the model’s words requires the private test fixtures, not the receipt. That is intentional.

## ADR-002 — Trust split is a type invariant (2026-08-23)

**Decision:** Evidence is stored in two labelled groups (`durable_knowledge`, `project_status`). Construction fails if an item is placed in the wrong group. Weak-fit items cannot be marked citable.

**Reason:** The core failure mode of “RAG demos” is silently blending sources. A comment in a prompt is not enough; the type must refuse.

**Consequence:** Federated questions still query both indexes. They never return one unlabelled list.

## ADR-001 — Framework-free baseline before LangGraph (2026-08-23)

**Decision:** Phase 1 is plain Python with Pydantic, fixture retrievers, and a deterministic extractive drafter. LangGraph is the next phase, measured against this suite.

**Reason:** “I used LangGraph” is only an honest claim if the same behavior exists without the framework. The baseline also keeps tests free of API keys.

**Consequence:** No LangChain/LangGraph dependency lands until the fixture suite is green here.
