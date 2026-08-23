# Decisions — Grounded Agent Lab

Newest first. These are public architecture tradeoffs for this repository.

## ADR-004 — LangGraph wraps the baseline; it does not replace it (2026-08-23)

**Decision:** Add a LangGraph `StateGraph` whose nodes call the same route/retrieve/draft/receipt functions as the framework-free loop. Fixture decisions must match. Optional checkpointed review can pause before a receipt is written and resume with `approve` or `reject`.

**Reason:** “I used LangGraph” is only honest if the graph is a visible wrap of already-tested behavior. Checkpoint/resume is the thing the framework actually adds.

**Consequence:** Live retrieval adapters stay out of this phase. A rejected review becomes `abstain_reason=review_rejected` and still emits one redacted receipt. Re-invoking a finished thread does not mint a second hash.

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
