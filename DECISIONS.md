# Decisions: Grounded Agent Lab

Newest first. These are public architecture tradeoffs for this repository.

## ADR-006: Shootout metrics come from a frozen fixture run (2026-08-23)

**Decision:** Phase 4 compares lexical overlap, a sha256 hashed-token vector index, and one-hop `[[wiki]]` expansion on labelled Harbor questions. Neural embeddings (Sentence Transformers) are an optional extra that fail closed if missing. Ragas is not the acceptance gate.

**Rejected:** Treating MiniLM or a hosted vector database as required for the first measurement. Also rejected: hand-writing a Markdown scoreboard without a JSON receipt.

**Reason:** A five-note corpus cannot support vendor ranking. The skill is measuring retrieval with gold paths, inspected misses, and written limits.

**Consequence:** CI reruns the shooter and checks the committed JSON ranking (latency excluded). Do not quote these recall numbers as a general embedding benchmark.

## ADR-005: Dual-index tools are allowlisted and read-only (2026-08-23)

**Decision:** Retrieval goes through two named tools, `query_knowledge` and `query_projects`. The allowlist, Pydantic argument schema, timeout, and write-name denylist sit in front of both the in-process client and the Harbor MCP server. Live MindGraph is a fail-closed optional adapter that does not run in CI.

**Rejected:** A `query_both` tool, third-party MCP connectors, and write tools "just for the demo."

**Reason:** MCP standardizes an interface. It does not make a tool safe. The skill to show is least privilege, schema checks, and labelled scopes.

**Consequence:** Unknown tools, write-shaped names, extra fields, and over-budget calls fail closed. Receipts record tool names and errors, never snippets. Scope `both` is not a tool.

## ADR-004: LangGraph wraps the baseline; it does not replace it (2026-08-23)

**Decision:** Add a LangGraph `StateGraph` whose nodes call the same route, retrieve, draft, and receipt functions as the framework-free loop. Fixture decisions must match. Optional checkpointed review can pause before a receipt is written and resume with `approve` or `reject`.

**Rejected:** Starting with LangGraph and reverse-engineering tests afterward.

**Reason:** "I used LangGraph" is only honest if the graph is a visible wrap of already-tested behavior. Checkpoint and resume are what the framework actually adds.

**Consequence:** Phase 2 stayed on fixtures. A rejected review becomes `abstain_reason=review_rejected` and still emits one redacted receipt. Re-invoking a finished thread does not mint a second hash.

## ADR-003: Redacted receipts are the public trace (2026-08-23)

**Decision:** Every run emits a receipt with route, outcome, citation paths, trust profiles, and a stable hash. Receipts omit the question text, prompts, and retrieved snippets.

**Rejected:** Logging raw retrieved text in the public artifact.

**Reason:** I want observability without leaking the working corpus. A reviewer can still see what was decided and which files were cited.

**Consequence:** Debugging the model's words requires the private test fixtures, not the receipt. That is intentional.

## ADR-002: Trust split is a type invariant (2026-08-23)

**Decision:** Evidence is stored in two labelled groups (`durable_knowledge`, `project_status`). Construction fails if an item is placed in the wrong group. Weak-fit items cannot be marked citable.

**Rejected:** A single unlabelled context list with a prompt instruction to "keep sources separate."

**Reason:** The usual RAG-demo failure is silently blending sources. A comment in a prompt is not enough. The type has to refuse.

**Consequence:** Federated questions still query both indexes. They never return one unlabelled list.

## ADR-001: Framework-free baseline before LangGraph (2026-08-23)

**Decision:** Phase 1 is plain Python with Pydantic, fixture retrievers, and a deterministic extractive drafter. LangGraph comes after, measured against this suite.

**Rejected:** A decorator demo with no baseline.

**Reason:** "I used LangGraph" is only an honest claim if the same behavior exists without the framework. The baseline also keeps tests free of API keys.

**Consequence:** Phase 1 shipped without LangGraph. The wrap landed later under ADR-004, against this same fixture suite.
