# Decisions: Grounded Agent Lab

Newest first. These are public architecture tradeoffs for this repository.

## ADR-008: Extractive stays default; Ollama and hosted fail closed (2026-08-23)

**Decision:** Keep the extractive drafter as the default. Put Ollama and an OpenAI-compatible hosted adapter behind a typed provider boundary. Missing daemon, missing model, or missing credentials is an error, not a silent fallback. Model JSON is schema-checked. Invented citation paths abstain.

**Rejected:** Making Ollama the CI default. Vendor SDKs as the portable layer. Running Temporal, Airflow, Prefect, or n8n as a phase gate.

**Reason:** The skill is a provider-neutral call with validated I/O. A green CI run that required a local GPU would be a lie. Mapping a durable workflow engine is a different skill from deploying one.

**Consequence:** `grounded-agent ask --provider ollama` exits 2 when the daemon or model is missing. Hosted needs `GROUNDED_AGENT_HOSTED_URL`, `GROUNDED_AGENT_HOSTED_API_KEY`, and `GROUNDED_AGENT_HOSTED_MODEL`. Receipts record `provider`. The workflow note is conceptual. I am not claiming I used OpenAI, Anthropic, Temporal, or n8n.

## ADR-007: Retrieved content cannot grant tools, and traces stay redacted (2026-08-23)

**Decision:** Scan citable evidence for the same injection markers as the question. Redact planted fixture secrets in copied sentences. Emit stage traces with paths and hashes only. Freeze the Harbor MCP server so `add_tool` cannot register writes. Review approve/reject still cannot widen the allowlist.

**Rejected:** Hosted guardrail SaaS as a Phase 5 requirement. Also rejected: logging retrieved snippets "just for debugging" in the public trace.

**Reason:** MCP and LangGraph interrupts do not make a tool safe. The failure I care about is a note or a tool description that tries to mint `write_index`.

**Consequence:** A forged tool card is retrieved, then the run abstains. A planted alias is copied as `[redacted]`. Receipts and JSONL traces fail closed if those markers appear.

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
