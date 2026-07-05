# RAG Hallucination Debug Report

Audit date: 2026-07-05

Scope: parsing, chunking, embedding, indexing, retrieval, reranking, prompt construction, LLM generation, citation, abstain, and evaluation for the NomNom Discord/API RAG system.

## Executable Baseline

- Full test suite: `python -m pytest -q` -> `163 passed, 22 warnings` in 22.06s.
- Focused RAG/eval tests: `15 passed` for retrieval metrics, eval wiring, embedding registry, KB reload dedup, prompt injection guard, and metrics persistence.
- Static KB parse/chunk diagnostic: `165` knowledge files, `12,763` local chunks.
- Frontmatter diagnostic: `160 / 165` knowledge files contain `source_url`, but chunk metadata drops it.
- Current local env diagnostic: `ENABLE_RAG=true`, `ENABLE_KNOWLEDGE_BASE=true`, `HYBRID_RAG_ENABLED=true`, `SELF_RAG_ENABLED=false`, `PERSONALITY_TEMPERATURE=0.8`, `RAG_TOP_K=8`, `KB_TOP_K=4`, `KB_THRESHOLD=0.35`.
- Current embedding factory diagnostic: `src.llm.embedding_factory.build_embedding_config()` raises `RuntimeError: EMBEDDING_PROVIDER is required: openai|openai_compatible|gemini`.
- Current API KB singleton diagnostic: initial `get_knowledge_manager().domains` is `{}` in a fresh process.
- Query preprocessor diagnostic:
  - `axe nao manh nhat?` equivalent (`axe nào mạnh nhất?`) -> `domain=None`, `intent=analytical`.
  - `Base.Axe_Old là gì?` -> `domain=None`, `intent=narrative`.
  - `cách chữa Knox Infection` -> `domain=pz`, `intent=hybrid`.

## Data-Flow Summary

1. Discord JSON ingestion normalizes messages in `rag.ingest._normalise_message`, stores each message as a full document through `rag.db.add_documents_batch`, and records reply/mention edges.
2. Static KB ingestion discovers Markdown/TXT/RST files in `knowledge.manager.KnowledgeManager.load_domain`, chunks Markdown with `MarkdownSemanticChunker`, then stores chunks in the `knowledge_base` collection.
3. Retrieval starts in `rag.retriever.build_rag_context`, preprocesses the query, searches KB and chat history, optionally runs Self-RAG relevance grading, formats retrieved blocks, and records `RAGMetric`.
4. Prompt assembly happens in `prompts.system_prompt.build_system_prompt`; Discord adds recent conversation history in `utils.context_manager.ContextManager.build_prompt`.
5. Generation runs through `src.ollama_provider.chat_completion` or `chat_with_tools`; API and Discord both use temperature `0.8` for factual answers unless caller overrides it.
6. Citation today is prompt-only: retrieved context has labels like `[KB-1]` and `[1]`, but generated answers are not required or validated to contain those labels.
7. Abstention today is prompt-only: metrics fields exist for `rag_decision`, `evidence_score`, and provenance, but runtime does not set them.

## Findings

### P0-1. Current embedding configuration prevents live RAG from working

- File/function: `src/llm/embedding_factory.py::build_embedding_config`, lines 10-16.
- Evidence: `build_embedding_config()` requires `EMBEDDING_PROVIDER`; local diagnostic raised `RuntimeError: EMBEDDING_PROVIDER is required: openai|openai_compatible|gemini`.
- Impact: ingestion, vector search, implicit replies, and KB reload can fail before retrieval. Empty retrieval then pushes the model toward general knowledge and hallucination.
- Metrics to measure: `embedding_config_ok`, `embedding_provider_health`, `empty_retrieval_rate`, `rag_unavailable_rate`.
- Methodology: add startup/health check that instantiates both chat and embedding providers; run a synthetic embedding query; count RAG queries served while embedding is unhealthy.
- Current value: `embedding_config_ok=false` locally. Full tests still pass because e2e tests mock or disable RAG.
- Target: `embedding_config_ok=true`; `rag_unavailable_rate=0` for production traffic; API returns degraded/503 for RAG routes if embeddings are unavailable.
- Patch needed: fail closed in `api.main.startup_event` and `NomNomClient.setup_hook` when `ENABLE_RAG=true` but embedding config is invalid; include embedding status in `/api/health`; document required envs; avoid silent defaults in `rag.db` that mask missing embedding config.

### P0-2. FastAPI startup does not load KB domains or BM25 indices

- File/function: `api/main.py::startup_event`, lines 78-86.
- Evidence: startup only calls `init_db()` and `get_metrics_manager().init_db()`. Fresh process diagnostic showed `api_process_initial_kb_domains 0 []`.
- Impact: API RAG can route through `build_rag_context` with no loaded `KnowledgeManager.domains`, producing empty KB retrieval unless an admin reload was called in that process.
- Metrics to measure: `kb_loaded_domain_count`, `kb_loaded_chunk_count`, `bm25_ready`, `api_empty_kb_retrieval_rate`.
- Methodology: record KB/BM25 status at API startup; run `/api/query` before and after `/api/admin/reload`; compare retrieved KB count.
- Current value: `kb_loaded_domain_count=0` in a fresh local process.
- Target: nonzero KB domains/chunks at API startup; `api_empty_kb_retrieval_rate <= 0.02` for answerable KB queries.
- Patch needed: call `get_knowledge_manager().load_all()` and `init_bm25_indices()` in `api.main.startup_event` when `ENABLE_KNOWLEDGE_BASE`/`ENABLE_RAG` are enabled, and expose readiness in `/api/health`.

### P0-3. Hybrid KB retrieval ignores requested domains

- File/function: `rag.retriever.retrieve_knowledge`, lines 223-257; `rag.hybrid_retriever.hybrid_search`, lines 186-193 and 215-222.
- Evidence: `build_rag_context` computes `domains_to_search` and passes it to `retrieve_knowledge` at lines 426-439. In the hybrid path, `retrieve_knowledge` calls `hybrid_search(... search_type="kb")` without passing domains. `hybrid_search` then calls `kb.search(query=..., k=..., score_threshold=...)` with no domain filter and BM25 filter is only constructed for chat channel filters.
- Impact: PZ questions can retrieve `general` or `server_rules` chunks; server-rule questions can retrieve PZ chunks. The top result also selects `primary_domain`, so the wrong domain prompt can be injected.
- Metrics to measure: `domain_mismatch_rate`, `context_precision_by_domain`, `wrong_domain_prompt_rate`.
- Methodology: add eval rows with expected domain; assert every retrieved KB result has `domain in expected_domains`; run with `HYBRID_RAG_ENABLED=true` and false.
- Current value: static code evidence; not live-executable locally because embeddings/DB are not configured.
- Target: `domain_mismatch_rate=0` when a domain is explicitly detected or requested.
- Patch needed: add `domains: Optional[list[str]]` to `hybrid_search`; pass it from `retrieve_knowledge`; apply vector filters (`domain` or `$in`) and BM25 metadata filters for KB; add regression tests.

### P0-4. No generated-answer citation enforcement

- File/function: `knowledge.domain_router.format_kb_results_for_prompt`, lines 128-163; `rag.retriever.format_retrieved_for_prompt`, lines 287-344; `api.main.execute_rag_query`, lines 249-283 and 329-340.
- Evidence: retrieved context labels sources as `[KB-1]` or `[1]`, but `prompts/templates/rag_instructions.txt` lines 1-14 never require final answers to cite those labels. API returns only `"response"` and token metrics. No post-generation validator checks answer citations.
- Impact: the model can answer correctly or incorrectly without citations, and the system has no way to reject unsupported claims.
- Metrics to measure: `answer_citation_rate`, `citation_validity_rate`, `citation_support_precision`, `unsupported_claim_rate`.
- Methodology: parse final answer citation tokens; verify each citation ID exists in retrieved context; run claim-level support check against cited chunks; fail factual eval rows without citations.
- Current value: no executable metric exists; current `citation_coverage` is not answer citation coverage.
- Target: `answer_citation_rate=1.0` for factual RAG/tool answers; `citation_validity_rate>=0.98`; `unsupported_claim_rate<=0.05`.
- Patch needed: update RAG prompt to require citations for factual claims; add a `validate_citations(answer, evidence)` postprocessor; return/record provenance and citation validation status; reject or regenerate when citations are missing.

### P0-5. Citation coverage metric is misleading

- File/function: `rag.retriever._citation_coverage`, lines 109-121; metric assignment at `rag.retriever.build_rag_context`, line 514.
- Evidence: coverage is computed from retrieved result metadata (`source`, `message_id`, `author_name`, `author_id`), not from the generated answer.
- Impact: dashboards can report high citation coverage even when the user-facing answer has zero citations.
- Metrics to measure: replace or split into `retrieved_provenance_coverage` and `answer_citation_coverage`.
- Methodology: compute retrieved provenance before generation; compute answer citation coverage after generation and compare both.
- Current value: current metric only measures retrieved metadata coverage; answer value is unavailable.
- Target: both metrics separately recorded; answer metric drives release gates.
- Patch needed: rename current metric, add `answer_citation_coverage`, and update API/Discord metric update after generation.

### P0-6. Abstention policy fields exist but are not implemented

- File/function: `rag.metrics.RAGMetric`, lines 194-201; `prompts/templates/rag_instructions.txt`, lines 3-6.
- Evidence: `rag_decision`, `decision_reason`, `provenance`, `trusted_source_count`, `untrusted_source_count`, and `evidence_score` default to unknown/zero. Search found no runtime code that sets a real answer/abstain decision.
- Impact: when retrieval is empty, low-confidence, untrusted, or contradictory, the model still generates; hallucinations are not blocked or measured.
- Metrics to measure: `abstain_precision`, `abstain_recall`, `wrong_abstention_rate`, `answered_without_evidence_rate`.
- Methodology: add unanswerable and insufficient-evidence eval rows; compute expected behavior (`answer` vs `abstain`) before generation; compare final behavior.
- Current value: `rag_decision="unknown"` by default; no executable abstain gate.
- Target: `abstain_recall>=0.95` for unanswerable factual rows; `answered_without_evidence_rate<=0.02`.
- Patch needed: implement an evidence decision layer in `build_rag_context` or immediately before generation; if evidence score is below threshold for factual/PZ-stat queries, return an abstention template and set metric fields.

### P0-7. Self-RAG is disabled locally and fail-open when enabled

- File/function: `rag.self_rag.grade_relevance`, lines 130-231; `_grade_batch`, lines 294-301; `_parse_batch_response`, lines 357-376.
- Evidence: local env has `SELF_RAG_ENABLED=false`. Code returns original results when disabled, and timeout/parse failures produce `RELEVANT` grades at score `0.5`.
- Impact: the reranking/filtering safety layer either does not run or silently lets irrelevant documents through.
- Metrics to measure: `self_rag_enabled_rate`, `grader_error_rate`, `irrelevant_pass_rate`, `self_rag_filtered_count`.
- Methodology: inject malformed grader responses in tests; run eval with known irrelevant top results; measure how many pass.
- Current value: `SELF_RAG_ENABLED=false` locally.
- Target: enabled for factual RAG queries; `grader_error_rate<0.02`; fail-safe behavior does not mark unknown documents relevant.
- Patch needed: make grader failure return ungraded/low-confidence metadata, not relevant; lower final evidence score on grader failure; add tests for timeout, parse failure, and partial grade arrays.

### P1-1. Chat history retrieval uses unapproved chat as evidence

- File/function: `rag.db._msg_to_document`, lines 176-180; `rag.db.search_similar`, lines 249-292; `rag.bm25_search.refresh_from_db`, lines 187-209.
- Evidence: ingested chat documents are marked `source_kind="discord_chat"`, `approval_status="unapproved"`, `trusted=False`. Retrieval filters only by channel ID; BM25 indexes all documents in the collection with no trust filter.
- Impact: jokes, wrong user claims, outdated chatter, and prompt-injection-like content can be retrieved as factual grounding.
- Metrics to measure: `untrusted_context_ratio`, `approved_chat_source_hit_rate`, `unsupported_claim_rate_by_source_kind`.
- Methodology: tag every retrieved chunk with trust/source kind; report ratios and compare factual answer quality with trusted-only chat retrieval.
- Current value: no runtime metric populated; schema exists but retrieval ignores it.
- Target: factual RAG uses `trusted=true` or static KB/tool evidence by default; untrusted chat may appear only as conversational memory with lower evidence weight.
- Patch needed: add trust filters to vector and BM25 chat search; make approval status configurable by query intent; populate `trusted_source_count` and `untrusted_source_count`.

### P1-2. Cache stores answer text without provenance or citation validation

- File/function: `api.main.execute_rag_query`, lines 175-220 and 313-325; `src.aclient.NomNomClient._generate_and_send`, lines 535-560 and 707-720.
- Evidence: cache hit returns cached `response_text` directly. Cache writes only `response_text`, `prompt_tokens`, and `completion_tokens`; no retrieved source IDs, citations, evidence score, or validation status.
- Impact: a hallucinated or uncited answer can be replayed without retrieval, reranking, or citation validation. Cache-hit metrics have no retrieval evidence.
- Metrics to measure: `cached_answer_citation_rate`, `cached_provenance_presence_rate`, `cache_stale_evidence_rate`.
- Methodology: require cached payload schema to include provenance and validation status; sample cache hits and revalidate against current KB version.
- Current value: cached payload lacks provenance by code inspection.
- Target: `cached_provenance_presence_rate=1.0`; no cache hit can bypass failed citation validation.
- Patch needed: cache only validated answers; store evidence IDs, prompt version, retrieval config, answer citation coverage, and evidence score; revalidate or bypass cache when evidence is missing.

### P1-3. Thread context enrichment fetches edge IDs but not message content

- File/function: `rag.db.get_message_context`, lines 316-333; `rag.retriever.format_retrieved_for_prompt`, lines 327-334.
- Evidence: `get_message_context` returns only `message_id` and `edge_type`. Formatter expects `author_name`, `author_id`, and `content`, so related context renders as unknown/empty.
- Impact: reply context does not supply usable evidence; the model may infer missing context or cite weak chat snippets.
- Metrics to measure: `thread_context_fill_rate`, `thread_context_token_coverage`, `reply_context_precision`.
- Methodology: for retrieved messages with edges, measure how many related rows include nonempty content and author metadata.
- Current value: content fill rate is structurally `0` for this function.
- Target: `thread_context_fill_rate>=0.95` for edges whose messages are still indexed.
- Patch needed: join `message_edges` to `langchain_pg_embedding` and return `document` plus metadata for parent/child messages.

### P1-4. Domain detection misses common PZ factual queries

- File/function: `rag.query_preprocessor.QueryPreprocessor._detect_domain`, domain keyword map around lines 178-205.
- Evidence: local diagnostic produced `domain=None` for `axe nào mạnh nhất?` and `Base.Axe_Old là gì?`, despite prompt rules saying game-related questions should assume Project Zomboid.
- Impact: retrieval may search all domains and miss PZ-specific domain prompting; API `domain` behaves like `channel_name`, not a strict filter.
- Metrics to measure: `domain_routing_accuracy`, `pz_false_negative_rate`, `wrong_domain_prompt_rate`.
- Methodology: build a labeled routing eval set for PZ item/stat/location/rule/general queries; compare preprocessor output.
- Current value: `2/3` tested PZ-specific examples missed `domain=pz`.
- Target: `pz_false_negative_rate<=0.05` for game-related PZ queries.
- Patch needed: align preprocessor with `game_domain_rules`; add keywords/item patterns (`axe`, `weapon`, `Base.*`, common stat fields); make API `payload.domain` a hard domain hint.

### P1-5. Factual generation uses high temperature

- File/function: `api.main.execute_rag_query`, lines 270-282; `utils.context_manager.PERSONALITY_TEMPERATURE`, line 30; `src.ollama_provider.chat_completion`, lines 56-69.
- Evidence: API passes `temperature=0.8`; Discord returns `PERSONALITY_TEMPERATURE=0.8` for all intents; default chat completion temperature is `0.8`.
- Impact: higher variance increases unsupported phrasing, invented numbers, and citation drift on factual answers.
- Metrics to measure: `repeat_answer_variance`, `unsupported_claim_rate`, `citation_stability_rate`.
- Methodology: run each factual eval example 3-5 times at current and lower temperatures; compare answer/citation consistency.
- Current value: `0.8`.
- Target: `0.1-0.3` for factual/tool/RAG answers; optional `0.7-0.8` for casual conversation.
- Patch needed: make temperature intent-based: analytical/tool and RAG factual low, conversation/persona higher.

### P1-6. Tool outputs are not validated against final answers

- File/function: `src.ollama_provider.chat_with_tools`, lines 122-188; API call site `api.main.execute_rag_query`, lines 270-277.
- Evidence: `_last_tool_results` stores tool outputs, but callers do not validate final answer claims against those outputs. `chat_with_tools` returns final text directly.
- Impact: the model can call tools, then invent a stat or omit source/citation in the final answer.
- Metrics to measure: `tool_answer_support_rate`, `tool_numeric_mismatch_rate`, `tool_citation_rate`.
- Methodology: parse tool JSON outputs and final answer numbers/names; assert all reported values appear in tool output or cited KB context.
- Current value: no validator exists.
- Target: `tool_numeric_mismatch_rate=0`; `tool_answer_support_rate>=0.98`.
- Patch needed: return a structured result containing answer plus tool outputs; validate final text before sending; cite tool result IDs.

### P2-1. KB citations drop upstream `source_url`

- File/function: `knowledge.manager.MarkdownSemanticChunker._process_section`, lines 345-362; `KnowledgeManager._load_and_chunk_file`, lines 847-865.
- Evidence: `source_url` appears in `160 / 165` knowledge files. `base_meta` only keeps `heading_path`, `content_mode`, `category`, and `type`; document metadata stores local `source` but not `source_url`.
- Impact: final citations can point only to local filenames, not the original wiki/source URL, making citations less useful and harder to audit.
- Metrics to measure: `source_url_metadata_coverage`, `external_citation_rate`.
- Methodology: inspect indexed metadata for all KB chunks; count chunks with nonempty `source_url`.
- Current value: source files have `source_url` coverage `160/165`; indexed chunk coverage would be `0` by code path.
- Target: `source_url_metadata_coverage>=0.95` for KB chunks derived from frontmatter.
- Patch needed: include frontmatter provenance fields (`title`, `source_url`, `scraped_at`, `method`) in chunk metadata and prompt formatting.

### P2-2. KB chunking/indexing likely over-fragments noisy wiki pages

- File/function: `knowledge.manager.SentenceChunker.chunk`, lines 68-142; `MarkdownSemanticChunker.NOISE_HEADINGS`, lines 187-190; prompt formatting `knowledge.domain_router.format_kb_results_for_prompt`, lines 128-163.
- Evidence: static diagnostic produced `12,763` chunks from `165` files. Sample PZ files contain wiki navigation text immediately after frontmatter. Noise filtering only skips specific headings, not nav/menu paragraphs.
- Impact: top-k `KB_TOP_K=4` and `max_chars=2000` can miss needed sibling facts, while noisy chunks consume retrieval slots.
- Metrics to measure: `context_recall@k`, `nav_noise_hit_rate`, `chunk_redundancy_rate`, `answerable_but_missing_context_rate`.
- Methodology: label expected source and expected context keywords; inspect top-k chunks for nav-only or low-information content.
- Current value: `12,763` static chunks; no nav-noise metric exists.
- Target: `context_recall@5>=0.85`; `nav_noise_hit_rate<=0.02`.
- Patch needed: add markdown cleanup for wiki nav/template noise; preserve parent headings; consider parent-document expansion or MMR for sibling chunks.

### P2-3. Evaluation e2e tests do not exercise live RAG

- File/function: `tests/e2e/conftest.py`, lines 71-72 and 146-168.
- Evidence: API e2e monkeypatches `ENABLE_RAG=False`; Discord e2e replaces `build_prompt` and also sets `ENABLE_RAG=False`.
- Impact: CI can pass while live RAG, citations, embedding config, KB load, and retrieval fail.
- Metrics to measure: `live_rag_test_coverage`, `citation_gate_coverage`.
- Methodology: add a hermetic fake vectorstore/embedding test that exercises `build_rag_context` through API/Discord without external services.
- Current value: full suite passes but live RAG not covered by e2e.
- Target: at least one CI test per release gate: retrieval, citation, abstain, domain filter, cache provenance.
- Patch needed: add integration tests with fake embeddings/vectorstore and a deterministic fake LLM.

### P2-4. Discord metric update can attach generation data to the wrong query

- File/function: `src.aclient.NomNomClient._generate_and_send`, lines 688-725.
- Evidence: after generation, code updates `metrics._recent[-1]`, not the metric matching `request_context.query_id`.
- Impact: concurrent Discord requests can corrupt response time, token, cache, and citation metrics, hiding hallucination patterns.
- Metrics to measure: `metric_query_id_mismatch_rate`.
- Methodology: run concurrent mocked Discord queries with different query IDs; assert response metrics update the matching metric.
- Current value: no test exists; code path is race-prone.
- Target: `metric_query_id_mismatch_rate=0`.
- Patch needed: return the metric object/query ID from prompt construction and update that exact record, not the latest global item.

### P3-1. Legacy LlamaIndex path is still present

- File/function: `llm_provider.py` and `discord_bot.py`.
- Evidence: legacy files load LlamaIndex storage and prompts independently from the newer pgvector path.
- Impact: operational ambiguity: running the wrong entrypoint can bypass the audited RAG/citation path entirely.
- Metrics to measure: `active_entrypoint_inventory`.
- Methodology: deployment check confirms `main.py`/`src.bot` is the only production entrypoint.
- Current value: legacy files remain in repo.
- Target: one supported RAG path or explicit deprecation guard.
- Patch needed: document legacy status or remove/disable old entrypoint after confirming it is unused.

## Remediation Roadmap

Priority 0:

1. Fail closed on missing embedding config and load KB/BM25 at API startup.
2. Fix hybrid KB domain filtering.
3. Implement evidence decision and abstention before generation.
4. Enforce generated-answer citations and split retrieved provenance coverage from answer citation coverage.
5. Make Self-RAG fail safe and enable it for factual RAG queries after tests pass.

Priority 1:

1. Filter or down-rank untrusted chat history for factual answers.
2. Store provenance and citation-validation status in cache; reject cache hits without evidence.
3. Fetch real thread-context message content.
4. Improve PZ domain routing and make API `domain` a hard hint.
5. Use low temperature for factual/tool/RAG intents.
6. Validate final answers against tool outputs.

Priority 2:

1. Preserve `source_url` and frontmatter provenance through chunk metadata and prompt formatting.
2. Clean wiki navigation noise and add parent/sibling chunk expansion.
3. Add hermetic live-RAG tests and citation/abstain tests.
4. Update Discord metrics by query ID instead of `_recent[-1]`.

Priority 3:

1. Deprecate or clearly fence legacy LlamaIndex entrypoints.
2. Add dashboards for domain mismatch, untrusted evidence ratio, and cache provenance.
3. Add repeated-run variance evals for temperature tuning.

## Proposed Release Gates

- `embedding_config_ok == true`
- `kb_loaded_domain_count >= 3` and `kb_loaded_chunk_count > 0`
- `domain_mismatch_rate == 0` for explicit-domain queries
- `answer_citation_rate >= 0.98` for factual answer rows
- `citation_validity_rate >= 0.98`
- `unsupported_claim_rate <= 0.05`
- `abstain_recall >= 0.95` on unanswerable rows
- `untrusted_context_ratio == 0` for factual PZ/stat/rule rows unless explicitly allowed
- `source_url_metadata_coverage >= 0.95` for KB chunks
- Existing RAGAS/retrieval gates remain: faithfulness, answer correctness, context recall, source hit rate, MRR, empty retrieval rate.
