# RAG P0 Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first P0 hallucination/citation remediations from the audit reports with regression tests.

**Architecture:** Keep fixes small and local: retrieval domain filtering in `rag.hybrid_retriever`, startup/readiness in `api.main`, citation parsing/validation in a focused helper module, and Self-RAG fail-safe behavior in `rag.self_rag`. Avoid DB-dependent tests by using monkeypatches and fake services.

**Tech Stack:** Python, pytest, FastAPI app module, existing RAG modules.

---

### Task 1: Hybrid KB Domain Filtering

**Files:**
- Modify: `rag/hybrid_retriever.py`
- Modify: `rag/retriever.py`
- Test: `tests/rag/test_hybrid_domain_filtering.py`

- [x] Write tests showing `retrieve_knowledge(... domains=["pz"])` passes domains into hybrid search.
- [x] Write tests showing `hybrid_search(... search_type="kb", domains=["pz"])` filters vector and BM25 KB results to `pz`.
- [x] Implement a `domains` parameter for `hybrid_search`.
- [x] Apply vector KB search per requested domain and BM25 metadata filtering for KB domains.
- [x] Run focused tests.

### Task 2: API RAG Readiness

**Files:**
- Modify: `api/main.py`
- Test: `tests/api/test_rag_startup_readiness.py`

- [x] Write tests showing `startup_event()` loads KB and initializes BM25 when RAG/KB are enabled.
- [x] Write tests showing `/api/health` reports embedding configuration status.
- [x] Implement startup loading with non-fatal logging consistent with existing startup behavior.
- [x] Implement embedding health in API health response.
- [x] Run focused tests.

### Task 3: Citation Validation Foundation

**Files:**
- Create: `rag/citations.py`
- Modify: `prompts/templates/rag_instructions.txt`
- Test: `tests/rag/test_citations.py`

- [x] Write tests for extracting `[KB-1]` and `[1]` citations from answers.
- [x] Write tests for rejecting citations not present in retrieved context.
- [x] Implement citation extraction and validation helpers.
- [x] Update prompt instructions to require citations for factual RAG/tool answers.
- [x] Add API enforcement for generated answers with missing/invalid RAG citations.
- [x] Run focused tests.

### Task 4: Self-RAG Fail-Safe

**Files:**
- Modify: `rag/self_rag.py`
- Test: `tests/rag/test_self_rag_fail_safe.py`

- [x] Write tests showing parse failures return `UNKNOWN`/low-confidence grades, not `RELEVANT`.
- [x] Write tests showing timeout/error fallback does not mark documents relevant.
- [x] Implement fail-safe grade helpers.
- [x] Run focused tests.

### Task 5: Verification

**Files:**
- No new files.

- [x] Run all new focused tests.
- [x] Run `python -m pytest -q`.
- [x] Summarize completed P0 items and remaining roadmap.
