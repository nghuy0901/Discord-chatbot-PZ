# RAG Quality And Golden Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve exact retrieval and clarification behavior, narrowly calibrate groundedness equivalence, and create a validated 300-row golden dataset for review.

**Architecture:** Reuse the existing hybrid fusion, evidence policy, prompt-based groundedness, and JSONL dataset validator. Add no dependencies, services, or secondary reranker.

**Tech Stack:** Python, pytest, PostgreSQL/pgvector, JSONL.

---

### Task 1: Exact Item/Title Boost

**Files:**
- Modify: `rag/hybrid_retriever.py`
- Test: `tests/rag/test_hybrid_domain_filtering.py`

- [ ] Add a failing test where `Axe` ranks below seven compound Axe titles.
- [ ] Verify the test fails because exact `Axe` is absent from final top-k.
- [ ] Add one phrase-boundary helper and use it as a stable pre-cut sort key.
- [ ] Verify exact `Axe` and explicit `Hand Axe` queries both pass.

### Task 2: Referential Clarification

**Files:**
- Modify: `rag/evidence.py`
- Test: `tests/rag/test_evidence_policy.py`

- [ ] Add failing Vietnamese and English pronoun-only tests.
- [ ] Verify explicit-subject regression tests remain green.
- [ ] Extend the existing referential and generic-property vocabularies only.
- [ ] Run evidence-policy and release-gate tests.

### Task 3: Direct Semantic Equivalence

**Files:**
- Modify: `rag/groundedness.py`
- Modify: `prompts/templates/rag_instructions.txt`
- Test: `tests/rag/test_groundedness.py`
- Test: `tests/rag/test_strict_rag_runtime.py`

- [ ] Add failing prompt-contract tests for direct equivalence and vague-probability rejection.
- [ ] Add the narrow rules without changing thresholds.
- [ ] Run groundedness/finalization tests.

### Task 4: Golden Dataset V2

**Files:**
- Create: `scripts/build_release_dataset_v2.py`
- Create: `evaluation/data/release_qa.v2.jsonl`
- Create: `evaluation/data/release_qa.v2.summary.json`
- Modify: `tests/evaluation/test_dataset_v2.py`

- [ ] Add failing tests requiring 300 rows, exact quotas, unique questions, valid sources, and no fact-group split leakage.
- [ ] Build 60 source-backed fact specifications and deterministic paraphrases.
- [ ] Add deterministic unanswerable, ambiguous, and adversarial banks.
- [ ] Generate JSONL and summary artifacts.
- [ ] Validate with `validate_dataset(..., require_release_quota=True)`.

### Task 5: Verification And Runtime

**Files:**
- Modify only if verification reveals a scoped defect.

- [ ] Run targeted tests.
- [ ] Run full `python -m pytest -q`.
- [ ] Run development release eval against V2 on a bounded sample or full split as runtime permits.
- [ ] Rebuild and recreate API/bot containers.
- [ ] Verify API health, active collection, and runtime model.

