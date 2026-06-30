# Ollama Cloud Authentication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Authenticate Ollama Cloud inference requests from both active and legacy bot entrypoints.

**Architecture:** Add a focused native Ollama HTTP client and select it through the existing LLM factory. Preserve compatibility environment variables and pass equivalent headers to the legacy LlamaIndex integration.

**Tech Stack:** Python 3.12, aiohttp, pytest, pytest-asyncio, LlamaIndex legacy adapter

---

### Task 1: Provider configuration

**Files:**
- Modify: `src/llm/factory.py`
- Test: `tests/llm/test_factory.py`

- [ ] Add failing tests for `LLM_PROVIDER=ollama`, compatibility variables, and missing cloud API key.
- [ ] Run `pytest tests/llm/test_factory.py -v` and confirm the new tests fail.
- [ ] Add `ollama` to supported providers and build its configuration from `OLLAMA_*` fallbacks.
- [ ] Run the factory tests and confirm they pass.

### Task 2: Native Ollama chat client

**Files:**
- Create: `src/llm/ollama_client.py`
- Create: `tests/llm/test_ollama_client.py`
- Modify: `src/llm/factory.py`

- [ ] Add a failing asynchronous test that captures the outgoing URL, JSON body, and Authorization header.
- [ ] Run `pytest tests/llm/test_ollama_client.py -v` and confirm it fails because the client is missing.
- [ ] Implement `OllamaClient.chat()` with `aiohttp`, `/api/chat`, Bearer authentication, safe errors, and token usage parsing.
- [ ] Select `OllamaClient` from `get_chat_client()`.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Legacy LlamaIndex authentication

**Files:**
- Modify: `llm_provider.py`
- Create: `tests/test_legacy_llm_provider.py`

- [ ] Add a failing isolated test that verifies LlamaIndex Ollama receives `Authorization: Bearer ...`.
- [ ] Run `pytest tests/test_legacy_llm_provider.py -v` and confirm the test fails.
- [ ] Centralize legacy Ollama construction and pass authenticated client headers.
- [ ] Run the legacy test and confirm it passes.

### Task 4: Configuration documentation and verification

**Files:**
- Modify: `.env.example`
- Modify: `README.md`

- [ ] Document `LLM_PROVIDER=ollama`, cloud base URL, model, and API key.
- [ ] Run all focused LLM tests.
- [ ] Run the complete pytest suite and inspect failures.
- [ ] Review the diff for accidental secret exposure and unrelated edits.

