import pytest
from fastapi import HTTPException

from rag.metrics import RAGMetric


class DummyMetricsManager:
    def __init__(self):
        self.updated = []

    async def update_db(self, metric):
        self.updated.append(metric)


class DummyCache:
    def __init__(self):
        self.saved = None

    async def get(self, query, domain):
        return None

    async def set(self, **kwargs):
        self.saved = kwargs


def _patch_common(monkeypatch, api_main, metrics, cache, answer):
    monkeypatch.setattr(api_main, "ENABLE_RAG", True)
    monkeypatch.setattr(api_main, "ENABLE_TOOL_CALLING", False)
    monkeypatch.setattr(api_main, "ENFORCE_RAG_CITATIONS", True)
    monkeypatch.setattr(api_main.api_rate_limiter, "allow", lambda _key: True)
    monkeypatch.setattr(api_main, "get_metrics_manager", lambda: metrics)
    monkeypatch.setattr(api_main, "get_query_cache", lambda: cache)
    monkeypatch.setattr(api_main, "get_last_token_usage", lambda: None)
    monkeypatch.setattr(
        "src.llm.embedding_factory.build_embedding_config",
        lambda: object(),
    )

    async def fake_build_rag_context(**kwargs):
        metric = RAGMetric(
            query_id=kwargs["request_context"].query_id,
            request_id=kwargs["request_context"].request_id,
            source=kwargs["request_context"].source,
            query_intent="analytical",
            num_results=1,
        )
        context = (
            "[Retrieved Knowledge Base - Relevant static documents]\n"
            "[KB-1] (90% match) [pz] pz/generator.md\n"
            "    Generators require the how-to-use-generators magazine."
        )
        return context, None, metric

    async def fake_chat_completion(**kwargs):
        return answer

    monkeypatch.setattr(api_main, "build_rag_context", fake_build_rag_context)
    monkeypatch.setattr(api_main, "chat_completion", fake_chat_completion)


@pytest.mark.asyncio
async def test_query_rejects_rag_answer_without_required_citation(monkeypatch):
    import api.main as api_main

    metrics = DummyMetricsManager()
    cache = DummyCache()
    _patch_common(
        monkeypatch,
        api_main,
        metrics,
        cache,
        "Generators require the magazine.",
    )

    payload = api_main.QueryRequest(
        query="How do generators work?",
        domain="pz",
        user_id="u1",
        channel_id="c1",
    )

    with pytest.raises(HTTPException) as exc_info:
        await api_main.execute_rag_query(payload, api_key="test-key")

    assert exc_info.value.status_code == 502
    assert "citation validation" in exc_info.value.detail
    assert metrics.updated[-1].rag_decision == "abstain"
    assert metrics.updated[-1].decision_reason == "missing_required_citation"
    assert cache.saved is None


@pytest.mark.asyncio
async def test_query_returns_and_caches_citation_validation_metadata(monkeypatch):
    import api.main as api_main

    metrics = DummyMetricsManager()
    cache = DummyCache()
    _patch_common(
        monkeypatch,
        api_main,
        metrics,
        cache,
        "Generators require the how-to-use-generators magazine [KB-1].",
    )

    payload = api_main.QueryRequest(
        query="How do generators work?",
        domain="pz",
        user_id="u1",
        channel_id="c1",
    )

    result = await api_main.execute_rag_query(payload, api_key="test-key")

    assert result["citation_validation"]["is_valid"] is True
    assert result["citation_validation"]["valid_ids"] == ["KB-1"]
    assert cache.saved["result"]["citation_validation"]["valid_ids"] == ["KB-1"]
    assert metrics.updated[-1].rag_decision == "answer"
    assert metrics.updated[-1].decision_reason == "citation_validated"
