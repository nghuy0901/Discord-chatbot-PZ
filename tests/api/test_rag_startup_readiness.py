import pytest
from fastapi import HTTPException


@pytest.mark.asyncio
async def test_startup_loads_knowledge_base_and_bm25_when_rag_enabled(monkeypatch):
    import api.main as api_main

    calls = []

    async def fake_init_db():
        calls.append("init_db")

    class FakeMetricsManager:
        async def init_db(self):
            calls.append("metrics")

    class FakeKnowledgeManager:
        async def load_all(self):
            calls.append("kb")
            return {"pz": 10}

    async def fake_init_bm25_indices():
        calls.append("bm25")
        return {"knowledge_base": 10, "chat_history": 0}

    monkeypatch.setattr("src.startup.validate_runtime_config", lambda require_api=True: None)
    monkeypatch.setattr("src.startup.is_production", lambda: False)
    monkeypatch.setattr(api_main, "ENABLE_RAG", True)
    monkeypatch.setattr(api_main, "ENABLE_KNOWLEDGE_BASE", True, raising=False)
    monkeypatch.setattr(api_main, "init_db", fake_init_db)
    monkeypatch.setattr(api_main, "get_metrics_manager", lambda: FakeMetricsManager())
    monkeypatch.setattr(api_main, "get_knowledge_manager", lambda: FakeKnowledgeManager())
    monkeypatch.setattr("rag.bm25_search.init_bm25_indices", fake_init_bm25_indices)

    await api_main.startup_event()

    assert calls == ["init_db", "metrics", "kb", "bm25"]


@pytest.mark.asyncio
async def test_health_reports_embedding_configuration(monkeypatch):
    import api.main as api_main

    class FakeAcquire:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def execute(self, *_args, **_kwargs):
            return None

    class FakePool:
        def acquire(self):
            return FakeAcquire()

    async def fake_get_pool():
        return FakePool()

    async def fake_provider_health_check():
        return True

    monkeypatch.setattr(api_main, "ENABLE_RAG", True)
    monkeypatch.setattr(api_main, "get_pool", fake_get_pool)
    monkeypatch.setattr(api_main, "provider_health_check", fake_provider_health_check)
    monkeypatch.setattr(api_main, "get_query_cache", lambda: type("Cache", (), {"_redis": None})())
    monkeypatch.setattr(
        "src.llm.embedding_factory.build_embedding_config",
        lambda: object(),
    )

    response = await api_main.health_check()

    assert response["services"]["embedding"] == "configured"


@pytest.mark.asyncio
async def test_query_fails_closed_when_embedding_provider_is_missing(monkeypatch):
    import api.main as api_main

    payload = api_main.QueryRequest(
        query="What is a generator?",
        domain="pz",
        user_id="u1",
        channel_id="c1",
    )

    monkeypatch.setattr(api_main, "ENABLE_RAG", True)
    monkeypatch.setattr(api_main.api_rate_limiter, "allow", lambda _key: True)

    def missing_embedding_config():
        raise RuntimeError("EMBEDDING_PROVIDER is required")

    monkeypatch.setattr(
        "src.llm.embedding_factory.build_embedding_config",
        missing_embedding_config,
    )

    with pytest.raises(HTTPException) as exc_info:
        await api_main.execute_rag_query(payload, api_key="test-key")

    assert exc_info.value.status_code == 503
    assert "Embedding provider is not configured" in exc_info.value.detail
